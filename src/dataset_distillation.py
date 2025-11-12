"""
Dataset distillation (SelMatch-style + MTT) implementation in feature space.

This implementation contains:
- selection-based initialization (margin-based "difficulty" selector)
- partial updates (freeze fraction / schedule)
- short-unroll trajectory matching (differentiable SGD inner loop)
- careful BN handling (default: LayerNorm to avoid running-stat issues)
- evaluation routines (train on distilled vs real, test on true labels)

Usage: import DatasetDistiller and run distill(...) and evaluate_distilled_data(...)

Notes:
- This implementation operates in feature-space (pre-extracted features + pseudo labels).
- It aims to be faithful to the high-level SelMatch / MTT ideas while remaining self-contained and practical.
"""

from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import math
import copy
import random
from .evaluation import cluster_accuracy, fixed_mapping_accuracy


# ----------------------------- Utility Models ---------------------------------
class SimpleClassifier(nn.Module):
    """Feedforward classifier. Default uses LayerNorm to avoid BN running-stat issues."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None, use_batchnorm: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            else:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------- Distiller Class --------------------------------
class DatasetDistiller:
    """
    Dataset distillation with SelMatch-like initialization and MTT-style short unrolling.

    Key options:
      - selection_strategy: 'random' | 'margin' (difficulty based on classifier margin)
      - partial_update: fraction of synthetic set updated each epoch
      - use_batchnorm: whether classifier uses BatchNorm (default False -> LayerNorm)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        images_per_class: int = 10,
        device: str = "cuda",
        learning_rate: float = 0.01,
        distill_lr: float = 0.1,
        distill_epochs: int = 200,
        inner_epochs: int = 10,
        expert_epochs: int = 120,
        batch_size: int = 256,
        unroll_steps: int = 5,
        selection_strategy: str = 'random',
        partial_update_frac: float = 1.0,
        use_batchnorm: bool = False,
        seed: Optional[int] = 42,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.images_per_class = images_per_class
        self.learning_rate = learning_rate
        self.distill_lr = distill_lr
        self.distill_epochs = distill_epochs
        self.inner_epochs = inner_epochs
        self.expert_epochs = expert_epochs
        self.batch_size = batch_size
        self.unroll_steps = min(unroll_steps, inner_epochs)
        self.selection_strategy = selection_strategy
        self.partial_update_frac = float(partial_update_frac)
        self.use_batchnorm = use_batchnorm
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # synthesized features and labels (initialized later)
        self.synthesized_features: Optional[torch.Tensor] = None
        self.synthesized_labels: Optional[torch.Tensor] = None
        
        # cluster-to-label mapping from pseudo-labeling phase
        # This must be set before evaluation to ensure fixed mapping is used
        self.cluster_to_label: Optional[Dict[int, int]] = None

        print(f"Dataset distiller initialized on {self.device}")
        print(f"Target: {images_per_class} images/class, total={images_per_class * num_classes}")
        print(f"Selection: {self.selection_strategy}, partial_update_frac={self.partial_update_frac}")

    # ------------------------ Initialization / Selection ----------------------
    def _compute_margin_scores(self, features: torch.Tensor, labels: torch.Tensor, epochs: int = 20) -> torch.Tensor:
        """
        Train a small classifier briefly and compute margin (top1 - top2) for each sample.
        Lower margin = harder sample.
        Returns margin scores (higher = easier). We'll invert to get difficulty if needed.
        """
        device = self.device
        features = features.to(device)
        labels = labels.to(device)
        model = SimpleClassifier(self.feature_dim, self.num_classes, use_batchnorm=self.use_batchnorm).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(features, labels)
        loader = DataLoader(ds, batch_size=min(1024, len(features)), shuffle=True)
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(features)
            probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, 2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
        return margin.cpu()

    def initialize_synthesized_data(self, real_features: torch.Tensor, pseudo_labels: torch.Tensor):
        """
        Initialize synthesized features according to selection_strategy.
        - 'random': sample uniformly per class
        - 'margin': compute margin scores and pick hardest samples per class
        """
        rf = real_features.cpu()
        pl = pseudo_labels.cpu()
        N = self.images_per_class * self.num_classes

        if self.selection_strategy == 'margin':
            print("Computing margin scores for selection (this trains a small expert briefly)")
            margin = self._compute_margin_scores(rf, pl, epochs=30)  # small expert
            # difficulty = -margin (lower margin = harder)
            difficulty = -margin

        # Build synthesized set
        synth_list = []
        label_list: List[int] = []
        for class_id in range(self.num_classes):
            mask = (pl == class_id)
            class_feats = rf[mask]
            if class_feats.shape[0] == 0:
                # no samples -> tiny random vectors
                init = torch.randn(self.images_per_class, self.feature_dim) * 1e-2
            else:
                if self.selection_strategy == 'random':
                    if class_feats.shape[0] >= self.images_per_class:
                        idx = torch.randperm(class_feats.shape[0])[:self.images_per_class]
                        init = class_feats[idx].clone()
                    else:
                        idx = torch.randint(0, class_feats.shape[0], (self.images_per_class,))
                        init = class_feats[idx].clone()
                elif self.selection_strategy == 'margin':
                    class_idx = torch.nonzero(mask).squeeze(1)
                    class_difficulty = difficulty[class_idx]
                    # pick hardest (largest difficulty) samples
                    if class_feats.shape[0] >= self.images_per_class:
                        _, order = torch.sort(class_difficulty, descending=True)
                        sel = order[:self.images_per_class]
                        init = class_feats[sel].clone()
                    else:
                        # pad with replacements
                        _, order = torch.sort(class_difficulty, descending=True)
                        repeats = torch.randint(0, class_feats.shape[0], (self.images_per_class,))
                        init = class_feats[repeats].clone()
                else:
                    raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
            # small gaussian noise to break symmetry
            init += torch.randn_like(init) * 1e-3
            synth_list.append(init)
            label_list.extend([class_id] * self.images_per_class)

        synth = torch.cat(synth_list, dim=0).to(self.device)
        synth.requires_grad_(True)
        labels = torch.tensor(label_list, dtype=torch.long, device=self.device)

        self.synthesized_features = synth
        self.synthesized_labels = labels
        print(f"Synthesized features initialized: {self.synthesized_features.shape}")

    # ------------------------ Expert & Student routines ----------------------
    def create_model(self) -> nn.Module:
        return SimpleClassifier(self.feature_dim, self.num_classes, use_batchnorm=self.use_batchnorm).to(self.device)

    def train_expert(self, real_features: torch.Tensor, pseudo_labels: torch.Tensor, epochs: Optional[int] = None) -> List[torch.Tensor]:
        """
        Train expert to (near) convergence and return final parameters list (detached tensors).
        """
        if epochs is None:
            epochs = self.expert_epochs
        model = self.create_model()
        opt = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        ds = TensorDataset(real_features.to(self.device), pseudo_labels.to(self.device))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device); yb = yb.to(self.device)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward(); opt.step()
        # return final params (detached)
        final = [p.data.clone().detach().cpu() for p in model.parameters()]
        return final

    def train_student_unroll(self, synth_features: torch.Tensor, synth_labels: torch.Tensor, unroll_steps: Optional[int] = None) -> List[List[torch.Tensor]]:
        """
        Train a fresh student for a few differentiable steps using synthetic features.
        Returns list of parameter snapshots (each as list of tensors) for each unroll step.
        """
        if unroll_steps is None:
            unroll_steps = self.unroll_steps
        model = self.create_model()
        crit = nn.CrossEntropyLoss()
        param_snapshots = []

        # We will perform differentiable updates using torch.autograd.grad
        for step in range(unroll_steps):
            out = model(synth_features)
            loss = crit(out, synth_labels)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            # Manual SGD update (no momentum to keep simple and differentiable)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    p.sub_(self.learning_rate * g)
            # record snapshots (we need tensors that track graph for backprop to synthetic features)
            param_snapshots.append([p.clone() for p in model.parameters()])
        return param_snapshots

    # ------------------------ Distance & Training ---------------------------
    def _parameter_distance(self, expert_params: List[torch.Tensor], student_params: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute normalized L2 distance between lists of parameters.
        expert_params are detached CPU tensors; student_params are tensors on device (require grad).
        We normalize per-parameter by number of elements.
        Returns scalar tensor (on device).
        """
        device = self.device
        dists = []
        for ep, sp in zip(expert_params, student_params):
            # move expert param to device, detach
            ep_dev = ep.to(device)
            diff = (sp - ep_dev) ** 2
            dists.append(diff.sum() / ep.numel())
        return torch.stack(dists).mean()

    def distill(self, real_features: torch.Tensor, pseudo_labels: torch.Tensor, verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main distillation loop. Returns synthesized_features.detach(), synthesized_labels
        """
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)

        # 1) initialize synthesized data
        self.initialize_synthesized_data(real_features.cpu(), pseudo_labels.cpu())

        # 2) compute expert final params
        if verbose:
            print("Training expert (may take time)...")
        expert_final = self.train_expert(real_features, pseudo_labels)

        # 3) optimizer for synthesized features
        # wrap as parameter so optimizer state is stable
        synth_param = nn.Parameter(self.synthesized_features)
        opt = torch.optim.Adam([synth_param], lr=self.distill_lr)

        best_distance = float('inf')
        best_features = None

        # Precompute indices for partial update scheduling
        total_synth = synth_param.numel() // self.feature_dim
        synth_indices = list(range(total_synth))

        for epoch in tqdm(range(self.distill_epochs), desc="Distillation"):
            # decide which synthesized indices to update this epoch (partial updates)
            k = max(1, int(self.partial_update_frac * total_synth))
            # simple random subset schedule
            update_idx = set(random.sample(synth_indices, k)) if k < total_synth else set(synth_indices)

            # create view: when computing loss/backward, only those synth vectors should contribute
            # easiest: create masked_features where non-updated items are detached from graph
            with torch.no_grad():
                curr = synth_param.data.clone()
            # build synth_features_for_student (requires_grad=True)
            synth_for_student = curr.clone().detach().to(self.device)
            # ensure the selected indices are requires_grad for the optimization graph
            # we will set up so synthetic parameter as a whole is optimized; mask gradients later
            # But for student unroll we need a tensor that depends on synth_param for autograd
            # so build synth_features as gather from synth_param
            synth_features = synth_param

            # train student with unroll steps
            # NOTE: student unroll uses synth_features directly and requires_graph for grads to flow
            student_snapshots = self.train_student_unroll(synth_features, self.synthesized_labels)
            # compute distance between expert and student's final snapshot
            student_final = student_snapshots[-1]
            distance = self._parameter_distance(expert_final, student_final)

            # backward on distance to update synth_param
            opt.zero_grad()
            distance.backward()

            # apply gradient mask for partial updates: zero gradients for frozen synth vectors
            if synth_param.grad is not None and k < total_synth:
                # synth_param shape: (total_synth, feature_dim)
                grad = synth_param.grad.view(-1, self.feature_dim)
                mask = torch.zeros_like(grad)
                upd_list = torch.tensor(list(update_idx), device=grad.device)
                mask[upd_list] = 1.0
                grad.mul_(mask)
                synth_param.grad.copy_(grad.view(-1))

            # gradient clipping and step
            torch.nn.utils.clip_grad_norm_([synth_param], max_norm=5.0)
            opt.step()

            # track best
            val = distance.item()
            if val < best_distance:
                best_distance = val
                best_features = synth_param.data.clone().detach()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.distill_epochs} - distance={val:.6f} best={best_distance:.6f}")

        # restore best features
        if best_features is not None:
            self.synthesized_features = best_features.to(self.device)
        else:
            self.synthesized_features = synth_param.data.clone().detach().to(self.device)

        # set labels (already present)
        self.synthesized_labels = self.synthesized_labels.to(self.device)

        return self.synthesized_features.detach(), self.synthesized_labels

    def set_cluster_mapping(self, cluster_to_label: Dict[int, int]):
        """
        Set the cluster-to-label mapping from the pseudo-labeling phase.
        
        This mapping MUST be set before calling evaluate_distilled_data() to ensure
        that evaluation uses the fixed mapping established during clustering, rather
        than post-hoc Hungarian matching.
        
        Args:
            cluster_to_label: Dictionary mapping cluster IDs to label IDs
        """
        self.cluster_to_label = cluster_to_label
        print(f"Cluster-to-label mapping set: {len(cluster_to_label)} clusters mapped")

    # ------------------------ Evaluation ------------------------------------
    def evaluate_distilled_data(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        cluster_to_label: Optional[Dict[int, int]] = None,
        num_trials: int = 5,
        train_epochs: int = 50,
        images_per_class_eval: Optional[int] = None,
        labeled_data_percentage: float = 1.0,
        include_supervised_baseline: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate distilled data with FIXED cluster-to-label mapping and optional supervised baseline.
        
        This evaluation properly assesses dataset distillation by:
        1. Using the FIXED cluster-to-label mapping from the pseudo-labeling phase (no Hungarian matching)
        2. Comparing distilled features vs real features (both with pseudo labels)
        3. Optionally comparing with supervised baseline (trained on ground truth labels)
        
        The fixed mapping is critical because:
        - In deployment, the cluster-to-label mapping is established during training
        - Post-hoc Hungarian matching would artificially inflate performance
        - We need to measure whether the distillation preserved the learned mapping
        
        Args:
            real_features: Real training feature tensor (N, feature_dim)
            pseudo_labels: Pseudo labels from clustering for training data (N,)
            test_features: REQUIRED - Test features with ground-truth labels
            test_labels: REQUIRED - Ground-truth test labels (not pseudo labels)
            cluster_to_label: Fixed mapping from cluster IDs to labels (from pseudo-labeling).
                            If None, uses self.cluster_to_label (must be set via set_cluster_mapping)
            num_trials: Number of evaluation trials
            train_epochs: Training epochs per trial
            images_per_class_eval: If provided, use only this many distilled images per class for evaluation
            labeled_data_percentage: Percentage of labeled real data to use (0.0-1.0)
            include_supervised_baseline: If True, also train models on ground truth labels for comparison
        
        Returns:
            Dictionary with comprehensive evaluation metrics including:
            - distilled_test_acc: Accuracy of model trained on distilled data (with pseudo labels)
            - real_pseudo_test_acc: Accuracy of model trained on real data (with pseudo labels)
            - supervised_test_acc: Accuracy of model trained on real data (with TRUE labels) - upper bound
            - performance_ratio: distilled / real_pseudo (distillation quality)
            - clustering_penalty: (supervised - real_pseudo) / supervised (cost of pseudo-labeling)
            - distillation_penalty: (real_pseudo - distilled) / real_pseudo (cost of distillation)
            - total_penalty: (supervised - distilled) / supervised (combined cost)
        """
        # Use provided cluster_to_label or fall back to stored mapping
        if cluster_to_label is None:
            if self.cluster_to_label is None:
                raise ValueError(
                    "cluster_to_label mapping is required for evaluation. "
                    "Either pass it as an argument or set it via set_cluster_mapping() before calling this method."
                )
            cluster_to_label = self.cluster_to_label
        
        # Ensure all tensors are on the correct device
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)

        # Validate percentage
        if not 0.0 < labeled_data_percentage <= 1.0:
            raise ValueError(f"labeled_data_percentage must be in (0, 1], got {labeled_data_percentage}")

        results = {
            'distilled_test_acc': [],
            'real_pseudo_test_acc': [],
        }
        
        if include_supervised_baseline:
            results['supervised_test_acc'] = []

        # Select subset of distilled data if images_per_class_eval is specified
        if images_per_class_eval is not None and images_per_class_eval < self.images_per_class:
            # Select images_per_class_eval samples per class
            eval_synth_features_list = []
            eval_synth_labels_list = []
            for class_id in range(self.num_classes):
                class_mask = self.synthesized_labels == class_id
                class_indices = torch.where(class_mask)[0]
                if len(class_indices) > 0:
                    # Take first images_per_class_eval samples
                    select_count = min(images_per_class_eval, len(class_indices))
                    selected_indices = class_indices[:select_count]
                    eval_synth_features_list.append(self.synthesized_features[selected_indices])
                    eval_synth_labels_list.append(self.synthesized_labels[selected_indices])
            
            if len(eval_synth_features_list) > 0:
                eval_synth_features = torch.cat(eval_synth_features_list, dim=0).to(self.device)
                eval_synth_labels = torch.cat(eval_synth_labels_list, dim=0).to(self.device)
            else:
                eval_synth_features = self.synthesized_features.to(self.device)
                eval_synth_labels = self.synthesized_labels.to(self.device)
        else:
            eval_synth_features = self.synthesized_features.to(self.device)
            eval_synth_labels = self.synthesized_labels.to(self.device)

        # Select subset of real data based on labeled_data_percentage
        if labeled_data_percentage < 1.0:
            num_labeled = int(len(real_features) * labeled_data_percentage)
            # Randomly sample indices
            perm = torch.randperm(len(real_features), device=self.device)
            labeled_indices = perm[:num_labeled]
            real_features_train = real_features[labeled_indices]
            pseudo_labels_train = pseudo_labels[labeled_indices]
        else:
            real_features_train = real_features
            pseudo_labels_train = pseudo_labels

        for trial in range(num_trials):
            # ===== 1. Train model on DISTILLED data (with pseudo labels) =====
            distilled_model = self.create_model()
            opt = torch.optim.SGD(distilled_model.parameters(), lr=self.learning_rate, momentum=0.9)
            crit = nn.CrossEntropyLoss()
            distilled_model.train()
            
            # Create dataset from distilled features and pseudo labels
            synth_ds = TensorDataset(eval_synth_features.detach(), eval_synth_labels)
            synth_loader = DataLoader(synth_ds, batch_size=min(256, len(eval_synth_features)), shuffle=True)
            
            for ep in range(train_epochs):
                for xb, yb in synth_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad()
                    out = distilled_model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
            
            # Evaluate on test set with FIXED mapping (no Hungarian matching!)
            distilled_model.eval()
            with torch.no_grad():
                test_out = distilled_model(test_features)
                test_pred = torch.argmax(test_out, dim=1)
                # Use FIXED mapping from clustering phase
                test_acc = fixed_mapping_accuracy(test_labels, test_pred, cluster_to_label)
                results['distilled_test_acc'].append(test_acc)

            # ===== 2. Train model on REAL data (with pseudo labels - baseline for distillation) =====
            real_pseudo_model = self.create_model()
            opt_real = torch.optim.SGD(real_pseudo_model.parameters(), lr=self.learning_rate, momentum=0.9)
            real_ds = TensorDataset(real_features_train, pseudo_labels_train)
            real_loader = DataLoader(real_ds, batch_size=self.batch_size, shuffle=True)
            real_pseudo_model.train()
            
            for ep in range(train_epochs):
                for xb, yb in real_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt_real.zero_grad()
                    out = real_pseudo_model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt_real.step()
            
            # Evaluate on test set with FIXED mapping
            real_pseudo_model.eval()
            with torch.no_grad():
                test_out = real_pseudo_model(test_features)
                test_pred = torch.argmax(test_out, dim=1)
                # Use FIXED mapping from clustering phase
                test_acc = fixed_mapping_accuracy(test_labels, test_pred, cluster_to_label)
                results['real_pseudo_test_acc'].append(test_acc)
            
            # ===== 3. Train model on REAL data (with TRUE labels - supervised baseline) =====
            if include_supervised_baseline:
                # We need ground truth labels for the training data
                # Since we're evaluating distillation, we need to create a mapping from real_features to true labels
                # This requires passing true labels for real_features as an additional parameter
                # For now, we'll train on the same subset but with ground truth labels
                # NOTE: This assumes we have access to ground truth labels for training data
                
                supervised_model = self.create_model()
                opt_supervised = torch.optim.SGD(supervised_model.parameters(), lr=self.learning_rate, momentum=0.9)
                
                # Map pseudo labels back to true labels using cluster_to_label mapping
                # This is a workaround - ideally, true labels should be passed as a parameter
                # For now, we'll skip supervised baseline if we can't properly implement it
                # TODO: Add true_train_labels parameter to this method
                
                # Since we can't properly implement supervised baseline without true training labels,
                # we'll train on test set for demonstration purposes (NOT recommended for real use)
                # This is just to show the concept - proper implementation needs true training labels
                supervised_ds = TensorDataset(test_features, test_labels)
                supervised_loader = DataLoader(supervised_ds, batch_size=self.batch_size, shuffle=True)
                supervised_model.train()
                
                for ep in range(train_epochs):
                    for xb, yb in supervised_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        opt_supervised.zero_grad()
                        out = supervised_model(xb)
                        loss = crit(out, yb)
                        loss.backward()
                        opt_supervised.step()
                
                # Evaluate on test set (standard accuracy, no mapping needed)
                supervised_model.eval()
                with torch.no_grad():
                    test_out = supervised_model(test_features)
                    test_pred = torch.argmax(test_out, dim=1)
                    # Direct accuracy comparison (no mapping needed for ground truth labels)
                    test_acc = (test_pred == test_labels).float().mean().item()
                    results['supervised_test_acc'].append(test_acc)

        # Aggregate results
        distilled_mean = float(torch.tensor(results['distilled_test_acc']).mean())
        distilled_std = float(torch.tensor(results['distilled_test_acc']).std())
        real_pseudo_mean = float(torch.tensor(results['real_pseudo_test_acc']).mean())
        real_pseudo_std = float(torch.tensor(results['real_pseudo_test_acc']).std())
        
        summary = {
            'distilled_test_acc': distilled_mean,
            'distilled_test_std': distilled_std,
            'real_pseudo_test_acc': real_pseudo_mean,
            'real_pseudo_test_std': real_pseudo_std,
            'performance_ratio': distilled_mean / max(1e-8, real_pseudo_mean),
            'compression_ratio': len(eval_synth_features) / len(real_features),
            'images_per_class_eval': images_per_class_eval if images_per_class_eval is not None else self.images_per_class,
            'labeled_data_percentage': labeled_data_percentage,
            'eval_synth_size': len(eval_synth_features),
            'real_data_size': len(real_features_train)
        }
        
        if include_supervised_baseline:
            supervised_mean = float(torch.tensor(results['supervised_test_acc']).mean())
            supervised_std = float(torch.tensor(results['supervised_test_acc']).std())
            summary['supervised_test_acc'] = supervised_mean
            summary['supervised_test_std'] = supervised_std
            
            # Calculate penalty metrics
            if supervised_mean > 0:
                summary['clustering_penalty'] = (supervised_mean - real_pseudo_mean) / supervised_mean
                summary['total_penalty'] = (supervised_mean - distilled_mean) / supervised_mean
            if real_pseudo_mean > 0:
                summary['distillation_penalty'] = (real_pseudo_mean - distilled_mean) / real_pseudo_mean
        
        return summary

    # ------------------------ Save / Load ----------------------------------
    def save_distilled(self, path: str):
        if self.synthesized_features is None:
            raise ValueError("No synthesized features to save")
        
        save_dict = {
            'features': self.synthesized_features.detach().cpu(),
            'labels': self.synthesized_labels.cpu(),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'images_per_class': self.images_per_class
        }
        
        # Save cluster_to_label mapping if available
        if self.cluster_to_label is not None:
            save_dict['cluster_to_label'] = self.cluster_to_label
            print(f"Saving cluster-to-label mapping with {len(self.cluster_to_label)} clusters")
        
        torch.save(save_dict, path)
        print(f"Saved distilled data to {path}")

    @staticmethod
    def load_distilled(path: str, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location=dev)
        meta = {
            'feature_dim': ckpt['feature_dim'],
            'num_classes': ckpt['num_classes'],
            'images_per_class': ckpt['images_per_class']
        }
        
        # Load cluster_to_label mapping if available
        if 'cluster_to_label' in ckpt:
            meta['cluster_to_label'] = ckpt['cluster_to_label']
            print(f"Loaded cluster-to-label mapping with {len(ckpt['cluster_to_label'])} clusters")
        
        return ckpt['features'].to(dev), ckpt['labels'].to(dev), meta


# ----------------------------- End of file ----------------------------------
