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
from src.evaluation import cluster_accuracy


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

    # ------------------------ Evaluation ------------------------------------
    def evaluate_distilled_data(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_trials: int = 5,
        train_epochs: int = 50,
        images_per_class_eval: Optional[int] = None,
        labeled_data_percentage: float = 1.0
    ) -> Dict[str, float]:
        """
        Evaluate distilled data by training on distilled/real data and testing on held-out test set with TRUE labels.
        
        This removes data leakage by:
        1. Training models on distilled data (with pseudo labels from clustering)
        2. Testing ONLY on a separate held-out test set with ground-truth labels
        3. No evaluation on training data with pseudo labels
        
        Args:
            real_features: Real training feature tensor (N, feature_dim)
            pseudo_labels: Pseudo labels from clustering for training data (N,)
            test_features: REQUIRED - Test features with ground-truth labels
            test_labels: REQUIRED - Ground-truth test labels (not pseudo labels)
            num_trials: Number of evaluation trials
            train_epochs: Training epochs per trial
            images_per_class_eval: If provided, use only this many distilled images per class for evaluation
            labeled_data_percentage: Percentage of labeled real data to use (0.0-1.0), simulates semi-supervised learning
        
        Returns:
            Dictionary with test accuracy metrics (no train accuracy to avoid confusion)
        """
        # Ensure all tensors are on the correct device
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)

        # Sanity checks to catch accidental passing of pseudo-labels as test labels
        if test_features.shape[0] != test_labels.shape[0]:
            raise ValueError(
                f"test_features and test_labels length mismatch: "
                f"{test_features.shape[0]} vs {test_labels.shape[0]}"
            )
        
        # test_labels must be integer class ids in [0, num_classes)
        if test_labels.dtype not in (torch.int64, torch.long):
            test_labels = test_labels.long()
        
        if test_labels.min().item() < 0 or test_labels.max().item() >= self.num_classes:
            raise ValueError(
                f"test_labels contains values outside expected class range [0, {self.num_classes}). "
                f"Got range [{test_labels.min().item()}, {test_labels.max().item()}]. "
                f"Are you passing pseudo-labels from clustering by mistake? "
                f"test_labels must be ground-truth labels from the dataset."
            )

        # Validate percentage
        if not 0.0 < labeled_data_percentage <= 1.0:
            raise ValueError(f"labeled_data_percentage must be in (0.0, 1.0], got {labeled_data_percentage}")

        results = {
            'distilled_test_acc': [],
            'real_test_acc': []
        }

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
            # ===== Train model on DISTILLED data (with pseudo labels) =====
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
            
            # ===== Evaluate ONLY on test set with TRUE labels =====
            distilled_model.eval()
            with torch.no_grad():
                test_out = distilled_model(test_features)
                test_pred = torch.argmax(test_out, dim=1)
                # Use Hungarian matching / cluster-aware accuracy function
                test_acc = cluster_accuracy(test_labels, test_pred)
                results['distilled_test_acc'].append(test_acc)

            # ===== Train model on REAL data (baseline, with pseudo labels) =====
            real_model = self.create_model()
            opt_real = torch.optim.SGD(real_model.parameters(), lr=self.learning_rate, momentum=0.9)
            real_ds = TensorDataset(real_features_train, pseudo_labels_train)
            real_loader = DataLoader(real_ds, batch_size=self.batch_size, shuffle=True)
            real_model.train()
            
            for ep in range(train_epochs):
                for xb, yb in real_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt_real.zero_grad()
                    out = real_model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt_real.step()
            
            # ===== Evaluate ONLY on test set with TRUE labels =====
            real_model.eval()
            with torch.no_grad():
                test_out = real_model(test_features)
                test_pred = torch.argmax(test_out, dim=1)
                # Use Hungarian matching / cluster-aware accuracy function
                test_acc = cluster_accuracy(test_labels, test_pred)
                results['real_test_acc'].append(test_acc)

        # Aggregate results - only test accuracy matters
        summary = {
            'distilled_test_acc': float(torch.tensor(results['distilled_test_acc']).mean()),
            'distilled_test_std': float(torch.tensor(results['distilled_test_acc']).std()),
            'real_test_acc': float(torch.tensor(results['real_test_acc']).mean()),
            'real_test_std': float(torch.tensor(results['real_test_acc']).std()),
            'performance_ratio': float(torch.tensor(results['distilled_test_acc']).mean()) / max(1e-8, float(torch.tensor(results['real_test_acc']).mean())),
            'compression_ratio': len(eval_synth_features) / len(real_features),
            'images_per_class_eval': images_per_class_eval if images_per_class_eval is not None else self.images_per_class,
            'labeled_data_percentage': labeled_data_percentage,
            'eval_synth_size': len(eval_synth_features),
            'real_data_size': len(real_features_train)
        }

        return summary

    # ------------------------ Save / Load ----------------------------------
    def save_distilled(self, path: str):
        if self.synthesized_features is None:
            raise ValueError("No synthesized features to save")
        torch.save({
            'features': self.synthesized_features.detach().cpu(),
            'labels': self.synthesized_labels.cpu(),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'images_per_class': self.images_per_class
        }, path)
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
        return ckpt['features'].to(dev), ckpt['labels'].to(dev), meta


# ----------------------------- End of file ----------------------------------
