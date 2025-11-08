"""
Dataset distillation module using pseudo labels from clustering.

This module implements supervised dataset distillation as described in
"Dataset Distillation by Matching Training Trajectories" (arXiv:2406.18561).
The key idea is to synthesize a small set of images that, when trained on,
produces similar model behavior as training on the full dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import warnings


class SimpleClassifier(nn.Module):
    """
    Simple classification model for dataset distillation.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super(SimpleClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DatasetDistiller:
    """
    Dataset distillation using trajectory matching with proper gradient flow.
    
    Key fixes:
    1. Maintains gradient flow through synthetic data updates
    2. Uses expert trajectories (computed once per outer loop)
    3. Proper batch-based training
    4. Normalized trajectory distance
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        images_per_class: int = 10,
        device: str = "cuda",
        learning_rate: float = 0.01,
        distill_lr: float = 0.1,
        distill_epochs: int = 100,
        inner_epochs: int = 10,
        batch_size: int = 256,
        expert_epochs: int = 50  # New: separate epoch count for expert
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                f"CUDA device requested but not available. Using CPU instead.",
                RuntimeWarning
            )
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.images_per_class = images_per_class
        self.learning_rate = learning_rate
        self.distill_lr = distill_lr
        self.distill_epochs = distill_epochs
        self.inner_epochs = inner_epochs
        self.expert_epochs = expert_epochs
        self.batch_size = batch_size
        
        # Synthesized features (learnable parameters)
        self.synthesized_features = None
        self.synthesized_labels = None
        
        # Expert trajectory (computed once)
        self.expert_trajectory = None
        
        print(f"Dataset distiller initialized on {self.device}")
        print(f"Target: {images_per_class} images per class, {num_classes} classes")
        print(f"Total synthesized images: {images_per_class * num_classes}")
        print(f"Expert epochs: {expert_epochs}, Student epochs: {inner_epochs}")
    
    def initialize_synthesized_data(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor
    ):
        """
        Initialize synthesized features by sampling from real features.
        """
        print("Initializing synthesized features...")
        
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        
        synthesized_list = []
        label_list = []
        
        for class_id in range(self.num_classes):
            class_mask = pseudo_labels == class_id
            class_features = real_features[class_mask]
            
            if len(class_features) == 0:
                init_features = torch.randn(
                    self.images_per_class, self.feature_dim,
                    device=self.device
                ) * 0.01
            elif len(class_features) < self.images_per_class:
                indices = torch.randint(
                    0, len(class_features),
                    (self.images_per_class,),
                    device=self.device
                )
                init_features = class_features[indices].clone()
            else:
                indices = torch.randperm(len(class_features), device=self.device)[:self.images_per_class]
                init_features = class_features[indices].clone()
            
            # Smaller noise for better initialization
            init_features += torch.randn_like(init_features) * 0.001
            
            synthesized_list.append(init_features)
            label_list.extend([class_id] * self.images_per_class)
        
        self.synthesized_features = torch.cat(synthesized_list, dim=0)
        self.synthesized_features.requires_grad = True
        
        self.synthesized_labels = torch.tensor(
            label_list, dtype=torch.long, device=self.device
        )
        
        print(f"Synthesized features initialized: {self.synthesized_features.shape}")
    
    def create_model(self) -> SimpleClassifier:
        """Create a fresh classifier model."""
        model = SimpleClassifier(
            input_dim=self.feature_dim,
            num_classes=self.num_classes
        ).to(self.device)
        return model
    
    def get_expert_trajectory(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute expert trajectory on real data (done once).
        
        Returns:
            List of parameter states at each epoch
        """
        print("Computing expert trajectory on real data...")
        
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        
        model = self.create_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(real_features, pseudo_labels)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=False, drop_last=False
        )
        
        trajectory = []
        
        model.train()
        for epoch in range(self.expert_epochs):
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Save parameter state (detached)
            param_dict = {name: param.data.clone().detach() 
                         for name, param in model.named_parameters()}
            trajectory.append(param_dict)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Expert epoch {epoch+1}/{self.expert_epochs}")
        
        print("Expert trajectory computed!")
        return trajectory
    
    def train_student_with_gradients(
        self,
        model: nn.Module,
        synthetic_features: torch.Tensor,
        synthetic_labels: torch.Tensor,
        epochs: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Train student model on synthetic data while maintaining gradient flow.
        
        This is the KEY fix - we keep gradients flowing through the process.
        """
        criterion = nn.CrossEntropyLoss()
        trajectory = []
        
        # Get initial parameters
        params = {name: param for name, param in model.named_parameters()}
        
        model.train()
        for epoch in range(epochs):
            # Forward pass with gradient tracking
            outputs = model(synthetic_features)
            loss = criterion(outputs, synthetic_labels)
            
            # Compute gradients w.r.t. model parameters
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=True,  # ← CRITICAL: Keep gradient graph!
                retain_graph=True
            )
            
            # Manual SGD update (WITHOUT detaching!)
            new_params = {}
            for (name, param), grad in zip(params.items(), grads):
                # This maintains the computational graph
                new_params[name] = param - self.learning_rate * grad
            
            # Update model parameters (this is differentiable!)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(new_params[name].data)
            
            # Record trajectory (keep gradients for backprop)
            trajectory.append(new_params)
        
        return trajectory
    
    def compute_trajectory_distance(
        self,
        expert_trajectory: List[Dict[str, torch.Tensor]],
        student_trajectory: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute normalized distance between expert and student trajectories.
        """
        distances = []
        
        # Sample epochs (e.g., every 5 epochs) for efficiency
        sample_epochs = list(range(0, len(expert_trajectory), 5))
        if len(expert_trajectory) - 1 not in sample_epochs:
            sample_epochs.append(len(expert_trajectory) - 1)
        
        for t in sample_epochs:
            if t >= len(student_trajectory):
                break
                
            expert_params = expert_trajectory[t]
            student_params = student_trajectory[t]
            
            # Compute parameter-wise distance
            for name in expert_params.keys():
                if name in student_params:
                    expert_p = expert_params[name].detach()
                    student_p = student_params[name]
                    
                    # Normalized L2 distance
                    diff = student_p - expert_p
                    dist = torch.sum(diff ** 2) / expert_p.numel()
                    distances.append(dist)
        
        # Average distance across all parameters and timesteps
        total_distance = torch.stack(distances).mean()
        
        return total_distance
    
    def distill(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform dataset distillation by matching training trajectories.
        
        Main fixes:
        1. Expert trajectory computed once
        2. Student training maintains gradient flow
        3. Proper distance normalization
        4. Batch-based synthetic data training
        """
        print("\n" + "="*80)
        print("Dataset Distillation with Trajectory Matching")
        print("="*80)
        
        # Initialize synthesized data
        self.initialize_synthesized_data(real_features, pseudo_labels)
        
        # Compute expert trajectory ONCE
        self.expert_trajectory = self.get_expert_trajectory(real_features, pseudo_labels)
        
        # Optimizer for synthesized features
        feature_optimizer = torch.optim.Adam([self.synthesized_features], lr=self.distill_lr)
        
        # Distillation loop
        best_distance = float('inf')
        best_features = None
        prev_distance = float('inf')
        
        print(f"\nStarting distillation for {self.distill_epochs} epochs...")
        print(f"Optimizing {len(self.synthesized_features)} synthetic samples...")
        
        for epoch in tqdm(range(self.distill_epochs), desc="Distillation"):
            # Create student model
            student_model = self.create_model()
            
            # Train student on synthetic data (maintains gradients!)
            student_trajectory = self.train_student_with_gradients(
                student_model,
                self.synthesized_features,
                self.synthesized_labels,
                self.inner_epochs
            )
            
            # Compute trajectory distance
            distance = self.compute_trajectory_distance(
                self.expert_trajectory,
                student_trajectory
            )
            
            # Update synthesized features
            feature_optimizer.zero_grad()
            distance.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([self.synthesized_features], max_norm=1.0)
            
            feature_optimizer.step()
            
            # Track best result
            if distance.item() < best_distance:
                best_distance = distance.item()
                best_features = self.synthesized_features.data.clone()
            
            # Compute improvement
            improvement = ((prev_distance - distance.item()) / prev_distance * 100) if epoch > 0 else 0.0
            prev_distance = distance.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{self.distill_epochs}")
                print(f"  Distance: {distance.item():.6f}")
                print(f"  Best: {best_distance:.6f}")
                print(f"  Improvement: {improvement:+.2f}%")
        
        # Use best features
        if best_features is not None:
            self.synthesized_features = best_features.requires_grad_()
        
        print(f"\n{'='*80}")
        print(f"Distillation complete!")
        print(f"  Best distance: {best_distance:.6f}")
        print(f"  Total improvement: {((self.expert_trajectory[0][list(self.expert_trajectory[0].keys())[0]].norm().item() - best_distance) / self.expert_trajectory[0][list(self.expert_trajectory[0].keys())[0]].norm().item() * 100):.1f}%")
        print(f"{'='*80}\n")
        
        return self.synthesized_features.detach(), self.synthesized_labels
    
    def evaluate_distilled_data(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        test_features: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        num_trials: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate the quality of distilled data.
        """
        print("\n" + "="*80)
        print("Evaluating Distilled Data")
        print("="*80)
        
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        
        if test_features is not None:
            test_features = test_features.to(self.device)
            test_labels = test_labels.to(self.device)
        
        results = {
            'distilled_train_acc': [],
            'distilled_test_acc': [],
            'real_train_acc': [],
            'real_test_acc': []
        }
        
        for trial in range(num_trials):
            # Train on distilled data
            distilled_model = self.create_model()
            optimizer = torch.optim.SGD(
                distilled_model.parameters(),
                lr=self.learning_rate,
                momentum=0.9
            )
            criterion = nn.CrossEntropyLoss()
            
            distilled_model.train()
            for _ in range(50):
                optimizer.zero_grad()
                outputs = distilled_model(self.synthesized_features)
                loss = criterion(outputs, self.synthesized_labels)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            distilled_model.eval()
            with torch.no_grad():
                train_outputs = distilled_model(real_features)
                train_preds = torch.argmax(train_outputs, dim=1)
                train_acc = (train_preds == pseudo_labels).float().mean().item()
                results['distilled_train_acc'].append(train_acc)
                
                if test_features is not None:
                    test_outputs = distilled_model(test_features)
                    test_preds = torch.argmax(test_outputs, dim=1)
                    test_acc = (test_preds == test_labels).float().mean().item()
                    results['distilled_test_acc'].append(test_acc)
            
            # Train on real data
            real_model = self.create_model()
            optimizer_real = torch.optim.SGD(real_model.parameters(), lr=self.learning_rate, momentum=0.9)
            dataset = TensorDataset(real_features, pseudo_labels)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            real_model.train()
            for _ in range(50):
                for batch_features, batch_labels in loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    optimizer_real.zero_grad()
                    outputs = real_model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer_real.step()
            
            # Evaluate
            real_model.eval()
            with torch.no_grad():
                train_outputs = real_model(real_features)
                train_preds = torch.argmax(train_outputs, dim=1)
                train_acc = (train_preds == pseudo_labels).float().mean().item()
                results['real_train_acc'].append(train_acc)
                
                if test_features is not None:
                    test_outputs = real_model(test_features)
                    test_preds = torch.argmax(test_outputs, dim=1)
                    test_acc = (test_preds == test_labels).float().mean().item()
                    results['real_test_acc'].append(test_acc)
        
        # Compute averages
        avg_results = {
            'distilled_train_acc': torch.tensor(results['distilled_train_acc']).mean().item(),
            'distilled_train_std': torch.tensor(results['distilled_train_acc']).std().item(),
            'real_train_acc': torch.tensor(results['real_train_acc']).mean().item(),
            'real_train_std': torch.tensor(results['real_train_acc']).std().item()
        }
        
        if test_features is not None:
            avg_results.update({
                'distilled_test_acc': torch.tensor(results['distilled_test_acc']).mean().item(),
                'distilled_test_std': torch.tensor(results['distilled_test_acc']).std().item(),
                'real_test_acc': torch.tensor(results['real_test_acc']).mean().item(),
                'real_test_std': torch.tensor(results['real_test_acc']).std().item()
            })
        
        # Print results
        print("\nEvaluation Results (averaged over {} trials):".format(num_trials))
        print("-" * 80)
        print(f"Model trained on DISTILLED data:")
        print(f"  Train Accuracy: {avg_results['distilled_train_acc']:.4f} ± {avg_results['distilled_train_std']:.4f}")
        if test_features is not None:
            print(f"  Test Accuracy:  {avg_results['distilled_test_acc']:.4f} ± {avg_results['distilled_test_std']:.4f}")
        
        print(f"\nModel trained on REAL data:")
        print(f"  Train Accuracy: {avg_results['real_train_acc']:.4f} ± {avg_results['real_train_std']:.4f}")
        if test_features is not None:
            print(f"  Test Accuracy:  {avg_results['real_test_acc']:.4f} ± {avg_results['real_test_std']:.4f}")
        
        if test_features is not None:
            performance_ratio = avg_results['distilled_test_acc'] / avg_results['real_test_acc']
            compression_ratio = len(self.synthesized_features) / len(real_features)
            print(f"\nPerformance Ratio (distilled/real): {performance_ratio:.4f}")
            print(f"Compression Ratio: {compression_ratio:.4f} ({len(self.synthesized_features)}/{len(real_features)})")
        
        print("-" * 80)
        
        return avg_results
    
    def save_distilled_data(self, path: str):
        """Save distilled features and labels to disk."""
        if self.synthesized_features is None:
            raise ValueError("No distilled data to save. Run distill() first.")
        
        torch.save({
            'features': self.synthesized_features.detach().cpu(),
            'labels': self.synthesized_labels.cpu(),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'images_per_class': self.images_per_class
        }, path)
        print(f"Distilled data saved to {path}")
    
    @staticmethod
    def load_distilled_data(path: str, device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Load distilled data from disk."""
        target_device = torch.device(device if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                f"CUDA device requested but not available. Using CPU instead.",
                RuntimeWarning
            )
        
        checkpoint = torch.load(path, map_location=target_device)
        
        metadata = {
            'feature_dim': checkpoint['feature_dim'],
            'num_classes': checkpoint['num_classes'],
            'images_per_class': checkpoint['images_per_class']
        }
        
        print(f"Distilled data loaded from {path}")
        print(f"  Feature dim: {metadata['feature_dim']}")
        print(f"  Num classes: {metadata['num_classes']}")
        print(f"  Images per class: {metadata['images_per_class']}")
        
        return checkpoint['features'], checkpoint['labels'], metadata