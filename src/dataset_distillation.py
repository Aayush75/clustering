"""
Dataset distillation module using pseudo labels from clustering.

This module implements supervised dataset distillation as described in
"Dataset Distillation by Matching Training Trajectories" (arXiv:2406.18561).
The key idea is to synthesize a small set of images that, when trained on,
produces similar model behavior as training on the full dataset.

We use pseudo labels generated from clustering to guide the distillation process.
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
    
    This model is used during the distillation process to learn from
    the synthesized images and ensure they capture the essential patterns.
    """
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of classes
            hidden_dims: List of hidden layer dimensions (default: [512, 256])
        """
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
        """Forward pass through the classifier."""
        return self.network(x)


class DatasetDistiller:
    """
    Dataset distillation using pseudo labels from clustering.
    
    This class implements the dataset distillation algorithm that creates
    a small synthetic dataset from the original dataset using pseudo labels.
    The distilled dataset preserves the essential patterns and can be used
    for efficient training.
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
        batch_size: int = 256
    ):
        """
        Initialize the dataset distiller.
        
        Args:
            feature_dim: Dimension of input features
            num_classes: Number of classes (clusters)
            images_per_class: Number of synthetic images per class
            device: Device to run computations on
            learning_rate: Learning rate for classifier training
            distill_lr: Learning rate for distilled image optimization
            distill_epochs: Number of distillation epochs
            inner_epochs: Number of inner training epochs per distillation step
            batch_size: Batch size for training
        """
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
        self.batch_size = batch_size
        
        # Synthesized features (learnable parameters)
        self.synthesized_features = None
        self.synthesized_labels = None
        
        print(f"Dataset distiller initialized on {self.device}")
        print(f"Target: {images_per_class} images per class, {num_classes} classes")
        print(f"Total synthesized images: {images_per_class * num_classes}")
    
    def initialize_synthesized_data(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor
    ):
        """
        Initialize synthesized features by sampling from real features.
        
        Args:
            real_features: Real feature tensor (n_samples, feature_dim)
            pseudo_labels: Pseudo labels (n_samples,)
        """
        print("Initializing synthesized features...")
        
        # Ensure inputs are on the correct device
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        
        synthesized_list = []
        label_list = []
        
        # For each class, sample images_per_class features from real data
        for class_id in range(self.num_classes):
            # Find samples belonging to this class
            class_mask = pseudo_labels == class_id
            class_features = real_features[class_mask]
            
            if len(class_features) == 0:
                # If no samples for this class, use random initialization
                init_features = torch.randn(
                    self.images_per_class, self.feature_dim,
                    device=self.device
                ) * 0.01
            elif len(class_features) < self.images_per_class:
                # If fewer samples than needed, sample with replacement
                indices = torch.randint(
                    0, len(class_features),
                    (self.images_per_class,),
                    device=self.device
                )
                init_features = class_features[indices].clone()
            else:
                # Sample without replacement
                indices = torch.randperm(len(class_features), device=self.device)[:self.images_per_class]
                init_features = class_features[indices].clone()
            
            # Add small noise for diversity
            init_features += torch.randn_like(init_features) * 0.01
            
            synthesized_list.append(init_features)
            label_list.extend([class_id] * self.images_per_class)
        
        # Concatenate all synthesized features
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
    
    def train_on_real_data(
        self,
        model: nn.Module,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        epochs: int
    ) -> List[torch.Tensor]:
        """
        Train model on real data and record parameter trajectories.
        
        Args:
            model: Model to train
            real_features: Real feature tensor
            pseudo_labels: Pseudo labels
            epochs: Number of training epochs
            
        Returns:
            List of parameter snapshots at each epoch
        """
        # Ensure inputs are on the correct device
        real_features = real_features.to(self.device)
        pseudo_labels = pseudo_labels.to(self.device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Create data loader
        dataset = TensorDataset(real_features, pseudo_labels)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=False, drop_last=False
        )
        
        param_trajectories = []
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Record parameter snapshot
            param_snapshot = [p.data.clone().detach() for p in model.parameters()]
            param_trajectories.append(param_snapshot)
        
        return param_trajectories
    
    def train_on_synthetic_data(
        self,
        model: nn.Module,
        epochs: int,
        retain_graph: bool = False
    ) -> Tuple[List[torch.Tensor], torch.nn.Module]:
        """
        Train model on synthetic data and record parameter trajectories.
        
        This version maintains computational graph for gradient flow.
        
        Args:
            model: Model to train
            epochs: Number of training epochs
            retain_graph: Whether to retain computational graph
            
        Returns:
            Tuple of (parameter trajectories, trained model)
        """
        criterion = nn.CrossEntropyLoss()
        
        param_trajectories = []
        
        # Manual SGD updates to maintain gradient flow
        model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = model(self.synthesized_features)
            loss = criterion(outputs, self.synthesized_labels)
            
            # Backward pass
            grads = torch.autograd.grad(
                loss, model.parameters(),
                create_graph=True,
                retain_graph=True
            )
            
            # Update parameters with SGD
            with torch.no_grad():
                for param, grad in zip(model.parameters(), grads):
                    param.data = param.data - self.learning_rate * grad.data
            
            # Record parameter snapshot (maintain graph)
            param_snapshot = [p.clone() for p in model.parameters()]
            param_trajectories.append(param_snapshot)
        
        return param_trajectories, model
    
    def compute_trajectory_distance(
        self,
        real_trajectory: List[torch.Tensor],
        synthetic_trajectory: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute distance between two parameter trajectories.
        
        Args:
            real_trajectory: List of parameter snapshots from real data training
            synthetic_trajectory: List of parameter snapshots from synthetic data training
            
        Returns:
            Scalar distance value with gradient
        """
        distances = []
        
        # Compare trajectories at each time step
        min_len = min(len(real_trajectory), len(synthetic_trajectory))
        
        for t in range(min_len):
            real_params = real_trajectory[t]
            synthetic_params = synthetic_trajectory[t]
            
            # Compute L2 distance for each parameter
            for real_p, synth_p in zip(real_params, synthetic_params):
                # Detach real parameters (they don't need gradients)
                # Keep synthetic parameters' computational graph for backprop
                dist = torch.sum((real_p.detach() - synth_p) ** 2)
                distances.append(dist)
        
        # Sum all distances efficiently
        total_distance = torch.stack(distances).sum()
        
        return total_distance
    
    def distill(
        self,
        real_features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform dataset distillation by matching training trajectories.
        
        Args:
            real_features: Real feature tensor (n_samples, feature_dim)
            pseudo_labels: Pseudo labels (n_samples,)
            verbose: Whether to print progress
            
        Returns:
            Tuple of (synthesized_features, synthesized_labels)
        """
        print("\n" + "="*80)
        print("Dataset Distillation")
        print("="*80)
        
        # Initialize synthesized data
        self.initialize_synthesized_data(real_features, pseudo_labels)
        
        # Optimizer for synthesized features
        feature_optimizer = torch.optim.Adam([self.synthesized_features], lr=self.distill_lr)
        
        # Distillation loop
        best_distance = float('inf')
        best_features = None
        
        print(f"\nStarting distillation for {self.distill_epochs} epochs...")
        
        for epoch in tqdm(range(self.distill_epochs), desc="Distillation"):
            # Train a model on real data to get reference trajectory
            real_model = self.create_model()
            real_trajectory = self.train_on_real_data(
                real_model, real_features, pseudo_labels, self.inner_epochs
            )
            
            # Train a model on synthetic data to get synthetic trajectory
            synthetic_model = self.create_model()
            synthetic_trajectory, _ = self.train_on_synthetic_data(
                synthetic_model, self.inner_epochs
            )
            
            # Compute trajectory distance
            distance = self.compute_trajectory_distance(real_trajectory, synthetic_trajectory)
            
            # Update synthesized features to minimize distance
            feature_optimizer.zero_grad()
            distance.backward()
            feature_optimizer.step()
            
            # Track best result
            if distance.item() < best_distance:
                best_distance = distance.item()
                best_features = self.synthesized_features.data.clone()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.distill_epochs}, Distance: {distance.item():.4f}, "
                      f"Best: {best_distance:.4f}")
        
        # Use best features
        if best_features is not None:
            self.synthesized_features = best_features
        
        print(f"\nDistillation complete! Best distance: {best_distance:.4f}")
        
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
        Evaluate the quality of distilled data by comparing models trained on
        distilled vs. real data.
        
        Args:
            real_features: Real training features
            pseudo_labels: Pseudo labels for real features
            test_features: Optional test features
            test_labels: Optional test labels
            num_trials: Number of trials to average over
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print("Evaluating Distilled Data")
        print("="*80)
        
        # Ensure inputs are on the correct device
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
            for _ in range(50):  # Train for 50 epochs
                optimizer.zero_grad()
                outputs = distilled_model(self.synthesized_features)
                loss = criterion(outputs, self.synthesized_labels)
                loss.backward()
                optimizer.step()
            
            # Evaluate on training data
            distilled_model.eval()
            with torch.no_grad():
                train_outputs = distilled_model(real_features)
                train_preds = torch.argmax(train_outputs, dim=1)
                train_acc = (train_preds == pseudo_labels).float().mean().item()
                results['distilled_train_acc'].append(train_acc)
                
                # Evaluate on test data if available
                if test_features is not None:
                    test_outputs = distilled_model(test_features)
                    test_preds = torch.argmax(test_outputs, dim=1)
                    test_acc = (test_preds == test_labels).float().mean().item()
                    results['distilled_test_acc'].append(test_acc)
            
            # Train on real data
            real_model = self.create_model()
            real_trajectory = self.train_on_real_data(
                real_model, real_features, pseudo_labels, 50
            )
            
            # Evaluate on training data
            real_model.eval()
            with torch.no_grad():
                train_outputs = real_model(real_features)
                train_preds = torch.argmax(train_outputs, dim=1)
                train_acc = (train_preds == pseudo_labels).float().mean().item()
                results['real_train_acc'].append(train_acc)
                
                # Evaluate on test data if available
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
        """
        Save distilled features and labels to disk.
        
        Args:
            path: Path where to save the distilled data
        """
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
        """
        Load distilled data from disk.
        
        Args:
            path: Path to the saved distilled data
            device: Device to load tensors to
            
        Returns:
            Tuple of (features, labels, metadata)
        """
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
