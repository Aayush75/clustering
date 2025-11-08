"""
TEMI (Transformation-Equivariant Multi-Instance) clustering implementation.

This module implements the TEMI clustering algorithm as described in the paper:
"Self-Supervised Clustering with Deep Learning" (arXiv:2303.17896)

TEMI uses transformation equivariance and multi-instance learning to improve
clustering performance without requiring labeled data during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import os
import warnings


class TEMIClusteringHead(nn.Module):
    """
    Clustering head for TEMI algorithm.
    
    This neural network projects features into a clustering-friendly space
    and produces cluster assignment probabilities. The head is trained using
    transformation equivariance and multi-instance objectives.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_clusters: int,
        hidden_dim: int = 2048,
        projection_dim: int = 256
    ):
        """
        Initialize the TEMI clustering head.
        
        Args:
            input_dim: Dimension of input features (from DINOv2)
            num_clusters: Number of clusters (k=100 for CIFAR100)
            hidden_dim: Dimension of hidden layer in projection network
            projection_dim: Dimension of projected features before clustering
        """
        super(TEMIClusteringHead, self).__init__()
        
        self.num_clusters = num_clusters
        
        # Projection network to map features to a clustering-friendly space
        # This follows the TEMI paper's architecture with multiple layers
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
        
        # Cluster assignment layer: maps projections to cluster probabilities
        # This is the actual clustering layer that learns cluster centroids
        self.cluster_layer = nn.Linear(projection_dim, num_clusters, bias=False)
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.
        
        Proper initialization is important for convergence in clustering tasks.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the clustering head.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (cluster_logits, projected_features)
            - cluster_logits: Shape (batch_size, num_clusters)
            - projected_features: Shape (batch_size, projection_dim)
        """
        # Project features to clustering space
        projected = self.projection(x)
        
        # Normalize projected features to unit sphere
        # This is important for stable clustering
        projected = F.normalize(projected, p=2, dim=1)
        
        # Compute cluster assignment logits
        cluster_logits = self.cluster_layer(projected)
        
        return cluster_logits, projected
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get hard cluster assignments for input features.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Cluster assignments of shape (batch_size,)
        """
        cluster_logits, _ = self.forward(x)
        return torch.argmax(cluster_logits, dim=1)


class TEMIClusterer:
    """
    TEMI clustering algorithm implementation.
    
    This class implements the full TEMI training and inference pipeline,
    including transformation equivariance losses and multi-instance learning.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        device: str = "cuda",
        hidden_dim: int = 2048,
        projection_dim: int = 256,
        learning_rate: float = 0.001,
        temperature: float = 0.1,
        use_sinkhorn: bool = True
    ):
        """
        Initialize the TEMI clusterer.
        
        Args:
            feature_dim: Dimension of input features
            num_clusters: Number of clusters to create
            device: Device to run computations on
            hidden_dim: Hidden layer dimension
            projection_dim: Projection space dimension
            learning_rate: Learning rate for optimizer
            temperature: Temperature parameter for softmax (controls sharpness)
            use_sinkhorn: Whether to use Sinkhorn-Knopp normalization (default: True)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.initial_temperature = temperature
        self.use_sinkhorn = use_sinkhorn
        
        # Create the clustering head
        self.model = TEMIClusteringHead(
            input_dim=feature_dim,
            num_clusters=num_clusters,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim
        ).to(self.device)
        
        # Optimizer for training the clustering head
        # Use SGD with momentum (as in SwAV - Caron et al., 2020) for better convergence in clustering
        # Adam can be unstable for cluster assignment learning in discrete spaces
        self.optimizer = SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        self.learning_rate = learning_rate
        
        # Storage for cluster centroids
        self.cluster_centers = None
        
        print(f"TEMI clusterer initialized with {num_clusters} clusters on {self.device}")
        print(f"Using Sinkhorn-Knopp normalization: {use_sinkhorn}")
    
    def initialize_clusters(self, features: torch.Tensor):
        """
        Initialize cluster assignments using K-means (PyTorch-native implementation).
        
        This provides a warm start for the TEMI algorithm, as recommended
        in the paper. K-means gives reasonable initial cluster assignments.
        We also use K-means centers to initialize the cluster layer weights.
        
        Args:
            features: Input features of shape (num_samples, feature_dim)
        """
        print("Initializing clusters with K-means (PyTorch-native)...")
        
        # Ensure features are on the correct device
        features = features.to(self.device)
        
        # PyTorch-native K-means implementation
        num_samples = features.shape[0]
        
        # Handle edge case: more clusters than samples
        effective_num_clusters = min(self.num_clusters, num_samples)
        if effective_num_clusters < self.num_clusters:
            warnings.warn(
                f"num_clusters ({self.num_clusters}) > num_samples ({num_samples}). "
                f"Using {effective_num_clusters} clusters instead.",
                RuntimeWarning
            )
        
        # Initialize cluster centers randomly from data points
        indices = torch.randperm(num_samples, device=self.device)[:effective_num_clusters]
        cluster_centers = features[indices].clone()
        
        # If we have fewer samples than clusters, pad with random noise
        if effective_num_clusters < self.num_clusters:
            # Add random centers for remaining clusters
            extra_centers = torch.randn(
                self.num_clusters - effective_num_clusters,
                features.shape[1],
                device=self.device
            ) * 0.1
            cluster_centers = torch.cat([cluster_centers, extra_centers], dim=0)
        
        # K-means iterations
        max_iters = 300
        n_init = 20
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        
        for init_idx in range(n_init):
            # Random initialization for this run
            if init_idx > 0:
                indices = torch.randperm(num_samples, device=self.device)[:effective_num_clusters]
                cluster_centers_init = features[indices].clone()
                
                # Pad with random noise if needed
                if effective_num_clusters < self.num_clusters:
                    extra_centers = torch.randn(
                        self.num_clusters - effective_num_clusters,
                        features.shape[1],
                        device=self.device
                    ) * 0.1
                    cluster_centers = torch.cat([cluster_centers_init, extra_centers], dim=0)
                else:
                    cluster_centers = cluster_centers_init
            
            for iter_idx in range(max_iters):
                # Assign samples to nearest cluster (using cosine similarity)
                # Normalize features and centers
                features_norm = F.normalize(features, p=2, dim=1)
                centers_norm = F.normalize(cluster_centers, p=2, dim=1)
                
                # Compute cosine similarity (batch_size, num_clusters)
                similarities = torch.mm(features_norm, centers_norm.t())
                
                # Assign to nearest cluster (highest similarity)
                labels = torch.argmax(similarities, dim=1)
                
                # Update cluster centers
                new_centers = torch.zeros_like(cluster_centers)
                for k in range(self.num_clusters):
                    mask = labels == k
                    if mask.sum() > 0:
                        new_centers[k] = features[mask].mean(dim=0)
                    else:
                        # Keep old center if cluster is empty
                        new_centers[k] = cluster_centers[k]
                
                # Check convergence
                center_shift = torch.norm(new_centers - cluster_centers)
                cluster_centers = new_centers
                
                if center_shift < 1e-4:
                    break
            
            # Compute inertia (sum of squared distances to nearest center)
            distances = 1 - similarities  # Convert similarity to distance
            min_distances = torch.min(distances, dim=1)[0]
            inertia = torch.sum(min_distances ** 2).item()
            
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = cluster_centers.clone()
                best_labels = labels.clone()
        
        # Store best cluster centers
        self.cluster_centers = best_centers
        
        print(f"K-means initialization complete. Inertia: {best_inertia:.2f}")
        
        # Initialize cluster layer with K-means centers in projection space
        # First, project the K-means centers through the projection network
        self.model.eval()
        with torch.no_grad():
            # Normalize K-means centers
            centers_normalized = F.normalize(self.cluster_centers, p=2, dim=1)
            
            # Project centers through the projection network
            projected_centers = self.model.projection(centers_normalized)
            projected_centers = F.normalize(projected_centers, p=2, dim=1)
            
            # Initialize cluster layer weights with projected centers
            self.model.cluster_layer.weight.data.copy_(projected_centers)
        
        self.model.train()
        print(f"Cluster layer initialized with K-means centers in projection space")
        
        return best_labels
    
    def sinkhorn_knopp(self, logits: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
        """
        Apply Sinkhorn-Knopp algorithm to balance cluster assignments.
        
        This iteratively normalizes rows and columns to enforce balanced clusters,
        which helps prevent cluster collapse. Implementation follows SwAV paper.
        
        Args:
            logits: Cluster logits (batch_size, num_clusters)
            num_iters: Number of Sinkhorn iterations (default: 5 for strong balance)
            
        Returns:
            Balanced probability distribution
        """
        with torch.no_grad():
            # Compute Q (soft assignments) with temperature scaling
            # Use self.temperature for consistency with softmax path
            # Transpose for easier column operations (normalize per cluster first, then per sample)
            Q = torch.exp(logits / self.temperature).T
            Q /= torch.sum(Q)  # Normalize total sum to 1
            
            K = Q.shape[0]  # num_clusters
            B = Q.shape[1]  # batch_size
            
            for _ in range(num_iters):
                # Normalize each row (make sum of assignments per cluster equal)
                Q /= torch.sum(Q, dim=1, keepdim=True)
                Q /= K
                
                # Normalize each column (make sum of cluster probs per sample equal)
                Q /= torch.sum(Q, dim=0, keepdim=True)
                Q /= B
            
            # Final normalization
            Q *= B  # Scale back
            
            return Q.T  # Transpose back to (batch_size, num_clusters)
    
    def compute_cluster_probs(
        self,
        cluster_logits: torch.Tensor,
        use_sinkhorn: bool = False
    ) -> torch.Tensor:
        """
        Convert cluster logits to probabilities using temperature-scaled softmax.
        
        Args:
            cluster_logits: Raw cluster assignment logits
            use_sinkhorn: Whether to apply Sinkhorn-Knopp normalization
            
        Returns:
            Cluster probability distribution
        """
        if use_sinkhorn:
            # Use Sinkhorn-Knopp for balanced assignments
            return self.sinkhorn_knopp(cluster_logits)
        else:
            # Apply temperature scaling and softmax
            return F.softmax(cluster_logits / self.temperature, dim=1)
    
    def compute_temi_loss(
        self,
        features: torch.Tensor,
        augmented_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the TEMI loss including equivariance and clustering objectives.
        
        The TEMI loss consists of:
        1. Clustering loss: Minimize entropy of individual predictions (confident assignments)
        2. Equivariance loss: Consistency between original and augmented views
        3. Uniformity loss: Maximize entropy of average predictions (balanced clusters)
        
        Args:
            features: Original features (batch_size, feature_dim)
            augmented_features: Augmented features (batch_size, feature_dim)
            
        Returns:
            Dictionary containing individual loss components and total loss
        """
        # Forward pass for original features
        logits_orig, proj_orig = self.model(features)
        probs_orig = self.compute_cluster_probs(logits_orig, use_sinkhorn=self.use_sinkhorn)
        
        # Forward pass for augmented features
        logits_aug, proj_aug = self.model(augmented_features)
        probs_aug = self.compute_cluster_probs(logits_aug, use_sinkhorn=self.use_sinkhorn)
        
        # 1. Clustering loss: Minimize entropy of individual predictions for confident assignments
        # Each sample should be confidently assigned to one cluster
        entropy_individual = -torch.sum(probs_orig * torch.log(probs_orig + 1e-8), dim=1)
        entropy_loss = torch.mean(entropy_individual)
        
        # 2. Equivariance loss: Ensure augmented views get similar assignments
        # Use symmetric cross-entropy for consistency between views
        equivariance_loss = -torch.mean(
            torch.sum(probs_orig * torch.log(probs_aug + 1e-8), dim=1) +
            torch.sum(probs_aug * torch.log(probs_orig + 1e-8), dim=1)
        ) / 2.0
        
        # 3. Uniformity loss: Maximize entropy of marginal distribution (prevent cluster collapse)
        # Average cluster probabilities should be roughly uniform
        avg_probs_orig = torch.mean(probs_orig, dim=0)
        avg_probs_aug = torch.mean(probs_aug, dim=0)
        avg_probs = (avg_probs_orig + avg_probs_aug) / 2.0
        
        # Entropy of marginal distribution - we want this to be high (uniform)
        marginal_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        # Loss is negative of marginal entropy (we want to maximize it)
        uniformity_loss = -marginal_entropy
        
        # 4. Projection consistency loss: projected features should also be similar
        # This adds an additional constraint in the embedding space
        proj_consistency_loss = F.mse_loss(proj_orig, proj_aug)
        
        # Total loss is weighted combination of all components
        # Weights are carefully tuned based on deep clustering literature (SwAV, SCAN):
        # - Strong emphasis on consistency (equivariance) - primary supervision signal
        # - Balanced uniformity with Sinkhorn-Knopp doing the heavy lifting for cluster balance
        # - Entropy for confident predictions
        # - Light projection consistency for stability
        #
        # Note: Sinkhorn-Knopp normalization is the primary mechanism preventing collapse
        # Uniformity loss provides additional gradient signal but should not dominate
        total_loss = (
            1.0 * entropy_loss +          # Minimize conditional entropy (confident predictions)
            2.0 * equivariance_loss +     # Maximize consistency (most important for self-supervised)
            1.5 * uniformity_loss +       # Moderate increase: Maximize marginal entropy (prevent collapse)
            0.1 * proj_consistency_loss   # Projection space consistency
        )
        
        # Check for NaN values to catch numerical issues early
        if torch.isnan(total_loss):
            raise ValueError("NaN detected in total loss! Check numerical stability.")
        
        return {
            'total_loss': total_loss,
            'entropy_loss': entropy_loss,
            'equivariance_loss': equivariance_loss,
            'uniformity_loss': uniformity_loss,
            'proj_consistency_loss': proj_consistency_loss
        }
    
    def train_epoch(
        self,
        features: torch.Tensor,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Train the clustering model for one epoch.
        
        Args:
            features: All training features (num_samples, feature_dim)
            batch_size: Batch size for training
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        
        num_samples = features.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_losses = {
            'total_loss': 0.0,
            'entropy_loss': 0.0,
            'equivariance_loss': 0.0,
            'uniformity_loss': 0.0,
            'proj_consistency_loss': 0.0
        }
        
        # Shuffle indices for this epoch
        indices = torch.randperm(num_samples)
        
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch features
            batch_features = features[batch_indices].to(self.device)
            
            # Create augmented view with mild augmentation for pre-extracted features
            # Note: DINOv2 features are already robust, so we use gentle augmentation
            # to create variation without destroying semantic content
            
            # Apply light Gaussian noise (2% of feature norm)
            noise = torch.randn_like(batch_features) * 0.02
            augmented_features = batch_features + noise
            
            # Renormalize to unit sphere to maintain feature distribution
            augmented_features = F.normalize(augmented_features, p=2, dim=1)
            
            # Compute losses
            losses = self.compute_temi_loss(batch_features, augmented_features)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
        
        # Average losses over all batches
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def fit(
        self,
        features: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the TEMI clustering model.
        
        Args:
            features: Input features (num_samples, feature_dim)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: If True, print training progress
            
        Returns:
            Dictionary containing training history
        """
        # Ensure features are L2-normalized for stable clustering
        features = F.normalize(features, p=2, dim=1)
        
        # Initialize clusters with K-means
        initial_assignments = self.initialize_clusters(features)
        
        # Store initial temperature for annealing
        initial_temp = self.temperature
        
        history = {
            'total_loss': [],
            'entropy_loss': [],
            'equivariance_loss': [],
            'uniformity_loss': [],
            'proj_consistency_loss': []
        }
        
        print(f"\nTraining TEMI clusterer for {num_epochs} epochs...")
        
        # Warmup period (10 epochs) - gradually increase learning rate
        warmup_epochs = min(10, num_epochs // 10)
        
        for epoch in range(num_epochs):
            # Temperature annealing: gradually decrease temperature for sharper assignments
            # More gentle annealing: initial_temp to 0.8*initial_temp (only 20% reduction)
            # This helps maintain diverse clusters while still sharpening over time
            progress = epoch / num_epochs
            self.temperature = initial_temp * (1.0 - 0.2 * progress)
            
            # Learning rate scheduling with warmup + cosine annealing
            if epoch < warmup_epochs:
                # Warmup: linearly increase from 0.1*lr to lr
                lr = self.learning_rate * (0.1 + 0.9 * epoch / warmup_epochs)
            else:
                # Cosine annealing: smoothly decrease from lr to 0.1*lr
                progress_after_warmup = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                # Using torch.pi for better readability and precision
                lr = self.learning_rate * (0.1 + 0.9 * (1 + torch.cos(torch.tensor(torch.pi * progress_after_warmup)).item()) / 2)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train for one epoch
            epoch_losses = self.train_epoch(features, batch_size)
            
            # Store losses in history
            for key, value in epoch_losses.items():
                history[key].append(value)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {epoch_losses['total_loss']:.4f}, "
                      f"Entropy: {epoch_losses['entropy_loss']:.4f}, "
                      f"Equivariance: {epoch_losses['equivariance_loss']:.4f}, "
                      f"Uniformity: {epoch_losses['uniformity_loss']:.4f}, "
                      f"Temp: {self.temperature:.4f}")
        
        print("Training complete!")
        return history
    
    def predict(self, features: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """
        Predict cluster assignments for input features.
        
        Args:
            features: Input features (num_samples, feature_dim)
            batch_size: Batch size for inference
            
        Returns:
            Cluster assignments as torch tensor
        """
        self.model.eval()
        
        all_assignments = []
        num_samples = features.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                
                batch_features = features[start_idx:end_idx].to(self.device)
                assignments = self.model.get_cluster_assignments(batch_features)
                all_assignments.append(assignments)
        
        return torch.cat(all_assignments)
    
    def save_checkpoint(self, path: str, epoch: int, history: dict):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            history: Training history
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cluster_centers': self.cluster_centers,
            'history': history,
            'num_clusters': self.num_clusters,
            'temperature': self.temperature,
            # Save architecture parameters for proper reconstruction
            'feature_dim': self.model.projection[0].in_features,
            'hidden_dim': self.model.projection[0].out_features,
            'projection_dim': self.model.cluster_layer.in_features
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Tuple[int, dict]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, history)
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Check if architecture parameters match
        if 'feature_dim' in checkpoint:
            saved_feature_dim = checkpoint['feature_dim']
            saved_hidden_dim = checkpoint['hidden_dim']
            saved_projection_dim = checkpoint['projection_dim']
            
            current_feature_dim = self.model.projection[0].in_features
            current_hidden_dim = self.model.projection[0].out_features
            current_projection_dim = self.model.cluster_layer.in_features
            
            if (saved_feature_dim != current_feature_dim or
                saved_hidden_dim != current_hidden_dim or
                saved_projection_dim != current_projection_dim):
                
                print(f"WARNING: Model architecture mismatch!")
                print(f"Checkpoint: feature_dim={saved_feature_dim}, hidden_dim={saved_hidden_dim}, projection_dim={saved_projection_dim}")
                print(f"Current: feature_dim={current_feature_dim}, hidden_dim={current_hidden_dim}, projection_dim={current_projection_dim}")
                print(f"Recreating model with saved architecture...")
                
                # Recreate model with correct architecture
                self.model = TEMIClusteringHead(
                    input_dim=saved_feature_dim,
                    num_clusters=checkpoint['num_clusters'],
                    hidden_dim=saved_hidden_dim,
                    projection_dim=saved_projection_dim
                ).to(self.device)
                
                # Recreate optimizer
                self.optimizer = Adam(self.model.parameters(), lr=self.optimizer.defaults['lr'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cluster_centers = checkpoint['cluster_centers']
        self.num_clusters = checkpoint['num_clusters']
        self.temperature = checkpoint['temperature']
        
        print(f"Checkpoint loaded from {path}")
        
        return checkpoint['epoch'], checkpoint['history']
