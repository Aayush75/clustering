"""
CLUDI (Clustering via Diffusion) deep clustering implementation.

This module provides an interface to the CLUDI algorithm for deep clustering
using diffusion models. CLUDI uses a self-supervised approach that leverages
denoising diffusion for learning cluster-friendly representations.

The implementation wraps the CLUDI source code and provides a consistent
interface compatible with the existing TEMI clustering pipeline.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from tqdm import tqdm
import warnings
import numpy as np

# Add cludi directory to path for imports
CLUDI_DIR = Path(__file__).parent.parent / "cludi"
if str(CLUDI_DIR) not in sys.path:
    sys.path.insert(0, str(CLUDI_DIR))


def clustering_accuracy(true_labels, predicted_labels):
    """
    Compute clustering accuracy using Hungarian algorithm.
    
    Args:
        true_labels: Ground truth labels (numpy array)
        predicted_labels: Predicted cluster assignments (numpy array)
        
    Returns:
        Clustering accuracy as a float between 0 and 1
    """
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(predicted_labels, torch.Tensor):
        predicted_labels = predicted_labels.cpu().numpy()
    
    true_labels = np.asarray(true_labels).flatten()
    predicted_labels = np.asarray(predicted_labels).flatten()
    
    D = max(true_labels.max(), predicted_labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(true_labels.shape[0]):
        w[int(true_labels[i]), int(predicted_labels[i])] += 1

    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / true_labels.shape[0]


class CLUDIModel(nn.Module):
    """
    CLUDI (Clustering via Diffusion) model for deep clustering.
    
    This neural network implements a diffusion-based approach to clustering,
    using self-conditioning and multi-step denoising to learn cluster assignments.
    
    The model combines:
    - Data feature projection: Maps input features to a latent space
    - Cluster assignment projection: Processes soft cluster assignments
    - Diffusion network: Combines features and assignments for denoising
    - Cluster centers: Learnable prototype vectors for each cluster
    """
    
    def __init__(
        self,
        feature_dim: int = 384,
        num_clusters: int = 100,
        embedding_dim: int = 64,
        self_condition: bool = True,
        dropout: float = 0.15
    ):
        """
        Initialize the CLUDI model.
        
        Args:
            feature_dim: Dimension of input features (from DINOv2/CLIP)
            num_clusters: Number of clusters
            embedding_dim: Dimension of cluster embeddings
            self_condition: Whether to use self-conditioning in diffusion
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.self_condition = self_condition
        
        # Embedding dimensions
        self.embedding_dim_start = (embedding_dim * 2) if self_condition else embedding_dim
        self.embedding_dim_end = feature_dim // 2
        
        # Network dimension
        self.network_dim = self.embedding_dim_end + feature_dim
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.network_dim, 4 * self.network_dim),
            nn.SiLU(),
            nn.Linear(4 * self.network_dim, self.network_dim),
        )
        
        # Data feature projection
        self.data_input_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim, bias=True),
            nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(feature_dim, feature_dim, bias=True),
        )
        
        # Cluster assignment projection
        self.cluster_input_proj = nn.Sequential(
            nn.Linear(self.embedding_dim_start, self.embedding_dim_start, bias=True),
            nn.SiLU(),
            nn.Linear(self.embedding_dim_start, self.embedding_dim_end, bias=True),
            nn.LayerNorm(self.embedding_dim_end, elementwise_affine=True, eps=1e-6),
            nn.Linear(self.embedding_dim_end, self.embedding_dim_end, bias=True),
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(self.network_dim, self.network_dim),
            nn.SiLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.SiLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.SiLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.SiLU(),
            nn.Linear(self.network_dim, self.network_dim),
            nn.SiLU(),
            nn.Linear(self.network_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Normalization
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Final projection
        self.final_layer = nn.Linear(embedding_dim, embedding_dim)
        
        # Last layer for cluster assignments
        self.last_layer = nn.Linear(embedding_dim, num_clusters)
        
        # Cluster centers (learnable)
        self.clusters_centers = nn.Parameter(
            torch.randn(1, num_clusters, embedding_dim)
        ).requires_grad_(False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: 1D tensor of N indices
            dim: Embedding dimension
            max_period: Maximum period for the embedding
            
        Returns:
            Tensor of shape (N, dim) with positional embeddings
        """
        import math
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def return_clusters_centers(self) -> torch.Tensor:
        """Return normalized cluster centers."""
        return F.normalize(self.clusters_centers, dim=-1)
    
    def return_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert soft cluster assignments to embeddings.
        
        Args:
            x: Soft cluster assignments of shape (batch, seq, num_clusters)
            
        Returns:
            Embeddings of shape (batch, seq, embedding_dim)
        """
        clusters_centers = F.normalize(self.clusters_centers, dim=-1)
        weighted_sum = torch.matmul(x, clusters_centers.to(x.dtype))
        return F.normalize(weighted_sum, dim=-1) * np.sqrt(self.embedding_dim)
    
    def forward(
        self,
        cluster_assignments: torch.Tensor,
        data_features: torch.Tensor,
        timesteps: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the CLUDI model.
        
        Args:
            cluster_assignments: Soft cluster assignments
            data_features: Input features from feature extractor
            timesteps: Diffusion timesteps
            x_self_cond: Self-conditioning tensor (optional)
            
        Returns:
            Denoised embeddings
        """
        cluster_assignments = cluster_assignments.to(data_features.dtype)
        
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(cluster_assignments)
            cluster_assignments = torch.cat((x_self_cond, cluster_assignments), dim=-1)
        
        # Normalize data features
        data_features = F.normalize(data_features, dim=-1, p=2)
        batch_size, num_points, _ = data_features.size()
        
        # Project features
        data_features = self.dropout1(self.data_input_proj(data_features))
        cluster_assignments = self.dropout2(self.cluster_input_proj(cluster_assignments))
        
        # Combine features
        combined_features = torch.cat((data_features, cluster_assignments), -1)
        
        # Add time embedding
        time_token = self.time_embed(
            self.timestep_embedding(timesteps, self.network_dim)
        )
        time_token = time_token.unsqueeze(dim=1)
        x = combined_features + time_token
        
        # Forward through network
        x = self.network(x)
        x = self.norm(x)
        final_output = self.final_layer(x)
        
        return final_output


class GaussianDiffusionCLUDI(nn.Module):
    """
    Gaussian diffusion process for CLUDI clustering.
    
    This class implements the forward (noising) and reverse (denoising)
    diffusion process used for cluster assignment learning.
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        objective: str = 'pred_v',
        vp_rf: bool = False,
        rescaling_factor: float = 1.0
    ):
        """
        Initialize the Gaussian diffusion process.
        
        Args:
            timesteps: Number of diffusion timesteps
            objective: Prediction objective ('pred_noise', 'pred_x0', 'pred_v')
            vp_rf: Whether to use variance-preserving rectified flow
            rescaling_factor: Factor for rescaling noise schedule
        """
        super().__init__()
        
        self.num_timesteps = timesteps
        self.sampling_timesteps = timesteps
        self.objective = objective
        self.ddim_sampling_eta = 1.0
        
        # Create beta schedule
        betas = self._sqrt_beta_schedule(timesteps)
        
        if vp_rf:
            f2 = rescaling_factor
            alpha_bar = lambda t: 1 - np.sqrt(t + 0.0001)
            rescaled_alpha_bar = lambda t: alpha_bar(t) / (f2 - (f2 - 1) * alpha_bar(t))
            betas = torch.from_numpy(self._betas_for_alpha_bar(timesteps, rescaled_alpha_bar))
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Register buffers
        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).float())
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())
        
        # Loss weight based on SNR
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max=5)
        
        if objective == 'pred_v':
            self.register_buffer('loss_weight', (maybe_clipped_snr / (snr + 1)).float())
        else:
            self.register_buffer('loss_weight', maybe_clipped_snr.float())
    
    @staticmethod
    def _sqrt_beta_schedule(timesteps: int, s: float = 0.0001) -> torch.Tensor:
        """Create square root beta schedule."""
        def alpha_bar(t):
            return 1 - np.sqrt(t + s)
        
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    
    @staticmethod
    def _betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999):
        """Create beta schedule from alpha_bar function."""
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    
    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract appropriate values from schedule tensor."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0).
        
        Args:
            x_start: Clean data (x_0)
            t: Timesteps
            noise: Optional noise tensor
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from v-parameterization."""
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v
    
    def predict_v(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Compute v from x_0 and noise."""
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start
    
    def predict_noise_from_start(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_start: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise from x_t and x_0."""
        sqrt_recip_alphas_cumprod_t = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip_alphas_cumprod_t * x_t - x_start) / sqrt_recipm1_alphas_cumprod_t


class CLUDIClusterer:
    """
    CLUDI clustering algorithm implementation.
    
    This class provides a high-level interface for training and evaluating
    the CLUDI deep clustering model, compatible with the existing pipeline.
    
    CLUDI uses diffusion-based denoising to learn cluster-friendly representations
    through a self-supervised objective that encourages consistent cluster assignments.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_clusters: int,
        device: str = "cuda",
        embedding_dim: int = 64,
        learning_rate: float = 0.0001,
        diffusion_steps: int = 1000,
        batch_diffusion: int = 8,
        rescaling_factor: float = 49.0,
        ce_lambda: float = 50.0,
        use_v_prediction: bool = True,
        warmup_epochs: int = 1
    ):
        """
        Initialize the CLUDI clusterer.
        
        Args:
            feature_dim: Dimension of input features
            num_clusters: Number of clusters to create
            device: Device to run computations on
            embedding_dim: Dimension of cluster embeddings
            learning_rate: Learning rate for optimizer
            diffusion_steps: Number of diffusion timesteps
            batch_diffusion: Batch size for diffusion process
            rescaling_factor: Factor for rescaling in diffusion
            ce_lambda: Weight for cross-entropy loss
            use_v_prediction: Whether to use v-parameterization
            warmup_epochs: Number of warmup epochs
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.diffusion_steps = diffusion_steps
        self.batch_diffusion = batch_diffusion
        self.rescaling_factor = rescaling_factor
        self.ce_lambda = ce_lambda
        self.use_v_prediction = use_v_prediction
        self.warmup_epochs = warmup_epochs
        
        # Initialize model
        self.model = CLUDIModel(
            feature_dim=feature_dim,
            num_clusters=num_clusters,
            embedding_dim=embedding_dim
        ).to(self.device)
        
        # Initialize diffusion
        self.diffusion = GaussianDiffusionCLUDI(
            timesteps=diffusion_steps,
            objective='pred_v' if use_v_prediction else 'pred_x0',
            rescaling_factor=rescaling_factor
        ).to(self.device)
        
        # Initialize EMA teacher (simple copy for now)
        self.teacher = None
        self._create_teacher()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self._get_params_groups(),
            lr=learning_rate,
            betas=(0.9, 0.98)
        )
        
        # FP16 scaler
        self.fp16_scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Cluster centers storage
        self.cluster_centers = None
        
        print(f"CLUDI clusterer initialized with {num_clusters} clusters on {self.device}")
        print(f"Feature dim: {feature_dim}, Embedding dim: {embedding_dim}")
    
    def _create_teacher(self):
        """Create EMA teacher model."""
        import copy
        self.teacher = copy.deepcopy(self.model)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def _get_params_groups(self) -> List[Dict[str, object]]:
        """Get parameter groups for optimizer."""
        regularized = []
        not_regularized = []
        cluster_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            elif "clusters_centers" in name:
                cluster_params.append(param)
            else:
                regularized.append(param)
        
        return [
            {'params': regularized},
            {'params': not_regularized, 'weight_decay': 0.},
            {'params': cluster_params, 'weight_decay': 0.}
        ]
    
    def _update_teacher(self, ema_val: float = 0.999):
        """Update teacher model with EMA."""
        with torch.no_grad():
            for student_param, teacher_param in zip(
                self.model.parameters(),
                self.teacher.parameters()
            ):
                teacher_param.data.mul_(ema_val).add_(
                    student_param.data,
                    alpha=1 - ema_val
                )
    
    def _feature_dropout(
        self,
        features: torch.Tensor,
        min_rate: float = 0.05,
        max_rate: float = 0.1
    ) -> torch.Tensor:
        """Apply feature dropout augmentation."""
        batch_size = features.shape[0]
        dropout_rates = torch.rand(batch_size, device=features.device)
        dropout_rates = dropout_rates * (max_rate - min_rate) + min_rate
        dropout_rates = dropout_rates.view(batch_size, 1, 1)
        
        masks = torch.rand_like(features) > dropout_rates
        return features * masks.float()
    
    def _add_noise(
        self,
        features: torch.Tensor,
        min_scale: float = 0.2,
        max_scale: float = 1.0
    ) -> torch.Tensor:
        """Add random noise to features."""
        batch_size = features.shape[0]
        noise_scale = torch.rand(batch_size, device=features.device)
        noise_scale = noise_scale * (max_scale - min_scale) + min_scale
        noise = torch.randn_like(features)
        noise = F.normalize(noise, p=2, dim=-1)
        return features + noise * noise_scale.view(batch_size, 1, 1)
    
    def _compute_loss(
        self,
        logits_teacher: torch.Tensor,
        logits_student: torch.Tensor,
        z_0: torch.Tensor,
        z_0_loop: torch.Tensor,
        loss_weights: torch.Tensor,
        row_tau: float = 0.1,
        col_tau: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute CLUDI training loss.
        
        Args:
            logits_teacher: Teacher model logits
            logits_student: Student model logits
            z_0: Target embeddings
            z_0_loop: Predicted embeddings
            loss_weights: Per-sample loss weights from diffusion
            row_tau: Temperature for row-wise softmax
            col_tau: Temperature for column-wise softmax
            
        Returns:
            Tuple of (total_loss, diffusion_loss, logits_loss)
        """
        # Diffusion loss (MSE between embeddings)
        loss_diff = F.mse_loss(z_0_loop, z_0, reduction='none')
        loss_diff = loss_diff.mean(dim=(-1, -2)) * loss_weights
        loss_diff = loss_diff.mean()
        
        # Cross-entropy-like logits loss
        def softmax_ce_loss(pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
            # Row-wise normalization
            log1 = torch.log(
                pred1.shape[1] / pred1.shape[-1] * 
                F.normalize(F.softmax(pred1 / row_tau, -1), p=1, dim=1, eps=1e-8) + 1e-8
            ).unsqueeze(1)
            log2 = torch.log(
                pred2.shape[1] / pred2.shape[-1] * 
                F.normalize(F.softmax(pred2 / row_tau, -1), p=1, dim=1, eps=1e-8) + 1e-8
            ).unsqueeze(0)
            
            # Column-wise normalization
            norm1 = F.normalize(F.softmax(pred1 / col_tau, 1), p=1, dim=2, eps=1e-8).unsqueeze(1)
            norm2 = F.normalize(F.softmax(pred2 / col_tau, 1), p=1, dim=2, eps=1e-8).unsqueeze(0)
            
            # Compute loss
            l1 = -torch.mean(torch.sum(norm1 * log2, dim=3), dim=-1)
            l2 = -torch.mean(torch.sum(norm2 * log1, dim=3), dim=-1)
            
            return (l1.mean() + l2.mean()) / 2
        
        loss_logits = softmax_ce_loss(logits_teacher, logits_student)
        
        # Total loss
        total_loss = loss_diff + loss_logits * self.ce_lambda
        
        return total_loss, loss_diff, loss_logits
    
    def _train_iter(
        self,
        features: torch.Tensor,
        soft_assignments: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single training iteration for student model.
        
        Args:
            features: Input features
            soft_assignments: Soft cluster assignments from teacher
            
        Returns:
            Tuple of (z_0_hat, logits, loss_weights, z_0_loop)
        """
        # Get target embeddings
        z_0 = self.model.return_embedding(soft_assignments)
        
        # Sample timesteps
        t = torch.randint(0, self.diffusion_steps, [len(z_0)], device=z_0.device)
        
        # Add noise
        noise = torch.randn_like(z_0) * self.rescaling_factor
        z_t = self.diffusion.q_sample(z_0, t, noise)
        
        # Self-conditioning
        z_self_cond = torch.zeros_like(z_0)
        
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                if self.use_v_prediction:
                    model_output = self.model(z_t, features, t, z_self_cond)
                    z_self_cond = self.diffusion.predict_start_from_v(z_t, t, model_output)
                    z_self_cond = torch.clamp(z_self_cond, min=-1., max=1.)
                else:
                    z_self_cond = self.model(z_t, features, t, z_self_cond)
                z_self_cond = z_self_cond.detach()
        
        # Forward pass
        if self.use_v_prediction:
            pred_v = self.model(z_t, features, t, z_self_cond)
            z_0_hat = self.diffusion.predict_start_from_v(z_t, t, pred_v)
        else:
            z_0_hat = self.model(z_t, features, t, z_self_cond)
        
        # Get logits
        logits = self.model.last_layer(z_0_hat)
        
        # Get loop embedding
        z_0_loop = self.model.return_embedding(F.softmax(logits / 0.1, dim=-1))
        
        return z_0_hat, logits, self.diffusion.loss_weight[t] / self.rescaling_factor, z_0_loop
    
    def _gen_loop(
        self,
        features: torch.Tensor,
        steps: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generation loop for inference.
        
        Args:
            features: Input features
            steps: Number of denoising steps
            
        Returns:
            Tuple of (final_embedding, logits)
        """
        batch = features.shape[0]
        device = features.device
        
        # Initialize from noise
        z_t = torch.randn(
            batch, features.shape[1], self.embedding_dim,
            device=device
        ) * self.rescaling_factor
        self_cond = torch.zeros_like(z_t)
        
        # Time scheduling
        times = torch.linspace(-1, self.diffusion_steps - 1, steps=steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            if self.use_v_prediction:
                model_output = self.teacher(z_t, features, time_cond, self_cond)
                z_start = self.diffusion.predict_start_from_v(z_t, time_cond, model_output)
                z_start = torch.clamp(z_start, min=-1., max=1.)
            else:
                z_start = self.teacher(z_t, features, time_cond, self_cond)
                z_start = torch.clamp(z_start, min=-1., max=1.)
            
            pred_noise = self.diffusion.predict_noise_from_start(z_t, time_cond, z_start)
            
            if time_next < 0:
                z_t = z_start
                continue
            
            alpha = self.diffusion.alphas_cumprod[time]
            alpha_next = self.diffusion.alphas_cumprod[time_next]
            
            sigma = ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(z_t) * self.rescaling_factor
            
            z_t = z_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            self_cond = z_start
        
        logits = self.teacher.last_layer(z_t)
        z_final = self.teacher.return_embedding(F.softmax(logits / 0.1, dim=-1))
        
        return z_final, logits
    
    def fit(
        self,
        features: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 20
    ) -> Dict[str, List[float]]:
        """
        Train the CLUDI clustering model.
        
        Args:
            features: Input features (num_samples, feature_dim)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: If True, print training progress
            save_checkpoints: Whether to save checkpoints during training
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Frequency of checkpoint saving
            
        Returns:
            Dictionary containing training history
        """
        # Ensure features are normalized and on device
        features = F.normalize(features, p=2, dim=1).to(self.device)
        
        # Create dataset and loader
        dataset = TensorDataset(features, torch.zeros(len(features)))
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )
        
        # Learning rate scheduling
        num_iters = len(data_loader) * num_epochs
        lr_schedule = self._cosine_scheduler(
            self.learning_rate,
            1e-6,
            num_epochs,
            len(data_loader),
            warmup_epochs=self.warmup_epochs
        )
        
        history = {
            'total_loss': [],
            'diff_loss': [],
            'logits_loss': [],
            'nmi': [],
            'ari': []
        }
        
        print(f"\nTraining CLUDI clusterer for {num_epochs} epochs...")
        print(f"Dataset size: {len(features)}, Batch size: {batch_size}")
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = {'total': 0., 'diff': 0., 'logits': 0.}
            
            for it, (batch_features, _) in enumerate(data_loader):
                global_it = len(data_loader) * epoch + it
                
                # Update learning rate
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group["lr"] = lr_schedule[global_it]
                    if i == 2:  # Cluster params
                        param_group["lr"] = lr_schedule[global_it] / 10
                
                batch_features = batch_features.to(self.device)
                
                # Expand batch for diffusion sampling
                batch_features = batch_features.unsqueeze(0).repeat(self.batch_diffusion, 1, 1)
                batch_features = F.normalize(batch_features, p=2, dim=-1)
                
                # Create augmented view
                student_feat = self._feature_dropout(
                    self._add_noise(batch_features, 0.2, 1.0)
                )
                
                # Teacher inference
                self.teacher.eval()
                with torch.no_grad():
                    z_0, logits_teacher = self._gen_loop(batch_features, steps=25)
                
                # Get soft assignments from teacher
                soft_assignments = F.softmax(logits_teacher / 0.1, dim=-1)
                
                # Student training step
                self.model.train()
                z_0_hat, logits_student, loss_weights, z_0_loop = self._train_iter(
                    student_feat,
                    soft_assignments
                )
                
                # Compute loss
                total_loss, diff_loss, logits_loss = self._compute_loss(
                    logits_teacher, logits_student,
                    z_0, z_0_loop,
                    loss_weights
                )
                
                # Backward pass
                if self.fp16_scaler is not None:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update teacher
                self._update_teacher()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['diff'] += diff_loss.item()
                epoch_losses['logits'] += logits_loss.item()
            
            # Average losses
            num_batches = len(data_loader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            history['total_loss'].append(epoch_losses['total'])
            history['diff_loss'].append(epoch_losses['diff'])
            history['logits_loss'].append(epoch_losses['logits'])
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Loss: {epoch_losses['total']:.4f}, "
                      f"Diff: {epoch_losses['diff']:.4f}, "
                      f"Logits: {epoch_losses['logits']:.4f}")
            
            # Save checkpoint
            if save_checkpoints and checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt"),
                    epoch + 1,
                    history
                )
        
        # Store cluster centers
        self.cluster_centers = self.model.clusters_centers.data.clone().squeeze(0)
        
        print("Training complete!")
        return history
    
    @staticmethod
    def _cosine_scheduler(
        base_value: float,
        final_value: float,
        epochs: int,
        niter_per_ep: int,
        warmup_epochs: int = 0
    ) -> np.ndarray:
        """Create cosine learning rate schedule with warmup."""
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(0, base_value, warmup_iters)
        
        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * \
                   (1 + np.cos(np.pi * iters / len(iters)))
        
        schedule = np.concatenate((warmup_schedule, schedule))
        return schedule
    
    def predict(self, features: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """
        Predict cluster assignments for input features.
        
        Args:
            features: Input features (num_samples, feature_dim)
            batch_size: Batch size for inference
            
        Returns:
            Cluster assignments as torch tensor
        """
        self.teacher.eval()
        features = F.normalize(features, p=2, dim=1).to(self.device)
        
        all_assignments = []
        num_samples = features.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Predicting", disable=False):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                
                batch_features = features[start_idx:end_idx].to(self.device)
                # Use batch_diffusion for consistency with training
                batch_features = batch_features.unsqueeze(0).repeat(self.batch_diffusion, 1, 1)
                
                # Run generation loop
                _, logits = self._gen_loop(batch_features, steps=50)
                
                # Average over diffusion samples
                logits_avg = logits.mean(0)
                assignments = logits_avg.argmax(dim=-1)
                all_assignments.append(assignments)
        
        return torch.cat(all_assignments)
    
    def save_checkpoint(self, path: str, epoch: int, history: Dict):
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
            'teacher_state_dict': self.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'cluster_centers': self.cluster_centers if self.cluster_centers is not None else self.model.clusters_centers.data,
            'history': history,
            'config': {
                'num_clusters': self.num_clusters,
                'feature_dim': self.feature_dim,
                'embedding_dim': self.embedding_dim,
                'learning_rate': self.learning_rate,
                'diffusion_steps': self.diffusion_steps,
                'rescaling_factor': self.rescaling_factor,
                'ce_lambda': self.ce_lambda,
                'use_v_prediction': self.use_v_prediction
            }
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Tuple of (epoch, history)
            
        Note:
            weights_only=False is used because we store optimizer state and history
            which contain non-tensor data. Only load checkpoints from trusted sources.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        self.cluster_centers = checkpoint['cluster_centers']
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['history']
