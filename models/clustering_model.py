"""
Multi-head clustering model with teacher-student architecture.

This module implements the clustering heads that operate on DINOv2 embeddings
using a teacher-student self-distillation framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusteringHead(nn.Module):
    """
    A simple MLP head for clustering.
    
    Takes embeddings and outputs cluster assignment logits.
    """
    
    def __init__(self, input_dim, num_clusters, hidden_dim=None):
        """
        Initialize clustering head.
        
        Args:
            input_dim: Dimension of input embeddings
            num_clusters: Number of clusters to predict
            hidden_dim: Hidden layer dimension (None for linear head)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        
        if hidden_dim is None:
            # Simple linear head
            self.head = nn.Linear(input_dim, num_clusters)
        else:
            # MLP head with one hidden layer
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_clusters)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input embeddings (batch_size, input_dim)
            
        Returns:
            Cluster logits (batch_size, num_clusters)
        """
        return self.head(x)


class MultiHeadClusteringModel(nn.Module):
    """
    Multi-head clustering model with multiple parallel heads.
    
    Each head independently predicts cluster assignments, and the ensemble
    of heads provides more robust clustering.
    """
    
    def __init__(self, config):
        """
        Initialize multi-head clustering model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.num_heads = config.NUM_HEADS
        self.embedding_dim = config.EMBEDDING_DIM
        self.num_clusters = config.NUM_CLUSTERS
        
        # Create multiple clustering heads
        self.heads = nn.ModuleList([
            ClusteringHead(
                input_dim=self.embedding_dim,
                num_clusters=self.num_clusters,
                hidden_dim=None  # Use linear heads for efficiency
            )
            for _ in range(self.num_heads)
        ])
    
    def forward(self, embeddings):
        """
        Forward pass through all heads.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            
        Returns:
            List of cluster logits from each head
        """
        outputs = []
        for head in self.heads:
            outputs.append(head(embeddings))
        return outputs


class TeacherStudentModel(nn.Module):
    """
    Teacher-student model for self-distillation clustering.
    
    The teacher and student share the same architecture but the teacher
    parameters are updated using exponential moving average (EMA) of
    the student parameters.
    """
    
    def __init__(self, config):
        """
        Initialize teacher-student model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.momentum = config.MOMENTUM_TEACHER
        
        # Create student and teacher models
        self.student = MultiHeadClusteringModel(config)
        self.teacher = MultiHeadClusteringModel(config)
        
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Teacher doesn't need gradients
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(self, embeddings):
        """
        Forward pass through both student and teacher.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            
        Returns:
            Tuple of (student_outputs, teacher_outputs)
        """
        student_outputs = self.student(embeddings)
        
        with torch.no_grad():
            teacher_outputs = self.teacher(embeddings)
        
        return student_outputs, teacher_outputs
    
    @torch.no_grad()
    def update_teacher(self):
        """
        Update teacher parameters using EMA of student parameters.
        
        This should be called after each optimizer step.
        """
        for student_param, teacher_param in zip(
            self.student.parameters(),
            self.teacher.parameters()
        ):
            teacher_param.data.mul_(self.momentum).add_(
                student_param.data, alpha=1 - self.momentum
            )
    
    def get_cluster_assignments(self, embeddings, use_teacher=True):
        """
        Get cluster assignments for given embeddings.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            use_teacher: Whether to use teacher or student for predictions
            
        Returns:
            Cluster assignments (batch_size,) - ensemble vote across heads
        """
        model = self.teacher if use_teacher else self.student
        
        with torch.no_grad():
            outputs = model(embeddings)
            
            # Get predictions from each head
            predictions = [torch.argmax(out, dim=1) for out in outputs]
            
            # Stack predictions
            predictions = torch.stack(predictions, dim=1)  # (batch_size, num_heads)
            
            # Ensemble: majority voting across heads
            # For each sample, find the most common prediction
            assignments = []
            for i in range(predictions.shape[0]):
                votes = predictions[i]
                unique_votes, counts = torch.unique(votes, return_counts=True)
                majority_vote = unique_votes[torch.argmax(counts)]
                assignments.append(majority_vote.item())
            
            return torch.tensor(assignments, device=embeddings.device)
    
    def get_cluster_probabilities(self, embeddings, use_teacher=True, temperature=1.0):
        """
        Get cluster probability distributions for given embeddings.
        
        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)
            use_teacher: Whether to use teacher or student
            temperature: Temperature for softmax
            
        Returns:
            Averaged probability distributions (batch_size, num_clusters)
        """
        model = self.teacher if use_teacher else self.student
        
        with torch.no_grad():
            outputs = model(embeddings)
            
            # Convert to probabilities
            probs = [F.softmax(out / temperature, dim=-1) for out in outputs]
            
            # Average across heads
            avg_probs = torch.stack(probs, dim=0).mean(dim=0)
            
            return avg_probs
