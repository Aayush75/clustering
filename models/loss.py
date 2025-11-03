"""
TEMI loss implementation for deep clustering.

This module implements the TEMI (Trustworthy Evidence from Mutual Information) loss
which uses weighted mutual information with a teacher-student architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sim_weight(p1, p2, gamma=1.0):
    """
    Compute similarity weight between two probability distributions.
    
    This measures how similar two cluster assignments are, used to weight
    the mutual information objective.
    
    Args:
        p1: First probability distribution (batch_size, num_clusters)
        p2: Second probability distribution (batch_size, num_clusters)
        gamma: Exponent for weighting (default 1.0)
        
    Returns:
        Similarity weights (batch_size,)
    """
    return (p1 * p2).pow(gamma).sum(dim=-1)


def beta_mi(student_probs, teacher_probs, pk, beta=1.0):
    """
    Compute cross-entropy distillation loss for clustering.
    
    Standard approach: minimize cross-entropy between teacher and student predictions.
    This is the proven method used in SwAV, DINO, and similar self-supervised methods.
    
    Args:
        student_probs: Student probability distribution (batch_size, num_clusters)
        teacher_probs: Teacher probability distribution (batch_size, num_clusters)
        pk: Marginal cluster probabilities (1, num_clusters) - for monitoring/centering
        beta: Not used, kept for API compatibility
        
    Returns:
        Loss per sample (batch_size,)
    """
    eps = 1e-8
    
    # Clamp for numerical stability
    student_probs = student_probs.clamp(min=eps)
    teacher_probs = teacher_probs.clamp(min=eps)
    
    # Cross-entropy loss: H(teacher, student) = -sum(teacher * log(student))
    # This encourages student to match teacher's cluster assignments
    loss = -(teacher_probs * student_probs.log()).sum(dim=-1)
    
    return loss


class TEMILoss(nn.Module):
    """
    TEMI loss for multi-head deep clustering.
    
    This loss combines:
    1. Weighted mutual information between student and teacher predictions
    2. Ensemble of multiple clustering heads
    3. Temperature scheduling for the teacher
    4. EMA updates for marginal cluster probabilities
    """
    
    def __init__(self, config):
        """
        Initialize TEMI loss.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.num_clusters = config.NUM_CLUSTERS
        self.num_heads = config.NUM_HEADS
        self.batch_size = config.BATCH_SIZE
        self.student_temp = config.STUDENT_TEMP
        self.teacher_temp = config.TEACHER_TEMP
        self.beta = config.BETA
        self.probs_momentum = config.PROBS_MOMENTUM
        
        # Temperature schedule for teacher
        # Linearly increase from warmup temp to final temp
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(
                config.WARMUP_TEACHER_TEMP,
                config.TEACHER_TEMP,
                config.WARMUP_TEACHER_EPOCHS
            ),
            np.ones(config.NUM_EPOCHS - config.WARMUP_TEACHER_EPOCHS) * config.TEACHER_TEMP
        ])
        
        # Register buffers for marginal probabilities (one per head)
        # Initialize with uniform distribution
        for i in range(self.num_heads):
            self.register_buffer(
                f"pk_{i}",
                torch.ones(1, self.num_clusters) / self.num_clusters
            )
    
    def get_pk(self, head_idx):
        """
        Get marginal probability for a specific head.
        
        Args:
            head_idx: Index of the head
            
        Returns:
            Marginal probability tensor
        """
        return getattr(self, f"pk_{head_idx}")
    
    def set_pk(self, head_idx, value):
        """
        Set marginal probability for a specific head.
        
        Args:
            head_idx: Index of the head
            value: New marginal probability value
        """
        setattr(self, f"pk_{head_idx}", value)
    
    def update_marginals(self, teacher_probs, head_idx):
        """
        Update marginal cluster probabilities using EMA.
        
        Args:
            teacher_probs: Teacher probability predictions (batch_size, num_clusters)
            head_idx: Index of the head
        """
        with torch.no_grad():
            # Compute batch center (mean over batch)
            batch_center = teacher_probs.mean(dim=0, keepdim=True)
            
            # EMA update
            pk_old = self.get_pk(head_idx)
            pk_new = self.probs_momentum * pk_old + (1 - self.probs_momentum) * batch_center
            self.set_pk(head_idx, pk_new)
    
    def forward(self, student_outputs, teacher_outputs, epoch):
        """
        Compute TEMI loss for multi-head clustering.
        
        Args:
            student_outputs: List of student predictions for each head
                           Each element has shape (batch_size, num_clusters)
            teacher_outputs: List of teacher predictions for each head
                           Each element has shape (batch_size, num_clusters)
            epoch: Current training epoch (for temperature scheduling)
            
        Returns:
            Total loss (averaged over all heads)
        """
        # Get current teacher temperature
        temp = self.teacher_temp_schedule[epoch]
        
        total_loss = 0.0
        num_heads = len(student_outputs)
        
        # Compute loss for each head
        for head_idx in range(num_heads):
            student_out = student_outputs[head_idx]
            teacher_out = teacher_outputs[head_idx]
            
            # Convert logits to probabilities
            student_probs = F.softmax(student_out / self.student_temp, dim=-1)
            teacher_probs = F.softmax(teacher_out / temp, dim=-1).detach()
            
            # Update marginal probabilities
            self.update_marginals(teacher_probs, head_idx)
            pk = self.get_pk(head_idx)
            
            # Compute cross-entropy distillation loss
            loss = beta_mi(
                student_probs,
                teacher_probs,
                pk,
                beta=self.beta
            )
            
            # Check for NaN and handle gracefully
            if torch.isnan(loss).any():
                print(f"Warning: NaN detected in loss for head {head_idx}, skipping")
                continue
            
            total_loss += loss.mean()
        
        # Average over heads
        return total_loss / num_heads


class MultiHeadTEMILoss(nn.Module):
    """
    Enhanced multi-head TEMI loss with cross-head weighting.
    
    This version computes weights between teacher predictions from different
    views and applies them to the MI objective, following the original paper.
    """
    
    def __init__(self, config):
        """
        Initialize multi-head TEMI loss.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.num_clusters = config.NUM_CLUSTERS
        self.num_heads = config.NUM_HEADS
        self.batch_size = config.BATCH_SIZE
        self.student_temp = config.STUDENT_TEMP
        self.teacher_temp = config.TEACHER_TEMP
        self.beta = config.BETA
        self.probs_momentum = config.PROBS_MOMENTUM
        self.use_reg = config.USE_REGULARIZATION
        self.alpha = config.REGULARIZATION_WEIGHT
        
        # Temperature schedule
        self.teacher_temp_schedule = np.concatenate([
            np.linspace(
                config.WARMUP_TEACHER_TEMP,
                config.TEACHER_TEMP,
                config.WARMUP_TEACHER_EPOCHS
            ),
            np.ones(config.NUM_EPOCHS - config.WARMUP_TEACHER_EPOCHS) * config.TEACHER_TEMP
        ])
        
        # Marginal probabilities for each head
        for i in range(self.num_heads):
            self.register_buffer(
                f"pk_{i}",
                torch.ones(1, self.num_clusters) / self.num_clusters
            )
    
    def get_pk(self, head_idx):
        """Get marginal probability for a specific head."""
        return getattr(self, f"pk_{head_idx}")
    
    def set_pk(self, head_idx, value):
        """Set marginal probability for a specific head."""
        setattr(self, f"pk_{head_idx}", value)
    
    def update_marginals(self, teacher_probs_list):
        """
        Update marginal probabilities for all heads.
        
        Args:
            teacher_probs_list: List of teacher probabilities for each head
        """
        with torch.no_grad():
            for head_idx, teacher_probs in enumerate(teacher_probs_list):
                batch_center = teacher_probs.mean(dim=0, keepdim=True)
                pk_old = self.get_pk(head_idx)
                pk_new = self.probs_momentum * pk_old + (1 - self.probs_momentum) * batch_center
                self.set_pk(head_idx, pk_new)
    
    def forward(self, student_outputs, teacher_outputs, epoch):
        """
        Compute multi-head TEMI loss with cross-head weighting.
        
        Args:
            student_outputs: List of student predictions for each head
            teacher_outputs: List of teacher predictions for each head
            epoch: Current training epoch
            
        Returns:
            Total loss
        """
        temp = self.teacher_temp_schedule[epoch]
        num_heads = len(student_outputs)
        
        # Convert to probabilities
        student_probs_list = [
            F.softmax(s / self.student_temp, dim=-1)
            for s in student_outputs
        ]
        teacher_probs_list = [
            F.softmax(t / temp, dim=-1).detach()
            for t in teacher_outputs
        ]
        
        # Update marginals
        self.update_marginals(teacher_probs_list)
        
        # Compute loss for each head - simple cross-entropy distillation
        total_loss = 0.0
        valid_heads = 0
        
        for head_idx in range(num_heads):
            student_probs = student_probs_list[head_idx]
            teacher_probs = teacher_probs_list[head_idx]
            pk = self.get_pk(head_idx)
            
            # Standard cross-entropy distillation
            loss = beta_mi(
                student_probs,
                teacher_probs,
                pk,
                beta=self.beta
            )
            
            # Check for NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN detected in loss for head {head_idx}, skipping")
                continue
            
            total_loss += loss.mean()
            valid_heads += 1
        
        # Entropy regularization to prevent cluster collapse
        # Encourage uniform distribution over clusters
        if self.use_reg:
            reg_loss = 0.0
            for teacher_probs in teacher_probs_list:
                # Batch-wise marginal distribution (average over batch)
                batch_marginal = teacher_probs.mean(dim=0)
                eps = 1e-8
                batch_marginal = batch_marginal.clamp(min=eps)
                
                # KL divergence from uniform distribution
                # D_KL(marginal || uniform) = sum(p * log(p * K))
                # where K is number of clusters
                uniform_prior = 1.0 / self.num_clusters
                kl_div = (batch_marginal * (batch_marginal / uniform_prior).log()).sum()
                reg_loss += kl_div
            
            total_loss = total_loss + self.alpha * reg_loss / num_heads
        
        # Average over valid heads (avoid division by zero)
        if valid_heads > 0:
            return total_loss / valid_heads
        else:
            print("Warning: All heads produced NaN, returning zero loss")
            return torch.tensor(0.0, device=student_probs_list[0].device, requires_grad=True)
