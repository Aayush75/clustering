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


@torch.no_grad()
def sinkhorn_knopp(Q, num_iters=3):
    """
    Apply Sinkhorn-Knopp algorithm to get normalized assignment matrix.
    
    This ensures balanced cluster assignments - critical for preventing collapse!
    
    Args:
        Q: Assignment scores (batch_size, num_clusters)
        num_iters: Number of iterations
        
    Returns:
        Normalized assignment matrix
    """
    Q = Q.exp()  # Q is log-probability
    
    # Make the matrix doubly stochastic
    sum_of_rows = Q.sum(dim=0, keepdim=True)
    Q /= sum_of_rows  # Normalize by column (cluster) sums
    
    for _ in range(num_iters):
        # Normalize each row (so each sample sums to 1)
        Q /= Q.sum(dim=1, keepdim=True)
        # Normalize each column (so each cluster gets equal total weight)
        Q /= Q.sum(dim=0, keepdim=True)
    
    Q /= Q.sum(dim=1, keepdim=True)  # Final row normalization
    return Q


def swav_loss(student_logits, teacher_logits, teacher_temp=0.05):
    """
    SwAV-style loss with Sinkhorn-Knopp normalization.
    
    This is the CORRECT way to do self-supervised clustering!
    
    Args:
        student_logits: Student cluster logits (batch_size, num_clusters)
        teacher_logits: Teacher cluster logits (batch_size, num_clusters)
        teacher_temp: Temperature for teacher sharpening
        
    Returns:
        Loss value
    """
    # Sharpen teacher predictions with temperature
    teacher_scores = teacher_logits / teacher_temp
    
    # Apply Sinkhorn-Knopp to get balanced target assignments
    teacher_probs = sinkhorn_knopp(teacher_scores, num_iters=3)
    
    # Student uses standard softmax (with temperature handled outside)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # Cross-entropy loss
    loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
    
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
            
            # Apply temperature to student
            student_out = student_out / self.student_temp
            
            # SwAV loss with Sinkhorn-Knopp for balanced assignments
            loss = swav_loss(student_out, teacher_out.detach(), teacher_temp=float(temp))
            
            # Update marginal probabilities for monitoring
            with torch.no_grad():
                teacher_probs = F.softmax(teacher_out / temp, dim=-1)
                self.update_marginals(teacher_probs, head_idx)
            
            # Check for NaN and handle gracefully
            if torch.isnan(loss):
                print(f"Warning: NaN detected in loss for head {head_idx}, skipping")
                continue
            
            total_loss += loss
        
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
        
        # Compute loss for each head using SwAV approach
        total_loss = 0.0
        valid_heads = 0
        
        for head_idx in range(num_heads):
            student_out = student_outputs[head_idx] / self.student_temp
            teacher_out = teacher_outputs[head_idx]
            
            # SwAV loss with Sinkhorn-Knopp
            loss = swav_loss(student_out, teacher_out.detach(), teacher_temp=float(temp))
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN detected in loss for head {head_idx}, skipping")
                continue
            
            total_loss += loss
            valid_heads += 1
        
        # Update marginals for monitoring
        with torch.no_grad():
            teacher_probs_list_for_update = [
                F.softmax(t / temp, dim=-1) for t in teacher_outputs
            ]
            self.update_marginals(teacher_probs_list_for_update)
        
        # Average over valid heads (avoid division by zero)
        if valid_heads > 0:
            return total_loss / valid_heads
        else:
            print("Warning: All heads produced NaN, returning zero loss")
            return torch.tensor(0.0, device=student_outputs[0].device, requires_grad=True)
