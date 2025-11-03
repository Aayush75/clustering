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


def beta_mi(p1, p2, pk, beta=1.0, clip_min=-torch.inf):
    """
    Compute beta-weighted mutual information.
    
    This is the core of the TEMI loss, measuring pointwise mutual information
    with a beta exponent and marginal probability normalization.
    
    Args:
        p1: First probability distribution (batch_size, num_clusters)
        p2: Second probability distribution (batch_size, num_clusters)
        pk: Marginal cluster probabilities (1, num_clusters)
        beta: Beta exponent for MI
        clip_min: Minimum value for log (for numerical stability)
        
    Returns:
        Negative beta-MI (batch_size,)
    """
    # Compute beta-weighted expected MI: E[(p1 * p2)^beta / pk]
    beta_emi = (((p1 * p2) ** beta) / pk).sum(dim=-1)
    
    # Take log to get pointwise MI
    beta_pmi = beta_emi.log().clamp(min=clip_min)
    
    # Return negative for loss minimization
    return -beta_pmi


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
            
            # Compute similarity weight between teacher predictions
            # This measures consistency of the teacher across the batch
            weight = sim_weight(teacher_probs, teacher_probs)
            weight = weight / weight.sum()
            
            # Compute weighted beta-MI loss
            loss = weight * beta_mi(
                student_probs,
                teacher_probs,
                pk,
                beta=self.beta,
                clip_min=-torch.inf if self.beta > 0 else 0
            )
            
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
        
        # Compute pairwise weights between all teacher heads
        total_weight = 0.0
        pair_count = 0
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                weight = sim_weight(teacher_probs_list[i], teacher_probs_list[j])
                total_weight += weight
                pair_count += 1
        
        # Normalize weight
        if pair_count > 0:
            avg_weight = total_weight / pair_count
            if isinstance(avg_weight, torch.Tensor):
                avg_weight = avg_weight / avg_weight.sum()
        else:
            avg_weight = torch.ones(1).to(teacher_outputs[0].device)
        
        # Compute MI loss for each head
        total_loss = 0.0
        
        for head_idx in range(num_heads):
            student_probs = student_probs_list[head_idx]
            teacher_probs = teacher_probs_list[head_idx]
            pk = self.get_pk(head_idx)
            
            # Weighted beta-MI
            loss = avg_weight * beta_mi(
                student_probs,
                teacher_probs,
                pk,
                beta=self.beta,
                clip_min=-torch.inf
            )
            
            total_loss += loss.mean()
        
        # Optional entropy regularization
        if self.use_reg:
            reg_loss = 0.0
            for student_probs in student_probs_list:
                # Self-entropy: H(p) = -sum(p * log(p))
                entropy = -(student_probs * student_probs.log()).sum(dim=-1)
                reg_loss += entropy.mean()
            
            total_loss = total_loss + self.alpha * reg_loss / num_heads
        
        # Average over heads
        return total_loss / num_heads
