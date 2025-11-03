# Critical Fixes Applied to TEMI Clustering Implementation

## Overview
This document summarizes the critical fixes applied to resolve the low accuracy issue (12% → expected 65-70%).

## Problem Diagnosis

### Symptoms
- Accuracy plateaued at ~12% (expected 65-70%)
- Loss values very small (~0.055) 
- No learning/improvement after first epoch
- Training appeared stable but ineffective

### Root Causes Identified
1. **Incorrect loss implementation**: `beta_mi` function did not correctly implement TEMI's weighted mutual information
2. **Missing architecture parameter**: `HIDDEN_DIM` not defined in config for MLP clustering heads
3. **Suboptimal hyperparameters**: Learning rate too low, regularization disabled
4. **Wrong entropy regularization**: Implementation didn't prevent cluster collapse

## Fixes Applied

### 1. Configuration Changes (`config.py`)

#### Added Parameters
```python
HIDDEN_DIM = 2048  # Hidden dimension for MLP clustering heads
```

#### Updated Hyperparameters
- `LEARNING_RATE`: 1e-4 → **1e-3** (10x increase for faster convergence)
- `MIN_LR`: 1e-4 → **1e-5** (allow lower final LR)
- `USE_REGULARIZATION`: False → **True** (MUST be enabled)
- `REGULARIZATION_WEIGHT`: 0.5 → **5.0** (stronger regularization for cluster balance)
- `WARMUP_EPOCHS`: 20 → **10** (shorter warmup)
- `WEIGHT_DECAY`: 0.04 → **1e-5** (lower to prevent over-regularization)

### 2. Loss Function Fixes (`models/loss.py`)

#### Fixed `beta_mi` Function
**OLD (Incorrect)**:
```python
# Computed: E[(p1 * p2)^beta / pk]
beta_emi = (((p1 * p2).clamp(min=eps) ** beta) / pk).sum(dim=-1)
beta_pmi = beta_emi.clamp(min=eps).log().clamp(min=clip_min)
return -beta_pmi
```

**NEW (Correct - TEMI Paper)**:
```python
# Compute joint probability: p(i,j) = p1(i) * p2(j)
joint = (p1 * p2).clamp(min=eps)

# Compute PMI: log(p(i,j) / (p(i) * p(j))) = log(joint / pk)
pmi = (joint / pk).clamp(min=eps).log().clamp(min=clip_min, max=100.0)

# Compute beta-weighted MI: (joint^beta / pk^(beta-1)) * PMI
weight = ((joint / pk).clamp(min=eps) ** beta) * pk

# Sum over clusters to get I_beta
beta_mi_value = (weight * pmi).sum(dim=-1)
return -beta_mi_value
```

**Key Changes**:
- Now correctly computes pointwise mutual information (PMI)
- Applies beta weighting as per TEMI paper formula
- Properly normalizes by marginal probabilities
- Added upper clipping (max=100.0) for stability

#### Fixed Entropy Regularization
**OLD (Wrong - maximized entropy incorrectly)**:
```python
# Per-sample entropy (not useful for cluster balance)
entropy = -(student_probs * (student_probs + eps).log()).sum(dim=-1)
reg_loss += entropy.mean()
```

**NEW (Correct - prevents cluster collapse)**:
```python
# Batch-wise marginal distribution
batch_marginal = student_probs.mean(dim=0)
batch_marginal = batch_marginal.clamp(min=eps)

# Negative entropy: -H(p) = sum(p * log(p))
# Minimizing this maximizes entropy → uniform cluster distribution
neg_entropy = (batch_marginal * batch_marginal.log()).sum()
reg_loss += neg_entropy
```

**Key Changes**:
- Uses batch-level marginal distribution (not per-sample)
- Minimizes negative entropy to maximize entropy
- Encourages uniform cluster distribution
- Prevents cluster collapse

### 3. Model Architecture Fixes (`models/clustering_model.py`)

#### Enhanced Clustering Heads
**OLD**:
```python
hidden_dim=None  # Linear heads
```

**NEW**:
```python
hidden_dim=self.hidden_dim  # MLP heads with BatchNorm + Dropout

# Architecture:
nn.Linear(input_dim, hidden_dim)  # 768 → 2048
nn.BatchNorm1d(hidden_dim)
nn.GELU()
nn.Dropout(0.1)
nn.Linear(hidden_dim, num_clusters)  # 2048 → 100
```

**Benefits**:
- Increased model capacity for complex clustering
- BatchNorm for training stability
- Dropout for regularization
- Better gradient flow

## Expected Results

### Before Fixes
- Accuracy: ~12%
- Loss: ~0.055
- No improvement after epoch 1

### After Fixes (Expected)
- Accuracy: **65-70%** (matching TEMI paper on CIFAR100)
- Loss: **Larger values** (0.5-2.0 initially, decreasing)
- Steady improvement over epochs
- Better cluster balance

## How to Test

1. **Delete old checkpoints**:
```powershell
Remove-Item "c:\Users\Aayush Kuloor\IITGN\clustering\checkpoints\*" -Recurse -Force
```

2. **Run training**:
```powershell
cd "c:\Users\Aayush Kuloor\IITGN\clustering"
python train.py
```

3. **Monitor progress**:
- Check loss values (should be larger, ~0.5-2.0 initially)
- Watch accuracy increase steadily
- Verify cluster balance (all clusters should be used)
- Expected to reach 65-70% accuracy by epoch 50-100

## Technical Details

### TEMI Loss Formula
The correct implementation follows:
$$I_\beta(p_1, p_2) = \sum_k \left[\frac{(p_1(k) \cdot p_2(k))^\beta}{p_k^{\beta-1}}\right] \cdot \log\frac{p_1(k) \cdot p_2(k)}{p_k}$$

Where:
- $p_1, p_2$: Student and teacher probability distributions
- $p_k$: Marginal cluster probabilities (EMA updated)
- $\beta$: Weighting parameter (0.6)

### Entropy Regularization
Maximizes batch-level entropy:
$$\mathcal{L}_{\text{reg}} = -H(\bar{p}) = \sum_k \bar{p}_k \log \bar{p}_k$$

Where $\bar{p}_k = \frac{1}{B}\sum_{i=1}^B p_i(k)$ is the batch marginal.

## Verification Checklist

- [x] `HIDDEN_DIM` added to config
- [x] Learning rate increased to 1e-3
- [x] `USE_REGULARIZATION` enabled
- [x] `beta_mi` function corrected to TEMI formula
- [x] Entropy regularization fixed to use batch marginals
- [x] Clustering heads use MLP architecture with BatchNorm
- [x] Gradient clipping configured (3.0)
- [x] All numerical stability checks in place

## Remaining Best Practices

All code already includes:
- NaN detection and batch skipping
- Gradient checking before optimizer step
- Checkpoint saving (best, latest, periodic)
- Resume from checkpoint capability
- TensorBoard logging
- Comprehensive evaluation metrics

## References

- Paper: "Exploring the Limits of Deep Image Clustering using Pretrained Models" (BMVC 2023)
- Expected CIFAR100 accuracy: 65-70% with 100 clusters
- DINOv2: facebook/dinov2-base (768-dim embeddings)
