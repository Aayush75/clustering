# NaN Loss Troubleshooting Guide

If you're experiencing NaN (Not a Number) losses during training, this guide will help you diagnose and fix the issue.

## Quick Diagnosis

Run the diagnostic script:
```bash
python test_nan_issues.py
```

This will test:
- Model forward pass
- Loss computation
- Gradient flow
- Numerical stability
- Edge cases

## Common Causes of NaN Loss

### 1. Learning Rate Too High

**Symptom**: Loss becomes NaN after a few batches

**Solution**: Reduce learning rate in `config.py`:
```python
LEARNING_RATE = 5e-5  # Lower from 1e-4
```

### 2. Gradient Explosion

**Symptom**: Loss suddenly jumps to NaN

**Solution**: Enable gradient clipping in `config.py`:
```python
CLIP_GRAD = 1.0  # Lower from 3.0
```

### 3. Numerical Instability in Loss

**Symptom**: NaN appears in specific heads

**Solution**: The code now automatically:
- Adds epsilon (1e-8) to prevent log(0)
- Clamps values to prevent extreme numbers
- Skips batches/heads with NaN
- Uses safer temperature values

### 4. Bad Initialization

**Symptom**: NaN from the very first batch

**Solution**: Model weights are now initialized with smaller std (0.01)

### 5. Temperature Issues

**Symptom**: Loss is NaN when temperature is very low

**Check temperatures in config.py**:
```python
STUDENT_TEMP = 0.1    # Don't go below 0.05
TEACHER_TEMP = 0.05   # Don't go below 0.04
```

## Fixes Applied

The following fixes have been implemented to prevent NaN:

### 1. Numerical Stability in `beta_mi` function
```python
# Added epsilon for numerical stability
eps = 1e-8
pk = pk.clamp(min=eps)
beta_emi = (((p1 * p2).clamp(min=eps) ** beta) / pk).sum(dim=-1)
beta_pmi = beta_emi.clamp(min=eps).log().clamp(min=clip_min)
```

### 2. Safe Weight Normalization
```python
# Check for zero sum before division
weight_sum = weight.sum()
if weight_sum > 1e-8:
    weight = weight / weight_sum
else:
    weight = torch.ones_like(weight) / weight.shape[0]
```

### 3. NaN Detection and Skipping
```python
# Skip batches with NaN
if torch.isnan(loss).any():
    print(f"Warning: NaN detected, skipping")
    continue
```

### 4. Gradient Checking
```python
# Check gradients before optimizer step
if not torch.isfinite(grad_norm):
    print(f"Warning: Non-finite gradients, skipping batch")
    self.optimizer.zero_grad()
    continue
```

### 5. Safer Loss Limits
```python
# Use -100.0 instead of -inf
clip_min=-100.0  # Prevent extreme values
```

## Step-by-Step Debugging

If you still get NaN, follow these steps:

### Step 1: Check Configuration
```bash
python test_installation.py
```
Ensure all packages are installed correctly.

### Step 2: Run Diagnostics
```bash
python test_nan_issues.py
```
This will pinpoint where NaN originates.

### Step 3: Try Conservative Settings

Edit `config.py` with safer values:
```python
# Conservative hyperparameters
LEARNING_RATE = 5e-5      # Lower learning rate
CLIP_GRAD = 1.0           # Aggressive clipping
STUDENT_TEMP = 0.1        # Safe temperature
TEACHER_TEMP = 0.05       # Safe temperature  
BETA = 0.5                # Lower beta
BATCH_SIZE = 128          # Smaller batches
```

### Step 4: Check Data

Verify embeddings are valid:
```python
import torch

# Load cached embeddings
train_emb = torch.load('data/embeddings/embeddings_train.pt')
embeddings = train_emb['embeddings']

# Check for NaN or inf
print(f"Has NaN: {torch.isnan(embeddings).any()}")
print(f"Has inf: {torch.isinf(embeddings).any()}")
print(f"Min: {embeddings.min()}, Max: {embeddings.max()}")
```

If embeddings have NaN, recompute them:
```bash
python train.py --force-recompute
```

### Step 5: Enable Verbose Logging

The code now prints warnings when:
- NaN is detected in loss
- NaN is detected in gradients
- Batches are skipped
- Heads produce invalid outputs

Watch for these messages during training.

## What the Code Does Now

### Automatic Recovery
- **Skips bad batches**: Continues training instead of crashing
- **Handles NaN gracefully**: Warns and continues with valid heads
- **Gradient checking**: Prevents optimizer from using bad gradients
- **Finite loss checking**: Verifies loss before backward pass

### Safety Mechanisms
- **Epsilon addition**: Prevents log(0) and division by zero
- **Value clamping**: Keeps values in safe numerical range
- **Temperature limits**: Prevents extreme softmax values
- **Smaller initialization**: Reduces initial instability

### Monitoring
- **NaN detection**: Automatic checking after each operation
- **Warning messages**: Clear indication of where issues occur
- **Loss tracking**: Shows when values become invalid

## Expected Behavior Now

With the fixes applied, you should see:
- Finite loss values at all times
- Occasional warnings if numerical issues are detected
- Automatic skipping of problematic batches
- Training continues smoothly

If loss becomes NaN consistently:
1. Run `python test_nan_issues.py`
2. Try conservative settings above
3. Check that embeddings are valid
4. Reduce learning rate and batch size

## Still Having Issues?

If NaN persists after all fixes:

1. **Delete cached embeddings and recompute**:
   ```bash
   rm -rf data/embeddings/
   python train.py --force-recompute
   ```

2. **Use even more conservative settings**:
   ```python
   LEARNING_RATE = 1e-5
   CLIP_GRAD = 0.5
   BATCH_SIZE = 64
   BETA = 0.4
   ```

3. **Try different DINOv2 model**:
   ```python
   DINOV2_MODEL = "dinov2_vits14"  # Smaller, more stable
   EMBEDDING_DIM = 384
   ```

4. **Check GPU/CUDA**:
   Some GPUs have numerical precision issues. Try:
   ```python
   USE_FP16 = False  # Ensure full precision
   ```

## Summary

The code now includes:
- ✅ Numerical stability fixes
- ✅ NaN detection and handling
- ✅ Automatic batch skipping
- ✅ Gradient checking
- ✅ Safe value clamping
- ✅ Warning messages
- ✅ Diagnostic tools

Training should be much more stable now!
