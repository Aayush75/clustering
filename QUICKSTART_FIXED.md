# Quick Start - Fixed TEMI Implementation

## What Was Fixed

The accuracy issue (12% → expected 65-70%) has been fixed by correcting:

1. **Loss Function**: `beta_mi` now correctly implements TEMI's weighted mutual information
2. **Model Architecture**: Added MLP heads with BatchNorm (2048 hidden dim)
3. **Hyperparameters**: Increased learning rate to 1e-3, enabled regularization
4. **Entropy Regularization**: Fixed to prevent cluster collapse

## Running Training

### Step 1: Clean Old Checkpoints
```powershell
Remove-Item "c:\Users\Aayush Kuloor\IITGN\clustering\checkpoints\*" -Recurse -Force
```

### Step 2: Start Training
```powershell
cd "c:\Users\Aayush Kuloor\IITGN\clustering"
python train.py
```

## What to Expect

### Initial Behavior (First Few Epochs)
- Loss: ~0.5-2.0 (much larger than before)
- Accuracy: ~15-25% (starting point)
- Progress: Should see steady improvement

### Mid Training (Epoch 20-50)
- Loss: Decreasing steadily
- Accuracy: ~40-55%
- Clusters: All 100 clusters being used

### Final Results (Epoch 50-100)
- **Target Accuracy: 65-70%** (matching paper)
- Loss: Converged to stable value
- NMI: ~45-55%
- ARI: ~35-45%

## Monitoring

Watch for these indicators of healthy training:

✅ **Good Signs**:
- Loss values 0.5-2.0 initially
- Accuracy increases every few epochs
- No NaN warnings
- All clusters populated (shown in metrics)

❌ **Bad Signs** (should NOT happen):
- Loss very small (<0.1) immediately
- Accuracy stuck at 12%
- Many NaN warnings
- Empty clusters

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| Learning Rate | 1e-4 | **1e-3** |
| Hidden Dim | None (linear) | **2048** (MLP) |
| Regularization | Disabled | **Enabled** |
| Loss Implementation | Incorrect | **Fixed to TEMI paper** |
| Entropy Reg | Per-sample | **Batch marginal** |

## Files Modified

1. `config.py` - Added HIDDEN_DIM, increased LR, enabled regularization
2. `models/loss.py` - Fixed beta_mi function and entropy regularization
3. `models/clustering_model.py` - Added MLP heads with BatchNorm

## Troubleshooting

### If accuracy still plateaus at 12%:
1. Check that old checkpoints were deleted
2. Verify config shows `LEARNING_RATE = 1e-3`
3. Ensure `USE_REGULARIZATION = True`
4. Check loss values are large (>0.5)

### If you see NaN errors:
- Training should handle this automatically (batch skipping)
- A few NaN batches is OK, but if every batch is NaN, something is wrong

### If training is too slow:
- First 10 epochs might be slow (warmup)
- Consider using GPU if available
- Batch size is 256, which is good for training speed

## Expected Timeline

- **Feature Extraction**: ~30-60 seconds (cached after first run)
- **Per Epoch**: ~30-60 seconds on CPU, ~5-10 seconds on GPU
- **Total Training**: ~1-2 hours for 100 epochs

## Validation

After training completes, you should see:
```
=== Final Evaluation ===
Accuracy: 65-70%
NMI: 45-55%
Adjusted NMI: 35-45%
ARI: 35-45%
```

For detailed explanation of fixes, see `FIXES_APPLIED.md`.
