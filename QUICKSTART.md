# TEMI Clustering - Quick Start Guide

This guide will help you get started with running TEMI clustering on CIFAR100.

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU with at least 8GB VRAM (recommended)
- At least 16GB system RAM

## Installation

### Step 1: Install Dependencies

Run the quickstart script to install all required packages:

```bash
python quickstart.py install
```

Or manually install:

```bash
pip install -r requirements.txt
```

## Running the Clustering Experiment

### Option 1: Using the Quickstart Script (Easiest)

To run everything in one command:

```bash
python quickstart.py all
```

This will:
1. Install dependencies
2. Run training
3. Visualize results

### Option 2: Using the Main Training Script

For more control, use the main training script:

```bash
python train.py
```

### Option 3: Step-by-Step

1. **Install dependencies:**
   ```bash
   python quickstart.py install
   ```

2. **Run training:**
   ```bash
   python quickstart.py train
   ```

3. **Visualize results:**
   ```bash
   python quickstart.py visualize
   ```

## What Happens During Training

### Phase 1: Data Loading
- Downloads CIFAR100 dataset (happens once)
- Dataset size: ~170MB
- Takes 1-2 minutes

### Phase 2: Feature Extraction
- Downloads DINOv2 model from torch hub (happens once)
- Extracts embeddings from all images
- Caches embeddings for future runs
- Takes 5-10 minutes on GPU

### Phase 3: Training
- Trains clustering heads for 100 epochs
- Evaluates every 5 epochs
- Saves checkpoints every 10 epochs
- Takes 30-60 minutes depending on GPU

### Phase 4: Results
- Final metrics are printed
- Results saved to `results/final_results.json`
- Best model saved to `checkpoints/checkpoint_best.pth`

## Expected Output

During training, you'll see output like this:

```
================================================================================
TEMI Clustering Configuration
================================================================================

Dataset:
  Dataset................................ CIFAR100
  Number of classes...................... 100
  Image size............................. 224x224

Model:
  DINOv2 model........................... dinov2_vitb14
  Embedding dimension.................... 768
  Number of heads........................ 16
  Number of clusters..................... 100

Training:
  Batch size............................. 256
  Number of epochs....................... 100
  Learning rate.......................... 0.0001
  Weight decay........................... 0.04
  Teacher momentum....................... 0.996

================================================================================

Epoch 1/100: 100%|████████████| 195/195 [00:45<00:00, 4.31it/s, loss=2.3456]

Evaluating at epoch 5...

Test Clustering Metrics:
  Accuracy:      58.34%
  NMI:           72.15%
  Adjusted NMI:  69.82%
  ARI:           45.23%

Test Cluster Statistics:
  Occupied clusters: 98/100 (98.0%)
  Cluster sizes - Min: 12, Max: 156, Mean: 100.0 (±23.4)
```

## Understanding the Results

### Accuracy
- Clustering accuracy after optimal assignment using Hungarian algorithm
- Higher is better (0-100%)
- Expected range: 55-70%

### NMI (Normalized Mutual Information)
- Measures information shared between clusters and true labels
- Higher is better (0-100%)
- Expected range: 65-80%

### ARI (Adjusted Rand Index)
- Similarity measure adjusted for chance
- Higher is better (0-100%)
- Expected range: 40-60%

### Cluster Occupancy
- Percentage of clusters that have at least one sample
- Should be close to 100%
- Low occupancy indicates cluster collapse

## Resuming Training

If training is interrupted, resume from the last checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_latest.pth
```

Or using quickstart:

```bash
python quickstart.py resume
```

## Monitoring Training

### Using TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir logs
```

Then open your browser to http://localhost:6006

### Checking Results

View results programmatically:

```python
import json

with open('results/final_results.json', 'r') as f:
    results = json.load(f)

print(f"Best accuracy: {results['best_accuracy']:.2f}%")
```

## Troubleshooting

### Out of Memory Error

**Problem:** GPU runs out of memory during training

**Solution:** Reduce batch size in `config.py`:
```python
BATCH_SIZE = 128  # or even 64
```

### Slow Training

**Problem:** Training is very slow

**Solutions:**
1. Check GPU is being used: Look for "cuda" in the output
2. Reduce number of workers: Set `NUM_WORKERS = 2` in `config.py`
3. Use smaller DINOv2 model: Set `DINOV2_MODEL = "dinov2_vits14"`

### Poor Results

**Problem:** Clustering accuracy is lower than expected

**Solutions:**
1. Train for more epochs: Set `NUM_EPOCHS = 150` in `config.py`
2. Adjust learning rate: Try `LEARNING_RATE = 5e-5`
3. Tune beta parameter: Try values between 0.5 and 0.8
4. Use larger model: Set `DINOV2_MODEL = "dinov2_vitl14"`

### Cache Issues

**Problem:** Want to recompute embeddings

**Solution:** Force recompute:
```bash
python train.py --force-recompute
```

Or delete cached embeddings:
```bash
rm -rf data/embeddings/
```

## Customization

### Changing Number of Clusters

Edit `config.py`:
```python
NUM_CLUSTERS = 50   # For 50 clusters
NUM_CLUSTERS = 200  # For overclustering
```

### Using Different DINOv2 Model

Edit `config.py`:
```python
# Small model (faster, less accurate)
DINOV2_MODEL = "dinov2_vits14"
EMBEDDING_DIM = 384

# Large model (slower, more accurate)
DINOV2_MODEL = "dinov2_vitl14"
EMBEDDING_DIM = 1024
```

### Adjusting Training Duration

Edit `config.py`:
```python
NUM_EPOCHS = 150        # More epochs
EVAL_FREQ = 10          # Evaluate less frequently
SAVE_CHECKPOINT_FREQ = 20  # Save less frequently
```

## File Locations

After running, your directory will look like:

```
clustering/
├── data/
│   ├── cifar-100-python/        # CIFAR100 dataset
│   └── embeddings/               # Cached DINOv2 embeddings
├── checkpoints/
│   ├── checkpoint_best.pth       # Best model
│   ├── checkpoint_latest.pth     # Latest checkpoint
│   └── checkpoint_epoch_*.pth    # Periodic checkpoints
├── logs/                         # TensorBoard logs
└── results/
    ├── final_results.json        # Final metrics
    └── training_curves.png       # Training plots
```

## Next Steps

After successful training:

1. **Visualize results:**
   ```bash
   python visualize_results.py
   ```

2. **View TensorBoard:**
   ```bash
   tensorboard --logdir logs
   ```

3. **Experiment with hyperparameters:**
   - Edit `config.py`
   - Rerun training
   - Compare results

4. **Try different configurations:**
   - Different cluster counts
   - Different DINOv2 models
   - Different loss parameters

## Getting Help

If you encounter issues:

1. Check this guide for common problems
2. Review the error messages carefully
3. Check the logs in the `logs/` directory
4. Verify all dependencies are installed correctly

## Performance Tips

For best performance:

1. **Use GPU:** Training on CPU is very slow
2. **Enable caching:** Don't use `--force-recompute` unless necessary
3. **Optimal batch size:** 256 works well for most GPUs
4. **Monitor GPU usage:** Use `nvidia-smi` to check utilization

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Adaloglou_2023_BMVC,
    author    = {Nikolas Adaloglou and Felix Michels and Hamza Kalisch and Markus Kollmann},
    title     = {Exploring the Limits of Deep Image Clustering using Pretrained Models},
    booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023},
    year      = {2023},
}
```
