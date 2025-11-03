# TEMI Deep Clustering on CIFAR100 using DINOv2

This project implements deep clustering on the CIFAR100 dataset using the TEMI (Trustworthy Evidence from Mutual Information) method with DINOv2 pretrained features.

## Overview

The implementation follows the methodology described in "Exploring the Limits of Deep Image Clustering using Pretrained Models" (BMVC 2023). The approach uses:

- **DINOv2** vision transformers for feature extraction
- **TEMI loss** with weighted mutual information for clustering
- **Multi-head architecture** for robust ensemble predictions
- **Teacher-student framework** with exponential moving average updates

## Project Structure

```
clustering/
├── config.py                    # Configuration and hyperparameters
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
├── models/
│   ├── clustering_model.py     # Multi-head clustering model
│   └── loss.py                  # TEMI loss implementation
├── utils/
│   ├── data_utils.py           # Data loading utilities
│   ├── feature_extractor.py    # DINOv2 feature extraction
│   ├── eval_utils.py           # Evaluation metrics
│   └── trainer.py              # Training loop and checkpointing
├── data/                        # Dataset and cached embeddings
├── checkpoints/                 # Model checkpoints
├── logs/                        # TensorBoard logs
└── results/                     # Final results and metrics
```

## Installation

### Requirements

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- At least 8GB RAM

### Setup

1. Clone this repository and navigate to the project directory:

```bash
cd clustering
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The installation will include:
- PyTorch and torchvision for deep learning
- DINOv2 models (automatically downloaded from torch hub)
- scikit-learn for evaluation metrics
- TensorBoard for training visualization

## Usage

### Basic Training

To start training with default settings:

```bash
python train.py
```

This will:
1. Download CIFAR100 dataset (if not already present)
2. Load pretrained DINOv2 model
3. Extract and cache image embeddings
4. Train clustering heads for 100 epochs
5. Save checkpoints and results

### Resume Training

To resume from a checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_latest.pth
```

### Force Recompute Embeddings

To recompute embeddings even if cache exists:

```bash
python train.py --force-recompute
```

## Configuration

All hyperparameters are defined in `config.py`. Key settings include:

### Model Configuration
- `DINOV2_MODEL`: DINOv2 variant (vits14, vitb14, vitl14, vitg14)
- `NUM_CLUSTERS`: Number of clusters (100 for CIFAR100)
- `NUM_HEADS`: Number of clustering heads (16 by default)

### Training Configuration
- `BATCH_SIZE`: Batch size for training (256)
- `NUM_EPOCHS`: Total training epochs (100)
- `LEARNING_RATE`: Initial learning rate (1e-4)
- `MOMENTUM_TEACHER`: EMA momentum for teacher (0.996)

### Loss Configuration
- `BETA`: Beta parameter for weighted MI (0.6)
- `STUDENT_TEMP`: Student temperature (0.1)
- `TEACHER_TEMP`: Teacher temperature (0.05)

To modify settings, edit the values in `config.py`.

## Results

After training completes, results are saved in multiple formats:

### Checkpoints
- `checkpoints/checkpoint_best.pth`: Best model by accuracy
- `checkpoints/checkpoint_latest.pth`: Most recent checkpoint
- `checkpoints/checkpoint_epoch_XXXX.pth`: Periodic checkpoints

### Results
- `results/final_results.json`: Complete metrics and training history
- Contains accuracy, NMI, ARI, and cluster statistics

### Logs
- `logs/`: TensorBoard logs for visualization

To view TensorBoard logs:

```bash
tensorboard --logdir logs
```

## Expected Performance

Based on the TEMI paper, expected clustering accuracy on CIFAR100:

- **K-means baseline**: ~52%
- **TEMI with DINOv2**: ~65-70%

The exact performance depends on:
- DINOv2 model size (larger models generally perform better)
- Number of training epochs
- Hyperparameter tuning

## Evaluation Metrics

The implementation computes several clustering metrics:

- **Accuracy**: Using Hungarian algorithm for optimal cluster-to-class assignment
- **NMI** (Normalized Mutual Information): Measures mutual information between clusters and classes
- **ANMI** (Adjusted NMI): Adjusted version accounting for chance
- **ARI** (Adjusted Rand Index): Similarity measure between clusterings

## Checkpointing and Recovery

The implementation includes robust checkpointing:

### Automatic Checkpointing
- Saves checkpoint every N epochs (configurable)
- Saves best model based on accuracy
- Saves latest checkpoint for recovery

### Manual Recovery
If training is interrupted:

1. The latest checkpoint is automatically saved
2. Resume with: `python train.py --resume checkpoints/checkpoint_latest.pth`
3. Training continues from the saved epoch

### Emergency Checkpoints
- Keyboard interrupt (Ctrl+C) triggers checkpoint save
- Unexpected errors also trigger emergency checkpoint

## Advanced Usage

### Using Different DINOv2 Models

Edit `config.py` to change the model:

```python
DINOV2_MODEL = "dinov2_vitl14"  # Larger model
EMBEDDING_DIM = 1024            # Update dimension accordingly
```

Available models:
- `dinov2_vits14`: Small (384 dim)
- `dinov2_vitb14`: Base (768 dim) - default
- `dinov2_vitl14`: Large (1024 dim)
- `dinov2_vitg14`: Giant (1536 dim)

### Adjusting Cluster Count

For different clustering granularities:

```python
NUM_CLUSTERS = 50   # Fewer clusters
NUM_CLUSTERS = 200  # More clusters (overclustering)
```

### Hyperparameter Tuning

Key hyperparameters to experiment with:

```python
BETA = 0.6              # Controls MI weighting (0.5-1.0)
NUM_HEADS = 16          # More heads = more robust (4-32)
LEARNING_RATE = 1e-4    # Learning rate (1e-5 to 1e-3)
MOMENTUM_TEACHER = 0.996 # Teacher update rate (0.99-0.999)
```

## Implementation Details

### Feature Extraction
- DINOv2 embeddings are extracted once and cached
- Embeddings are stored in `data/embeddings/`
- Caching significantly speeds up training iterations

### Loss Function
The TEMI loss combines:
1. Weighted mutual information between student and teacher
2. Similarity weighting based on teacher predictions
3. Temperature-scaled softmax distributions
4. EMA updates for marginal cluster probabilities

### Teacher-Student Architecture
- Student network is trained with gradient descent
- Teacher network updated via EMA of student weights
- Provides stable training targets and prevents collapse

### Multi-Head Ensemble
- Multiple clustering heads trained in parallel
- Final predictions use majority voting
- Improves robustness and handles ambiguous samples

## Troubleshooting

### Out of Memory Errors

Reduce batch size in `config.py`:
```python
BATCH_SIZE = 128  # or 64
```

### Poor Clustering Performance

Try adjusting:
- Increase training epochs
- Use larger DINOv2 model
- Tune beta parameter (0.5-0.8)
- Increase number of heads

### Training Divergence

If loss becomes NaN or diverges:
- Lower learning rate
- Increase warmup epochs
- Enable gradient clipping

## Citation

If you use this code, please cite the original TEMI paper:

```bibtex
@inproceedings{Adaloglou_2023_BMVC,
    author    = {Nikolas Adaloglou and Felix Michels and Hamza Kalisch and Markus Kollmann},
    title     = {Exploring the Limits of Deep Image Clustering using Pretrained Models},
    booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023},
    year      = {2023},
    url       = {https://papers.bmvc2023.org/0297.pdf}
}
```

## License

This implementation is provided for research and educational purposes. The original TEMI method is from the paper cited above.

## Contact

For questions or issues, please open an issue on the repository.
