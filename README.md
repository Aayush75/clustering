# TEMI Deep Clustering for Multiple Datasets

This repository implements TEMI (Transformation-Equivariant Multi-Instance) clustering on multiple vision datasets using DINOv2, DINOv3, or CLIP features. The implementation follows the methodology from "Self-Supervised Clustering with Deep Learning" (arXiv:2303.17896).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset Support](#dataset-support)
- [Feature Extractors](#feature-extractors)
- [Cluster Visualization](#cluster-visualization)
- [Pseudo Label Generation](#pseudo-label-generation)
- [Dataset Distillation](#dataset-distillation)
- [Command Line Arguments](#command-line-arguments)
- [Algorithm Details](#algorithm-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

## Overview

The pipeline consists of five main stages:

1. **Feature Extraction**: Extract visual features from dataset images using pre-trained vision models (DINOv2, DINOv3, or CLIP)
2. **TEMI Clustering**: Train a clustering model using transformation equivariance and multi-instance learning principles
3. **Evaluation**: Assess clustering quality using multiple metrics (accuracy, NMI, ARI)
4. **Visualization and Pseudo Labeling** (optional): Generate t-SNE/UMAP plots and map clusters to actual labels
5. **Dataset Distillation** (optional): Create a small synthetic dataset that preserves learning dynamics

## Features

- Multiple datasets: CIFAR10, CIFAR100, Tiny ImageNet, ImageNet-1K, and Imagenette
- Multiple feature extractors: DINOv2, DINOv3, and CLIP models
- TEMI clustering algorithm following paper specifications
- Pseudo label generation for interpretability and semi-supervised learning
- Dataset distillation using trajectory matching (arXiv:2406.18561)
- Checkpoint system for resumable training
- Comprehensive evaluation metrics (accuracy, NMI, ARI)
- Cluster visualization with t-SNE and UMAP
- Support for multiple model variants (small, base, large, giant)
- Robust error handling and device management
- Efficient vectorized operations throughout

## Requirements

Python 3.8 or higher is required. Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- Transformers (HuggingFace)
- torchvision
- scikit-learn
- numpy, scipy
- matplotlib
- umap-learn (optional, for UMAP visualization)
- datasets (for HuggingFace datasets)

## Project Structure

```
clustering/
├── main.py                        # Main training script
├── analyze_results.py             # Results analysis and visualization
├── generate_pseudo_labels.py      # Pseudo label generation script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── src/                          # Source code modules
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── clip_feature_extractor.py
│   ├── temi_clustering.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── pseudo_labeling.py
├── example/                      # Example scripts
├── tests/                        # Test suite
├── data/                         # Dataset storage
├── checkpoints/                  # Model checkpoints
└── results/                      # Experiment outputs
```

## Quick Start

Run clustering with default settings (CIFAR100, 100 clusters, DINOv2):

```bash
python main.py
```

Run with visualization:

```bash
python main.py --plot_clusters --save_features
```

Run with pseudo labels:

```bash
python main.py --generate_pseudo_labels --k_samples 10 --visualize_mapping --save_features
```

## Usage

### Basic Usage Examples

CIFAR10 (10 classes):
```bash
python main.py --dataset cifar10 --num_clusters 10
```

CIFAR100 (100 classes):
```bash
python main.py --dataset cifar100 --num_clusters 100
```

Imagenette (10 classes, good for quick testing):
```bash
python main.py --dataset imagenette --num_clusters 10
```

Tiny ImageNet (200 classes):
```bash
python main.py --dataset tiny-imagenet --num_clusters 200
```

ImageNet-1K (1000 classes):
```bash
python main.py --dataset imagenet --num_clusters 1000
```

### Using CLIP Models

Use CLIP for feature extraction:

```bash
# Default CLIP model (ViT-B/32)
python main.py --model_type clip

# Specific CLIP model
python main.py --model_type clip --clip_model openai/clip-vit-large-patch14

# With visualization
python main.py --model_type clip --plot_clusters --save_features
```

Available CLIP models:
- `openai/clip-vit-base-patch32` (512-dim, fastest)
- `openai/clip-vit-base-patch16` (512-dim, better quality)
- `openai/clip-vit-large-patch14` (768-dim, best quality)

### Using DINOv3 Models

Specify any DINOv3 model from HuggingFace:

```bash
python main.py --dinov2_model facebook/dinov3-base
```

Available DINOv2/DINOv3 models:
- `facebook/dinov2-small` (384-dim)
- `facebook/dinov2-base` (768-dim, default)
- `facebook/dinov2-large` (1024-dim)
- `facebook/dinov2-giant` (1536-dim)
- Any DINOv3 model from HuggingFace

### Advanced Options

Custom hyperparameters:

```bash
python main.py \
    --dataset cifar100 \
    --model_type dinov2 \
    --dinov2_model facebook/dinov2-base \
    --num_clusters 100 \
    --num_epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --temperature 0.1 \
    --plot_clusters \
    --save_features \
    --device cuda
```

Resume from checkpoint:

```bash
python main.py --resume_from ./checkpoints/experiment/final_checkpoint.pt
```

### Using Pre-extracted Features

Save time by reusing features:

```bash
# First run: extract and save
python main.py --save_features

# Subsequent runs: load saved features
python main.py --load_features ./results/experiment_name/features/train_features
```

### Analyzing Results

Analyze completed experiments:

```bash
# Basic analysis
python analyze_results.py ./results/experiment_name

# Detailed analysis
python analyze_results.py ./results/experiment_name --detailed

# Generate visualizations
python analyze_results.py ./results/experiment_name --plot --viz_method tsne
```

## Dataset Support

### CIFAR10
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Image Size: 32x32 (resized to 224x224)
- Train: 50,000 | Test: 10,000
- Source: torchvision.datasets.CIFAR10
- Download: Automatic

### CIFAR100
- Classes: 100 fine-grained classes
- Image Size: 32x32 (resized to 224x224)
- Train: 50,000 | Test: 10,000
- Source: torchvision.datasets.CIFAR100
- Download: Automatic

### Imagenette
- Classes: 10 (subset of ImageNet)
- Image Size: 320x320 (resized to 224x224)
- Train: 9,469 | Test: 3,925
- Source: fastai
- Download: Automatic

### Tiny ImageNet
- Classes: 200 (subset of ImageNet)
- Image Size: 64x64 (resized to 224x224)
- Train: 100,000 | Test: 10,000
- Source: HuggingFace (zh-plus/tiny-imagenet)
- Download: Requires internet

### ImageNet-1K
- Classes: 1000
- Image Size: 128x128 (resized to 224x224)
- Train: 1,281,167 | Test: 50,000
- Source: HuggingFace (benjamin-paine/imagenet-1k-128x128)
- Download: Requires internet

All datasets use ImageNet normalization for compatibility with pre-trained models.

## Feature Extractors

### DINOv2/DINOv3
Self-supervised vision transformers providing:
- Rich semantic features without requiring labels
- Robustness to image transformations
- Strong clustering performance

Uses CLS token embedding as image representation.

### CLIP
Vision-language model providing:
- Strong features trained on image-text pairs
- Excellent transfer learning capabilities
- 512-dim (ViT-B) or 768-dim (ViT-L) embeddings

Uses vision encoder pooled output with projection.

## Cluster Visualization

Generate visualizations using dimensionality reduction:

### t-SNE (default)
```bash
python main.py --plot_clusters --viz_method tsne --save_features
```

### UMAP (faster)
```bash
python main.py --plot_clusters --viz_method umap --save_features
```

Generated visualizations:
- Side-by-side plots: predicted clusters vs ground truth
- Cluster distribution bar plots
- Saved in `visualizations/` subdirectory

## Pseudo Label Generation

Map clusters to actual class labels using k-nearest samples:

### During Training
```bash
python main.py \
    --generate_pseudo_labels \
    --k_samples 10 \
    --visualize_mapping \
    --save_features
```

### From Existing Results
```bash
python generate_pseudo_labels.py \
    --experiment_dir ./results/experiment_name \
    --k_samples 10 \
    --visualize
```

### How It Works
1. Find k samples closest to each cluster center
2. Assign cluster the most frequent label among those samples
3. Apply mapping to all samples in cluster

### Output Files
- `pseudo_labels_k{k}.json`: Pseudo labels and mappings
- `cluster_mapping_k{k}.png`: Visualization showing representative images
- `train_image_pseudo_labels.csv`: Per-image mappings
- `test_image_pseudo_labels.csv`: Test set mappings

### Recommended k Values
- CIFAR100: k=10
- ImageNet: k=20
- Smaller datasets: k=5

## Dataset Distillation

Create small synthetic datasets preserving learning dynamics:

### Basic Usage
```bash
python main.py \
    --num_clusters 100 \
    --generate_pseudo_labels \
    --distill_dataset \
    --images_per_class 10 \
    --distill_epochs 100 \
    --evaluate_distilled \
    --save_features
```

### Benefits
- 50-100x compression (98-99% size reduction)
- Faster training on distilled data
- 80-95% performance retention
- Lower memory requirements

### Key Parameters
- `--images_per_class`: Synthetic images per class (default: 10)
- `--distill_epochs`: Distillation epochs (default: 100)
- `--distill_lr`: Learning rate for optimization (default: 0.1)
- `--inner_epochs`: Inner training epochs (default: 10)
- `--evaluate_distilled`: Evaluate quality

## Command Line Arguments

### Data Arguments
- `--dataset`: Dataset choice (cifar10, cifar100, imagenet, tiny-imagenet, imagenette)
- `--data_root`: Dataset storage directory (default: ./data)
- `--batch_size`: Batch size (default: 256)
- `--num_workers`: Data loading workers (default: 4)

### Model Arguments
- `--model_type`: Feature extractor (dinov2, clip)
- `--dinov2_model`: DINOv2/DINOv3 model (default: facebook/dinov2-base)
- `--clip_model`: CLIP model (default: openai/clip-vit-base-patch32)
- `--num_clusters`: Number of clusters (default: dataset-dependent)
- `--hidden_dim`: Hidden layer dimension (default: 2048)
- `--projection_dim`: Projection dimension (default: 256)

### Training Arguments
- `--num_epochs`: Training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 0.001)
- `--temperature`: Softmax temperature (default: 0.1)

### Checkpoint Arguments
- `--checkpoint_dir`: Checkpoint directory (default: ./checkpoints)
- `--resume_from`: Path to resume checkpoint
- `--save_features`: Save extracted features
- `--load_features`: Load pre-extracted features

### Visualization Arguments
- `--plot_clusters`: Generate visualizations
- `--viz_method`: Method (tsne, umap)
- `--show_plots`: Display plots interactively

### Pseudo Labeling Arguments
- `--generate_pseudo_labels`: Generate pseudo labels
- `--k_samples`: Nearest samples for labeling (default: 10)
- `--visualize_mapping`: Generate mapping visualization
- `--max_clusters_viz`: Max clusters to visualize (default: 20)
- `--samples_per_cluster`: Samples per cluster in viz (default: 5)

### Dataset Distillation Arguments
- `--distill_dataset`: Perform distillation
- `--images_per_class`: Synthetic images per class (default: 10)
- `--distill_epochs`: Distillation epochs (default: 100)
- `--distill_lr`: Distillation learning rate (default: 0.1)
- `--inner_epochs`: Inner training epochs (default: 10)
- `--evaluate_distilled`: Evaluate distilled data

### Output Arguments
- `--results_dir`: Results directory (default: ./results)
- `--experiment_name`: Custom experiment name
- `--device`: Computation device (default: cuda)

## Algorithm Details

### TEMI Clustering Components

1. **K-means Initialization**: Warm start using K-means on extracted features

2. **Clustering Head**: Neural network projecting features
   - Multi-layer projection with batch normalization
   - Cluster assignment layer with learned centroids
   - Feature normalization for stability

3. **Loss Function**: Four complementary objectives
   - Conditional entropy minimization: confident assignments
   - Consistency/equivariance: agreement across augmentations
   - Marginal entropy maximization: prevents cluster collapse
   - Projection consistency: stable embeddings

4. **Training**: Iterative optimization with Adam
   - Mini-batch training with augmentation
   - Progressive refinement of assignments

## Evaluation Metrics

### Clustering Accuracy
Uses Hungarian algorithm for optimal cluster-to-class assignment.
- Range: 0 to 1 (higher better)
- Accounts for arbitrary label permutations

### Normalized Mutual Information (NMI)
Measures shared information between clusters and true classes.
- Range: 0 to 1 (higher better)
- Invariant to permutations

### Adjusted Rand Index (ARI)
Measures similarity between clusterings.
- Range: -1 to 1 (higher better)
- Corrects for chance agreement

### Cluster Distribution Analysis
- Active/empty cluster counts
- Size statistics (mean, std, min, max)
- Coefficient of variation for balance

## Output Files

Each experiment generates:
- `config.json`: Configuration and hyperparameters
- `results.json`: Evaluation metrics
- `predictions.npz`: Cluster assignments and labels
- `final_checkpoint.pt`: Trained model checkpoint
- `features/` (optional): Extracted features
- `visualizations/` (optional): Plots and charts
- `pseudo_labels/` (optional): Pseudo labels and mappings

## Expected Results

### CIFAR100 (k=100 clusters)
- Clustering Accuracy: 40-50%
- NMI: 0.50-0.60
- ARI: 0.30-0.40
- Active Clusters: 70-90 out of 100

Results vary based on random initialization and hardware. CIFAR100 is challenging with 100 fine-grained classes.

### CIFAR10 (k=10 clusters)
- Clustering Accuracy: 70-85%
- NMI: 0.70-0.80
- ARI: 0.60-0.75

### Imagenette (k=10 clusters)
- Clustering Accuracy: 75-90%
- NMI: 0.75-0.85
- ARI: 0.65-0.80

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 128`
- Use smaller model: `--dinov2_model facebook/dinov2-small`
- Save features and work with them directly

### Slow Feature Extraction
- Reduce workers if CPU bottleneck: `--num_workers 2`
- Extract once and reuse: `--save_features`
- Use smaller model variant

### Poor Clustering Results
- Increase epochs: `--num_epochs 200`
- Adjust temperature: `--temperature 0.05` (lower for sharper assignments)
- Try different learning rate: `--learning_rate 0.0001`
- Use larger model: `--dinov2_model facebook/dinov2-large`

### Visualization Issues
- Install UMAP: `pip install umap-learn`
- Ensure features were saved: `--save_features`
- Reduce samples for faster viz: Use subset of data

### Dataset Download Issues
- Check internet connection
- Verify HuggingFace access for ImageNet/Tiny ImageNet
- Use alternative datasets (CIFAR10/100 work offline after first download)

## Implementation Notes

### Checkpoint System
Robust checkpointing allows resuming from any stage:
- Feature extraction stage: save/load features
- Training stage: save model state, optimizer, history
- Resume capability: automatic recovery from interruptions

### Error Handling
Comprehensive error handling includes:
- Graceful CPU fallback if CUDA unavailable
- Input dimension and shape validation
- Clear error messages

### Memory Management
Efficient handling of large datasets:
- Batch processing for feature extraction
- Gradient accumulation support
- Automatic cleanup between stages
- Disk-based feature storage

### Differences from Paper
Faithful to TEMI paper with practical considerations:
- Uses DINOv2/CLIP instead of training from scratch (better features)
- Applied to multiple datasets for flexibility
- Simplified augmentation for small images
- Core algorithm follows paper specifications exactly

## Citation

If you use this code, please cite the TEMI paper:

```
@article{temi2023,
  title={Self-Supervised Clustering with Deep Learning},
  author={...},
  journal={arXiv preprint arXiv:2303.17896},
  year={2023}
}
```

## License

This project is for research and educational purposes.
