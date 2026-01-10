# CLUDI Deep Clustering Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Algorithm Overview](#algorithm-overview)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Usage Guide](#usage-guide)
6. [Configuration Options](#configuration-options)
7. [Hyperparameter Search](#hyperparameter-search)
8. [Checkpoints and Recovery](#checkpoints-and-recovery)
9. [Pseudo-Labeling Pipeline](#pseudo-labeling-pipeline)
10. [CSV Output Format](#csv-output-format)
11. [Dataset Compatibility](#dataset-compatibility)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)

---

## Introduction

CLUDI (Clustering via Diffusion) is a state-of-the-art self-supervised learning (SSL) deep clustering algorithm that leverages denoising diffusion models for learning cluster-friendly representations. Unlike traditional clustering methods, CLUDI uses a diffusion-based approach to iteratively refine cluster assignments through a teacher-student framework.

### Key Features

- **Diffusion-based clustering**: Uses denoising diffusion for robust cluster assignment learning
- **Self-conditioning**: Improves consistency through self-conditioning in the diffusion process
- **Teacher-student framework**: EMA-based teacher model for stable training
- **Multiple dataset support**: Compatible with CIFAR10, CIFAR100, ImageNet-1K, Tiny-ImageNet, and Imagenette
- **Feature extractor agnostic**: Works with DINOv2, DINOv3, and CLIP features
- **Comprehensive checkpointing**: Saves progress at regular intervals to prevent data loss
- **Pseudo-label generation**: Maps clusters to actual labels for downstream tasks
- **CSV export**: Exports pseudo labels and cluster mappings in CSV format
- **Hyperparameter search**: Grid, random, and Bayesian optimization for finding optimal hyperparameters

---

## Algorithm Overview

### How CLUDI Works

CLUDI operates in three main phases:

1. **Feature Extraction**: Extract visual features from images using pre-trained vision models (DINOv2, DINOv3, or CLIP)

2. **Diffusion-Based Clustering**: Train a neural network that learns to map noisy cluster embeddings back to clean cluster assignments through denoising

3. **Pseudo-Label Generation**: Map learned clusters to actual class labels using k-nearest neighbors to cluster centers

### Mathematical Foundation

The CLUDI algorithm uses a diffusion process defined by:

```
q(z_t | z_0) = N(z_t; √(α_t) * z_0, (1 - α_t) * I)
```

Where:
- `z_0` is the clean cluster embedding
- `z_t` is the noisy embedding at timestep t
- `α_t` is the cumulative product of (1 - β) where β is the noise schedule

The model learns to predict either:
- The clean embedding `z_0` directly (x0-prediction)
- The velocity `v = √(α_t) * ε - √(1 - α_t) * z_0` (v-prediction, recommended)

### Training Objective

CLUDI optimizes a combination of:

1. **Diffusion Loss**: MSE between predicted and target embeddings
2. **Cross-Entropy Loss**: Ensures consistent cluster assignments between teacher and student

```
L_total = L_diff + λ * L_ce
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLUDI CLUSTERING PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐     ┌──────────────────┐     ┌─────────────┐ │
│  │  Dataset      │────▶│  Feature         │────▶│  CLUDI      │ │
│  │  (CIFAR100)   │     │  Extractor       │     │  Training   │ │
│  └───────────────┘     │  (DINOv2/CLIP)   │     └──────┬──────┘ │
│                        └──────────────────┘            │        │
│                                                        ▼        │
│  ┌───────────────┐     ┌──────────────────┐     ┌─────────────┐ │
│  │  CSV Export   │◀────│  Pseudo-Label    │◀────│  Cluster    │ │
│  │  (Results)    │     │  Generation      │     │  Evaluation │ │
│  └───────────────┘     └──────────────────┘     └─────────────┘ │
│                                                                   │
│  CHECKPOINTS SAVED AT: Feature Extraction, Training (every 20    │
│  epochs), Final Model, Pseudo Labels                             │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Data Loading** (`src/data_loader.py`)
   - Loads and preprocesses images
   - Applies appropriate transformations for feature extractors
   - Supports multiple dataset formats

2. **Feature Extraction** (`src/feature_extractor.py`, `src/clip_feature_extractor.py`)
   - Extracts visual features using DINOv2/DINOv3 or CLIP
   - Features are L2-normalized for consistency

3. **CLUDI Model** (`src/cludi_clustering.py`)
   - Implements the diffusion-based clustering model
   - Includes teacher-student framework with EMA updates
   - Provides training loop with checkpointing

4. **Pseudo-Labeling** (`src/pseudo_labeling.py`)
   - Maps cluster IDs to actual class labels
   - Uses k-nearest neighbors to cluster centers
   - Generates confidence scores

5. **Evaluation** (`src/evaluation.py`)
   - Computes clustering accuracy, NMI, and ARI
   - Analyzes cluster distribution

---

## Installation and Setup

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
pillow>=10.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

### Verify Installation

```bash
# Run a quick test
python -c "from src.cludi_clustering import CLUDIClusterer; print('CLUDI installed successfully!')"
```

---

## Usage Guide

### Basic Usage

Run CLUDI clustering with default settings:

```bash
python main.py --clustering_method cludi --dataset cifar100
```

### Complete Example

Run the full pipeline with pseudo-label generation:

```bash
python main.py \
    --clustering_method cludi \
    --dataset cifar100 \
    --num_clusters 100 \
    --num_epochs 100 \
    --generate_pseudo_labels \
    --k_samples 10 \
    --visualize_mapping \
    --save_features
```

### Using Different Datasets

```bash
# CIFAR-10 (10 classes)
python main.py --clustering_method cludi --dataset cifar10 --num_clusters 10

# CIFAR-100 (100 classes)
python main.py --clustering_method cludi --dataset cifar100 --num_clusters 100

# Imagenette (10 classes, good for quick testing)
python main.py --clustering_method cludi --dataset imagenette --num_clusters 10

# Tiny ImageNet (200 classes)
python main.py --clustering_method cludi --dataset tiny-imagenet --num_clusters 200

# ImageNet-1K (1000 classes)
python main.py --clustering_method cludi --dataset imagenet-1k --num_clusters 1000
```

### Using Different Feature Extractors

```bash
# DINOv2 (default)
python main.py --clustering_method cludi --model_type dinov2 --dinov2_model facebook/dinov2-base

# CLIP
python main.py --clustering_method cludi --model_type clip --clip_model openai/clip-vit-base-patch32

# DINOv2-Large (better quality, slower)
python main.py --clustering_method cludi --model_type dinov2 --dinov2_model facebook/dinov2-large
```

### Advanced Configuration

```bash
python main.py \
    --clustering_method cludi \
    --dataset cifar100 \
    --num_clusters 100 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --batch_size 256 \
    --cludi_embedding_dim 64 \
    --cludi_diffusion_steps 1000 \
    --cludi_batch_diffusion 8 \
    --cludi_rescaling_factor 49.0 \
    --cludi_ce_lambda 50.0 \
    --cludi_use_v_prediction \
    --cludi_warmup_epochs 1 \
    --generate_pseudo_labels \
    --k_samples 10 \
    --visualize_mapping \
    --save_features \
    --device cuda
```

---

## Configuration Options

### General Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--clustering_method` | `temi` | Clustering algorithm (`temi` or `cludi`) |
| `--dataset` | `cifar100` | Dataset to use |
| `--num_clusters` | Auto | Number of clusters (defaults to dataset classes) |
| `--num_epochs` | `100` | Training epochs |
| `--learning_rate` | `0.005` | Learning rate |
| `--batch_size` | `256` | Batch size |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

### CLUDI-Specific Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--cludi_embedding_dim` | `64` | Dimension of cluster embeddings |
| `--cludi_diffusion_steps` | `1000` | Number of diffusion timesteps |
| `--cludi_batch_diffusion` | `8` | Batch size for diffusion sampling |
| `--cludi_rescaling_factor` | `49.0` | Noise rescaling factor |
| `--cludi_ce_lambda` | `50.0` | Cross-entropy loss weight |
| `--cludi_use_v_prediction` | `True` | Use v-parameterization |
| `--cludi_warmup_epochs` | `1` | Warmup epochs for learning rate |

### Feature Extractor Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `dinov2` | Feature extractor (`dinov2` or `clip`) |
| `--dinov2_model` | `facebook/dinov2-base` | DINOv2/v3 model variant |
| `--clip_model` | `openai/clip-vit-base-patch32` | CLIP model variant |
| `--save_features` | `False` | Save extracted features to disk |
| `--load_features` | `None` | Path to pre-extracted features |

### Pseudo-Labeling Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--generate_pseudo_labels` | `False` | Enable pseudo-label generation |
| `--k_samples` | `10` | K-nearest samples for label voting |
| `--visualize_mapping` | `False` | Generate cluster-to-label visualization |
| `--max_clusters_viz` | `20` | Max clusters to visualize |
| `--samples_per_cluster` | `5` | Samples per cluster in visualization |

---

## Hyperparameter Search

CLUDI provides comprehensive hyperparameter search capabilities with three different methods: Grid Search, Random Search, and Bayesian Optimization.

### Why Use Hyperparameter Search?

Finding optimal hyperparameters can significantly improve clustering performance. Key hyperparameters that affect CLUDI performance include:

- **embedding_dim**: Dimension of cluster embeddings (larger may capture more complex patterns)
- **learning_rate**: Learning rate for optimizer (critical for convergence)
- **diffusion_steps**: Number of diffusion timesteps (more steps = finer denoising)
- **ce_lambda**: Cross-entropy loss weight (balances diffusion and assignment losses)
- **rescaling_factor**: Noise rescaling factor (affects training dynamics)

### Quick Start

```bash
# Run random search with 20 trials
python main.py \
    --clustering_method cludi \
    --dataset cifar100 \
    --hyperparam_search \
    --search_method random \
    --search_trials 20 \
    --search_epochs 50 \
    --search_metric accuracy
```

### Search Methods

#### Grid Search
Exhaustive search over all combinations of specified parameter values.

```bash
python main.py \
    --clustering_method cludi \
    --hyperparam_search \
    --search_method grid \
    --search_epochs 30
```

**Best for**: Small search spaces, when you want to test all combinations.

#### Random Search
Randomly samples configurations from the parameter space.

```bash
python main.py \
    --clustering_method cludi \
    --hyperparam_search \
    --search_method random \
    --search_trials 30 \
    --search_seed 42
```

**Best for**: Large search spaces, continuous hyperparameters, limited compute budget.

#### Bayesian Optimization
Uses sequential model-based optimization (via Optuna) to efficiently explore the space.

```bash
python main.py \
    --clustering_method cludi \
    --hyperparam_search \
    --search_method bayesian \
    --search_trials 50
```

**Best for**: When you want to find good hyperparameters with fewer trials, expensive evaluations.

**Note**: Requires Optuna: `pip install optuna`

### Search Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--hyperparam_search` | `False` | Enable hyperparameter search |
| `--search_method` | `random` | Search method (`grid`, `random`, `bayesian`) |
| `--search_trials` | `20` | Number of trials for random/bayesian search |
| `--search_epochs` | `50` | Training epochs per trial |
| `--search_metric` | `accuracy` | Metric to optimize (`accuracy`, `nmi`, `ari`) |
| `--search_seed` | `None` | Random seed for reproducibility |

### Default Search Space

The default search space covers the following hyperparameters:

| Hyperparameter | Search Values |
|----------------|---------------|
| `embedding_dim` | [32, 64, 128] |
| `learning_rate` | 1e-5 to 1e-3 (log scale) |
| `diffusion_steps` | [500, 750, 1000] |
| `batch_diffusion` | [4, 8, 16] |
| `rescaling_factor` | [25.0, 49.0, 100.0] |
| `ce_lambda` | [25.0, 50.0, 75.0, 100.0] |
| `warmup_epochs` | [0, 1, 2] |

### Search Results

After search completes, results are saved in:

```
results/
└── cludi_{dataset}_{clusters}clusters_{timestamp}/
    ├── best_hyperparameters.json      # Best configuration found
    └── hyperparam_search/
        ├── {method}_search_final_{timestamp}.json  # All trial results
        └── optuna_study.json           # (Bayesian only) Optuna study info
```

### Programmatic Usage

```python
from src.cludi_hyperparam_search import (
    CLUDIHyperparameterSearch,
    CLUDIHyperparameterSpace
)

# Define custom search space
search_space = CLUDIHyperparameterSpace(
    embedding_dim=[64, 128, 256],
    learning_rate=(1e-5, 5e-4),  # Tuple = continuous range
    ce_lambda=[30.0, 50.0, 70.0]
)

# Create searcher
searcher = CLUDIHyperparameterSearch(
    feature_dim=768,
    num_clusters=100,
    device="cuda",
    metric="accuracy"
)

# Run search
best_params, results = searcher.search(
    features=train_features,
    labels=train_labels,
    search_space=search_space,
    method="random",
    n_trials=30,
    num_epochs=50
)

# Print summary
searcher.print_summary()
```

---

## Checkpoints and Recovery

### Automatic Checkpointing

CLUDI automatically saves checkpoints at multiple stages:

1. **Feature Extraction**: Saved when `--save_features` is enabled
2. **Training Checkpoints**: Saved every 20 epochs during training
3. **Final Model**: Saved at the end of training
4. **Pseudo Labels**: Saved after pseudo-label generation

### Checkpoint Locations

```
results/
└── cludi_{dataset}_{clusters}clusters_{timestamp}/
    ├── config.json                    # Experiment configuration
    ├── final_checkpoint.pt            # Final trained model
    ├── checkpoints/
    │   ├── checkpoint_epoch20.pt      # Training checkpoint
    │   ├── checkpoint_epoch40.pt
    │   └── ...
    ├── features/
    │   ├── train_features.pt          # Saved features (if enabled)
    │   └── test_features.pt
    ├── pseudo_labels/
    │   ├── pseudo_labels_k10.json     # Pseudo labels (JSON)
    │   ├── train_pseudo_labels_k10.csv  # Training labels (CSV)
    │   ├── test_pseudo_labels_k10.csv   # Test labels (CSV)
    │   ├── cluster_mapping_k10.csv      # Cluster mappings (CSV)
    │   └── cluster_mapping_k10.png      # Visualization
    └── results.json                   # Evaluation results
```

### Resuming from Checkpoint

```bash
# Resume training from a checkpoint
python main.py \
    --clustering_method cludi \
    --resume_from ./results/cludi_cifar100_100clusters_20240101_120000/checkpoints/checkpoint_epoch40.pt
```

### Using Pre-extracted Features

```bash
# First run: Extract and save features
python main.py --clustering_method cludi --save_features

# Subsequent runs: Load saved features (saves time)
python main.py \
    --clustering_method cludi \
    --load_features ./results/experiment_name/features
```

---

## Pseudo-Labeling Pipeline

### How Pseudo-Labeling Works

1. **Compute Cluster Assignments**: Run inference on trained CLUDI model
2. **Find K-Nearest Samples**: For each cluster, find K samples closest to the cluster center
3. **Majority Vote**: Assign cluster the most frequent true label among K-nearest samples
4. **Apply Mapping**: Map all samples to pseudo labels based on their cluster assignment
5. **Compute Confidence**: Calculate confidence based on cluster purity and distance to center

### Example Usage

```python
from src.pseudo_labeling import generate_pseudo_labels, save_pseudo_labels_to_csv

# Generate pseudo labels
pseudo_labels, cluster_to_label, k_nearest, confidence, cluster_conf = generate_pseudo_labels(
    features=train_features,
    cluster_assignments=cluster_predictions,
    true_labels=train_labels,
    cluster_centers=clusterer.cluster_centers,
    k=10,
    verbose=True
)

# Save to CSV
save_pseudo_labels_to_csv(
    pseudo_labels=pseudo_labels,
    cluster_assignments=cluster_predictions,
    true_labels=train_labels,
    confidence_scores=confidence,
    output_path="pseudo_labels.csv",
    class_names=class_names
)
```

### Choosing K Value

| Dataset | Recommended K | Notes |
|---------|--------------|-------|
| CIFAR-10 | 5-10 | Small dataset, fewer clusters |
| CIFAR-100 | 10-20 | Fine-grained classes |
| ImageNet | 20-50 | Large dataset, more variance |
| Tiny ImageNet | 10-20 | Similar to CIFAR-100 |

---

## CSV Output Format

### Training/Test Pseudo Labels CSV

`train_pseudo_labels_k10.csv` / `test_pseudo_labels_k10.csv`:

| Column | Description |
|--------|-------------|
| `sample_idx` | Sample index in dataset |
| `cluster_id` | Assigned cluster ID |
| `pseudo_label` | Mapped pseudo label |
| `true_label` | Ground truth label |
| `confidence_score` | Confidence (0-1) |
| `pseudo_label_name` | Class name for pseudo label |
| `true_label_name` | Class name for true label |
| `is_correct` | 1 if pseudo == true, 0 otherwise |

Example:
```csv
sample_idx,cluster_id,pseudo_label,true_label,confidence_score,pseudo_label_name,true_label_name,is_correct
0,42,5,5,0.8234,dog,dog,1
1,42,5,8,0.8234,dog,cat,0
2,15,3,3,0.9123,car,car,1
```

### Cluster Mapping CSV

`cluster_mapping_k10.csv`:

| Column | Description |
|--------|-------------|
| `cluster_id` | Cluster ID |
| `pseudo_label` | Mapped class label |
| `cluster_size` | Number of samples in cluster |
| `cluster_accuracy` | Accuracy within cluster |
| `confidence` | Cluster confidence score |
| `pseudo_label_name` | Class name |

Example:
```csv
cluster_id,pseudo_label,cluster_size,cluster_accuracy,confidence,pseudo_label_name
0,15,523,0.8912,0.9234,bicycle
1,42,412,0.7834,0.8123,airplane
```

---

## Dataset Compatibility

### Supported Datasets

| Dataset | Classes | Train Size | Test Size | Notes |
|---------|---------|------------|-----------|-------|
| CIFAR-10 | 10 | 50,000 | 10,000 | Quick testing |
| CIFAR-100 | 100 | 50,000 | 10,000 | Standard benchmark |
| Imagenette | 10 | 9,469 | 3,925 | Fast experiments |
| Tiny ImageNet | 200 | 100,000 | 10,000 | Medium scale |
| ImageNet-1K | 1000 | 1.2M | 50,000 | Large scale |

### Adding New Datasets

To add a new dataset:

1. Add dataset class to `src/data_loader.py`
2. Add transform function for preprocessing
3. Update `create_data_loaders()` to include new dataset
4. Add dataset to CLI choices in `main.py`

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
python main.py --clustering_method cludi --batch_size 128

# Solution 2: Reduce diffusion batch
python main.py --clustering_method cludi --cludi_batch_diffusion 4

# Solution 3: Use smaller embedding dimension
python main.py --clustering_method cludi --cludi_embedding_dim 32
```

#### Slow Training

```bash
# Solution 1: Reduce diffusion steps
python main.py --clustering_method cludi --cludi_diffusion_steps 500

# Solution 2: Use pre-extracted features
python main.py --save_features  # First run
python main.py --load_features ./results/exp/features  # Subsequent runs
```

#### Poor Clustering Results

```bash
# Solution 1: Increase training epochs
python main.py --clustering_method cludi --num_epochs 200

# Solution 2: Adjust learning rate
python main.py --clustering_method cludi --learning_rate 0.00005

# Solution 3: Increase embedding dimension
python main.py --clustering_method cludi --cludi_embedding_dim 128

# Solution 4: Use larger feature extractor
python main.py --clustering_method cludi --dinov2_model facebook/dinov2-large
```

#### Checkpoint Loading Errors

```bash
# Ensure checkpoint was saved with same configuration
python main.py --clustering_method cludi \
    --num_clusters 100 \
    --cludi_embedding_dim 64 \
    --resume_from ./checkpoint.pt
```

---

## API Reference

### CLUDIClusterer

Main class for CLUDI clustering.

```python
from src.cludi_clustering import CLUDIClusterer

clusterer = CLUDIClusterer(
    feature_dim=768,           # Input feature dimension
    num_clusters=100,          # Number of clusters
    device="cuda",             # Device
    embedding_dim=64,          # Cluster embedding dimension
    learning_rate=0.0001,      # Learning rate
    diffusion_steps=1000,      # Diffusion timesteps
    batch_diffusion=8,         # Diffusion batch size
    rescaling_factor=49.0,     # Noise rescaling
    ce_lambda=50.0,            # CE loss weight
    use_v_prediction=True,     # Use v-parameterization
    warmup_epochs=1            # Warmup epochs
)

# Train
history = clusterer.fit(
    features=train_features,
    num_epochs=100,
    batch_size=256,
    verbose=True,
    save_checkpoints=True,
    checkpoint_dir="./checkpoints",
    checkpoint_freq=20
)

# Predict
predictions = clusterer.predict(features, batch_size=256)

# Save/Load checkpoints
clusterer.save_checkpoint("checkpoint.pt", epoch=100, history=history)
epoch, history = clusterer.load_checkpoint("checkpoint.pt")
```

### generate_pseudo_labels

Generate pseudo labels from cluster assignments.

```python
from src.pseudo_labeling import generate_pseudo_labels

pseudo_labels, cluster_to_label, k_nearest, confidence, cluster_conf = generate_pseudo_labels(
    features=features,           # Feature tensor
    cluster_assignments=preds,   # Cluster assignments
    true_labels=labels,          # Ground truth labels
    cluster_centers=centers,     # Cluster centers
    k=10,                        # K-nearest samples
    verbose=True,                # Print progress
    return_confidence=True       # Return confidence scores
)
```

### save_pseudo_labels_to_csv

Save pseudo labels to CSV file.

```python
from src.pseudo_labeling import save_pseudo_labels_to_csv

save_pseudo_labels_to_csv(
    pseudo_labels=pseudo_labels,
    cluster_assignments=predictions,
    true_labels=true_labels,
    confidence_scores=confidence,
    output_path="output.csv",
    class_names=["cat", "dog", ...]  # Optional
)
```

---

## Citation

If you use CLUDI in your research, please cite:

```bibtex
@article{cludi2024,
  title={CLUDI: Clustering via Diffusion for Deep Image Clustering},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

This project is for research and educational purposes. See LICENSE file for details.
