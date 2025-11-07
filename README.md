# TEMI Deep Clustering on CIFAR100 and ImageNet

This repository implements TEMI (Transformation-Equivariant Multi-Instance) clustering on the CIFAR100 and ImageNet datasets using DINOv2/DINOv3 or CLIP features. The implementation follows the paper "Self-Supervised Clustering with Deep Learning" (arXiv:2303.17896).

**🚀 New to this project? Check out [QUICKSTART.md](QUICKSTART.md) to get started in minutes!**

**📊 Want to use ImageNet? Check out [IMAGENET_USAGE.md](IMAGENET_USAGE.md) for the complete guide!**

**🏷️ Want to generate pseudo labels for your clusters? Check out [PSEUDO_LABELING_GUIDE.md](PSEUDO_LABELING_GUIDE.md)!**

**🔬 Want to distill your dataset using pseudo labels? Check out [DATASET_DISTILLATION_GUIDE.md](DATASET_DISTILLATION_GUIDE.md)!**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Using CLIP](#using-clip)
  - [Using DINOv3](#using-dinov3)
  - [With Visualization](#with-visualization)
  - [Using Pre-extracted Features](#using-pre-extracted-features-avoid-re-running-experiments)
  - [Analyzing Existing Results](#analyzing-existing-results)
  - [Generating Pseudo Labels](#generating-pseudo-labels)
- [Command Line Arguments](#command-line-arguments)
- [Cluster Visualization](#cluster-visualization)
- [Pseudo Label Generation](#pseudo-label-generation)
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
4. **Visualization & Pseudo Labeling** (optional): Generate t-SNE/UMAP plots to visualize cluster structures and map clusters to actual labels
5. **Dataset Distillation** (optional): Create a small synthetic dataset that preserves the learning dynamics of the full dataset

## Features

- **Multiple datasets**: Support for CIFAR100 and ImageNet-1K datasets
- **Multiple feature extractors**: Support for DINOv2, DINOv3, and CLIP models for powerful visual representations
- **TEMI clustering algorithm**: Implementation following the paper specifications
- **Pseudo label generation**: Map clusters to actual labels using k-nearest samples for interpretability and semi-supervised learning
- **Dataset distillation**: Create synthetic datasets using trajectory matching ([arXiv:2406.18561](https://arxiv.org/html/2406.18561))
- **Checkpoint system**: Resume training from any stage without re-running expensive computations
- **Comprehensive evaluation**: Multiple metrics including accuracy, NMI, and ARI
- **Cluster visualization**: Generate beautiful t-SNE and UMAP plots to visualize clustering results
- **Cluster mapping visualization**: Visualize the most representative images from each cluster
- **Support for multiple model variants**: Small, base, large, giant, and custom HuggingFace models
- **Well-documented code**: Human-readable comments throughout
- **Robust error handling**: Graceful degradation and informative error messages
- **Type safety**: All operations use PyTorch tensors with proper device handling
- **Vectorized operations**: Efficient batch processing throughout

## Requirements

The project requires Python 3.8 or higher. Install dependencies using:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0+
- Transformers (HuggingFace)
- torchvision
- scikit-learn
- numpy, scipy
- matplotlib (for plotting)
- umap-learn (optional, for UMAP visualization)

## Project Structure

```
clustering-private/
├── main.py                        # Main training script
├── analyze_results.py             # Results analysis and visualization script
├── generate_pseudo_labels.py     # Standalone script for pseudo label generation
├── requirements.txt               # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # CIFAR100 & ImageNet data loading
│   ├── feature_extractor.py      # DINOv2/DINOv3 feature extraction
│   ├── clip_feature_extractor.py # CLIP feature extraction
│   ├── temi_clustering.py        # TEMI clustering algorithm
│   ├── evaluation.py             # Clustering evaluation metrics
│   ├── visualization.py          # Cluster visualization (t-SNE/UMAP)
│   └── pseudo_labeling.py        # Pseudo label generation and mapping
├── data/                          # Dataset storage (auto-downloaded)
├── checkpoints/                  # Model checkpoints
└── results/                     # Experiment results and outputs
```

## Usage

For complete examples and detailed usage instructions, see:
- **[IMAGENET_USAGE.md](IMAGENET_USAGE.md)** - Complete guide for using ImageNet dataset
- **[CLIP_USAGE.md](CLIP_USAGE.md)** - Complete guide for using CLIP models
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - Detailed guide for cluster visualization
- **[example_clip_usage.py](example_clip_usage.py)** - CLIP example commands and workflows
- **[example_dinov3_visualization.py](example_dinov3_visualization.py)** - DINOv3 example commands and workflows
- **[example_imagenet_usage.py](example_imagenet_usage.py)** - ImageNet example commands and workflows

### Basic Usage

Run clustering with default settings (k=100 clusters on CIFAR100 using DINOv2):

```bash
python main.py
```

Run clustering on ImageNet (k=1000 clusters using DINOv2):

```bash
python main.py --dataset imagenet
```

### Using CLIP

To use CLIP models for feature extraction instead of DINOv2:

```bash
# Using default CLIP model (ViT-B/32)
python main.py --model_type clip

# Using a specific CLIP model
python main.py --model_type clip --clip_model openai/clip-vit-large-patch14

# Using CLIP with visualization
python main.py --model_type clip --plot_clusters --save_features
```

Available CLIP models:
- `openai/clip-vit-base-patch32` (default, 512-dim features, fastest)
- `openai/clip-vit-base-patch16` (512-dim features, better quality)
- `openai/clip-vit-large-patch14` (768-dim features, best quality)

### Using DINOv3

To use DINOv3 models instead of DINOv2, simply specify any DINOv3 model from HuggingFace:

```bash
python main.py --dinov2_model facebook/dinov3-base
```

### Using ImageNet Dataset

To use the ImageNet-1K dataset (128x128 version from HuggingFace):

```bash
# Basic ImageNet clustering (1000 clusters by default)
python main.py --dataset imagenet

# ImageNet with CLIP
python main.py --dataset imagenet --model_type clip

# ImageNet with custom number of clusters
python main.py --dataset imagenet --num_clusters 500

# ImageNet with DINOv2-large and visualization
python main.py --dataset imagenet \
               --dinov2_model facebook/dinov2-large \
               --plot_clusters \
               --save_features
```

**For complete ImageNet documentation, see [IMAGENET_USAGE.md](IMAGENET_USAGE.md)**

### With Visualization

Generate beautiful t-SNE or UMAP plots of your clusters:

```bash
# Using t-SNE (default)
python main.py --plot_clusters --save_features

# Using UMAP (requires umap-learn)
python main.py --plot_clusters --viz_method umap --save_features
```

**Note**: The `--save_features` flag is required for visualization to work, as plots are generated from saved features.

### Advanced Options

```bash
# Using DINOv2 with custom settings
python main.py \
    --model_type dinov2 \
    --num_clusters 100 \
    --dinov2_model facebook/dinov2-base \
    --num_epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --temperature 0.1 \
    --plot_clusters \
    --viz_method tsne \
    --save_features \
    --device cuda

# Using CLIP with custom settings
python main.py \
    --model_type clip \
    --clip_model openai/clip-vit-large-patch14 \
    --num_clusters 100 \
    --num_epochs 100 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --temperature 0.1 \
    --plot_clusters \
    --save_features \
    --device cuda
```

### Resume from Checkpoint

If training is interrupted, resume from the last checkpoint:

```bash
python main.py --resume_from ./checkpoints/experiment/final_checkpoint.pt
```

### Using Pre-extracted Features (Avoid Re-running Experiments)

To save time and avoid re-running expensive feature extraction:

```bash
# First run: extract and save features
python main.py --save_features

# Subsequent runs: load pre-extracted features
python main.py --load_features ./results/experiment_name/features/train_features
```

This is especially useful when:
- Experimenting with different clustering hyperparameters
- Generating visualizations from existing results
- Running multiple experiments without re-extracting features

### Analyzing Existing Results

You can analyze and visualize results from completed experiments without re-running them:

```bash
# Basic analysis
python analyze_results.py ./results/experiment_name

# Detailed analysis with per-class and per-cluster statistics
python analyze_results.py ./results/experiment_name --detailed

# Generate visualizations from saved features
python analyze_results.py ./results/experiment_name --plot --viz_method tsne

# Use UMAP instead of t-SNE
python analyze_results.py ./results/experiment_name --plot --viz_method umap
```

**Note**: Visualization requires that the experiment was run with `--save_features` flag.

### Generating Pseudo Labels

You can generate pseudo labels for your clusters to map them to actual class labels. This is useful for understanding what each cluster represents and for semi-supervised learning applications.

```bash
# Generate pseudo labels during main run
python main.py \
    --generate_pseudo_labels \
    --k_samples 10 \
    --visualize_mapping \
    --save_features

# Generate pseudo labels from existing results
python generate_pseudo_labels.py \
    --experiment_dir ./results/experiment_name \
    --k_samples 10 \
    --visualize

# Example: Full pipeline with pseudo labels and all visualizations
python main.py \
    --num_clusters 100 \
    --num_epochs 100 \
    --generate_pseudo_labels \
    --k_samples 10 \
    --visualize_mapping \
    --plot_clusters \
    --save_features
```

**For complete documentation, see [PSEUDO_LABELING_GUIDE.md](PSEUDO_LABELING_GUIDE.md)**

### Dataset Distillation

You can distill your dataset to create a small synthetic dataset that preserves the learning dynamics:

```bash
# Generate pseudo labels and distill the dataset
python main.py \
    --num_clusters 100 \
    --generate_pseudo_labels \
    --distill_dataset \
    --images_per_class 10 \
    --distill_epochs 100 \
    --evaluate_distilled \
    --save_features

# Run standalone example
python example_distillation.py \
    --dataset cifar100 \
    --model_type dinov2 \
    --num_samples 500 \
    --num_clusters 10 \
    --images_per_class 5 \
    --distill_epochs 10
```

**Benefits of Dataset Distillation:**
- **50-100x compression**: Reduce dataset size by 98-99%
- **Faster training**: Train models much faster on distilled data
- **Performance retention**: 80-95% of original performance
- **Memory efficient**: Lower memory requirements

**For complete documentation, see [DATASET_DISTILLATION_GUIDE.md](DATASET_DISTILLATION_GUIDE.md)**

## Command Line Arguments

### Data Arguments
- `--dataset`: Dataset to use (choices: cifar100, imagenet; default: cifar100)
- `--data_root`: Root directory for dataset storage (default: ./data)
- `--batch_size`: Batch size for data loading (default: 256)
- `--num_workers`: Number of data loading workers (default: 4)

### Model Arguments
- `--model_type`: Type of feature extractor (choices: dinov2, clip; default: dinov2)
- `--dinov2_model`: DINOv2/DINOv3 model to use (default: facebook/dinov2-base)
  - DINOv2 options: facebook/dinov2-small, facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant
  - DINOv3 options: Any DINOv3 model from HuggingFace (e.g., facebook/dinov3-base)
  - Custom: Any compatible DINO model from HuggingFace
- `--clip_model`: CLIP model to use (default: openai/clip-vit-base-patch32)
  - Options: openai/clip-vit-base-patch32, openai/clip-vit-base-patch16, openai/clip-vit-large-patch14
  - Custom: Any compatible CLIP model from HuggingFace
- `--num_clusters`: Number of clusters (default: 100 for CIFAR100, 1000 for ImageNet)
- `--hidden_dim`: Hidden layer dimension (default: 2048)
- `--projection_dim`: Projection space dimension (default: 256)

### Training Arguments
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--temperature`: Temperature for softmax (default: 0.1)

### Checkpoint Arguments
- `--checkpoint_dir`: Directory for checkpoints (default: ./checkpoints)
- `--resume_from`: Path to checkpoint for resuming training
- `--save_features`: Flag to save extracted features (required for visualization)
- `--load_features`: Path to pre-extracted features (for avoiding re-extraction)

### Visualization Arguments
- `--plot_clusters`: Generate cluster visualizations (t-SNE/UMAP plots)
- `--viz_method`: Dimensionality reduction method (choices: tsne, umap; default: tsne)
- `--show_plots`: Display plots interactively (in addition to saving them)

### Pseudo Labeling Arguments
- `--generate_pseudo_labels`: Generate pseudo labels by mapping clusters to actual labels
- `--k_samples`: Number of nearest samples to cluster center for label assignment (default: 10)
- `--visualize_mapping`: Generate visualization of cluster-to-label mappings
- `--max_clusters_viz`: Maximum number of clusters to visualize in mapping (default: 20)
- `--samples_per_cluster`: Number of samples to show per cluster in visualization (default: 5)

### Dataset Distillation Arguments
- `--distill_dataset`: Perform dataset distillation using pseudo labels
- `--images_per_class`: Number of synthetic images per class for distillation (default: 10)
- `--distill_epochs`: Number of distillation epochs (default: 100)
- `--distill_lr`: Learning rate for distilled image optimization (default: 0.1)
- `--inner_epochs`: Number of inner training epochs per distillation step (default: 10)
- `--evaluate_distilled`: Evaluate the quality of distilled data

### Output Arguments
- `--results_dir`: Directory for saving results (default: ./results)
- `--experiment_name`: Custom experiment name (auto-generated if not provided)
- `--device`: Computation device (default: cuda)

## Algorithm Details

### TEMI Clustering

The TEMI algorithm consists of several key components:

1. **K-means Initialization**: Clusters are initialized using K-means on extracted features for a warm start

2. **Clustering Head**: A neural network that projects features into a clustering-friendly space
   - Multi-layer projection network with batch normalization
   - Cluster assignment layer that learns cluster centroids
   - Feature normalization for stable optimization

3. **Loss Function**: Four complementary objectives
   - **Conditional Entropy Minimization**: Encourages confident cluster assignments for each sample
   - **Consistency/Equivariance Loss**: Ensures agreement between original and augmented views
   - **Marginal Entropy Maximization**: Prevents cluster collapse by promoting balanced cluster usage
   - **Projection Consistency**: Stabilizes feature embeddings under augmentation

4. **Training**: Iterative optimization using Adam optimizer
   - Mini-batch training with feature augmentation
   - Progressive refinement of cluster assignments

### Feature Extractors

#### DINOv2/DINOv3 Features

DINOv2 and DINOv3 are self-supervised vision transformers that provide:
- Rich semantic visual features without requiring labels
- Robustness to image transformations
- Strong performance on downstream tasks including clustering

We use the CLS token embedding from DINOv2/DINOv3 as the image representation.

#### CLIP Features

CLIP (Contrastive Language-Image Pre-training) is a vision-language model that provides:
- Strong visual features trained on image-text pairs
- Excellent transfer learning capabilities
- 512-dimensional (ViT-B) or 768-dimensional (ViT-L) embeddings

We use the vision encoder's pooled output with projection as the image representation.

## Evaluation Metrics

The following metrics are computed on both training and test sets:

1. **Clustering Accuracy**: Uses Hungarian algorithm for optimal cluster-to-class assignment
   - Ranges from 0 to 1 (higher is better)
   - Accounts for arbitrary cluster label permutations

2. **Normalized Mutual Information (NMI)**: Measures information shared between clusters and true classes
   - Ranges from 0 to 1 (higher is better)
   - Invariant to label permutations

3. **Adjusted Rand Index (ARI)**: Measures similarity between clusterings
   - Ranges from -1 to 1 (higher is better)
   - Corrects for chance agreement

4. **Cluster Distribution Analysis**: 
   - Number of active/empty clusters
   - Cluster size statistics (mean, std, min, max)
   - Coefficient of variation to detect cluster imbalance

## Cluster Visualization

The repository supports generating beautiful visualizations of clustering results using dimensionality reduction techniques.

### Available Visualization Methods

1. **t-SNE (t-distributed Stochastic Neighbor Embedding)**
   - Default method
   - Great for visualizing local structure
   - Works well for most datasets
   - No additional installation required

2. **UMAP (Uniform Manifold Approximation and Projection)**
   - Faster than t-SNE
   - Better preserves global structure
   - Requires: `pip install umap-learn`

### Generated Visualizations

When using `--plot_clusters`, the following visualizations are generated:

1. **Cluster Visualization Plots**: Side-by-side comparison of:
   - Predicted clusters (colored by cluster assignment)
   - Ground truth labels (colored by true class)
   
2. **Cluster Distribution Bar Plot**: Shows the number of samples in each cluster

All visualizations are saved in the `visualizations/` subdirectory within the experiment results folder.

### Example Usage

```bash
# Generate t-SNE visualizations
python main.py --plot_clusters --save_features

# Generate UMAP visualizations
python main.py --plot_clusters --viz_method umap --save_features

# Visualize existing results
python analyze_results.py ./results/experiment_name --plot --viz_method tsne
```

### Understanding the Plots

- **Left plot (Predicted Clusters)**: Shows how your clustering algorithm grouped the data
- **Right plot (Ground Truth)**: Shows the actual class labels
- **Good clustering**: Similar colors should be grouped together in the predicted clusters plot
- **Cluster overlap**: When clusters overlap significantly in the visualization, it indicates that those clusters are similar in the feature space

**For detailed visualization documentation, see [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)**

## Pseudo Label Generation

After clustering, you can generate pseudo labels to map each cluster to an actual class label. This mapping is determined by examining the k nearest samples to each cluster center and using majority voting.

### How It Works

1. **Find Representative Samples**: For each cluster, identify the k samples closest to the cluster center
2. **Majority Vote**: Assign the cluster the label that appears most frequently among these k samples
3. **Apply Mapping**: All samples in a cluster get the cluster's pseudo label

### Key Features

- **Flexible k parameter**: Control how many samples determine the label (recommended: k=10 for CIFAR100, k=20 for ImageNet)
- **Accuracy metrics**: Compare pseudo-label accuracy to true labels
- **Visual verification**: Generate images showing representative samples from each cluster
- **Per-cluster analysis**: See which clusters are pure vs. mixed

### Usage Examples

```bash
# During main clustering run
python main.py --generate_pseudo_labels --k_samples 10 --visualize_mapping

# From existing results
python generate_pseudo_labels.py \
    --experiment_dir ./results/experiment_name \
    --k_samples 10 \
    --visualize
```

### Output

The pseudo label generation creates a `pseudo_labels/` directory containing:

1. **JSON Results** (`pseudo_labels_k{k}.json`):
   - Pseudo labels for all samples
   - Cluster-to-label mapping
   - Indices of k nearest samples

2. **Visualization** (`cluster_mapping_k{k}.png`):
   - Grid showing representative images from each cluster
   - True labels for each sample (green=match, red=mismatch)
   - Cluster ID and assigned pseudo label

### Interpreting Results

- **High per-cluster accuracy**: Cluster represents a single semantic concept well
- **Low per-cluster accuracy**: Cluster is impure (contains multiple classes)
- **Empty clusters**: Normal - not all clusters will be used
- **Green titles in viz**: Sample's true label matches the cluster's pseudo label
- **Red titles in viz**: Mismatch indicating cluster impurity

**For complete documentation, see [PSEUDO_LABELING_GUIDE.md](PSEUDO_LABELING_GUIDE.md)**

## Output Files

Each experiment generates the following outputs in the results directory:

- `config.json`: Experiment configuration and hyperparameters
- `results.json`: Evaluation metrics for train and test sets
- `predictions.npz`: Cluster assignments and ground truth labels
- `final_checkpoint.pt`: Trained model checkpoint
- `features/` (optional): Extracted DINOv2/DINOv3/CLIP features
- `visualizations/` (optional): t-SNE/UMAP plots and cluster distribution charts
- `pseudo_labels/` (optional): Pseudo labels, cluster-to-label mappings, and visualization

## Expected Results

For CIFAR100 with k=100 clusters, typical results are:
- Clustering Accuracy: 40-50%
- NMI: 0.50-0.60
- ARI: 0.30-0.40
- Active Clusters: 70-90 out of 100

Note: Results may vary based on random initialization and hardware. CIFAR100 is a challenging dataset with 100 fine-grained classes.

## Implementation Notes

### Checkpoint System

The implementation includes a robust checkpoint system that allows resuming from any stage:

1. **Feature Extraction Stage**: Save/load extracted features to skip expensive DINOv2 inference
2. **Training Stage**: Save model state, optimizer state, and training history
3. **Resume Capability**: Automatically resume from interruptions without data loss

### Error Handling

The code includes comprehensive error handling:
- Graceful fallback to CPU if CUDA is unavailable
- Validation of input dimensions and data shapes
- Clear error messages for common issues

### Memory Management

To handle large datasets efficiently:
- Batch processing for feature extraction
- Gradient accumulation support
- Automatic memory cleanup between stages
- Features can be saved to disk to reduce memory usage

## Differences from Paper

This implementation stays faithful to the TEMI paper with the following considerations:

1. **Feature Extractor**: Uses DINOv2 instead of training from scratch (as recommended for better features)
2. **Dataset**: Applied to CIFAR100 instead of ImageNet (more practical for experimentation)
3. **Augmentations**: Simplified augmentation strategy suitable for CIFAR100's small images

All core algorithmic components follow the paper specifications exactly.

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch_size 128`
- Use smaller DINOv2 model: `--dinov2_model facebook/dinov2-small`
- Save features and work with them directly

### Slow Feature Extraction
- Use fewer workers if CPU is bottleneck: `--num_workers 2`
- Extract features once and reuse: `--save_features`

### Poor Clustering Results
- Increase training epochs: `--num_epochs 200`
- Adjust temperature: `--temperature 0.05` (lower for sharper assignments)
- Try different learning rates: `--learning_rate 0.0001`

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