#!/usr/bin/env python3
"""
Example usage of CIFAR10 and Tiny ImageNet datasets with the clustering framework.

This script demonstrates how to use the newly added dataset support.
"""

# Example 1: CIFAR10 with DINOv2
print("=" * 60)
print("Example 1: CIFAR10 with DINOv2")
print("=" * 60)
print("""
Run clustering on CIFAR10 (10 classes) using DINOv2:

python main.py --dataset cifar10 --num_clusters 10

This will:
- Download CIFAR10 automatically (or use pre-downloaded data)
- Extract features using DINOv2
- Perform TEMI clustering with 10 clusters
- Evaluate clustering quality
""")

# Example 2: CIFAR10 with CLIP
print("\n" + "=" * 60)
print("Example 2: CIFAR10 with CLIP")
print("=" * 60)
print("""
Run clustering on CIFAR10 using CLIP features:

python main.py --dataset cifar10 --model_type clip --num_clusters 10

This uses CLIP's vision encoder for feature extraction.
""")

# Example 3: CIFAR10 with visualization
print("\n" + "=" * 60)
print("Example 3: CIFAR10 with visualization")
print("=" * 60)
print("""
Run CIFAR10 clustering with t-SNE visualization:

python main.py \\
    --dataset cifar10 \\
    --num_clusters 10 \\
    --plot_clusters \\
    --save_features

This will generate:
- Cluster visualization plots
- Feature embeddings saved for later use
""")

# Example 4: CIFAR10 with pseudo labels
print("\n" + "=" * 60)
print("Example 4: CIFAR10 with pseudo labels")
print("=" * 60)
print("""
Generate pseudo labels for CIFAR10 clusters:

python main.py \\
    --dataset cifar10 \\
    --num_clusters 10 \\
    --generate_pseudo_labels \\
    --k_samples 5 \\
    --visualize_mapping \\
    --save_features

This will:
- Cluster the data
- Map clusters to actual class labels
- Generate visualization of cluster assignments
""")

# Example 5: Tiny ImageNet with DINOv2
print("\n" + "=" * 60)
print("Example 5: Tiny ImageNet with DINOv2")
print("=" * 60)
print("""
Run clustering on Tiny ImageNet (200 classes) using DINOv2:

python main.py --dataset tiny-imagenet --num_clusters 200

This will:
- Download Tiny ImageNet from HuggingFace (zh-plus/tiny-imagenet)
- Extract features using DINOv2
- Perform TEMI clustering with 200 clusters
- Evaluate clustering quality

Note: Requires internet access to download the dataset.
""")

# Example 6: Tiny ImageNet with CLIP
print("\n" + "=" * 60)
print("Example 6: Tiny ImageNet with CLIP")
print("=" * 60)
print("""
Run clustering on Tiny ImageNet using CLIP features:

python main.py \\
    --dataset tiny-imagenet \\
    --model_type clip \\
    --clip_model openai/clip-vit-large-patch14 \\
    --num_clusters 200

This uses CLIP's large vision encoder for better features.
""")

# Example 7: Tiny ImageNet with fewer clusters
print("\n" + "=" * 60)
print("Example 7: Tiny ImageNet with fewer clusters")
print("=" * 60)
print("""
Run Tiny ImageNet clustering with fewer clusters:

python main.py --dataset tiny-imagenet --num_clusters 50

This can be useful for finding higher-level semantic groupings.
""")

# Example 8: Compare all datasets
print("\n" + "=" * 60)
print("Example 8: Compare all datasets")
print("=" * 60)
print("""
Compare clustering performance across all datasets:

# CIFAR10 (10 classes)
python main.py --dataset cifar10 --save_features --experiment_name exp_cifar10

# CIFAR100 (100 classes)  
python main.py --dataset cifar100 --save_features --experiment_name exp_cifar100

# Tiny ImageNet (200 classes)
python main.py --dataset tiny-imagenet --save_features --experiment_name exp_tinyimagenet

# ImageNet-1K (1000 classes)
python main.py --dataset imagenet --save_features --experiment_name exp_imagenet

Results will be saved in ./results/ directory.
""")

# Dataset information
print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)
print("""
Supported datasets and their characteristics:

1. CIFAR10
   - Classes: 10
   - Image size: 32x32 (resized to 224x224)
   - Train samples: 50,000
   - Test samples: 10,000
   - Default clusters: 10
   - Source: torchvision.datasets.CIFAR10

2. CIFAR100
   - Classes: 100
   - Image size: 32x32 (resized to 224x224)
   - Train samples: 50,000
   - Test samples: 10,000
   - Default clusters: 100
   - Source: torchvision.datasets.CIFAR100

3. Tiny ImageNet
   - Classes: 200
   - Image size: 64x64 (resized to 224x224)
   - Train samples: 100,000
   - Test samples: 10,000
   - Default clusters: 200
   - Source: HuggingFace (zh-plus/tiny-imagenet)
   - Requires: Internet access for first download

4. ImageNet-1K
   - Classes: 1000
   - Image size: 128x128 (resized to 224x224)
   - Train samples: 1,281,167
   - Test samples: 50,000
   - Default clusters: 1000
   - Source: HuggingFace (benjamin-paine/imagenet-1k-128x128)
   - Requires: Internet access for first download
""")

print("\n" + "=" * 60)
print("Notes")
print("=" * 60)
print("""
1. CIFAR10 can work with pre-downloaded data at ./data/cifar-10-batches-py/
   or will download automatically if not present.

2. Tiny ImageNet requires internet access to HuggingFace Hub for first use.

3. All datasets are automatically preprocessed with ImageNet normalization
   for compatibility with DINOv2 and CLIP models.

4. The number of clusters can be customized using --num_clusters argument.

5. Use --save_features to avoid re-extracting features in subsequent runs.
""")
