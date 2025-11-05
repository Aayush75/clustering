"""
Example usage of TEMI clustering with ImageNet dataset.

This script demonstrates how to use the ImageNet-1K dataset
from HuggingFace (benjamin-paine/imagenet-1k-128x128) for clustering.

Requirements:
    - Install datasets library: pip install datasets
    - Network connection to download ImageNet from HuggingFace

Usage Examples:
    # Basic usage with ImageNet and DINOv2
    python main.py --dataset imagenet
    
    # ImageNet with CLIP
    python main.py --dataset imagenet --model_type clip
    
    # ImageNet with custom number of clusters (default is 1000)
    python main.py --dataset imagenet --num_clusters 500
    
    # ImageNet with DINOv2-large and visualization
    python main.py --dataset imagenet \
                   --dinov2_model facebook/dinov2-large \
                   --plot_clusters \
                   --save_features
    
    # ImageNet with CLIP-large and fewer epochs
    python main.py --dataset imagenet \
                   --model_type clip \
                   --clip_model openai/clip-vit-large-patch14 \
                   --num_epochs 50 \
                   --save_features
"""

print(__doc__)

# Quick validation that imports work
try:
    from src.data_loader import ImageNetDataset, create_data_loaders
    print("\n✓ ImageNet support is available!")
    print("✓ Use the commands above to run clustering on ImageNet")
    print("\nNote: Make sure you have:")
    print("  1. Installed the datasets library (pip install datasets)")
    print("  2. Network access to download from HuggingFace")
    print("  3. Sufficient disk space (~150GB for full ImageNet)")
    print("\nThe ImageNet-1K dataset has:")
    print("  - Training set: ~1.28M images")
    print("  - Validation set: ~50K images")
    print("  - 1000 classes")
    print("  - Images are 128x128 (resized to 224x224 for models)")
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("Please run: pip install -r requirements.txt")
