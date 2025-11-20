"""
Extract features for Tiny ImageNet and save image paths to create accurate pseudo-label mappings.
Run this to generate features with image path tracking (no shuffle).
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
sys.path.append('/home/ssl.distillation/clustering')

from src.data_loader import TinyImageNetDataset, get_tiny_imagenet_transforms
from src.feature_extractor import DINOv2FeatureExtractor
from tqdm import tqdm

def extract_features_with_paths(dataset_root='data', output_dir='results/dinov2_large_tinyimagenet_with_paths'):
    """Extract features while tracking image paths"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING TINY IMAGENET FEATURES WITH IMAGE PATH TRACKING")
    print("="*80)
    
    # Create datasets
    train_transform, test_transform = get_tiny_imagenet_transforms(224)
    
    hf_dataset_name = "zh-plus/tiny-imagenet"
    
    train_dataset = TinyImageNetDataset(
        dataset_name=hf_dataset_name,
        split='train',
        transform=train_transform,
        streaming=False
    )
    
    test_dataset = TinyImageNetDataset(
        dataset_name=hf_dataset_name,
        split='valid',
        transform=test_transform,
        streaming=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create loaders WITHOUT shuffle to maintain order
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,  # IMPORTANT: No shuffle to maintain order
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize feature extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    feature_extractor = DINOv2FeatureExtractor(
        model_name='facebook/dinov2-large',
        device=device
    )
    
    # Extract features and save paths
    print("\n>>> Extracting training features...")
    train_features, train_labels = feature_extractor.extract_features(
        train_loader,
        return_labels=True
    )
    
    # Save training image identifiers
    # For HuggingFace datasets, we need to get image IDs/indices
    print("Collecting training image paths/identifiers...")
    train_paths = []
    if hasattr(train_dataset.dataset, 'data') and 'image' in train_dataset.dataset.data.column_names:
        # Try to get image paths or create identifiers based on index
        for idx in range(len(train_dataset)):
            # Create identifier: split_index format (e.g., "train_0", "train_1")
            train_paths.append(f"train_{idx}")
    else:
        # Fallback: use sequential indices
        train_paths = [f"train_{idx}" for idx in range(len(train_dataset))]
    
    print("\n>>> Extracting test features...")
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    
    # Save test image identifiers
    print("Collecting test image paths/identifiers...")
    test_paths = []
    if hasattr(test_dataset.dataset, 'data') and 'image' in test_dataset.dataset.data.column_names:
        for idx in range(len(test_dataset)):
            test_paths.append(f"valid_{idx}")
    else:
        test_paths = [f"valid_{idx}" for idx in range(len(test_dataset))]
    
    # Verify counts
    assert len(train_features) == len(train_paths), \
        f"Mismatch: {len(train_features)} features vs {len(train_paths)} paths"
    assert len(test_features) == len(test_paths), \
        f"Mismatch: {len(test_features)} features vs {len(test_paths)} paths"
    
    # Save everything
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Save features
    print(f"\nSaving features to {features_dir}...")
    torch.save({
        'features': train_features,
        'labels': train_labels,
        'paths': train_paths,
        'feature_dim': train_features.shape[1],
        'model_name': 'dinov2-large'
    }, features_dir / 'train_features.pt')
    
    torch.save({
        'features': test_features,
        'labels': test_labels,
        'paths': test_paths,
        'feature_dim': test_features.shape[1],
        'model_name': 'dinov2-large'
    }, features_dir / 'test_features.pt')
    
    # Save path mappings as JSON
    path_mapping = {
        'train_paths': train_paths,
        'test_paths': test_paths,
        'dataset': 'tiny-imagenet',
        'note': 'Images are in the same order as features (no shuffle). Identifiers are in format: split_index'
    }
    
    with open(output_dir / 'image_paths.json', 'w') as f:
        json.dump(path_mapping, f, indent=2)
    
    print(f"\nFeatures saved to {features_dir}")
    print(f"Image paths saved to {output_dir / 'image_paths.json'}")
    print(f"\nTrain: {len(train_paths)} images")
    print(f"Test: {len(test_paths)} images")
    print(f"Feature dimension: {train_features.shape[1]}")
    print("\n" + "="*80)
    print("NEXT STEP: Run clustering with these features")
    print("="*80)
    print("\npython main.py \\")
    print("  --dataset tiny-imagenet \\")
    print("  --model_type dinov2 \\")
    print("  --dinov2_model facebook/dinov2-large \\")
    print("  --num_clusters 200 \\")
    print("  --num_epochs 100 \\")
    print("  --batch_size 256 \\")
    print("  --load_features results/dinov2_large_tinyimagenet_with_paths/features \\")
    print("  --experiment_name tinyimagenet_pseudolabels \\")
    print("  --generate_pseudo_labels \\")
    print("  --k_samples 10")

if __name__ == "__main__":
    extract_features_with_paths()
