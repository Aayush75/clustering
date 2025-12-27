"""
Extract features for ImageNet-1K and save image indices to create accurate pseudo-label mappings.
Run this to generate features with image index tracking (no shuffle).
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
sys.path.append('/home/ssl.distillation/clustering')

from src.data_loader import ImageNet1KDataset, get_imagenet1k_transforms
from src.feature_extractor import DINOv2FeatureExtractor
from tqdm import tqdm

def extract_features_with_paths(dataset_root='/home/ssl.distillation/WMDD/datasets', 
                                output_dir='results/dinov2_large_imagenet1k_with_paths'):
    """Extract features while tracking image indices"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING IMAGENET-1K FEATURES WITH IMAGE INDEX TRACKING")
    print("="*80)
    print(f"Dataset root: {dataset_root}/imagenet_data/train and val")
    
    # Create datasets
    train_transform, test_transform = get_imagenet1k_transforms(224)
    
    train_dataset = ImageNet1KDataset(
        root=dataset_root,
        train=True,
        transform=train_transform
    )
    
    test_dataset = ImageNet1KDataset(
        root=dataset_root,
        train=False,
        transform=test_transform
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
    
    # Extract features
    print("\n>>> Extracting training features...")
    train_features, train_labels = feature_extractor.extract_features(
        train_loader,
        return_labels=True
    )
    
    # Get actual image paths
    print("Collecting training image paths...")
    train_paths = []
    for img_path in train_dataset.samples:
        # Get relative path: train/0000/train_0000000.JPEG
        rel_path = str(Path(*img_path.parts[-3:]))  # split/class/image.JPEG
        train_paths.append(rel_path)
    
    # Validate paths
    print(f"  Collected {len(train_paths)} training paths")
    print(f"  Example paths: {train_paths[:3]}")
    
    print("\n>>> Extracting test features...")
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    
    # Get test image paths
    print("Collecting test image paths...")
    test_paths = []
    for img_path in test_dataset.samples:
        rel_path = str(Path(*img_path.parts[-3:]))  # split/class/image.JPEG
        test_paths.append(rel_path)
    
    # Validate paths
    print(f"  Collected {len(test_paths)} test paths")
    print(f"  Example paths: {test_paths[:3]}")
    
    # Verify counts match exactly
    print("\nValidating data integrity...")
    assert len(train_features) == len(train_paths), \
        f"ERROR: Mismatch in training data - {len(train_features)} features vs {len(train_paths)} paths"
    assert len(test_features) == len(test_paths), \
        f"ERROR: Mismatch in test data - {len(test_features)} features vs {len(test_paths)} paths"
    assert len(train_features) == len(train_labels), \
        f"ERROR: Mismatch in training data - {len(train_features)} features vs {len(train_labels)} labels"
    assert len(test_features) == len(test_labels), \
        f"ERROR: Mismatch in test data - {len(test_features)} features vs {len(test_labels)} labels"
    
    print("  All data counts match perfectly")
    print(f"  Training: {len(train_features)} features, {len(train_paths)} paths, {len(train_labels)} labels")
    print(f"  Test: {len(test_features)} features, {len(test_paths)} paths, {len(test_labels)} labels")
    
    # Get class names
    class_names = train_dataset.class_names
    print(f"  Class names loaded: {len(class_names)} classes")
    
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
        'model_name': 'facebook/dinov2-large'
    }, features_dir / 'train_features.pt')
    
    torch.save({
        'features': test_features,
        'labels': test_labels,
        'paths': test_paths,
        'feature_dim': test_features.shape[1],
        'model_name': 'facebook/dinov2-large'
    }, features_dir / 'test_features.pt')
    
    # Save path mappings as JSON with validation info
    path_mapping = {
        'train_paths': train_paths,
        'test_paths': test_paths,
        'dataset': 'imagenet-1k',
        'num_classes': 1000,
        'class_names': class_names,
        'num_train': len(train_paths),
        'num_test': len(test_paths),
        'note': 'Images are in the same order as features (no shuffle). Paths are relative: train/0000/image.JPEG',
        'path_format': 'split/class/filename.JPEG'
    }
    
    json_path = output_dir / 'image_paths.json'
    with open(json_path, 'w') as f:
        json.dump(path_mapping, f, indent=2)
    
    print(f"\nFeatures saved to {features_dir}")
    print(f"Image paths saved to {json_path}")
    print(f"\nTrain: {len(train_paths)} images")
    print(f"Test: {len(test_paths)} images")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Number of classes: {len(class_names)}")
    
    # Final validation
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Path format: Relative file paths (train/0000/image.JPEG)")
    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Example training paths: {train_paths[:3]}")
    print(f"Example test paths: {test_paths[:3]}")
    print("\n" + "="*80)
    print("NEXT STEP: Run clustering with these features")
    print("="*80)
    print("\npython main.py \\")
    print("  --dataset imagenet-1k \\")
    print("  --model_type dinov2 \\")
    print("  --dinov2_model facebook/dinov2-large \\")
    print("  --num_clusters 1000 \\")
    print("  --num_epochs 100 \\")
    print("  --batch_size 256 \\")
    print("  --load_features results/dinov2_large_imagenet1k_with_paths/features \\")
    print("  --experiment_name imagenet1k_pseudolabels \\")
    print("  --generate_pseudo_labels \\")
    print("  --k_samples 10")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract ImageNet-1K features with path tracking')
    parser.add_argument('--dataset_root', type=str, 
                        default='/home/ssl.distillation/WMDD/datasets',
                        help='Root directory containing imagenet_data folder')
    parser.add_argument('--output_dir', type=str,
                        default='results/dinov2_large_imagenet1k_with_paths',
                        help='Output directory for features and paths')
    args = parser.parse_args()
    
    extract_features_with_paths(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir
    )
