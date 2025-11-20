"""
Extract features for CIFAR-100 and save image indices to create accurate pseudo-label mappings.
Run this to generate features with image index tracking (no shuffle).
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
sys.path.append('/home/ssl.distillation/clustering')

from src.data_loader import CIFAR100Dataset, get_cifar100_transforms
from src.feature_extractor import DINOv2FeatureExtractor
from tqdm import tqdm

def extract_features_with_paths(dataset_root='data', output_dir='results/dinov2_large_cifar100_with_paths',
                                use_folder_structure=False):
    """Extract features while tracking image indices"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING CIFAR-100 FEATURES WITH IMAGE INDEX TRACKING")
    print("="*80)
    
    if use_folder_structure:
        print("Using folder structure: data/cifar100/train/0/*.png")
    else:
        print("Using standard CIFAR-100 format")
    
    # Create datasets
    train_transform, test_transform = get_cifar100_transforms(224)
    
    train_dataset = CIFAR100Dataset(
        root=dataset_root,
        train=True,
        transform=train_transform,
        download=not use_folder_structure,  # Don't download if using folder structure
        use_folder_structure=use_folder_structure
    )
    
    test_dataset = CIFAR100Dataset(
        root=dataset_root,
        train=False,
        transform=test_transform,
        download=not use_folder_structure,
        use_folder_structure=use_folder_structure
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
    if use_folder_structure:
        # Get paths from folder structure
        train_paths = []
        for img_path in train_dataset.samples:
            # Get relative path: train/0/000002.png
            rel_path = str(Path(*img_path.parts[-3:]))  # split/class/image.png
            train_paths.append(rel_path)
        
        # Validate paths
        print(f"  Collected {len(train_paths)} training paths")
        print(f"  Example paths: {train_paths[:3]}")
    else:
        # For standard CIFAR-100, use sequential indices
        train_paths = [f"train_{idx}" for idx in range(len(train_dataset))]
        print(f"  Using sequential indices: {len(train_paths)} images")
    
    print("\n>>> Extracting test features...")
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    
    # Get test image paths
    print("Collecting test image paths...")
    if use_folder_structure:
        test_paths = []
        for img_path in test_dataset.samples:
            rel_path = str(Path(*img_path.parts[-3:]))  # split/class/image.png
            test_paths.append(rel_path)
        
        # Validate paths
        print(f"  Collected {len(test_paths)} test paths")
        print(f"  Example paths: {test_paths[:3]}")
    else:
        test_paths = [f"test_{idx}" for idx in range(len(test_dataset))]
        print(f"  Using sequential indices: {len(test_paths)} images")
    
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
    
    # Get class names from the underlying CIFAR100 dataset
    if use_folder_structure:
        class_names = train_dataset.class_names
    else:
        class_names = train_dataset.cifar100.classes
    
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
        'dataset': 'cifar-100',
        'num_classes': 100,
        'class_names': class_names,
        'use_folder_structure': use_folder_structure,
        'num_train': len(train_paths),
        'num_test': len(test_paths),
        'note': 'Images are in the same order as features (no shuffle). Paths are relative: train/0/image.png',
        'path_format': 'split/class/filename.png' if use_folder_structure else 'split_index'
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
    print(f"Path format: {'Relative file paths (train/class/image.png)' if use_folder_structure else 'Sequential indices (train_0, train_1, ...)'}")
    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Example training paths: {train_paths[:3]}")
    print(f"Example test paths: {test_paths[:3]}")
    print("\n" + "="*80)
    print("NEXT STEP: Run clustering with these features")
    print("="*80)
    print("\npython main.py \\")
    print("  --dataset cifar100 \\")
    print("  --model_type dinov2 \\")
    print("  --dinov2_model facebook/dinov2-large \\")
    print("  --num_clusters 100 \\")
    print("  --num_epochs 100 \\")
    print("  --batch_size 256 \\")
    print("  --load_features results/dinov2_large_cifar100_with_paths/features \\")
    print("  --experiment_name cifar100_pseudolabels \\")
    print("  --generate_pseudo_labels \\")
    print("  --k_samples 10")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract CIFAR-100 features with path tracking')
    parser.add_argument('--use_folder_structure', action='store_true',
                        help='Load from data/cifar100/train/0/*.png structure instead of standard format')
    args = parser.parse_args()
    
    extract_features_with_paths(use_folder_structure=args.use_folder_structure)
