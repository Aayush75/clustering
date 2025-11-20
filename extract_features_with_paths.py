"""
Extract features and save image paths to create accurate pseudo-label mappings.
Run this to regenerate features with image path tracking.
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
sys.path.append('/home/ssl.distillation/clustering')

from src.data_loader import ImagenetteDataset, get_imagenette_transforms
from src.feature_extractor import DINOv2FeatureExtractor
from tqdm import tqdm

def extract_features_with_paths(dataset_root='data', output_dir='results/dinov2_large_imagenette_with_paths'):
    """Extract features while tracking image paths"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING FEATURES WITH IMAGE PATH TRACKING")
    print("="*80)
    
    # Create datasets
    train_transform, test_transform = get_imagenette_transforms(224)
    
    train_dataset = ImagenetteDataset(
        root=dataset_root,
        split='train',
        transform=train_transform,
        download=True,
        version='320'
    )
    
    test_dataset = ImagenetteDataset(
        root=dataset_root,
        split='val',
        transform=test_transform,
        download=True,
        version='320'
    )
    
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
    feature_extractor = DINOv2FeatureExtractor(
        model_name='facebook/dinov2-large',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Extract features and save paths
    print("\n>>> Extracting training features...")
    train_features, train_labels = feature_extractor.extract_features(
        train_loader,
        return_labels=True
    )
    
    # Save training image paths in order
    train_paths = []
    for img_path in train_dataset.samples:
        rel_path = str(Path(*img_path.parts[-3:]))  # split/class/image.JPEG
        train_paths.append(rel_path)
    
    print("\n>>> Extracting test features...")
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    
    # Save test image paths in order
    test_paths = []
    for img_path in test_dataset.samples:
        rel_path = str(Path(*img_path.parts[-3:]))
        test_paths.append(rel_path)
    
    # Save everything
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Save features
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
        'note': 'Images are in the same order as features (no shuffle)'
    }
    
    with open(output_dir / 'image_paths.json', 'w') as f:
        json.dump(path_mapping, f, indent=2)
    
    print(f"\nFeatures saved to {features_dir}")
    print(f"Image paths saved to {output_dir / 'image_paths.json'}")
    print(f"\nTrain: {len(train_paths)} images")
    print(f"Test: {len(test_paths)} images")
    print("\nNow you can run clustering on these features and")
    print("the pseudo-labels will correspond to the correct image paths!")

if __name__ == "__main__":
    extract_features_with_paths()
