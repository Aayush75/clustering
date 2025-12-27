"""
Create correct path-based pseudo-label mappings for ImageNet-1K.
Run this AFTER clustering completes.

This version extracts image paths/identifiers from the parquet files
to create mappings similar to CIFAR-100.
"""
import json
import numpy as np
from pathlib import Path
import torch
import argparse
import pandas as pd
from tqdm import tqdm


def load_parquet_image_paths(parquet_path, split='train'):
    """
    Load image paths/identifiers from ImageNet-1K parquet files.
    
    Args:
        parquet_path: Path to the ImageNet parquet directory
        split: Dataset split ('train', 'validation', or 'test')
    
    Returns:
        List of image identifiers in the order they appear
    """
    parquet_path = Path(parquet_path)
    
    # Try different directory structures
    split_dir = parquet_path / split
    if not split_dir.exists():
        # Maybe it's already the split directory
        split_dir = parquet_path
    
    # Find all parquet files for this split
    parquet_files = sorted(list(split_dir.glob('*.parquet')))
    
    if not parquet_files:
        # Try alternative structure with prefixes
        parquet_files = sorted(list(parquet_path.glob(f'{split}-*.parquet')))
    
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for split '{split}' in {parquet_path}.\n"
            f"Checked: {split_dir} and {parquet_path}/{split}-*.parquet"
        )
    
    print(f"Found {len(parquet_files)} parquet file(s) for {split} split")
    
    # Load all parquet files and extract identifiers
    image_paths = []
    
    for parquet_file in tqdm(parquet_files, desc=f"Loading {split} parquet files"):
        df = pd.read_parquet(parquet_file)
        
        # Try to find image identifier columns
        # Common columns: 'image', 'file_name', 'path', 'id', etc.
        if 'file_name' in df.columns:
            paths = df['file_name'].tolist()
        elif 'path' in df.columns:
            paths = df['path'].tolist()
        elif 'id' in df.columns:
            paths = df['id'].tolist()
        else:
            # If no identifier column, create synthetic ones based on index
            # Format: split_fileindex_rowindex
            file_idx = parquet_files.index(parquet_file)
            paths = [f"{split}_{file_idx}_{i}" for i in range(len(df))]
        
        image_paths.extend(paths)
    
    return image_paths


def create_mappings(
    experiment_dir="results/temi_imagenet-1k_1000clusters_20251224_194551",
    parquet_path=None
):
    """Create path-based pseudo-label mappings for ImageNet-1K
    
    Args:
        experiment_dir: Path to the experiment directory containing features, pseudo_labels, and predictions
        parquet_path: Path to ImageNet parquet files to extract image identifiers
    """
    
    print("="*80)
    print("CREATING PATH-BASED PSEUDO-LABEL MAPPING FOR IMAGENET-1K")
    print("="*80)
    
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Load experiment configuration to get parquet path if not provided
    config_path = experiment_dir / "config.json"
    if config_path.exists() and parquet_path is None:
        with open(config_path) as f:
            config = json.load(f)
            parquet_path = config.get('imagenet_path')
    
    if parquet_path is None:
        raise ValueError(
            "parquet_path must be provided either as argument or in experiment config.\n"
            "Use: --parquet_path /path/to/imagenet/parquet/files"
        )
    
    parquet_path = Path(parquet_path).expanduser()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path not found: {parquet_path}")
    
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"Parquet path: {parquet_path}")
    
    # Load features to get the number of samples
    features_dir = experiment_dir / "features"
    train_features_path = features_dir / "train_features.pt"
    test_features_path = features_dir / "test_features.pt"
    
    print(f"\nLoading features from {features_dir}...")
    train_data = torch.load(train_features_path)
    test_data = torch.load(test_features_path)
    
    n_train = len(train_data['features'])
    n_test = len(test_data['features'])
    
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    # Load image paths from parquet files
    print("\n" + "-"*80)
    print("Loading image paths from parquet files...")
    print("-"*80)
    
    train_paths = load_parquet_image_paths(parquet_path, split='train')
    
    # For test, try 'validation' first, then 'test'
    try:
        test_paths = load_parquet_image_paths(parquet_path, split='validation')
    except FileNotFoundError:
        try:
            test_paths = load_parquet_image_paths(parquet_path, split='test')
        except FileNotFoundError:
            print("Warning: Could not find test/validation parquet files")
            test_paths = [f"test_{i}" for i in range(n_test)]
    
    print(f"\nLoaded {len(train_paths)} training paths")
    print(f"Loaded {len(test_paths)} test paths")
    
    # Verify lengths match
    if len(train_paths) != n_train:
        print(f"\nWARNING: Mismatch in training set!")
        print(f"  Features: {n_train}, Paths: {len(train_paths)}")
        if len(train_paths) > n_train:
            print(f"  Truncating paths to match features...")
            train_paths = train_paths[:n_train]
        else:
            raise ValueError("Cannot proceed: fewer paths than features")
    
    if len(test_paths) != n_test:
        print(f"\nWARNING: Mismatch in test set!")
        print(f"  Features: {n_test}, Paths: {len(test_paths)}")
        if len(test_paths) > n_test:
            print(f"  Truncating paths to match features...")
            test_paths = test_paths[:n_test]
        else:
            raise ValueError("Cannot proceed: fewer paths than features")
    
    # Load the pseudo-labels
    pseudo_dir = experiment_dir / "pseudo_labels"
    pseudo_file = pseudo_dir / "pseudo_labels_k10.json"
    
    print(f"\nLoading pseudo-labels from {pseudo_file}...")
    with open(pseudo_file) as f:
        pseudo_data = json.load(f)
    
    train_pseudo_labels = pseudo_data['train_pseudo_labels']
    test_pseudo_labels = pseudo_data['test_pseudo_labels']
    cluster_to_label = pseudo_data['train_cluster_to_label']
    
    # Also load confidence scores if available
    train_confidence = pseudo_data.get('train_confidence_scores', None)
    test_confidence = pseudo_data.get('test_confidence_scores', None)
    train_cluster_confidence = pseudo_data.get('train_cluster_to_confidence', {})
    
    print(f"Loaded {len(train_pseudo_labels)} training pseudo-labels")
    print(f"Loaded {len(test_pseudo_labels)} test pseudo-labels")
    
    # Verify lengths match
    assert n_train == len(train_pseudo_labels), \
        f"Mismatch: {n_train} samples vs {len(train_pseudo_labels)} labels"
    assert n_test == len(test_pseudo_labels), \
        f"Mismatch: {n_test} samples vs {len(test_pseudo_labels)} labels"
    
    # Load cluster assignments and true labels from predictions.npz
    pred_file = experiment_dir / "predictions.npz"
    print(f"\nLoading predictions from {pred_file}...")
    pred_data = np.load(pred_file)
    train_cluster_assignments = pred_data['train_predictions']
    test_cluster_assignments = pred_data['test_predictions']
    train_true_labels = pred_data['train_labels']
    test_true_labels = pred_data['test_labels']
    
    # ImageNet-1K has 1000 classes
    num_classes = 1000
    print(f"\nDataset: ImageNet-1K with {num_classes} classes")
    
    print("\nCreating mappings...")
    
    # Create detailed mapping (similar to CIFAR-100 structure)
    full_mapping = {
        "train": {},
        "test": {},
        "num_classes": num_classes,
        "cluster_to_label": {int(k): int(v) for k, v in cluster_to_label.items()},
        "cluster_to_confidence": {int(k): float(v) for k, v in train_cluster_confidence.items()},
        "dataset": "imagenet-1k"
    }
    
    # Training set
    for idx in range(n_train):
        path = train_paths[idx]
        cluster_id = int(train_cluster_assignments[idx])
        pseudo_label = int(train_pseudo_labels[idx])
        true_label = int(train_true_labels[idx])
        
        full_mapping["train"][path] = {
            "index": idx,
            "cluster_id": cluster_id,
            "pseudo_label": pseudo_label,
            "pseudo_label_name": f"Class_{pseudo_label}",
            "true_label": true_label,
            "true_label_name": f"Class_{true_label}",
            "correct": pseudo_label == true_label
        }
        
        # Add confidence if available
        if train_confidence is not None:
            full_mapping["train"][path]["confidence"] = float(train_confidence[idx])
    
    # Test/Validation set
    for idx in range(n_test):
        path = test_paths[idx]
        cluster_id = int(test_cluster_assignments[idx])
        pseudo_label = int(test_pseudo_labels[idx])
        true_label = int(test_true_labels[idx])
        
        full_mapping["test"][path] = {
            "index": idx,
            "cluster_id": cluster_id,
            "pseudo_label": pseudo_label,
            "pseudo_label_name": f"Class_{pseudo_label}",
            "true_label": true_label,
            "true_label_name": f"Class_{true_label}",
            "correct": pseudo_label == true_label
        }
        
        # Add confidence if available
        if test_confidence is not None:
            full_mapping["test"][path]["confidence"] = float(test_confidence[idx])
    
    # Save detailed mapping
    output_file = pseudo_dir / "image_path_to_pseudolabel.json"
    with open(output_file, 'w') as f:
        json.dump(full_mapping, f, indent=2)
    
    print(f"\nSaved detailed mapping to {output_file}")
    
    # Create simple mapping (path -> pseudo_label only)
    simple_mapping = {}
    for path, info in full_mapping["train"].items():
        simple_mapping[path] = info["pseudo_label"]
    for path, info in full_mapping["test"].items():
        simple_mapping[path] = info["pseudo_label"]
    
    simple_output = pseudo_dir / "image_path_to_pseudolabel_simple.json"
    with open(simple_output, 'w') as f:
        json.dump(simple_mapping, f, indent=2)
    
    print(f"Saved simple mapping to {simple_output}")
    
    # Create CSV files (matching CIFAR-100 format)
    print("\nCreating CSV files...")
    
    train_csv = pseudo_dir / "train_image_pseudo_labels.csv"
    with open(train_csv, 'w') as f:
        # Header matching CIFAR-100 format
        if train_confidence is not None:
            f.write("image_path,cluster_id,pseudo_label_class_index,pseudo_label_class_name,true_label_index,true_label_name,correct,confidence\n")
        else:
            f.write("image_path,cluster_id,pseudo_label_class_index,pseudo_label_class_name,true_label_index,true_label_name,correct\n")
        
        for path in train_paths:
            info = full_mapping["train"][path]
            correct_str = "yes" if info["correct"] else "no"
            
            if train_confidence is not None:
                f.write(f'"{path}",{info["cluster_id"]},{info["pseudo_label"]},"{info["pseudo_label_name"]}",{info["true_label"]},"{info["true_label_name"]}",{correct_str},{info["confidence"]:.4f}\n')
            else:
                f.write(f'"{path}",{info["cluster_id"]},{info["pseudo_label"]},"{info["pseudo_label_name"]}",{info["true_label"]},"{info["true_label_name"]}",{correct_str}\n')
    
    print(f"Saved {train_csv}")
    
    test_csv = pseudo_dir / "test_image_pseudo_labels.csv"
    with open(test_csv, 'w') as f:
        # Header matching CIFAR-100 format
        if test_confidence is not None:
            f.write("image_path,cluster_id,pseudo_label_class_index,pseudo_label_class_name,true_label_index,true_label_name,correct,confidence\n")
        else:
            f.write("image_path,cluster_id,pseudo_label_class_index,pseudo_label_class_name,true_label_index,true_label_name,correct\n")
        
        for path in test_paths:
            info = full_mapping["test"][path]
            correct_str = "yes" if info["correct"] else "no"
            
            if test_confidence is not None:
                f.write(f'"{path}",{info["cluster_id"]},{info["pseudo_label"]},"{info["pseudo_label_name"]}",{info["true_label"]},"{info["true_label_name"]}",{correct_str},{info["confidence"]:.4f}\n')
            else:
                f.write(f'"{path}",{info["cluster_id"]},{info["pseudo_label"]},"{info["pseudo_label_name"]}",{info["true_label"]},"{info["true_label_name"]}",{correct_str}\n')
    
    print(f"Saved {test_csv}")
    
    # Calculate and show accuracy
    train_correct = sum(1 for info in full_mapping["train"].values() if info["correct"])
    test_correct = sum(1 for info in full_mapping["test"].values() if info["correct"])
    train_acc = train_correct / n_train * 100
    test_acc = test_correct / n_test * 100
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training: {n_train} images, {train_acc:.2f}% accuracy")
    print(f"Test: {n_test} images, {test_acc:.2f}% accuracy")
    
    print("\n" + "="*80)
    print("EXAMPLE MAPPINGS (first 5 training images):")
    print("="*80)
    for i, path in enumerate(list(full_mapping["train"].keys())[:5]):
        info = full_mapping["train"][path]
        match = "[OK]" if info["correct"] else "[FAIL]"
        print(f"\n{path}")
        print(f"  Cluster: {info['cluster_id']} → Pseudo: {info['pseudo_label']} ({info['pseudo_label_name']})")
        print(f"  True: {info['true_label']} ({info['true_label_name']}) {match}")
        if train_confidence is not None and 'confidence' in info:
            print(f"  Confidence: {info['confidence']:.4f}")
    
    print("\n" + "="*80)
    print("USAGE IN ANOTHER REPO:")
    print("="*80)
    print("""
# Load simple mapping
import json
with open('image_path_to_pseudolabel_simple.json') as f:
    mapping = json.load(f)
# Use image path as key
pseudo_label = mapping['train_0_1234']  # Example path

# Or use CSV
import pandas as pd
train_df = pd.read_csv('train_image_pseudo_labels.csv')
test_df = pd.read_csv('test_image_pseudo_labels.csv')

# Get pseudo label for a specific image
pseudo_label = train_df[train_df['image_path'] == 'train_0_1234']['pseudo_label_class_index'].values[0]
""")
    
    print("\nAll mappings created successfully!")
    print(f"Files saved in: {pseudo_dir}")
    print("\nGenerated files:")
    print(f"  - train_image_pseudo_labels.csv")
    print(f"  - test_image_pseudo_labels.csv")
    print(f"  - image_path_to_pseudolabel.json (detailed)")
    print(f"  - image_path_to_pseudolabel_simple.json (simple lookup)")
    print(f"\nNote: The pseudo_labels_k10.json already exists from the clustering run.")
    print(f"      The cluster_mapping_k10.png visualization requires running with --visualize_mapping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create pseudo-label mappings for ImageNet-1K clustering results'
    )
    parser.add_argument('--experiment_dir', type=str,
                        default='results/temi_imagenet-1k_1000clusters_20251224_194551',
                        help='Path to experiment directory')
    parser.add_argument('--parquet_path', type=str, default=None,
                        help='Path to ImageNet parquet files (required to extract image paths)')
    
    args = parser.parse_args()
    
    if args.parquet_path is None:
        print("\nWARNING: --parquet_path not provided. Will try to load from experiment config.")
        print("If this fails, please provide: --parquet_path /path/to/imagenet/parquet/files\n")
    
    create_mappings(experiment_dir=args.experiment_dir, parquet_path=args.parquet_path)
