"""
Create correct path-based pseudo-label mappings for Tiny ImageNet.
Run this AFTER clustering completes.
"""
import json
import numpy as np
from pathlib import Path
import torch

def create_mappings(
    paths_file="results/dinov2_large_tinyimagenet_with_paths/image_paths.json",
    pseudo_labels_dir="results/tinyimagenet_pseudo_labels/pseudo_labels",
    predictions_file="results/tinyimagenet_pseudo_labels/predictions.npz"
):
    """Create path-based pseudo-label mappings"""
    
    print("="*80)
    print("CREATING CORRECT PATH-BASED PSEUDO-LABEL MAPPING FOR TINY IMAGENET")
    print("="*80)
    
    # Load the image paths (from feature extraction with paths)
    paths_file = Path(paths_file)
    with open(paths_file) as f:
        path_data = json.load(f)
    
    train_paths = path_data['train_paths']
    test_paths = path_data['test_paths']
    
    print(f"\nLoaded {len(train_paths)} training paths")
    print(f"Loaded {len(test_paths)} test paths")
    
    # Load the pseudo-labels
    pseudo_dir = Path(pseudo_labels_dir)
    with open(pseudo_dir / "pseudo_labels_k10.json") as f:
        pseudo_data = json.load(f)
    
    train_pseudo_labels = pseudo_data['train_pseudo_labels']
    test_pseudo_labels = pseudo_data['test_pseudo_labels']
    cluster_to_label = pseudo_data['train_cluster_to_label']
    
    print(f"Loaded {len(train_pseudo_labels)} training pseudo-labels")
    print(f"Loaded {len(test_pseudo_labels)} test pseudo-labels")
    
    # Verify lengths match
    assert len(train_paths) == len(train_pseudo_labels), \
        f"Mismatch: {len(train_paths)} paths vs {len(train_pseudo_labels)} labels"
    assert len(test_paths) == len(test_pseudo_labels), \
        f"Mismatch: {len(test_paths)} paths vs {len(test_pseudo_labels)} labels"
    
    # Load cluster assignments and true labels
    pred_file = Path(predictions_file)
    pred_data = np.load(pred_file)
    train_cluster_assignments = pred_data['train_predictions']
    test_cluster_assignments = pred_data['test_predictions']
    train_true_labels = pred_data['train_labels']
    test_true_labels = pred_data['test_labels']
    
    # Tiny ImageNet has 200 classes
    num_classes = 200
    print(f"\nDataset: Tiny ImageNet with {num_classes} classes")
    
    print("\nCreating mappings...")
    
    # Create detailed mapping
    full_mapping = {
        "train": {},
        "valid": {},
        "num_classes": num_classes,
        "cluster_to_label": cluster_to_label,
        "dataset": "tiny-imagenet"
    }
    
    # Training set
    for idx, path in enumerate(train_paths):
        cluster_id = int(train_cluster_assignments[idx])
        pseudo_label = int(train_pseudo_labels[idx])
        true_label = int(train_true_labels[idx])
        
        full_mapping["train"][path] = {
            "index": idx,
            "cluster_id": cluster_id,
            "pseudo_label": pseudo_label,
            "true_label": true_label,
            "correct": pseudo_label == true_label
        }
    
    # Test set
    for idx, path in enumerate(test_paths):
        cluster_id = int(test_cluster_assignments[idx])
        pseudo_label = int(test_pseudo_labels[idx])
        true_label = int(test_true_labels[idx])
        
        full_mapping["valid"][path] = {
            "index": idx,
            "cluster_id": cluster_id,
            "pseudo_label": pseudo_label,
            "true_label": true_label,
            "correct": pseudo_label == true_label
        }
    
    # Save detailed mapping
    output_file = pseudo_dir / "image_path_to_pseudo_label.json"
    with open(output_file, 'w') as f:
        json.dump(full_mapping, f, indent=2)
    
    print(f"\nSaved detailed mapping to {output_file}")
    
    # Create simple mapping (path -> pseudo_label only)
    simple_mapping = {}
    for path, info in full_mapping["train"].items():
        simple_mapping[path] = info["pseudo_label"]
    for path, info in full_mapping["valid"].items():
        simple_mapping[path] = info["pseudo_label"]
    
    simple_output = pseudo_dir / "image_path_to_pseudo_label_simple.json"
    with open(simple_output, 'w') as f:
        json.dump(simple_mapping, f, indent=2)
    
    print(f"Saved simple mapping to {simple_output}")
    
    # Create CSV files
    print("\nCreating CSV files...")
    
    train_csv = pseudo_dir / "train_image_pseudo_labels.csv"
    with open(train_csv, 'w') as f:
        f.write("image_path,image_index,cluster_id,pseudo_label_class_index,true_label_index,correct\n")
        for path in train_paths:
            info = full_mapping["train"][path]
            f.write(f'"{path}",{info["index"]},{info["cluster_id"]},{info["pseudo_label"]},{info["true_label"]},{"yes" if info["correct"] else "no"}\n')
    
    print(f"Saved {train_csv}")
    
    test_csv = pseudo_dir / "test_image_pseudo_labels.csv"
    with open(test_csv, 'w') as f:
        f.write("image_path,image_index,cluster_id,pseudo_label_class_index,true_label_index,correct\n")
        for path in test_paths:
            info = full_mapping["valid"][path]
            f.write(f'"{path}",{info["index"]},{info["cluster_id"]},{info["pseudo_label"]},{info["true_label"]},{"yes" if info["correct"] else "no"}\n')
    
    print(f"Saved {test_csv}")
    
    # Calculate and show accuracy
    train_correct = sum(1 for info in full_mapping["train"].values() if info["correct"])
    test_correct = sum(1 for info in full_mapping["valid"].values() if info["correct"])
    train_acc = train_correct / len(train_paths) * 100
    test_acc = test_correct / len(test_paths) * 100
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training: {len(train_paths)} images, {train_acc:.2f}% accuracy")
    print(f"Test: {len(test_paths)} images, {test_acc:.2f}% accuracy")
    
    print("\n" + "="*80)
    print("EXAMPLE MAPPINGS (first 5 training images):")
    print("="*80)
    for i, path in enumerate(list(full_mapping["train"].keys())[:5]):
        info = full_mapping["train"][path]
        match = "[OK]" if info["correct"] else "[FAIL]"
        print(f"\n{path}")
        print(f"  Cluster: {info['cluster_id']} → Pseudo: {info['pseudo_label']}")
        print(f"  True: {info['true_label']} {match}")
    
    print("\n" + "="*80)
    print("USAGE IN ANOTHER REPO:")
    print("="*80)
    print("""
# Load simple mapping
import json
mapping = json.load(open('image_path_to_pseudo_label_simple.json'))
pseudo_label = mapping['train_0']  # For training image at index 0

# Or use CSV
import pandas as pd
df = pd.read_csv('train_image_pseudo_labels.csv')
pseudo_label = df[df['image_path'] == 'train_0']['pseudo_label_class_index'].values[0]
""")
    
    print("\nAll mappings created successfully!")
    print(f"Files saved in: {pseudo_dir}")

if __name__ == "__main__":
    create_mappings()
