"""
Standalone script to generate pseudo labels from existing TEMI clustering results.

This script loads pre-extracted features and clustering results, then generates
pseudo labels by mapping clusters to actual labels based on the k nearest samples
to each cluster center. It also creates visualizations of the mappings.
"""

import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path

from src.feature_extractor import DINOv2FeatureExtractor
from src.clip_feature_extractor import CLIPFeatureExtractor
from src.temi_clustering import TEMIClusterer
from src.data_loader import create_data_loaders
from src.pseudo_labeling import (
    generate_pseudo_labels,
    print_cluster_mapping_summary,
    visualize_cluster_mapping,
    map_clusters_to_labels
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate pseudo labels for TEMI clustering results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Path to experiment directory (e.g., ./results/experiment_name)')
    
    # Pseudo-labeling arguments
    parser.add_argument('--k_samples', type=int, default=10,
                        help='Number of nearest samples to cluster center for label assignment')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of cluster-to-label mappings')
    parser.add_argument('--max_clusters_viz', type=int, default=20,
                        help='Maximum number of clusters to visualize')
    parser.add_argument('--samples_per_cluster', type=int, default=5,
                        help='Number of samples to show per cluster in visualization')
    
    # Data arguments (needed for visualization)
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset (needed for visualization)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')
    
    return parser.parse_args()


def load_experiment_config(experiment_dir: Path) -> dict:
    """Load experiment configuration from JSON file."""
    config_path = experiment_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def load_features_and_labels(experiment_dir: Path, config: dict):
    """Load pre-extracted features and labels."""
    features_dir = experiment_dir / "features"
    
    # Determine which feature loader to use
    model_type = config.get('model_type', 'dinov2')
    
    # Load training features
    train_path = features_dir / "train_features.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")
    
    print(f"Loading training features from {train_path}...")
    if model_type == 'clip':
        train_features, train_labels, _ = CLIPFeatureExtractor.load_features(str(train_path))
    else:
        train_features, train_labels, _ = DINOv2FeatureExtractor.load_features(str(train_path))
    
    # Load test features (optional)
    test_path = features_dir / "test_features.pt"
    test_features, test_labels = None, None
    
    if test_path.exists():
        print(f"Loading test features from {test_path}...")
        if model_type == 'clip':
            test_features, test_labels, _ = CLIPFeatureExtractor.load_features(str(test_path))
        else:
            test_features, test_labels, _ = DINOv2FeatureExtractor.load_features(str(test_path))
    
    return train_features, train_labels, test_features, test_labels


def load_clustering_model(experiment_dir: Path, config: dict, device: str):
    """Load trained clustering model."""
    checkpoint_path = experiment_dir / "final_checkpoint.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading clustering model from {checkpoint_path}...")
    
    # Get model parameters from config
    feature_dim = config.get('feature_dim', 768)  # Default for DINOv2-base
    num_clusters = config.get('num_clusters', 100)
    hidden_dim = config.get('hidden_dim', 2048)
    projection_dim = config.get('projection_dim', 256)
    learning_rate = config.get('learning_rate', 0.001)
    temperature = config.get('temperature', 0.1)
    
    # Create clusterer
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        learning_rate=learning_rate,
        temperature=temperature
    )
    
    # Load checkpoint
    clusterer.load_checkpoint(str(checkpoint_path))
    
    return clusterer


def load_dataset_for_visualization(config: dict, data_root: str):
    """Load dataset for visualization."""
    dataset_name = config.get('dataset', 'cifar100')
    batch_size = config.get('batch_size', 256)
    num_workers = config.get('num_workers', 4)
    
    print(f"Loading {dataset_name} dataset for visualization...")
    train_loader, test_loader = create_data_loaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name
    )
    
    # Get all training images and labels (keep as tensors)
    all_images = []
    all_labels = []
    
    for images, labels in train_loader:
        all_images.append(images)
        all_labels.append(labels)
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Get class names if available
    class_names = None
    if dataset_name.lower() == 'cifar100':
        from torchvision.datasets import CIFAR100
        cifar = CIFAR100(root=data_root, train=True, download=False)
        class_names = cifar.classes
    elif dataset_name.lower() == 'imagenet':
        # ImageNet has 1000 classes - using indices is fine
        class_names = [f"Class_{i}" for i in range(1000)]
    
    return all_images, all_labels, class_names


def main():
    """Main function."""
    args = parse_arguments()
    
    experiment_dir = Path(args.experiment_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    print("\n" + "="*80)
    print("Pseudo Label Generation for TEMI Clustering")
    print("="*80)
    print(f"\nExperiment directory: {experiment_dir}")
    print(f"K samples for labeling: {args.k_samples}")
    
    # Step 1: Load experiment configuration
    print("\n" + "-"*80)
    print("Step 1: Loading experiment configuration")
    print("-"*80)
    config = load_experiment_config(experiment_dir)
    print(f"Dataset: {config.get('dataset', 'cifar100')}")
    print(f"Number of clusters: {config.get('num_clusters', 100)}")
    print(f"Model type: {config.get('model_type', 'dinov2')}")
    
    # Step 2: Load features and labels
    print("\n" + "-"*80)
    print("Step 2: Loading features and labels")
    print("-"*80)
    train_features, train_labels, test_features, test_labels = load_features_and_labels(
        experiment_dir, config
    )
    print(f"Training samples: {len(train_features)}")
    if test_features is not None:
        print(f"Test samples: {len(test_features)}")
    
    # Step 3: Load clustering model
    print("\n" + "-"*80)
    print("Step 3: Loading clustering model")
    print("-"*80)
    clusterer = load_clustering_model(experiment_dir, config, args.device)
    
    # Step 4: Get cluster assignments
    print("\n" + "-"*80)
    print("Step 4: Computing cluster assignments")
    print("-"*80)
    print("Predicting cluster assignments for training data...")
    train_assignments = clusterer.predict(train_features)
    
    test_assignments = None
    if test_features is not None:
        print("Predicting cluster assignments for test data...")
        test_assignments = clusterer.predict(test_features)
    
    # Step 5: Generate pseudo labels
    print("\n" + "-"*80)
    print("Step 5: Generating pseudo labels")
    print("-"*80)
    
    # Generate for training data
    print("\n>>> Training Data <<<")
    train_pseudo_labels, train_cluster_to_label, train_k_nearest, train_confidence, train_cluster_confidence = generate_pseudo_labels(
        features=train_features,
        cluster_assignments=train_assignments,
        true_labels=train_labels,
        cluster_centers=clusterer.cluster_centers,
        k=args.k_samples,
        verbose=True,
        return_confidence=True
    )
    
    # Print detailed summary for training
    class_names = None
    if config.get('dataset', 'cifar100').lower() == 'cifar100':
        try:
            from torchvision.datasets import CIFAR100
            cifar = CIFAR100(root=args.data_root, train=True, download=False)
            class_names = cifar.classes
        except Exception as e:
            print(f"Note: Could not load class names: {e}")
            pass
    
    print_cluster_mapping_summary(
        train_cluster_to_label,
        train_assignments,
        train_labels,
        class_names,
        cluster_to_confidence=train_cluster_confidence,
        confidence_scores=train_confidence
    )
    
    # Generate for test data if available
    test_confidence = None
    test_cluster_confidence = None
    if test_features is not None:
        print("\n>>> Test Data <<<")
        test_pseudo_labels, test_cluster_to_label, test_k_nearest, test_confidence, test_cluster_confidence = generate_pseudo_labels(
            features=test_features,
            cluster_assignments=test_assignments,
            true_labels=test_labels,
            cluster_centers=clusterer.cluster_centers,
            k=args.k_samples,
            verbose=True,
            return_confidence=True
        )
        
        print_cluster_mapping_summary(
            test_cluster_to_label,
            test_assignments,
            test_labels,
            class_names,
            cluster_to_confidence=test_cluster_confidence,
            confidence_scores=test_confidence
        )
    
    # Step 6: Save results
    print("\n" + "-"*80)
    print("Step 6: Saving results")
    print("-"*80)
    
    pseudo_labels_dir = experiment_dir / "pseudo_labels"
    pseudo_labels_dir.mkdir(exist_ok=True)
    
    # Save pseudo labels and mappings with confidence scores
    results = {
        'k_samples': args.k_samples,
        'train_pseudo_labels': train_pseudo_labels.tolist(),
        'train_cluster_to_label': {int(k): int(v) for k, v in train_cluster_to_label.items()},
        'train_cluster_to_confidence': {int(k): float(v) for k, v in train_cluster_confidence.items()},
        'train_confidence_scores': train_confidence.tolist(),
        'train_k_nearest_indices': {int(k): v.tolist() for k, v in train_k_nearest.items()}
    }
    
    if test_features is not None:
        results['test_pseudo_labels'] = test_pseudo_labels.tolist()
        results['test_cluster_to_label'] = {int(k): int(v) for k, v in test_cluster_to_label.items()}
        results['test_cluster_to_confidence'] = {int(k): float(v) for k, v in test_cluster_confidence.items()}
        results['test_confidence_scores'] = test_confidence.tolist()
        results['test_k_nearest_indices'] = {int(k): v.tolist() for k, v in test_k_nearest.items()}
    
    results_path = pseudo_labels_dir / f"pseudo_labels_k{args.k_samples}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    
    # Step 7: Generate visualization if requested
    if args.visualize:
        print("\n" + "-"*80)
        print("Step 7: Generating visualization")
        print("-"*80)
        
        try:
            # Load dataset images
            all_images, all_labels, class_names = load_dataset_for_visualization(
                config, args.data_root
            )
            
            # Generate visualization for training data
            viz_path = pseudo_labels_dir / f"cluster_mapping_k{args.k_samples}.png"
            print(f"Creating visualization with {args.max_clusters_viz} clusters...")
            
            visualize_cluster_mapping(
                images=all_images,
                true_labels=train_labels,
                cluster_assignments=train_assignments,
                cluster_to_label=train_cluster_to_label,
                k_nearest_indices=train_k_nearest,
                save_path=str(viz_path),
                class_names=class_names,
                max_clusters_to_show=args.max_clusters_viz,
                samples_per_cluster=args.samples_per_cluster
            )
            
            print(f"Visualization saved to {viz_path}")
            
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
            print("This is not critical - pseudo labels were still generated successfully.")
    
    print("\n" + "="*80)
    print("Pseudo Label Generation Complete!")
    print("="*80)
    print(f"\nResults saved to: {pseudo_labels_dir}")
    print(f"You can find:")
    print(f"  - Pseudo labels and mappings: pseudo_labels_k{args.k_samples}.json")
    if args.visualize:
        print(f"  - Visualization: cluster_mapping_k{args.k_samples}.png")
    print()


if __name__ == "__main__":
    main()
