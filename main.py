"""
Main script for TEMI clustering on CIFAR100 using DINOv2.

This script orchestrates the complete pipeline:
1. Load CIFAR100 dataset
2. Extract features using DINOv2
3. Train TEMI clustering model
4. Evaluate and save results

The script includes checkpoint support for resuming from any stage.
"""

import os
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.data_loader import create_data_loaders
from src.feature_extractor import DINOv2FeatureExtractor
from src.clip_feature_extractor import CLIPFeatureExtractor
from src.temi_clustering import TEMIClusterer
from src.evaluation import (
    evaluate_clustering,
    print_evaluation_results,
    analyze_cluster_distribution,
    print_cluster_distribution
)
from src.visualization import visualize_clustering_results
from src.pseudo_labeling import (
    generate_pseudo_labels,
    apply_pseudo_labels,
    print_cluster_mapping_summary,
    visualize_cluster_mapping,
    map_clusters_to_labels
)
from src.dataset_distillation import DatasetDistiller


def parse_arguments():
    """
    Parse command line arguments for the clustering script.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description='TEMI Clustering on CIFAR100 with DINOv2, DINOv3, or CLIP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'],
                        help='Dataset to use (cifar10, cifar100, imagenet, or tiny-imagenet)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for dataset storage')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='dinov2',
                        choices=['dinov2', 'clip'],
                        help='Type of feature extractor to use (dinov2 for DINOv2/DINOv3, clip for CLIP)')
    parser.add_argument('--dinov2_model', type=str, default='facebook/dinov2-base',
                        help='DINOv2/DINOv3 model variant to use (e.g., facebook/dinov2-base, facebook/dinov2-large, or any HuggingFace DINO model)')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                        help='CLIP model variant to use (e.g., openai/clip-vit-base-patch32, openai/clip-vit-large-patch14)')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='Number of clusters (default: 100 for CIFAR100, 1000 for ImageNet)')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs for TEMI')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate for clustering head (increased from 0.001 for SGD)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for softmax in clustering')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='Hidden dimension in clustering head')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Projection dimension in clustering head')
    parser.add_argument('--use_sinkhorn', action='store_true', default=True,
                        help='Use Sinkhorn-Knopp normalization to prevent cluster collapse')
    parser.add_argument('--no_sinkhorn', dest='use_sinkhorn', action='store_false',
                        help='Disable Sinkhorn-Knopp normalization')
    
    # Checkpoint and resume arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_features', action='store_true',
                        help='Save extracted features to disk')
    parser.add_argument('--load_features', type=str, default=None,
                        help='Path to pre-extracted features')
    
    # Output arguments
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for this experiment (auto-generated if not provided)')
    
    # Visualization arguments
    parser.add_argument('--plot_clusters', action='store_true',
                        help='Generate cluster visualizations (t-SNE/UMAP plots)')
    parser.add_argument('--viz_method', type=str, default='tsne',
                        choices=['tsne', 'umap'],
                        help='Dimensionality reduction method for visualization')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively (in addition to saving them)')
    
    # Pseudo-labeling arguments
    parser.add_argument('--generate_pseudo_labels', action='store_true',
                        help='Generate pseudo labels by mapping clusters to actual labels')
    parser.add_argument('--k_samples', type=int, default=10,
                        help='Number of nearest samples to cluster center for pseudo label assignment')
    parser.add_argument('--visualize_mapping', action='store_true',
                        help='Generate visualization of cluster-to-label mappings')
    parser.add_argument('--max_clusters_viz', type=int, default=20,
                        help='Maximum number of clusters to visualize in mapping')
    parser.add_argument('--samples_per_cluster', type=int, default=5,
                        help='Number of samples to show per cluster in mapping visualization')
    
    # Dataset distillation arguments
    parser.add_argument('--distill_dataset', action='store_true',
                        help='Perform dataset distillation using pseudo labels')
    parser.add_argument('--images_per_class', type=int, default=10,
                        help='Number of synthetic images per class for distillation')
    parser.add_argument('--distill_epochs', type=int, default=100,
                        help='Number of distillation epochs')
    parser.add_argument('--distill_lr', type=float, default=0.1,
                        help='Learning rate for distilled image optimization')
    parser.add_argument('--inner_epochs', type=int, default=10,
                        help='Number of inner training epochs per distillation step')
    parser.add_argument('--evaluate_distilled', action='store_true',
                        help='Evaluate the quality of distilled data')
    parser.add_argument('--selection_strategy', type=str, default='random',
                        choices=['random', 'margin'],
                        help='Strategy for initializing synthetic data (default: random).')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for computation')
    
    return parser.parse_args()


def setup_directories(args):
    """
    Create necessary directories for checkpoints and results.
    
    Args:
        args: Parsed command line arguments
    """
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Create results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Set default number of clusters based on dataset if not specified
    if args.num_clusters is None:
        if args.dataset.lower() == 'cifar10':
            args.num_clusters = 10
        elif args.dataset.lower() == 'cifar100':
            args.num_clusters = 100
        elif args.dataset.lower() == 'imagenet':
            args.num_clusters = 1000
        elif args.dataset.lower() == 'tiny-imagenet':
            args.num_clusters = 200
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"temi_{args.dataset}_{args.num_clusters}clusters_{timestamp}"
    
    # Create experiment subdirectory in results
    experiment_dir = Path(args.results_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def save_config(args, experiment_dir):
    """
    Save experiment configuration to JSON file.
    
    Args:
        args: Parsed command line arguments
        experiment_dir: Directory for this experiment
    """
    config_path = experiment_dir / "config.json"
    config = vars(args)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")


def extract_features(args, experiment_dir):
    """
    Extract features from CIFAR100 using DINOv2/DINOv3 or CLIP.
    
    This function handles both extracting new features and loading
    pre-extracted features from disk.
    
    Args:
        args: Parsed command line arguments
        experiment_dir: Directory for this experiment
        
    Returns:
        Tuple of (train_features, train_labels, test_features, test_labels)
    """
    # Check if we should load pre-extracted features
    if args.load_features:
        print(f"Loading pre-extracted features from {args.load_features}")
        # Use the appropriate feature loader based on model type
        if args.model_type == 'clip':
            train_data = CLIPFeatureExtractor.load_features(args.load_features + "_train.pt")
            test_data = CLIPFeatureExtractor.load_features(args.load_features + "_test.pt")
        else:
            train_data = DINOv2FeatureExtractor.load_features(args.load_features + "_train.pt")
            test_data = DINOv2FeatureExtractor.load_features(args.load_features + "_test.pt")
        
        train_features, train_labels, _ = train_data
        test_features, test_labels, _ = test_data
        
        return train_features, train_labels, test_features, test_labels
    
    # Extract features using selected model
    model_display_name = "CLIP" if args.model_type == 'clip' else "DINOv2/DINOv3"
    print("\n" + "="*60)
    print(f"Step 1: Feature Extraction with {model_display_name}")
    print("="*60)
    
    # Create data loaders
    dataset_display_name = args.dataset.upper()
    print(f"Loading {dataset_display_name} dataset...")
    train_loader, test_loader = create_data_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_name=args.dataset
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize feature extractor based on model type
    if args.model_type == 'clip':
        feature_extractor = CLIPFeatureExtractor(
            model_name=args.clip_model,
            device=args.device
        )
    else:
        feature_extractor = DINOv2FeatureExtractor(
            model_name=args.dinov2_model,
            device=args.device
        )
    
    # Extract training features
    print("\nExtracting training features...")
    train_features, train_labels = feature_extractor.extract_features(
        train_loader,
        return_labels=True
    )
    
    # Extract test features
    print("\nExtracting test features...")
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    
    # Save features if requested
    if args.save_features:
        features_dir = experiment_dir / "features"
        features_dir.mkdir(exist_ok=True)
        
        train_path = features_dir / "train_features.pt"
        test_path = features_dir / "test_features.pt"
        
        feature_extractor.save_features(train_features, train_labels, str(train_path))
        feature_extractor.save_features(test_features, test_labels, str(test_path))
    
    return train_features, train_labels, test_features, test_labels


def train_temi_clustering(args, train_features, experiment_dir):
    """
    Train the TEMI clustering model.
    
    Args:
        args: Parsed command line arguments
        train_features: Training features from DINOv2
        experiment_dir: Directory for this experiment
        
    Returns:
        Trained TEMIClusterer object
    """
    print("\n" + "="*60)
    print("Step 2: TEMI Clustering")
    print("="*60)
    
    # Get feature dimension
    feature_dim = train_features.shape[1]
    
    # Initialize TEMI clusterer
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=args.num_clusters,
        device=args.device,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        use_sinkhorn=args.use_sinkhorn
    )
    
    # Check if resuming from checkpoint
    start_epoch = 0
    history = None
    
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, history = clusterer.load_checkpoint(args.resume_from)
        print(f"Resuming from epoch {start_epoch}")
    
    # Train the clustering model
    if start_epoch < args.num_epochs:
        remaining_epochs = args.num_epochs - start_epoch
        
        new_history = clusterer.fit(
            features=train_features,
            num_epochs=remaining_epochs,
            batch_size=args.batch_size,
            verbose=True
        )
        
        # Merge histories if resuming
        if history is not None:
            for key in history.keys():
                history[key].extend(new_history[key])
        else:
            history = new_history
    
    # Save final checkpoint
    checkpoint_path = experiment_dir / "final_checkpoint.pt"
    clusterer.save_checkpoint(str(checkpoint_path), args.num_epochs, history)
    
    return clusterer


def evaluate_results(args, clusterer, train_features, train_labels, 
                     test_features, test_labels, experiment_dir):
    """
    Evaluate clustering results on both training and test sets.
    
    Args:
        args: Parsed command line arguments
        clusterer: Trained TEMI clusterer
        train_features: Training features
        train_labels: Training ground truth labels
        test_features: Test features
        test_labels: Test ground truth labels
        experiment_dir: Directory for this experiment
    """
    print("\n" + "="*60)
    print("Step 3: Evaluation")
    print("="*60)
    
    # Predict clusters for training set
    print("\nPredicting clusters for training set...")
    train_predictions = clusterer.predict(train_features, batch_size=args.batch_size)
    
    # Predict clusters for test set
    print("Predicting clusters for test set...")
    test_predictions = clusterer.predict(test_features, batch_size=args.batch_size)
    
    # Evaluate training set
    print("\nEvaluating training set...")
    train_results = evaluate_clustering(
        train_labels,
        train_predictions,
        return_confusion_matrix=False
    )
    print_evaluation_results(train_results, "Training Set")
    
    # Analyze training cluster distribution
    train_distribution = analyze_cluster_distribution(train_predictions, args.num_clusters)
    print_cluster_distribution(train_distribution)
    
    # Evaluate test set
    print("\nEvaluating test set...")
    test_results = evaluate_clustering(
        test_labels,
        test_predictions,
        return_confusion_matrix=False
    )
    print_evaluation_results(test_results, "Test Set")
    
    # Analyze test cluster distribution
    test_distribution = analyze_cluster_distribution(test_predictions, args.num_clusters)
    print_cluster_distribution(test_distribution)
    
    # Save results to JSON
    results = {
        'experiment_name': args.experiment_name,
        'num_clusters': args.num_clusters,
        'training': {
            'accuracy': float(train_results['accuracy']),
            'nmi': float(train_results['nmi']),
            'ari': float(train_results['ari']),
            'num_samples': len(train_labels),
            'num_active_clusters': train_distribution['num_active_clusters'],
            'num_empty_clusters': train_distribution['num_empty_clusters']
        },
        'test': {
            'accuracy': float(test_results['accuracy']),
            'nmi': float(test_results['nmi']),
            'ari': float(test_results['ari']),
            'num_samples': len(test_labels),
            'num_active_clusters': test_distribution['num_active_clusters'],
            'num_empty_clusters': test_distribution['num_empty_clusters']
        }
    }
    
    results_path = experiment_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")
    
    # Save predictions (convert to numpy only for saving to file)
    predictions_path = experiment_dir / "predictions.npz"
    import numpy as np
    np.savez(
        predictions_path,
        train_predictions=train_predictions.cpu().numpy() if isinstance(train_predictions, torch.Tensor) else train_predictions,
        train_labels=train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else train_labels,
        test_predictions=test_predictions.cpu().numpy() if isinstance(test_predictions, torch.Tensor) else test_predictions,
        test_labels=test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels
    )
    print(f"Predictions saved to {predictions_path}")
    
    # Generate visualizations if requested
    if args.plot_clusters:
        print("\n" + "="*60)
        print("Step 4: Generating Visualizations")
        print("="*60)
        
        viz_dir = experiment_dir / "visualizations"
        
        # Visualize training set (pass torch tensors directly)
        visualize_clustering_results(
            features=train_features,
            labels=train_labels,
            predictions=train_predictions,
            num_clusters=args.num_clusters,
            output_dir=str(viz_dir),
            dataset_name="Training",
            method=args.viz_method,
            show_plots=args.show_plots
        )
        
        # Visualize test set (pass torch tensors directly)
        visualize_clustering_results(
            features=test_features,
            labels=test_labels,
            predictions=test_predictions,
            num_clusters=args.num_clusters,
            output_dir=str(viz_dir),
            dataset_name="Test",
            method=args.viz_method,
            show_plots=args.show_plots
        )
        
        print(f"\nVisualizations saved to {viz_dir}")


def generate_and_save_pseudo_labels(
    args,
    clusterer,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    experiment_dir: Path
):
    """
    Generate pseudo labels for clusters and save results.
    
    Args:
        args: Command line arguments
        clusterer: Trained TEMI clusterer
        train_features: Training features
        train_labels: Training labels
        test_features: Test features
        test_labels: Test labels
        experiment_dir: Experiment directory
    """
    print("\n" + "="*60)
    print("Step 4: Pseudo Label Generation")
    print("="*60)
    
    # Get cluster assignments
    print("\nComputing cluster assignments...")
    train_predictions = clusterer.predict(train_features)
    test_predictions = clusterer.predict(test_features)
    
    # Get class names if available
    class_names = None
    if args.dataset.lower() == 'cifar10':
        try:
            from torchvision.datasets import CIFAR10
            cifar = CIFAR10(root=args.data_root, train=True, download=False)
            class_names = cifar.classes
        except Exception as e:
            print(f"Could not load class names: {e}")
    elif args.dataset.lower() == 'cifar100':
        try:
            from torchvision.datasets import CIFAR100
            cifar = CIFAR100(root=args.data_root, train=True, download=False)
            class_names = cifar.classes
        except Exception as e:
            print(f"Could not load class names: {e}")
    elif args.dataset.lower() == 'imagenet':
        # ImageNet has 1000 classes
        class_names = [f"Class_{i}" for i in range(1000)]
    elif args.dataset.lower() == 'tiny-imagenet':
        # Tiny ImageNet has 200 classes
        class_names = [f"Class_{i}" for i in range(200)]
    
    # Generate pseudo labels for training set
    print("\n>>> Training Set <<<")
    train_pseudo_labels, train_cluster_to_label, train_k_nearest, train_confidence, train_cluster_confidence = generate_pseudo_labels(
        features=train_features,
        cluster_assignments=train_predictions,
        true_labels=train_labels,
        cluster_centers=clusterer.cluster_centers,
        k=args.k_samples,
        verbose=True,
        return_confidence=True
    )
    
    print_cluster_mapping_summary(
        train_cluster_to_label,
        train_predictions,
        train_labels,
        class_names,
        cluster_to_confidence=train_cluster_confidence,
        confidence_scores=train_confidence
    )
    
    # Apply TRAINING cluster mapping to test set (NO data leakage)
    print("\n>>> Test Set (using training cluster-to-label mapping) <<<")
    test_pseudo_labels = apply_pseudo_labels(test_predictions, train_cluster_to_label)
    
    # Compute test accuracy using training mapping
    test_accuracy = (test_pseudo_labels == test_labels).float().mean().item() * 100
    print(f"Test pseudo-label accuracy (with training mapping): {test_accuracy:.2f}%")
    print(f"Using cluster-to-label mapping from training set (no test label leakage)")
    
    # Note: We don't create a separate test_cluster_to_label mapping
    # to avoid any data leakage. The training mapping is used for distillation.
    
    # Save pseudo labels (convert to numpy/list only for JSON serialization)
    pseudo_labels_dir = experiment_dir / "pseudo_labels"
    pseudo_labels_dir.mkdir(exist_ok=True)
    
    results = {
        'k_samples': args.k_samples,
        'train_pseudo_labels': train_pseudo_labels.cpu().tolist() if isinstance(train_pseudo_labels, torch.Tensor) else train_pseudo_labels.tolist(),
        'train_cluster_to_label': {int(k): int(v) for k, v in train_cluster_to_label.items()},
        'train_cluster_to_confidence': {int(k): float(v) for k, v in train_cluster_confidence.items()},
        'train_confidence_scores': train_confidence.cpu().tolist() if isinstance(train_confidence, torch.Tensor) else train_confidence.tolist() if train_confidence is not None else None,
        'train_k_nearest_indices': {int(k): (v.cpu().tolist() if isinstance(v, torch.Tensor) else v.tolist()) for k, v in train_k_nearest.items()},
        'test_pseudo_labels': test_pseudo_labels.cpu().tolist() if isinstance(test_pseudo_labels, torch.Tensor) else test_pseudo_labels.tolist(),
        # Note: test set uses the same cluster_to_label mapping as training (no separate mapping to avoid data leakage)
        'note': 'Test pseudo labels generated using training cluster-to-label mapping to prevent data leakage'
    }
    
    results_path = pseudo_labels_dir / f"pseudo_labels_k{args.k_samples}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nPseudo labels saved to {results_path}")
    
    # Generate visualization if requested
    if args.visualize_mapping:
        print("\nGenerating cluster-to-label mapping visualization...")
        
        try:
            # Load dataset for visualization
            print("Loading dataset images for visualization...")
            train_loader, _ = create_data_loaders(
                root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_name=args.dataset
            )
            
            # Get all training images (keep as torch tensor until final visualization step)
            all_images = []
            for images, _ in train_loader:
                all_images.append(images)
            all_images = torch.cat(all_images, dim=0)
            
            # Generate visualization
            viz_path = pseudo_labels_dir / f"cluster_mapping_k{args.k_samples}.png"
            visualize_cluster_mapping(
                images=all_images,
                true_labels=train_labels,
                cluster_assignments=train_predictions,
                cluster_to_label=train_cluster_to_label,
                k_nearest_indices=train_k_nearest,
                save_path=str(viz_path),
                class_names=class_names,
                max_clusters_to_show=args.max_clusters_viz,
                samples_per_cluster=args.samples_per_cluster
            )
            
            print(f"Visualization saved to {viz_path}")
            
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as e:
            print(f"Warning: Could not generate visualization: {e}")
            print("Pseudo labels were still generated successfully.")


def main():
    """
    Main function to run the complete TEMI clustering pipeline.
    
    This function coordinates all steps:
    1. Parse arguments and setup directories
    2. Extract features using DINOv2/DINOv3 or CLIP
    3. Train TEMI clustering model
    4. Evaluate and save results
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    experiment_dir = setup_directories(args)
    print(f"\nExperiment directory: {experiment_dir}")
    
    # Save configuration
    save_config(args, experiment_dir)
    
    # Print experiment configuration
    model_name = "CLIP" if args.model_type == 'clip' else "DINOv2/DINOv3"
    dataset_name = args.dataset.upper()
    print("\n" + "="*60)
    print(f"TEMI Clustering on {dataset_name} with {model_name}")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Model type: {args.model_type}")
    print(f"Number of clusters: {args.num_clusters}")
    if args.model_type == 'clip':
        print(f"CLIP model: {args.clip_model}")
    else:
        print(f"DINOv2 model: {args.dinov2_model}")
    print(f"Training epochs: {args.num_epochs}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Step 1: Extract features
    train_features, train_labels, test_features, test_labels = extract_features(
        args, experiment_dir
    )
    
    # Step 2: Train TEMI clustering
    clusterer = train_temi_clustering(args, train_features, experiment_dir)
    
    # Step 3: Evaluate results
    evaluate_results(
        args, clusterer, train_features, train_labels,
        test_features, test_labels, experiment_dir
    )
    
    # Step 4: Generate pseudo labels if requested
    if args.generate_pseudo_labels:
        generate_and_save_pseudo_labels(
            args, clusterer, train_features, train_labels,
            test_features, test_labels, experiment_dir
        )
    
    # Step 5: Perform dataset distillation if requested
    if args.distill_dataset:
        print("\n" + "="*60)
        print("Step 5: Dataset Distillation")
        print("="*60)
        
        # Ensure we have pseudo labels
        if not args.generate_pseudo_labels:
            print("Generating pseudo labels for distillation...")
            train_predictions = clusterer.predict(train_features)
            train_pseudo_labels, train_cluster_to_label, train_k_nearest, train_confidence, train_cluster_confidence = generate_pseudo_labels(
                features=train_features,
                cluster_assignments=train_predictions,
                true_labels=train_labels,
                cluster_centers=clusterer.cluster_centers,
                k=args.k_samples,
                verbose=True,
                return_confidence=True
            )
        else:
            # Load previously generated pseudo labels
            pseudo_labels_dir = experiment_dir / "pseudo_labels"
            results_path = pseudo_labels_dir / f"pseudo_labels_k{args.k_samples}.json"
            with open(results_path, 'r') as f:
                pseudo_results = json.load(f)
            train_pseudo_labels = torch.tensor(pseudo_results['train_pseudo_labels'], device=args.device)
        
        # Initialize distiller
        feature_dim = train_features.shape[1]
        distiller = DatasetDistiller(
            feature_dim=feature_dim,
            num_classes=args.num_clusters,
            images_per_class=args.images_per_class,
            device=args.device,
            learning_rate=args.learning_rate,
            distill_lr=args.distill_lr,
            distill_epochs=args.distill_epochs,
            inner_epochs=args.inner_epochs,
            batch_size=args.batch_size,
            selection_strategy=args.selection_strategy
        )
        
        # Perform distillation
        synthesized_features, synthesized_labels = distiller.distill(
            real_features=train_features,
            pseudo_labels=train_pseudo_labels,
            verbose=True
        )
        
        # Save distilled data
        distilled_dir = experiment_dir / "distilled_data"
        distilled_dir.mkdir(exist_ok=True)
        distilled_path = distilled_dir / "distilled_features.pt"
        distiller.save_distilled(str(distilled_path))
        
        # Evaluate distilled data if requested
        if args.evaluate_distilled:
            if test_labels is None:
                print("\nWarning: Cannot evaluate distilled data - test labels not available")
            else:
                print("\nEvaluating distilled data...")
                eval_results = distiller.evaluate_distilled_data(
                    real_features=train_features,
                    pseudo_labels=train_pseudo_labels,
                    test_features=test_features,
                    test_labels=test_labels,
                    num_trials=5
                )
                
                # Print evaluation results
                print("\n" + "="*60)
                print("Distillation Evaluation Results")
                print("="*60)
                print(f"Distilled test accuracy: {eval_results['distilled_test_acc']:.4f} ± {eval_results['distilled_test_std']:.4f}")
                print(f"Real test accuracy: {eval_results['real_test_acc']:.4f} ± {eval_results['real_test_std']:.4f}")
                print(f"Performance ratio: {eval_results['performance_ratio']:.4f}")
                print(f"Compression ratio: {eval_results['compression_ratio']:.4f}")
                print(f"Eval synth size: {eval_results['eval_synth_size']}")
                print(f"Real data size: {eval_results['real_data_size']}")
                print("="*60)
            print("="*60)
            
            # Save evaluation results
            eval_path = distilled_dir / "evaluation_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=4)
            print(f"\nEvaluation results saved to {eval_path}")
    
    print("\n" + "="*60)
    print("Clustering pipeline completed successfully!")
    print(f"All results saved to: {experiment_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
