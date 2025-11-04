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
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for CIFAR100 dataset')
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
    parser.add_argument('--num_clusters', type=int, default=100,
                        help='Number of clusters (k=100 for CIFAR100)')
    
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
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"temi_cifar100_{args.num_clusters}clusters_{timestamp}"
    
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
    print("Loading CIFAR100 dataset...")
    train_loader, test_loader = create_data_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
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
        train_labels.numpy(),
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
        test_labels.numpy(),
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
    
    # Save predictions
    predictions_path = experiment_dir / "predictions.npz"
    import numpy as np
    np.savez(
        predictions_path,
        train_predictions=train_predictions,
        train_labels=train_labels.numpy(),
        test_predictions=test_predictions,
        test_labels=test_labels.numpy()
    )
    print(f"Predictions saved to {predictions_path}")
    
    # Generate visualizations if requested
    if args.plot_clusters:
        print("\n" + "="*60)
        print("Step 4: Generating Visualizations")
        print("="*60)
        
        viz_dir = experiment_dir / "visualizations"
        
        # Visualize training set
        visualize_clustering_results(
            features=train_features.numpy(),
            labels=train_labels.numpy(),
            predictions=train_predictions,
            num_clusters=args.num_clusters,
            output_dir=str(viz_dir),
            dataset_name="Training",
            method=args.viz_method,
            show_plots=args.show_plots
        )
        
        # Visualize test set
        visualize_clustering_results(
            features=test_features.numpy(),
            labels=test_labels.numpy(),
            predictions=test_predictions,
            num_clusters=args.num_clusters,
            output_dir=str(viz_dir),
            dataset_name="Test",
            method=args.viz_method,
            show_plots=args.show_plots
        )
        
        print(f"\nVisualizations saved to {viz_dir}")


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
    print("\n" + "="*60)
    print(f"TEMI Clustering on CIFAR100 with {model_name}")
    print("="*60)
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
    
    print("\n" + "="*60)
    print("Clustering pipeline completed successfully!")
    print(f"All results saved to: {experiment_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
