"""
Example usage of the TEMI clustering implementation.

This script demonstrates how to use the TEMI clustering implementation
programmatically, without using the command-line interface.
"""

import torch
from pathlib import Path

from src.data_loader import create_data_loaders
from src.feature_extractor import DINOv2FeatureExtractor
from src.temi_clustering import TEMIClusterer
from src.evaluation import (
    evaluate_clustering,
    print_evaluation_results,
    analyze_cluster_distribution,
    print_cluster_distribution
)


def example_basic_usage():
    """
    Example 1: Basic usage with default settings.
    
    This example shows the simplest way to use TEMI clustering.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)
    
    # Configuration
    data_root = './data'
    batch_size = 256
    num_clusters = 100
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Clusters: {num_clusters}")
    print(f"  Epochs: {num_epochs}")
    
    # Step 1: Load CIFAR100 data
    print("\nStep 1: Loading CIFAR100 dataset...")
    train_loader, test_loader = create_data_loaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=4,
        image_size=224
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Step 2: Extract features with DINOv2
    print("\nStep 2: Extracting features with DINOv2...")
    feature_extractor = DINOv2FeatureExtractor(
        model_name='facebook/dinov2-base',
        device=device
    )
    
    train_features, train_labels = feature_extractor.extract_features(
        train_loader,
        return_labels=True
    )
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    print(f"  Training features shape: {train_features.shape}")
    print(f"  Test features shape: {test_features.shape}")
    
    # Step 3: Train TEMI clustering
    print("\nStep 3: Training TEMI clustering model...")
    clusterer = TEMIClusterer(
        feature_dim=train_features.shape[1],
        num_clusters=num_clusters,
        device=device,
        hidden_dim=2048,
        projection_dim=256,
        learning_rate=0.001,
        temperature=0.1
    )
    
    history = clusterer.fit(
        features=train_features,
        num_epochs=num_epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Step 4: Evaluate results
    print("\nStep 4: Evaluating clustering results...")
    
    # Predict on training set
    train_predictions = clusterer.predict(train_features, batch_size=batch_size)
    train_results = evaluate_clustering(train_labels.numpy(), train_predictions)
    print_evaluation_results(train_results, "Training Set")
    
    # Predict on test set
    test_predictions = clusterer.predict(test_features, batch_size=batch_size)
    test_results = evaluate_clustering(test_labels.numpy(), test_predictions)
    print_evaluation_results(test_results, "Test Set")
    
    print("\nExample 1 complete!")


def example_with_feature_caching():
    """
    Example 2: Using feature caching for faster experimentation.
    
    This example shows how to extract features once and reuse them.
    """
    print("\n" + "="*70)
    print("Example 2: Feature Caching")
    print("="*70)
    
    # Configuration
    data_root = './data'
    feature_cache_dir = Path('./cached_features')
    feature_cache_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if features are already cached
    train_features_path = feature_cache_dir / 'train_features.pt'
    test_features_path = feature_cache_dir / 'test_features.pt'
    
    if train_features_path.exists() and test_features_path.exists():
        print("\nLoading cached features...")
        train_features, train_labels, _ = DINOv2FeatureExtractor.load_features(
            str(train_features_path)
        )
        test_features, test_labels, _ = DINOv2FeatureExtractor.load_features(
            str(test_features_path)
        )
        print("  Features loaded from cache")
    else:
        print("\nExtracting and caching features...")
        
        # Load data
        train_loader, test_loader = create_data_loaders(
            root=data_root,
            batch_size=256,
            num_workers=4
        )
        
        # Extract features
        feature_extractor = DINOv2FeatureExtractor(
            model_name='facebook/dinov2-base',
            device=device
        )
        
        train_features, train_labels = feature_extractor.extract_features(
            train_loader,
            return_labels=True
        )
        test_features, test_labels = feature_extractor.extract_features(
            test_loader,
            return_labels=True
        )
        
        # Save to cache
        feature_extractor.save_features(
            train_features,
            train_labels,
            str(train_features_path)
        )
        feature_extractor.save_features(
            test_features,
            test_labels,
            str(test_features_path)
        )
        print("  Features cached for future use")
    
    # Now train clustering with cached features
    print("\nTraining TEMI clustering with cached features...")
    clusterer = TEMIClusterer(
        feature_dim=train_features.shape[1],
        num_clusters=100,
        device=device
    )
    
    history = clusterer.fit(
        features=train_features,
        num_epochs=50,  # Fewer epochs since we can iterate faster
        batch_size=256,
        verbose=True
    )
    
    print("\nExample 2 complete!")


def example_with_checkpointing():
    """
    Example 3: Using checkpoints for resumable training.
    
    This example shows how to save and load checkpoints.
    """
    print("\n" + "="*70)
    print("Example 3: Checkpointing")
    print("="*70)
    
    # Configuration
    checkpoint_dir = Path('./example_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / 'clustering_checkpoint.pt'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # For this example, we'll use synthetic data
    print("\nCreating synthetic data for demonstration...")
    num_samples = 1000
    feature_dim = 768
    num_clusters = 10
    
    train_features = torch.randn(num_samples, feature_dim)
    train_features = torch.nn.functional.normalize(train_features, p=2, dim=1)
    train_labels = torch.randint(0, num_clusters, (num_samples,))
    
    # Initialize clusterer
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device
    )
    
    # Train for a few epochs
    print("\nTraining for 10 epochs...")
    history = clusterer.fit(
        features=train_features,
        num_epochs=10,
        batch_size=128,
        verbose=False
    )
    
    # Save checkpoint
    print(f"\nSaving checkpoint to {checkpoint_path}")
    clusterer.save_checkpoint(str(checkpoint_path), 10, history)
    
    # Simulate resuming from checkpoint
    print("\nSimulating training interruption and resume...")
    
    # Create new clusterer and load checkpoint
    clusterer2 = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device
    )
    
    loaded_epoch, loaded_history = clusterer2.load_checkpoint(str(checkpoint_path))
    print(f"  Resumed from epoch {loaded_epoch}")
    
    # Continue training
    print("\nContinuing training for 10 more epochs...")
    new_history = clusterer2.fit(
        features=train_features,
        num_epochs=10,
        batch_size=128,
        verbose=False
    )
    
    # Merge histories
    for key in loaded_history.keys():
        loaded_history[key].extend(new_history[key])
    
    print(f"  Total training: {len(loaded_history['total_loss'])} epochs")
    
    print("\nExample 3 complete!")


def example_hyperparameter_comparison():
    """
    Example 4: Comparing different hyperparameters.
    
    This example shows how to run multiple experiments with different settings.
    """
    print("\n" + "="*70)
    print("Example 4: Hyperparameter Comparison")
    print("="*70)
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    num_samples = 1000
    feature_dim = 768
    num_clusters = 10
    
    train_features = torch.randn(num_samples, feature_dim)
    train_features = torch.nn.functional.normalize(train_features, p=2, dim=1)
    train_labels = torch.randint(0, num_clusters, (num_samples,))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different temperature values
    temperatures = [0.05, 0.1, 0.2]
    
    results = {}
    
    for temp in temperatures:
        print(f"\nTesting temperature = {temp}...")
        
        clusterer = TEMIClusterer(
            feature_dim=feature_dim,
            num_clusters=num_clusters,
            device=device,
            temperature=temp
        )
        
        history = clusterer.fit(
            features=train_features,
            num_epochs=20,
            batch_size=128,
            verbose=False
        )
        
        predictions = clusterer.predict(train_features)
        eval_results = evaluate_clustering(train_labels.numpy(), predictions)
        
        results[temp] = eval_results
        print(f"  Accuracy: {eval_results['accuracy']:.4f}")
        print(f"  NMI: {eval_results['nmi']:.4f}")
    
    # Print comparison
    print("\n" + "="*70)
    print("Hyperparameter Comparison Results")
    print("="*70)
    print(f"{'Temperature':<15} {'Accuracy':<15} {'NMI':<15} {'ARI':<15}")
    print("-"*60)
    
    for temp, res in results.items():
        print(f"{temp:<15.2f} {res['accuracy']:<15.4f} {res['nmi']:<15.4f} {res['ari']:<15.4f}")
    
    print("\nExample 4 complete!")


def main():
    """
    Run all examples.
    
    Note: Example 1 requires internet access to download CIFAR100.
    Comment it out if running in an offline environment.
    """
    print("="*70)
    print("TEMI Clustering - Example Usage")
    print("="*70)
    
    # Example 1: Basic usage (requires CIFAR100 download)
    # Uncomment to run:
    # example_basic_usage()
    
    # Example 2: Feature caching (requires CIFAR100 download)
    # Uncomment to run:
    # example_with_feature_caching()
    
    # Example 3: Checkpointing (works offline with synthetic data)
    example_with_checkpointing()
    
    # Example 4: Hyperparameter comparison (works offline with synthetic data)
    example_hyperparameter_comparison()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
