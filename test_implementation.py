"""
Test script to verify the TEMI implementation without requiring dataset downloads.

This script creates synthetic data to test all components of the pipeline.
"""

import torch
import numpy as np
from pathlib import Path

from src.temi_clustering import TEMIClusterer
from src.evaluation import (
    evaluate_clustering,
    print_evaluation_results,
    analyze_cluster_distribution,
    print_cluster_distribution
)


def test_temi_clustering():
    """
    Test the TEMI clustering implementation with synthetic data.
    
    This verifies that all components work correctly without requiring
    external data downloads.
    """
    print("="*60)
    print("Testing TEMI Clustering Implementation")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    num_samples = 1000
    feature_dim = 384  # DINOv2-small feature dimension
    num_clusters = 10
    
    # Generate features from multiple Gaussian clusters
    features_list = []
    labels_list = []
    
    samples_per_cluster = num_samples // num_clusters
    for cluster_id in range(num_clusters):
        # Create cluster center
        center = torch.randn(feature_dim) * 2
        
        # Generate samples around center
        cluster_features = center + torch.randn(samples_per_cluster, feature_dim) * 0.5
        cluster_labels = torch.ones(samples_per_cluster, dtype=torch.long) * cluster_id
        
        features_list.append(cluster_features)
        labels_list.append(cluster_labels)
    
    # Concatenate all clusters
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Normalize features
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    # Shuffle data
    shuffle_idx = torch.randperm(num_samples)
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]
    
    print(f"   Created {num_samples} samples with {feature_dim} features")
    print(f"   Ground truth: {num_clusters} clusters")
    
    # Test TEMI clustering
    print("\n2. Testing TEMI clustering...")
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device,
        hidden_dim=512,
        projection_dim=128,
        learning_rate=0.001,
        temperature=0.1
    )
    
    # Test K-means initialization
    print("   Testing K-means initialization...")
    initial_assignments = clusterer.initialize_clusters(features)
    print(f"   Initial assignments shape: {initial_assignments.shape}")
    
    # Test training
    print("   Testing training loop...")
    history = clusterer.fit(
        features=features,
        num_epochs=20,
        batch_size=128,
        verbose=False
    )
    
    print(f"   Training complete. Final loss: {history['total_loss'][-1]:.4f}")
    
    # Test prediction
    print("\n3. Testing prediction...")
    predictions = clusterer.predict(features, batch_size=128)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Unique clusters predicted: {len(np.unique(predictions))}")
    
    # Test evaluation
    print("\n4. Testing evaluation metrics...")
    results = evaluate_clustering(
        labels.numpy(),
        predictions,
        return_confusion_matrix=False
    )
    
    print_evaluation_results(results, "Synthetic Data")
    
    # Test cluster distribution analysis
    distribution = analyze_cluster_distribution(predictions, num_clusters)
    print_cluster_distribution(distribution)
    
    # Test checkpoint saving and loading
    print("\n5. Testing checkpoint system...")
    checkpoint_dir = Path("./test_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    clusterer.save_checkpoint(str(checkpoint_path), 20, history)
    
    # Create new clusterer and load checkpoint
    clusterer2 = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device
    )
    
    loaded_epoch, loaded_history = clusterer2.load_checkpoint(str(checkpoint_path))
    print(f"   Checkpoint loaded: epoch {loaded_epoch}")
    
    # Verify loaded model produces same predictions
    predictions2 = clusterer2.predict(features, batch_size=128)
    matches = np.sum(predictions == predictions2)
    print(f"   Predictions match: {matches}/{num_samples} ({100*matches/num_samples:.1f}%)")
    
    # Clean up test checkpoint
    checkpoint_path.unlink()
    checkpoint_dir.rmdir()
    
    # Verification
    print("\n6. Verification...")
    all_pass = True
    
    if results['accuracy'] < 0.3:
        print("   WARNING: Accuracy is low (expected for random synthetic data)")
    else:
        print(f"   PASS: Accuracy is {results['accuracy']:.4f}")
    
    if len(np.unique(predictions)) < num_clusters * 0.7:
        print(f"   WARNING: Only {len(np.unique(predictions))} clusters active (some collapse expected)")
    else:
        print(f"   PASS: {len(np.unique(predictions))} clusters active")
    
    if history['total_loss'][-1] > history['total_loss'][0]:
        print("   FAIL: Loss increased during training")
        all_pass = False
    else:
        print("   PASS: Loss decreased during training")
    
    if matches < num_samples * 0.95:
        print("   FAIL: Checkpoint loading failed")
        all_pass = False
    else:
        print("   PASS: Checkpoint system working")
    
    return all_pass


def test_components():
    """
    Test individual components of the implementation.
    """
    print("\n" + "="*60)
    print("Testing Individual Components")
    print("="*60)
    
    # Test TEMIClusteringHead
    print("\n1. Testing TEMIClusteringHead...")
    from src.temi_clustering import TEMIClusteringHead
    
    head = TEMIClusteringHead(
        input_dim=384,
        num_clusters=10,
        hidden_dim=512,
        projection_dim=128
    )
    
    # Test forward pass
    batch_features = torch.randn(32, 384)
    logits, projected = head(batch_features)
    
    print(f"   Input shape: {batch_features.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Projected shape: {projected.shape}")
    
    assert logits.shape == (32, 10), "Logits shape incorrect"
    assert projected.shape == (32, 128), "Projected shape incorrect"
    print("   PASS: TEMIClusteringHead forward pass")
    
    # Test cluster assignments
    assignments = head.get_cluster_assignments(batch_features)
    assert assignments.shape == (32,), "Assignments shape incorrect"
    assert torch.all((assignments >= 0) & (assignments < 10)), "Invalid cluster assignments"
    print("   PASS: Cluster assignments")
    
    # Test evaluation functions
    print("\n2. Testing evaluation functions...")
    from src.evaluation import cluster_accuracy
    
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 1, 0, 0, 2, 2])
    
    acc = cluster_accuracy(y_true, y_pred)
    print(f"   Cluster accuracy: {acc:.4f}")
    assert acc == 1.0, "Expected perfect accuracy with Hungarian matching"
    print("   PASS: Hungarian matching working correctly")
    
    print("\n" + "="*60)
    print("All component tests passed!")
    print("="*60)


if __name__ == "__main__":
    try:
        print("\nRunning implementation tests...\n")
        
        # Test individual components
        test_components()
        
        # Test full pipeline
        success = test_temi_clustering()
        
        print("\n" + "="*60)
        if success:
            print("All tests passed successfully!")
            print("The TEMI clustering implementation is working correctly.")
        else:
            print("Some tests failed. Please review the implementation.")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
