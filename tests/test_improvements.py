"""
Test script to verify the improvements to TEMI clustering.

This script tests the updated implementation with synthetic data
to ensure cluster collapse is prevented and all mechanisms work correctly.
"""

import torch
import numpy as np
from src.temi_clustering import TEMIClusterer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import Counter

def create_synthetic_clusters(num_clusters=10, samples_per_cluster=50, feature_dim=768, seed=42):
    """Create synthetic clustered data for testing."""
    # Configuration for synthetic data generation
    CLUSTER_SEPARATION = 5.0  # Standard deviations between cluster centers
    
    np.random.seed(seed)
    
    features_list = []
    labels_list = []
    
    for i in range(num_clusters):
        # Create samples for this cluster
        cluster_features = np.random.randn(samples_per_cluster, feature_dim)
        # Add distinct cluster center with controlled separation
        cluster_center = np.random.randn(feature_dim) * CLUSTER_SEPARATION
        cluster_features += cluster_center
        
        features_list.append(cluster_features)
        labels_list.extend([i] * samples_per_cluster)
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    
    # Normalize features
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    return torch.from_numpy(features).float(), labels


def evaluate_clustering(true_labels, pred_labels):
    """Compute clustering evaluation metrics."""
    from scipy.optimize import linear_sum_assignment
    
    # Compute accuracy using Hungarian algorithm
    num_clusters = len(np.unique(true_labels))
    confusion_matrix = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    
    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    
    # NMI and ARI
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    return accuracy, nmi, ari


def test_clustering_improvements():
    """Main test function."""
    print("="*70)
    print("Testing TEMI Clustering Improvements")
    print("="*70)
    
    # Test parameters
    num_clusters = 10
    samples_per_cluster = 50
    feature_dim = 768
    num_epochs = 20
    
    print(f"\nTest Configuration:")
    print(f"  Clusters: {num_clusters}")
    print(f"  Samples per cluster: {samples_per_cluster}")
    print(f"  Total samples: {num_clusters * samples_per_cluster}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Training epochs: {num_epochs}")
    
    # Create synthetic data
    print("\nCreating synthetic clustered data...")
    features, true_labels = create_synthetic_clusters(
        num_clusters=num_clusters,
        samples_per_cluster=samples_per_cluster,
        feature_dim=feature_dim
    )
    print(f"✓ Created {len(features)} samples with {num_clusters} ground truth clusters")
    
    # Initialize clusterer with improved settings
    print("\nInitializing TEMI clusterer with improvements:")
    print("  - Sinkhorn-Knopp normalization: ENABLED")
    print("  - Optimizer: SGD with momentum")
    print("  - Learning rate: 0.005 (with warmup + cosine annealing)")
    print("  - Temperature: 0.1 (with gentle 20% annealing)")
    print("  - Loss weights: entropy=1.0, equivariance=2.0, uniformity=1.5")
    print("  - K-means initialization: ENABLED for cluster layer")
    
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device='cpu',
        use_sinkhorn=True,
        learning_rate=0.005,
        temperature=0.1
    )
    
    # Train
    print(f"\nTraining for {num_epochs} epochs...")
    history = clusterer.fit(
        features,
        num_epochs=num_epochs,
        batch_size=128,
        verbose=True
    )
    
    # Predict
    print("\nPredicting cluster assignments...")
    predictions = clusterer.predict(features, batch_size=256)
    
    # Analyze cluster distribution
    print("\n" + "="*70)
    print("Cluster Distribution Analysis")
    print("="*70)
    
    unique_clusters = np.unique(predictions)
    num_active = len(unique_clusters)
    num_empty = num_clusters - num_active
    
    print(f"Active clusters: {num_active}/{num_clusters}")
    print(f"Empty clusters: {num_empty}/{num_clusters}")
    
    if num_active > 0:
        cluster_sizes = Counter(predictions)
        sizes = list(cluster_sizes.values())
        print(f"Cluster size statistics:")
        print(f"  Mean: {np.mean(sizes):.1f}")
        print(f"  Std: {np.std(sizes):.1f}")
        print(f"  Min: {np.min(sizes)}")
        print(f"  Max: {np.max(sizes)}")
        print(f"  Balance (CV): {np.std(sizes)/np.mean(sizes):.3f}")
    
    # Evaluate clustering quality
    print("\n" + "="*70)
    print("Clustering Quality Metrics")
    print("="*70)
    
    accuracy, nmi, ari = evaluate_clustering(true_labels, predictions)
    
    print(f"Accuracy (Hungarian): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Normalized MI: {nmi:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Check if improvements are working
    print("\n" + "="*70)
    print("Verification Results")
    print("="*70)
    
    # Test thresholds
    MIN_ACTIVE_CLUSTER_RATIO = 0.8  # At least 80% of clusters should be active
    MIN_ACCURACY_SYNTHETIC = 0.70   # Expect >70% accuracy on synthetic data
    MAX_BALANCE_CV = 0.5            # Coefficient of variation should be < 0.5
    
    success = True
    
    # Check 1: Most clusters should be active
    min_active_clusters = int(MIN_ACTIVE_CLUSTER_RATIO * num_clusters)
    if num_active >= min_active_clusters:
        print(f"✓ Cluster collapse prevented (>={MIN_ACTIVE_CLUSTER_RATIO*100:.0f}% clusters active)")
    else:
        print(f"✗ Cluster collapse detected (only {num_active}/{num_clusters} active)")
        success = False
    
    # Check 2: Reasonable accuracy
    if accuracy >= MIN_ACCURACY_SYNTHETIC:
        print(f"✓ Good clustering accuracy ({accuracy*100:.1f}%)")
    else:
        print(f"✗ Low clustering accuracy ({accuracy*100:.1f}%)")
        success = False
    
    # Check 3: Balanced clusters
    if num_active > 0:
        cv = np.std(sizes) / np.mean(sizes)
        if cv < MAX_BALANCE_CV:
            print(f"✓ Balanced cluster distribution (CV={cv:.3f} < {MAX_BALANCE_CV})")
        else:
            print(f"✗ Unbalanced clusters (CV={cv:.3f} >= {MAX_BALANCE_CV})")
            success = False
    
    # Check 4: Loss decreased
    if history['total_loss'][0] > history['total_loss'][-1]:
        print("✓ Loss decreased during training")
    else:
        print("✗ Loss did not decrease properly")
        success = False
    
    print("\n" + "="*70)
    if success:
        print("🎉 All improvements verified successfully!")
        print("="*70)
        return 0
    else:
        print("❌ Some tests failed - further improvements needed")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit(test_clustering_improvements())
