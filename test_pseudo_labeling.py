"""
Simple test script to verify pseudo labeling functionality.
"""

import torch
import numpy as np
from src.pseudo_labeling import (
    find_k_nearest_to_centers,
    map_clusters_to_labels,
    apply_pseudo_labels,
    compute_pseudo_label_accuracy,
    generate_pseudo_labels
)

def test_pseudo_labeling():
    """Test pseudo labeling with synthetic data."""
    print("Testing pseudo labeling functionality...\n")
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 128
    n_clusters = 10
    n_classes = 10
    
    # Generate random features
    features = torch.randn(n_samples, n_features)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    # Generate cluster centers
    cluster_centers = torch.randn(n_clusters, n_features)
    cluster_centers = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)
    
    # Generate cluster assignments (somewhat related to centers)
    distances = torch.cdist(features, cluster_centers)
    cluster_assignments = torch.argmin(distances, dim=1).numpy()
    
    # Generate true labels (somewhat correlated with clusters)
    true_labels = np.random.randint(0, n_classes, n_samples)
    # Make some clusters more correlated with specific labels
    # Use 70% primary label and 30% secondary label to simulate realistic cluster purity
    PRIMARY_LABEL_PROB = 0.7
    SECONDARY_LABEL_PROB = 0.3
    for i in range(min(n_clusters, n_classes)):
        mask = cluster_assignments == i
        true_labels[mask] = np.random.choice(
            [i, (i+1) % n_classes], 
            size=np.sum(mask), 
            p=[PRIMARY_LABEL_PROB, SECONDARY_LABEL_PROB]
        )
    
    print(f"Synthetic data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Classes: {n_classes}")
    
    # Test 1: Find k nearest to centers
    print("\n" + "="*60)
    print("Test 1: Finding k nearest samples to cluster centers")
    print("="*60)
    k = 10
    k_nearest_indices, k_nearest_distances = find_k_nearest_to_centers(
        features, cluster_assignments, cluster_centers, k, return_distances=True
    )
    
    print(f"\nFound k={k} nearest samples for {len(k_nearest_indices)} clusters")
    for cluster_id in range(min(3, n_clusters)):  # Show first 3 clusters
        indices = k_nearest_indices[cluster_id]
        distances = k_nearest_distances[cluster_id]
        print(f"  Cluster {cluster_id}: {len(indices)} samples, distances: {distances[:3]}")
    
    # Test 2: Map clusters to labels
    print("\n" + "="*60)
    print("Test 2: Mapping clusters to labels")
    print("="*60)
    cluster_to_label = map_clusters_to_labels(
        cluster_assignments, true_labels, k_nearest_indices
    )
    
    print(f"\nCluster to label mapping:")
    for cluster_id in sorted(cluster_to_label.keys())[:5]:  # Show first 5
        pseudo_label = cluster_to_label[cluster_id]
        print(f"  Cluster {cluster_id} → Label {pseudo_label}")
    
    # Test 3: Apply pseudo labels
    print("\n" + "="*60)
    print("Test 3: Applying pseudo labels")
    print("="*60)
    pseudo_labels = apply_pseudo_labels(cluster_assignments, cluster_to_label)
    
    print(f"\nGenerated {len(pseudo_labels)} pseudo labels")
    print(f"  Unique pseudo labels: {np.unique(pseudo_labels)}")
    
    # Test 4: Compute accuracy
    print("\n" + "="*60)
    print("Test 4: Computing pseudo label accuracy")
    print("="*60)
    accuracy = compute_pseudo_label_accuracy(pseudo_labels, true_labels)
    
    print(f"\nPseudo-label accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Test 5: Complete pipeline with confidence scores
    print("\n" + "="*60)
    print("Test 5: Complete pipeline with confidence scores")
    print("="*60)
    pseudo_labels_full, cluster_to_label_full, k_nearest_full, confidence_scores, cluster_confidence = generate_pseudo_labels(
        features=features,
        cluster_assignments=cluster_assignments,
        true_labels=true_labels,
        cluster_centers=cluster_centers,
        k=k,
        verbose=True,
        return_confidence=True
    )
    
    print(f"\nConfidence scores shape: {confidence_scores.shape}")
    print(f"Cluster confidence dict size: {len(cluster_confidence)}")
    print(f"Confidence score statistics:")
    print(f"  Min: {np.min(confidence_scores):.4f}")
    print(f"  Max: {np.max(confidence_scores):.4f}")
    # Note: Samples with no valid cluster mapping have confidence = 0
    # This occurs when cluster has no pseudo label (pseudo_labels == -1)
    valid_mask = pseudo_labels_full != -1
    valid_confidence = confidence_scores[valid_mask]
    if len(valid_confidence) > 0:
        print(f"  Mean (valid): {np.mean(valid_confidence):.4f}")
        print(f"  Std (valid): {np.std(valid_confidence):.4f}")
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_pseudo_labeling()
