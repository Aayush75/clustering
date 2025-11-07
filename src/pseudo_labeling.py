"""
Pseudo-labeling module for TEMI clustering.

This module provides functionality to generate pseudo labels for clusters
by mapping them to actual labels based on the most representative samples
(samples closest to cluster centers).
"""

import warnings
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt


def find_k_nearest_to_centers(
    features: torch.Tensor,
    cluster_assignments: np.ndarray,
    cluster_centers: torch.Tensor,
    k: int = 10,
    return_distances: bool = False
) -> Dict[int, np.ndarray]:
    """
    Find k nearest samples to each cluster center.
    
    For each cluster, find the k samples that are closest to the cluster center.
    These samples are the most representative of their cluster.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_centers: Cluster centers of shape (n_clusters, n_features)
        k: Number of nearest samples to find per cluster
        return_distances: If True, also return distances
        
    Returns:
        Dictionary mapping cluster_id -> array of k sample indices
        If return_distances=True, returns (indices_dict, distances_dict)
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(cluster_centers, torch.Tensor):
        cluster_centers = cluster_centers.cpu().numpy()
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.cpu().numpy()
    
    # Normalize features and centers for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    centers_norm = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)
    
    k_nearest_indices = {}
    k_nearest_distances = {}
    
    unique_clusters = np.unique(cluster_assignments)
    
    for cluster_id in unique_clusters:
        # Get indices of samples in this cluster
        cluster_mask = cluster_assignments == cluster_id
        cluster_sample_indices = np.where(cluster_mask)[0]
        
        if len(cluster_sample_indices) == 0:
            k_nearest_indices[cluster_id] = np.array([])
            if return_distances:
                k_nearest_distances[cluster_id] = np.array([])
            continue
        
        # Get features for samples in this cluster
        cluster_features = features_norm[cluster_mask]
        
        # Compute cosine similarity to cluster center (use dot product since normalized)
        # Check if cluster_id is valid index; if not, skip this cluster with a warning
        if cluster_id >= len(centers_norm):
            warnings.warn(
                f"Cluster ID {cluster_id} is out of bounds for cluster centers "
                f"(max index: {len(centers_norm)-1}). This may indicate a mismatch "
                f"between cluster assignments and cluster centers. Skipping this cluster."
            )
            k_nearest_indices[cluster_id] = np.array([])
            if return_distances:
                k_nearest_distances[cluster_id] = np.array([])
            continue
        
        center = centers_norm[cluster_id]
        similarities = np.dot(cluster_features, center)
        
        # Convert similarity to distance (1 - similarity for cosine distance)
        distances = 1 - similarities
        
        # Find k nearest (smallest distances)
        k_actual = min(k, len(cluster_sample_indices))
        nearest_idx_in_cluster = np.argpartition(distances, k_actual-1)[:k_actual]
        
        # Sort by distance (ascending)
        nearest_idx_in_cluster = nearest_idx_in_cluster[np.argsort(distances[nearest_idx_in_cluster])]
        
        # Map back to original indices
        k_nearest_indices[cluster_id] = cluster_sample_indices[nearest_idx_in_cluster]
        
        if return_distances:
            k_nearest_distances[cluster_id] = distances[nearest_idx_in_cluster]
    
    if return_distances:
        return k_nearest_indices, k_nearest_distances
    return k_nearest_indices


def map_clusters_to_labels(
    cluster_assignments: np.ndarray,
    true_labels: np.ndarray,
    k_nearest_indices: Dict[int, np.ndarray],
    return_confidence: bool = False
) -> Dict[int, int]:
    """
    Map cluster IDs to actual labels using k-nearest samples to cluster centers.
    
    For each cluster, the pseudo label is determined by the majority vote
    among the k nearest samples to the cluster center.
    
    Args:
        cluster_assignments: Cluster assignments of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        k_nearest_indices: Dictionary mapping cluster_id -> array of k sample indices
        return_confidence: If True, also return confidence scores for each cluster
        
    Returns:
        Dictionary mapping cluster_id -> pseudo_label
        If return_confidence=True, returns (cluster_to_label, cluster_to_confidence)
    """
    cluster_to_label = {}
    cluster_to_confidence = {}
    
    for cluster_id, sample_indices in k_nearest_indices.items():
        if len(sample_indices) == 0:
            # Empty cluster - no mapping
            cluster_to_label[cluster_id] = -1
            cluster_to_confidence[cluster_id] = 0.0
            continue
        
        # Get true labels for k nearest samples
        k_labels = true_labels[sample_indices]
        
        # Find the most common label (majority vote)
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        majority_count = np.max(counts)
        
        # Compute confidence as the proportion of k-nearest samples with majority label
        # This represents cluster purity for the k-nearest samples
        confidence = majority_count / len(sample_indices)
        
        cluster_to_label[cluster_id] = majority_label
        cluster_to_confidence[cluster_id] = confidence
    
    if return_confidence:
        return cluster_to_label, cluster_to_confidence
    return cluster_to_label


def apply_pseudo_labels(
    cluster_assignments: np.ndarray,
    cluster_to_label: Dict[int, int]
) -> np.ndarray:
    """
    Apply pseudo labels to all samples based on cluster-to-label mapping.
    
    Args:
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        
    Returns:
        Pseudo labels of shape (n_samples,)
    """
    pseudo_labels = np.zeros_like(cluster_assignments)
    
    for i, cluster_id in enumerate(cluster_assignments):
        pseudo_labels[i] = cluster_to_label.get(cluster_id, -1)
    
    return pseudo_labels


def compute_sample_confidence_scores(
    features: np.ndarray,
    cluster_assignments: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_to_label: Dict[int, int],
    cluster_to_confidence: Dict[int, float]
) -> np.ndarray:
    """
    Compute sample-wise confidence scores for pseudo labels.
    
    The confidence score for each sample is based on:
    1. Distance to cluster center (closer = higher confidence)
    2. Cluster purity (from k-nearest samples majority vote)
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_centers: Cluster centers of shape (n_clusters, n_features)
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        cluster_to_confidence: Dictionary mapping cluster_id -> cluster confidence
        
    Returns:
        Confidence scores of shape (n_samples,) with values in [0, 1]
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(cluster_centers, torch.Tensor):
        cluster_centers = cluster_centers.cpu().numpy()
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.cpu().numpy()
    
    # Normalize features and centers for cosine similarity
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    centers_norm = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-8)
    
    n_samples = len(cluster_assignments)
    confidence_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_id = cluster_assignments[i]
        
        # Get cluster confidence (from k-nearest majority vote)
        cluster_conf = cluster_to_confidence.get(cluster_id, 0.0)
        
        # Skip if cluster has no valid mapping
        if cluster_to_label.get(cluster_id, -1) == -1:
            confidence_scores[i] = 0.0
            continue
        
        # Compute distance-based confidence
        if cluster_id < len(centers_norm):
            # Compute cosine similarity to cluster center
            similarity = np.dot(features_norm[i], centers_norm[cluster_id])
            # Convert to confidence (similarity ranges from -1 to 1, map to 0-1)
            distance_conf = (similarity + 1) / 2.0
        else:
            distance_conf = 0.0
        
        # Combine cluster confidence and distance confidence
        # Use geometric mean for balanced combination
        # Ensure non-negative values to avoid NaN in sqrt (defense-in-depth)
        # While theoretically cluster_conf and distance_conf are always >= 0,
        # this guards against numerical instability and edge cases
        cluster_conf = max(0.0, cluster_conf)
        distance_conf = max(0.0, distance_conf)
        confidence_scores[i] = np.sqrt(cluster_conf * distance_conf)
    
    return confidence_scores


def compute_pseudo_label_accuracy(
    pseudo_labels: np.ndarray,
    true_labels: np.ndarray
) -> float:
    """
    Compute accuracy of pseudo labels against true labels.
    
    Args:
        pseudo_labels: Pseudo labels of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Filter out samples with no pseudo label (-1)
    valid_mask = pseudo_labels != -1
    
    if not np.any(valid_mask):
        return 0.0
    
    accuracy = np.mean(pseudo_labels[valid_mask] == true_labels[valid_mask])
    return accuracy


def visualize_cluster_mapping(
    images: np.ndarray,
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    cluster_to_label: Dict[int, int],
    k_nearest_indices: Dict[int, np.ndarray],
    save_path: str,
    class_names: Optional[List[str]] = None,
    max_clusters_to_show: int = 20,
    samples_per_cluster: int = 5
) -> None:
    """
    Visualize the cluster-to-label mapping by showing sample images from each cluster.
    
    For each cluster, shows the k nearest samples to the cluster center along with:
    - Cluster ID
    - Mapped pseudo label
    - True labels of the samples
    
    Args:
        images: Image array of shape (n_samples, H, W, C) or (n_samples, C, H, W)
        true_labels: True labels of shape (n_samples,)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        k_nearest_indices: Dictionary mapping cluster_id -> array of k sample indices
        save_path: Path to save the visualization
        class_names: Optional list of class names for better readability
        max_clusters_to_show: Maximum number of clusters to visualize
        samples_per_cluster: Number of samples to show per cluster
    """
    # Convert tensors to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.cpu().numpy()
    
    # Get unique clusters
    unique_clusters = sorted(list(k_nearest_indices.keys()))[:max_clusters_to_show]
    
    if len(unique_clusters) == 0:
        print("No clusters to visualize!")
        return
    
    # Create figure
    n_clusters = len(unique_clusters)
    fig, axes = plt.subplots(
        n_clusters, 
        samples_per_cluster, 
        figsize=(3 * samples_per_cluster, 3 * n_clusters)
    )
    
    # Handle single cluster or single sample per cluster cases
    if n_clusters == 1 and samples_per_cluster == 1:
        axes = np.array([[axes]])  # Make it 2D with shape (1, 1)
    elif n_clusters == 1:
        axes = axes.reshape(1, -1)  # Make it 2D with shape (1, samples_per_cluster)
    elif samples_per_cluster == 1:
        axes = axes.reshape(-1, 1)  # Make it 2D with shape (n_clusters, 1)
    
    for row_idx, cluster_id in enumerate(unique_clusters):
        sample_indices = k_nearest_indices[cluster_id][:samples_per_cluster]
        pseudo_label = cluster_to_label.get(cluster_id, -1)
        
        for col_idx in range(samples_per_cluster):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(sample_indices):
                sample_idx = sample_indices[col_idx]
                
                # Get image and normalize for display
                img = images[sample_idx]
                
                # Handle different image formats
                if img.shape[0] == 3 or img.shape[0] == 1:  # (C, H, W)
                    img = np.transpose(img, (1, 2, 0))
                
                # Normalize to [0, 1] if needed
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Convert grayscale to RGB if needed
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                
                ax.imshow(np.clip(img, 0, 1))
                
                # Get true label for this sample
                true_label = true_labels[sample_idx]
                
                # Create title
                if class_names:
                    title = f"True: {class_names[true_label]}"
                else:
                    title = f"True: {true_label}"
                
                # Color title based on match
                title_color = 'green' if true_label == pseudo_label else 'red'
                ax.set_title(title, fontsize=8, color=title_color)
            else:
                # No more samples for this cluster
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add cluster info as ylabel for first column
        if class_names and pseudo_label != -1:
            ylabel = f"Cluster {cluster_id}\n→ {class_names[pseudo_label]}"
        else:
            ylabel = f"Cluster {cluster_id}\n→ Label {pseudo_label}"
        
        axes[row_idx, 0].set_ylabel(ylabel, fontsize=10, fontweight='bold')
    
    plt.suptitle(
        f"Cluster-to-Label Mapping Visualization\n"
        f"(Showing {samples_per_cluster} nearest samples to each cluster center)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Cluster mapping visualization saved to {save_path}")
    plt.close()


def print_cluster_mapping_summary(
    cluster_to_label: Dict[int, int],
    cluster_assignments: np.ndarray,
    true_labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    cluster_to_confidence: Optional[Dict[int, float]] = None,
    confidence_scores: Optional[np.ndarray] = None
) -> None:
    """
    Print a summary of the cluster-to-label mapping with confidence scores.
    
    Args:
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        cluster_assignments: Cluster assignments of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        class_names: Optional list of class names for better readability
        cluster_to_confidence: Optional dictionary mapping cluster_id -> confidence
        confidence_scores: Optional sample-wise confidence scores
    """
    print("\n" + "="*80)
    print("Cluster-to-Label Mapping Summary")
    print("="*80)
    
    # Apply pseudo labels
    pseudo_labels = apply_pseudo_labels(cluster_assignments, cluster_to_label)
    
    # Compute accuracy
    accuracy = compute_pseudo_label_accuracy(pseudo_labels, true_labels)
    
    print(f"\nOverall Pseudo-Label Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nTotal Clusters: {len(cluster_to_label)}")
    print(f"Empty Clusters: {sum(1 for v in cluster_to_label.values() if v == -1)}")
    
    # Print confidence statistics if available
    if confidence_scores is not None:
        valid_mask = pseudo_labels != -1
        if np.any(valid_mask):
            avg_confidence = np.mean(confidence_scores[valid_mask])
            print(f"Average Sample Confidence: {avg_confidence:.4f}")
            print(f"Confidence Range: [{np.min(confidence_scores[valid_mask]):.4f}, "
                  f"{np.max(confidence_scores[valid_mask]):.4f}]")
    
    # Show per-cluster mapping
    print("\nCluster Mappings:")
    print("-" * 95)
    if cluster_to_confidence is not None:
        print(f"{'Cluster ID':<12} {'→':<3} {'Pseudo Label':<20} {'Cluster Size':<15} "
              f"{'Accuracy':<12} {'Confidence':<10}")
    else:
        print(f"{'Cluster ID':<12} {'→':<3} {'Pseudo Label':<20} {'Cluster Size':<15} {'Accuracy':<10}")
    print("-" * 95)
    
    for cluster_id in sorted(cluster_to_label.keys()):
        pseudo_label = cluster_to_label[cluster_id]
        
        # Get samples in this cluster
        cluster_mask = cluster_assignments == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        if cluster_size == 0:
            continue
        
        # Compute per-cluster accuracy
        cluster_true_labels = true_labels[cluster_mask]
        cluster_accuracy = np.mean(cluster_true_labels == pseudo_label) if pseudo_label != -1 else 0.0
        
        # Format label
        if class_names and pseudo_label != -1:
            label_str = f"{pseudo_label} ({class_names[pseudo_label]})"
        else:
            label_str = str(pseudo_label)
        
        # Print with or without confidence
        if cluster_to_confidence is not None:
            cluster_conf = cluster_to_confidence.get(cluster_id, 0.0)
            print(f"{cluster_id:<12} {'→':<3} {label_str:<20} {cluster_size:<15} "
                  f"{cluster_accuracy:<12.4f} {cluster_conf:<10.4f}")
        else:
            print(f"{cluster_id:<12} {'→':<3} {label_str:<20} {cluster_size:<15} {cluster_accuracy:.4f}")
    
    print("-" * 95)
    print()


def generate_pseudo_labels(
    features: torch.Tensor,
    cluster_assignments: np.ndarray,
    true_labels: np.ndarray,
    cluster_centers: torch.Tensor,
    k: int = 10,
    verbose: bool = True,
    return_confidence: bool = True
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, np.ndarray], Optional[np.ndarray], Dict[int, float]]:
    """
    Complete pipeline to generate pseudo labels for clusters with confidence scores.
    
    This function:
    1. Finds k nearest samples to each cluster center
    2. Maps clusters to labels based on majority vote
    3. Computes confidence scores for each cluster based on label purity
    4. Applies pseudo labels to all samples
    5. Computes sample-wise confidence scores based on distance and cluster purity
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        cluster_centers: Cluster centers of shape (n_clusters, n_features)
        k: Number of nearest samples to use for label assignment
        verbose: If True, print summary
        return_confidence: If True, return confidence scores (default: True)
        
    Returns:
        Tuple of (pseudo_labels, cluster_to_label, k_nearest_indices, confidence_scores, cluster_to_confidence)
        - pseudo_labels: Assigned pseudo labels for all samples
        - cluster_to_label: Mapping from cluster ID to pseudo label
        - k_nearest_indices: Indices of k nearest samples per cluster
        - confidence_scores: Sample-wise confidence scores (None if return_confidence=False)
        - cluster_to_confidence: Cluster-level confidence scores
    """
    # Step 1: Find k nearest samples to each cluster center
    if verbose:
        print(f"\nFinding {k} nearest samples to each cluster center...")
    k_nearest_indices = find_k_nearest_to_centers(
        features, cluster_assignments, cluster_centers, k
    )
    
    # Step 2: Map clusters to labels with confidence
    if verbose:
        print("Mapping clusters to labels using majority vote...")
    cluster_to_label, cluster_to_confidence = map_clusters_to_labels(
        cluster_assignments, true_labels, k_nearest_indices, return_confidence=True
    )
    
    # Step 3: Apply pseudo labels
    if verbose:
        print("Applying pseudo labels to all samples...")
    pseudo_labels = apply_pseudo_labels(cluster_assignments, cluster_to_label)
    
    # Step 4: Compute sample-wise confidence scores
    confidence_scores = None
    if return_confidence:
        if verbose:
            print("Computing sample-wise confidence scores...")
        confidence_scores = compute_sample_confidence_scores(
            features, cluster_assignments, cluster_centers,
            cluster_to_label, cluster_to_confidence
        )
    
    # Print summary if verbose
    if verbose:
        accuracy = compute_pseudo_label_accuracy(pseudo_labels, true_labels)
        print(f"\nPseudo-label generation complete!")
        print(f"Pseudo-Label Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if confidence_scores is not None:
            valid_mask = pseudo_labels != -1
            if np.any(valid_mask):
                avg_confidence = np.mean(confidence_scores[valid_mask])
                print(f"Average Confidence Score: {avg_confidence:.4f}")
                print(f"Confidence Score Range: [{np.min(confidence_scores[valid_mask]):.4f}, "
                      f"{np.max(confidence_scores[valid_mask]):.4f}]")
    
    return pseudo_labels, cluster_to_label, k_nearest_indices, confidence_scores, cluster_to_confidence
