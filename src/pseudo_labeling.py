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


# Constant to represent samples with no valid pseudo label
NO_PSEUDO_LABEL = -1


def find_k_nearest_to_centers(
    features: torch.Tensor,
    cluster_assignments: torch.Tensor,
    cluster_centers: torch.Tensor,
    k: int = 10,
    return_distances: bool = False
) -> Dict[int, torch.Tensor]:
    """
    Find k nearest samples to each cluster center (PyTorch-native).
    
    For each cluster, find the k samples that are closest to the cluster center.
    These samples are the most representative of their cluster.
    
    Args:
        features: Feature tensor of shape (n_samples, n_features)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_centers: Cluster centers of shape (n_clusters, n_features)
        k: Number of nearest samples to find per cluster
        return_distances: If True, also return distances
        
    Returns:
        Dictionary mapping cluster_id -> tensor of k sample indices
        If return_distances=True, returns (indices_dict, distances_dict)
    """
    # Ensure all inputs are tensors on the same device
    device = features.device
    if not isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = torch.tensor(cluster_assignments, device=device)
    else:
        cluster_assignments = cluster_assignments.to(device)
    
    if not isinstance(cluster_centers, torch.Tensor):
        cluster_centers = torch.tensor(cluster_centers, device=device)
    else:
        cluster_centers = cluster_centers.to(device)
    
    features = features.to(device)
    
    # Normalize features and centers for cosine similarity
    features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
    centers_norm = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)
    
    k_nearest_indices = {}
    k_nearest_distances = {}
    
    unique_clusters = torch.unique(cluster_assignments)
    
    for cluster_id in unique_clusters:
        cluster_id_int = int(cluster_id.item())
        
        # Get indices of samples in this cluster
        cluster_mask = cluster_assignments == cluster_id
        cluster_sample_indices = torch.where(cluster_mask)[0]
        
        if len(cluster_sample_indices) == 0:
            k_nearest_indices[cluster_id_int] = torch.tensor([], dtype=torch.long, device=device)
            if return_distances:
                k_nearest_distances[cluster_id_int] = torch.tensor([], dtype=torch.float32, device=device)
            continue
        
        # Get features for samples in this cluster
        cluster_features = features_norm[cluster_mask]
        
        # Check if cluster_id is valid index; if not, skip this cluster with a warning
        if cluster_id_int >= len(centers_norm):
            warnings.warn(
                f"Cluster ID {cluster_id_int} is out of bounds for cluster centers "
                f"(max index: {len(centers_norm)-1}). This may indicate a mismatch "
                f"between cluster assignments and cluster centers. Skipping this cluster."
            )
            k_nearest_indices[cluster_id_int] = torch.tensor([], dtype=torch.long, device=device)
            if return_distances:
                k_nearest_distances[cluster_id_int] = torch.tensor([], dtype=torch.float32, device=device)
            continue
        
        # Compute cosine similarity to cluster center
        center = centers_norm[cluster_id_int]
        similarities = torch.mm(cluster_features, center.unsqueeze(1)).squeeze(1)
        
        # Convert similarity to distance (1 - similarity for cosine distance)
        distances = 1 - similarities
        
        # Find k nearest (smallest distances)
        k_actual = min(k, len(cluster_sample_indices))
        
        # Use topk to get k smallest distances
        nearest_distances, nearest_idx_in_cluster = torch.topk(distances, k_actual, largest=False, sorted=True)
        
        # Map back to original indices
        k_nearest_indices[cluster_id_int] = cluster_sample_indices[nearest_idx_in_cluster]
        
        if return_distances:
            k_nearest_distances[cluster_id_int] = nearest_distances
    
    if return_distances:
        return k_nearest_indices, k_nearest_distances
    return k_nearest_indices


def map_clusters_to_labels(
    cluster_assignments: torch.Tensor,
    true_labels: torch.Tensor,
    k_nearest_indices: Dict[int, torch.Tensor],
    return_confidence: bool = False
) -> Dict[int, int]:
    """
    Map cluster IDs to actual labels using k-nearest samples to cluster centers (PyTorch-native).
    
    For each cluster, the pseudo label is determined by the majority vote
    among the k nearest samples to the cluster center.
    
    Args:
        cluster_assignments: Cluster assignments of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        k_nearest_indices: Dictionary mapping cluster_id -> tensor of k sample indices
        return_confidence: If True, also return confidence scores for each cluster
        
    Returns:
        Dictionary mapping cluster_id -> pseudo_label
        If return_confidence=True, returns (cluster_to_label, cluster_to_confidence)
    """
    # Ensure inputs are tensors and determine the device from k_nearest_indices
    # Get device from first non-empty tensor in k_nearest_indices
    device = None
    for indices in k_nearest_indices.values():
        if len(indices) > 0 and isinstance(indices, torch.Tensor):
            device = indices.device
            break
    
    # If no device found from indices, use CPU as default
    if device is None:
        device = torch.device('cpu')
    
    if not isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = torch.tensor(cluster_assignments, device=device)
    else:
        cluster_assignments = cluster_assignments.to(device)
        
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels, device=device)
    else:
        true_labels = true_labels.to(device)
    
    cluster_to_label = {}
    cluster_to_confidence = {}
    
    for cluster_id, sample_indices in k_nearest_indices.items():
        if len(sample_indices) == 0:
            # Empty cluster - no mapping
            cluster_to_label[cluster_id] = NO_PSEUDO_LABEL
            cluster_to_confidence[cluster_id] = 0.0
            continue
        
        # Ensure sample_indices is on the same device as true_labels
        sample_indices = sample_indices.to(device)
        
        # Get true labels for k nearest samples
        k_labels = true_labels[sample_indices]
        
        # Find the most common label (majority vote)
        unique_labels, counts = torch.unique(k_labels, return_counts=True)
        majority_idx = torch.argmax(counts)
        majority_label = int(unique_labels[majority_idx].item())
        majority_count = int(counts[majority_idx].item())
        
        # Compute confidence as the proportion of k-nearest samples with majority label
        # This represents cluster purity for the k-nearest samples
        confidence = majority_count / len(sample_indices)
        
        cluster_to_label[cluster_id] = majority_label
        cluster_to_confidence[cluster_id] = confidence
    
    if return_confidence:
        return cluster_to_label, cluster_to_confidence
    return cluster_to_label


def apply_pseudo_labels(
    cluster_assignments: torch.Tensor,
    cluster_to_label: Dict[int, int]
) -> torch.Tensor:
    """
    Apply pseudo labels to all samples based on cluster-to-label mapping (PyTorch-native).
    
    Args:
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        
    Returns:
        Pseudo labels of shape (n_samples,)
    """
    # Ensure input is tensor
    if not isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = torch.tensor(cluster_assignments)
    
    device = cluster_assignments.device
    pseudo_labels = torch.full_like(cluster_assignments, NO_PSEUDO_LABEL, dtype=torch.long, device=device)
    
    # Vectorized approach for better performance
    unique_clusters = torch.unique(cluster_assignments)
    for cluster_id in unique_clusters:
        cluster_id_int = int(cluster_id.item())
        if cluster_id_int in cluster_to_label:
            mask = cluster_assignments == cluster_id
            pseudo_labels[mask] = cluster_to_label[cluster_id_int]
    
    return pseudo_labels


def compute_sample_confidence_scores(
    features: torch.Tensor,
    cluster_assignments: torch.Tensor,
    cluster_centers: torch.Tensor,
    cluster_to_label: Dict[int, int],
    cluster_to_confidence: Dict[int, float]
) -> torch.Tensor:
    """
    Compute sample-wise confidence scores for pseudo labels (PyTorch-native).
    
    The confidence score for each sample is based on:
    1. Distance to cluster center (closer = higher confidence)
    2. Cluster purity (from k-nearest samples majority vote)
    
    Args:
        features: Feature tensor of shape (n_samples, n_features)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_centers: Cluster centers of shape (n_clusters, n_features)
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        cluster_to_confidence: Dictionary mapping cluster_id -> cluster confidence
        
    Returns:
        Confidence scores of shape (n_samples,) with values in [0, 1]
    """
    # Ensure all inputs are tensors on the same device
    device = features.device
    if not isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = torch.tensor(cluster_assignments, device=device)
    else:
        cluster_assignments = cluster_assignments.to(device)
    
    if not isinstance(cluster_centers, torch.Tensor):
        cluster_centers = torch.tensor(cluster_centers, device=device)
    else:
        cluster_centers = cluster_centers.to(device)
    
    features = features.to(device)
    
    # Normalize features and centers for cosine similarity
    features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
    centers_norm = torch.nn.functional.normalize(cluster_centers, p=2, dim=1)
    
    n_samples = len(cluster_assignments)
    confidence_scores = torch.zeros(n_samples, device=device)
    
    for i in range(n_samples):
        cluster_id = int(cluster_assignments[i].item())
        
        # Get cluster confidence (from k-nearest majority vote)
        cluster_conf = cluster_to_confidence.get(cluster_id, 0.0)
        
        # Skip if cluster has no valid mapping
        if cluster_to_label.get(cluster_id, NO_PSEUDO_LABEL) == NO_PSEUDO_LABEL:
            confidence_scores[i] = 0.0
            continue
        
        # Compute distance-based confidence
        if cluster_id < len(centers_norm):
            # Compute cosine similarity to cluster center
            similarity = torch.dot(features_norm[i], centers_norm[cluster_id])
            # Convert to confidence (similarity ranges from -1 to 1, map to 0-1)
            distance_conf = (similarity + 1) / 2.0
        else:
            distance_conf = 0.0
        
        # Combine cluster confidence and distance confidence
        # Use geometric mean for balanced combination
        # Ensure non-negative values to avoid NaN in sqrt
        cluster_conf = max(0.0, cluster_conf)
        distance_conf = max(0.0, distance_conf.item() if isinstance(distance_conf, torch.Tensor) else distance_conf)
        confidence_scores[i] = torch.sqrt(torch.tensor(cluster_conf * distance_conf, device=device))
    
    return confidence_scores


def compute_pseudo_label_accuracy(
    pseudo_labels: torch.Tensor,
    true_labels: torch.Tensor
) -> float:
    """
    Compute accuracy of pseudo labels against true labels (PyTorch-native).
    
    Args:
        pseudo_labels: Pseudo labels of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Ensure inputs are tensors on the same device
    if not isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = torch.tensor(pseudo_labels)
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels)
    
    # Move true_labels to same device as pseudo_labels
    true_labels = true_labels.to(pseudo_labels.device)
    
    # Filter out samples with no pseudo label
    valid_mask = pseudo_labels != NO_PSEUDO_LABEL
    
    if not torch.any(valid_mask):
        return 0.0
    
    accuracy = torch.mean((pseudo_labels[valid_mask] == true_labels[valid_mask]).float()).item()
    return accuracy


def visualize_cluster_mapping(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    cluster_assignments: torch.Tensor,
    cluster_to_label: Dict[int, int],
    k_nearest_indices: Dict[int, torch.Tensor],
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
        images: Image tensor of shape (n_samples, H, W, C) or (n_samples, C, H, W)
        true_labels: True labels of shape (n_samples,)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        cluster_to_label: Dictionary mapping cluster_id -> pseudo_label
        k_nearest_indices: Dictionary mapping cluster_id -> tensor of k sample indices
        save_path: Path to save the visualization
        class_names: Optional list of class names for better readability
        max_clusters_to_show: Maximum number of clusters to visualize
        samples_per_cluster: Number of samples to show per cluster
    """
    # Convert tensors to numpy only for visualization (at the final step)
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.cpu().numpy()
    
    # Convert k_nearest_indices tensors to numpy
    k_nearest_indices_np = {}
    for k, v in k_nearest_indices.items():
        if isinstance(v, torch.Tensor):
            k_nearest_indices_np[k] = v.cpu().numpy()
        else:
            k_nearest_indices_np[k] = v
    
    # Get unique clusters
    unique_clusters = sorted(list(k_nearest_indices_np.keys()))[:max_clusters_to_show]
    
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
        sample_indices = k_nearest_indices_np[cluster_id][:samples_per_cluster]
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
        if class_names and pseudo_label != NO_PSEUDO_LABEL:
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
    cluster_assignments: torch.Tensor,
    true_labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    cluster_to_confidence: Optional[Dict[int, float]] = None,
    confidence_scores: Optional[torch.Tensor] = None
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
    
    # Ensure inputs are tensors on the same device
    if not isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = torch.tensor(cluster_assignments)
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels)
    
    # Ensure both tensors are on the same device
    device = cluster_assignments.device
    true_labels = true_labels.to(device)
    
    # Apply pseudo labels
    pseudo_labels = apply_pseudo_labels(cluster_assignments, cluster_to_label)
    
    # Compute accuracy
    accuracy = compute_pseudo_label_accuracy(pseudo_labels, true_labels)
    
    print(f"\nOverall Pseudo-Label Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nTotal Clusters: {len(cluster_to_label)}")
    print(f"Empty Clusters: {sum(1 for v in cluster_to_label.values() if v == NO_PSEUDO_LABEL)}")
    
    # Print confidence statistics if available
    if confidence_scores is not None:
        if not isinstance(confidence_scores, torch.Tensor):
            confidence_scores = torch.tensor(confidence_scores)
        # Ensure confidence_scores is on the same device
        confidence_scores = confidence_scores.to(device)
        valid_mask = pseudo_labels != NO_PSEUDO_LABEL
        if torch.any(valid_mask):
            avg_confidence = torch.mean(confidence_scores[valid_mask]).item()
            print(f"Average Sample Confidence: {avg_confidence:.4f}")
            print(f"Confidence Range: [{torch.min(confidence_scores[valid_mask]).item():.4f}, "
                  f"{torch.max(confidence_scores[valid_mask]).item():.4f}]")
    
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
        cluster_size = torch.sum(cluster_mask).item()
        
        if cluster_size == 0:
            continue
        
        # Compute per-cluster accuracy
        cluster_true_labels = true_labels[cluster_mask]
        cluster_accuracy = torch.mean((cluster_true_labels == pseudo_label).float()).item() if pseudo_label != NO_PSEUDO_LABEL else 0.0
        
        # Format label
        if class_names and pseudo_label != NO_PSEUDO_LABEL:
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
    cluster_assignments: torch.Tensor,
    true_labels: torch.Tensor,
    cluster_centers: torch.Tensor,
    k: int = 10,
    verbose: bool = True,
    return_confidence: bool = True
) -> Tuple[torch.Tensor, Dict[int, int], Dict[int, torch.Tensor], Optional[torch.Tensor], Dict[int, float]]:
    """
    Complete pipeline to generate pseudo labels for clusters with confidence scores (PyTorch-native).
    
    This function:
    1. Finds k nearest samples to each cluster center
    2. Maps clusters to labels based on majority vote
    3. Computes confidence scores for each cluster based on label purity
    4. Applies pseudo labels to all samples
    5. Computes sample-wise confidence scores based on distance and cluster purity
    
    Args:
        features: Feature tensor of shape (n_samples, n_features)
        cluster_assignments: Cluster assignments of shape (n_samples,)
        true_labels: True labels of shape (n_samples,)
        cluster_centers: Cluster centers of shape (n_clusters, n_features)
        k: Number of nearest samples to use for label assignment
        verbose: If True, print summary
        return_confidence: If True, return confidence scores (default: True)
        
    Returns:
        Tuple of (pseudo_labels, cluster_to_label, k_nearest_indices, confidence_scores, cluster_to_confidence)
        - pseudo_labels: Assigned pseudo labels for all samples (torch.Tensor)
        - cluster_to_label: Mapping from cluster ID to pseudo label
        - k_nearest_indices: Indices of k nearest samples per cluster (Dict[int, torch.Tensor])
        - confidence_scores: Sample-wise confidence scores (torch.Tensor or None)
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
            valid_mask = pseudo_labels != NO_PSEUDO_LABEL
            if torch.any(valid_mask):
                valid_mask = valid_mask.to(confidence_scores.device)
                avg_confidence = torch.mean(confidence_scores[valid_mask]).item()
                print(f"Average Confidence Score: {avg_confidence:.4f}")
                print(f"Confidence Score Range: [{torch.min(confidence_scores[valid_mask]).item():.4f}, "
                      f"{torch.max(confidence_scores[valid_mask]).item():.4f}]")
    
    return pseudo_labels, cluster_to_label, k_nearest_indices, confidence_scores, cluster_to_confidence


def save_pseudo_labels_to_csv(
    pseudo_labels: torch.Tensor,
    cluster_assignments: torch.Tensor,
    true_labels: torch.Tensor,
    confidence_scores: Optional[torch.Tensor],
    output_path: str,
    image_paths: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Save pseudo labels to a CSV file for easy inspection and downstream use.
    
    This function creates a comprehensive CSV file with:
    - Sample index
    - Image path (if provided)
    - Cluster assignment
    - Pseudo label
    - True label
    - Confidence score
    - Class names (if provided)
    
    Args:
        pseudo_labels: Pseudo labels for each sample (n_samples,)
        cluster_assignments: Cluster assignments for each sample (n_samples,)
        true_labels: True labels for each sample (n_samples,)
        confidence_scores: Confidence scores for each sample (n_samples,) or None
        output_path: Path to save the CSV file
        image_paths: Optional list of image paths for each sample
        class_names: Optional list of class names for label interpretation
    """
    import csv
    
    # Convert tensors to numpy
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.cpu().numpy()
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if confidence_scores is not None and isinstance(confidence_scores, torch.Tensor):
        confidence_scores = confidence_scores.cpu().numpy()
    
    n_samples = len(pseudo_labels)
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV header
    header = ['sample_idx', 'cluster_id', 'pseudo_label', 'true_label']
    if image_paths is not None:
        header.insert(1, 'image_path')
    if confidence_scores is not None:
        header.append('confidence_score')
    if class_names is not None:
        header.extend(['pseudo_label_name', 'true_label_name'])
    header.append('is_correct')
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        for i in range(n_samples):
            row = {
                'sample_idx': i,
                'cluster_id': int(cluster_assignments[i]),
                'pseudo_label': int(pseudo_labels[i]),
                'true_label': int(true_labels[i])
            }
            
            if image_paths is not None:
                row['image_path'] = image_paths[i] if i < len(image_paths) else ''
            
            if confidence_scores is not None:
                row['confidence_score'] = f"{confidence_scores[i]:.4f}"
            
            if class_names is not None:
                pseudo_label_val = int(pseudo_labels[i])
                true_label_val = int(true_labels[i])
                row['pseudo_label_name'] = class_names[pseudo_label_val] if 0 <= pseudo_label_val < len(class_names) else 'UNKNOWN'
                row['true_label_name'] = class_names[true_label_val] if 0 <= true_label_val < len(class_names) else 'UNKNOWN'
            
            row['is_correct'] = 1 if pseudo_labels[i] == true_labels[i] else 0
            
            writer.writerow(row)
    
    print(f"Pseudo labels CSV saved to {output_path}")


def save_cluster_mapping_to_csv(
    cluster_to_label: Dict[int, int],
    cluster_to_confidence: Dict[int, float],
    cluster_assignments: torch.Tensor,
    true_labels: torch.Tensor,
    output_path: str,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Save cluster-to-label mapping to a CSV file.
    
    This function creates a summary CSV with:
    - Cluster ID
    - Mapped pseudo label
    - Cluster size
    - Per-cluster accuracy
    - Confidence score
    - Class names (if provided)
    
    Args:
        cluster_to_label: Mapping from cluster ID to pseudo label
        cluster_to_confidence: Mapping from cluster ID to confidence score
        cluster_assignments: Cluster assignments for each sample
        true_labels: True labels for each sample
        output_path: Path to save the CSV file
        class_names: Optional list of class names
    """
    import csv
    
    # Convert tensors to numpy if needed
    if isinstance(cluster_assignments, torch.Tensor):
        cluster_assignments = cluster_assignments.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare header
    header = ['cluster_id', 'pseudo_label', 'cluster_size', 'cluster_accuracy', 'confidence']
    if class_names is not None:
        header.append('pseudo_label_name')
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        for cluster_id in sorted(cluster_to_label.keys()):
            pseudo_label = cluster_to_label[cluster_id]
            confidence = cluster_to_confidence.get(cluster_id, 0.0)
            
            # Compute cluster size and accuracy
            cluster_mask = cluster_assignments == cluster_id
            cluster_size = int(np.sum(cluster_mask))
            
            if cluster_size > 0 and pseudo_label != NO_PSEUDO_LABEL:
                cluster_true_labels = true_labels[cluster_mask]
                cluster_accuracy = float(np.mean(cluster_true_labels == pseudo_label))
            else:
                cluster_accuracy = 0.0
            
            row = {
                'cluster_id': cluster_id,
                'pseudo_label': pseudo_label,
                'cluster_size': cluster_size,
                'cluster_accuracy': f"{cluster_accuracy:.4f}",
                'confidence': f"{confidence:.4f}"
            }
            
            if class_names is not None and 0 <= pseudo_label < len(class_names):
                row['pseudo_label_name'] = class_names[pseudo_label]
            elif class_names is not None:
                row['pseudo_label_name'] = 'UNKNOWN' if pseudo_label == NO_PSEUDO_LABEL else f'class_{pseudo_label}'
            
            writer.writerow(row)
    
    print(f"Cluster mapping CSV saved to {output_path}")
