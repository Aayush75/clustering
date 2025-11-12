"""
Evaluation metrics for clustering performance.

This module provides various metrics to evaluate the quality of clustering
results, including accuracy, normalized mutual information, and adjusted rand index.
"""

import torch
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict, Union


def cluster_accuracy(y_true: Union[torch.Tensor, list], y_pred: Union[torch.Tensor, list]) -> float:
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    
    WARNING: This function should ONLY be used during the initial clustering phase
    to discover the optimal cluster-to-label mapping. It should NOT be used for
    evaluating distilled datasets or in deployment scenarios where the cluster-to-label
    mapping is already established.
    
    For evaluation with a fixed cluster-to-label mapping, use fixed_mapping_accuracy() instead.
    
    Since cluster labels are arbitrary during clustering, we need to find the best matching
    between predicted cluster labels and true class labels. This is done using the
    Hungarian algorithm for optimal assignment.
    
    Args:
        y_true: Ground truth labels (num_samples,) - torch.Tensor or list
        y_pred: Predicted cluster labels (num_samples,) - torch.Tensor or list
        
    Returns:
        Accuracy score between 0 and 1
    """
    # Convert to tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Ensure int64 dtype
    y_true = y_true.to(dtype=torch.int64)
    y_pred = y_pred.to(dtype=torch.int64)
    
    # Check that arrays have the same length
    assert y_true.shape[0] == y_pred.shape[0], "Input arrays must have same length"
    
    # Convert to CPU numpy for sklearn compatibility (only at final step)
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Build confusion matrix
    # Rows are true labels, columns are predicted clusters
    conf_matrix = confusion_matrix(y_true_np, y_pred_np)
    
    # Use Hungarian algorithm to find optimal assignment
    # This finds the best one-to-one mapping between clusters and classes
    row_indices, col_indices = linear_sum_assignment(-conf_matrix)
    
    # Calculate accuracy based on optimal assignment
    total_correct = conf_matrix[row_indices, col_indices].sum()
    accuracy = total_correct / y_true.shape[0]
    
    return accuracy


def fixed_mapping_accuracy(
    y_true: Union[torch.Tensor, list],
    y_pred: Union[torch.Tensor, list],
    cluster_to_label: Dict[int, int]
) -> float:
    """
    Calculate accuracy using a FIXED cluster-to-label mapping.
    
    This function should be used when evaluating distilled datasets or in deployment
    scenarios where the cluster-to-label mapping has already been established during
    the clustering/pseudo-labeling phase.
    
    Unlike cluster_accuracy() which uses Hungarian matching to find the optimal
    post-hoc alignment, this function respects the learned mapping from the training
    phase, which is critical for:
    1. Realistic evaluation that reflects deployment performance
    2. Proper assessment of dataset distillation quality
    3. Avoiding artificially inflated performance metrics
    
    Args:
        y_true: Ground truth labels (num_samples,) - torch.Tensor or list
        y_pred: Predicted cluster IDs (num_samples,) - torch.Tensor or list
        cluster_to_label: Dictionary mapping cluster_id -> label (from clustering phase)
        
    Returns:
        Accuracy score between 0 and 1 based on the fixed mapping
        
    Example:
        # During clustering, we establish: cluster_0 -> label_1, cluster_1 -> label_2
        cluster_to_label = {0: 1, 1: 2, 2: 0}
        
        # Model predicts cluster IDs: [0, 1, 2, 0, 1]
        predictions = torch.tensor([0, 1, 2, 0, 1])
        
        # Map to labels using the fixed mapping: [1, 2, 0, 1, 2]
        # Compare with true labels: [1, 2, 1, 1, 2]
        # Accuracy: 4/5 = 0.8
        true_labels = torch.tensor([1, 2, 1, 1, 2])
        acc = fixed_mapping_accuracy(true_labels, predictions, cluster_to_label)
    """
    # Convert to tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Ensure int64 dtype
    y_true = y_true.to(dtype=torch.int64)
    y_pred = y_pred.to(dtype=torch.int64)
    
    # Check that arrays have the same length
    assert y_true.shape[0] == y_pred.shape[0], "Input arrays must have same length"
    
    device = y_pred.device
    
    # Map predicted cluster IDs to labels using the fixed mapping
    mapped_predictions = torch.full_like(y_pred, -1, dtype=torch.int64, device=device)
    
    for cluster_id, label in cluster_to_label.items():
        mask = y_pred == cluster_id
        mapped_predictions[mask] = label
    
    # Filter out samples with unmapped clusters (cluster_id not in mapping)
    valid_mask = mapped_predictions != -1
    
    if not torch.any(valid_mask):
        # No valid predictions
        return 0.0
    
    # Calculate accuracy on valid samples
    correct = (mapped_predictions[valid_mask] == y_true[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy


def evaluate_clustering(
    y_true: Union[torch.Tensor, list],
    y_pred: Union[torch.Tensor, list],
    return_confusion_matrix: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive clustering evaluation metrics.
    
    This function calculates multiple standard metrics for evaluating
    clustering quality against ground truth labels.
    
    Args:
        y_true: Ground truth labels - torch.Tensor or list
        y_pred: Predicted cluster labels - torch.Tensor or list
        return_confusion_matrix: If True, include confusion matrix in results
        
    Returns:
        Dictionary containing evaluation metrics:
        - accuracy: Clustering accuracy (using Hungarian matching)
        - nmi: Normalized Mutual Information
        - ari: Adjusted Rand Index
        - confusion_matrix (optional): Confusion matrix
    """
    # Convert to tensors if needed
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Calculate clustering accuracy with Hungarian matching
    acc = cluster_accuracy(y_true, y_pred)
    
    # Convert to CPU numpy for sklearn compatibility (only at final step)
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    
    # Calculate Normalized Mutual Information
    # NMI measures the mutual information between cluster assignments and true labels
    # Range: [0, 1], higher is better
    nmi = normalized_mutual_info_score(y_true_np, y_pred_np, average_method='geometric')
    
    # Calculate Adjusted Rand Index
    # ARI measures the similarity between two clusterings
    # Range: [-1, 1], higher is better, 0 means random clustering
    ari = adjusted_rand_score(y_true_np, y_pred_np)
    
    results = {
        'accuracy': acc,
        'nmi': nmi,
        'ari': ari
    }
    
    if return_confusion_matrix:
        conf_mat = confusion_matrix(y_true_np, y_pred_np)
        results['confusion_matrix'] = conf_mat
    
    return results


def print_evaluation_results(results: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Pretty print evaluation results.
    
    Args:
        results: Dictionary of evaluation metrics
        dataset_name: Name of the dataset being evaluated
    """
    print(f"\n{'=' * 60}")
    print(f"Clustering Evaluation Results - {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy (Hungarian):  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Normalized MI:         {results['nmi']:.4f}")
    print(f"Adjusted Rand Index:   {results['ari']:.4f}")
    print(f"{'=' * 60}\n")


def analyze_cluster_distribution(y_pred: Union[torch.Tensor, list], num_clusters: int) -> Dict[str, any]:
    """
    Analyze the distribution of samples across clusters.
    
    This helps identify if clusters are balanced or if some clusters
    are dominating (cluster collapse).
    
    Args:
        y_pred: Predicted cluster labels - torch.Tensor or list
        num_clusters: Total number of clusters
        
    Returns:
        Dictionary with distribution statistics
    """
    # Convert to tensor if needed
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Count samples in each cluster using torch
    unique, counts = torch.unique(y_pred, return_counts=True)
    cluster_sizes = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
    
    # Calculate statistics using torch
    counts_tensor = counts.to(dtype=torch.float32)
    mean_size = torch.mean(counts_tensor).item()
    std_size = torch.std(counts_tensor).item()
    min_size = torch.min(counts_tensor).item()
    max_size = torch.max(counts_tensor).item()
    
    # Calculate coefficient of variation (std/mean)
    # Lower values indicate more balanced clusters
    cv = std_size / mean_size if mean_size > 0 else float('inf')
    
    # Count empty clusters
    empty_clusters = num_clusters - len(unique)
    
    return {
        'cluster_sizes': cluster_sizes,
        'mean_size': mean_size,
        'std_size': std_size,
        'min_size': min_size,
        'max_size': max_size,
        'cv': cv,
        'num_empty_clusters': empty_clusters,
        'num_active_clusters': len(unique)
    }


def print_cluster_distribution(distribution: Dict[str, any]):
    """
    Pretty print cluster distribution statistics.
    
    Args:
        distribution: Dictionary from analyze_cluster_distribution
    """
    print(f"\n{'=' * 60}")
    print("Cluster Distribution Analysis")
    print(f"{'=' * 60}")
    print(f"Active Clusters:       {distribution['num_active_clusters']}")
    print(f"Empty Clusters:        {distribution['num_empty_clusters']}")
    print(f"Mean Cluster Size:     {distribution['mean_size']:.1f}")
    print(f"Std Cluster Size:      {distribution['std_size']:.1f}")
    print(f"Min Cluster Size:      {distribution['min_size']}")
    print(f"Max Cluster Size:      {distribution['max_size']}")
    print(f"Coefficient of Var:    {distribution['cv']:.3f}")
    print(f"{'=' * 60}\n")
