"""
Evaluation metrics for clustering performance.

This module provides various metrics to evaluate the quality of clustering
results, including accuracy, normalized mutual information, and adjusted rand index.
"""

import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    
    Since cluster labels are arbitrary, we need to find the best matching
    between predicted cluster labels and true class labels. This is done
    using the Hungarian algorithm for optimal assignment.
    
    Args:
        y_true: Ground truth labels (num_samples,)
        y_pred: Predicted cluster labels (num_samples,)
        
    Returns:
        Accuracy score between 0 and 1
    """
    # Ensure inputs are numpy arrays
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    # Check that arrays have the same length
    assert y_true.shape[0] == y_pred.shape[0], "Input arrays must have same length"
    
    # Build confusion matrix
    # Rows are true labels, columns are predicted clusters
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Use Hungarian algorithm to find optimal assignment
    # This finds the best one-to-one mapping between clusters and classes
    row_indices, col_indices = linear_sum_assignment(-conf_matrix)
    
    # Calculate accuracy based on optimal assignment
    total_correct = conf_matrix[row_indices, col_indices].sum()
    accuracy = total_correct / y_true.shape[0]
    
    return accuracy


def evaluate_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_confusion_matrix: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive clustering evaluation metrics.
    
    This function calculates multiple standard metrics for evaluating
    clustering quality against ground truth labels.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster labels
        return_confusion_matrix: If True, include confusion matrix in results
        
    Returns:
        Dictionary containing evaluation metrics:
        - accuracy: Clustering accuracy (using Hungarian matching)
        - nmi: Normalized Mutual Information
        - ari: Adjusted Rand Index
        - confusion_matrix (optional): Confusion matrix
    """
    # Calculate clustering accuracy with Hungarian matching
    acc = cluster_accuracy(y_true, y_pred)
    
    # Calculate Normalized Mutual Information
    # NMI measures the mutual information between cluster assignments and true labels
    # Range: [0, 1], higher is better
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='geometric')
    
    # Calculate Adjusted Rand Index
    # ARI measures the similarity between two clusterings
    # Range: [-1, 1], higher is better, 0 means random clustering
    ari = adjusted_rand_score(y_true, y_pred)
    
    results = {
        'accuracy': acc,
        'nmi': nmi,
        'ari': ari
    }
    
    if return_confusion_matrix:
        conf_mat = confusion_matrix(y_true, y_pred)
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


def analyze_cluster_distribution(y_pred: np.ndarray, num_clusters: int) -> Dict[str, any]:
    """
    Analyze the distribution of samples across clusters.
    
    This helps identify if clusters are balanced or if some clusters
    are dominating (cluster collapse).
    
    Args:
        y_pred: Predicted cluster labels
        num_clusters: Total number of clusters
        
    Returns:
        Dictionary with distribution statistics
    """
    # Count samples in each cluster
    unique, counts = np.unique(y_pred, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # Calculate statistics
    counts_array = np.array(list(cluster_sizes.values()))
    mean_size = np.mean(counts_array)
    std_size = np.std(counts_array)
    min_size = np.min(counts_array)
    max_size = np.max(counts_array)
    
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
