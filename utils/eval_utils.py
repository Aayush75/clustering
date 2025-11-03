"""
Evaluation metrics for clustering performance.

This module provides various metrics to evaluate clustering quality including
accuracy, NMI, ARI, and ANMI using the Hungarian algorithm for cluster assignment.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score
)


def compute_cluster_accuracy(targets, predictions):
    """
    Compute clustering accuracy using Hungarian algorithm.
    
    The Hungarian algorithm finds the best one-to-one mapping between
    predicted clusters and ground truth classes.
    
    Args:
        targets: Ground truth labels (N,)
        predictions: Predicted cluster assignments (N,)
        
    Returns:
        Tuple of (accuracy, reassignment_mapping)
    """
    # Ensure numpy arrays
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Build confusion matrix
    num_classes = max(targets.max(), predictions.max()) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for i in range(len(targets)):
        confusion_matrix[predictions[i], targets[i]] += 1
    
    # Apply Hungarian algorithm to find optimal assignment
    # We maximize the sum, so we use negative of confusion matrix
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    
    # Compute accuracy
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(targets)
    
    # Create reassignment mapping
    reassignment = np.zeros(num_classes, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        reassignment[i] = j
    
    return accuracy * 100.0, reassignment


def compute_nmi(targets, predictions):
    """
    Compute Normalized Mutual Information.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted cluster assignments
        
    Returns:
        NMI score (0-100)
    """
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    score = normalized_mutual_info_score(targets, predictions)
    return score * 100.0


def compute_adjusted_nmi(targets, predictions):
    """
    Compute Adjusted Normalized Mutual Information.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted cluster assignments
        
    Returns:
        Adjusted NMI score (0-100)
    """
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    score = adjusted_mutual_info_score(targets, predictions)
    return score * 100.0


def compute_ari(targets, predictions):
    """
    Compute Adjusted Rand Index.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted cluster assignments
        
    Returns:
        ARI score (0-100)
    """
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    score = adjusted_rand_score(targets, predictions)
    return score * 100.0


def compute_all_metrics(targets, predictions):
    """
    Compute all clustering metrics at once.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted cluster assignments
        
    Returns:
        Dictionary containing all metrics
    """
    accuracy, reassignment = compute_cluster_accuracy(targets, predictions)
    nmi = compute_nmi(targets, predictions)
    anmi = compute_adjusted_nmi(targets, predictions)
    ari = compute_ari(targets, predictions)
    
    return {
        'accuracy': accuracy,
        'nmi': nmi,
        'adjusted_nmi': anmi,
        'ari': ari,
        'reassignment': reassignment
    }


def print_metrics(metrics, prefix=""):
    """
    Print clustering metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for the output
    """
    print(f"\n{prefix}Clustering Metrics:")
    print(f"  Accuracy:      {metrics['accuracy']:.2f}%")
    print(f"  NMI:           {metrics['nmi']:.2f}%")
    print(f"  Adjusted NMI:  {metrics['adjusted_nmi']:.2f}%")
    print(f"  ARI:           {metrics['ari']:.2f}%")


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, temperature=0.07):
    """
    K-nearest neighbors classifier for embeddings.
    
    This provides a baseline accuracy by performing KNN in the embedding space.
    
    Args:
        train_features: Training features (N_train, D)
        train_labels: Training labels (N_train,)
        test_features: Test features (N_test, D)
        test_labels: Test labels (N_test,)
        k: Number of neighbors
        temperature: Temperature for distance weighting
        
    Returns:
        Tuple of (top1_accuracy, top5_accuracy)
    """
    # Ensure tensors
    if not torch.is_tensor(train_features):
        train_features = torch.tensor(train_features)
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels)
    if not torch.is_tensor(test_features):
        test_features = torch.tensor(test_features)
    if not torch.is_tensor(test_labels):
        test_labels = torch.tensor(test_labels)
    
    num_classes = int(train_labels.max()) + 1
    num_test = test_features.shape[0]
    
    # Normalize features
    train_features = F.normalize(train_features, dim=1, p=2)
    test_features = F.normalize(test_features, dim=1, p=2)
    
    # Move to same device
    device = train_features.device
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    top1 = 0
    top5 = 0
    
    # Process in batches to avoid memory issues
    batch_size = 100
    
    for idx in range(0, num_test, batch_size):
        end_idx = min(idx + batch_size, num_test)
        batch_features = test_features[idx:end_idx]
        batch_labels = test_labels[idx:end_idx]
        batch_size_actual = batch_features.shape[0]
        
        # Compute similarity with training features
        similarity = torch.mm(batch_features, train_features.t())
        
        # Get top-k neighbors
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        
        # Get labels of neighbors
        candidates = train_labels.view(1, -1).expand(batch_size_actual, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        
        # Create one-hot encoding
        retrieval_one_hot = torch.zeros(batch_size_actual * k, num_classes, device=device)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        
        # Weight by distance
        distances_transform = distances.clone().div_(temperature).exp_()
        
        # Compute weighted votes
        probs = torch.mul(
            retrieval_one_hot.view(batch_size_actual, -1, num_classes),
            distances_transform.view(batch_size_actual, -1, 1),
        )
        probs = torch.sum(probs, 1)
        
        # Get predictions
        _, predictions = probs.sort(1, True)
        
        # Find matches
        correct = predictions.eq(batch_labels.data.view(-1, 1))
        top1 += correct.narrow(1, 0, 1).sum().item()
        top5 += correct.narrow(1, 0, min(5, k)).sum().item()
    
    top1 = top1 * 100.0 / num_test
    top5 = top5 * 100.0 / num_test
    
    return top1, top5


def compute_cluster_statistics(predictions, num_clusters):
    """
    Compute statistics about cluster assignments.
    
    Args:
        predictions: Predicted cluster assignments (N,)
        num_clusters: Expected number of clusters
        
    Returns:
        Dictionary of statistics
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    unique_clusters, counts = np.unique(predictions, return_counts=True)
    
    stats = {
        'num_occupied_clusters': len(unique_clusters),
        'num_total_clusters': num_clusters,
        'occupancy_rate': len(unique_clusters) / num_clusters * 100,
        'min_cluster_size': counts.min(),
        'max_cluster_size': counts.max(),
        'mean_cluster_size': counts.mean(),
        'std_cluster_size': counts.std(),
    }
    
    return stats


def print_cluster_statistics(stats, prefix=""):
    """
    Print cluster statistics in a formatted way.
    
    Args:
        stats: Dictionary of statistics
        prefix: Optional prefix for output
    """
    print(f"\n{prefix}Cluster Statistics:")
    print(f"  Occupied clusters: {stats['num_occupied_clusters']}/{stats['num_total_clusters']} "
          f"({stats['occupancy_rate']:.1f}%)")
    print(f"  Cluster sizes - Min: {stats['min_cluster_size']:.0f}, "
          f"Max: {stats['max_cluster_size']:.0f}, "
          f"Mean: {stats['mean_cluster_size']:.1f} (±{stats['std_cluster_size']:.1f})")
