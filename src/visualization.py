"""
Visualization module for clustering results.

This module provides functionality to visualize high-dimensional clustering
results using dimensionality reduction techniques like t-SNE and UMAP.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


def plot_clusters_tsne(
    features: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    predictions: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Cluster Visualization (t-SNE)",
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: Tuple[int, int] = (16, 6),
    show_plot: bool = False
) -> None:
    """
    Visualize clusters using t-SNE dimensionality reduction.
    
    Creates two side-by-side plots:
    - Left: colored by predicted clusters
    - Right: colored by ground truth labels
    
    Args:
        features: Feature array of shape (n_samples, n_features) - torch.Tensor or np.ndarray
        labels: Ground truth labels of shape (n_samples,) - torch.Tensor or np.ndarray
        predictions: Predicted cluster labels of shape (n_samples,) - torch.Tensor or np.ndarray
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        perplexity: t-SNE perplexity parameter (default: 30)
        n_iter: Number of t-SNE iterations (default: 1000)
        random_state: Random seed for reproducibility
        figsize: Figure size as (width, height)
        show_plot: Whether to display the plot interactively
    """
    print(f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    
    # Convert to numpy only at the final step for sklearn compatibility
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=0
    )
    features_2d = tsne.fit_transform(features)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Predicted clusters
    scatter1 = axes[0].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=predictions,
        cmap='tab20',
        s=10,
        alpha=0.6
    )
    axes[0].set_title(f'Predicted Clusters\n(k={len(np.unique(predictions))} clusters)')
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
    
    # Plot 2: Ground truth labels
    scatter2 = axes[1].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab20',
        s=10,
        alpha=0.6
    )
    axes[1].set_title(f'Ground Truth Labels\n(n={len(np.unique(labels))} classes)')
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter2, ax=axes[1], label='Class ID')
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_clusters_umap(
    features: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    predictions: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Cluster Visualization (UMAP)",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    figsize: Tuple[int, int] = (16, 6),
    show_plot: bool = False
) -> None:
    """
    Visualize clusters using UMAP dimensionality reduction.
    
    Creates two side-by-side plots:
    - Left: colored by predicted clusters
    - Right: colored by ground truth labels
    
    Args:
        features: Feature array of shape (n_samples, n_features) - torch.Tensor or np.ndarray
        labels: Ground truth labels of shape (n_samples,) - torch.Tensor or np.ndarray
        predictions: Predicted cluster labels of shape (n_samples,) - torch.Tensor or np.ndarray
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        n_neighbors: UMAP n_neighbors parameter (default: 15)
        min_dist: UMAP min_dist parameter (default: 0.1)
        random_state: Random seed for reproducibility
        figsize: Figure size as (width, height)
        show_plot: Whether to display the plot interactively
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP is not installed. Install with: pip install umap-learn"
        )
    
    print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    
    # Convert to numpy only at the final step for sklearn compatibility
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    elif not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False
    )
    features_2d = reducer.fit_transform(features)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Predicted clusters
    scatter1 = axes[0].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=predictions,
        cmap='tab20',
        s=10,
        alpha=0.6
    )
    axes[0].set_title(f'Predicted Clusters\n(k={len(np.unique(predictions))} clusters)')
    axes[0].set_xlabel('UMAP Component 1')
    axes[0].set_ylabel('UMAP Component 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster ID')
    
    # Plot 2: Ground truth labels
    scatter2 = axes[1].scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels,
        cmap='tab20',
        s=10,
        alpha=0.6
    )
    axes[1].set_title(f'Ground Truth Labels\n(n={len(np.unique(labels))} classes)')
    axes[1].set_xlabel('UMAP Component 1')
    axes[1].set_ylabel('UMAP Component 2')
    plt.colorbar(scatter2, ax=axes[1], label='Class ID')
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_cluster_distribution(
    predictions: np.ndarray,
    num_clusters: int,
    save_path: Optional[str] = None,
    title: str = "Cluster Distribution",
    figsize: Tuple[int, int] = (12, 6),
    show_plot: bool = False
) -> None:
    """
    Plot the distribution of samples across clusters.
    
    Args:
        predictions: Predicted cluster labels of shape (n_samples,)
        num_clusters: Total number of clusters
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
        figsize: Figure size as (width, height)
        show_plot: Whether to display the plot interactively
    """
    # Convert to numpy if needed
    if not isinstance(predictions, np.ndarray):
        # Handle torch tensors
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        else:
            predictions = np.array(predictions)
    
    # Count samples per cluster
    unique, counts = np.unique(predictions, return_counts=True)
    
    # Create array with all clusters (including empty ones)
    all_counts = np.zeros(num_clusters, dtype=int)
    all_counts[unique] = counts
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(
        range(num_clusters),
        all_counts,
        color='steelblue',
        alpha=0.7,
        edgecolor='black'
    )
    
    # Highlight empty clusters
    for i, count in enumerate(all_counts):
        if count == 0:
            bars[i].set_color('lightcoral')
            bars[i].set_alpha(0.5)
    
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    active_clusters = np.sum(all_counts > 0)
    empty_clusters = num_clusters - active_clusters
    mean_size = np.mean(all_counts[all_counts > 0]) if active_clusters > 0 else 0
    
    stats_text = (
        f"Active clusters: {active_clusters}/{num_clusters}\n"
        f"Empty clusters: {empty_clusters}\n"
        f"Mean cluster size: {mean_size:.1f}"
    )
    ax.text(
        0.98, 0.97,
        stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_clustering_results(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_clusters: int,
    output_dir: str,
    dataset_name: str = "Dataset",
    method: str = "tsne",
    show_plots: bool = False
) -> None:
    """
    Create comprehensive visualization of clustering results.
    
    This function generates multiple plots:
    - Dimensionality reduction plot (t-SNE or UMAP)
    - Cluster distribution bar plot
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Ground truth labels of shape (n_samples,)
        predictions: Predicted cluster labels of shape (n_samples,)
        num_clusters: Total number of clusters
        output_dir: Directory to save plots
        dataset_name: Name of the dataset (for plot titles)
        method: Dimensionality reduction method ('tsne' or 'umap')
        show_plots: Whether to display plots interactively
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {dataset_name} visualizations...")
    
    # 1. Dimensionality reduction plot
    if method.lower() == 'umap':
        if UMAP_AVAILABLE:
            plot_clusters_umap(
                features=features,
                labels=labels,
                predictions=predictions,
                save_path=str(output_path / f"{dataset_name.lower()}_clusters_umap.png"),
                title=f"{dataset_name} - Cluster Visualization (UMAP)",
                show_plot=show_plots
            )
        else:
            print("UMAP not available, falling back to t-SNE")
            method = 'tsne'
    
    if method.lower() == 'tsne':
        plot_clusters_tsne(
            features=features,
            labels=labels,
            predictions=predictions,
            save_path=str(output_path / f"{dataset_name.lower()}_clusters_tsne.png"),
            title=f"{dataset_name} - Cluster Visualization (t-SNE)",
            show_plot=show_plots
        )
    
    # 2. Cluster distribution plot
    plot_cluster_distribution(
        predictions=predictions,
        num_clusters=num_clusters,
        save_path=str(output_path / f"{dataset_name.lower()}_distribution.png"),
        title=f"{dataset_name} - Cluster Distribution",
        show_plot=show_plots
    )
    
    print(f"{dataset_name} visualizations complete!")
