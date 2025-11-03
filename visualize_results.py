"""
Visualization script for clustering results.

This script loads saved results and creates visualizations of the clustering
performance over training.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_results(results_path):
    """
    Load results from JSON file.
    
    Args:
        results_path: Path to results JSON file
        
    Returns:
        Dictionary of results
    """
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_curves(training_stats, save_path=None):
    """
    Plot training curves for loss and metrics.
    
    Args:
        training_stats: List of training statistics per epoch
        save_path: Optional path to save the figure
    """
    if not training_stats:
        print("No training statistics to plot")
        return
    
    epochs = [s['epoch'] for s in training_stats]
    losses = [s['train_loss'] for s in training_stats]
    accuracies = [s['test_accuracy'] for s in training_stats]
    nmis = [s['test_nmi'] for s in training_stats]
    aris = [s['test_ari'] for s in training_stats]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TEMI Clustering Training Progress', fontsize=16, fontweight='bold')
    
    # Plot training loss
    axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(epochs, accuracies, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Clustering Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # Plot NMI
    axes[1, 0].plot(epochs, nmis, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('NMI (%)')
    axes[1, 0].set_title('Normalized Mutual Information')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 100])
    
    # Plot ARI
    axes[1, 1].plot(epochs, aris, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ARI (%)')
    axes[1, 1].set_title('Adjusted Rand Index')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def print_summary(results):
    """
    Print a summary of the results.
    
    Args:
        results: Dictionary of results
    """
    print("\n" + "="*80)
    print("TEMI Clustering Results Summary")
    print("="*80)
    
    print("\nConfiguration:")
    config = results['config']
    print(f"  Number of clusters: {config['num_clusters']}")
    print(f"  Number of heads: {config['num_heads']}")
    print(f"  Number of epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Beta parameter: {config['beta']}")
    
    print("\nFinal Performance:")
    metrics = results['final_metrics']
    print(f"  Accuracy:      {metrics['accuracy']:.2f}%")
    print(f"  NMI:           {metrics['nmi']:.2f}%")
    print(f"  Adjusted NMI:  {metrics['adjusted_nmi']:.2f}%")
    print(f"  ARI:           {metrics['ari']:.2f}%")
    
    print("\nCluster Statistics:")
    print(f"  Occupied clusters: {metrics['num_occupied_clusters']}/{metrics['num_total_clusters']} "
          f"({metrics['occupancy_rate']:.1f}%)")
    print(f"  Mean cluster size: {metrics['mean_cluster_size']:.1f} (±{metrics['std_cluster_size']:.1f})")
    print(f"  Size range: {metrics['min_cluster_size']:.0f} - {metrics['max_cluster_size']:.0f}")
    
    print("\nBest Performance:")
    print(f"  Best accuracy: {results['best_accuracy']:.2f}%")
    
    print("\n" + "="*80)


def main():
    """
    Main function to visualize results.
    """
    results_dir = Path("results")
    results_file = results_dir / "final_results.json"
    
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run training first: python train.py")
        return
    
    # Load results
    print(f"Loading results from {results_file}")
    results = load_results(results_file)
    
    # Print summary
    print_summary(results)
    
    # Plot training curves
    if results['training_stats']:
        print("\nGenerating training curves...")
        plot_path = results_dir / "training_curves.png"
        plot_training_curves(results['training_stats'], save_path=plot_path)
        print(f"Training curves saved to {plot_path}")
    else:
        print("\nNo training statistics available for plotting")


if __name__ == "__main__":
    main()
