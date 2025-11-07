"""
Script to analyze and visualize clustering results.

This script loads results from a completed experiment and provides
detailed analysis and visualizations.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
import sys
from src.visualization import visualize_clustering_results


def load_experiment_results(experiment_dir):
    """
    Load all results from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary containing all experiment data
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return None
    
    results = {}
    
    # Load config
    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            results['config'] = json.load(f)
    
    # Load metrics
    metrics_path = experiment_dir / "results.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results['metrics'] = json.load(f)
    
    # Load predictions
    predictions_path = experiment_dir / "predictions.npz"
    if predictions_path.exists():
        data = np.load(predictions_path)
        results['predictions'] = {
            'train_predictions': data['train_predictions'],
            'train_labels': data['train_labels'],
            'test_predictions': data['test_predictions'],
            'test_labels': data['test_labels']
        }
    
    return results


def print_summary(results):
    """
    Print a summary of the experiment results.
    
    Args:
        results: Dictionary of experiment results
    """
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    if 'config' in results:
        config = results['config']
        print("\nConfiguration:")
        print(f"  Experiment Name:     {config.get('experiment_name', 'N/A')}")
        print(f"  DINOv2 Model:        {config.get('dinov2_model', 'N/A')}")
        print(f"  Number of Clusters:  {config.get('num_clusters', 'N/A')}")
        print(f"  Training Epochs:     {config.get('num_epochs', 'N/A')}")
        print(f"  Learning Rate:       {config.get('learning_rate', 'N/A')}")
        print(f"  Temperature:         {config.get('temperature', 'N/A')}")
        print(f"  Batch Size:          {config.get('batch_size', 'N/A')}")
    
    if 'metrics' in results:
        metrics = results['metrics']
        print("\nTraining Set Results:")
        train = metrics['training']
        print(f"  Accuracy:            {train['accuracy']:.4f} ({train['accuracy']*100:.2f}%)")
        print(f"  NMI:                 {train['nmi']:.4f}")
        print(f"  ARI:                 {train['ari']:.4f}")
        print(f"  Active Clusters:     {train['num_active_clusters']}")
        print(f"  Empty Clusters:      {train['num_empty_clusters']}")
        
        print("\nTest Set Results:")
        test = metrics['test']
        print(f"  Accuracy:            {test['accuracy']:.4f} ({test['accuracy']*100:.2f}%)")
        print(f"  NMI:                 {test['nmi']:.4f}")
        print(f"  ARI:                 {test['ari']:.4f}")
        print(f"  Active Clusters:     {test['num_active_clusters']}")
        print(f"  Empty Clusters:      {test['num_empty_clusters']}")
    
    print("="*70)


def analyze_cluster_distribution(predictions, labels, num_clusters, split_name="Dataset"):
    """
    Analyze the distribution of clusters.
    
    Args:
        predictions: Cluster assignments
        labels: Ground truth labels
        num_clusters: Total number of clusters
        split_name: Name of the data split (e.g., "Training", "Test")
    """
    print(f"\n{split_name} Cluster Distribution Analysis")
    print("-"*70)
    
    # Count samples per cluster
    cluster_counts = Counter(predictions)
    
    # Statistics
    sizes = list(cluster_counts.values())
    print(f"Active Clusters:       {len(cluster_counts)}/{num_clusters}")
    print(f"Empty Clusters:        {num_clusters - len(cluster_counts)}")
    print(f"Mean Cluster Size:     {np.mean(sizes):.1f}")
    print(f"Std Cluster Size:      {np.std(sizes):.1f}")
    print(f"Min Cluster Size:      {np.min(sizes)}")
    print(f"Max Cluster Size:      {np.max(sizes)}")
    
    # Show largest clusters
    print("\nTop 10 Largest Clusters:")
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (cluster_id, count) in enumerate(sorted_clusters[:10], 1):
        percentage = 100 * count / len(predictions)
        print(f"  {i:2d}. Cluster {cluster_id:3d}: {count:5d} samples ({percentage:5.2f}%)")


def analyze_class_to_cluster_mapping(predictions, labels, num_classes=100):
    """
    Analyze how ground truth classes map to clusters.
    
    Args:
        predictions: Cluster assignments
        labels: Ground truth labels
        num_classes: Number of ground truth classes
    """
    print("\nClass to Cluster Mapping Analysis")
    print("-"*70)
    
    # For each class, find the dominant cluster
    print("\nTop classes and their dominant clusters:")
    
    class_to_cluster = {}
    for class_id in range(min(num_classes, 20)):  # Show first 20 classes
        mask = labels == class_id
        if np.sum(mask) == 0:
            continue
        
        class_predictions = predictions[mask]
        cluster_counts = Counter(class_predictions)
        
        if len(cluster_counts) == 0:
            continue
        
        # Find dominant cluster
        dominant_cluster, dominant_count = cluster_counts.most_common(1)[0]
        total_count = len(class_predictions)
        purity = dominant_count / total_count
        
        class_to_cluster[class_id] = {
            'cluster': dominant_cluster,
            'purity': purity,
            'total': total_count
        }
        
        print(f"  Class {class_id:2d}: {total_count:4d} samples -> "
              f"Cluster {dominant_cluster:3d} ({purity*100:5.1f}% purity)")
    
    # Overall purity statistics
    if class_to_cluster:
        purities = [info['purity'] for info in class_to_cluster.values()]
        print(f"\nOverall Purity Statistics:")
        print(f"  Mean Purity:   {np.mean(purities):.4f}")
        print(f"  Median Purity: {np.median(purities):.4f}")
        print(f"  Min Purity:    {np.min(purities):.4f}")
        print(f"  Max Purity:    {np.max(purities):.4f}")


def analyze_cluster_purity(predictions, labels):
    """
    Calculate purity for each cluster.
    
    Args:
        predictions: Cluster assignments
        labels: Ground truth labels
    """
    print("\nCluster Purity Analysis")
    print("-"*70)
    
    unique_clusters = np.unique(predictions)
    purities = []
    
    print("\nTop 10 Purest Clusters:")
    cluster_info = []
    
    for cluster_id in unique_clusters:
        mask = predictions == cluster_id
        cluster_labels = labels[mask]
        
        if len(cluster_labels) == 0:
            continue
        
        # Find most common class in this cluster
        label_counts = Counter(cluster_labels)
        dominant_label, dominant_count = label_counts.most_common(1)[0]
        purity = dominant_count / len(cluster_labels)
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'size': len(cluster_labels),
            'purity': purity,
            'dominant_label': dominant_label
        })
        purities.append(purity)
    
    # Sort by purity
    cluster_info.sort(key=lambda x: x['purity'], reverse=True)
    
    for i, info in enumerate(cluster_info[:10], 1):
        print(f"  {i:2d}. Cluster {info['cluster_id']:3d}: "
              f"{info['size']:4d} samples, {info['purity']*100:5.1f}% purity "
              f"(class {info['dominant_label']:2d})")
    
    print(f"\nOverall Cluster Purity Statistics:")
    print(f"  Mean Purity:   {np.mean(purities):.4f}")
    print(f"  Median Purity: {np.median(purities):.4f}")
    print(f"  Min Purity:    {np.min(purities):.4f}")
    print(f"  Max Purity:    {np.max(purities):.4f}")


def main():
    """
    Main function to analyze experiment results.
    """
    parser = argparse.ArgumentParser(
        description='Analyze TEMI clustering experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('experiment_dir', type=str,
                        help='Path to experiment directory')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed analysis including per-class and per-cluster stats')
    parser.add_argument('--plot', action='store_true',
                        help='Generate cluster visualizations')
    parser.add_argument('--viz_method', type=str, default='tsne',
                        choices=['tsne', 'umap'],
                        help='Dimensionality reduction method for visualization')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading experiment results...")
    results = load_experiment_results(args.experiment_dir)
    
    if results is None:
        sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Detailed analysis if requested
    if args.detailed and 'predictions' in results and 'config' in results:
        preds = results['predictions']
        config = results['config']
        num_clusters = config.get('num_clusters', 100)
        
        # Analyze training set
        print("\n" + "="*70)
        print("DETAILED ANALYSIS - TRAINING SET")
        print("="*70)
        
        analyze_cluster_distribution(
            preds['train_predictions'],
            preds['train_labels'],
            num_clusters,
            "Training Set"
        )
        
        analyze_class_to_cluster_mapping(
            preds['train_predictions'],
            preds['train_labels']
        )
        
        analyze_cluster_purity(
            preds['train_predictions'],
            preds['train_labels']
        )
        
        # Analyze test set
        print("\n" + "="*70)
        print("DETAILED ANALYSIS - TEST SET")
        print("="*70)
        
        analyze_cluster_distribution(
            preds['test_predictions'],
            preds['test_labels'],
            num_clusters,
            "Test Set"
        )
        
        analyze_class_to_cluster_mapping(
            preds['test_predictions'],
            preds['test_labels']
        )
        
        analyze_cluster_purity(
            preds['test_predictions'],
            preds['test_labels']
        )
    
    # Generate visualizations if requested
    if args.plot and 'predictions' in results and 'config' in results:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        preds = results['predictions']
        config = results['config']
        num_clusters = config.get('num_clusters', 100)
        
        # Check if features are available
        experiment_dir = Path(args.experiment_dir)
        features_dir = experiment_dir / "features"
        
        if features_dir.exists():
            train_features_path = features_dir / "train_features.pt"
            test_features_path = features_dir / "test_features.pt"
            
            if train_features_path.exists() and test_features_path.exists():
                import torch
                print("\nLoading features for visualization...")
                
                # Load features (keep as tensors - visualization functions handle conversion)
                train_data = torch.load(train_features_path)
                test_data = torch.load(test_features_path)
                
                train_features = train_data['features']
                test_features = test_data['features']
                
                viz_dir = experiment_dir / "visualizations"
                
                # Visualize training set (pass tensors directly)
                visualize_clustering_results(
                    features=train_features,
                    labels=preds['train_labels'],
                    predictions=preds['train_predictions'],
                    num_clusters=num_clusters,
                    output_dir=str(viz_dir),
                    dataset_name="Training",
                    method=args.viz_method,
                    show_plots=args.show_plots
                )
                
                # Visualize test set (pass tensors directly)
                visualize_clustering_results(
                    features=test_features,
                    labels=preds['test_labels'],
                    predictions=preds['test_predictions'],
                    num_clusters=num_clusters,
                    output_dir=str(viz_dir),
                    dataset_name="Test",
                    method=args.viz_method,
                    show_plots=args.show_plots
                )
                
                print(f"\nVisualizations saved to {viz_dir}")
            else:
                print("\nFeatures not found. Cannot generate visualizations.")
                print("Run the experiment with --save_features to enable visualization.")
        else:
            print("\nFeatures directory not found. Cannot generate visualizations.")
            print("Run the experiment with --save_features to enable visualization.")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
