"""
Example usage of pseudo label generation for TEMI clustering.

This script demonstrates how to use the pseudo labeling functionality
both programmatically and via command-line interface.
"""

def example_1_with_main_pipeline():
    """
    Example 1: Generate pseudo labels during main clustering run.
    """
    print("\n" + "="*70)
    print("Example 1: Pseudo Labels During Main Pipeline")
    print("="*70)
    print("""
Run TEMI clustering with pseudo label generation:

    python main.py \\
        --dataset cifar100 \\
        --num_clusters 100 \\
        --num_epochs 100 \\
        --generate_pseudo_labels \\
        --k_samples 10 \\
        --visualize_mapping \\
        --max_clusters_viz 20 \\
        --samples_per_cluster 5 \\
        --save_features

This will:
1. Train TEMI clustering on CIFAR100
2. Generate pseudo labels using 10 nearest samples per cluster
3. Create visualization showing representative images
4. Save results to pseudo_labels/ directory
    """)


def example_2_from_existing_results():
    """
    Example 2: Generate pseudo labels from existing results.
    """
    print("\n" + "="*70)
    print("Example 2: Pseudo Labels From Existing Results")
    print("="*70)
    print("""
If you already have clustering results, generate pseudo labels without re-running:

    python generate_pseudo_labels.py \\
        --experiment_dir ./results/temi_cifar100_100clusters_20241107_143025 \\
        --k_samples 10 \\
        --visualize \\
        --max_clusters_viz 20 \\
        --samples_per_cluster 5

This will:
1. Load pre-computed features and cluster assignments
2. Generate pseudo labels
3. Create visualization
4. Save results to experiment_dir/pseudo_labels/
    """)


def example_3_programmatic_usage():
    """
    Example 3: Use pseudo labeling in your own Python code.
    """
    print("\n" + "="*70)
    print("Example 3: Programmatic Usage")
    print("="*70)
    print("""
Use the pseudo labeling API in your Python code:

```python
from src.pseudo_labeling import (
    generate_pseudo_labels,
    print_cluster_mapping_summary,
    visualize_cluster_mapping
)
import torch
import numpy as np

# Assuming you have these from your clustering experiment:
# - features: torch.Tensor of shape (n_samples, n_features)
# - cluster_assignments: np.ndarray of shape (n_samples,)
# - true_labels: np.ndarray of shape (n_samples,)
# - cluster_centers: torch.Tensor of shape (n_clusters, n_features)

# Generate pseudo labels
pseudo_labels, cluster_to_label, k_nearest_indices = generate_pseudo_labels(
    features=features,
    cluster_assignments=cluster_assignments,
    true_labels=true_labels,
    cluster_centers=cluster_centers,
    k=10,
    verbose=True
)

# Print detailed summary
print_cluster_mapping_summary(
    cluster_to_label=cluster_to_label,
    cluster_assignments=cluster_assignments,
    true_labels=true_labels,
    class_names=class_names  # optional
)

# Create visualization (requires images)
visualize_cluster_mapping(
    images=images,
    true_labels=true_labels,
    cluster_assignments=cluster_assignments,
    cluster_to_label=cluster_to_label,
    k_nearest_indices=k_nearest_indices,
    save_path='cluster_mapping.png',
    class_names=class_names,
    max_clusters_to_show=20,
    samples_per_cluster=5
)
```
    """)


def example_4_comparing_k_values():
    """
    Example 4: Compare different k values.
    """
    print("\n" + "="*70)
    print("Example 4: Comparing Different k Values")
    print("="*70)
    print("""
Try different k values to see how it affects the mapping:

    # Generate with k=5 (very strict, only most central samples)
    python generate_pseudo_labels.py \\
        --experiment_dir ./results/my_experiment \\
        --k_samples 5

    # Generate with k=10 (recommended for CIFAR100)
    python generate_pseudo_labels.py \\
        --experiment_dir ./results/my_experiment \\
        --k_samples 10

    # Generate with k=20 (more robust, includes more samples)
    python generate_pseudo_labels.py \\
        --experiment_dir ./results/my_experiment \\
        --k_samples 20

    # Generate with k=50 (very inclusive)
    python generate_pseudo_labels.py \\
        --experiment_dir ./results/my_experiment \\
        --k_samples 50

Results for each k will be saved separately as:
- pseudo_labels/pseudo_labels_k5.json
- pseudo_labels/pseudo_labels_k10.json
- pseudo_labels/pseudo_labels_k20.json
- pseudo_labels/pseudo_labels_k50.json

Compare the accuracy metrics to see which k works best for your data.
    """)


def example_5_imagenet():
    """
    Example 5: Pseudo labels for ImageNet.
    """
    print("\n" + "="*70)
    print("Example 5: Pseudo Labels for ImageNet")
    print("="*70)
    print("""
For ImageNet (larger dataset, more classes), use larger k:

    python main.py \\
        --dataset imagenet \\
        --num_clusters 1000 \\
        --num_epochs 100 \\
        --generate_pseudo_labels \\
        --k_samples 20 \\
        --visualize_mapping \\
        --max_clusters_viz 30 \\
        --samples_per_cluster 5 \\
        --save_features

Or from existing results:

    python generate_pseudo_labels.py \\
        --experiment_dir ./results/temi_imagenet_1000clusters \\
        --k_samples 20 \\
        --visualize \\
        --max_clusters_viz 30
    """)


def example_6_with_clip():
    """
    Example 6: Pseudo labels with CLIP features.
    """
    print("\n" + "="*70)
    print("Example 6: Pseudo Labels with CLIP Features")
    print("="*70)
    print("""
Generate pseudo labels when using CLIP instead of DINOv2:

    python main.py \\
        --model_type clip \\
        --clip_model openai/clip-vit-large-patch14 \\
        --num_clusters 100 \\
        --generate_pseudo_labels \\
        --k_samples 10 \\
        --visualize_mapping \\
        --save_features

The pseudo labeling works the same regardless of feature extractor!
    """)


def main():
    """Display all examples."""
    print("\n" + "="*70)
    print("TEMI Clustering - Pseudo Label Generation Examples")
    print("="*70)
    print("""
This script demonstrates various ways to use pseudo label generation
for TEMI clustering results.

For complete documentation, see PSEUDO_LABELING_GUIDE.md
    """)
    
    example_1_with_main_pipeline()
    example_2_from_existing_results()
    example_3_programmatic_usage()
    example_4_comparing_k_values()
    example_5_imagenet()
    example_6_with_clip()
    
    print("\n" + "="*70)
    print("Key Takeaways")
    print("="*70)
    print("""
1. Use --generate_pseudo_labels during main run for convenience
2. Use generate_pseudo_labels.py script for existing results
3. Adjust k based on dataset size (k=10 for CIFAR100, k=20 for ImageNet)
4. Always use --save_features to enable pseudo label generation later
5. Visualizations help verify cluster quality and semantic meaning
6. Compare different k values to find optimal setting for your data

For more information:
- Complete guide: PSEUDO_LABELING_GUIDE.md
- API documentation: src/pseudo_labeling.py
- Main README: README.md
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
