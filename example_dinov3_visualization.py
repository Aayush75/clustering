"""
Example script demonstrating DINOv3 support and cluster visualization.

This script shows how to:
1. Use DINOv3 models for feature extraction
2. Generate cluster visualizations
3. Reuse saved features to avoid re-running experiments
"""

import os
import sys
import argparse


def example_dinov3_with_visualization():
    """
    Example 1: Run clustering with DINOv3 and generate visualizations.
    """
    print("\n" + "="*70)
    print("Example 1: DINOv3 Clustering with Visualization")
    print("="*70)
    
    print("\nThis example demonstrates:")
    print("  • Using DINOv3 model for feature extraction")
    print("  • Generating t-SNE visualizations")
    print("  • Saving features for later reuse")
    
    print("\nCommand to run:")
    cmd = """python main.py \\
    --dinov2_model facebook/dinov3-base \\
    --num_clusters 100 \\
    --num_epochs 50 \\
    --batch_size 256 \\
    --plot_clusters \\
    --viz_method tsne \\
    --save_features \\
    --experiment_name dinov3_example"""
    
    print(cmd)
    print("\nNote: Replace 'facebook/dinov3-base' with any DINOv3 model from HuggingFace")


def example_compare_dinov2_dinov3():
    """
    Example 2: Compare DINOv2 and DINOv3 models.
    """
    print("\n" + "="*70)
    print("Example 2: Comparing DINOv2 and DINOv3")
    print("="*70)
    
    print("\nThis example shows how to run experiments with both models:")
    
    print("\n1. Run with DINOv2:")
    cmd_v2 = """python main.py \\
    --dinov2_model facebook/dinov2-base \\
    --num_clusters 100 \\
    --num_epochs 50 \\
    --plot_clusters \\
    --save_features \\
    --experiment_name comparison_dinov2"""
    print(cmd_v2)
    
    print("\n2. Run with DINOv3:")
    cmd_v3 = """python main.py \\
    --dinov2_model facebook/dinov3-base \\
    --num_clusters 100 \\
    --num_epochs 50 \\
    --plot_clusters \\
    --save_features \\
    --experiment_name comparison_dinov3"""
    print(cmd_v3)
    
    print("\n3. Compare results:")
    print("python analyze_results.py ./results/comparison_dinov2 --detailed")
    print("python analyze_results.py ./results/comparison_dinov3 --detailed")


def example_reuse_features():
    """
    Example 3: Reuse saved features to avoid re-running experiments.
    """
    print("\n" + "="*70)
    print("Example 3: Reusing Features (Avoid Re-running Experiments)")
    print("="*70)
    
    print("\nThis is the most efficient way to experiment!")
    
    print("\n1. First run - extract and save features:")
    cmd1 = """python main.py \\
    --dinov2_model facebook/dinov2-base \\
    --save_features \\
    --num_epochs 100 \\
    --experiment_name base_experiment"""
    print(cmd1)
    
    print("\n2. Subsequent runs - reuse features with different hyperparameters:")
    
    print("\n   Try different number of clusters:")
    cmd2 = """python main.py \\
    --load_features ./results/base_experiment/features/train_features \\
    --num_clusters 50 \\
    --num_epochs 100 \\
    --plot_clusters \\
    --experiment_name experiment_k50"""
    print(cmd2)
    
    print("\n   Try different temperature:")
    cmd3 = """python main.py \\
    --load_features ./results/base_experiment/features/train_features \\
    --num_clusters 100 \\
    --temperature 0.05 \\
    --num_epochs 100 \\
    --plot_clusters \\
    --experiment_name experiment_temp005"""
    print(cmd3)
    
    print("\n   Try different learning rate:")
    cmd4 = """python main.py \\
    --load_features ./results/base_experiment/features/train_features \\
    --num_clusters 100 \\
    --learning_rate 0.001 \\
    --num_epochs 100 \\
    --plot_clusters \\
    --experiment_name experiment_lr001"""
    print(cmd4)
    
    print("\nBenefit: Feature extraction (the slowest part) only runs once!")


def example_visualize_existing():
    """
    Example 4: Generate visualizations from existing results.
    """
    print("\n" + "="*70)
    print("Example 4: Visualize Existing Results")
    print("="*70)
    
    print("\nIf you already ran experiments with --save_features, you can")
    print("generate visualizations without re-running the experiment:")
    
    print("\n1. Generate t-SNE plots:")
    print("python analyze_results.py ./results/experiment_name --plot --viz_method tsne")
    
    print("\n2. Generate UMAP plots (faster):")
    print("python analyze_results.py ./results/experiment_name --plot --viz_method umap")
    
    print("\n3. Get detailed analysis with plots:")
    print("python analyze_results.py ./results/experiment_name --detailed --plot")
    
    print("\nNote: This requires that the original experiment was run with --save_features")


def example_umap_visualization():
    """
    Example 5: Use UMAP instead of t-SNE for faster visualization.
    """
    print("\n" + "="*70)
    print("Example 5: UMAP Visualization (Faster)")
    print("="*70)
    
    print("\nUMAP is faster than t-SNE and better for large datasets.")
    print("First, install UMAP:")
    print("  pip install umap-learn")
    
    print("\nThen run with UMAP:")
    cmd = """python main.py \\
    --dinov2_model facebook/dinov2-base \\
    --plot_clusters \\
    --viz_method umap \\
    --save_features \\
    --experiment_name umap_example"""
    print(cmd)


def main():
    """
    Main function to run examples.
    """
    parser = argparse.ArgumentParser(
        description='Examples for DINOv3 and Visualization Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example_dinov3_visualization.py --show-all
  python example_dinov3_visualization.py --example 1
  python example_dinov3_visualization.py --example reuse
        """
    )
    
    parser.add_argument('--show-all', action='store_true',
                        help='Show all examples')
    parser.add_argument('--example', type=str, default=None,
                        help='Show specific example (1-5 or name)')
    
    args = parser.parse_args()
    
    if not args.show_all and args.example is None:
        parser.print_help()
        print("\n" + "="*70)
        print("DINOv3 and Visualization Examples")
        print("="*70)
        print("\nAvailable examples:")
        print("  1. dinov3        - DINOv3 clustering with visualization")
        print("  2. compare       - Compare DINOv2 and DINOv3")
        print("  3. reuse         - Reuse features (most efficient!)")
        print("  4. visualize     - Visualize existing results")
        print("  5. umap          - Use UMAP for faster visualization")
        print("\nRun with --show-all to see all examples")
        print("Or --example <number|name> to see a specific example")
        return
    
    if args.show_all or args.example in ['1', 'dinov3']:
        example_dinov3_with_visualization()
    
    if args.show_all or args.example in ['2', 'compare']:
        example_compare_dinov2_dinov3()
    
    if args.show_all or args.example in ['3', 'reuse']:
        example_reuse_features()
    
    if args.show_all or args.example in ['4', 'visualize']:
        example_visualize_existing()
    
    if args.show_all or args.example in ['5', 'umap']:
        example_umap_visualization()
    
    print("\n" + "="*70)
    print("For more details, see:")
    print("  • README.md - General documentation")
    print("  • VISUALIZATION_GUIDE.md - Detailed visualization guide")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
