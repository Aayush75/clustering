"""
Example usage of CLIP with TEMI clustering.

This script demonstrates how to use CLIP for feature extraction
instead of DINOv2/DINOv3, both via command-line and programmatically.
"""

import torch
from pathlib import Path

from src.data_loader import create_data_loaders
from src.clip_feature_extractor import CLIPFeatureExtractor
from src.temi_clustering import TEMIClusterer
from src.evaluation import (
    evaluate_clustering,
    print_evaluation_results,
    analyze_cluster_distribution,
    print_cluster_distribution
)


def example_clip_basic():
    """
    Example: Basic usage with CLIP.
    
    This example shows how to use CLIP for feature extraction
    and clustering programmatically.
    """
    print("\n" + "="*70)
    print("Example: CLIP Feature Extraction and Clustering")
    print("="*70)
    
    # Configuration
    data_root = './data'
    batch_size = 256
    num_clusters = 100
    num_epochs = 50  # Fewer epochs for quick testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Feature Extractor: CLIP (ViT-B/32)")
    print(f"  Clusters: {num_clusters}")
    print(f"  Epochs: {num_epochs}")
    
    # Step 1: Load CIFAR100 data
    print("\nStep 1: Loading CIFAR100 dataset...")
    train_loader, test_loader = create_data_loaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=4,
        image_size=224
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Step 2: Extract features with CLIP
    print("\nStep 2: Extracting features with CLIP...")
    feature_extractor = CLIPFeatureExtractor(
        model_name='openai/clip-vit-base-patch32',
        device=device
    )
    
    train_features, train_labels = feature_extractor.extract_features(
        train_loader,
        return_labels=True
    )
    test_features, test_labels = feature_extractor.extract_features(
        test_loader,
        return_labels=True
    )
    print(f"  Training features shape: {train_features.shape}")
    print(f"  Test features shape: {test_features.shape}")
    print(f"  Feature dimension: {feature_extractor.get_feature_dim()}")
    
    # Step 3: Train TEMI clustering
    print("\nStep 3: Training TEMI clustering model...")
    clusterer = TEMIClusterer(
        feature_dim=train_features.shape[1],
        num_clusters=num_clusters,
        device=device,
        hidden_dim=2048,
        projection_dim=256,
        learning_rate=0.001,
        temperature=0.1
    )
    
    history = clusterer.fit(
        features=train_features,
        num_epochs=num_epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Step 4: Evaluate results
    print("\nStep 4: Evaluating clustering results...")
    
    # Predict on training set
    train_predictions = clusterer.predict(train_features, batch_size=batch_size)
    train_results = evaluate_clustering(train_labels.numpy(), train_predictions)
    print_evaluation_results(train_results, "Training Set")
    
    # Analyze cluster distribution
    train_distribution = analyze_cluster_distribution(train_predictions, num_clusters)
    print_cluster_distribution(train_distribution)
    
    # Predict on test set
    test_predictions = clusterer.predict(test_features, batch_size=batch_size)
    test_results = evaluate_clustering(test_labels.numpy(), test_predictions)
    print_evaluation_results(test_results, "Test Set")
    
    # Analyze cluster distribution
    test_distribution = analyze_cluster_distribution(test_predictions, num_clusters)
    print_cluster_distribution(test_distribution)
    
    print("\nExample complete!")


def example_command_line_usage():
    """
    Example: Command-line usage with CLIP.
    
    This example shows the equivalent command-line commands.
    """
    print("\n" + "="*70)
    print("Command-Line Examples for CLIP")
    print("="*70)
    
    print("\n1. Basic CLIP usage:")
    print("   python main.py --model_type clip")
    
    print("\n2. CLIP with specific model:")
    print("   python main.py --model_type clip --clip_model openai/clip-vit-large-patch14")
    
    print("\n3. CLIP with visualization:")
    print("   python main.py --model_type clip --plot_clusters --save_features")
    
    print("\n4. CLIP with custom hyperparameters:")
    print("   python main.py \\")
    print("       --model_type clip \\")
    print("       --clip_model openai/clip-vit-base-patch16 \\")
    print("       --num_clusters 100 \\")
    print("       --num_epochs 100 \\")
    print("       --batch_size 256 \\")
    print("       --learning_rate 0.001 \\")
    print("       --temperature 0.1 \\")
    print("       --plot_clusters \\")
    print("       --save_features")
    
    print("\n5. Load pre-extracted CLIP features:")
    print("   # First run: extract and save")
    print("   python main.py --model_type clip --save_features")
    print("\n   # Subsequent runs: load saved features")
    print("   python main.py --model_type clip --load_features ./results/experiment_name/features/train_features")
    

def example_comparison():
    """
    Example: Comparing different CLIP models.
    """
    print("\n" + "="*70)
    print("Comparing Different CLIP Models")
    print("="*70)
    
    print("\nAvailable CLIP models:")
    print("1. openai/clip-vit-base-patch32")
    print("   - Feature dimension: 512")
    print("   - Speed: Fastest")
    print("   - Quality: Good")
    print("   - Best for: Quick experiments")
    
    print("\n2. openai/clip-vit-base-patch16")
    print("   - Feature dimension: 512")
    print("   - Speed: Medium")
    print("   - Quality: Better")
    print("   - Best for: Balanced performance")
    
    print("\n3. openai/clip-vit-large-patch14")
    print("   - Feature dimension: 768")
    print("   - Speed: Slower")
    print("   - Quality: Best")
    print("   - Best for: Maximum accuracy")
    
    print("\nTo compare models, run:")
    print("python main.py --model_type clip --clip_model openai/clip-vit-base-patch32")
    print("python main.py --model_type clip --clip_model openai/clip-vit-base-patch16")
    print("python main.py --model_type clip --clip_model openai/clip-vit-large-patch14")


def main():
    """
    Run all CLIP examples.
    """
    print("="*70)
    print("TEMI Clustering with CLIP - Example Usage")
    print("="*70)
    
    # Show command-line examples (doesn't require data)
    example_command_line_usage()
    
    # Show model comparison info
    example_comparison()
    
    # For the basic example, uncomment to run:
    # NOTE: Requires CIFAR100 dataset and internet connection
    # example_clip_basic()
    
    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70)
    print("\nTo run the full example with CLIP:")
    print("  Uncomment 'example_clip_basic()' in the main() function")
    print("  Or run: python main.py --model_type clip")


if __name__ == "__main__":
    main()
