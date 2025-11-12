"""
Example script demonstrating dataset distillation with pseudo labels.

This script shows how to use the complete pipeline:
1. Extract features using DINOv2 or CLIP
2. Perform TEMI clustering
3. Generate pseudo labels
4. Distill dataset using pseudo labels
5. Evaluate distilled dataset

The distilled dataset can be used for efficient training.
"""

import torch
import argparse
from pathlib import Path

from src.data_loader import create_data_loaders
from src.feature_extractor import DINOv2FeatureExtractor
from src.clip_feature_extractor import CLIPFeatureExtractor
from src.temi_clustering import TEMIClusterer
from src.pseudo_labeling import generate_pseudo_labels
from src.dataset_distillation import DatasetDistiller


def main():
    """Run the complete dataset distillation pipeline."""
    
    # Configuration
    parser = argparse.ArgumentParser(description='Dataset Distillation Example')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'])
    parser.add_argument('--model_type', type=str, default='dinov2', choices=['dinov2', 'clip'])
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to use (for testing)')
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--images_per_class', type=int, default=5)
    parser.add_argument('--distill_epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("\n" + "="*80)
    print("DATASET DISTILLATION EXAMPLE")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_type}")
    print(f"Num clusters: {args.num_clusters}")
    print(f"Images per class: {args.images_per_class}")
    print(f"Distillation epochs: {args.distill_epochs}")
    
    # Step 1: Load data and extract features
    print("\n" + "="*80)
    print("Step 1: Feature Extraction")
    print("="*80)
    
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader = create_data_loaders(
        root='./data',
        batch_size=256,
        num_workers=2,
        dataset_name=args.dataset
    )
    
    # Extract features from first batch only (for quick testing)
    print("\nExtracting features from training data...")
    if args.model_type == 'clip':
        feature_extractor = CLIPFeatureExtractor(device=device)
    else:
        feature_extractor = DINOv2FeatureExtractor(device=device)
    
    # Get limited samples for quick testing
    features_list = []
    labels_list = []
    
    for i, (images, labels) in enumerate(train_loader):
        if len(features_list) * train_loader.batch_size >= args.num_samples:
            break
        
        images = images.to(device)
        
        # Extract features
        if args.model_type == 'clip':
            # For CLIP, we need special preprocessing
            with torch.no_grad():
                outputs = feature_extractor.model.vision_model(pixel_values=images)
                batch_features = feature_extractor.model.visual_projection(outputs.pooler_output)
                batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
        else:
            # For DINOv2
            with torch.no_grad():
                outputs = feature_extractor.model(pixel_values=images)
                batch_features = outputs.last_hidden_state[:, 0, :]
                batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
        
        features_list.append(batch_features.cpu())
        labels_list.append(labels)
    
    train_features = torch.cat(features_list, dim=0)[:args.num_samples]
    train_labels = torch.cat(labels_list, dim=0)[:args.num_samples]
    
    print(f"Features extracted: {train_features.shape}")
    feature_dim = train_features.shape[1]
    
    # Step 2: Perform TEMI clustering
    print("\n" + "="*80)
    print("Step 2: TEMI Clustering")
    print("="*80)
    
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=args.num_clusters,
        device=device,
        hidden_dim=512,
        projection_dim=128,
        learning_rate=0.001,
        temperature=0.1
    )
    
    # Initialize clusters
    clusterer.initialize_clusters(train_features)
    
    # Train clustering model
    print("\nTraining TEMI clustering model...")
    clusterer.train(train_features, num_epochs=20, batch_size=128)
    
    # Get cluster assignments
    predictions = clusterer.predict(train_features)
    print(f"Cluster assignments computed: {predictions.shape}")
    
    # Step 3: Generate pseudo labels
    print("\n" + "="*80)
    print("Step 3: Pseudo Label Generation")
    print("="*80)
    
    pseudo_labels, cluster_to_label, k_nearest, confidence, cluster_confidence = generate_pseudo_labels(
        features=train_features,
        cluster_assignments=predictions,
        true_labels=train_labels,
        cluster_centers=clusterer.cluster_centers,
        k=10,
        verbose=True,
        return_confidence=True
    )
    
    print(f"Pseudo labels generated: {pseudo_labels.shape}")
    print(f"Average confidence: {torch.mean(confidence[pseudo_labels != -1]).item():.4f}")
    
    # Step 4: Dataset distillation
    print("\n" + "="*80)
    print("Step 4: Dataset Distillation")
    print("="*80)
    
    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=args.num_clusters,
        images_per_class=args.images_per_class,
        device=device,
        learning_rate=0.001,
        distill_lr=0.1,
        distill_epochs=args.distill_epochs,
        inner_epochs=5,
        batch_size=128
    )
    
    synthesized_features, synthesized_labels = distiller.distill(
        real_features=train_features,
        pseudo_labels=pseudo_labels,
        verbose=True
    )
    
    # Set the cluster-to-label mapping for evaluation
    distiller.set_cluster_mapping(cluster_to_label)
    
    print(f"\nDistilled dataset created:")
    print(f"  Original size: {len(train_features)} samples")
    print(f"  Distilled size: {len(synthesized_features)} samples")
    print(f"  Compression ratio: {len(synthesized_features)/len(train_features):.2%}")
    
    # Step 5: Evaluate distilled dataset
    print("\n" + "="*80)
    print("Step 5: Evaluation")
    print("="*80)
    
    # Create a test set for proper evaluation (normally you'd have this from data loading)
    print("  Creating test set for evaluation...")
    test_features = torch.randn(200, feature_dim, device=device)
    test_labels = torch.randint(0, args.num_clusters, (200,), device=device)
    
    results = distiller.evaluate_distilled_data(
        real_features=train_features,
        pseudo_labels=pseudo_labels,
        test_features=test_features,
        test_labels=test_labels,
        cluster_to_label=cluster_to_label,  # Pass the mapping from pseudo-labeling
        num_trials=3,
        include_supervised_baseline=False  # Set to True if you have ground truth train labels
    )
    
    print(f"\n  Distilled test accuracy: {results['distilled_test_acc']:.4f} ± {results['distilled_test_std']:.4f}")
    print(f"  Real pseudo test accuracy: {results['real_pseudo_test_acc']:.4f} ± {results['real_pseudo_test_std']:.4f}")
    print(f"  Performance ratio: {results['performance_ratio']:.4f}")
    
    # Save distilled dataset
    output_dir = Path('./results/distillation_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    distilled_path = output_dir / 'distilled_features.pt'
    distiller.save_distilled(str(distilled_path))
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Distilled dataset saved to: {distilled_path}")
    print("\nKey Results:")
    print(f"  - Original dataset: {len(train_features)} samples")
    print(f"  - Distilled dataset: {len(synthesized_features)} samples")
    print(f"  - Compression: {len(synthesized_features)/len(train_features):.2%}")
    print(f"  - Test accuracy (distilled): {results['distilled_test_acc']:.4f}")
    print(f"  - Test accuracy (real): {results['real_pseudo_test_acc']:.4f}")
    print(f"  - Performance ratio: {results['performance_ratio']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
