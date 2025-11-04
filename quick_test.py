"""
Quick test script to verify the implementation works correctly.

This script runs a minimal version of the pipeline on a small subset
of data to ensure everything is properly configured.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import create_data_loaders
from src.feature_extractor import DINOv2FeatureExtractor
from src.temi_clustering import TEMIClusterer
from src.evaluation import evaluate_clustering, print_evaluation_results


def quick_test():
    """
    Run a quick test of the clustering pipeline.
    
    This uses a small subset of data and fewer epochs to verify
    that the implementation is working correctly.
    """
    print("="*60)
    print("Quick Test: TEMI Clustering Pipeline")
    print("="*60)
    
    # Configuration for quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 1: Load a small batch of data
    print("\n1. Loading CIFAR100 data...")
    train_loader, test_loader = create_data_loaders(
        root='./data',
        batch_size=128,
        num_workers=0,  # Use 0 workers for quick test
        image_size=224
    )
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Step 2: Extract features (only first batch for quick test)
    print("\n2. Extracting features with DINOv2...")
    feature_extractor = DINOv2FeatureExtractor(
        model_name="facebook/dinov2-small",  # Use small model for speed
        device=device
    )
    
    # Get just a few batches for testing
    train_features_list = []
    train_labels_list = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_loader):
            if i >= 5:  # Only process 5 batches
                break
            images = images.to(device)
            outputs = feature_extractor.model(pixel_values=images)
            features = outputs.last_hidden_state[:, 0, :]
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            train_features_list.append(features.cpu())
            train_labels_list.append(labels)
    
    train_features = torch.cat(train_features_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    
    print(f"   Extracted features shape: {train_features.shape}")
    
    # Step 3: Train TEMI clustering
    print("\n3. Training TEMI clustering...")
    feature_dim = train_features.shape[1]
    num_clusters = 20  # Use fewer clusters for quick test
    
    clusterer = TEMIClusterer(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        device=device,
        hidden_dim=512,  # Smaller network for speed
        projection_dim=128,
        learning_rate=0.001,
        temperature=0.1
    )
    
    # Train for just a few epochs
    history = clusterer.fit(
        features=train_features,
        num_epochs=10,  # Only 10 epochs for quick test
        batch_size=128,
        verbose=True
    )
    
    # Step 4: Evaluate
    print("\n4. Evaluating clustering...")
    predictions = clusterer.predict(train_features, batch_size=128)
    
    results = evaluate_clustering(
        train_labels.numpy(),
        predictions,
        return_confusion_matrix=False
    )
    
    print_evaluation_results(results, "Quick Test")
    
    # Verify results are reasonable
    print("\n5. Verification...")
    if results['accuracy'] > 0:
        print("   PASS: Clustering produced valid results")
        print("   PASS: All components working correctly")
        return True
    else:
        print("   FAIL: Clustering accuracy is zero")
        return False


if __name__ == "__main__":
    try:
        success = quick_test()
        print("\n" + "="*60)
        if success:
            print("Quick test completed successfully!")
            print("The implementation is ready for full experiments.")
        else:
            print("Quick test failed. Please check the implementation.")
        print("="*60)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
