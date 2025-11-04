"""
Quick test with CIFAR100 to verify the clustering works properly.
Uses a small subset for fast validation.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from src.temi_clustering import TEMIClusterer
from src.evaluation import evaluate_clustering, print_evaluation_results, analyze_cluster_distribution, print_cluster_distribution
import numpy as np

def test_cifar100_subset():
    """Test with a small subset of CIFAR100"""
    print("="*60)
    print("Quick CIFAR100 Test")
    print("="*60)
    
    # Load a subset of CIFAR100
    print("\nLoading CIFAR100...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    # Use only first 5000 samples for quick test
    num_samples = 5000
    indices = list(range(num_samples))
    subset = torch.utils.data.Subset(dataset, indices)
    
    # Create simple features by flattening images (3x32x32 = 3072 dims)
    print(f"Creating simple features from {num_samples} images...")
    features_list = []
    labels_list = []
    
    loader = torch.utils.data.DataLoader(subset, batch_size=256, shuffle=False)
    
    for images, labels in loader:
        # Flatten images to create simple features
        batch_features = images.view(images.size(0), -1)
        # Truncate to 384 dimensions (simple dimensionality reduction)
        batch_features = batch_features[:, :384]
        features_list.append(batch_features)
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of classes: {len(torch.unique(labels))}")
    
    # Test clustering with k=100
    print("\nTesting TEMI clustering with k=100...")
    clusterer = TEMIClusterer(
        feature_dim=384,
        num_clusters=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        hidden_dim=2048,
        projection_dim=256,
        learning_rate=0.001,
        temperature=0.1
    )
    
    # Train
    history = clusterer.fit(
        features=features,
        num_epochs=50,  # Fewer epochs for quick test
        batch_size=256,
        verbose=True
    )
    
    # Predict
    print("\nPredicting clusters...")
    predictions = clusterer.predict(features, batch_size=256)
    
    # Evaluate
    print("\nEvaluating results...")
    results = evaluate_clustering(
        labels.numpy(),
        predictions,
        return_confusion_matrix=False
    )
    
    print_evaluation_results(results, "CIFAR100 Subset (5000 samples)")
    
    # Analyze distribution
    distribution = analyze_cluster_distribution(predictions, 100)
    print_cluster_distribution(distribution)
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_cifar100_subset()
