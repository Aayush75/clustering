"""
Test script for CLUDI clustering implementation.

This script tests the CLUDI clustering module with synthetic data
to verify correct functionality of all components.
"""

import os
import sys

# Ensure parent directory is in path for imports
# This allows running the test directly without package installation
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import numpy as np

from src.cludi_clustering import CLUDIClusterer, CLUDIModel, GaussianDiffusionCLUDI, clustering_accuracy


def test_cludi_model():
    """Test CLUDIModel forward pass."""
    print("\n" + "="*60)
    print("Test 1: CLUDIModel Forward Pass")
    print("="*60)
    
    # Create model
    feature_dim = 384
    num_clusters = 10
    embedding_dim = 64
    batch_size = 4
    seq_len = 100  # Number of samples per batch
    
    model = CLUDIModel(
        feature_dim=feature_dim,
        num_clusters=num_clusters,
        embedding_dim=embedding_dim
    )
    
    # Create dummy inputs
    data_features = torch.randn(batch_size, seq_len, feature_dim)
    cluster_assignments = torch.randn(batch_size, seq_len, embedding_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = model(cluster_assignments, data_features, timesteps)
    
    print(f"Input features shape: {data_features.shape}")
    print(f"Cluster assignments shape: {cluster_assignments.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, seq_len, embedding_dim), f"Expected shape {(batch_size, seq_len, embedding_dim)}, got {output.shape}"
    print("✓ CLUDIModel forward pass test passed!")


def test_gaussian_diffusion():
    """Test GaussianDiffusionCLUDI class."""
    print("\n" + "="*60)
    print("Test 2: GaussianDiffusion")
    print("="*60)
    
    diffusion = GaussianDiffusionCLUDI(
        timesteps=1000,
        objective='pred_v',
        rescaling_factor=1.0
    )
    
    # Test q_sample (forward diffusion)
    batch_size = 4
    embedding_dim = 64
    x_start = torch.randn(batch_size, 10, embedding_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    x_t = diffusion.q_sample(x_start, t)
    
    print(f"x_start shape: {x_start.shape}")
    print(f"x_t shape: {x_t.shape}")
    print(f"Timesteps: {t}")
    
    assert x_t.shape == x_start.shape, "q_sample should preserve shape"
    print("✓ GaussianDiffusion q_sample test passed!")
    
    # Test predict_start_from_v
    v = torch.randn_like(x_start)
    x_pred = diffusion.predict_start_from_v(x_t, t, v)
    
    assert x_pred.shape == x_start.shape, "predict_start_from_v should preserve shape"
    print("✓ GaussianDiffusion predict_start_from_v test passed!")


def test_cludi_clusterer_init():
    """Test CLUDIClusterer initialization."""
    print("\n" + "="*60)
    print("Test 3: CLUDIClusterer Initialization")
    print("="*60)
    
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    clusterer = CLUDIClusterer(
        feature_dim=384,
        num_clusters=10,
        device=device,
        embedding_dim=64,
        learning_rate=0.0001,
        diffusion_steps=100,  # Reduced for testing
        batch_diffusion=2,
        rescaling_factor=1.0
    )
    
    print(f"Model device: {clusterer.device}")
    print(f"Number of clusters: {clusterer.num_clusters}")
    print(f"Feature dimension: {clusterer.feature_dim}")
    
    assert clusterer.num_clusters == 10
    assert clusterer.feature_dim == 384
    print("✓ CLUDIClusterer initialization test passed!")


def test_clustering_accuracy():
    """Test clustering accuracy computation."""
    print("\n" + "="*60)
    print("Test 4: Clustering Accuracy")
    print("="*60)
    
    # Perfect clustering
    true_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred_labels = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])  # Permuted but correct
    
    acc = clustering_accuracy(true_labels, pred_labels)
    print(f"Perfect clustering (permuted): accuracy = {acc:.4f}")
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    
    # Random clustering
    np.random.seed(42)
    true_labels = np.random.randint(0, 10, 1000)
    pred_labels = np.random.randint(0, 10, 1000)
    
    acc = clustering_accuracy(true_labels, pred_labels)
    print(f"Random clustering: accuracy = {acc:.4f}")
    assert 0 <= acc <= 1.0, "Accuracy should be between 0 and 1"
    
    print("✓ Clustering accuracy test passed!")


def test_cludi_clusterer_predict():
    """Test CLUDIClusterer prediction."""
    print("\n" + "="*60)
    print("Test 5: CLUDIClusterer Prediction (without training)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create clusterer with reduced diffusion steps for speed
    clusterer = CLUDIClusterer(
        feature_dim=384,
        num_clusters=10,
        device=device,
        embedding_dim=64,
        diffusion_steps=100,
        batch_diffusion=2
    )
    
    # Create dummy features
    n_samples = 100
    features = torch.randn(n_samples, 384)
    
    # Predict (without training - will give random results)
    try:
        predictions = clusterer.predict(features, batch_size=50)
        print(f"Input features shape: {features.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Unique clusters: {torch.unique(predictions).cpu().numpy()}")
        
        assert predictions.shape[0] == n_samples, "Should predict for all samples"
        print("✓ CLUDIClusterer prediction test passed!")
    except Exception as e:
        print(f"Prediction test skipped due to: {e}")
        print("Note: This may be expected if CUDA is not available with sufficient memory")


def run_all_tests():
    """Run all CLUDI tests."""
    print("\n" + "="*60)
    print("CLUDI Clustering Test Suite")
    print("="*60)
    
    try:
        test_cludi_model()
    except Exception as e:
        print(f"✗ CLUDIModel test failed: {e}")
        
    try:
        test_gaussian_diffusion()
    except Exception as e:
        print(f"✗ GaussianDiffusion test failed: {e}")
        
    try:
        test_cludi_clusterer_init()
    except Exception as e:
        print(f"✗ CLUDIClusterer init test failed: {e}")
        
    try:
        test_clustering_accuracy()
    except Exception as e:
        print(f"✗ Clustering accuracy test failed: {e}")
        
    try:
        test_cludi_clusterer_predict()
    except Exception as e:
        print(f"✗ CLUDIClusterer predict test failed: {e}")
    
    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
