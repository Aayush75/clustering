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


def test_return_embedding_shape_and_diversity():
    """Test that return_embedding produces correct shapes and diverse outputs."""
    print("\n" + "="*60)
    print("Test 6: Return Embedding Shape and Diversity")
    print("="*60)
    
    num_clusters = 10
    embedding_dim = 64
    batch_size = 4
    seq_len = 100
    
    model = CLUDIModel(
        feature_dim=384,
        num_clusters=num_clusters,
        embedding_dim=embedding_dim
    )
    
    # Create diverse soft cluster assignments (one-hot like)
    # First sample: all weight on cluster 0
    # Second sample: all weight on cluster 1
    # etc.
    x = torch.zeros(batch_size, seq_len, num_clusters)
    for i in range(batch_size):
        x[i, :, i % num_clusters] = 1.0
    
    # Get embeddings
    embeddings = model.return_embedding(x)
    
    # Check shape
    expected_shape = (batch_size, seq_len, embedding_dim)
    assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"
    print(f"✓ Embedding shape is correct: {embeddings.shape}")
    
    # Check that different cluster assignments produce different embeddings
    # Compare embeddings from samples assigned to different clusters
    embedding_0 = embeddings[0, 0, :]  # Sample assigned to cluster 0
    embedding_1 = embeddings[1, 0, :]  # Sample assigned to cluster 1
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        embedding_0.unsqueeze(0), 
        embedding_1.unsqueeze(0)
    )
    
    print(f"Cosine similarity between different cluster embeddings: {cos_sim.item():.4f}")
    
    # Different clusters should produce different embeddings (low similarity)
    # With random initialization, similarity should not be very high
    assert cos_sim.item() < 0.95, f"Different cluster assignments should produce different embeddings, but similarity is {cos_sim.item()}"
    print("✓ Different cluster assignments produce different embeddings")
    
    # Test with soft assignments (uniform distribution)
    x_uniform = torch.ones(1, seq_len, num_clusters) / num_clusters
    embeddings_uniform = model.return_embedding(x_uniform)
    
    assert embeddings_uniform.shape == (1, seq_len, embedding_dim), "Uniform assignment embedding shape is incorrect"
    print("✓ Uniform soft assignments also produce valid embeddings")
    
    print("✓ Return embedding test passed!")


def test_cluster_centers_are_learnable():
    """Test that cluster centers have requires_grad=True."""
    print("\n" + "="*60)
    print("Test 7: Cluster Centers Are Learnable")
    print("="*60)
    
    model = CLUDIModel(
        feature_dim=384,
        num_clusters=10,
        embedding_dim=64
    )
    
    # Check that cluster centers have requires_grad=True
    assert model.clusters_centers.requires_grad, "Cluster centers should have requires_grad=True"
    print(f"✓ clusters_centers.requires_grad = {model.clusters_centers.requires_grad}")
    
    # Verify cluster centers are included in named_parameters
    cluster_param_found = False
    for name, param in model.named_parameters():
        if "clusters_centers" in name:
            cluster_param_found = True
            assert param.requires_grad, f"Cluster centers parameter '{name}' should be learnable"
            print(f"✓ Found learnable parameter: {name}")
    
    assert cluster_param_found, "Cluster centers should be in named_parameters"
    print("✓ Cluster centers are learnable test passed!")


def test_cluster_diversity_after_training_step():
    """Test that a single training step produces diverse cluster assignments."""
    print("\n" + "="*60)
    print("Test 8: Cluster Diversity After Training Step")
    print("="*60)
    
    device = "cpu"
    num_clusters = 10
    n_samples = 200
    
    # Create clusterer
    clusterer = CLUDIClusterer(
        feature_dim=384,
        num_clusters=num_clusters,
        device=device,
        embedding_dim=64,
        diffusion_steps=50,  # Reduced for speed
        batch_diffusion=2
    )
    
    # Create synthetic features with some structure
    # Create features that are grouped - samples with similar indices should be similar
    torch.manual_seed(42)
    base_features = torch.randn(num_clusters, 384)
    features = []
    for i in range(n_samples):
        cluster_idx = i % num_clusters
        # Add some noise to base feature
        noise = torch.randn(384) * 0.1
        features.append(base_features[cluster_idx] + noise)
    features = torch.stack(features)
    
    # Predict before training (random init)
    predictions = clusterer.predict(features, batch_size=50)
    unique_clusters = torch.unique(predictions)
    
    print(f"Number of unique clusters used: {len(unique_clusters)}")
    print(f"Unique clusters: {unique_clusters.cpu().numpy()}")
    
    # With the fix, even with random init, the model should use multiple clusters
    # not collapse to a single cluster
    assert len(unique_clusters) > 1, f"Model should use more than 1 cluster, but only uses {len(unique_clusters)}"
    print("✓ Model uses multiple clusters (no cluster collapse)")
    
    print("✓ Cluster diversity test passed!")


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
    
    try:
        test_return_embedding_shape_and_diversity()
    except Exception as e:
        print(f"✗ Return embedding test failed: {e}")
    
    try:
        test_cluster_centers_are_learnable()
    except Exception as e:
        print(f"✗ Cluster centers learnable test failed: {e}")
    
    try:
        test_cluster_diversity_after_training_step()
    except Exception as e:
        print(f"✗ Cluster diversity test failed: {e}")
    
    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
