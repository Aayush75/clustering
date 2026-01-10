"""
Test script for CLUDI hyperparameter search functionality.

This script tests the hyperparameter search module with synthetic data
to verify correct functionality of all search methods.
"""

import os
import sys

# Ensure parent directory is in path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import numpy as np

from src.cludi_hyperparam_search import (
    CLUDIHyperparameterSpace,
    CLUDIHyperparameterSearch,
    SearchResult,
    run_cludi_hyperparam_search
)


def test_hyperparameter_space():
    """Test CLUDIHyperparameterSpace creation and serialization."""
    print("\n" + "="*60)
    print("Test 1: CLUDIHyperparameterSpace")
    print("="*60)
    
    # Create default space
    space = CLUDIHyperparameterSpace()
    
    print(f"Default embedding_dim: {space.embedding_dim}")
    print(f"Default learning_rate: {space.learning_rate}")
    print(f"Default ce_lambda: {space.ce_lambda}")
    
    # Test to_dict
    space_dict = space.to_dict()
    assert isinstance(space_dict, dict), "to_dict should return a dictionary"
    assert 'embedding_dim' in space_dict, "Should contain embedding_dim"
    
    # Test from_dict
    custom_space = CLUDIHyperparameterSpace.from_dict({
        'embedding_dim': [16, 32],
        'learning_rate': (1e-4, 1e-3)
    })
    assert custom_space.embedding_dim == [16, 32], "Should load custom embedding_dim"
    
    print("✓ CLUDIHyperparameterSpace test passed!")


def test_search_result():
    """Test SearchResult dataclass."""
    print("\n" + "="*60)
    print("Test 2: SearchResult")
    print("="*60)
    
    result = SearchResult(
        params={'embedding_dim': 64, 'learning_rate': 0.001},
        metrics={'accuracy': 0.75, 'nmi': 0.65, 'ari': 0.55},
        train_time=120.5,
        trial_id=0
    )
    
    print(f"Params: {result.params}")
    print(f"Metrics: {result.metrics}")
    print(f"Score (accuracy): {result.score}")
    print(f"Train time: {result.train_time}s")
    
    assert result.score == 0.75, "Score should be accuracy value"
    assert result.trial_id == 0, "Trial ID should be 0"
    
    print("✓ SearchResult test passed!")


def test_grid_config_generation():
    """Test grid configuration generation."""
    print("\n" + "="*60)
    print("Test 3: Grid Configuration Generation")
    print("="*60)
    
    # Create a small search space
    space = CLUDIHyperparameterSpace(
        embedding_dim=[32, 64],
        learning_rate=[0.0001, 0.001],
        diffusion_steps=[500],
        batch_diffusion=[8],
        rescaling_factor=[49.0],
        ce_lambda=[50.0],
        warmup_epochs=[1],
        dropout=[0.15]
    )
    
    # Create searcher
    searcher = CLUDIHyperparameterSearch(
        feature_dim=384,
        num_clusters=10,
        device="cpu"
    )
    
    # Generate configs
    configs = searcher._generate_grid_configs(space)
    
    print(f"Number of configurations: {len(configs)}")
    for i, cfg in enumerate(configs[:5]):  # Show first 5
        print(f"  Config {i}: {cfg}")
    
    # Should be 2 * 2 = 4 configs (embedding_dim x learning_rate)
    assert len(configs) == 4, f"Expected 4 configs, got {len(configs)}"
    
    print("✓ Grid config generation test passed!")


def test_random_config_sampling():
    """Test random configuration sampling."""
    print("\n" + "="*60)
    print("Test 4: Random Configuration Sampling")
    print("="*60)
    
    space = CLUDIHyperparameterSpace(
        embedding_dim=[32, 64, 128],
        learning_rate=(1e-5, 1e-3),  # Continuous range
        ce_lambda=[25.0, 50.0, 100.0]
    )
    
    searcher = CLUDIHyperparameterSearch(
        feature_dim=384,
        num_clusters=10,
        device="cpu"
    )
    
    # Sample multiple configs
    configs = []
    for _ in range(5):
        cfg = searcher._sample_random_config(space)
        configs.append(cfg)
        print(f"  Sampled: embedding_dim={cfg['embedding_dim']}, "
              f"lr={cfg['learning_rate']:.6f}, ce_lambda={cfg['ce_lambda']}")
    
    # Check types
    for cfg in configs:
        assert isinstance(cfg['embedding_dim'], int), "embedding_dim should be int"
        assert isinstance(cfg['learning_rate'], float), "learning_rate should be float"
        assert 1e-5 <= cfg['learning_rate'] <= 1e-3, "learning_rate should be in range"
    
    print("✓ Random config sampling test passed!")


def test_searcher_initialization():
    """Test CLUDIHyperparameterSearch initialization."""
    print("\n" + "="*60)
    print("Test 5: CLUDIHyperparameterSearch Initialization")
    print("="*60)
    
    searcher = CLUDIHyperparameterSearch(
        feature_dim=768,
        num_clusters=100,
        device="cpu",
        metric="nmi",
        results_dir="/tmp/test_search_results"
    )
    
    print(f"Feature dim: {searcher.feature_dim}")
    print(f"Num clusters: {searcher.num_clusters}")
    print(f"Metric: {searcher.metric}")
    print(f"Results dir: {searcher.results_dir}")
    
    assert searcher.feature_dim == 768, "Feature dim should be 768"
    assert searcher.num_clusters == 100, "Num clusters should be 100"
    assert searcher.metric == "nmi", "Metric should be nmi"
    
    print("✓ CLUDIHyperparameterSearch initialization test passed!")


def run_all_tests():
    """Run all hyperparameter search tests."""
    print("\n" + "="*60)
    print("CLUDI Hyperparameter Search Test Suite")
    print("="*60)
    
    try:
        test_hyperparameter_space()
    except Exception as e:
        print(f"✗ Hyperparameter space test failed: {e}")
    
    try:
        test_search_result()
    except Exception as e:
        print(f"✗ Search result test failed: {e}")
    
    try:
        test_grid_config_generation()
    except Exception as e:
        print(f"✗ Grid config generation test failed: {e}")
    
    try:
        test_random_config_sampling()
    except Exception as e:
        print(f"✗ Random config sampling test failed: {e}")
    
    try:
        test_searcher_initialization()
    except Exception as e:
        print(f"✗ Searcher initialization test failed: {e}")
    
    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
