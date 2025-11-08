"""
Comprehensive test suite for dataset distillation pipeline.

This test file validates all aspects of the dataset distillation pipeline:
- Device handling (GPU consistency)
- Type consistency (PyTorch tensors)
- No indexing errors
- No typecasting errors  
- No device mismatch errors
- Complete distillation pipeline
- Evaluation with different configurations
- Integration with clustering and pseudo-labeling

Run: python test_distillation.py
"""

import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import traceback

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_distillation import DatasetDistiller, SimpleClassifier


def test_device_handling_gpu():
    """Test that all operations correctly handle GPU tensors."""
    print("\n" + "="*80)
    print("TEST: Device Handling (GPU)")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 64
    num_samples = 80
    num_classes = 8

    # Create tensors on GPU
    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=2,
        device=device,
        distill_epochs=2,
        inner_epochs=2,
        expert_epochs=3,
        seed=42
    )

    # Initialize - should handle GPU tensors
    distiller.initialize_synthesized_data(features, labels)

    # Verify all tensors are on correct device
    assert distiller.synthesized_features.device.type == device.replace('cuda', 'cuda')[:4], \
        f"Synthesized features on wrong device: {distiller.synthesized_features.device}"
    assert distiller.synthesized_labels.device.type == device.replace('cuda', 'cuda')[:4], \
        f"Synthesized labels on wrong device: {distiller.synthesized_labels.device}"

    # Verify model is on correct device
    model = distiller.create_model()
    model_device = next(model.parameters()).device.type
    assert model_device == device.replace('cuda', 'cuda')[:4], \
        f"Model on wrong device: {model_device}"

    print("✓ All tensors and models correctly placed on device:", device)
    print("✓ Device handling test PASSED")


def test_type_consistency():
    """Test that all operations use PyTorch tensors consistently."""
    print("\n" + "="*80)
    print("TEST: Type Consistency")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 32
    num_samples = 40
    num_classes = 4

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=2,
        device=device,
        distill_epochs=1,
        inner_epochs=1,
        expert_epochs=2,
        seed=42
    )

    distiller.initialize_synthesized_data(features, labels)

    # Verify types
    assert isinstance(distiller.synthesized_features, torch.Tensor), \
        f"Synthesized features not a tensor: {type(distiller.synthesized_features)}"
    assert isinstance(distiller.synthesized_labels, torch.Tensor), \
        f"Synthesized labels not a tensor: {type(distiller.synthesized_labels)}"

    # Verify model outputs are tensors
    model = distiller.create_model()
    outputs = model(distiller.synthesized_features)
    assert isinstance(outputs, torch.Tensor), \
        f"Model outputs not a tensor: {type(outputs)}"

    print("✓ All operations use PyTorch tensors")
    print("✓ Type consistency test PASSED")


def test_no_indexing_errors():
    """Test that there are no indexing errors during distillation."""
    print("\n" + "="*80)
    print("TEST: No Indexing Errors")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 48
    num_samples = 120
    num_classes = 6

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=3,
        device=device,
        distill_epochs=2,
        inner_epochs=2,
        expert_epochs=3,
        selection_strategy='margin',  # Test margin-based selection
        seed=42
    )

    # This should not raise any indexing errors
    try:
        synth_features, synth_labels = distiller.distill(features, labels, verbose=False)
        
        # Verify dimensions
        expected_size = num_classes * 3
        assert synth_features.shape[0] == expected_size, \
            f"Wrong number of synthesized features: {synth_features.shape[0]} vs {expected_size}"
        assert synth_features.shape[1] == feature_dim, \
            f"Wrong feature dimension: {synth_features.shape[1]} vs {feature_dim}"
        assert len(synth_labels) == expected_size, \
            f"Wrong number of labels: {len(synth_labels)} vs {expected_size}"
        
        print("✓ No indexing errors during distillation")
        print("✓ Indexing test PASSED")
    except IndexError as e:
        raise AssertionError(f"Indexing error occurred: {e}")


def test_no_device_mismatch():
    """Test that there are no device mismatch errors."""
    print("\n" + "="*80)
    print("TEST: No Device Mismatch Errors")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 64
    num_samples = 100
    num_classes = 5

    # Create features on GPU
    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=2,
        device=device,
        distill_epochs=2,
        inner_epochs=2,
        expert_epochs=3,
        seed=42
    )

    try:
        # This should not raise device mismatch errors
        synth_features, synth_labels = distiller.distill(features, labels, verbose=False)
        
        # Verify all results are on same device
        assert synth_features.device.type == device.replace('cuda', 'cuda')[:4], \
            f"Synth features on wrong device: {synth_features.device}"
        assert synth_labels.device.type == device.replace('cuda', 'cuda')[:4], \
            f"Synth labels on wrong device: {synth_labels.device}"
        
        print("✓ No device mismatch errors")
        print("✓ Device mismatch test PASSED")
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            raise AssertionError(f"Device mismatch error: {e}")
        raise


def test_complete_distillation_pipeline():
    """Test the complete distillation pipeline from start to finish."""
    print("\n" + "="*80)
    print("TEST: Complete Distillation Pipeline")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 64
    num_samples = 150
    num_classes = 10

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=5,
        device=device,
        distill_epochs=5,
        inner_epochs=3,
        expert_epochs=5,
        selection_strategy='random',
        partial_update_frac=1.0,
        seed=42
    )

    # Step 1: Distill
    print("  1. Running distillation...")
    synth_features, synth_labels = distiller.distill(features, labels, verbose=False)
    
    # Verify output
    assert synth_features.shape[0] == num_classes * 5
    assert synth_features.shape[1] == feature_dim
    assert synth_labels.shape[0] == num_classes * 5
    print(f"     ✓ Distilled {synth_features.shape[0]} samples")

    # Step 2: Save
    print("  2. Saving distilled data...")
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    try:
        distiller.save_distilled(temp_path)
        print(f"     ✓ Saved to {temp_path}")
        
        # Step 3: Load
        print("  3. Loading distilled data...")
        loaded_features, loaded_labels, meta = DatasetDistiller.load_distilled(temp_path, device=device)
        
        # Verify loaded data
        assert loaded_features.shape == synth_features.shape
        assert loaded_labels.shape == synth_labels.shape
        assert meta['feature_dim'] == feature_dim
        assert meta['num_classes'] == num_classes
        print("     ✓ Loaded successfully")
        
        # Step 4: Evaluate
        print("  4. Evaluating distilled data...")
        results = distiller.evaluate_distilled_data(
            real_features=features,
            pseudo_labels=labels,
            num_trials=2,
            train_epochs=10
        )
        
        assert 'distilled_train_acc' in results
        assert 'real_train_acc' in results
        assert 'compression_ratio' in results
        print(f"     ✓ Distilled train acc: {results['distilled_train_acc']:.4f}")
        print(f"     ✓ Real train acc: {results['real_train_acc']:.4f}")
        print(f"     ✓ Compression ratio: {results['compression_ratio']:.4f}")
        
        print("✓ Complete pipeline test PASSED")
    finally:
        Path(temp_path).unlink()


def test_evaluation_with_options():
    """Test evaluation with different configuration options."""
    print("\n" + "="*80)
    print("TEST: Evaluation with Options")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 32
    num_samples = 100
    num_classes = 5

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=10,
        device=device,
        distill_epochs=2,
        inner_epochs=2,
        expert_epochs=3,
        seed=42
    )

    # Distill
    synth_features, synth_labels = distiller.distill(features, labels, verbose=False)
    
    # Test 1: Evaluate with subset of images per class
    print("  1. Testing images_per_class_eval option...")
    results1 = distiller.evaluate_distilled_data(
        real_features=features,
        pseudo_labels=labels,
        num_trials=2,
        train_epochs=5,
        images_per_class_eval=5  # Use only 5 images per class
    )
    assert results1['images_per_class_eval'] == 5
    assert results1['eval_synth_size'] == num_classes * 5
    print(f"     ✓ Used {results1['eval_synth_size']} samples (5 per class)")
    
    # Test 2: Evaluate with percentage of labeled data
    print("  2. Testing labeled_data_percentage option...")
    results2 = distiller.evaluate_distilled_data(
        real_features=features,
        pseudo_labels=labels,
        num_trials=2,
        train_epochs=5,
        labeled_data_percentage=0.5  # Use only 50% of labeled data
    )
    assert results2['labeled_data_percentage'] == 0.5
    assert results2['real_data_size'] == int(num_samples * 0.5)
    print(f"     ✓ Used {results2['real_data_size']} samples (50% of real data)")
    
    # Test 3: Combine both options
    print("  3. Testing combined options...")
    results3 = distiller.evaluate_distilled_data(
        real_features=features,
        pseudo_labels=labels,
        num_trials=2,
        train_epochs=5,
        images_per_class_eval=3,
        labeled_data_percentage=0.3
    )
    assert results3['images_per_class_eval'] == 3
    assert results3['labeled_data_percentage'] == 0.3
    print(f"     ✓ Combined: {results3['eval_synth_size']} synth samples, "
          f"{results3['real_data_size']} real samples")
    
    print("✓ Evaluation options test PASSED")


def test_partial_updates():
    """Test partial update functionality."""
    print("\n" + "="*80)
    print("TEST: Partial Updates")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 32
    num_samples = 80
    num_classes = 4

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    # Test with partial updates (50%)
    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=3,
        device=device,
        distill_epochs=3,
        inner_epochs=2,
        expert_epochs=3,
        partial_update_frac=0.5,  # Update only 50% each epoch
        seed=42
    )

    try:
        synth_features, synth_labels = distiller.distill(features, labels, verbose=False)
        assert synth_features.shape[0] == num_classes * 3
        print("✓ Partial update (50%) works correctly")
        print("✓ Partial updates test PASSED")
    except Exception as e:
        raise AssertionError(f"Partial updates failed: {e}")


def test_selection_strategies():
    """Test different selection strategies."""
    print("\n" + "="*80)
    print("TEST: Selection Strategies")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 48
    num_samples = 100
    num_classes = 5

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    # Test random selection
    print("  1. Testing random selection...")
    distiller_random = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=4,
        device=device,
        distill_epochs=2,
        inner_epochs=2,
        expert_epochs=3,
        selection_strategy='random',
        seed=42
    )
    synth1, _ = distiller_random.distill(features, labels, verbose=False)
    print("     ✓ Random selection works")

    # Test margin-based selection
    print("  2. Testing margin-based selection...")
    distiller_margin = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=4,
        device=device,
        distill_epochs=2,
        inner_epochs=2,
        expert_epochs=3,
        selection_strategy='margin',
        seed=42
    )
    synth2, _ = distiller_margin.distill(features, labels, verbose=False)
    print("     ✓ Margin-based selection works")

    # Verify outputs have correct shape
    assert synth1.shape == synth2.shape
    assert synth1.shape[0] == num_classes * 4
    
    print("✓ Selection strategies test PASSED")


def test_integration_with_pseudo_labeling():
    """Test integration with pseudo-labeling from clustering."""
    print("\n" + "="*80)
    print("TEST: Integration with Pseudo-Labeling")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim = 64
    num_samples = 200
    num_clusters = 10

    # Simulate clustered data with pseudo labels
    features = torch.randn(num_samples, feature_dim, device=device)
    
    # Create synthetic cluster assignments (pseudo labels)
    cluster_assignments = torch.randint(0, num_clusters, (num_samples,), device=device)
    
    # Simulate pseudo label generation (mapping clusters to classes)
    # In real scenario, this comes from pseudo_labeling.generate_pseudo_labels
    pseudo_labels = cluster_assignments.clone()
    
    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_clusters,
        images_per_class=5,
        device=device,
        distill_epochs=3,
        inner_epochs=2,
        expert_epochs=3,
        seed=42
    )

    # Distill using pseudo labels
    synth_features, synth_labels = distiller.distill(features, pseudo_labels, verbose=False)
    
    # Verify
    assert synth_features.shape[0] == num_clusters * 5
    assert synth_labels.shape[0] == num_clusters * 5
    
    # Verify all labels are valid cluster IDs
    assert torch.all(synth_labels >= 0)
    assert torch.all(synth_labels < num_clusters)
    
    print(f"✓ Distilled {synth_features.shape[0]} samples from {num_clusters} clusters")
    print("✓ Integration with pseudo-labeling test PASSED")


def test_robustness_edge_cases():
    """Test robustness with edge cases."""
    print("\n" + "="*80)
    print("TEST: Robustness - Edge Cases")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test 1: Small dataset
    print("  1. Testing with very small dataset...")
    features = torch.randn(20, 32, device=device)
    labels = torch.randint(0, 4, (20,), device=device)
    distiller = DatasetDistiller(
        feature_dim=32, num_classes=4, images_per_class=2,
        device=device, distill_epochs=1, inner_epochs=1, expert_epochs=1, seed=42
    )
    synth, _ = distiller.distill(features, labels, verbose=False)
    assert synth.shape[0] == 8
    print("     ✓ Small dataset works")
    
    # Test 2: Imbalanced classes
    print("  2. Testing with imbalanced classes...")
    features = torch.randn(100, 48, device=device)
    # Create imbalanced labels (most samples in first 2 classes)
    labels = torch.cat([
        torch.zeros(40, device=device, dtype=torch.long),
        torch.ones(40, device=device, dtype=torch.long),
        torch.full((10,), 2, device=device, dtype=torch.long),
        torch.full((10,), 3, device=device, dtype=torch.long),
    ])
    distiller = DatasetDistiller(
        feature_dim=48, num_classes=4, images_per_class=3,
        device=device, distill_epochs=2, inner_epochs=1, expert_epochs=2, seed=42
    )
    synth, _ = distiller.distill(features, labels, verbose=False)
    assert synth.shape[0] == 12
    print("     ✓ Imbalanced classes work")
    
    # Test 3: Single sample per class
    print("  3. Testing with single sample per class...")
    distiller = DatasetDistiller(
        feature_dim=32, num_classes=5, images_per_class=1,
        device=device, distill_epochs=1, inner_epochs=1, expert_epochs=1, seed=42
    )
    features = torch.randn(50, 32, device=device)
    labels = torch.randint(0, 5, (50,), device=device)
    synth, _ = distiller.distill(features, labels, verbose=False)
    assert synth.shape[0] == 5
    print("     ✓ Single sample per class works")
    
    print("✓ Robustness test PASSED")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_device_handling_gpu,
        test_type_consistency,
        test_no_indexing_errors,
        test_no_device_mismatch,
        test_complete_distillation_pipeline,
        test_evaluation_with_options,
        test_partial_updates,
        test_selection_strategies,
        test_integration_with_pseudo_labeling,
        test_robustness_edge_cases,
    ]

    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE DATASET DISTILLATION TEST SUITE")
    print("="*80)
    print(f"Device: {'GPU (cuda)' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*80)

    passed = 0
    failed = 0
    failed_tests = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            failed_tests.append((test.__name__, e))
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("TEST SUITE SUMMARY")
    print("="*80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    
    if failed > 0:
        print("\nFailed tests:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {str(error)[:100]}")
        print("="*80)
        return False
    else:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("="*80)
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
