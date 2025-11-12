"""
Test suite for evaluation functions with fixed mapping.

This test validates the corrected evaluation methodology that uses
fixed cluster-to-label mappings instead of post-hoc Hungarian matching.
"""

import torch
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import cluster_accuracy, fixed_mapping_accuracy


def test_fixed_mapping_accuracy():
    """Test the fixed mapping accuracy function."""
    print("\n" + "="*80)
    print("TEST: Fixed Mapping Accuracy")
    print("="*80)
    
    # Example: clustering established this mapping
    # cluster_0 -> label 1 (automobile)
    # cluster_1 -> label 2 (bird)
    # cluster_2 -> label 3 (cat)
    cluster_to_label = {0: 1, 1: 2, 2: 3}
    
    # Test 1: Perfect predictions
    print("\n  Test 1: Perfect predictions")
    y_true = torch.tensor([1, 2, 3, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 1])  # Cluster IDs
    
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    print(f"    ✓ Accuracy: {acc:.4f} (100% correct)")
    
    # Test 2: Partial correctness
    print("\n  Test 2: Partial correctness")
    y_true = torch.tensor([1, 2, 3, 1, 2])
    y_pred = torch.tensor([0, 1, 0, 0, 1])  # 4 out of 5 correct
    
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    expected = 4.0 / 5.0
    assert abs(acc - expected) < 1e-6, f"Expected {expected}, got {acc}"
    print(f"    ✓ Accuracy: {acc:.4f} (80% correct)")
    
    # Test 3: All wrong
    print("\n  Test 3: All wrong")
    y_true = torch.tensor([1, 1, 1, 1, 1])
    y_pred = torch.tensor([1, 1, 1, 1, 1])  # All predict cluster_1 -> label 2
    
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    assert acc == 0.0, f"Expected 0.0, got {acc}"
    print(f"    ✓ Accuracy: {acc:.4f} (0% correct)")
    
    # Test 4: Unmapped clusters (should be filtered out)
    print("\n  Test 4: Unmapped clusters")
    y_true = torch.tensor([1, 2, 3, 1, 2])
    y_pred = torch.tensor([0, 5, 2, 0, 1])  # cluster_5 is not in mapping
    
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    # Only 4 samples have valid predictions: [1, 3, 1, 2] vs [1, 3, 1, 2] = 100%
    assert acc == 1.0, f"Expected 1.0 (ignoring unmapped), got {acc}"
    print(f"    ✓ Accuracy: {acc:.4f} (unmapped clusters filtered)")
    
    print("\n✓ Fixed mapping accuracy test PASSED")


def test_fixed_vs_hungarian_mapping():
    """
    Demonstrate the difference between fixed mapping and Hungarian matching.
    
    This test shows how Hungarian matching can artificially inflate performance
    by retroactively finding the optimal alignment.
    """
    print("\n" + "="*80)
    print("TEST: Fixed Mapping vs Hungarian Matching")
    print("="*80)
    
    # Scenario: Model learned cluster_0->1, cluster_1->2, cluster_2->3
    # But predictions are systematically wrong
    cluster_to_label = {0: 1, 1: 2, 2: 3}
    
    # Ground truth labels
    y_true = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    
    # Model predictions (cluster IDs)
    # Model consistently predicts wrong cluster for each label
    # When label=1, it predicts cluster_2 (should predict cluster_0)
    # When label=2, it predicts cluster_0 (should predict cluster_1)
    # When label=3, it predicts cluster_1 (should predict cluster_2)
    y_pred = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    
    # Test with FIXED mapping (CORRECT for evaluation)
    print("\n  With FIXED mapping (correct approach):")
    fixed_acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    # cluster_0->1, cluster_1->2, cluster_2->3
    # Predictions: [2,0,1,2,0,1,2,0,1,2] -> [3,1,2,3,1,2,3,1,2,3]
    # True labels: [1,2,3,1,2,3,1,2,3,1]
    # Matches: [F,F,F,F,F,F,F,F,F,F] = 0/10
    print(f"    Accuracy: {fixed_acc:.4f} (0% - model failed to learn mapping)")
    assert fixed_acc == 0.0, f"Expected 0.0, got {fixed_acc}"
    
    # Test with HUNGARIAN matching (WRONG for evaluation, but common mistake)
    print("\n  With HUNGARIAN matching (incorrect approach):")
    hungarian_acc = cluster_accuracy(y_true, y_pred)
    # Hungarian will find: cluster_0->2, cluster_1->3, cluster_2->1
    # This retroactively makes the predictions "correct"!
    print(f"    Accuracy: {hungarian_acc:.4f} (artificially high!)")
    assert hungarian_acc > 0.5, "Hungarian matching should give high accuracy"
    
    print(f"\n  Difference: {hungarian_acc - fixed_acc:.4f}")
    print("  ⚠️  Hungarian matching gives artificially high results!")
    print("  ✓  Fixed mapping shows the TRUE performance")
    
    print("\n✓ Fixed vs Hungarian comparison PASSED")


def test_cuda_compatibility():
    """Test that fixed_mapping_accuracy works with CUDA tensors."""
    print("\n" + "="*80)
    print("TEST: CUDA Compatibility")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, skipping test")
        return
    
    device = torch.device('cuda')
    cluster_to_label = {0: 1, 1: 2, 2: 3}
    
    y_true = torch.tensor([1, 2, 3, 1, 2], device=device)
    y_pred = torch.tensor([0, 1, 2, 0, 1], device=device)
    
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    print(f"  ✓ CUDA tensors work correctly: {acc:.4f}")
    
    print("\n✓ CUDA compatibility test PASSED")


def test_mixed_types():
    """Test that fixed_mapping_accuracy handles mixed input types."""
    print("\n" + "="*80)
    print("TEST: Mixed Input Types")
    print("="*80)
    
    cluster_to_label = {0: 1, 1: 2, 2: 3}
    
    # Test with lists
    print("  Test 1: Lists")
    y_true = [1, 2, 3, 1, 2]
    y_pred = [0, 1, 2, 0, 1]
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    print(f"    ✓ Lists: {acc:.4f}")
    
    # Test with numpy arrays (converted to lists)
    print("  Test 2: NumPy arrays")
    import numpy as np
    y_true = np.array([1, 2, 3, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1])
    acc = fixed_mapping_accuracy(y_true.tolist(), y_pred.tolist(), cluster_to_label)
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    print(f"    ✓ NumPy arrays: {acc:.4f}")
    
    # Test with PyTorch tensors
    print("  Test 3: PyTorch tensors")
    y_true = torch.tensor([1, 2, 3, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 1])
    acc = fixed_mapping_accuracy(y_true, y_pred, cluster_to_label)
    assert acc == 1.0, f"Expected 1.0, got {acc}"
    print(f"    ✓ PyTorch tensors: {acc:.4f}")
    
    print("\n✓ Mixed types test PASSED")


def run_all_tests():
    """Run all evaluation tests."""
    print("\n" + "="*80)
    print("EVALUATION TEST SUITE")
    print("Testing Fixed Mapping Evaluation Functions")
    print("="*80)
    
    try:
        test_fixed_mapping_accuracy()
        test_fixed_vs_hungarian_mapping()
        test_cuda_compatibility()
        test_mixed_types()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
