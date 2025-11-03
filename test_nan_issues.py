"""
Diagnostic script to check for potential NaN issues.

Run this before training to verify the setup is working correctly.
"""

import torch
import numpy as np
from config import Config
from models.clustering_model import TeacherStudentModel
from models.loss import MultiHeadTEMILoss


def test_model_forward():
    """Test that model forward pass doesn't produce NaN."""
    print("Testing model forward pass...")
    
    config = Config()
    model = TeacherStudentModel(config)
    
    # Create dummy input
    batch_size = 4
    dummy_embeddings = torch.randn(batch_size, config.EMBEDDING_DIM)
    
    # Forward pass
    student_outputs, teacher_outputs = model(dummy_embeddings)
    
    # Check for NaN
    for i, (s_out, t_out) in enumerate(zip(student_outputs, teacher_outputs)):
        if torch.isnan(s_out).any():
            print(f"  ERROR: Student head {i} produced NaN")
            return False
        if torch.isnan(t_out).any():
            print(f"  ERROR: Teacher head {i} produced NaN")
            return False
    
    print("  OK: No NaN in model outputs")
    return True


def test_loss_computation():
    """Test that loss computation doesn't produce NaN."""
    print("Testing loss computation...")
    
    config = Config()
    model = TeacherStudentModel(config)
    loss_fn = MultiHeadTEMILoss(config)
    
    # Create dummy input
    batch_size = 4
    dummy_embeddings = torch.randn(batch_size, config.EMBEDDING_DIM)
    
    # Forward pass
    student_outputs, teacher_outputs = model(dummy_embeddings)
    
    # Compute loss
    loss = loss_fn(student_outputs, teacher_outputs, epoch=0)
    
    # Check loss
    if torch.isnan(loss):
        print(f"  ERROR: Loss is NaN: {loss.item()}")
        return False
    if torch.isinf(loss):
        print(f"  ERROR: Loss is infinite: {loss.item()}")
        return False
    
    print(f"  OK: Loss is finite: {loss.item():.4f}")
    return True


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("Testing gradient flow...")
    
    config = Config()
    model = TeacherStudentModel(config)
    loss_fn = MultiHeadTEMILoss(config)
    
    # Create dummy input
    batch_size = 4
    dummy_embeddings = torch.randn(batch_size, config.EMBEDDING_DIM)
    
    # Forward pass
    student_outputs, teacher_outputs = model(dummy_embeddings)
    
    # Compute loss
    loss = loss_fn(student_outputs, teacher_outputs, epoch=0)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    has_nan_grad = False
    
    for name, param in model.student.named_parameters():
        if param.grad is not None:
            has_grad = True
            if torch.isnan(param.grad).any():
                print(f"  ERROR: NaN gradient in {name}")
                has_nan_grad = True
    
    if not has_grad:
        print("  ERROR: No gradients computed")
        return False
    
    if has_nan_grad:
        print("  ERROR: NaN gradients detected")
        return False
    
    print("  OK: Gradients are finite")
    return True


def test_numerical_stability():
    """Test numerical stability of loss components."""
    print("Testing numerical stability...")
    
    from models.loss import beta_mi, sim_weight
    
    # Test with small probabilities
    batch_size = 4
    num_clusters = 100
    
    # Create probability distributions
    p1 = torch.softmax(torch.randn(batch_size, num_clusters) * 0.1, dim=-1)
    p2 = torch.softmax(torch.randn(batch_size, num_clusters) * 0.1, dim=-1)
    pk = torch.ones(1, num_clusters) / num_clusters
    
    # Test similarity weight
    weight = sim_weight(p1, p2)
    if torch.isnan(weight).any():
        print("  ERROR: sim_weight produced NaN")
        return False
    
    # Test beta_mi
    mi = beta_mi(p1, p2, pk, beta=0.6)
    if torch.isnan(mi).any():
        print("  ERROR: beta_mi produced NaN")
        return False
    if torch.isinf(mi).any():
        print("  ERROR: beta_mi produced inf")
        return False
    
    print("  OK: Loss components are numerically stable")
    return True


def test_extreme_cases():
    """Test extreme cases that might cause NaN."""
    print("Testing extreme cases...")
    
    from models.loss import beta_mi
    
    batch_size = 4
    num_clusters = 100
    
    # Test 1: Very confident predictions (might cause log(0))
    p1 = torch.zeros(batch_size, num_clusters)
    p1[:, 0] = 1.0  # All probability on first cluster
    p2 = torch.softmax(torch.randn(batch_size, num_clusters), dim=-1)
    pk = torch.ones(1, num_clusters) / num_clusters
    
    mi = beta_mi(p1, p2, pk, beta=0.6)
    if torch.isnan(mi).any():
        print("  WARNING: Confident predictions cause NaN (should be handled)")
    else:
        print("  OK: Handles confident predictions")
    
    # Test 2: Uniform distributions
    p1 = torch.ones(batch_size, num_clusters) / num_clusters
    p2 = torch.ones(batch_size, num_clusters) / num_clusters
    pk = torch.ones(1, num_clusters) / num_clusters
    
    mi = beta_mi(p1, p2, pk, beta=0.6)
    if torch.isnan(mi).any() or torch.isinf(mi).any():
        print("  ERROR: Uniform distributions cause NaN/inf")
        return False
    else:
        print("  OK: Handles uniform distributions")
    
    return True


def main():
    """Run all diagnostic tests."""
    print("="*80)
    print("NaN Diagnostic Tests")
    print("="*80)
    print()
    
    tests = [
        ("Model Forward Pass", test_model_forward),
        ("Loss Computation", test_loss_computation),
        ("Gradient Flow", test_gradient_flow),
        ("Numerical Stability", test_numerical_stability),
        ("Extreme Cases", test_extreme_cases),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
        print()
    
    # Summary
    print("="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("All tests passed! The model should be stable during training.")
    else:
        print("Some tests failed. There may be numerical instability issues.")
        print("Review the error messages above for details.")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()
