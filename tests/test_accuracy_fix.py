"""
Test script to demonstrate the difference between Hungarian matching and direct accuracy.
"""
import torch
import numpy as np
from src.evaluation import cluster_accuracy

# Simulate a scenario
np.random.seed(42)
torch.manual_seed(42)

# Simulate test predictions and labels
num_samples = 1000
num_classes = 100

# Case 1: Model trained on pseudo-labels (which are true labels)
# Model should predict in true label space
true_labels = torch.randint(0, num_classes, (num_samples,))

# Simulate ~70% accuracy (matching your pseudo-label accuracy)
predictions = true_labels.clone()
num_errors = int(num_samples * 0.3)
error_indices = torch.randperm(num_samples)[:num_errors]
predictions[error_indices] = torch.randint(0, num_classes, (num_errors,))

print("=" * 80)
print("ACCURACY COMPARISON TEST")
print("=" * 80)

# Direct accuracy (CORRECT for supervised models)
direct_acc = (predictions == true_labels).float().mean().item()
print(f"\nDirect Accuracy (correct method): {direct_acc:.4f} ({direct_acc*100:.2f}%)")

# Hungarian matching (WRONG for supervised models, only for clustering)
hungarian_acc = cluster_accuracy(true_labels, predictions)
print(f"Hungarian Accuracy (incorrect for this case): {hungarian_acc:.4f} ({hungarian_acc*100:.2f}%)")

print(f"\nInflation factor: {hungarian_acc / direct_acc:.2f}x")

print("\n" + "=" * 80)
print("EXPLANATION")
print("=" * 80)
print("""
WHY THIS MATTERS:

1. Your model is trained on PSEUDO-LABELS (which are actually TRUE labels)
   - Training: model(features) → true_labels
   - The model learns to predict CIFAR-100 classes directly

2. During evaluation:
   - WRONG: cluster_accuracy() finds optimal post-hoc alignment
   - This artificially inflates accuracy by finding the best permutation
   - It's "cheating" because the model already predicts true labels!

3. The fix:
   - Use direct comparison: (predictions == true_labels).mean()
   - This gives the REAL accuracy of your model

Hungarian matching is ONLY appropriate for:
- Unsupervised clustering (where cluster IDs are arbitrary)
- NOT for supervised classifiers trained on labels!
""")

