"""
Test showing when Hungarian matching artificially inflates accuracy.
"""
import torch
from src.evaluation import cluster_accuracy

print("=" * 80)
print("DEMONSTRATION: Hungarian Algorithm Inflation")
print("=" * 80)

# Scenario: Model predicts poorly (only 5% direct accuracy)
# But if labels are permuted, Hungarian can find better alignment

num_samples = 1000
num_classes = 100

# True labels
true_labels = torch.arange(num_classes).repeat_interleave(10)  # 10 samples per class

# Case 1: Model predicts with systematic offset (shifted by 1)
predictions_shifted = (true_labels + 1) % num_classes

direct_acc = (predictions_shifted == true_labels).float().mean().item()
hungarian_acc = cluster_accuracy(true_labels, predictions_shifted)

print("\nCase 1: Systematic Shift (predictions = (true + 1) % 100)")
print(f"  Direct accuracy: {direct_acc:.4f} ({direct_acc*100:.2f}%)")
print(f"  Hungarian accuracy: {hungarian_acc:.4f} ({hungarian_acc*100:.2f}%)")
print(f"  Inflation: {hungarian_acc/max(direct_acc, 1e-8):.2f}x")

# Case 2: Random permutation of labels
perm = torch.randperm(num_classes)
predictions_perm = perm[true_labels]

direct_acc2 = (predictions_perm == true_labels).float().mean().item()
hungarian_acc2 = cluster_accuracy(true_labels, predictions_perm)

print("\nCase 2: Random Permutation")
print(f"  Direct accuracy: {direct_acc2:.4f} ({direct_acc2*100:.2f}%)")
print(f"  Hungarian accuracy: {hungarian_acc2:.4f} ({hungarian_acc2*100:.2f}%)")
print(f"  Inflation: {hungarian_acc2/max(direct_acc2, 1e-8):.2f}x")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
Hungarian matching finds the BEST POST-HOC alignment between predictions and labels.

This is appropriate when:
[OK] Evaluating UNSUPERVISED clustering (cluster IDs are arbitrary)
[OK] Cluster 0 could mean any class - Hungarian finds the optimal mapping

This is WRONG when:
[FAIL] Model is trained on specific labels (supervised learning)
[FAIL] Model already learned label=42 means "apple" - no remapping needed!
[FAIL] Your distillation pipeline (model trained on pseudo-labels)

The fix applied:
- Changed from cluster_accuracy() (Hungarian matching)
- To direct comparison: (predictions == true_labels).mean()
- This gives REAL accuracy without artificial inflation
""")

