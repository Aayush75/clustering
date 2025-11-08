"""
Updated test script for dataset distillation (aligned to the "working" MTT-style implementation
provided in `src.dataset_distillation`).

This file is intentionally lightweight and fast so it can run on CI/locally in < 30s for CPU
by default. It tests:
 - device handling
 - type consistency
 - basic vectorization
 - the full distillation pipeline (short runs)
 - pseudo-labeling integration (using a small simulated generator)

Run: python test_distillation.py
"""

import torch
from pathlib import Path
import sys
import tempfile
import traceback

# ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset_distillation import DatasetDistiller, SimpleClassifier


def test_device_handling():
    device = "cpu"
    feature_dim = 64
    num_samples = 80
    num_classes = 8

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=2,
        device=device,
        distill_epochs=1,
        inner_epochs=1,
        expert_epochs=2
    )

    distiller.initialize_synthesized_data(features, labels)

    assert distiller.synthesized_features.device.type == device
    assert distiller.synthesized_labels.device.type == device

    model = distiller.create_model()
    assert next(model.parameters()).device.type == device

    # quick expert run (short)
    expert_params = distiller.get_expert_final_params(features, labels)
    assert isinstance(expert_params, list)

    print("test_device_handling: OK")


def test_type_consistency():
    device = "cpu"
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
        expert_epochs=2
    )

    distiller.initialize_synthesized_data(features, labels)

    assert isinstance(distiller.synthesized_features, torch.Tensor)
    assert isinstance(distiller.synthesized_labels, torch.Tensor)

    model = distiller.create_model()
    outputs = model(distiller.synthesized_features)
    assert isinstance(outputs, torch.Tensor)

    print("test_type_consistency: OK")


def test_vectorization_and_unroll():
    device = "cpu"
    feature_dim = 48
    num_samples = 120
    num_classes = 6

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=2,
        device=device,
        distill_epochs=1,
        inner_epochs=3,
        expert_epochs=3,
        batch_size=32
    )

    distiller.initialize_synthesized_data(features, labels)

    # create model and run a small unroll
    student_traj = distiller.train_student_with_unroll(
        distiller.synthesized_features, distiller.synthesized_labels
    )

    # trajectory length should equal num_unroll_steps
    assert len(student_traj) == distiller.num_unroll_steps
    # each snapshot should be a list of tensors (parameters)
    assert isinstance(student_traj[0], list) and isinstance(student_traj[0][0], torch.Tensor)

    # compute distance between expert (quick) and student
    expert = distiller.get_expert_final_params(features, labels)
    dist = distiller.compute_distance(expert[:len(student_traj)], student_traj)
    assert isinstance(dist, torch.Tensor)
    assert dist.dim() == 0 or dist.numel() == 1

    print("test_vectorization_and_unroll: OK")


def test_full_pipeline_and_save_load():
    device = "cpu"
    feature_dim = 64
    num_samples = 100
    num_classes = 5

    features = torch.randn(num_samples, feature_dim, device=device)
    labels = torch.randint(0, num_classes, (num_samples,), device=device)

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_classes,
        images_per_class=3,
        device=device,
        distill_epochs=2,      # short for test
        inner_epochs=2,
        expert_epochs=3,
        batch_size=32
    )

    synth_feats, synth_labels = distiller.distill(features, labels, verbose=False)

    assert isinstance(synth_feats, torch.Tensor)
    assert isinstance(synth_labels, torch.Tensor)
    assert synth_feats.shape[0] == num_classes * 3
    assert synth_feats.shape[1] == feature_dim

    # save / load
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name

    try:
        distiller.save_distilled_data(temp_path)
        loaded_feats, loaded_labels, meta = DatasetDistiller.load_distilled_data(temp_path, device=device)
        # loaded tensors are CPU tensors per implementation contract
        assert loaded_feats.shape == synth_feats.detach().cpu().shape
        assert loaded_labels.shape == synth_labels.detach().cpu().shape
        assert meta['feature_dim'] == feature_dim
        print("test_full_pipeline_and_save_load: OK")
    finally:
        Path(temp_path).unlink()


def test_pseudo_labeling_integration_simulated():
    # create a small simulated clustering -> pseudo labels
    device = "cpu"
    feature_dim = 32
    num_samples = 80
    num_clusters = 8

    features = torch.randn(num_samples, feature_dim, device=device)
    centers = torch.randn(num_clusters, feature_dim, device=device)
    fn = torch.nn.functional.normalize
    similarities = torch.mm(fn(features, dim=1), fn(centers, dim=1).t())
    cluster_assignments = torch.argmax(similarities, dim=1)

    # create pseudo labels by majority vote within each cluster (simple)
    pseudo = torch.zeros_like(cluster_assignments)
    for c in range(num_clusters):
        mask = cluster_assignments == c
        if mask.sum() == 0:
            continue
        # assign the cluster id as pseudo label (identity mapping)
        pseudo[mask] = c

    distiller = DatasetDistiller(
        feature_dim=feature_dim,
        num_classes=num_clusters,
        images_per_class=1,
        device=device,
        distill_epochs=1,
        inner_epochs=1,
        expert_epochs=2
    )

    synth_feats, synth_labels = distiller.distill(features, pseudo, verbose=False)
    assert synth_feats.shape[0] == num_clusters * 1

    print("test_pseudo_labeling_integration_simulated: OK")


def run_all_tests():
    tests = [
        test_device_handling,
        test_type_consistency,
        test_vectorization_and_unroll,
        test_full_pipeline_and_save_load,
        test_pseudo_labeling_integration_simulated
    ]

    all_ok = True
    for t in tests:
        try:
            t()
        except Exception as e:
            all_ok = False
            print("FAILED test:", t.__name__)
            traceback.print_exc()

    if all_ok:
        print("\\nALL TESTS PASSED")
        return True
    else:
        print("\\nSOME TESTS FAILED")
        return False


if __name__ == '__main__':
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
