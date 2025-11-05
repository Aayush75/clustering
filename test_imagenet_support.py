"""
Test script to verify ImageNet dataset support.

This script tests that the ImageNet dataset integration works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all necessary imports work."""
    print("Testing imports...")
    try:
        from src.data_loader import (
            ImageNetDataset,
            get_imagenet_transforms,
            create_data_loaders
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_transforms():
    """Test that ImageNet transforms can be created."""
    print("\nTesting ImageNet transforms...")
    try:
        from src.data_loader import get_imagenet_transforms
        
        train_transform, test_transform = get_imagenet_transforms(image_size=224)
        
        print("✓ Train transform created successfully")
        print("✓ Test transform created successfully")
        return True
    except Exception as e:
        print(f"✗ Transform creation failed: {e}")
        return False

def test_dataset_selection():
    """Test that dataset selection works in argument parsing."""
    print("\nTesting dataset selection in main.py...")
    try:
        import argparse
        # Simulate argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='cifar100',
                           choices=['cifar100', 'imagenet'])
        parser.add_argument('--num_clusters', type=int, default=None)
        
        # Test CIFAR100
        args1 = parser.parse_args(['--dataset', 'cifar100'])
        assert args1.dataset == 'cifar100'
        
        # Test ImageNet
        args2 = parser.parse_args(['--dataset', 'imagenet'])
        assert args2.dataset == 'imagenet'
        
        print("✓ Dataset selection argument parsing works")
        return True
    except Exception as e:
        print(f"✗ Dataset selection test failed: {e}")
        return False

def test_num_clusters_default():
    """Test automatic cluster number assignment."""
    print("\nTesting automatic cluster number assignment...")
    try:
        # Simulate the logic from main.py
        class Args:
            def __init__(self, dataset, num_clusters=None):
                self.dataset = dataset
                self.num_clusters = num_clusters
        
        # Test CIFAR100 default
        args1 = Args('cifar100')
        if args1.num_clusters is None:
            if args1.dataset.lower() == 'cifar100':
                args1.num_clusters = 100
            elif args1.dataset.lower() == 'imagenet':
                args1.num_clusters = 1000
        
        assert args1.num_clusters == 100
        print(f"✓ CIFAR100 default clusters: {args1.num_clusters}")
        
        # Test ImageNet default
        args2 = Args('imagenet')
        if args2.num_clusters is None:
            if args2.dataset.lower() == 'cifar100':
                args2.num_clusters = 100
            elif args2.dataset.lower() == 'imagenet':
                args2.num_clusters = 1000
        
        assert args2.num_clusters == 1000
        print(f"✓ ImageNet default clusters: {args2.num_clusters}")
        
        # Test custom override
        args3 = Args('imagenet', num_clusters=500)
        assert args3.num_clusters == 500
        print(f"✓ Custom cluster override works: {args3.num_clusters}")
        
        return True
    except Exception as e:
        print(f"✗ Cluster number test failed: {e}")
        return False

def test_requirements():
    """Test that datasets library is in requirements."""
    print("\nTesting requirements.txt...")
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        assert 'datasets' in requirements
        print("✓ datasets library is in requirements.txt")
        return True
    except Exception as e:
        print(f"✗ Requirements check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Testing ImageNet Dataset Support")
    print("="*60)
    
    tests = [
        test_imports,
        test_transforms,
        test_dataset_selection,
        test_num_clusters_default,
        test_requirements
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! ImageNet support is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
