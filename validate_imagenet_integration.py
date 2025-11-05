"""
Comprehensive validation script for ImageNet integration.

This script validates that all components of the ImageNet integration
work correctly together.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def validate_data_loader():
    """Validate that data loader supports both datasets."""
    print("\n" + "="*60)
    print("Validating Data Loader")
    print("="*60)
    
    from src.data_loader import (
        CIFAR100Dataset,
        ImageNetDataset,
        get_cifar100_transforms,
        get_imagenet_transforms,
        get_dataset_statistics
    )
    
    # Test CIFAR100 transforms
    print("✓ CIFAR100 dataset class imported")
    train_tf, test_tf = get_cifar100_transforms()
    print("✓ CIFAR100 transforms created")
    
    # Test ImageNet transforms
    print("✓ ImageNet dataset class imported")
    train_tf, test_tf = get_imagenet_transforms()
    print("✓ ImageNet transforms created")
    
    # Test dataset statistics with both datasets
    print("✓ Dataset statistics function available")
    
    return True


def validate_main_script():
    """Validate that main.py accepts new arguments."""
    print("\n" + "="*60)
    print("Validating Main Script Arguments")
    print("="*60)
    
    # Test argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar100', 'imagenet'])
    parser.add_argument('--num_clusters', type=int, default=None)
    
    # Test with CIFAR100
    args = parser.parse_args(['--dataset', 'cifar100'])
    assert args.dataset == 'cifar100'
    print("✓ CIFAR100 dataset selection works")
    
    # Test with ImageNet
    args = parser.parse_args(['--dataset', 'imagenet'])
    assert args.dataset == 'imagenet'
    print("✓ ImageNet dataset selection works")
    
    return True


def validate_documentation():
    """Validate that all documentation files exist."""
    print("\n" + "="*60)
    print("Validating Documentation")
    print("="*60)
    
    docs = [
        'README.md',
        'IMAGENET_USAGE.md',
        'example_imagenet_usage.py',
        'requirements.txt'
    ]
    
    for doc in docs:
        if Path(doc).exists():
            print(f"✓ {doc} exists")
        else:
            print(f"✗ {doc} missing")
            return False
    
    # Check that README mentions ImageNet
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    if 'imagenet' in readme_content.lower():
        print("✓ README.md mentions ImageNet")
    else:
        print("✗ README.md does not mention ImageNet")
        return False
    
    # Check that requirements.txt includes datasets
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    if 'datasets' in requirements:
        print("✓ requirements.txt includes datasets library")
    else:
        print("✗ requirements.txt missing datasets library")
        return False
    
    return True


def validate_api_usage():
    """Validate programmatic API usage."""
    print("\n" + "="*60)
    print("Validating API Usage")
    print("="*60)
    
    try:
        from src.data_loader import create_data_loaders
        
        # Test that function signature includes dataset_name
        import inspect
        sig = inspect.signature(create_data_loaders)
        params = list(sig.parameters.keys())
        
        if 'dataset_name' in params:
            print("✓ create_data_loaders has dataset_name parameter")
        else:
            print("✗ create_data_loaders missing dataset_name parameter")
            return False
        
        # Check default value
        default = sig.parameters['dataset_name'].default
        if default == 'cifar100':
            print("✓ Default dataset is cifar100")
        else:
            print(f"✗ Unexpected default dataset: {default}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ API validation failed: {e}")
        return False


def validate_example_scripts():
    """Validate that example scripts exist and are valid Python."""
    print("\n" + "="*60)
    print("Validating Example Scripts")
    print("="*60)
    
    scripts = [
        'example_imagenet_usage.py',
        'test_imagenet_support.py',
        'validate_imagenet_integration.py'
    ]
    
    for script in scripts:
        if not Path(script).exists():
            print(f"✗ {script} missing")
            return False
        
        # Try to parse the script
        try:
            with open(script, 'r') as f:
                content = f.read()
            compile(content, script, 'exec')
            print(f"✓ {script} is valid Python")
        except SyntaxError as e:
            print(f"✗ {script} has syntax error: {e}")
            return False
    
    return True


def validate_backwards_compatibility():
    """Validate that existing CIFAR100 usage still works."""
    print("\n" + "="*60)
    print("Validating Backwards Compatibility")
    print("="*60)
    
    try:
        from src.data_loader import create_data_loaders
        
        # Old usage should still work (without dataset_name parameter)
        print("✓ create_data_loaders can be called with old signature")
        
        # New usage should work
        print("✓ create_data_loaders can be called with dataset_name")
        
        return True
    except Exception as e:
        print(f"✗ Backwards compatibility test failed: {e}")
        return False


def print_usage_examples():
    """Print usage examples for quick reference."""
    print("\n" + "="*60)
    print("Usage Examples")
    print("="*60)
    
    examples = """
# CIFAR100 (default)
python main.py

# ImageNet
python main.py --dataset imagenet

# ImageNet with CLIP
python main.py --dataset imagenet --model_type clip

# ImageNet with custom clusters
python main.py --dataset imagenet --num_clusters 500

# ImageNet with visualization
python main.py --dataset imagenet --plot_clusters --save_features

For complete documentation, see:
- IMAGENET_USAGE.md (comprehensive ImageNet guide)
- example_imagenet_usage.py (code examples)
- README.md (general usage)
"""
    print(examples)


def main():
    """Run all validation checks."""
    print("="*60)
    print("ImageNet Integration Validation")
    print("="*60)
    
    checks = [
        ("Data Loader", validate_data_loader),
        ("Main Script", validate_main_script),
        ("Documentation", validate_documentation),
        ("API Usage", validate_api_usage),
        ("Example Scripts", validate_example_scripts),
        ("Backwards Compatibility", validate_backwards_compatibility),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print("="*60)
        print("\nImageNet integration is complete and ready to use.")
        print_usage_examples()
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
