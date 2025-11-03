"""
Test script to verify installation and basic functionality.

This script performs basic checks to ensure the environment is set up correctly.
"""

import sys
import importlib


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"  ERROR: Python 3.7+ required, found {version.major}.{version.minor}")
        return False
    print(f"  OK: Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """
    Check if a package is installed.
    
    Args:
        package_name: Display name of the package
        import_name: Name to use for import (if different)
    """
    if import_name is None:
        import_name = package_name.lower().replace("-", "_")
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  OK: {package_name} ({version})")
        return True
    except ImportError:
        print(f"  ERROR: {package_name} not found")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  OK: CUDA available")
            print(f"      Device: {torch.cuda.get_device_name(0)}")
            print(f"      CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  WARNING: CUDA not available, will use CPU (slower)")
            return False
    except Exception as e:
        print(f"  ERROR: Could not check CUDA: {e}")
        return False


def check_project_structure():
    """Check if project files exist."""
    print("Checking project structure...")
    from pathlib import Path
    
    required_files = [
        "config.py",
        "train.py",
        "models/clustering_model.py",
        "models/loss.py",
        "utils/data_utils.py",
        "utils/feature_extractor.py",
        "utils/eval_utils.py",
        "utils/trainer.py",
    ]
    
    all_ok = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  OK: {file}")
        else:
            print(f"  ERROR: {file} not found")
            all_ok = False
    
    return all_ok


def test_basic_import():
    """Test if project modules can be imported."""
    print("Testing module imports...")
    
    try:
        from config import Config
        print("  OK: config.Config")
        
        from models.clustering_model import TeacherStudentModel
        print("  OK: models.clustering_model.TeacherStudentModel")
        
        from models.loss import TEMILoss
        print("  OK: models.loss.TEMILoss")
        
        from utils.data_utils import get_cifar100_dataloaders
        print("  OK: utils.data_utils.get_cifar100_dataloaders")
        
        from utils.feature_extractor import DINOv2FeatureExtractor
        print("  OK: utils.feature_extractor.DINOv2FeatureExtractor")
        
        from utils.eval_utils import compute_all_metrics
        print("  OK: utils.eval_utils.compute_all_metrics")
        
        from utils.trainer import Trainer
        print("  OK: utils.trainer.Trainer")
        
        return True
    except Exception as e:
        print(f"  ERROR: Import failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_creation():
    """Test if configuration can be created."""
    print("Testing configuration...")
    
    try:
        from config import Config
        config = Config()
        print(f"  OK: Config created")
        print(f"      Dataset: {config.DATASET_NAME}")
        print(f"      Clusters: {config.NUM_CLUSTERS}")
        print(f"      Heads: {config.NUM_HEADS}")
        return True
    except Exception as e:
        print(f"  ERROR: Config creation failed - {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("TEMI Clustering - Installation Test")
    print("="*80)
    print()
    
    results = {}
    
    # Check Python version
    results['python'] = check_python_version()
    print()
    
    # Check required packages
    print("Checking required packages...")
    packages = [
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("tqdm", "tqdm"),
        ("TensorBoard", "tensorboard"),
        ("Matplotlib", "matplotlib"),
    ]
    
    results['packages'] = all(check_package(name, imp) for name, imp in packages)
    print()
    
    # Check CUDA
    results['cuda'] = check_cuda()
    print()
    
    # Check project structure
    results['structure'] = check_project_structure()
    print()
    
    # Test imports
    results['imports'] = test_basic_import()
    print()
    
    # Test config
    results['config'] = test_config_creation()
    print()
    
    # Summary
    print("="*80)
    print("Test Summary")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name.capitalize():.<20} {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("All tests passed! You're ready to run training.")
        print()
        print("To start training, run:")
        print("  python train.py")
        print()
        print("Or use the quickstart script:")
        print("  python quickstart.py train")
    else:
        print("Some tests failed. Please check the errors above.")
        print()
        print("To install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        print()
        print("If you encounter issues, please check:")
        print("  1. Python version is 3.7 or higher")
        print("  2. All dependencies are installed")
        print("  3. All project files are present")
    
    print()
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
