"""
Simple test script to verify CLIP integration works correctly.
"""

import torch
from src.clip_feature_extractor import CLIPFeatureExtractor

def test_clip_import():
    """Test that CLIP feature extractor can be imported and initialized."""
    print("Testing CLIP import and initialization...")
    try:
        extractor = CLIPFeatureExtractor(
            model_name='openai/clip-vit-base-patch32',
            device='cpu'
        )
        print(f"✓ CLIP initialized successfully")
        print(f"✓ Feature dimension: {extractor.get_feature_dim()}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_clip_extraction():
    """Test that CLIP can extract features from dummy images."""
    print("\nTesting CLIP feature extraction...")
    try:
        # Create dummy data loader
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create dummy CIFAR100-like images (3x32x32)
        num_samples = 10
        dummy_images = torch.randn(num_samples, 3, 32, 32)
        # Normalize like ImageNet (as used by data_loader.py)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        dummy_images = (dummy_images - mean) / std
        
        dummy_labels = torch.randint(0, 100, (num_samples,))
        
        dataset = TensorDataset(dummy_images, dummy_labels)
        loader = DataLoader(dataset, batch_size=5)
        
        # Extract features
        extractor = CLIPFeatureExtractor(
            model_name='openai/clip-vit-base-patch32',
            device='cpu'
        )
        
        features, labels = extractor.extract_features(loader, return_labels=True)
        
        print(f"✓ Feature extraction successful")
        print(f"✓ Features shape: {features.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        
        # Verify features are normalized
        feature_norms = torch.norm(features, p=2, dim=1)
        print(f"✓ Feature norms (should be ~1.0): min={feature_norms.min():.4f}, max={feature_norms.max():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_load():
    """Test that CLIP features can be saved and loaded."""
    print("\nTesting CLIP save/load...")
    try:
        import tempfile
        import os
        
        # Create dummy features
        features = torch.randn(10, 512)
        labels = torch.randint(0, 100, (10,))
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name
        
        try:
            # Initialize extractor
            extractor = CLIPFeatureExtractor(
                model_name='openai/clip-vit-base-patch32',
                device='cpu'
            )
            
            # Save features
            extractor.save_features(features, labels, temp_path)
            print(f"✓ Features saved to {temp_path}")
            
            # Load features
            loaded_features, loaded_labels, feature_dim = CLIPFeatureExtractor.load_features(temp_path)
            print(f"✓ Features loaded successfully")
            print(f"✓ Feature dimension: {feature_dim}")
            
            # Verify they match
            assert torch.allclose(features, loaded_features), "Features don't match!"
            assert torch.equal(labels, loaded_labels), "Labels don't match!"
            print(f"✓ Saved and loaded features match")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("CLIP Feature Extractor Test Suite")
    print("="*60)
    
    results = []
    
    results.append(("Import and Initialization", test_clip_import()))
    results.append(("Feature Extraction", test_clip_extraction()))
    results.append(("Save/Load", test_save_load()))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("="*60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    
    exit(0 if all_passed else 1)
