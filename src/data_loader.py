"""
Data loading module for CIFAR100 and ImageNet datasets.

This module handles loading and preprocessing of the CIFAR100 and ImageNet datasets
for clustering tasks. It provides utilities for creating data loaders
with appropriate transformations for the DINOv2 and CLIP models.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image

# Supported datasets
SUPPORTED_DATASETS = ['cifar100', 'imagenet']


class CIFAR100Dataset(Dataset):
    """
    Wrapper for CIFAR100 dataset with custom transformations.
    
    This dataset class handles both training and test splits of CIFAR100,
    applying transformations suitable for DINOv2 feature extraction.
    """
    
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        """
        Initialize the CIFAR100 dataset.
        
        Args:
            root: Root directory where dataset will be stored
            train: If True, use training set, otherwise use test set
            transform: Optional transform to apply to images
            download: If True, download the dataset if not present
        """
        self.cifar100 = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.cifar100)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image tensor, label)
        """
        return self.cifar100[idx]


class ImageNetDataset(Dataset):
    """
    Wrapper for ImageNet dataset from HuggingFace with custom transformations.
    
    This dataset class handles ImageNet-1K dataset (128x128 version),
    applying transformations suitable for DINOv2 and CLIP feature extraction.
    """
    
    def __init__(self, dataset_name: str = "benjamin-paine/imagenet-1k-128x128", 
                 split: str = "train", transform=None, streaming: bool = False):
        """
        Initialize the ImageNet dataset from HuggingFace.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split ('train' or 'validation')
            transform: Optional transform to apply to images
            streaming: If True, use streaming mode (for large datasets)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library (version >= 2.14.0) is required for loading ImageNet. "
                "Install it with: pip install 'datasets>=2.14.0'"
            )
        
        self.transform = transform
        self.split = split
        self.streaming = streaming
        
        # Load dataset from HuggingFace
        # Note: Streaming mode is not fully supported yet - use streaming=False
        if streaming:
            raise NotImplementedError(
                "Streaming mode is not currently supported. "
                "Please use streaming=False (default) for full dataset loading."
            )
        
        self.dataset = load_dataset(dataset_name, split=split)
        self._length = len(self.dataset)
        
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image tensor, label)
        """
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']
        
        # Convert to PIL Image if not already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def get_cifar100_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transformations for CIFAR100 suitable for DINOv2.
    
    DINOv2 expects images normalized with ImageNet statistics.
    We resize CIFAR100 images from 32x32 to 224x224 to match DINOv2 input size.
    
    Args:
        image_size: Target image size (default: 224 for DINOv2)
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # ImageNet normalization values used by DINOv2
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with data augmentation for TEMI
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, test_transform


def get_imagenet_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transformations for ImageNet suitable for DINOv2 and CLIP.
    
    ImageNet images are already 128x128, so we resize to the target size.
    DINOv2 and CLIP expect images normalized with ImageNet statistics.
    
    Args:
        image_size: Target image size (default: 224 for DINOv2/CLIP)
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # ImageNet normalization values used by DINOv2 and CLIP
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with data augmentation for TEMI
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, test_transform


def create_data_loaders(
    root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    image_size: int = 224,
    dataset_name: str = 'cifar100'
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test data loaders for specified dataset.
    
    Args:
        root: Root directory for dataset storage
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        image_size: Target image size for resizing
        dataset_name: Name of dataset to load ('cifar100' or 'imagenet')
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_name.lower() == 'cifar100':
        train_transform, test_transform = get_cifar100_transforms(image_size)
        
        train_dataset = CIFAR100Dataset(
            root=root,
            train=True,
            transform=train_transform,
            download=True
        )
        
        test_dataset = CIFAR100Dataset(
            root=root,
            train=False,
            transform=test_transform,
            download=True
        )
    elif dataset_name.lower() == 'imagenet':
        train_transform, test_transform = get_imagenet_transforms(image_size)
        
        # Use the HuggingFace dataset
        hf_dataset_name = "benjamin-paine/imagenet-1k-128x128"
        
        train_dataset = ImageNetDataset(
            dataset_name=hf_dataset_name,
            split='train',
            transform=train_transform,
            streaming=False
        )
        
        test_dataset = ImageNetDataset(
            dataset_name=hf_dataset_name,
            split='validation',
            transform=test_transform,
            streaming=False
        )
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_dataset_statistics(data_loader: DataLoader, dataset_name: str = 'cifar100') -> dict:
    """
    Compute statistics for the dataset.
    
    Args:
        data_loader: DataLoader to compute statistics for
        dataset_name: Name of the dataset ('cifar100' or 'imagenet')
        
    Returns:
        Dictionary containing dataset statistics
    """
    num_samples = len(data_loader.dataset)
    
    # Determine number of classes based on dataset
    if dataset_name.lower() == 'cifar100':
        num_classes = 100
    elif dataset_name.lower() == 'imagenet':
        num_classes = 1000  # ImageNet-1K has 1000 classes
    else:
        num_classes = None
    
    return {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'batch_size': data_loader.batch_size
    }
