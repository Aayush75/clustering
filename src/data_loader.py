"""
Data loading module for CIFAR100 dataset.

This module handles loading and preprocessing of the CIFAR100 dataset
for clustering tasks. It provides utilities for creating data loaders
with appropriate transformations for the DINOv2 model.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np


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


def create_data_loaders(
    root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test data loaders for CIFAR100.
    
    Args:
        root: Root directory for dataset storage
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        image_size: Target image size for resizing
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
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


def get_dataset_statistics(data_loader: DataLoader) -> dict:
    """
    Compute statistics for the dataset.
    
    Args:
        data_loader: DataLoader to compute statistics for
        
    Returns:
        Dictionary containing dataset statistics
    """
    num_samples = len(data_loader.dataset)
    num_classes = 100  # CIFAR100 has 100 classes
    
    return {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'batch_size': data_loader.batch_size
    }
