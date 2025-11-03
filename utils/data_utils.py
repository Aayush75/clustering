"""
Data loading utilities for CIFAR100 dataset.

This module provides data loaders with appropriate preprocessing for DINOv2.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


class CIFAR100DatasetWrapper(Dataset):
    """
    Wrapper around CIFAR100 dataset to provide additional functionality.
    """
    
    def __init__(self, base_dataset, transform=None):
        """
        Initialize the dataset wrapper.
        
        Args:
            base_dataset: The base CIFAR100 dataset
            transform: Optional transform to apply to images
        """
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx


def get_dinov2_transforms(is_training=False, image_size=224):
    """
    Get the appropriate transforms for DINOv2 preprocessing.
    
    DINOv2 expects images normalized with ImageNet statistics.
    
    Args:
        is_training: Whether this is for training (unused for feature extraction)
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    # ImageNet normalization statistics (used by DINOv2)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform_list = [
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    return transforms.Compose(transform_list)


def get_cifar100_dataloaders(config, extract_features=True):
    """
    Create data loaders for CIFAR100 dataset.
    
    Args:
        config: Configuration object
        extract_features: If True, uses transforms suitable for feature extraction
        
    Returns:
        Tuple of (train_loader, test_loader, train_dataset, test_dataset)
    """
    # Get transforms
    transform = get_dinov2_transforms(
        is_training=False,  # No augmentation during feature extraction
        image_size=config.IMAGE_SIZE
    )
    
    # Load CIFAR100 datasets
    train_dataset_base = datasets.CIFAR100(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset_base = datasets.CIFAR100(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    
    # Wrap datasets
    train_dataset = CIFAR100DatasetWrapper(train_dataset_base)
    test_dataset = CIFAR100DatasetWrapper(test_dataset_base)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=not extract_features,  # Don't shuffle when extracting features
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


class EmbeddingDataset(Dataset):
    """
    Dataset wrapper for precomputed embeddings.
    
    This is used during clustering head training to work with cached embeddings
    instead of raw images.
    """
    
    def __init__(self, embeddings, labels, indices=None):
        """
        Initialize the embedding dataset.
        
        Args:
            embeddings: Tensor of shape (N, D) containing embeddings
            labels: Tensor of shape (N,) containing labels
            indices: Optional tensor of indices
        """
        self.embeddings = embeddings
        self.labels = labels
        self.indices = indices if indices is not None else torch.arange(len(embeddings))
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.indices[idx]


def get_embedding_dataloaders(config, train_embeddings, train_labels, 
                               test_embeddings, test_labels):
    """
    Create data loaders for precomputed embeddings.
    
    Args:
        config: Configuration object
        train_embeddings: Training embeddings tensor
        train_labels: Training labels tensor
        test_embeddings: Test embeddings tensor
        test_labels: Test labels tensor
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    test_dataset = EmbeddingDataset(test_embeddings, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Embeddings are already in memory
        pin_memory=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    return train_loader, test_loader
