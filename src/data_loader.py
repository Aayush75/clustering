"""
Data loading module for CIFAR10, CIFAR100, ImageNet, Tiny ImageNet, and Imagenette datasets.

This module handles loading and preprocessing of multiple datasets
for clustering tasks. It provides utilities for creating data loaders
with appropriate transformations for the DINOv2 and CLIP models.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import os
import tarfile
import urllib.request
from pathlib import Path

# Supported datasets
SUPPORTED_DATASETS = ['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', 'imagenette']


class CIFAR100Dataset(Dataset):
    """
    Wrapper for CIFAR100 dataset with custom transformations.
    
    This dataset class handles both training and test splits of CIFAR100,
    applying transformations suitable for DINOv2 feature extraction.
    Supports both standard CIFAR-100 format and folder-organized format.
    """
    
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True, 
                 use_folder_structure: bool = False):
        """
        Initialize the CIFAR100 dataset.
        
        Args:
            root: Root directory where dataset will be stored
            train: If True, use training set, otherwise use test set
            transform: Optional transform to apply to images
            download: If True, download the dataset if not present
            use_folder_structure: If True, load from cifar100/train or cifar100/test folder structure
        """
        self.transform = transform
        self.use_folder_structure = use_folder_structure
        
        if use_folder_structure:
            # Load from folder structure: data/cifar100/train/0/*.png or data/cifar100/test/0/*.png
            self._load_from_folders(root, train)
        else:
            # Use standard torchvision CIFAR100
            self.cifar100 = datasets.CIFAR100(
                root=root,
                train=train,
                download=download,
                transform=transform
            )
    
    def _load_from_folders(self, root: str, train: bool):
        """Load CIFAR-100 from folder structure organized by class."""
        root_path = Path(root) / 'cifar100'
        
        # Try 'train' and 'test' first, then 'train' and 'val'
        if train:
            split_dir = root_path / 'train'
        else:
            # Try 'test' first, then 'val'
            test_dir = root_path / 'test'
            val_dir = root_path / 'val'
            if test_dir.exists():
                split_dir = test_dir
            elif val_dir.exists():
                split_dir = val_dir
            else:
                raise FileNotFoundError(
                    f"CIFAR-100 test/val folder not found. Tried: {test_dir} and {val_dir}. "
                    f"Expected structure: {root}/cifar100/train/0/*.png and {root}/cifar100/test/0/*.png (or val/)"
                )
        
        if not split_dir.exists():
            raise FileNotFoundError(
                f"CIFAR-100 folder structure not found at {split_dir}. "
                f"Expected structure: {root}/cifar100/train/0/*.png or {root}/cifar100/test/0/*.png (or val/)"
            )
        
        # Collect all images and labels
        self.samples = []
        self.labels = []
        
        # Get all class directories (sorted numerically)
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], 
                           key=lambda x: int(x.name))
        
        # Load CIFAR-100 class names
        self.class_names = self._get_cifar100_class_names()
        
        # Build file list
        for class_idx, class_dir in enumerate(class_dirs):
            # Get all image files in this class directory
            image_files = sorted(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
            
            for img_path in image_files:
                self.samples.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images from {split_dir} with {len(class_dirs)} classes")
    
    def _get_cifar100_class_names(self):
        """Get CIFAR-100 class names in order."""
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
            'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
        
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        if self.use_folder_structure:
            return len(self.samples)
        return len(self.cifar100)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image tensor, label)
        """
        if self.use_folder_structure:
            img_path = self.samples[idx]
            label = self.labels[idx]
            
            # Load image
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                # Return a blank image if loading fails
                image = Image.new('RGB', (32, 32), color='black')
            
            # Apply transformations
            if self.transform is not None:
                image = self.transform(image)
            
            return image, label
        else:
            return self.cifar100[idx]


class CIFAR10Dataset(Dataset):
    """
    Wrapper for CIFAR10 dataset with custom transformations.
    
    This dataset class handles both training and test splits of CIFAR10,
    applying transformations suitable for DINOv2 feature extraction.
    CIFAR10 has 10 classes instead of 100.
    """
    
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        """
        Initialize the CIFAR10 dataset.
        
        Args:
            root: Root directory where dataset will be stored
            train: If True, use training set, otherwise use test set
            transform: Optional transform to apply to images
            download: If True, download the dataset if not present
        """
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.cifar10)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image tensor, label)
        """
        return self.cifar10[idx]


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


class TinyImageNetDataset(Dataset):
    """
    Wrapper for Tiny ImageNet dataset from HuggingFace with custom transformations.
    
    This dataset class handles Tiny ImageNet dataset (64x64 version with 200 classes),
    applying transformations suitable for DINOv2 and CLIP feature extraction.
    """
    
    def __init__(self, dataset_name: str = "zh-plus/tiny-imagenet", 
                 split: str = "train", transform=None, streaming: bool = False):
        """
        Initialize the Tiny ImageNet dataset from HuggingFace.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split ('train' or 'valid')
            transform: Optional transform to apply to images
            streaming: If True, use streaming mode (for large datasets)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library (version >= 2.14.0) is required for loading Tiny ImageNet. "
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


class ImagenetteDataset(Dataset):
    """
    Wrapper for Imagenette dataset from fastai with custom transformations.
    
    Imagenette is a subset of 10 easily classified classes from ImageNet.
    This dataset class handles downloading and loading the dataset.
    """
    
    # Imagenette class names (WordNet IDs to class names mapping)
    CLASS_NAMES = [
        'tench', 'English springer', 'cassette player', 'chain saw', 'church',
        'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
    ]
    
    # URLs for different versions
    URLS = {
        'full': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
        '160': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
        '320': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
    }
    
    def __init__(self, root: str, split: str = "train", transform=None, 
                 download: bool = True, version: str = '320'):
        """
        Initialize the Imagenette dataset.
        
        Args:
            root: Root directory where dataset will be stored
            split: Dataset split ('train' or 'val')
            transform: Optional transform to apply to images
            download: If True, download the dataset if not present
            version: Version of dataset ('full', '160', or '320')
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.version = version
        
        # Determine dataset directory name based on version
        if version == 'full':
            self.dataset_dir = self.root / 'imagenette2'
        else:
            self.dataset_dir = self.root / f'imagenette2-{version}'
        
        self.split_dir = self.dataset_dir / split
        
        # Download if needed
        if download and not self.split_dir.exists():
            self._download_and_extract()
        
        if not self.split_dir.exists():
            raise RuntimeError(
                f"Dataset not found at {self.split_dir}. "
                f"Set download=True to download it automatically."
            )
        
        # Build file list and labels
        self.samples = []
        self.labels = []
        self._build_file_list()
        
    def _download_and_extract(self):
        """Download and extract the Imagenette dataset."""
        url = self.URLS[self.version]
        
        if self.version == 'full':
            filename = 'imagenette2.tgz'
        else:
            filename = f'imagenette2-{self.version}.tgz'
        
        filepath = self.root / filename
        
        # Create root directory if needed
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Download if not already present
        if not filepath.exists():
            print(f"Downloading Imagenette {self.version} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Download complete: {filepath}")
            except Exception as e:
                raise RuntimeError(f"Failed to download Imagenette: {e}")
        
        # Extract if not already extracted
        if not self.dataset_dir.exists():
            print(f"Extracting {filepath}...")
            try:
                with tarfile.open(filepath, 'r:gz') as tar:
                    tar.extractall(self.root)
                print(f"Extraction complete: {self.dataset_dir}")
            except Exception as e:
                raise RuntimeError(f"Failed to extract Imagenette: {e}")
        
    def _build_file_list(self):
        """Build list of image files and their labels."""
        # Get all class directories (sorted for consistent ordering)
        class_dirs = sorted([d for d in self.split_dir.iterdir() if d.is_dir()])
        
        # Build samples list with labels
        for class_idx, class_dir in enumerate(class_dirs):
            # Get all image files in this class directory
            image_files = sorted(list(class_dir.glob('*.JPEG')))
            
            for img_path in image_files:
                self.samples.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.samples)} images from {self.split} split with {len(class_dirs)} classes")
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


def get_cifar10_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transformations for CIFAR10 suitable for DINOv2.
    
    DINOv2 expects images normalized with ImageNet statistics.
    We resize CIFAR10 images from 32x32 to 224x224 to match DINOv2 input size.
    
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


def get_tiny_imagenet_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transformations for Tiny ImageNet suitable for DINOv2 and CLIP.
    
    Tiny ImageNet images are 64x64, so we resize to the target size.
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


def get_imagenette_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get transformations for Imagenette suitable for DINOv2 and CLIP.
    
    Imagenette images vary in size (160-320px depending on version),
    so we resize to the target size. DINOv2 and CLIP expect images 
    normalized with ImageNet statistics.
    
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
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transforms without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, test_transform


def create_data_loaders(
    root: str,
    batch_size: int = 256,
    num_workers: int = 4,
    image_size: int = 224,
    dataset_name: str = 'cifar100',
    use_folder_structure: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test data loaders for specified dataset.
    
    Args:
        root: Root directory for dataset storage
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        image_size: Target image size for resizing
        dataset_name: Name of dataset to load ('cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', or 'imagenette')
        use_folder_structure: If True, load CIFAR-100 from folder structure (data/cifar100/train/0/*.png)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_name.lower() == 'cifar10':
        train_transform, test_transform = get_cifar10_transforms(image_size)
        
        train_dataset = CIFAR10Dataset(
            root=root,
            train=True,
            transform=train_transform,
            download=True
        )
        
        test_dataset = CIFAR10Dataset(
            root=root,
            train=False,
            transform=test_transform,
            download=True
        )
    elif dataset_name.lower() == 'cifar100':
        train_transform, test_transform = get_cifar100_transforms(image_size)
        
        train_dataset = CIFAR100Dataset(
            root=root,
            train=True,
            transform=train_transform,
            download=True,
            use_folder_structure=use_folder_structure
        )
        
        test_dataset = CIFAR100Dataset(
            root=root,
            train=False,
            transform=test_transform,
            download=True,
            use_folder_structure=use_folder_structure
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
    elif dataset_name.lower() == 'tiny-imagenet':
        train_transform, test_transform = get_tiny_imagenet_transforms(image_size)
        
        # Use the HuggingFace dataset
        hf_dataset_name = "zh-plus/tiny-imagenet"
        
        train_dataset = TinyImageNetDataset(
            dataset_name=hf_dataset_name,
            split='train',
            transform=train_transform,
            streaming=False
        )
        
        test_dataset = TinyImageNetDataset(
            dataset_name=hf_dataset_name,
            split='valid',
            transform=test_transform,
            streaming=False
        )
    elif dataset_name.lower() == 'imagenette':
        train_transform, test_transform = get_imagenette_transforms(image_size)
        
        train_dataset = ImagenetteDataset(
            root=root,
            split='train',
            transform=train_transform,
            download=True,
            version='320'  # Use 320px version for good balance of quality and speed
        )
        
        test_dataset = ImagenetteDataset(
            root=root,
            split='val',
            transform=test_transform,
            download=True,
            version='320'
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
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'imagenet', 'tiny-imagenet', or 'imagenette')
        
    Returns:
        Dictionary containing dataset statistics
    """
    num_samples = len(data_loader.dataset)
    
    # Determine number of classes based on dataset
    if dataset_name.lower() == 'cifar10':
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        num_classes = 100
    elif dataset_name.lower() == 'imagenet':
        num_classes = 1000  # ImageNet-1K has 1000 classes
    elif dataset_name.lower() == 'tiny-imagenet':
        num_classes = 200  # Tiny ImageNet has 200 classes
    elif dataset_name.lower() == 'imagenette':
        num_classes = 10  # Imagenette has 10 classes
    else:
        num_classes = None
    
    return {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'batch_size': data_loader.batch_size
    }
