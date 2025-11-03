"""
Feature extraction module using DINOv2 pretrained models.

This module handles extracting and caching embeddings from CIFAR100 images
using the DINOv2 vision transformer.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np


class DINOv2FeatureExtractor:
    """
    Extracts features from images using pretrained DINOv2 models.
    """
    
    def __init__(self, config):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        
        # Load pretrained DINOv2 model
        print(f"Loading DINOv2 model: {config.DINOV2_MODEL}")
        self.model = self._load_dinov2_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Embedding dimension: {config.EMBEDDING_DIM}")
    
    def _load_dinov2_model(self):
        """
        Load the DINOv2 model from torch hub.
        
        Returns:
            DINOv2 model
        """
        # Available models:
        # - dinov2_vits14: Small model, 384 dims
        # - dinov2_vitb14: Base model, 768 dims
        # - dinov2_vitl14: Large model, 1024 dims
        # - dinov2_vitg14: Giant model, 1536 dims
        
        model = torch.hub.load('facebookresearch/dinov2', self.config.DINOV2_MODEL)
        
        # Freeze all parameters since we only extract features
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    @torch.no_grad()
    def extract_features(self, data_loader, desc="Extracting features"):
        """
        Extract features from a data loader.
        
        Args:
            data_loader: DataLoader providing images
            desc: Description for progress bar
            
        Returns:
            Tuple of (embeddings, labels, indices) as tensors
        """
        embeddings_list = []
        labels_list = []
        indices_list = []
        
        self.model.eval()
        
        for batch in tqdm(data_loader, desc=desc):
            images, labels, indices = batch
            images = images.to(self.device)
            
            # Extract features
            # DINOv2 returns CLS token embedding
            features = self.model(images)
            
            embeddings_list.append(features.cpu())
            labels_list.append(labels)
            indices_list.append(indices)
        
        # Concatenate all batches
        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        indices = torch.cat(indices_list, dim=0)
        
        return embeddings, labels, indices
    
    def compute_and_cache_embeddings(self, train_loader, test_loader, force_recompute=False):
        """
        Compute embeddings for train and test sets, with caching.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            force_recompute: If True, recompute even if cache exists
            
        Returns:
            Tuple of (train_embeddings, train_labels, test_embeddings, test_labels)
        """
        train_path = self.config.get_embeddings_path("train")
        test_path = self.config.get_embeddings_path("test")
        
        # Check if cached embeddings exist
        if not force_recompute and train_path.exists() and test_path.exists():
            print("Loading cached embeddings...")
            train_data = torch.load(train_path)
            test_data = torch.load(test_path)
            
            train_embeddings = train_data['embeddings']
            train_labels = train_data['labels']
            test_embeddings = test_data['embeddings']
            test_labels = test_data['labels']
            
            print(f"Loaded train embeddings: {train_embeddings.shape}")
            print(f"Loaded test embeddings: {test_embeddings.shape}")
        else:
            print("Computing embeddings from scratch...")
            
            # Extract train embeddings
            train_embeddings, train_labels, train_indices = self.extract_features(
                train_loader, desc="Extracting train features"
            )
            
            # Extract test embeddings
            test_embeddings, test_labels, test_indices = self.extract_features(
                test_loader, desc="Extracting test features"
            )
            
            # Save to cache
            print("Saving embeddings to cache...")
            torch.save({
                'embeddings': train_embeddings,
                'labels': train_labels,
                'indices': train_indices
            }, train_path)
            
            torch.save({
                'embeddings': test_embeddings,
                'labels': test_labels,
                'indices': test_indices
            }, test_path)
            
            print(f"Saved train embeddings: {train_embeddings.shape}")
            print(f"Saved test embeddings: {test_embeddings.shape}")
        
        return train_embeddings, train_labels, test_embeddings, test_labels
    
    def compute_nearest_neighbors(self, embeddings, k=50):
        """
        Compute k-nearest neighbors for each embedding.
        
        This is used by the TEMI loss to weight the mutual information objective.
        
        Args:
            embeddings: Tensor of shape (N, D)
            k: Number of nearest neighbors
            
        Returns:
            Tuple of (distances, indices) for k-nearest neighbors
        """
        print(f"Computing {k}-nearest neighbors...")
        
        embeddings = embeddings.to(self.device)
        
        # Normalize embeddings
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        
        # Compute cosine similarity matrix
        # We do this in chunks to avoid memory issues
        chunk_size = 1000
        num_samples = embeddings.shape[0]
        
        all_distances = []
        all_indices = []
        
        for i in tqdm(range(0, num_samples, chunk_size), desc="Computing neighbors"):
            end_idx = min(i + chunk_size, num_samples)
            chunk = embeddings_norm[i:end_idx]
            
            # Compute similarity with all embeddings
            similarities = torch.mm(chunk, embeddings_norm.t())
            
            # Get top-k (including self)
            distances, indices = similarities.topk(k + 1, dim=1, largest=True)
            
            # Remove self (first entry)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
            
            all_distances.append(distances.cpu())
            all_indices.append(indices.cpu())
        
        distances = torch.cat(all_distances, dim=0)
        indices = torch.cat(all_indices, dim=0)
        
        print(f"Computed nearest neighbors: {distances.shape}")
        
        return distances, indices
