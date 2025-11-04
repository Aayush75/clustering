"""
Feature extraction module using DINOv2.

This module provides functionality to extract features from images using
the DINOv2 (self-DIstillation with NO labels v2) vision transformer model.
DINOv2 provides powerful visual features suitable for clustering tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import Optional, Tuple
from tqdm import tqdm


class DINOv2FeatureExtractor:
    """
    Feature extractor using the DINOv2 vision transformer model.
    
    This class wraps the DINOv2 model and provides methods for extracting
    features from batches of images. The extracted features can be used
    for downstream clustering tasks.
    """
    
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        """
        Initialize the DINOv2 feature extractor.
        
        Args:
            model_name: Name of the DINOv2 model to use from HuggingFace
                       Options: dinov2-small, dinov2-base, dinov2-large, dinov2-giant
            device: Device to run the model on (cuda or cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing DINOv2 feature extractor on {self.device}")
        
        # Load the DINOv2 model and processor
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Get the feature dimension from the model
        self.feature_dim = self.model.config.hidden_size
        print(f"DINOv2 model loaded: {model_name}, feature dimension: {self.feature_dim}")
        
    def extract_features(
        self,
        data_loader,
        return_labels: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features from all images in the data loader.
        
        This method processes the entire dataset through DINOv2 and extracts
        the CLS token embeddings which serve as image-level representations.
        
        Args:
            data_loader: DataLoader containing the images
            return_labels: If True, also return the ground truth labels
            
        Returns:
            Tuple of (features tensor, labels tensor) if return_labels=True
            Otherwise just features tensor
            
        Note:
            Features are normalized to unit length for better clustering performance
        """
        features_list = []
        labels_list = []
        
        print("Extracting features using DINOv2...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Feature extraction")):
                # Move images to the correct device
                images = images.to(self.device)
                
                # Extract features using DINOv2
                # The model returns a dict with 'last_hidden_state' and 'pooler_output'
                outputs = self.model(pixel_values=images)
                
                # Use the CLS token embedding (first token) as the image representation
                # Shape: (batch_size, hidden_size)
                batch_features = outputs.last_hidden_state[:, 0, :]
                
                # Normalize features to unit length
                # This helps with clustering by making distances more meaningful
                batch_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
                
                # Move to CPU to avoid GPU memory issues
                features_list.append(batch_features.cpu())
                
                if return_labels:
                    labels_list.append(labels)
        
        # Concatenate all batches
        all_features = torch.cat(features_list, dim=0)
        print(f"Feature extraction complete. Shape: {all_features.shape}")
        
        if return_labels:
            all_labels = torch.cat(labels_list, dim=0)
            return all_features, all_labels
        
        return all_features, None
    
    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of the extracted features.
        
        Returns:
            Integer representing the feature dimension
        """
        return self.feature_dim
    
    def save_features(self, features: torch.Tensor, labels: torch.Tensor, path: str):
        """
        Save extracted features and labels to disk.
        
        This allows resuming from extracted features without re-running
        the expensive feature extraction step.
        
        Args:
            features: Feature tensor to save
            labels: Label tensor to save
            path: Path where to save the features
        """
        torch.save({
            'features': features,
            'labels': labels,
            'feature_dim': self.feature_dim
        }, path)
        print(f"Features saved to {path}")
    
    @staticmethod
    def load_features(path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Load previously saved features from disk.
        
        Args:
            path: Path to the saved features file
            
        Returns:
            Tuple of (features, labels, feature_dim)
        """
        checkpoint = torch.load(path)
        print(f"Features loaded from {path}")
        return checkpoint['features'], checkpoint['labels'], checkpoint['feature_dim']
