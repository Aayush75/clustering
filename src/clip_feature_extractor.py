"""
CLIP feature extraction module.

This module provides functionality to extract features from images using
the CLIP (Contrastive Language-Image Pre-training) model from OpenAI.
CLIP provides powerful visual features suitable for clustering tasks.
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Optional, Tuple
from tqdm import tqdm


class CLIPFeatureExtractor:
    """
    Feature extractor using the CLIP vision transformer model.
    
    This class wraps the CLIP model and provides methods for extracting
    features from batches of images. The extracted features can be used
    for downstream clustering tasks.
    
    Common CLIP models:
    - openai/clip-vit-base-patch32 (default, 512-dim features)
    - openai/clip-vit-base-patch16 (512-dim features, better quality)
    - openai/clip-vit-large-patch14 (768-dim features, highest quality)
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        Initialize the CLIP feature extractor.
        
        Args:
            model_name: Name of the CLIP model to use from HuggingFace
                       Options: clip-vit-base-patch32, clip-vit-base-patch16, 
                               clip-vit-large-patch14, or any compatible CLIP model
            device: Device to run the model on (cuda or cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Initializing CLIP feature extractor on {self.device}")
        
        # Load the model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Get the feature dimension from the model
        self.feature_dim = self.model.config.vision_config.hidden_size
        print(f"CLIP model loaded: {model_name}, feature dimension: {self.feature_dim}")
        
        # ImageNet normalization values (used by data_loader.py)
        # Store as tensors for efficient reuse, on the same device as the model
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
    def extract_features(
        self,
        data_loader,
        return_labels: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract features from all images in the data loader.
        
        This method processes the entire dataset through CLIP and extracts
        the vision embeddings which serve as image-level representations.
        
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
        
        print(f"Extracting features using CLIP...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Feature extraction")):
                # Move images to device
                images = images.to(self.device)
                
                # CLIP expects images in [0, 1] range with specific normalization
                # The data loader returns ImageNet-normalized tensors, we need to denormalize
                # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                
                # Denormalize images from ImageNet normalization
                images_denorm = images * self.imagenet_std + self.imagenet_mean
                # Clamp to [0, 1] range
                images_denorm = torch.clamp(images_denorm, 0, 1)
                
                # Process images through CLIP processor
                # Note: CLIP processor expects images in [0, 255] range or PIL images
                # We'll convert to [0, 255] range
                images_255 = (images_denorm * 255).to(torch.uint8)
                
                # Convert to list of numpy arrays for processor
                # Use batch operation for better performance
                images_numpy = images_255.permute(0, 2, 3, 1).cpu().numpy()
                images_list = [images_numpy[i] for i in range(images_numpy.shape[0])]
                
                # Process through CLIP processor
                inputs = self.processor(images=images_list, return_tensors="pt", padding=True)
                pixel_values = inputs['pixel_values'].to(self.device)
                
                # Extract features using CLIP vision model
                vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                
                # Use the pooled output as the image representation
                # Shape: (batch_size, hidden_size)
                batch_features = vision_outputs.pooler_output
                
                # Apply the vision projection layer to get the final embeddings
                batch_features = self.model.visual_projection(batch_features)
                
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
            'feature_dim': self.feature_dim,
            'model_name': self.model_name
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
        model_name = checkpoint.get('model_name', 'unknown')
        print(f"Features loaded from {path} (model: {model_name})")
        return checkpoint['features'], checkpoint['labels'], checkpoint['feature_dim']
