"""
Configuration file for TEMI clustering on CIFAR100 using DINOv2.

This file contains all hyperparameters and settings for the clustering experiment.
The configuration is organized into logical sections for easy modification.
"""

import os
from pathlib import Path


class Config:
    """
    Main configuration class containing all hyperparameters and settings.
    """
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    
    # Dataset settings
    DATASET_NAME = "CIFAR100"
    NUM_CLASSES = 100
    NUM_TRAIN_SAMPLES = 50000
    NUM_TEST_SAMPLES = 10000
    IMAGE_SIZE = 224  # DINOv2 expects 224x224 images
    
    # DINOv2 model settings
    DINOV2_MODEL = "dinov2_vitb14"  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    EMBEDDING_DIM = 768  # 384 for vits14, 768 for vitb14, 1024 for vitl14, 1536 for vitg14
    FREEZE_BACKBONE = True  # We only train the clustering heads
    
    # TEMI clustering settings
    NUM_CLUSTERS = 100  # K value for clustering
    NUM_HEADS = 16  # Number of clustering heads for ensemble
    HIDDEN_DIM = 2048  # Hidden dimension for clustering heads
    KNN_NEIGHBORS = 50  # Number of nearest neighbors to consider
    
    # Loss function parameters
    STUDENT_TEMP = 0.1  # Student temperature for softmax
    TEACHER_TEMP = 0.07  # Final teacher temperature (slightly higher for 100 clusters)
    WARMUP_TEACHER_TEMP = 0.04  # Starting teacher temperature
    WARMUP_TEACHER_EPOCHS = 30  # Epochs to warmup teacher temperature (longer warmup)
    CENTER_MOMENTUM = 0.9  # Momentum for center update
    PROBS_MOMENTUM = 0.95  # Momentum for probability distribution (higher for stability)
    BETA = 0.6  # Beta parameter for weighted MI
    REGULARIZATION_WEIGHT = 10.0  # Alpha parameter for entropy regularization (HIGHER for 100 clusters!)
    USE_REGULARIZATION = True  # Whether to use entropy regularization - MUST be True
    
    # Training settings
    BATCH_SIZE = 256  # Batch size for training
    NUM_EPOCHS = 100  # Total number of training epochs
    LEARNING_RATE = 1e-3  # Initial learning rate (10x higher for better convergence)
    MIN_LR = 1e-5  # Minimum learning rate
    WARMUP_EPOCHS = 10  # Number of epochs for learning rate warmup (shorter)
    WEIGHT_DECAY = 1e-5  # Weight decay for optimizer (lower to prevent over-regularization)
    MOMENTUM_TEACHER = 0.996  # EMA momentum for teacher network
    
    # Optimizer settings
    OPTIMIZER = "adamw"  # Options: adam, adamw, sgd
    CLIP_GRAD = 3.0  # Gradient clipping value (None to disable)
    
    # Data augmentation (for embeddings)
    USE_DATA_AUGMENTATION = False  # Whether to use augmentation during embedding extraction
    
    # Training behavior
    USE_FP16 = False  # Use mixed precision training
    NUM_WORKERS = 4  # Number of data loading workers
    PIN_MEMORY = True  # Pin memory for faster data transfer
    
    # Checkpointing
    SAVE_CHECKPOINT_FREQ = 10  # Save checkpoint every N epochs
    KEEP_ALL_CHECKPOINTS = False  # Keep all checkpoints or only best and latest
    RESUME_FROM_CHECKPOINT = None  # Path to checkpoint to resume from (None for fresh start)
    
    # Evaluation settings
    EVAL_FREQ = 5  # Evaluate every N epochs
    COMPUTE_KNN_ACCURACY = True  # Whether to compute KNN accuracy on embeddings
    KNN_K = 20  # K for KNN evaluation
    KNN_TEMPERATURE = 0.07  # Temperature for KNN classifier
    
    # Logging
    LOG_FREQ = 50  # Log training stats every N iterations
    TENSORBOARD_ENABLED = True  # Enable TensorBoard logging
    SAVE_RESULTS_JSON = True  # Save results to JSON file
    VERBOSE = True  # Print detailed logging
    
    # Reproducibility
    SEED = 42  # Random seed for reproducibility
    DETERMINISTIC = True  # Use deterministic algorithms
    
    # Device settings
    DEVICE = "cuda"  # Options: cuda, cpu
    
    @classmethod
    def create_directories(cls):
        """
        Create all necessary directories if they don't exist.
        """
        directories = [
            cls.DATA_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
            cls.EMBEDDINGS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("Created project directories:")
        for directory in directories:
            print(f"  - {directory}")
    
    @classmethod
    def print_config(cls):
        """
        Print the current configuration.
        """
        print("\n" + "="*80)
        print("TEMI Clustering Configuration")
        print("="*80)
        
        sections = {
            "Dataset": [
                ("Dataset", cls.DATASET_NAME),
                ("Number of classes", cls.NUM_CLASSES),
                ("Image size", f"{cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}"),
            ],
            "Model": [
                ("DINOv2 model", cls.DINOV2_MODEL),
                ("Embedding dimension", cls.EMBEDDING_DIM),
                ("Number of heads", cls.NUM_HEADS),
                ("Number of clusters", cls.NUM_CLUSTERS),
            ],
            "Training": [
                ("Batch size", cls.BATCH_SIZE),
                ("Number of epochs", cls.NUM_EPOCHS),
                ("Learning rate", cls.LEARNING_RATE),
                ("Weight decay", cls.WEIGHT_DECAY),
                ("Teacher momentum", cls.MOMENTUM_TEACHER),
            ],
            "Loss": [
                ("Beta", cls.BETA),
                ("Student temperature", cls.STUDENT_TEMP),
                ("Teacher temperature", cls.TEACHER_TEMP),
                ("KNN neighbors", cls.KNN_NEIGHBORS),
            ],
        }
        
        for section_name, items in sections.items():
            print(f"\n{section_name}:")
            for name, value in items:
                print(f"  {name:.<40} {value}")
        
        print("\n" + "="*80 + "\n")
    
    @classmethod
    def get_checkpoint_path(cls, epoch=None):
        """
        Get the path for a checkpoint file.
        
        Args:
            epoch: Epoch number (None for latest checkpoint)
            
        Returns:
            Path to checkpoint file
        """
        if epoch is None:
            return cls.CHECKPOINT_DIR / "checkpoint_latest.pth"
        else:
            return cls.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:04d}.pth"
    
    @classmethod
    def get_best_checkpoint_path(cls):
        """
        Get the path for the best checkpoint file.
        
        Returns:
            Path to best checkpoint file
        """
        return cls.CHECKPOINT_DIR / "checkpoint_best.pth"
    
    @classmethod
    def get_embeddings_path(cls, split="train"):
        """
        Get the path for cached embeddings.
        
        Args:
            split: Dataset split (train or test)
            
        Returns:
            Path to embeddings file
        """
        return cls.EMBEDDINGS_DIR / f"embeddings_{split}.pt"
