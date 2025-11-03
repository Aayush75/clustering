"""
Main script for TEMI clustering on CIFAR100 using DINOv2.

This script coordinates the entire clustering pipeline:
1. Load CIFAR100 dataset
2. Extract features using pretrained DINOv2
3. Train clustering heads using TEMI loss
4. Evaluate clustering performance
"""

import torch
import numpy as np
import random
import argparse
from pathlib import Path

# Import project modules
from config import Config
from utils.data_utils import get_cifar100_dataloaders, get_embedding_dataloaders
from utils.feature_extractor import DINOv2FeatureExtractor
from models.clustering_model import TeacherStudentModel
from models.loss import TEMILoss, MultiHeadTEMILoss
from utils.trainer import Trainer
from utils.eval_utils import knn_classifier


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main function to run the clustering experiment.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = Config()
    
    # Override config with command line arguments if provided
    if args.resume is not None:
        config.RESUME_FROM_CHECKPOINT = args.resume
    if args.force_recompute:
        print("Force recomputing embeddings...")
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create directories
    config.create_directories()
    
    # Print configuration
    config.print_config()
    
    print("\n" + "="*80)
    print("Step 1: Loading CIFAR100 Dataset")
    print("="*80)
    
    # Load CIFAR100 data loaders
    train_loader, test_loader, train_dataset, test_dataset = get_cifar100_dataloaders(
        config, extract_features=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\n" + "="*80)
    print("Step 2: Extracting DINOv2 Features")
    print("="*80)
    
    # Initialize feature extractor
    feature_extractor = DINOv2FeatureExtractor(config)
    
    # Extract and cache embeddings
    train_embeddings, train_labels, test_embeddings, test_labels = \
        feature_extractor.compute_and_cache_embeddings(
            train_loader, test_loader, force_recompute=args.force_recompute
        )
    
    print(f"\nTrain embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    # Compute KNN baseline if requested
    if config.COMPUTE_KNN_ACCURACY:
        print("\n" + "="*80)
        print("Computing KNN Baseline Accuracy")
        print("="*80)
        
        top1, top5 = knn_classifier(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
            k=config.KNN_K,
            temperature=config.KNN_TEMPERATURE
        )
        
        print(f"KNN (k={config.KNN_K}) Accuracy:")
        print(f"  Top-1: {top1:.2f}%")
        print(f"  Top-5: {top5:.2f}%")
    
    print("\n" + "="*80)
    print("Step 3: Creating Data Loaders for Embeddings")
    print("="*80)
    
    # Create data loaders for embeddings
    train_embed_loader, test_embed_loader = get_embedding_dataloaders(
        config,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels
    )
    
    print(f"Training batches: {len(train_embed_loader)}")
    print(f"Test batches: {len(test_embed_loader)}")
    
    print("\n" + "="*80)
    print("Step 4: Initializing Clustering Model")
    print("="*80)
    
    # Initialize teacher-student model
    model = TeacherStudentModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.student.parameters())
    trainable_params = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
    
    print(f"Model initialized with {config.NUM_HEADS} heads")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*80)
    print("Step 5: Initializing TEMI Loss")
    print("="*80)
    
    # Initialize loss function
    if config.NUM_HEADS > 1:
        loss_fn = MultiHeadTEMILoss(config)
        print(f"Using Multi-Head TEMI loss with {config.NUM_HEADS} heads")
    else:
        loss_fn = TEMILoss(config)
        print("Using single-head TEMI loss")
    
    print(f"Beta parameter: {config.BETA}")
    print(f"Student temperature: {config.STUDENT_TEMP}")
    print(f"Teacher temperature: {config.TEACHER_TEMP}")
    
    print("\n" + "="*80)
    print("Step 6: Training Clustering Heads")
    print("="*80)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        train_loader=train_embed_loader,
        test_loader=test_embed_loader,
        train_labels=train_labels,
        test_labels=test_labels
    )
    
    # Train
    try:
        final_metrics = trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        print("Checkpoint saved. You can resume training later.")
        return
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("Saving emergency checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        raise
    
    print("\n" + "="*80)
    print("Clustering Experiment Completed Successfully!")
    print("="*80)
    print(f"\nBest test accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Final test accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"Final NMI: {final_metrics['nmi']:.2f}%")
    print(f"Final ARI: {final_metrics['ari']:.2f}%")
    print(f"\nCheckpoints saved in: {config.CHECKPOINT_DIR}")
    print(f"Results saved in: {config.RESULTS_DIR}")
    print(f"Logs saved in: {config.LOG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TEMI Deep Clustering on CIFAR100 using DINOv2"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
