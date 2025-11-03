"""
Training utilities for TEMI clustering.

This module provides functions for training, checkpointing, and managing
the training loop.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np


def convert_to_json_serializable(obj):
    """
    Convert numpy types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor,)):
        return obj.cpu().numpy().tolist()
    else:
        return obj


class Trainer:
    """
    Trainer class for TEMI clustering.
    
    Handles the training loop, evaluation, checkpointing, and logging.
    """
    
    def __init__(self, model, loss_fn, config, train_loader, test_loader,
                 train_labels, test_labels):
        """
        Initialize trainer.
        
        Args:
            model: TeacherStudentModel instance
            loss_fn: TEMI loss function
            config: Configuration object
            train_loader: Training data loader (embeddings)
            test_loader: Test data loader (embeddings)
            train_labels: Training labels tensor
            test_labels: Test labels tensor
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler
        self.lr_scheduler = self._setup_lr_scheduler()
        
        # Tracking
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_stats = []
        
        # TensorBoard writer
        if config.TENSORBOARD_ENABLED:
            self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        else:
            self.writer = None
    
    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        params = self.model.student.parameters()
        
        if self.config.OPTIMIZER.lower() == "adam":
            optimizer = optim.Adam(
                params,
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER.lower() == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER.lower() == "sgd":
            optimizer = optim.SGD(
                params,
                lr=self.config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.OPTIMIZER}")
        
        return optimizer
    
    def _setup_lr_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.config.WARMUP_EPOCHS:
                # Linear warmup
                return epoch / self.config.WARMUP_EPOCHS
            else:
                # Cosine annealing
                progress = (epoch - self.config.WARMUP_EPOCHS) / (self.config.NUM_EPOCHS - self.config.WARMUP_EPOCHS)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (embeddings, labels, indices) in enumerate(pbar):
            embeddings = embeddings.to(self.device)
            
            # Forward pass
            student_outputs, teacher_outputs = self.model(embeddings)
            
            # Compute loss
            loss = self.loss_fn(student_outputs, teacher_outputs, epoch)
            
            # Check for NaN or inf
            if not torch.isfinite(loss):
                print(f"\nWarning: Non-finite loss detected at batch {batch_idx}, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.CLIP_GRAD is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.student.parameters(),
                    self.config.CLIP_GRAD
                )
                # Check for exploding gradients
                if not torch.isfinite(grad_norm):
                    print(f"\nWarning: Non-finite gradients at batch {batch_idx}, skipping batch")
                    self.optimizer.zero_grad()
                    continue
            
            self.optimizer.step()
            
            # Update teacher with EMA
            self.model.update_teacher()
            
            # Track loss
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1
            
            # Update progress bar
            if num_batches > 0:
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
            else:
                pbar.set_postfix({'loss': 'skipped'})
            
            # Log to TensorBoard
            if self.writer is not None and batch_idx % self.config.LOG_FREQ == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/loss_step', loss_value, global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Compute average loss (handle case where all batches were skipped)
        avg_loss = total_loss / max(num_batches, 1)
        
        # Log epoch statistics
        if self.writer is not None:
            self.writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(self, epoch, split='test'):
        """
        Evaluate clustering performance.
        
        Args:
            epoch: Current epoch number
            split: Which split to evaluate ('train' or 'test')
            
        Returns:
            Dictionary of evaluation metrics
        """
        from utils.eval_utils import compute_all_metrics, compute_cluster_statistics
        
        self.model.eval()
        
        # Select data loader and labels
        if split == 'test':
            loader = self.test_loader
            labels = self.test_labels
        else:
            loader = self.train_loader
            labels = self.train_labels
        
        # Collect all embeddings and predictions
        all_embeddings = []
        all_indices = []
        
        for embeddings, _, indices in loader:
            all_embeddings.append(embeddings)
            all_indices.append(indices)
        
        all_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        all_indices = torch.cat(all_indices, dim=0)
        
        # Get cluster assignments
        predictions = self.model.get_cluster_assignments(all_embeddings, use_teacher=True)
        predictions = predictions.cpu()
        
        # Compute metrics
        metrics = compute_all_metrics(labels, predictions)
        
        # Compute cluster statistics
        stats = compute_cluster_statistics(predictions, self.config.NUM_CLUSTERS)
        metrics.update(stats)
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar(f'{split}/accuracy', metrics['accuracy'], epoch)
            self.writer.add_scalar(f'{split}/nmi', metrics['nmi'], epoch)
            self.writer.add_scalar(f'{split}/ari', metrics['ari'], epoch)
            self.writer.add_scalar(f'{split}/occupancy_rate', metrics['occupancy_rate'], epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config.__dict__,
        }
        
        # Save latest checkpoint
        latest_path = self.config.get_checkpoint_path()
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        if self.config.KEEP_ALL_CHECKPOINTS or epoch % self.config.SAVE_CHECKPOINT_FREQ == 0:
            epoch_path = self.config.get_checkpoint_path(epoch)
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.config.get_best_checkpoint_path()
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy: {self.best_accuracy:.2f}%")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        
        print(f"Resumed from epoch {self.current_epoch}, best accuracy: {self.best_accuracy:.2f}%")
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            Dictionary of final results
        """
        from utils.eval_utils import print_metrics, print_cluster_statistics
        
        print("\nStarting training...")
        print(f"Training on device: {self.device}")
        
        # Resume from checkpoint if specified
        if self.config.RESUME_FROM_CHECKPOINT is not None:
            checkpoint_path = Path(self.config.RESUME_FROM_CHECKPOINT)
            if checkpoint_path.exists():
                self.load_checkpoint(checkpoint_path)
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}, starting from scratch")
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_stats = self.train_epoch(epoch)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Evaluate
            if (epoch + 1) % self.config.EVAL_FREQ == 0 or epoch == self.config.NUM_EPOCHS - 1:
                print(f"\nEvaluating at epoch {epoch+1}...")
                
                # Evaluate on test set
                test_metrics = self.evaluate(epoch, split='test')
                print_metrics(test_metrics, prefix="Test ")
                print_cluster_statistics(test_metrics, prefix="Test ")
                
                # Check if best model
                is_best = test_metrics['accuracy'] > self.best_accuracy
                if is_best:
                    self.best_accuracy = test_metrics['accuracy']
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Store statistics
                self.training_stats.append({
                    'epoch': epoch + 1,
                    'train_loss': train_stats['loss'],
                    'test_accuracy': test_metrics['accuracy'],
                    'test_nmi': test_metrics['nmi'],
                    'test_ari': test_metrics['ari'],
                    'occupancy_rate': test_metrics['occupancy_rate'],
                })
            else:
                # Save checkpoint periodically
                if (epoch + 1) % self.config.SAVE_CHECKPOINT_FREQ == 0:
                    self.save_checkpoint(epoch, is_best=False)
        
        # Final evaluation
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        
        print("\nFinal evaluation on test set:")
        final_metrics = self.evaluate(self.config.NUM_EPOCHS - 1, split='test')
        print_metrics(final_metrics, prefix="Final Test ")
        print_cluster_statistics(final_metrics, prefix="Final Test ")
        
        # Save final results
        if self.config.SAVE_RESULTS_JSON:
            results = {
                'best_accuracy': self.best_accuracy,
                'final_metrics': {k: v for k, v in final_metrics.items() if k != 'reassignment'},
                'training_stats': self.training_stats,
                'config': {
                    'num_clusters': self.config.NUM_CLUSTERS,
                    'num_heads': self.config.NUM_HEADS,
                    'num_epochs': self.config.NUM_EPOCHS,
                    'learning_rate': self.config.LEARNING_RATE,
                    'beta': self.config.BETA,
                }
            }
            
            # Convert all numpy types to JSON-serializable Python types
            results = convert_to_json_serializable(results)
            
            results_path = self.config.RESULTS_DIR / 'final_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"\nResults saved to {results_path}")
        
        if self.writer is not None:
            self.writer.close()
        
        return final_metrics
