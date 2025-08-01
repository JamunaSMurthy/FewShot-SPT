"""
Main training script for FewShot-SPT.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import sys
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fewshot_spt import create_fewshot_spt
from datasets.video_dataset import create_dataloaders, create_few_shot_loader
from configs.config import Config, get_default_config, get_few_shot_config
from training.train_utils import (
    EarlyStopping, LearningRateScheduler, TrainingLogger,
    MetricTracker, count_parameters, print_model_summary,
    save_checkpoint, load_checkpoint, get_device
)
from utils.losses import CombinedAnomalyLoss, AnomalyLoss, PrototypicalLoss
from utils.metrics import AnomalyDetectionMetrics, MetricsVisualizer


class FewShotSPTTrainer:
    """Trainer class for FewShot-SPT."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = get_device()
        
        # Setup logging
        self._setup_logging()
        
        # Create model
        self.model = create_fewshot_spt(
            num_classes=config.model.num_classes,
            keyframe_ratio=config.model.keyframe_ratio
        ).to(self.device)
        
        # Print model info
        print_model_summary(self.model)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create loss function
        self.loss_fn = CombinedAnomalyLoss(
            lambda_focal=config.training.lambda_classification,
            lambda_contrastive=config.training.lambda_contrastive
        )
        
        # Learning rate scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            strategy=config.training.lr_scheduler,
            initial_lr=config.training.learning_rate,
            total_epochs=config.training.num_epochs,
            warmup_epochs=config.training.warmup_epochs
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta
        )
        
        # Training logger
        self.logger = TrainingLogger(
            log_dir=config.data.log_dir,
            experiment_name=config.experiment_name
        )
        self.logger.log_config(config.to_dict())
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_mixed_precision else None
        self.use_amp = config.training.use_mixed_precision
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.data.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger_obj = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        metric_tracker = MetricTracker([
            'loss', 'loss_focal', 'loss_contrastive',
            'accuracy', 'auc'
        ])
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self._forward_pass(batch, is_training=True)
                    loss_dict = outputs['loss']
                    loss = loss_dict['total']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_pass(batch, is_training=True)
                loss_dict = outputs['loss']
                loss = loss_dict['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                if 'labels' in batch:
                    labels = batch['labels'].cpu().numpy()
                    logits = outputs['logits'].cpu().numpy()
                    preds = np.argmax(logits, axis=1)
                    
                    accuracy = (preds == labels).mean()
                    metric_tracker.update(
                        loss=loss.item(),
                        loss_focal=loss_dict.get('focal', 0).item(),
                        loss_contrastive=loss_dict.get('contrastive', 0).item(),
                        accuracy=accuracy
                    )
            
            progress_bar.set_postfix({
                'loss': f'{metric_tracker.metrics["loss"].avg:.4f}',
                'acc': f'{metric_tracker.metrics["accuracy"].avg:.4f}'
            })
        
        return metric_tracker.get_averages()
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                batch = self._move_batch_to_device(batch)
                
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = self._forward_pass(batch, is_training=False)
                else:
                    outputs = self._forward_pass(batch, is_training=False)
                
                logits = outputs['logits'].cpu().numpy()
                all_logits.append(logits)
                
                if 'labels' in batch:
                    labels = batch['labels'].cpu().numpy()
                    all_labels.append(labels)
                    all_preds.append(np.argmax(logits, axis=1))
        
        # Compute metrics
        if all_labels:
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            all_logits = np.concatenate(all_logits)
            
            # Get anomaly scores (probability of anomaly class)
            anomaly_scores = np.softmax(all_logits, axis=1)[:, 1]
            
            metrics = AnomalyDetectionMetrics.compute_all_metrics(
                anomaly_scores, all_labels
            )
        else:
            metrics = {}
        
        return metrics
    
    def _forward_pass(self, batch: Dict, is_training: bool = True) -> Dict:
        """Forward pass through model."""
        
        # Extract modalities
        video = batch.get('video', None)
        audio = batch.get('audio', None)
        text = batch.get('text', None)
        labels = batch.get('labels', None)
        
        # Forward pass
        logits = self.model(
            video_frames=video,
            audio_features=audio,
            text_features=text
        )
        
        # Compute loss if labels available
        outputs = {'logits': logits}
        
        if labels is not None:
            loss_dict = self.loss_fn(
                logits=logits,
                targets=labels,
                features=None
            )
            outputs['loss'] = loss_dict
        
        return outputs
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.config.experiment_name}")
        print(f"{'='*60}\n")
        
        best_val_auc = 0
        
        for epoch in range(self.config.training.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            epoch_metrics = {
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()},
                'lr': current_lr
            }
            self.logger.log_epoch(epoch, epoch_metrics)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_metrics.get('loss', 0):.4f}")
            if val_metrics:
                print(f"  Val AUC: {val_metrics.get('auc', 0):.4f}")
                print(f"  Val AP: {val_metrics.get('ap', 0):.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.checkpoint_every == 0:
                checkpoint_path = (
                    self.checkpoint_dir /
                    f"checkpoint_epoch_{epoch+1:03d}.pt"
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    epoch_metrics, str(checkpoint_path)
                )
            
            # Early stopping
            val_loss = val_metrics.get('auc', 0)
            if self.early_stopping(val_loss, self.model, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                self.early_stopping.restore_best_model(self.model)
                break
            
            # Track best model
            if val_metrics.get('auc', 0) > best_val_auc:
                best_val_auc = val_metrics.get('auc', 0)
                best_checkpoint = self.checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    epoch_metrics, str(best_checkpoint)
                )
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Val AUC: {best_val_auc:.4f}")
        print(f"{'='*60}\n")


def main():
    """Main training function."""
    
    # Load configuration
    config = get_few_shot_config()
    
    # Set random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create trainer
    trainer = FewShotSPTTrainer(config)
    
    # Create datasets
    print(f"\nLoading datasets from {config.data.dataset_path}...\n")
    dataloaders = create_dataloaders(
        dataset_path=config.data.dataset_path,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        modalities=config.model.modalities
    )
    
    # Train
    trainer.train(dataloaders['train'], dataloaders['val'])


if __name__ == "__main__":
    main()
