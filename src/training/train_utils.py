"""
Training utilities and helpers for FewShot-SPT.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional, Tuple
import csv
from pathlib import Path
from datetime import datetime
import json


class AverageMeter:
    """Tracks average metric values."""
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __repr__(self):
        return f'{self.name}: {self.avg:.4f}'


class MetricTracker:
    """Tracks multiple metrics across training."""
    
    def __init__(self, metric_names: list):
        self.metrics = {name: AverageMeter(name) for name in metric_names}
        self.metric_names = metric_names
    
    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].update(value)
    
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
    
    def get_averages(self) -> Dict[str, float]:
        return {name: metric.avg for name, metric in self.metrics.items()}
    
    def __repr__(self):
        return ' | '.join([str(self.metrics[name]) for name in self.metric_names])


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 best_model_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_model_path = best_model_path
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module = None, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
            if model is not None:
                self.best_model_state = model.state_dict()
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True
        
        return False
    
    def restore_best_model(self, model: nn.Module):
        """Restore model to best checkpoint."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"Restored model from epoch {self.best_epoch}")


class LearningRateScheduler:
    """Custom learning rate scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 strategy: str = 'linear',
                 initial_lr: float = 0.001,
                 total_epochs: int = 100,
                 warmup_epochs: int = 5):
        """
        Args:
            optimizer: PyTorch optimizer
            strategy: 'linear', 'exponential', 'cosine', 'step'
            initial_lr: Starting learning rate
            total_epochs: Total training epochs
            warmup_epochs: Number of warmup epochs
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self, epoch: int = None):
        """Update learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """Compute learning rate based on strategy."""
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_ratio = self.current_epoch / max(1, self.warmup_epochs)
            return self.initial_lr * warmup_ratio
        
        progress = self.current_epoch / self.total_epochs
        
        if self.strategy == 'linear':
            return self.initial_lr * (1 - progress)
        
        elif self.strategy == 'exponential':
            return self.initial_lr * (0.1 ** progress)
        
        elif self.strategy == 'cosine':
            import math
            return self.initial_lr * (1 + math.cos(math.pi * progress)) / 2
        
        elif self.strategy == 'step':
            # Decay by 0.1 every third of training
            num_decays = int(progress * 3)
            return self.initial_lr * (0.1 ** num_decays)
        
        return self.initial_lr


class TrainingLogger:
    """Logs training metrics to CSV and console."""
    
    def __init__(self, log_dir: str = 'logs', experiment_name: str = 'experiment'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = f"{experiment_name}_{timestamp}"
        
        self.csv_path = self.log_dir / f"{self.experiment_name}.csv"
        self.config_path = self.log_dir / f"{self.experiment_name}_config.json"
        
        self.fieldnames = None
        self.csv_file = None
        self.writer = None
    
    def log_config(self, config: Dict):
        """Save configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_epoch(self, epoch: int, metrics: Dict):
        """Log metrics for an epoch."""
        
        if self.fieldnames is None:
            # Initialize CSV on first write
            self.fieldnames = ['epoch'] + list(metrics.keys())
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        
        # Write epoch metrics
        row = {'epoch': epoch}
        row.update(metrics)
        self.writer.writerow(row)
        self.csv_file.flush()
    
    def close(self):
        """Close logger."""
        if self.csv_file is not None:
            self.csv_file.close()
    
    def __del__(self):
        self.close()


class FewShotBatchSampler:
    """Batch sampler for few-shot learning."""
    
    def __init__(self, dataset, n_way: int = 2, n_shot: int = 2, 
                 n_query: int = 5, batch_size: int = 1):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.batch_size = batch_size
    
    def __iter__(self):
        """Yield batches of episodes."""
        batch = []
        for _ in range(len(self.dataset)):
            batch.append(len(batch))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0:
            yield batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict,
                   checkpoint_path: str):
    """Save training checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   checkpoint_path: str = ''):
    """Load training checkpoint."""
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, {}
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    
    return epoch, metrics


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def print_model_summary(model: nn.Module):
    """Print model summary."""
    
    total, trainable = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test utilities
    print("Training utilities loaded successfully")
