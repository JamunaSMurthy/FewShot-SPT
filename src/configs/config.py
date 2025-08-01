"""
Configuration for FewShot-SPT training and inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Input dimensions
    video_input_shape: tuple = (3, 224, 224)
    audio_input_dim: int = 128
    text_input_dim: int = 768
    
    # Feature dimensions
    feature_dim: int = 512
    hidden_dim: int = 256
    
    # Number of classes
    num_classes: int = 2
    
    # EGKE parameters
    keyframe_ratio: float = 0.3
    egke_window_size: int = 5
    egke_memory_size: int = 10
    
    # AMG parameters
    num_heads: int = 8
    amg_dropout: float = 0.1
    
    # Perceiver IO parameters
    num_latents: int = 64
    num_perceiver_blocks: int = 4
    
    # APFSL parameters
    num_refinement_steps: int = 3
    prototype_temperature: float = 0.05
    prototype_decay: float = 0.99
    
    # Modalities
    modalities: List[str] = field(default_factory=lambda: ['video', 'audio', 'text'])


@dataclass
class TrainingConfig:
    """Training hyperparameter configuration."""
    
    # Optimization
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Learning rate scheduler
    lr_scheduler: str = 'cosine'  # 'linear', 'exponential', 'cosine', 'step'
    warmup_epochs: int = 5
    
    # Loss weights
    lambda_classification: float = 1.0
    lambda_contrastive: float = 0.5
    lambda_prototype: float = 0.3
    lambda_anomaly: float = 0.2
    
    # Regularization
    dropout: float = 0.2
    max_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Device
    device: str = 'cuda'  # 'cuda', 'cpu', 'mps'
    
    # Checkpointing
    checkpoint_every: int = 5
    save_best_only: bool = True


@dataclass
class FewShotConfig:
    """Few-shot learning configuration."""
    
    # Episode parameters
    n_way: int = 2
    n_shot: int = 2
    n_query: int = 5
    n_episodes: int = 500
    
    # Whether to use adaptive refinement
    use_adaptive_refinement: bool = True
    
    # Support set size
    max_support_samples: int = 100


@dataclass
class DataConfig:
    """Data loading configuration."""
    
    # Paths
    dataset_path: str = './data/video_anomaly'
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Data processing
    sequence_length: int = 16
    frame_rate: int = 30  # frames per second
    image_size: int = 224
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_strength: float = 0.5


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        'auc', 'ap', 'accuracy', 'f1', 'confusion_matrix'
    ])
    
    # Threshold for classification
    decision_threshold: float = 0.5
    
    # Visualization
    visualize_results: bool = True
    save_visualizations: bool = True
    
    # Frame-level vs video-level evaluation
    evaluation_level: str = 'frame'  # 'frame' or 'video'


@dataclass
class Config:
    """Master configuration combining all components."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment info
    experiment_name: str = 'FewShot-SPT'
    seed: int = 42
    verbose: bool = True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'few_shot': self.few_shot.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'verbose': self.verbose
        }
    
    def save(self, path: str):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        
        # Create config objects
        config = cls()
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'few_shot' in config_dict:
            config.few_shot = FewShotConfig(**config_dict['few_shot'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        if 'experiment_name' in config_dict:
            config.experiment_name = config_dict['experiment_name']
        if 'seed' in config_dict:
            config.seed = config_dict['seed']
        if 'verbose' in config_dict:
            config.verbose = config_dict['verbose']
        
        return config
    
    def update(self, **kwargs):
        """Update config values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)


# Default configurations for different scenarios

def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_few_shot_config() -> Config:
    """Get configuration optimized for few-shot learning."""
    config = Config()
    config.training.batch_size = 8
    config.training.num_epochs = 50
    config.few_shot.n_way = 2
    config.few_shot.n_shot = 2
    config.few_shot.n_query = 5
    return config


def get_production_config() -> Config:
    """Get configuration optimized for inference/deployment."""
    config = Config()
    config.model.num_latents = 32  # Reduce for speed
    config.model.num_perceiver_blocks = 2
    config.training.use_mixed_precision = True
    config.data.num_workers = 2
    return config


def get_debug_config() -> Config:
    """Get configuration for debugging."""
    config = Config()
    config.training.num_epochs = 2
    config.training.batch_size = 2
    config.data.sequence_length = 4
    config.training.early_stopping_patience = 1
    config.verbose = True
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Default Config:")
    print(config.to_dict())
    
    # Save and load
    config_path = "./test_config.json"
    config.save(config_path)
    loaded_config = Config.load(config_path)
    print("\nConfig loaded and verified!")
    
    # Cleanup
    Path(config_path).unlink()
