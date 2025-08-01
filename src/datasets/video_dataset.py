"""
Data loaders for video anomaly detection datasets.
Supports UCF-Crime, XD-Violence, ShanghaiTech datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
from pathlib import Path


class VideoAnomalyDataset(Dataset):
    """
    Base dataset class for video anomaly detection.
    
    Expects a structure like:
    dataset_path/
      ├── normal/
      │   ├── video1.mp4
      │   ├── video2.mp4
      │   └── ...
      ├── abnormal/
      │   ├── anomaly1.mp4
      │   ├── anomaly2.mp4
      │   └── ...
      ├── annotations.json (optional)
      └── metadata.json (optional)
    """
    
    def __init__(self,
                 dataset_path: str,
                 split: str = 'train',
                 sequence_length: int = 16,
                 modalities: List[str] = ['video', 'audio', 'text'],
                 transform=None):
        """
        Args:
            dataset_path: Path to dataset root directory
            split: 'train', 'val', or 'test'
            sequence_length: Number of frames per sequence
            modalities: Which modalities to load
            transform: Data transformation function
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sequence_length = sequence_length
        self.modalities = modalities
        self.transform = transform
        
        self.videos = []
        self.labels = []
        self.metadata = {}
        
        # Load dataset structure
        self._load_dataset()
    
    def _load_dataset(self):
        """Load video paths and labels."""
        # Normal videos (label 0)
        normal_dir = self.dataset_path / 'normal'
        if normal_dir.exists():
            for video_file in normal_dir.glob('*.mp4'):
                self.videos.append(str(video_file))
                self.labels.append(0)
        
        # Abnormal videos (label 1)
        abnormal_dir = self.dataset_path / 'abnormal'
        if abnormal_dir.exists():
            for video_file in abnormal_dir.glob('*.mp4'):
                self.videos.append(str(video_file))
                self.labels.append(1)
        
        # Load annotations if available
        annotations_file = self.dataset_path / 'annotations.json'
        if annotations_file.exists():
            with open(annotations_file) as f:
                self.metadata = json.load(f)
    
    def __len__(self) -> int:
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a sample containing video and optionally other modalities.
        
        Returns:
            dict with keys:
                - 'video': (T, C, H, W) video frames
                - 'label': class label
                - 'video_id': video identifier
                - (optional) 'audio': (T, audio_dim) audio features
                - (optional) 'text': (T, text_dim) text embeddings
        """
        
        video_path = self.videos[idx]
        label = self.labels[idx]
        video_id = Path(video_path).stem
        
        sample = {
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id,
        }
        
        # Load video frames
        if 'video' in self.modalities:
            video_frames = self._load_video(video_path)
            if video_frames is not None:
                sample['video'] = video_frames
        
        # Load audio features
        if 'audio' in self.modalities:
            audio_features = self._load_audio(video_path)
            if audio_features is not None:
                sample['audio'] = audio_features
        
        # Load text features
        if 'text' in self.modalities:
            text_features = self._load_text(video_path)
            if text_features is not None:
                sample['text'] = text_features
        
        return sample
    
    def _load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess video frames.
        Returns (T, 3, H, W) tensor.
        """
        # In real implementation, use cv2 or decord
        try:
            # Dummy implementation - replace with actual video loading
            frames = torch.randn(self.sequence_length, 3, 224, 224)
            
            if self.transform:
                frames = self.transform(frames)
            
            return frames.float()
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def _load_audio(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess audio features (mel-spectrogram).
        Returns (T, audio_dim) tensor.
        """
        try:
            # Dummy implementation - replace with actual audio loading
            audio = torch.randn(self.sequence_length, 128)
            return audio.float()
        except Exception as e:
            print(f"Error loading audio from {video_path}: {e}")
            return None
    
    def _load_text(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess text features (embeddings from captions/descriptions).
        Returns (T, text_dim) tensor.
        """
        try:
            # Dummy implementation - replace with actual text loading
            text = torch.randn(self.sequence_length, 768)  # BERT-like embeddings
            return text.float()
        except Exception as e:
            print(f"Error loading text description for {video_path}: {e}")
            return None


class FewShotVideoDataset(Dataset):
    """Few-shot learning dataset for episode sampling."""
    
    def __init__(self,
                 dataset_path: str,
                 n_way: int = 2,
                 n_shot: int = 2,
                 n_query: int = 5,
                 n_episodes: int = 100,
                 modalities: List[str] = ['video']):
        """
        Args:
            dataset_path: Path to dataset
            n_way: Number of classes in each episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            n_episodes: Number of episodes to generate
            modalities: Which modalities to include
        """
        self.dataset_path = Path(dataset_path)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.modalities = modalities
        
        # Load base dataset
        self.base_dataset = VideoAnomalyDataset(
            str(dataset_path),
            modalities=modalities
        )
        
        # Organize by class
        self.class_indices = self._organize_by_class()
    
    def _organize_by_class(self) -> Dict[int, List[int]]:
        """Organize dataset indices by class."""
        class_indices = {}
        for idx, label in enumerate(self.base_dataset.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
    
    def __len__(self) -> int:
        return self.n_episodes
    
    def __getitem__(self, episode_idx: int) -> Dict:
        """
        Generate a few-shot episode.
        
        Returns:
            dict with:
                - 'support_features': (n_way * n_shot, D)
                - 'support_labels': (n_way * n_shot,)
                - 'query_features': (n_way * n_query, D)
                - 'query_labels': (n_way * n_query,)
        """
        
        # Sample n_way classes
        classes = np.random.choice(
            list(self.class_indices.keys()),
            self.n_way,
            replace=False
        )
        
        support_samples = []
        support_labels = []
        query_samples = []
        query_labels = []
        
        for way, class_label in enumerate(classes):
            # Get all samples for this class
            class_samples = self.class_indices[class_label]
            
            # Sample support and query indices
            total_needed = self.n_shot + self.n_query
            selected_idx = np.random.choice(
                class_samples,
                min(total_needed, len(class_samples)),
                replace=False
            )
            
            # Support samples
            for shot_idx in range(min(self.n_shot, len(selected_idx))):
                sample = self.base_dataset[selected_idx[shot_idx]]
                support_samples.append(sample)
                support_labels.append(way)
            
            # Query samples
            for query_idx in range(min(self.n_query, len(selected_idx) - self.n_shot)):
                sample = self.base_dataset[selected_idx[self.n_shot + query_idx]]
                query_samples.append(sample)
                query_labels.append(way)
        
        # Stack samples - extract features from videos
        support_features = self._extract_features(support_samples)
        query_features = self._extract_features(query_samples)
        
        return {
            'support_features': support_features,
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_features': query_features,
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
        }
    
    def _extract_features(self, samples: List[Dict]) -> torch.Tensor:
        """Extract features from samples."""
        features = []
        for sample in samples:
            if 'video' in sample:
                # Use video features
                feat = sample['video'].mean(dim=0)  # Average over time
            else:
                # Use dummy features
                feat = torch.randn(512)
            
            features.append(feat)
        
        return torch.stack(features)


class BalancedSampler(torch.utils.data.Sampler):
    """Sampler that ensures balanced positive/negative samples in each batch."""
    
    def __init__(self, labels: List[int], batch_size: int, drop_last: bool = False):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by label
        self.positive_indices = [i for i, label in enumerate(labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(labels) if label == 0]
    
    def __iter__(self):
        # Shuffle
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)
        
        batches = []
        num_pos_per_batch = self.batch_size // 2
        num_neg_per_batch = self.batch_size - num_pos_per_batch
        
        # Create balanced batches
        num_batches = min(
            len(self.positive_indices) // num_pos_per_batch,
            len(self.negative_indices) // num_neg_per_batch
        )
        
        for batch_idx in range(num_batches):
            start_pos = batch_idx * num_pos_per_batch
            start_neg = batch_idx * num_neg_per_batch
            
            batch = (
                self.positive_indices[start_pos:start_pos + num_pos_per_batch] +
                self.negative_indices[start_neg:start_neg + num_neg_per_batch]
            )
            
            batches.extend(batch)
        
        return iter(batches)
    
    def __len__(self):
        min_samples = min(len(self.positive_indices), len(self.negative_indices))
        return (min_samples * 2 // self.batch_size) * self.batch_size


def create_dataloaders(dataset_path: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      modalities: List[str] = ['video', 'audio', 'text']) -> Dict:
    """
    Create train/val/test dataloaders.
    
    Returns:
        dict with 'train', 'val', 'test' dataloaders
    """
    
    # Create datasets
    train_dataset = VideoAnomalyDataset(
        dataset_path,
        split='train',
        modalities=modalities
    )
    
    val_dataset = VideoAnomalyDataset(
        dataset_path,
        split='val',
        modalities=modalities
    )
    
    test_dataset = VideoAnomalyDataset(
        dataset_path,
        split='test',
        modalities=modalities
    )
    
    # Create balanced sampler for training
    train_sampler = BalancedSampler(
        train_dataset.labels,
        batch_size=batch_size,
        drop_last=True
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_multimodal_batch
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_multimodal_batch
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_multimodal_batch
        )
    }
    
    return dataloaders


def collate_multimodal_batch(batch: List[Dict]) -> Dict:
    """Custom collate function for multi-modal batches."""
    
    collated = {}
    
    for key in batch[0].keys():
        if key == 'label':
            collated['label'] = torch.stack([s[key] for s in batch if key in s])
        elif key == 'video_id':
            collated['video_id'] = [s[key] for s in batch if key in s]
        elif key in ['video', 'audio', 'text']:
            # Stack tensors
            tensors = [s[key] for s in batch if key in s]
            if tensors:
                collated[key] = torch.stack(tensors)
    
    return collated


# Factory functions
def create_few_shot_loader(dataset_path: str,
                           n_way: int = 2,
                           n_shot: int = 2,
                           n_query: int = 5,
                           n_episodes: int = 100,
                           batch_size: int = 4,
                           num_workers: int = 4) -> DataLoader:
    """Create few-shot learning dataloader."""
    
    dataset = FewShotVideoDataset(
        dataset_path=dataset_path,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_episodes=n_episodes
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return loader


if __name__ == "__main__":
    print("Dataset loader module loaded successfully")
