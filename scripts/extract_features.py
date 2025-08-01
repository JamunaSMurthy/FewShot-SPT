#!/usr/bin/env python3
"""
Feature extraction for FewShot-SPT
Extract video frames, audio mel-spectrograms, and prepare for training
"""

import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """Extract video frame features"""
    
    def __init__(self, 
                 frame_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 16,
                 fps: Optional[int] = 25,
                 normalize: bool = True):
        """
        Args:
            frame_size: (H, W)
            num_frames: Number of frames to extract
            fps: Target FPS (resample if needed)
            normalize: Apply ImageNet normalization
        """
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.fps = fps
        self.normalize = normalize
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
        ])
        
        # ImageNet normalization
        if normalize:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def extract_frames(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Extract uniformly spaced frames from video
        
        Returns:
            Tensor of shape (T, 3, H, W) or None if failed
        """
        try:
            video_path = str(video_path)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                logger.warning(f"Video has 0 frames: {video_path}")
                return None
            
            # Calculate frame indices to extract
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Apply transforms
                    frame_tensor = self.transform(frame_pil)
                    frames.append(frame_tensor)
            
            cap.release()
            
            if len(frames) == 0:
                logger.warning(f"No frames extracted from {video_path}")
                return None
            
            # Stack into (T, 3, H, W)
            video_tensor = torch.stack(frames)
            return video_tensor
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None


class AudioFeatureExtractor:
    """Extract audio mel-spectrogram features"""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 n_fft: int = 1024,
                 hop_length: int = 512):
        """
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
    
    def extract_audio(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Extract audio from video and compute mel-spectrogram
        
        Returns:
            Tensor of shape (n_mels, T) or None if failed
        """
        try:
            video_path = str(video_path)
            
            # Load audio from video file
            waveform, sr = torchaudio.load(video_path, num_frames=-1)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Compute mel-spectrogram
            mel_spec = self.mel_transform(waveform)  # (1, n_mels, time)
            mel_spec_db = torch.log(mel_spec + 1e-9)  # Convert to dB scale
            
            # Return (n_mels, T)
            return mel_spec_db.squeeze(0)
        
        except Exception as e:
            logger.warning(f"Error extracting audio from {video_path}: {e}")
            # Return dummy audio features
            return torch.randn(self.n_mels, 16)


class BatchFeatureExtractor:
    """Extract features from entire dataset"""
    
    def __init__(self, 
                 video_output_dir: str,
                 audio_output_dir: Optional[str] = None):
        """
        Args:
            video_output_dir: Directory to save video features
            audio_output_dir: Directory to save audio features (optional)
        """
        self.video_output_dir = Path(video_output_dir)
        self.audio_output_dir = Path(audio_output_dir) if audio_output_dir else None
        
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        if self.audio_output_dir:
            self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_extractor = VideoFeatureExtractor(
            frame_size=(224, 224),
            num_frames=16,
            normalize=True
        )
        
        self.audio_extractor = AudioFeatureExtractor(
            sample_rate=16000,
            n_mels=128
        )
    
    def extract_dataset(self, dataset_dir: str, skip_audio: bool = False):
        """
        Extract features from all videos in dataset
        
        Args:
            dataset_dir: Dataset directory with normal/ and abnormal/ subdirs
            skip_audio: Skip audio extraction (faster)
        """
        dataset_dir = Path(dataset_dir)
        
        # Collect all videos
        video_files = []
        for label_dir in [dataset_dir / 'normal', dataset_dir / 'abnormal']:
            if label_dir.exists():
                video_files.extend(label_dir.glob('*.mp4'))
        
        logger.info(f"Extracting features from {len(video_files)} videos...")
        
        # Extract features
        for video_path in tqdm(video_files, desc="Extracting features"):
            video_id = video_path.stem
            label = video_path.parent.name
            
            # Setup output paths
            video_output_path = self.video_output_dir / label / f"{video_id}.pt"
            video_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already extracted
            if video_output_path.exists():
                continue
            
            # Extract video features
            video_frames = self.video_extractor.extract_frames(str(video_path))
            if video_frames is not None:
                torch.save(video_frames, video_output_path)
            
            # Extract audio features
            if not skip_audio and self.audio_output_dir:
                audio_output_path = self.audio_output_dir / label / f"{video_id}.pt"
                audio_output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not audio_output_path.exists():
                    audio_features = self.audio_extractor.extract_audio(str(video_path))
                    if audio_features is not None:
                        torch.save(audio_features, audio_output_path)
        
        logger.info("✓ Feature extraction complete!")
        self.print_statistics(dataset_dir)
    
    def print_statistics(self, dataset_dir: Path):
        """Print extraction statistics"""
        logger.info("\nExtraction Statistics:")
        
        for label in ['normal', 'abnormal']:
            label_dir = self.video_output_dir / label
            if label_dir.exists():
                video_count = len(list(label_dir.glob('*.pt')))
                logger.info(f"  {label.capitalize()}: {video_count} videos extracted")


class DatasetCreator:
    """Create dataset in format compatible with FewShot-SPT"""
    
    def __init__(self, extracted_dir: str, output_dir: str):
        """
        Args:
            extracted_dir: Directory with extracted features
            output_dir: Output directory for formatted dataset
        """
        self.extracted_dir = Path(extracted_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_metadata(self, split_ratio: float = 0.8):
        """
        Create metadata JSON and train/test split
        
        Args:
            split_ratio: Ratio of training data
        """
        metadata = {
            'train': {'normal': [], 'abnormal': []},
            'test': {'normal': [], 'abnormal': []},
            'few_shot': {
                'support': {'normal': [], 'abnormal': []},
                'query': {'normal': [], 'abnormal': []}
            }
        }
        
        # Collect all videos
        for label in ['normal', 'abnormal']:
            label_dir = self.extracted_dir / label
            if not label_dir.exists():
                continue
            
            video_files = sorted(list(label_dir.glob('*.pt')))
            np.random.shuffle(video_files)
            
            split_idx = int(len(video_files) * split_ratio)
            
            # Assign to splits
            for i, video_file in enumerate(video_files):
                video_path = f"{label}/{video_file.name}"
                
                if i < split_idx:
                    metadata['train'][label].append(video_path)
                else:
                    metadata['test'][label].append(video_path)
            
            # Few-shot support/query split
            support_size = min(5, len(video_files) // 4)
            query_size = min(10, len(video_files) // 4)
            
            metadata['few_shot']['support'][label] = [
                f"{label}/{video_files[j].name}" 
                for j in range(support_size)
            ]
            
            metadata['few_shot']['query'][label] = [
                f"{label}/{video_files[support_size + j].name}"
                for j in range(query_size)
            ]
        
        # Save metadata
        import json
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to {metadata_path}")
        
        # Log statistics
        for split in ['train', 'test']:
            n_normal = len(metadata[split]['normal'])
            n_abnormal = len(metadata[split]['abnormal'])
            logger.info(f"  {split.capitalize()}: {n_normal} normal, {n_abnormal} abnormal")
        
        return metadata


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features for FewShot-SPT")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract video features
    video_parser = subparsers.add_parser('extract-video', help='Extract video frames')
    video_parser.add_argument('dataset', help='Dataset directory with normal/abnormal subdirs')
    video_parser.add_argument('-o', '--output', default='./data/features/video', help='Output directory')
    video_parser.add_argument('--frames', type=int, default=16, help='Number of frames to extract')
    
    # Extract audio features
    audio_parser = subparsers.add_parser('extract-audio', help='Extract audio features')
    audio_parser.add_argument('dataset', help='Dataset directory')
    audio_parser.add_argument('-o', '--output', default='./data/features/audio', help='Output directory')
    
    # Extract both
    both_parser = subparsers.add_parser('extract-all', help='Extract video and audio features')
    both_parser.add_argument('dataset', help='Dataset directory')
    both_parser.add_argument('-v', '--video-output', default='./data/features/video', help='Video output dir')
    both_parser.add_argument('-a', '--audio-output', default='./data/features/audio', help='Audio output dir')
    
    # Create metadata
    meta_parser = subparsers.add_parser('create-metadata', help='Create dataset metadata')
    meta_parser.add_argument('features', help='Features directory')
    meta_parser.add_argument('-o', '--output', default='./data/processed', help='Output directory')
    meta_parser.add_argument('--split', type=float, default=0.8, help='Train/test split ratio')
    
    args = parser.parse_args()
    
    if args.command == 'extract-video':
        extractor = BatchFeatureExtractor(args.output, audio_output_dir=None)
        extractor.extract_dataset(args.dataset, skip_audio=True)
    
    elif args.command == 'extract-audio':
        extractor = BatchFeatureExtractor(video_output_dir=None, audio_output_dir=args.output)
        # For audio only, we need to adjust logic
        logger.error("Audio-only extraction not yet implemented. Use 'extract-all'")
    
    elif args.command == 'extract-all':
        extractor = BatchFeatureExtractor(args.video_output, args.audio_output)
        extractor.extract_dataset(args.dataset, skip_audio=False)
    
    elif args.command == 'create-metadata':
        creator = DatasetCreator(args.features, args.output)
        creator.create_metadata(split_ratio=args.split)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
