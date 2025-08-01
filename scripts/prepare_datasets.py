#!/usr/bin/env python3
"""
Dataset Preparation Scripts for FewShot-SPT
Complete automated pipeline for downloading, processing, and organizing datasets
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Base dataset processor"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (self.output_dir / 'normal').mkdir(exist_ok=True)
        (self.output_dir / 'abnormal').mkdir(exist_ok=True)
    
    def organize_videos(self, source_dir: str, category_mapping: Dict[str, str]):
        """
        Organize videos into normal/abnormal by category
        
        Args:
            source_dir: Root directory containing category folders
            category_mapping: {'category_name': 'normal'/'abnormal'}
        """
        source = Path(source_dir)
        
        for category, label in category_mapping.items():
            category_dir = source / category
            if not category_dir.exists():
                logger.warning(f"Category directory not found: {category_dir}")
                continue
            
            target_dir = self.output_dir / label
            video_count = 0
            
            for video_file in category_dir.glob('*.mp4'):
                try:
                    shutil.copy2(video_file, target_dir / video_file.name)
                    video_count += 1
                except Exception as e:
                    logger.error(f"Failed to copy {video_file}: {e}")
            
            logger.info(f"  ✓ Copied {video_count} videos from {category} → {label}/")


class UCFCrimeProcessor(DatasetProcessor):
    """Process UCF-Crime dataset"""
    
    CRIME_CATEGORIES = {
        'Abuse': 'abnormal',
        'Arrest': 'abnormal',
        'Arson': 'abnormal',
        'Assault': 'abnormal',
        'Robbery': 'abnormal',
        'Shooting': 'abnormal',
        'Shoplifting': 'abnormal',
        'Stealing': 'abnormal',
        'Tagging': 'abnormal',
        'Trespassing': 'abnormal',
        'Vandalism': 'abnormal',
        'Burglary': 'abnormal',
        'Explosion': 'abnormal',
        'Fighting': 'abnormal',
        'RoadAccidents': 'abnormal',
        'Seizure': 'abnormal',
    }
    
    def process(self, source_dir: str):
        """Process UCF-Crime dataset"""
        logger.info("Processing UCF-Crime dataset...")
        
        source = Path(source_dir)
        
        # Copy crime videos
        for crime_type, label in self.CRIME_CATEGORIES.items():
            crime_dir = source / crime_type
            if crime_dir.exists():
                video_count = len(list(crime_dir.glob('*.mp4')))
                target_dir = self.output_dir / label
                
                for video in crime_dir.glob('*.mp4'):
                    try:
                        shutil.copy2(video, target_dir / video.name)
                    except Exception as e:
                        logger.error(f"Failed: {video.name} - {e}")
        
        # Copy normal videos
        for normal_dir in [source / 'Normal_Videos', source / 'Normal_Events_v2']:
            if normal_dir.exists():
                for video in normal_dir.glob('*.mp4'):
                    try:
                        shutil.copy2(video, self.output_dir / 'normal' / video.name)
                    except Exception as e:
                        logger.error(f"Failed: {video.name} - {e}")
        
        logger.info(f"✓ UCF-Crime processing complete")
        self.print_statistics()
    
    def print_statistics(self):
        normal_count = len(list((self.output_dir / 'normal').glob('*.mp4')))
        abnormal_count = len(list((self.output_dir / 'abnormal').glob('*.mp4')))
        
        logger.info(f"  Normal videos: {normal_count}")
        logger.info(f"  Abnormal videos: {abnormal_count}")
        logger.info(f"  Total: {normal_count + abnormal_count}")


class XDViolenceProcessor(DatasetProcessor):
    """Process XD-Violence dataset"""
    
    def process(self, source_dir: str):
        """Process XD-Violence dataset"""
        logger.info("Processing XD-Violence dataset...")
        
        source = Path(source_dir)
        mapping = {
            'Violence': 'abnormal',
            'violent': 'abnormal',
            'NonViolence': 'normal',
            'normal': 'normal',
        }
        
        for category, label in mapping.items():
            category_dir = source / category
            if category_dir.exists():
                video_count = 0
                target_dir = self.output_dir / label
                
                for video in category_dir.glob('*.mp4'):
                    try:
                        shutil.copy2(video, target_dir / video.name)
                        video_count += 1
                    except Exception as e:
                        logger.error(f"Failed: {video.name} - {e}")
                
                if video_count > 0:
                    logger.info(f"  ✓ Copied {video_count} videos from {category}")
        
        logger.info(f"✓ XD-Violence processing complete")
        self.print_statistics()
    
    def print_statistics(self):
        normal_count = len(list((self.output_dir / 'normal').glob('*.mp4')))
        abnormal_count = len(list((self.output_dir / 'abnormal').glob('*.mp4')))
        
        logger.info(f"  Normal videos: {normal_count}")
        logger.info(f"  Abnormal videos: {abnormal_count}")
        logger.info(f"  Total: {normal_count + abnormal_count}")


class ShanghaiTechProcessor:
    """Process ShanghaiTech dataset (frames to videos)"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        (self.output_dir / 'normal').mkdir(exist_ok=True)
        (self.output_dir / 'abnormal').mkdir(exist_ok=True)
    
    def frames_to_video(self, frame_dir: str, output_video: str, fps: int = 25) -> bool:
        """Convert directory of JPG frames to MP4 video"""
        frame_dir = Path(frame_dir)
        frames = sorted([f for f in frame_dir.glob('*.jpg') if f.is_file()])
        
        if not frames:
            logger.warning(f"No frames found in {frame_dir}")
            return False
        
        try:
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frames[0]))
            if first_frame is None:
                logger.error(f"Could not read first frame: {frames[0]}")
                return False
            
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Could not create video writer for {output_video}")
                return False
            
            # Write frames
            for frame_file in frames:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_file}")
                    continue
                
                out.write(frame)
            
            out.release()
            return True
        
        except Exception as e:
            logger.error(f"Error converting frames to video: {e}")
            return False
    
    def process(self, source_dir: str):
        """Process ShanghaiTech dataset"""
        logger.info("Processing ShanghaiTech dataset...")
        
        source = Path(source_dir)
        
        # Process training and testing splits
        for split in ['training', 'testing']:
            split_dir = source / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            frames_dir = split_dir / 'frames'
            gt_dir = split_dir / 'gt'
            
            if not frames_dir.exists():
                logger.warning(f"Frames directory not found: {frames_dir}")
                continue
            
            logger.info(f"Processing {split} split...")
            
            # Get unique video IDs
            video_dirs = set()
            for item in frames_dir.iterdir():
                if item.is_dir():
                    video_dirs.add(item.name)
            
            for video_id in sorted(video_dirs):
                video_frame_dir = frames_dir / video_id
                
                # Determine if normal or abnormal
                is_abnormal = self._is_abnormal_video(video_id, gt_dir)
                label = 'abnormal' if is_abnormal else 'normal'
                
                output_video = self.output_dir / label / f"{video_id}.mp4"
                
                if output_video.exists():
                    continue
                
                success = self.frames_to_video(str(video_frame_dir), str(output_video))
                
                if success:
                    logger.info(f"  ✓ {video_id} → {label}/")
                else:
                    logger.error(f"  ✗ Failed: {video_id}")
        
        logger.info(f"✓ ShanghaiTech processing complete")
        self.print_statistics()
    
    def _is_abnormal_video(self, video_id: str, gt_dir: Path) -> bool:
        """Check if video contains anomalies"""
        gt_file = gt_dir / f"{video_id}_gt.npy"
        
        if gt_file.exists():
            try:
                gt = np.load(gt_file)
                return np.any(gt == 1)
            except Exception as e:
                logger.warning(f"Error reading ground truth for {video_id}: {e}")
        
        return False
    
    def print_statistics(self):
        normal_count = len(list((self.output_dir / 'normal').glob('*.mp4')))
        abnormal_count = len(list((self.output_dir / 'abnormal').glob('*.mp4')))
        
        logger.info(f"  Normal videos: {normal_count}")
        logger.info(f"  Abnormal videos: {abnormal_count}")
        logger.info(f"  Total: {normal_count + abnormal_count}")


class VideoFrameExtractor:
    """Extract frames from videos"""
    
    def __init__(self, output_dir: str, frame_size: Tuple[int, int] = (224, 224), 
                 fps: Optional[int] = None, num_frames: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.frame_size = frame_size
        self.fps = fps
        self.num_frames = num_frames
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, video_path: str, video_id: str) -> int:
        """
        Extract frames from video
        
        Returns:
            Number of frames extracted
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return 0
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame skip rate
            frame_skip = 1
            if self.fps and original_fps:
                frame_skip = max(1, int(original_fps / self.fps))
            
            # Extract frames
            frame_idx = 0
            extracted = 0
            
            frame_dir = self.output_dir / video_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip == 0:
                    # Resize
                    frame_resized = cv2.resize(frame, self.frame_size)
                    
                    # Save
                    frame_path = frame_dir / f"frame_{extracted:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame_resized)
                    extracted += 1
                    
                    if self.num_frames and extracted >= self.num_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            return extracted
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return 0
    
    def batch_extract(self, dataset_dir: str, label_filter: Optional[str] = None):
        """Extract frames from all videos in dataset"""
        
        dataset_dir = Path(dataset_dir)
        
        # Find all videos
        if label_filter:
            video_files = list((dataset_dir / label_filter).glob('*.mp4'))
        else:
            video_files = list(dataset_dir.glob('**/*.mp4'))
        
        logger.info(f"Extracting frames from {len(video_files)} videos...")
        
        for i, video_path in enumerate(tqdm(video_files)):
            video_id = video_path.stem
            
            # Check if already extracted
            if (self.output_dir / video_id).exists():
                continue
            
            num_frames = self.extract_frames(str(video_path), video_id)
            
            if num_frames == 0:
                logger.warning(f"Failed to extract frames from {video_path.name}")


def create_dataset_structure(base_dir: str):
    """Create standard dataset directory structure"""
    
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    
    for dataset in ['UCF_Crime', 'XD_Violence', 'ShanghaiTech']:
        for split in ['train', 'test']:
            for label in ['normal', 'abnormal']:
                (base / dataset / split / label).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Created directory structure in {base}")


def create_train_test_split(dataset_dir: str, train_ratio: float = 0.8):
    """Create train/test split from processed videos"""
    
    dataset_dir = Path(dataset_dir)
    
    # Get all normal and abnormal videos
    normal_videos = list((dataset_dir / 'normal').glob('*.mp4'))
    abnormal_videos = list((dataset_dir / 'abnormal').glob('*.mp4'))
    
    np.random.seed(42)
    np.random.shuffle(normal_videos)
    np.random.shuffle(abnormal_videos)
    
    # Split indices
    n_normal_train = int(len(normal_videos) * train_ratio)
    n_abnormal_train = int(len(abnormal_videos) * train_ratio)
    
    # Create directories
    for label in ['normal', 'abnormal']:
        for split in ['train', 'test']:
            (dataset_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Copy training videos
    for video in normal_videos[:n_normal_train]:
        shutil.copy2(video, dataset_dir / 'train' / 'normal' / video.name)
    
    for video in abnormal_videos[:n_abnormal_train]:
        shutil.copy2(video, dataset_dir / 'train' / 'abnormal' / video.name)
    
    # Copy test videos
    for video in normal_videos[n_normal_train:]:
        shutil.copy2(video, dataset_dir / 'test' / 'normal' / video.name)
    
    for video in abnormal_videos[n_abnormal_train:]:
        shutil.copy2(video, dataset_dir / 'test' / 'abnormal' / video.name)
    
    # Log statistics
    logger.info(f"Train/Test Split ({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f}):")
    logger.info(f"  Train - Normal: {n_normal_train}, Abnormal: {n_abnormal_train}")
    logger.info(f"  Test  - Normal: {len(normal_videos)-n_normal_train}, Abnormal: {len(abnormal_videos)-n_abnormal_train}")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FewShot-SPT Dataset Preparation")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # UCF-Crime processing
    ucf_parser = subparsers.add_parser('process-ucf', help='Process UCF-Crime dataset')
    ucf_parser.add_argument('source', help='Path to UCF-Crime dataset')
    ucf_parser.add_argument('-o', '--output', default='./data/UCF_Crime', help='Output directory')
    
    # XD-Violence processing
    xd_parser = subparsers.add_parser('process-xd', help='Process XD-Violence dataset')
    xd_parser.add_argument('source', help='Path to XD-Violence dataset')
    xd_parser.add_argument('-o', '--output', default='./data/XD_Violence', help='Output directory')
    
    # ShanghaiTech processing
    st_parser = subparsers.add_parser('process-st', help='Process ShanghaiTech dataset')
    st_parser.add_argument('source', help='Path to ShanghaiTech dataset')
    st_parser.add_argument('-o', '--output', default='./data/ShanghaiTech', help='Output directory')
    
    # Frame extraction
    extract_parser = subparsers.add_parser('extract-frames', help='Extract frames from videos')
    extract_parser.add_argument('dataset', help='Path to processed dataset')
    extract_parser.add_argument('-o', '--output', default='./data/frames', help='Output directory')
    extract_parser.add_argument('--size', type=int, nargs=2, default=[224, 224], help='Frame size')
    extract_parser.add_argument('--fps', type=int, help='Target FPS')
    
    # Create directory structure
    struct_parser = subparsers.add_parser('create-structure', help='Create dataset directory structure')
    struct_parser.add_argument('base', help='Base directory')
    
    # Create splits
    split_parser = subparsers.add_parser('create-split', help='Create train/test split')
    split_parser.add_argument('dataset', help='Processed dataset directory')
    split_parser.add_argument('--ratio', type=float, default=0.8, help='Train ratio')
    
    args = parser.parse_args()
    
    if args.command == 'process-ucf':
        processor = UCFCrimeProcessor(args.output)
        processor.process(args.source)
    
    elif args.command == 'process-xd':
        processor = XDViolenceProcessor(args.output)
        processor.process(args.source)
    
    elif args.command == 'process-st':
        processor = ShanghaiTechProcessor(args.output)
        processor.process(args.source)
    
    elif args.command == 'extract-frames':
        extractor = VideoFrameExtractor(
            args.output,
            frame_size=tuple(args.size),
            fps=args.fps
        )
        extractor.batch_extract(args.dataset)
    
    elif args.command == 'create-structure':
        create_dataset_structure(args.base)
    
    elif args.command == 'create-split':
        create_train_test_split(args.dataset, args.ratio)
    
    else:
        parser.print_help()
