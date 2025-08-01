#!/usr/bin/env python3
"""
Dataset validation script for FewShot-SPT
Verify dataset integrity, structure, and statistics
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate FewShot-SPT datasets"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.results = {}
    
    def validate_structure(self) -> bool:
        """Check if dataset has correct directory structure"""
        logger.info("Validating dataset structure...")
        
        required_dirs = ['normal', 'abnormal']
        
        for label_dir in required_dirs:
            path = self.dataset_dir / label_dir
            if not path.exists():
                logger.error(f"Missing directory: {path}")
                return False
            logger.info(f"  ✓ {label_dir}/ exists")
        
        return True
    
    def validate_videos(self) -> Dict[str, int]:
        """Check all video files are readable"""
        logger.info("Validating video files...")
        
        results = {'valid': 0, 'corrupted': 0, 'total': 0}
        
        for label_dir in ['normal', 'abnormal']:
            label_path = self.dataset_dir / label_dir
            video_files = list(label_path.glob('*.mp4'))
            
            for video_file in tqdm(video_files, desc=f"Checking {label_dir}"):
                results['total'] += 1
                
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    
                    if not cap.isOpened():
                        logger.warning(f"Cannot open: {video_file.name}")
                        results['corrupted'] += 1
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    if frame_count < 10:
                        logger.warning(f"Too few frames ({frame_count}): {video_file.name}")
                        results['corrupted'] += 1
                        cap.release()
                        continue
                    
                    results['valid'] += 1
                    cap.release()
                
                except Exception as e:
                    logger.warning(f"Error reading {video_file.name}: {e}")
                    results['corrupted'] += 1
        
        return results
    
    def validate_features(self, feature_dir: str) -> Dict[str, int]:
        """Check extracted feature files"""
        logger.info("Validating feature files...")
        
        feature_path = Path(feature_dir)
        results = {'valid': 0, 'corrupted': 0, 'total': 0}
        
        if not feature_path.exists():
            logger.warning(f"Feature directory not found: {feature_dir}")
            return results
        
        for label_dir in ['normal', 'abnormal']:
            label_path = feature_path / label_dir
            if not label_path.exists():
                continue
            
            feature_files = list(label_path.glob('*.pt'))
            
            for feature_file in tqdm(feature_files, desc=f"Checking {label_dir} features"):
                results['total'] += 1
                
                try:
                    tensor = torch.load(feature_file)
                    
                    # Check shape
                    if tensor.ndim not in [2, 3, 4]:
                        logger.warning(f"Invalid shape: {feature_file.name} - {tensor.shape}")
                        results['corrupted'] += 1
                        continue
                    
                    # Check values
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        logger.warning(f"Invalid values in: {feature_file.name}")
                        results['corrupted'] += 1
                        continue
                    
                    results['valid'] += 1
                
                except Exception as e:
                    logger.warning(f"Error loading {feature_file.name}: {e}")
                    results['corrupted'] += 1
        
        return results
    
    def validate_metadata(self, metadata_file: str) -> bool:
        """Validate metadata JSON structure"""
        logger.info("Validating metadata...")
        
        metadata_path = Path(metadata_file)
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return False
        
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Check required keys
            required_keys = ['train', 'test', 'few_shot']
            for key in required_keys:
                if key not in metadata:
                    logger.error(f"Missing key in metadata: {key}")
                    return False
            
            # Validate splits
            for split in ['train', 'test']:
                if 'normal' not in metadata[split] or 'abnormal' not in metadata[split]:
                    logger.error(f"Invalid {split} split structure")
                    return False
            
            logger.info("  ✓ Metadata structure valid")
            return True
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        logger.info("Computing statistics...")
        
        stats = {
            'normal': {'count': 0, 'avg_frames': 0, 'avg_fps': 0},
            'abnormal': {'count': 0, 'avg_frames': 0, 'avg_fps': 0},
        }
        
        for label in ['normal', 'abnormal']:
            label_path = self.dataset_dir / label
            video_files = list(label_path.glob('*.mp4'))
            
            if not video_files:
                continue
            
            stats[label]['count'] = len(video_files)
            
            frame_counts = []
            fps_list = []
            
            for video_file in tqdm(video_files, desc=f"Analyzing {label}", leave=False):
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        frame_counts.append(frame_count)
                        fps_list.append(fps)
                        cap.release()
                except:
                    pass
            
            if frame_counts:
                stats[label]['avg_frames'] = int(np.mean(frame_counts))
                stats[label]['min_frames'] = int(np.min(frame_counts))
                stats[label]['max_frames'] = int(np.max(frame_counts))
            
            if fps_list:
                stats[label]['avg_fps'] = float(np.mean(fps_list))
        
        return stats
    
    def print_report(self, video_results: Dict = None, feature_results: Dict = None, stats: Dict = None):
        """Print validation report"""
        
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        
        if video_results:
            print("\n📹 Video Files:")
            print(f"  Total: {video_results['total']}")
            print(f"  Valid: {video_results['valid']} ✓")
            print(f"  Corrupted: {video_results['corrupted']} ✗")
            
            if video_results['total'] > 0:
                valid_pct = 100 * video_results['valid'] / video_results['total']
                print(f"  Health: {valid_pct:.1f}%")
        
        if feature_results:
            print("\n🎬 Feature Files:")
            print(f"  Total: {feature_results['total']}")
            print(f"  Valid: {feature_results['valid']} ✓")
            print(f"  Corrupted: {feature_results['corrupted']} ✗")
            
            if feature_results['total'] > 0:
                valid_pct = 100 * feature_results['valid'] / feature_results['total']
                print(f"  Health: {valid_pct:.1f}%")
        
        if stats:
            print("\n📊 Dataset Statistics:")
            for label, label_stats in stats.items():
                print(f"\n  {label.upper()}:")
                print(f"    Videos: {label_stats['count']}")
                if label_stats['count'] > 0:
                    print(f"    Avg Frames: {label_stats['avg_frames']} (min: {label_stats.get('min_frames', 0)}, max: {label_stats.get('max_frames', 0)})")
                    print(f"    Avg FPS: {label_stats['avg_fps']:.2f}")
        
        print("\n" + "="*60 + "\n")
    
    def validate_all(self, feature_dir: str = None, metadata_file: str = None) -> bool:
        """Run all validations"""
        
        # Validate structure
        if not self.validate_structure():
            logger.error("❌ Structure validation failed")
            return False
        
        # Validate videos
        video_results = self.validate_videos()
        
        # Validate features if provided
        feature_results = None
        if feature_dir:
            feature_results = self.validate_features(feature_dir)
        
        # Validate metadata if provided
        if metadata_file:
            self.validate_metadata(metadata_file)
        
        # Get statistics
        stats = self.get_statistics()
        
        # Print report
        self.print_report(video_results, feature_results, stats)
        
        # Overall health check
        if video_results['valid'] == video_results['total']:
            logger.info("✓ Dataset validation PASSED")
            return True
        else:
            logger.warning(f"⚠ {video_results['corrupted']} corrupted videos found")
            return True  # Still return True but with warning


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate FewShot-SPT datasets")
    parser.add_argument('dataset', help='Path to dataset directory')
    parser.add_argument('-f', '--features', help='Path to features directory')
    parser.add_argument('-m', '--metadata', help='Path to metadata JSON file')
    parser.add_argument('--fix-corrupted', action='store_true', help='Remove corrupted files')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset)
    validator.validate_all(
        feature_dir=args.features,
        metadata_file=args.metadata
    )


if __name__ == "__main__":
    main()
