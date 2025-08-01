#!/usr/bin/env python3
"""
Setup helper for FewShot-SPT dataset preparation
Provides interactive configuration and directory setup
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
import configparser


class SetupHelper:
    """Interactive setup helper for dataset preparation"""
    
    def __init__(self):
        self.config = {}
        self.base_dir = Path.cwd()
    
    def get_user_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default"""
        if default:
            prompt = f"{prompt} [{default}]: "
        else:
            prompt = f"{prompt}: "
        
        response = input(prompt).strip()
        return response or default or ""
    
    def setup_directories(self) -> Dict[str, Path]:
        """Create and setup dataset directories"""
        
        print("\n" + "="*60)
        print("FewShot-SPT Dataset Directory Setup")
        print("="*60 + "\n")
        
        # Get data directory
        data_dir = self.get_user_input(
            "Enter data directory path",
            default="./data"
        )
        data_dir = Path(data_dir).expanduser()
        
        # Create directory structure
        dirs = {
            'root': data_dir,
            'raw': data_dir / 'raw',
            'processed': data_dir / 'processed',
            'features': data_dir / 'features',
            'features_video': data_dir / 'features' / 'video',
            'features_audio': data_dir / 'features' / 'audio',
            'ucf_crime': data_dir / 'UCF_Crime',
            'xd_violence': data_dir / 'XD_Violence',
            'shanghaitech': data_dir / 'ShanghaiTech',
            'models': data_dir / 'models',
            'logs': data_dir / 'logs',
        }
        
        # Create all directories
        for dir_name, dir_path in dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {dir_path}")
        
        return dirs
    
    def setup_raw_dataset_paths(self) -> Dict[str, Path]:
        """Get paths to raw datasets"""
        
        print("\n" + "="*60)
        print("Raw Dataset Paths")
        print("="*60 + "\n")
        
        print("Enter paths to your downloaded datasets")
        print("(Press Enter to skip if not downloaded yet)\n")
        
        raw_paths = {
            'UCF_Crime': self.get_user_input(
                "UCF-Crime directory",
                default=str(Path.home() / "datasets" / "UCF_Crime")
            ),
            'XD_Violence': self.get_user_input(
                "XD-Violence directory",
                default=str(Path.home() / "datasets" / "XD_Violence")
            ),
            'ShanghaiTech': self.get_user_input(
                "ShanghaiTech directory",
                default=str(Path.home() / "datasets" / "ShanghaiTech")
            ),
        }
        
        # Verify paths exist
        for dataset_name, path_str in list(raw_paths.items()):
            if path_str:
                path = Path(path_str).expanduser()
                if path.exists():
                    print(f"✓ {dataset_name}: {path}")
                else:
                    print(f"⚠ {dataset_name}: Not found at {path}")
                    raw_paths[dataset_name] = None
            else:
                raw_paths[dataset_name] = None
        
        return raw_paths
    
    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        
        print("\n" + "="*60)
        print("Checking Dependencies")
        print("="*60 + "\n")
        
        dependencies = {
            'torch': 'PyTorch (ML framework)',
            'cv2': 'OpenCV (video processing)',
            'torchaudio': 'TorchAudio (audio processing)',
            'numpy': 'NumPy (numerical computing)',
            'librosa': 'Librosa (audio feature extraction)',
        }
        
        missing = []
        
        for module, description in dependencies.items():
            try:
                __import__(module)
                print(f"✓ {description}")
            except ImportError:
                print(f"✗ {description} (missing)")
                missing.append(module)
        
        if missing:
            print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
            print("\nInstall with:")
            print(f"  pip install -r scripts/requirements.txt")
            return False
        
        return True
    
    def create_config_file(self, output_path: str, dirs: Dict, raw_paths: Dict) -> None:
        """Create configuration file"""
        
        config = configparser.ConfigParser()
        
        # Paths section
        config['paths'] = {
            'ucf_crime_raw': raw_paths.get('UCF_Crime', ''),
            'xd_violence_raw': raw_paths.get('XD_Violence', ''),
            'shanghaitech_raw': raw_paths.get('ShanghaiTech', ''),
            'output_dir': str(dirs['root']),
            'features_dir': str(dirs['features']),
            'processed_dir': str(dirs['processed']),
        }
        
        # Processing section
        config['processing'] = {
            'frame_size_h': '224',
            'frame_size_w': '224',
            'num_frames': '16',
            'target_fps': '25',
            'sample_rate': '16000',
            'n_mels': '128',
            'num_workers': '4',
        }
        
        # Splits section
        config['splits'] = {
            'train_ratio': '0.8',
            'few_shot_n_way': '5',
            'few_shot_n_shot': '5',
            'few_shot_n_query': '10',
        }
        
        # Write config
        config_path = Path(output_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            config.write(f)
        
        print(f"\n✓ Configuration saved to: {config_path}")
    
    def create_metadata_template(self, output_path: str) -> None:
        """Create metadata template"""
        
        metadata_template = {
            "dataset_info": {
                "name": "FewShot-SPT Dataset",
                "version": "1.0",
                "created": "2024-03-22",
                "description": "Combined video anomaly detection dataset"
            },
            "statistics": {
                "total_videos": 0,
                "normal_videos": 0,
                "abnormal_videos": 0,
                "total_frames": 0,
                "datasets": {
                    "UCF_Crime": {"videos": 0, "normal": 0, "abnormal": 0},
                    "XD_Violence": {"videos": 0, "normal": 0, "abnormal": 0},
                    "ShanghaiTech": {"videos": 0, "normal": 0, "abnormal": 0},
                }
            },
            "splits": {
                "train": {"normal": [], "abnormal": []},
                "test": {"normal": [], "abnormal": []},
                "few_shot": {
                    "support": {"normal": [], "abnormal": []},
                    "query": {"normal": [], "abnormal": []}
                }
            },
            "processing_notes": {
                "frame_size": "224x224",
                "num_frames": 16,
                "fps": 25,
                "audio_sample_rate": 16000,
                "audio_mels": 128
            }
        }
        
        metadata_path = Path(output_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_template, f, indent=2)
        
        print(f"✓ Metadata template saved to: {metadata_path}")
    
    def generate_setup_report(self, dirs: Dict, raw_paths: Dict) -> str:
        """Generate setup report"""
        
        report = []
        report.append("\n" + "="*60)
        report.append("SETUP COMPLETE")
        report.append("="*60)
        
        report.append("\n📁 Directories Created:")
        for dir_name, dir_path in dirs.items():
            report.append(f"  {dir_name:20s} → {dir_path}")
        
        report.append("\n📂 Raw Datasets:")
        for dataset, path in raw_paths.items():
            status = "✓" if path else "✗"
            report.append(f"  {status} {dataset:20s} → {path or 'Not found'}")
        
        report.append("\n📝 Next Steps:")
        report.append("  1. Download datasets (if not already done):")
        report.append("     See DATASETS_SETUP.md for download links")
        report.append("")
        report.append("  2. Process datasets:")
        report.append("     python scripts/prepare_datasets.py process-ucf ~/datasets/UCF_Crime \\")
        report.append("       -o ./data/UCF_Crime")
        report.append("")
        report.append("  3. Extract features:")
        report.append("     python scripts/extract_features.py extract-all ./data \\")
        report.append("       -v ./data/features/video -a ./data/features/audio")
        report.append("")
        report.append("  4. Validate dataset:")
        report.append("     python scripts/validate_dataset.py ./data/UCF_Crime \\")
        report.append("       -f ./data/features/video -m ./data/processed/metadata.json")
        report.append("")
        report.append("  5. Start training:")
        report.append("     cd src/training")
        report.append("     python train.py --data_dir ../../data/processed")
        
        report.append("\n📚 Documentation:")
        report.append("  - Full setup guide: DATASETS_SETUP.md")
        report.append("  - Quick reference: QUICK_REFERENCE.md")
        report.append("  - Scripts documentation: scripts/README.md")
        
        report.append("\n" + "="*60 + "\n")
        
        return "\n".join(report)
    
    def run_setup(self) -> None:
        """Run complete setup wizard"""
        
        print("\n" + "="*70)
        print("FewShot-SPT Dataset Preparation Setup Wizard")
        print("="*70)
        
        # Check dependencies first
        if not self.check_dependencies():
            print("\n⚠ Please install missing dependencies and run again")
            return
        
        # Setup directories
        dirs = self.setup_directories()
        
        # Setup raw dataset paths
        raw_paths = self.setup_raw_dataset_paths()
        
        # Create config file
        self.create_config_file("./data/.fewshot_config.ini", dirs, raw_paths)
        
        # Create metadata template
        self.create_metadata_template(str(dirs['processed'] / "metadata_template.json"))
        
        # Generate and print report
        report = self.generate_setup_report(dirs, raw_paths)
        print(report)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FewShot-SPT Setup Helper")
    parser.add_argument('--auto', action='store_true', help='Use default settings (no prompts)')
    parser.add_argument('--data-dir', default='./data', help='Data directory path')
    
    args = parser.parse_args()
    
    helper = SetupHelper()
    
    if args.auto:
        dirs = {
            'root': Path(args.data_dir),
            'raw': Path(args.data_dir) / 'raw',
            'processed': Path(args.data_dir) / 'processed',
            'features': Path(args.data_dir) / 'features',
            'features_video': Path(args.data_dir) / 'features' / 'video',
            'features_audio': Path(args.data_dir) / 'features' / 'audio',
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✓ Setup complete (auto mode)")
    else:
        helper.run_setup()


if __name__ == "__main__":
    main()
