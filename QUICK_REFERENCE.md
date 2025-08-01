# Complete Dataset Preparation Guide for FewShot-SPT

## Overview

This document provides step-by-step instructions for downloading, processing, and preparing the three public video anomaly detection datasets for use with the FewShot-SPT model.

**Datasets**:
- 🎬 **UCF-Crime** (~32 GB) - 1,900 untrimmed videos from surveillance
- 🎬 **XD-Violence** (~45 GB) - 4,754 videos with diverse violence types  
- 🎬 **ShanghaiTech** (~24 GB) - 13 scenes with frame-level ground truth

---

## Part 1: Dataset Download

### UCF-Crime (32 GB)

**Paper**: "Real-world Anomaly Detection in Surveillance Videos" - CVPR 2018

#### Download
```bash
# Official source
# Website: https://www.crcv.ucf.edu/datasets/
# Dataset page: "Real-World Anomaly Detection in Surveillance Videos"

# Option 1: Direct download (recommended)
wget -O ~/datasets/UCF_Crime.rar http://crcv.ucf.edu/data/UCF_Crime.rar

# Option 2: Via curl
curl -o ~/datasets/UCF_Crime.rar http://crcv.ucf.edu/data/UCF_Crime.rar

# Extract
mkdir -p ~/datasets/UCF_Crime
unrar x ~/datasets/UCF_Crime.rar ~/datasets/UCF_Crime/

# Or with 7z if available
7z x ~/datasets/UCF_Crime.rar -o ~/datasets/UCF_Crime/
```

#### Expected Structure
After extraction, your UCF-Crime directory should contain:
```
UCF_Crime/
├── Abuse/               (150+ videos)
├── Arrest/              (75+ videos)
├── Arson/               (50+ videos)
├── Assault/             (120+ videos)
├── Robbery/             (200+ videos)
├── Shooting/            (60+ videos)
├── Shoplifting/         (80+ videos)
├── Stealing/            (100+ videos)
├── Tagging/             (40+ videos)
├── Trespassing/         (60+ videos)
├── Vandalism/           (70+ videos)
├── Burglary/            (120+ videos)
├── Explosion/           (50+ videos)
├── Fighting/            (100+ videos)
├── RoadAccidents/       (70+ videos)
├── Seizure/             (40+ videos)
├── Normal_Videos/       (800+ videos)
└── Normal_Events_v2/    (additional normal videos)
```

---

### XD-Violence (45 GB)

**Paper**: "Exploring the Limits of Weakly Supervised Learning" - ECCV 2020

#### Download
```bash
# Official source
# Website: https://roc.unict.it/xd-violence/
# Contact: https://roc.unict.it/xd-violence/download.html

# Steps:
# 1. Visit https://roc.unict.it/xd-violence/download.html
# 2. Fill request form (email required)
# 3. Download links sent via email
# 4. Download all parts (typically split into multiple archives)

# Example if you have links in links.txt:
aria2c \
    --auto-file-renaming=false \
    --conditional-get=true \
    --max-concurrent-downloads=3 \
    --max-connection-per-server=4 \
    --split=4 \
    --continue=true \
    --input-file=links.txt
```

#### Expected Structure
```
XD_Violence/
├── Violence/           (1,140 videos)
│   ├── v_ActionTrimmed_g001_c001.mp4
│   ├── v_ActionTrimmed_g001_c002.mp4
│   └── ...
├── NonViolence/        (3,614 videos)
│   ├── v_ActionTrimmed_g024_c001.mp4
│   ├── v_ActionTrimmed_g024_c002.mp4
│   └── ...
└── metadata.json       (video metadata)
```

---

### ShanghaiTech (24 GB)

**Paper**: "Anomaly Detection with Robust Deep Autoencoders" - ICPR 2014

#### Download
```bash
# Official source
# Website: http://www.ahcip.com/
# Direct link: http://www.ahcip.com/sample/TRAIN-TEST.rar

# Download
wget -O ~/datasets/ShanghaiTech.rar http://www.ahcip.com/sample/TRAIN-TEST.rar

# Or with curl
curl -o ~/datasets/ShanghaiTech.rar http://www.ahcip.com/sample/TRAIN-TEST.rar

# Extract
mkdir -p ~/datasets/ShanghaiTech
unrar x ~/datasets/ShanghaiTech.rar ~/datasets/ShanghaiTech/

# If .zip format
unzip ~/datasets/ShanghaiTech.zip -d ~/datasets/ShanghaiTech/
```

#### Expected Structure
```
ShanghaiTech/
├── training/
│   ├── frames/
│   │   ├── 01_0014_0/   (frame images 0-549)
│   │   ├── 01_0014_1/
│   │   └── ...
│   ├── gt/
│   │   ├── 01_0014_gt.npy  (ground truth binary masks)
│   │   ├── 01_0014_gt.npy
│   │   └── ...
│   └── video_label.txt
├── testing/
│   ├── frames/ (same structure)
│   ├── gt/
│   └── video_label.txt
└── split_info.txt
```

**Note**: ShanghaiTech comes as individual JPEG frames, not videos. We'll convert these to MP4 in Step 3.

---

## Part 2: Dataset Preprocessing

### Prerequisites

Install required Python packages:
```bash
pip install torch torchvision torchaudio opencv-python librosa soundfile pillow tqdm numpy scipy scikit-learn

# Optional but recommended
pip install moviepy  # For audio extraction
```

### Step 1: Create Output Directory Structure

```bash
# Method 1: Using Python script
python scripts/prepare_datasets.py create-structure ./data

# Method 2: Manual
mkdir -p ./data/{UCF_Crime,XD_Violence,ShanghaiTech}/{normal,abnormal}
```

### Step 2: Organize Videos

Process each dataset into binary (normal/abnormal) classification:

```bash
# UCF-Crime
python scripts/prepare_datasets.py process-ucf ~/datasets/UCF_Crime -o ./data/UCF_Crime

# XD-Violence
python scripts/prepare_datasets.py process-xd ~/datasets/XD_Violence -o ./data/XD_Violence

# ShanghaiTech (converts frames to videos)
python scripts/prepare_datasets.py process-st ~/datasets/ShanghaiTech -o ./data/ShanghaiTech
```

Output structure for each:
```
data/UCF_Crime/
├── normal/       (800+ files)
│   ├── Normal001.mp4
│   ├── Normal002.mp4
│   └── ...
└── abnormal/     (1,100+ files)
    ├── Abuse001.mp4
    ├── Assault001.mp4
    └── ...
```

### Step 3: Create Train/Test Splits

Split each dataset into 80% training and 20% testing:

```bash
# For each dataset
python scripts/prepare_datasets.py create-split ./data/UCF_Crime --ratio 0.8
python scripts/prepare_datasets.py create-split ./data/XD_Violence --ratio 0.8
python scripts/prepare_datasets.py create-split ./data/ShanghaiTech --ratio 0.8
```

Creates structure:
```
data/UCF_Crime/
├── train/
│   ├── normal/ (640+ videos)
│   └── abnormal/ (880+ videos)
└── test/
    ├── normal/ (160+ videos)
    └── abnormal/ (220+ videos)
```

---

## Part 3: Feature Extraction

### Extract Video Frames

Extract 16 uniformly-spaced frames (224×224) from each video:

```bash
# Extract all video features
python scripts/extract_features.py extract-all \
    ./data \
    -v ./data/features/video \
    -a ./data/features/audio
```

Or extract specific dataset:
```bash
# Video only for quick start
python scripts/extract_features.py extract-video \
    ./data/UCF_Crime/train \
    -o ./data/features/video
```

Output structure:
```
data/features/
├── video/
│   ├── normal/
│   │   ├── video1.pt    # torch tensor (16, 3, 224, 224)
│   │   ├── video2.pt
│   │   └── ...
│   └── abnormal/ (same)
└── audio/
    ├── normal/
    │   ├── video1.pt    # torch tensor (128, T)
    │   └── ...
    └── abnormal/ (same)
```

### Extract Audio Features

Audio features are mel-spectrograms (128 bins, 16kHz):

```bash
# Audio extraction happens in extract-all above
# Or manually extract from videos

python scripts/extract_features.py extract-all \
    ./data/UCF_Crime/train \
    -a ./data/features/audio/ucf_crime \
    --skip-video  # Skip video extraction
```

Audio file dimensions: `(128, T)` where T depends on video duration

---

## Part 4: Create Final Dataset

### Generate Metadata

Create metadata JSON and splits for training:

```bash
python scripts/extract_features.py create-metadata \
    ./data/features/video \
    -o ./data/processed \
    --split 0.8
```

Creates:
- `./data/processed/metadata.json` - Train/test/few-shot splits
- Train/test/few-shot video lists

### Metadata Format

```json
{
  "train": {
    "normal": ["normal/video1.pt", "normal/video2.pt", ...],
    "abnormal": ["abnormal/anom1.pt", ...]
  },
  "test": {
    "normal": [...],
    "abnormal": [...]
  },
  "few_shot": {
    "support": {
      "normal": ["normal/sup1.pt", "normal/sup2.pt", ...],
      "abnormal": ["abnormal/sup1.pt", ...]
    },
    "query": {
      "normal": [...],
      "abnormal": [...]
    }
  }
}
```

---

## Part 5: Using Datasets with FewShot-SPT

### Load Data for Training

```python
import torch
from pathlib import Path
sys.path.insert(0, 'src')

from datasets.video_dataset import PreprocessedVideoAnomalyDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = PreprocessedVideoAnomalyDataset(
    dataset_root="./data/processed",
    split='train',
    modalities=['video', 'audio']
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Iterate
for batch in train_loader:
    video = batch['video']  # (B, T, 3, H, W)
    audio = batch['audio']  # (B, 128, T)
    labels = batch['label']  # (B,)
    
    # Your training code here
    break
```

### DataLoader Outputs

```python
{
    'video_id': str,                    # Video identifier
    'label': torch.Tensor (1,),         # 0 or 1
    'video': torch.Tensor (T, 3, H, W), # (16, 3, 224, 224)
    'audio': torch.Tensor (n_mels, T),  # (128, T) 
    'text': torch.Tensor (768,),        # Optional text embeddings
}
```

---

## Complete Automation Script

Use the comprehensive setup script:

```bash
# Interactive mode
bash scripts/quick_setup.sh

# Or run specific commands
bash scripts/quick_setup.sh check        # Check dependencies
bash scripts/quick_setup.sh download     # Show download instructions
bash scripts/quick_setup.sh process      # Process all datasets
bash scripts/quick_setup.sh extract      # Extract features
bash scripts/quick_setup.sh verify       # Verify dataset
bash scripts/quick_setup.sh full         # Complete pipeline

# Make executable first (on macOS/Linux)
chmod +x scripts/quick_setup.sh scripts/*.py
```

---

## Dataset Statistics

After processing, expected dataset sizes:

| Dataset | Total Videos | Normal | Abnormal | Train (80%) | Test (20%) | Storage |
|---------|-------------|--------|----------|------------|-----------|---------|
| **UCF-Crime** | 1,900+ | 800+ | 1,100+ | 1,520+ | 380+ | 32 GB |
| **XD-Violence** | 4,754 | 3,614 | 1,140 | 3,803 | 951 | 45 GB |
| **ShanghaiTech** | 437 clips | ~220 | ~217 | ~350 | ~87 | 24 GB |
| **Total** | ~7,100 | ~4,600 | ~2,500 | ~5,700 | ~1,400 | **101 GB** |

**Feature Storage** (extracted):
- Video features: ~150-200 GB (depending on quality/compression)
- Audio features: ~10-15 GB
- **Total with features: ~250-300 GB**

---

## Troubleshooting

### Download Issues

```bash
# Test connection
curl -I https://crcv.ucf.edu/datasets/

# Resume interrupted download
wget --continue -O dataset.rar http://source.url

# Use proxy if needed
wget -O dataset.rar --proxy=on http://source.url
```

### Processing Issues

```bash
# Check video format compatibility
ffprobe -v error -select_streams v:0 video.mp4

# Convert incompatible videos
ffmpeg -i input.mp4 -c:v h264 -c:a aac output.mp4

# If frame extraction fails, try fallback
python scripts/extract_features.py extract-video dataset_dir --fallback-fps 1
```

### Memory Issues

```bash
# Reduce batch size during extraction
# Modify extract_features.py: MAX_BATCH_SIZE = 8

# Process datasets one-by-one instead of simultaneously
for dataset in UCF_Crime XD_Violence ShanghaiTech; do
    python scripts/extract_features.py extract-all "./data/$dataset/train" \
        -v "./data/features/video/$dataset"
done
```

### Storage Issues

```bash
# Use external SSD for faster I/O
# Store raw videos on external drive
# Extract features locally for faster training

# Example: Mount external SSD
mkdir -p /Volumes/datasets
# Mount your external drive here
cp -r ~/datasets/* /Volumes/datasets/

# Train with: --data_dir /Volumes/datasets/features
```

---

## Verification Checklist

- [ ] **Download**: All three datasets downloaded (101 GB total)
- [ ] **Organization**: Videos organized in normal/abnormal structure
- [ ] **Processing**: Datasets processed (mp4 format)
- [ ] **Splits**: Train/test splits created (80/20)
- [ ] **Features**: Video frames extracted (16 per video, 224×224)
- [ ] **Audio**: Audio mel-spectrograms extracted
- [ ] **Metadata**: metadata.json created in format above
- [ ] **Testing**: Training pipeline initialization successful

---

## Next Steps

After dataset preparation:

1. **Train Model**:
   ```bash
   cd src/training
   python train.py --config config.json --data_dir ./data/processed
   ```

2. **Evaluate**:
   ```bash
   python train.py --mode eval --checkpoint model.pth
   ```

3. **Few-Shot Learning**:
   ```bash
   python train.py --mode few_shot --n_way 5 --n_shot 5 --n_query 10
   ```

---

## References

- **UCF-Crime**: [Real-world Anomaly Detection in Surveillance Videos](https://arxiv.org/pdf/1801.04264.pdf)
- **XD-Violence**: [Exploring the Limits of Weakly Supervised Learning](https://arxiv.org/pdf/2012.03665.pdf)
- **ShanghaiTech**: [Anomaly Detection with Robust Deep Autoencoders](https://arxiv.org/pdf/1406.0943.pdf)

---

**Date**: March 22, 2026  
**Status**: Complete  
**Version**: 1.0
