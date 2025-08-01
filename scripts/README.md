# FewShot-SPT Dataset Scripts

Automated tools for downloading, processing, validating, and managing video anomaly detection datasets.

## Scripts Overview

### 1. `prepare_datasets.py` - Dataset Processing
Organize raw datasets into binary classification format (normal/abnormal).

**Features**:
- Process UCF-Crime, XD-Violence, ShanghaiTech datasets
- Convert frame sequences to videos (ShanghaiTech)
- Create train/test splits
- CLI interface with multiple commands

**Usage**:
```bash
# Process UCF-Crime
python prepare_datasets.py process-ucf ~/datasets/UCF_Crime -o ./data/UCF_Crime

# Process XD-Violence
python prepare_datasets.py process-xd ~/datasets/XD_Violence -o ./data/XD_Violence

# Process ShanghaiTech (frames → videos)
python prepare_datasets.py process-st ~/datasets/ShanghaiTech -o ./data/ShanghaiTech

# Create train/test split
python prepare_datasets.py create-split ./data/UCF_Crime --ratio 0.8

# Create directory structure
python prepare_datasets.py create-structure ./data
```

**Output**:
```
./data/UCF_Crime/
├── normal/       (800+ videos)
├── abnormal/     (1,100+ videos)
└── [train/test/] (if splits created)
```

---

### 2. `extract_features.py` - Feature Extraction
Extract video frames and audio mel-spectrograms for training.

**Features**:
- Extract 16 uniformly-spaced frames at 224×224
- Extract 128-bin mel-spectrograms (16kHz)
- ImageNet normalization
- Batch processing with progress bars
- Create metadata and splits

**Usage**:
```bash
# Extract both video and audio features
python extract_features.py extract-all \
    ./data/UCF_Crime/train \
    -v ./data/features/video \
    -a ./data/features/audio

# Extract video only
python extract_features.py extract-video \
    ./data/UCF_Crime/train \
    -o ./data/features/video \
    --size 224 224 \
    --frames 16

# Create metadata for training
python extract_features.py create-metadata \
    ./data/features/video \
    -o ./data/processed \
    --split 0.8
```

**Output**:
```
./data/features/
├── video/
│   ├── normal/
│   │   └── video1.pt (16, 3, 224, 224)
│   └── abnormal/
└── audio/
    ├── normal/
    │   └── video1.pt (128, T)
    └── abnormal/

./data/processed/
└── metadata.json (train/test/few-shot splits)
```

---

### 3. `validate_dataset.py` - Dataset Validation
Verify dataset integrity, structure, and statistics.

**Features**:
- Check directory structure
- Validate all video files
- Check extracted features
- Verify metadata JSON
- Compute dataset statistics
- Generate validation report

**Usage**:
```bash
# Validate dataset structure and videos
python validate_dataset.py ./data/UCF_Crime

# Validate with features
python validate_dataset.py ./data/UCF_Crime \
    -f ./data/features/video \
    -m ./data/processed/metadata.json

# Check specific dataset
python validate_dataset.py ./data/XD_Violence
```

**Output Report**:
```
════════════════════════════════════════════════════════════
DATASET VALIDATION REPORT
════════════════════════════════════════════════════════════

📹 Video Files:
  Total: 1900
  Valid: 1897 ✓
  Corrupted: 3 ✗
  Health: 99.8%

🎬 Feature Files:
  Total: 1900
  Valid: 1900 ✓
  Corrupted: 0 ✗
  Health: 100.0%

📊 Dataset Statistics:

  NORMAL:
    Videos: 800
    Avg Frames: 348 (min: 120, max: 645)
    Avg FPS: 25.00

  ABNORMAL:
    Videos: 1100
    Avg Frames: 312 (min: 100, max: 600)
    Avg FPS: 25.00
════════════════════════════════════════════════════════════
```

---

### 4. `quick_setup.sh` - Interactive Setup
Complete automation with interactive menu.

**Features**:
- Dependency checking
- Download instructions for all datasets
- Complete pipeline execution
- Progress tracking
- Verification and testing

**Usage**:
```bash
# Interactive menu
bash quick_setup.sh

# Command mode
bash quick_setup.sh full        # Complete pipeline
bash quick_setup.sh check       # Check dependencies
bash quick_setup.sh download    # Download instructions
bash quick_setup.sh process     # Process all datasets
bash quick_setup.sh extract     # Extract features
bash quick_setup.sh verify      # Verify dataset
bash quick_setup.sh test        # Test training
```

---

## Typical Workflow

### Step 1: Download Datasets
```bash
# Manual download (see DATASETS_SETUP.md)
# UCF-Crime: ~32 GB
# XD-Violence: ~45 GB
# ShanghaiTech: ~24 GB
# Total: ~101 GB

# Organize in ~/datasets/
~/datasets/
├── UCF_Crime/
├── XD_Violence/
└── ShanghaiTech/
```

### Step 2: Process Datasets
```bash
# Option A: Interactive
bash quick_setup.sh

# Option B: Manual
python prepare_datasets.py create-structure ./data
python prepare_datasets.py process-ucf ~/datasets/UCF_Crime -o ./data/UCF_Crime
python prepare_datasets.py process-xd ~/datasets/XD_Violence -o ./data/XD_Violence
python prepare_datasets.py process-st ~/datasets/ShanghaiTech -o ./data/ShanghaiTech
python prepare_datasets.py create-split ./data/UCF_Crime
python prepare_datasets.py create-split ./data/XD_Violence
python prepare_datasets.py create-split ./data/ShanghaiTech
```

### Step 3: Extract Features
```bash
python extract_features.py extract-all \
    ./data \
    -v ./data/features/video \
    -a ./data/features/audio

python extract_features.py create-metadata \
    ./data/features/video \
    -o ./data/processed
```

### Step 4: Validate Dataset
```bash
python validate_dataset.py ./data/UCF_Crime \
    -f ./data/features/video \
    -m ./data/processed/metadata.json
```

### Step 5: Train Model
```bash
cd src/training
python train.py --data_dir ../../data/processed
```

---

## Command Reference

| Script | Purpose | Main Commands |
|--------|---------|--------------|
| `prepare_datasets.py` | Organize raw videos | `process-ucf`, `process-xd`, `process-st`, `create-split` |
| `extract_features.py` | Extract video/audio | `extract-all`, `extract-video`, `create-metadata` |
| `validate_dataset.py` | Check dataset health | (single command, multiple flags) |
| `quick_setup.sh` | Automation | `full`, `check`, `process`, `extract`, `verify`, `test` |

---

## System Requirements

**Dependencies**:
```bash
pip install torch torchvision torchaudio
pip install opencv-python librosa soundfile
pip install tqdm numpy scipy scikit-learn pillow
```

**System**:
- Python 3.8+
- 150-250 GB disk space (raw data + features)
- 8+ GB RAM
- GPU recommended (CUDA/MPS)

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
```bash
pip install -r ../requirements.txt
```

### Issue: Out of memory during extraction
Reduce batch size in `extract_features.py`:
```python
# Modify: MAX_BATCH_SIZE = 8
```

### Issue: Video files corrupted
```bash
# List corrupted files
python validate_dataset.py ./data --fix-corrupted

# Re-download specific dataset
```

### Issue: FFmpeg not found (audio extraction)
```bash
# macOS
brew install ffmpeg

# Linux
apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

---

## Performance Tips

1. **Use SSD**: Store datasets on SSD for 2-3x faster I/O
2. **Parallel Downloads**: Use aria2c for multi-threaded downloads
3. **Batch Processing**: Extract features in batches to optimize memory
4. **Pre-compute**: Extract all features before starting training

---

## Examples

### Process Single Dataset
```bash
python prepare_datasets.py process-ucf ~/datasets/UCF_Crime -o ./data/UCF_Crime
python extract_features.py extract-all ./data/UCF_Crime/train -v ./data/features/video
python validate_dataset.py ./data/UCF_Crime
```

### Create Few-Shot Episodes
```bash
# Metadata automatically creates few-shot splits:
python extract_features.py create-metadata ./data/features/video -o ./data/processed

# Check metadata:
cat ./data/processed/metadata.json | grep -A 5 '"few_shot"'
```

### Validate All Dataset
```bash
for dataset in UCF_Crime XD_Violence ShanghaiTech; do
    python validate_dataset.py "./data/$dataset" \
        -f "./data/features/video/$dataset" \
        -m ./data/processed/metadata.json
done
```

---

## Next Steps

After dataset preparation:

1. **Training**:
   ```bash
   cd src/training && python train.py
   ```

2. **Few-Shot Learning**:
   ```bash
   python train.py --mode few_shot --n_way 5 --n_shot 5
   ```

3. **Evaluation**:
   ```bash
   python train.py --mode eval --checkpoint model.pth
   ```

---

**Created**: March 22, 2026  
**Version**: 1.0  
**Status**: Production-ready
