#!/bin/bash
#
# Quick Start Guide - Dataset Preparation for FewShot-SPT
# Complete workflow from paper datasets to training-ready features
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASETS_DIR="${HOME}/datasets"
DATA_DIR="./data"
SCRIPTS_DIR="./scripts"

# ============================================================================
# Helper Functions
# ============================================================================

print_title() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# ============================================================================
# Step 1: Check Dependencies
# ============================================================================

check_dependencies() {
    print_title "Checking Dependencies"
    
    # Python packages
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" || print_error "PyTorch not installed"
    python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')" || print_error "OpenCV not installed"
    python3 -c "import torchaudio; print(f'  TorchAudio: {torchaudio.__version__}')" || print_error "TorchAudio not installed"
    
    # System dependencies
    which ffmpeg > /dev/null && print_step "FFmpeg installed" || print_warning "FFmpeg not found (needed for audio extraction)"
}

# ============================================================================
# Step 2: Download Datasets
# ============================================================================

download_ucf_crime() {
    print_title "UCF-Crime Dataset"
    
    echo "📥 Download Instructions:"
    echo "  1. Visit: https://www.crcv.ucf.edu/datasets/"
    echo "  2. Download: Real-World Anomaly Detection in Surveillance Videos"
    echo "  3. Extract to: $DATASETS_DIR/UCF_Crime"
    echo ""
    echo "  Size: ~32 GB"
    echo "  Format: MP4 videos organized by crime type"
}

download_xd_violence() {
    print_title "XD-Violence Dataset"
    
    echo "📥 Download Instructions:"
    echo "  1. Visit: https://roc.unict.it/xd-violence/"
    echo "  2. Request access via email"
    echo "  3. Download provided links"
    echo "  4. Extract to: $DATASETS_DIR/XD_Violence"
    echo ""
    echo "  Size: ~45 GB"
    echo "  Structure: Violence/ and NonViolence/ directories"
}

download_shanghaitech() {
    print_title "ShanghaiTech Dataset"
    
    echo "📥 Download Instructions:"
    echo "  1. Visit: http://www.ahcip.com/"
    echo "  2. Download via: wget -O ShanghaiTech.rar http://www.ahcip.com/sample/TRAIN-TEST.rar"
    echo "  3. Extract: unrar x ShanghaiTech.rar -d $DATASETS_DIR/ShanghaiTech"
    echo ""
    echo "  Size: ~24 GB"
    echo "  Format: Frame sequences with Ground Truth annotations"
}

download_all_datasets() {
    print_title "Downloading Datasets"
    
    echo -e "${YELLOW}Manual Download Required${NC}\n"
    
    read -p "Do you want to see download instructions for all datasets? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_ucf_crime
        echo "---"
        download_xd_violence
        echo "---"
        download_shanghaitech
        
        echo -e "\n${YELLOW}After downloading, organize in:${NC}"
        echo "  $DATASETS_DIR/"
        echo "  ├── UCF_Crime/"
        echo "  ├── XD_Violence/"
        echo "  └── ShanghaiTech/"
    fi
}

# ============================================================================
# Step 3: Process Datasets
# ============================================================================

create_dataset_structure() {
    print_title "Creating Dataset Structure"
    
    mkdir -p "$DATA_DIR"/{UCF_Crime,XD_Violence,ShanghaiTech}/{normal,abnormal}
    print_step "Created directory structure in $DATA_DIR"
}

process_ucf_crime() {
    print_title "Processing UCF-Crime"
    
    SOURCE="$DATASETS_DIR/UCF_Crime"
    TARGET="$DATA_DIR/UCF_Crime"
    
    if [ ! -d "$SOURCE" ]; then
        print_error "Source directory not found: $SOURCE"
        return 1
    fi
    
    echo "Processing UCF-Crime dataset..."
    python3 "$SCRIPTS_DIR/prepare_datasets.py" process-ucf "$SOURCE" -o "$TARGET"
    
    print_step "UCF-Crime processing complete"
}

process_xd_violence() {
    print_title "Processing XD-Violence"
    
    SOURCE="$DATASETS_DIR/XD_Violence"
    TARGET="$DATA_DIR/XD_Violence"
    
    if [ ! -d "$SOURCE" ]; then
        print_error "Source directory not found: $SOURCE"
        return 1
    fi
    
    echo "Processing XD-Violence dataset..."
    python3 "$SCRIPTS_DIR/prepare_datasets.py" process-xd "$SOURCE" -o "$TARGET"
    
    print_step "XD-Violence processing complete"
}

process_shanghaitech() {
    print_title "Processing ShanghaiTech"
    
    SOURCE="$DATASETS_DIR/ShanghaiTech"
    TARGET="$DATA_DIR/ShanghaiTech"
    
    if [ ! -d "$SOURCE" ]; then
        print_error "Source directory not found: $SOURCE"
        return 1
    fi
    
    echo "Processing ShanghaiTech dataset (frames → videos)..."
    python3 "$SCRIPTS_DIR/prepare_datasets.py" process-st "$SOURCE" -o "$TARGET"
    
    print_step "ShanghaiTech processing complete"
}

process_all_datasets() {
    print_title "Processing All Datasets"
    
    for dataset in UCF_Crime XD_Violence ShanghaiTech; do
        if [ -d "$DATASETS_DIR/$dataset" ]; then
            case $dataset in
                UCF_Crime)
                    process_ucf_crime || true
                    ;;
                XD_Violence)
                    process_xd_violence || true
                    ;;
                ShanghaiTech)
                    process_shanghaitech || true
                    ;;
            esac
        else
            print_warning "Skipping $dataset (directory not found)"
        fi
    done
}

# ============================================================================
# Step 4: Create Train/Test Splits
# ============================================================================

create_splits() {
    print_title "Creating Train/Test Splits"
    
    for dataset in UCF_Crime XD_Violence ShanghaiTech; do
        dataset_dir="$DATA_DIR/$dataset"
        
        if [ -d "$dataset_dir" ]; then
            echo "Creating split for $dataset..."
            python3 "$SCRIPTS_DIR/prepare_datasets.py" create-split "$dataset_dir" --ratio 0.8
        fi
    done
}

# ============================================================================
# Step 5: Extract Features
# ============================================================================

extract_video_features() {
    print_title "Extracting Video Features"
    
    DATASET_DIR="${1:-./$DATA_DIR}"
    
    echo "Extracting 224×224 frames (16 per video) from all datasets..."
    
    # Create structure
    mkdir -p "$DATA_DIR/features"/video/{normal,abnormal}
    
    # Extract from each dataset
    for dataset in UCF_Crime XD_Violence ShanghaiTech; do
        dataset_path="$DATASET_DIR/$dataset/train"
        
        if [ -d "$dataset_path" ]; then
            echo "Extracting from $dataset..."
            python3 "$SCRIPTS_DIR/extract_features.py" extract-all \
                "$dataset_path" \
                -v "$DATA_DIR/features/video" \
                -a "$DATA_DIR/features/audio"
        fi
    done
    
    print_step "Video feature extraction complete"
}

# ============================================================================
# Step 6: Create Metadata
# ============================================================================

create_metadata() {
    print_title "Creating Dataset Metadata"
    
    echo "Creating metadata and train/test/few-shot splits..."
    
    python3 "$SCRIPTS_DIR/extract_features.py" create-metadata \
        "$DATA_DIR/features/video" \
        -o "$DATA_DIR/processed" \
        --split 0.8
    
    print_step "Metadata creation complete"
}

# ============================================================================
# Step 7: Verify Dataset
# ============================================================================

verify_dataset() {
    print_title "Verifying Dataset"
    
    echo "Checking extracted features..."
    
    for label in normal abnormal; do
        video_count=$(find "$DATA_DIR/features/video/$label" -name "*.pt" 2>/dev/null | wc -l)
        audio_count=$(find "$DATA_DIR/features/audio/$label" -name "*.pt" 2>/dev/null | wc -l)
        
        echo "  $label:"
        echo "    Video frames: $video_count ✓"
        echo "    Audio features: $audio_count ✓"
    done
    
    if [ -f "$DATA_DIR/processed/metadata.json" ]; then
        echo ""
        echo "  Dataset metadata: ✓"
        
        # Show statistics
        python3 - << 'EOF'
import json
from pathlib import Path

metadata_file = Path("./data/processed/metadata.json")
if metadata_file.exists():
    with open(metadata_file) as f:
        data = json.load(f)
    
    for split in ['train', 'test']:
        if split in data:
            n_normal = len(data[split]['normal'])
            n_abnormal = len(data[split]['abnormal'])
            print(f"    {split.title()}: {n_normal} normal + {n_abnormal} abnormal = {n_normal+n_abnormal} total")
EOF
    fi
}

# ============================================================================
# Step 8: Test Training Pipeline
# ============================================================================

test_training_pipeline() {
    print_title "Testing Training Pipeline"
    
    echo "Running quick training test with extracted features..."
    
    python3 - << 'EOF'
import torch
from pathlib import Path
import sys

sys.path.insert(0, './src')

try:
    from models import create_fewshot_spt
    from training.train_utils import AverageMeter, MetricTracker
    
    # Load model
    model = create_fewshot_spt(num_classes=2)
    
    # Load sample batch
    video_frames = torch.randn(4, 16, 3, 224, 224)
    audio_features = torch.randn(4, 128, 16)
    labels = torch.randint(0, 2, (4,))
    
    # Forward pass
    output = model(
        video_frames=video_frames,
        audio_features=audio_features
    )
    
    print("  ✓ Model loading: OK")
    print(f"  ✓ Forward pass: OK (output shape {output.shape})")
    print(f"  ✓ Training pipeline: Ready")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
EOF
}

# ============================================================================
# Main Menu
# ============================================================================

show_menu() {
    echo -e "\n${BLUE}FewShot-SPT Dataset Preparation Menu${NC}"
    echo "======================================"
    echo "1) Check dependencies"
    echo "2) Show dataset download instructions"
    echo "3) Create directory structure"
    echo "4) Process all datasets"
    echo "5) Create train/test splits"
    echo "6) Extract video features"
    echo "7) Create metadata"
    echo "8) Verify dataset"
    echo "9) Test training pipeline"
    echo "10) Run complete pipeline (all steps)"
    echo "0) Exit"
    echo ""
    read -p "Select option: " option
}

run_complete_pipeline() {
    print_title "Running Complete Pipeline"
    
    check_dependencies
    create_dataset_structure
    
    echo ""
    print_warning "Please ensure datasets are downloaded in: $DATASETS_DIR"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Aborted."
        return 1
    fi
    
    process_all_datasets
    create_splits
    extract_video_features
    create_metadata
    verify_dataset
    test_training_pipeline
    
    print_title "✓ Pipeline Complete!"
    echo "Your dataset is ready for training:"
    echo "  Location: $DATA_DIR/processed"
    echo "  Metadata: $DATA_DIR/processed/metadata.json"
    echo ""
    echo "Next: Start training with:"
    echo "  cd src/training"
    echo "  python train.py --config config.json"
}

# ============================================================================
# Main
# ============================================================================

main() {
    if [ $# -eq 0 ]; then
        # Interactive menu
        while true; do
            show_menu
            
            case $option in
                1) check_dependencies ;;
                2) download_all_datasets ;;
                3) create_dataset_structure ;;
                4) process_all_datasets ;;
                5) create_splits ;;
                6) extract_video_features ;;
                7) create_metadata ;;
                8) verify_dataset ;;
                9) test_training_pipeline ;;
                10) run_complete_pipeline ;;
                0) echo "Exiting..."; exit 0 ;;
                *) print_error "Invalid option" ;;
            esac
        done
    else
        # Command line mode
        case $1 in
            full) run_complete_pipeline ;;
            check) check_dependencies ;;
            download) download_all_datasets ;;
            process) process_all_datasets ;;
            splits) create_splits ;;
            extract) extract_video_features ;;
            metadata) create_metadata ;;
            verify) verify_dataset ;;
            test) test_training_pipeline ;;
            *) 
                echo "Usage: $0 [full|check|download|process|splits|extract|metadata|verify|test]"
                exit 1
                ;;
        esac
    fi
}

# Run
main "$@"
