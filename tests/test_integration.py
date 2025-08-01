"""
Integration tests for FewShot-SPT components.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.fewshot_spt import create_fewshot_spt
from models.components.egke import create_egke
from models.components.amg import create_amg
from models.components.perceiver_io import create_perceiver_io
from models.components.apfsl import create_apfsl
from utils.metrics import AnomalyDetectionMetrics
from utils.losses import CombinedAnomalyLoss


def test_egke():
    """Test Event-Guided Keyframe Extraction."""
    print("\n" + "="*60)
    print("Testing EGKE (Event-Guided Keyframe Extraction)")
    print("="*60)
    
    egke = create_egke(feature_dim=512, keyframe_ratio=0.3)
    
    # Input: batch of 2, 16 frames, 512-dim features
    batch_size, num_frames, feature_dim = 2, 16, 512
    video_frames = torch.randn(batch_size, num_frames, feature_dim)
    
    # Extract keyframes
    keyframes = egke(video_frames)
    
    print(f"✓ Input shape: {video_frames.shape}")
    print(f"✓ Output shape: {keyframes.shape}")
    print(f"✓ Keyframe ratio: {keyframes.shape[1] / num_frames:.2%}")
    
    # Test with return flags
    keyframes, indices, scores = egke(
        video_frames,
        return_indices=True,
        return_scores=True
    )
    
    print(f"✓ Anomaly scores shape: {scores.shape}")
    print(f"✓ Extracted {len(indices[0])} keyframes from {num_frames} frames")
    print("✓ EGKE test PASSED\n")


def test_amg():
    """Test Adaptive Modality Gating."""
    print("="*60)
    print("Testing AMG (Adaptive Modality Gating)")
    print("="*60)
    
    amg = create_amg(
        modality_dims={'video': 512, 'audio': 512, 'text': 512},
        output_dim=512
    )
    
    batch_size, num_frames, feature_dim = 2, 16, 512
    
    # Multi-modal features
    features = {
        'video': torch.randn(batch_size, num_frames, feature_dim),
        'audio': torch.randn(batch_size, num_frames, feature_dim),
        'text': torch.randn(batch_size, num_frames, feature_dim)
    }
    
    # Fuse modalities
    fused, gates = amg(features, return_gates=True)
    
    print(f"✓ Fused features shape: {fused.shape}")
    print(f"✓ Gating weights shape: {gates.shape}")
    print(f"✓ Average gating weights: {gates.mean(dim=0).tolist()}")
    print("✓ AMG test PASSED\n")


def test_perceiver_io():
    """Test Perceiver IO attention."""
    print("="*60)
    print("Testing Perceiver IO Attention")
    print("="*60)
    
    perceiver = create_perceiver_io(
        input_dim=512,
        output_dim=512,
        num_latents=64
    )
    
    batch_size, num_frames, feature_dim = 2, 16, 512
    features = torch.randn(batch_size, num_frames, feature_dim)
    
    # Process through perceiver
    output = perceiver(features)
    
    print(f"✓ Input shape: {features.shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ Perceiver IO test PASSED\n")


def test_apfsl():
    """Test Adaptive Prototypical Few-Shot Learning."""
    print("="*60)
    print("Testing APFSL (Adaptive Prototypical Few-Shot Learning)")
    print("="*60)
    
    apfsl = create_apfsl(feature_dim=512)
    
    # Create few-shot episode
    n_way, n_shot, n_query = 2, 3, 5
    feature_dim = 512
    
    # Support set
    support_features = torch.randn(n_way * n_shot, feature_dim)
    support_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    
    # Query set
    query_features = torch.randn(n_way * n_query, feature_dim)
    query_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Few-shot classification
    logits = apfsl(
        support_features, support_labels,
        query_features, query_labels,
        adaptive_refinement=True
    )
    
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Expected shape: ({n_way * n_query}, {n_way})")
    
    # Classification accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == query_labels).float().mean()
    print(f"✓ Few-shot accuracy: {accuracy:.2%}")
    print("✓ APFSL test PASSED\n")


def test_fewshot_spt():
    """Test full FewShot-SPT model."""
    print("="*60)
    print("Testing Full FewShot-SPT Model")
    print("="*60)
    
    model = create_fewshot_spt(num_classes=2, keyframe_ratio=0.3)
    model.eval()
    
    batch_size, num_frames = 2, 16
    
    # Prepare inputs
    video = torch.randn(batch_size, num_frames, 3, 224, 224)
    audio = torch.randn(batch_size, num_frames, 128)
    text = torch.randn(batch_size, num_frames, 768)
    
    print("\nInput shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Text: {text.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(
            video_frames=video,
            audio_features=audio,
            text_features=text,
            return_intermediate=True
        )
    
    print("\nOutput shapes:")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Fused features: {output['fused_features'].shape}")
    print(f"  Refined features: {output['refined_features'].shape}")
    
    # Predictions
    predictions = torch.softmax(output['logits'], dim=1)
    print(f"\nPredictions (probability of anomaly):")
    print(f"  Sample 0: {predictions[0, 1]:.4f}")
    print(f"  Sample 1: {predictions[1, 1]:.4f}")
    
    print("✓ FewShot-SPT test PASSED\n")


def test_metrics():
    """Test metrics computation."""
    print("="*60)
    print("Testing Metrics Computation")
    print("="*60)
    
    # Create dummy predictions and labels
    num_samples = 100
    predictions = np.random.rand(num_samples)
    labels = np.random.randint(0, 2, num_samples)
    
    # Compute metrics
    metrics = AnomalyDetectionMetrics.compute_all_metrics(predictions, labels)
    
    print("\nComputed metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
    
    print("✓ Metrics test PASSED\n")


def test_losses():
    """Test loss functions."""
    print("="*60)
    print("Testing Loss Functions")
    print("="*60)
    
    loss_fn = CombinedAnomalyLoss()
    
    batch_size, num_classes = 32, 2
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    features = torch.randn(batch_size, 512)
    
    # Compute losses
    losses = loss_fn(
        logits=logits,
        targets=targets,
        features=features
    )
    
    print("\nLoss components:")
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            print(f"  {loss_name}: {loss_value.item():.4f}")
    
    print("✓ Losses test PASSED\n")


def test_few_shot_pipeline():
    """Test complete few-shot learning pipeline."""
    print("="*60)
    print("Testing Few-Shot Learning Pipeline")
    print("="*60)
    
    model = create_fewshot_spt(num_classes=2, keyframe_ratio=0.3)
    model.eval()
    
    # Create few-shot episode
    support_features = torch.randn(4, 512)  # 2-way, 2-shot
    support_labels = torch.tensor([0, 0, 1, 1])
    
    query_features = torch.randn(10, 512)
    query_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Modality data (required for the model to process)
    video = torch.randn(1, 16, 3, 224, 224)
    audio = torch.randn(1, 16, 128)
    text = torch.randn(1, 16, 768)
    
    support_set = {'features': support_features, 'labels': support_labels}
    query_set = {'features': query_features, 'labels': query_labels}
    
    # Test 1: Classification with modality input
    with torch.no_grad():
        logits = model(
            video_frames=video,
            audio_features=audio,
            text_features=text
        )
    
    print(f"✓ Standard classification logits shape: {logits.shape}")
    
    # Test 2: Few-shot prediction with support/query (using pre-extracted features)
    # In real scenario, these would be extracted from the model's feature space
    with torch.no_grad():
        logits_fs = model.apfsl(
            support_features, support_labels,
            query_features, query_labels,
            adaptive_refinement=True
        )
    
    predictions = torch.argmax(logits_fs, dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    print(f"✓ Few-shot logits shape: {logits_fs.shape}")
    print(f"✓ Few-shot accuracy: {accuracy:.2%}")
    print("✓ Few-shot pipeline test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FewShot-SPT Component Tests")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Component tests
        test_egke()
        test_amg()
        test_perceiver_io()
        test_apfsl()
        
        # Model tests
        test_fewshot_spt()
        test_metrics()
        test_losses()
        test_few_shot_pipeline()
        
        # Summary
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
