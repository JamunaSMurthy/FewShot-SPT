"""
FewShot-SPT: Few-Shot Spatiotemporal Perception Transformer
Main model combining EGKE, AMG, Perceiver IO, and Adaptive Prototypical Few-Shot Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import sys
import os

# Import components
sys.path.insert(0, os.path.dirname(__file__))
from components.egke import EventGuidedKeyframeExtraction
from components.amg import AdaptiveModalityGating, MultiModalEncoder
from components.perceiver_io import PerceiverIOStack
from components.apfsl import AdaptivePrototypicalFewShotLearning


class VideoEncoder(nn.Module):
    """Encodes video frames into feature representations."""
    
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Calculate the size after backbone
        self.backbone_output_dim = 128 * 7 * 7
        
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W)
        Returns:
            features: (B, T, D)
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.reshape(B * T, C, H, W)
        
        features_backbone = self.backbone(frames_flat)
        features_flat = features_backbone.reshape(B * T, -1)
        features = self.projection(features_flat)
        
        return features.reshape(B, T, -1)


class AudioEncoder(nn.Module):
    """Encodes audio features (mel-spectrogram)."""
    
    def __init__(self, input_dim: int = 128, output_dim: int = 512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (B, T, input_dim) - mel-spectrogram features
        Returns:
            features: (B, T, D)
        """
        B, T, D = audio_features.shape
        audio_flat = audio_features.reshape(B * T, D)
        features = self.encoder(audio_flat)
        return features.reshape(B, T, -1)


class TextEncoder(nn.Module):
    """Encodes text features (from pre-trained embeddings)."""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: (B, T, input_dim) - text embeddings
        Returns:
            features: (B, T, D)
        """
        return self.encoder(text_features)


class FewShotSPT(nn.Module):
    """
    Few-Shot Spatiotemporal Perception Transformer (FewShot-SPT)
    Main architecture combining all components for video anomaly detection.
    """
    
    def __init__(self,
                 video_input_shape: Tuple = (3, 224, 224),
                 audio_input_dim: int = 128,
                 text_input_dim: int = 768,
                 feature_dim: int = 512,
                 num_classes: int = 2,
                 keyframe_ratio: float = 0.3,
                 num_modalities: int = 3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        print("Initializing FewShot-SPT components...")
        
        # ===== Modality Encoders =====
        self.video_encoder = VideoEncoder(output_dim=feature_dim)
        self.audio_encoder = AudioEncoder(input_dim=audio_input_dim, output_dim=feature_dim)
        self.text_encoder = TextEncoder(input_dim=text_input_dim, output_dim=feature_dim)
        print(f"✓ Initialized modality encoders")
        
        # ===== Event-Guided Keyframe Extraction =====
        self.egke = EventGuidedKeyframeExtraction(
            feature_dim=feature_dim,
            hidden_dim=512,
            keyframe_ratio=keyframe_ratio,
            window_size=5,
            memory_size=10
        )
        print(f"✓ Initialized EGKE (keyframe_ratio={keyframe_ratio})")
        
        # ===== Adaptive Modality Gating =====
        modality_dims = {
            'video': feature_dim,
            'audio': feature_dim,
            'text': feature_dim
        }
        self.amg = AdaptiveModalityGating(
            modality_dims=modality_dims,
            output_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )
        print(f"✓ Initialized AMG (num_modalities={num_modalities})")
        
        # ===== Perceiver IO Attention =====
        self.perceiver_io = PerceiverIOStack(
            dim=feature_dim,
            num_heads=8,
            num_blocks=4,
            num_latents=64
        )
        print(f"✓ Initialized Perceiver IO")
        
        # ===== Adaptive Prototypical Few-Shot Learning =====
        self.apfsl = AdaptivePrototypicalFewShotLearning(
            feature_dim=feature_dim,
            hidden_dim=256,
            num_refinement_steps=3,
            temperature=0.05,
            decay=0.99
        )
        print(f"✓ Initialized APFSL")
        
        # ===== Classification Head =====
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        print(f"✓ Initialized classification head")
        
        print("✓ FewShot-SPT initialized successfully\n")
    
    def forward(self, 
                video_frames: Optional[torch.Tensor] = None,
                audio_features: Optional[torch.Tensor] = None,
                text_features: Optional[torch.Tensor] = None,
                support_set: Optional[Dict] = None,
                query_set: Optional[Dict] = None,
                return_intermediate: bool = False) -> torch.Tensor or Dict:
        """
        Forward pass for FewShot-SPT.
        
        Args:
            video_frames: (B, T, 3, H, W) - video frames
            audio_features: (B, T, audio_dim) - audio features
            text_features: (B, T, text_dim) - text features
            support_set: dict with 'features' and 'labels' for few-shot
            query_set: dict with 'features' for inference
            return_intermediate: if True, return intermediate representations
        
        Returns:
            logits: (B, num_classes) classification logits
            or dict with intermediate representations
        """
        
        # ===== Step 1: Encode each modality =====
        encoded_modalities = {}
        B = None
        
        if video_frames is not None:
            video_features = self.video_encoder(video_frames)  # (B, T, D)
            encoded_modalities['video'] = video_features
            B = video_frames.shape[0]
        
        if audio_features is not None:
            audio_encoded = self.audio_encoder(audio_features)  # (B, T, D)
            encoded_modalities['audio'] = audio_encoded
            B = audio_features.shape[0]
        
        if text_features is not None:
            text_encoded = self.text_encoder(text_features)  # (B, T, D)
            encoded_modalities['text'] = text_encoded
            B = text_features.shape[0]
        
        # ===== Step 2: Event-Guided Keyframe Extraction =====
        if 'video' in encoded_modalities:
            video_features = encoded_modalities['video']
            keyframes, keyframe_indices, anomaly_scores = self.egke(
                video_features,
                return_indices=True,
                return_scores=True
            )  # keyframes: (B, K, D), anomaly_scores: (B, T, 1)
            
            # Update other modalities to match keyframe selection
            for modality in encoded_modalities:
                if modality != 'video':
                    # Select corresponding frames for other modalities
                    mod_keyframes = []
                    for b in range(B):
                        indices = keyframe_indices[b]
                        selected = encoded_modalities[modality][b, indices, :]
                        mod_keyframes.append(selected)
                    
                    # Pad to same length
                    max_len = max(k.shape[0] for k in mod_keyframes)
                    for i, k in enumerate(mod_keyframes):
                        if k.shape[0] < max_len:
                            pad = torch.zeros(max_len - k.shape[0], self.feature_dim,
                                            device=k.device)
                            mod_keyframes[i] = torch.cat([k, pad], dim=0)
                    
                    encoded_modalities[modality] = torch.stack(mod_keyframes)
            
            encoded_modalities['video'] = keyframes
        
        # ===== Step 3: Adaptive Modality Gating =====
        fused_features, gates = self.amg(
            encoded_modalities,
            return_gates=True
        )  # (B, K, D) or (B, T, D)
        
        # ===== Step 4: Perceiver IO Spatiotemporal Attention =====
        refined_features = self.perceiver_io(fused_features)  # (B, K, D) or (B, T, D)
        
        # ===== Step 5: Aggregate temporal information =====
        # Use mean and max pooling
        mean_features = refined_features.mean(dim=1)  # (B, D)
        max_features = refined_features.max(dim=1)[0]  # (B, D)
        aggregated = (mean_features + max_features) / 2  # (B, D)
        
        # ===== Step 6: Few-shot classification if support/query provided =====
        if support_set is not None and query_set is not None:
            logits = self.apfsl(
                support_set['features'],
                support_set['labels'],
                query_set['features'],
                query_set.get('labels', None),
                adaptive_refinement=True
            )
        else:
            # Standard classification
            logits = self.classification_head(aggregated)  # (B, num_classes)
        
        # ===== Return based on flags =====
        if return_intermediate:
            return {
                'logits': logits,
                'video_features': encoded_modalities.get('video', None),
                'fused_features': fused_features,
                'refined_features': refined_features,
                'anomaly_scores': anomaly_scores if 'video' in encoded_modalities else None,
                'gating_weights': gates,
                'aggregated_features': aggregated
            }
        
        return logits
    
    def extract_features(self, 
                        video_frames: torch.Tensor,
                        audio_features: Optional[torch.Tensor] = None,
                        text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract feature representation without classification."""
        intermediate = self.forward(
            video_frames=video_frames,
            audio_features=audio_features,
            text_features=text_features,
            return_intermediate=True
        )
        return intermediate['aggregated_features']
    
    def get_anomaly_scores(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Get anomaly scores for each frame."""
        intermediate = self.forward(
            video_frames=video_frames,
            return_intermediate=True
        )
        return intermediate['anomaly_scores']


# Factory function
def create_fewshot_spt(num_classes: int = 2,
                       keyframe_ratio: float = 0.3) -> FewShotSPT:
    """Factory function to create FewShot-SPT model."""
    return FewShotSPT(
        video_input_shape=(3, 224, 224),
        audio_input_dim=128,
        text_input_dim=768,
        feature_dim=512,
        num_classes=num_classes,
        keyframe_ratio=keyframe_ratio,
        num_modalities=3
    )


if __name__ == "__main__":
    # Test the model
    print("Testing FewShot-SPT model...\n")
    
    model = create_fewshot_spt(num_classes=2, keyframe_ratio=0.3)
    model.eval()
    
    # Create dummy inputs
    batch_size, num_frames = 2, 16
    video = torch.randn(batch_size, num_frames, 3, 224, 224)
    audio = torch.randn(batch_size, num_frames, 128)
    text = torch.randn(batch_size, num_frames, 768)
    
    print("Input shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Text: {text.shape}\n")
    
    # Forward pass
    with torch.no_grad():
        output = model(
            video_frames=video,
            audio_features=audio,
            text_features=text,
            return_intermediate=True
        )
    
    print("Output shapes:")
    for key in output:
        if output[key] is not None:
            if isinstance(output[key], torch.Tensor):
                print(f"  {key}: {output[key].shape}")
            elif isinstance(output[key], dict):
                print(f"  {key}: dict with keys {output[key].keys()}")
    
    print("\n✓ Model test passed!")
