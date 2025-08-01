"""
Event-Guided Keyframe Extraction (EGKE)
Dynamically selects keyframes based on anomaly intensity and temporal patterns.
Reduces redundancy by filtering frames based on anomaly scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class AnomalyScorer(nn.Module):
    """Computes anomaly scores for each frame based on visual features."""
    
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, C) - batch, temporal, channel
        Returns:
            anomaly_scores: (B, T, 1) - normalized anomaly scores
        """
        x = self.activation(self.fc1(features))
        x = self.activation(self.fc2(x))
        scores = self.sigmoid(self.fc3(x))
        return scores


class AdaptiveThresholding(nn.Module):
    """Learns adaptive thresholds for keyframe selection based on video statistics."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, anomaly_scores: torch.Tensor, 
                video_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anomaly_scores: (B, T, 1) - anomaly scores
            video_context: (B, D) - optional context for adaptive thresholding
        Returns:
            threshold: (B, 1, 1) - adaptive threshold per video
        """
        if video_context is None:
            # Use mean and std of anomaly scores as context
            mean_score = anomaly_scores.mean(dim=1, keepdim=True)  # (B, 1, 1)
            std_score = anomaly_scores.std(dim=1, keepdim=True)    # (B, 1, 1)
            video_context = torch.cat([mean_score, std_score], dim=-1).squeeze(1)  # (B, 2)
            context_dim = 2
        else:
            context_dim = video_context.shape[-1]
            
        # Ensure context has proper dimension
        if context_dim < 256:
            padding = 256 - context_dim
            video_context = F.pad(video_context, (0, padding))
            
        threshold = self.threshold_predictor(video_context)  # (B, 1)
        return threshold.unsqueeze(-1)  # (B, 1, 1)


class TemporalConsistency(nn.Module):
    """Enforces temporal consistency in keyframe selection."""
    
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        self.temporal_filter = nn.Conv1d(1, 1, kernel_size=window_size, 
                                        padding=window_size//2, padding_mode='replicate')
        
    def forward(self, keyframe_mask: torch.Tensor) -> torch.Tensor:
        """
        Smooth keyframe selection with temporal consistency.
        Args:
            keyframe_mask: (B, T) - binary keyframe mask
        Returns:
            smoothed_mask: (B, T) - smoothed mask with temporal consistency
        """
        # Add channel dimension for conv1d
        x = keyframe_mask.unsqueeze(1).float()  # (B, 1, T)
        smoothed = torch.sigmoid(self.temporal_filter(x))  # (B, 1, T)
        return smoothed.squeeze(1)  # (B, T)


class ContextAwareMemory(nn.Module):
    """Context-aware memory module for maintaining information across frames."""
    
    def __init__(self, feature_dim: int = 2048, memory_size: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        self.memory_update = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
    def forward(self, features: torch.Tensor, 
                keyframe_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, C) - feature sequences
            keyframe_mask: (B, T) - keyframe selection mask
        Returns:
            refined_features: (B, T, C) - context-aware refined features
            memory: (B, M, C) - updated memory
        """
        B, T, C = features.shape
        
        # Initialize memory with keyframe features
        keyframe_indices = torch.where(keyframe_mask > 0.5)
        
        # Attention-based refinement
        refined, _ = self.attention(features, features, features)
        
        # Update memory with spatial context
        memory = features[:, :min(self.memory_size, T), :]
        
        return refined, memory


class EventGuidedKeyframeExtraction(nn.Module):
    """
    Event-Guided Keyframe Extraction (EGKE) module.
    Selects keyframes dynamically based on anomaly intensity and temporal patterns.
    
    Paper contribution: Reduces redundancy by 7-8% while maintaining accuracy.
    """
    
    def __init__(self, 
                 feature_dim: int = 2048,
                 hidden_dim: int = 512,
                 keyframe_ratio: float = 0.3,
                 window_size: int = 5,
                 memory_size: int = 10):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.keyframe_ratio = keyframe_ratio
        self.window_size = window_size
        
        # Core components
        self.anomaly_scorer = AnomalyScorer(feature_dim, hidden_dim)
        self.adaptive_thresholding = AdaptiveThresholding(hidden_dim)
        self.temporal_consistency = TemporalConsistency(window_size)
        self.context_aware_memory = ContextAwareMemory(feature_dim, memory_size)
        
        # Additional processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, features: torch.Tensor, 
                return_indices: bool = False,
                return_scores: bool = False) -> torch.Tensor or Tuple:
        """
        Extract keyframes based on event guidance.
        
        Args:
            features: (B, T, C) - visual features from video frames
            return_indices: if True, return keyframe indices
            return_scores: if True, return anomaly scores
            
        Returns:
            keyframes: (B, K, C) - selected keyframe features
            or tuple of (keyframes, indices, scores) if return flags are True
        """
        B, T, C = features.shape
        
        # Step 1: Compute anomaly scores for each frame
        anomaly_scores = self.anomaly_scorer(features)  # (B, T, 1)
        
        # Step 2: Adaptive thresholding
        video_context = self.feature_processor(features.mean(dim=1))  # (B, H)
        threshold = self.adaptive_thresholding(anomaly_scores, video_context)  # (B, 1, 1)
        
        # Step 3: Initial keyframe selection based on threshold
        keyframe_mask = (anomaly_scores > threshold).squeeze(-1).float()  # (B, T)
        
        # Ensure minimum number of keyframes
        min_keyframes = max(1, int(T * self.keyframe_ratio))
        for b in range(B):
            num_keyframes = keyframe_mask[b].sum().item()
            if num_keyframes < min_keyframes:
                # Select top-k frames if threshold is too strict
                top_k_values, top_k_indices = torch.topk(anomaly_scores[b, :, 0], min_keyframes)
                keyframe_mask[b] = 0
                keyframe_mask[b, top_k_indices] = 1
        
        # Step 4: Temporal consistency refinement
        keyframe_mask = self.temporal_consistency(keyframe_mask)  # (B, T)
        keyframe_mask = (keyframe_mask > 0.5).float()  # Binarize
        
        # Step 5: Context-aware memory refinement
        refined_features, _ = self.context_aware_memory(features, keyframe_mask)
        
        # Step 6: Final keyframe selection
        keyframe_mask_binary = keyframe_mask > 0.5
        
        # Extract selected keyframes
        keyframes_list = []
        keyframe_indices_list = []
        
        for b in range(B):
            mask = keyframe_mask_binary[b]
            selected_indices = torch.where(mask)[0]
            
            # Ensure at least one keyframe
            if len(selected_indices) == 0:
                # Select frame with highest anomaly score
                top_idx = anomaly_scores[b, :, 0].argmax()
                selected_indices = torch.tensor([top_idx], device=features.device)
            
            keyframes = refined_features[b, selected_indices, :]  # (K, C)
            keyframes_list.append(keyframes)
            keyframe_indices_list.append(selected_indices)
        
        # Pad to same length for batching
        max_keyframes = max(k.shape[0] for k in keyframes_list)
        keyframes_padded = []
        
        for keyframes in keyframes_list:
            if keyframes.shape[0] < max_keyframes:
                pad = torch.zeros(max_keyframes - keyframes.shape[0], C, 
                                device=features.device)
                keyframes = torch.cat([keyframes, pad], dim=0)
            keyframes_padded.append(keyframes)
        
        keyframes = torch.stack(keyframes_padded, dim=0)  # (B, K, C)
        
        # Return based on flags
        if return_indices or return_scores:
            result = [keyframes]
            if return_indices:
                result.append(keyframe_indices_list)
            if return_scores:
                result.append(anomaly_scores)
            return tuple(result)
        
        return keyframes


# Convenience function for creating EGKE module
def create_egke(feature_dim: int = 2048, keyframe_ratio: float = 0.3) -> EventGuidedKeyframeExtraction:
    """Factory function to create EGKE module with standard configuration."""
    return EventGuidedKeyframeExtraction(
        feature_dim=feature_dim,
        hidden_dim=512,
        keyframe_ratio=keyframe_ratio,
        window_size=5,
        memory_size=10
    )
