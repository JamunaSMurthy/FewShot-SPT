"""
Adaptive Modality Gating (AMG)
Dynamically weights and fuses multi-modal features from video, audio, and text.
Uses attention mechanisms to prioritize relevant modalities for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math


class ModalityFeatureProjector(nn.Module):
    """Projects each modality to a common semantic space."""
    
    def __init__(self, input_dims: Dict[str, int], output_dim: int = 512):
        super().__init__()
        self.projectors = nn.ModuleDict()
        
        for modality, dim in input_dims.items():
            self.projectors[modality] = nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project each modality to common space.
        Args:
            features: dict of modality -> (B, T, C) tensors
        Returns:
            projected: dict of modality -> (B, T, D) tensors
        """
        projected = {}
        for modality, feature in features.items():
            if modality in self.projectors:
                B, T, C = feature.shape
                # Reshape for projection
                feature_flat = feature.reshape(B * T, C)
                proj = self.projectors[modality](feature_flat)
                projected[modality] = proj.reshape(B, T, -1)
            else:
                projected[modality] = feature
        return projected


class GatingFunction(nn.Module):
    """Learns modality-specific gating weights."""
    
    def __init__(self, feature_dim: int = 512, num_modalities: int = 3):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Context encoder for computing gating weights
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Gating weight predictors for each modality
        self.gate_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_modalities)
        ])
        
    def forward(self, context: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute gating weights for each modality.
        Args:
            context: (B, D) - context vector
        Returns:
            gates: list of (B, 1) gating weights, one per modality
        """
        encoded = self.context_encoder(context)  # (B, 128)
        
        gates = []
        for predictor in self.gate_predictors:
            gate = torch.sigmoid(predictor(encoded))  # (B, 1)
            gates.append(gate)
        
        return gates


class CrossModalAttention(nn.Module):
    """Computes attention between different modalities."""
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )
    
    def forward(self, query_mod: torch.Tensor, 
                key_mod: torch.Tensor, 
                value_mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_mod: (B, T, D) - query modality
            key_mod: (B, T, D) - key modality
            value_mod: (B, T, D) - value modality
        Returns:
            output: (B, T, D) - refined query modality
        """
        # Self-attention with cross-modal features
        attn_out, _ = self.attention(query_mod, key_mod, value_mod)
        out = self.norm1(query_mod + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out


class AdaptiveModalityGating(nn.Module):
    """
    Adaptive Modality Gating (AMG) module.
    Dynamically fuses multi-modal features through learned gating and attention.
    
    Paper contribution: Efficiently scales to multiple modalities with adaptive weighting.
    """
    
    def __init__(self, 
                 modality_dims: Dict[str, int],
                 output_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.modalities = list(modality_dims.keys())
        self.num_modalities = len(self.modalities)
        self.output_dim = output_dim
        
        # Feature projection
        self.feature_projector = ModalityFeatureProjector(modality_dims, output_dim)
        
        # Gating mechanism
        self.gating_function = GatingFunction(output_dim, self.num_modalities)
        
        # Cross-modal attention layers
        self.cross_modal_attentions = nn.ModuleList([
            CrossModalAttention(output_dim, num_heads)
            for _ in range(self.num_modalities)
        ])
        
        # Modality fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * self.num_modalities, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, features: Dict[str, torch.Tensor],
                return_gates: bool = False,
                return_attention_weights: bool = False) -> torch.Tensor or Tuple:
        """
        Adaptively gate and fuse multi-modal features.
        
        Args:
            features: dict mapping modality name to tensor (B, T, C_modality)
            return_gates: if True, return gating weights
            return_attention_weights: if True, return attention weights
            
        Returns:
            fused_features: (B, T, D) - fused multi-modal features
            or tuple of (fused_features, gates, attention_weights)
        """
        B, T = list(features.values())[0].shape[:2]
        
        # Step 1: Project features to common space
        projected = self.feature_projector(features)  # Dict[str, (B, T, D)]
        
        # Step 2: Compute global context for gating
        # Average features across time for each modality, then concatenate
        all_features = torch.cat([f.mean(dim=1) for f in projected.values()], dim=-1)  # (B, D*M)
        context = all_features  # (B, D*M)
        
        # Project to match gating function input expectation
        if context.shape[-1] > self.output_dim:
            # Reduce dimension by averaging embeddings
            context_reduced = context[:, :self.output_dim]
            # Pad with zeros if needed
            if context_reduced.shape[-1] < self.output_dim:
                padding = torch.zeros(B, self.output_dim - context_reduced.shape[-1], device=context.device)
                context = torch.cat([context_reduced, padding], dim=-1)
            else:
                context = context_reduced
        elif context.shape[-1] < self.output_dim:
            # Pad with zeros if needed
            padding = torch.zeros(B, self.output_dim - context.shape[-1], device=context.device)
            context = torch.cat([context, padding], dim=-1)
        
        # Step 3: Compute modality-specific gating weights
        gates = self.gating_function(context)  # List of (B, 1), one per modality
        
        # Normalize gating weights to sum to 1
        gates_tensor = torch.cat(gates, dim=-1)  # (B, M)
        gates_normalized = F.softmax(gates_tensor / self.temperature, dim=-1)  # (B, M)
        
        # Step 4: Apply gating weights to each modality
        gated_features = {}
        attention_weights_dict = {}
        
        modality_list = list(projected.keys())
        for i, modality in enumerate(modality_list):
            feat = projected[modality]  # (B, T, D)
            gate_weight = gates_normalized[:, i:i+1].unsqueeze(1)  # (B, 1, 1)
            
            # Apply gating
            gated = feat * gate_weight  # (B, T, D)
            
            gated_features[modality] = gated
            attention_weights_dict[modality] = gate_weight.squeeze()
        
        # Step 5: Cross-modal attention refinement
        refined_features = {}
        for i, query_modality in enumerate(modality_list):
            query = gated_features[query_modality]
            
            # Attend to other modalities
            attended = query
            for j, key_modality in enumerate(modality_list):
                if i != j:
                    key = gated_features[key_modality]
                    value = gated_features[key_modality]
                    attended = self.cross_modal_attentions[i](attended, key, value)
            
            refined_features[query_modality] = attended
        
        # Step 6: Multi-modal fusion
        # Concatenate all refined modalities
        fused = torch.cat([refined_features[m] for m in modality_list], dim=-1)  # (B, T, D*M)
        fused = self.fusion_layer(fused)  # (B, T, D)
        
        # Apply residual connection and normalization
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)
        
        # Return based on flags
        if return_gates or return_attention_weights:
            result = [fused]
            if return_gates:
                result.append(gates_normalized)
            if return_attention_weights:
                result.append(attention_weights_dict)
            return tuple(result)
        
        return fused


class MultiModalEncoder(nn.Module):
    """Encodes each modality independently before fusion."""
    
    def __init__(self, modality_configs: Dict[str, Dict]):
        super().__init__()
        self.encoders = nn.ModuleDict()
        
        for modality, config in modality_configs.items():
            input_dim = config.get('input_dim', 512)
            hidden_dim = config.get('hidden_dim', 512)
            num_layers = config.get('num_layers', 2)
            
            # Build encoder as stacked transformer blocks
            encoder = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            
            # Input projection
            self.encoders[modality] = nn.ModuleDict({
                'projector': nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity(),
                'encoder': encoder,
                'layer_norm': nn.LayerNorm(hidden_dim)
            })
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode each modality."""
        encoded = {}
        for modality, feature in features.items():
            if modality in self.encoders:
                proj = self.encoders[modality]['projector']
                enc = self.encoders[modality]['encoder']
                norm = self.encoders[modality]['layer_norm']
                
                # Project and encode
                x = proj(feature)
                x = enc(x)
                encoded[modality] = norm(x)
            else:
                encoded[modality] = feature
        
        return encoded


# Convenience function
def create_amg(modality_dims: Dict[str, int],
               output_dim: int = 512) -> AdaptiveModalityGating:
    """Factory function to create AMG module."""
    return AdaptiveModalityGating(
        modality_dims=modality_dims,
        output_dim=output_dim,
        num_heads=8,
        dropout=0.1
    )
