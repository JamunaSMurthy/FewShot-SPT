"""
Perceiver IO Attention
Efficient architecture for processing long-range dependencies in multimodal sequences.
Reduces computational complexity while maintaining expressiveness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PerceiverAttention(nn.Module):
    """
    Perceiver-style attention mechanism that processes sequences efficiently.
    Uses latent vectors to attend over input sequences.
    """
    
    def __init__(self, 
                 input_dim: int = 512,
                 num_latents: int = 64,
                 latent_dim: int = 512,
                 num_heads: int = 8,
                 num_blocks: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # Learnable latent vectors
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        nn.init.normal_(self.latents, std=0.02)
        
        # Cross-attention: latents attend to input
        self.cross_attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_blocks)
        ])
        
        # Self-attention: refine latents
        self.self_attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_blocks)
        ])
        
        # Project input to latent dim for attention
        self.input_proj = nn.Linear(input_dim, latent_dim) if input_dim != latent_dim else nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, input_dim) if latent_dim != input_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor, return_latents: bool = False) -> torch.Tensor or Tuple:
        """
        Args:
            x: (B, T, D) - input sequence
            return_latents: if True, return both output and latent vectors
        Returns:
            output: (B, T, D) - attended sequence
            or (output, latents) if return_latents=True
        """
        B, T, D = x.shape
        
        # Project input to latent dimension
        x_projected = self.input_proj(x)  # (B, T, L)
        
        # Expand latents for batch
        latents = self.latents.expand(B, -1, -1)  # (B, N, L)
        
        # Process through blocks
        for cross_attn, self_attn in zip(self.cross_attn_blocks, self.self_attn_blocks):
            # Cross-attention: latents attend to input
            latents = cross_attn(latents)  # (B, N, L)
            
            # Self-attention: refine latents
            latents = self_attn(latents)  # (B, N, L)
        
        # Project latents back to input dimension
        latents_output = self.output_proj(latents)  # (B, N, D)
        
        # Upsample latents to match input length using attention
        if return_latents:
            return latents_output, latents
        
        return latents_output


class PerceiverIOAttention(nn.Module):
    """
    Perceiver IO: Extended Perceiver with output processing.
    Enables structured input/output handling for different modalities.
    """
    
    def __init__(self,
                 input_dim: int = 512,
                 output_dim: int = 512,
                 num_latents: int = 64,
                 latent_dim: int = 512,
                 num_heads: int = 8,
                 num_perceiver_blocks: int = 4,
                 num_output_blocks: int = 2):
        super().__init__()
        
        # Input perceiver
        self.perceiver = PerceiverAttention(
            input_dim=input_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_blocks=num_perceiver_blocks
        )
        
        # Output decoder
        self.output_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_output_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x: torch.Tensor, x_reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_in) - input sequence
            x_reference: (B, T_ref) - optional reference for output positions
        Returns:
            output: (B, T, D_out) - decoded output
        """
        # Process through perceiver
        latents = self.perceiver(x)  # (B, N, D_in)
        
        # Optional: use reference positions for decoding
        if x_reference is not None:
            query = x_reference  # (B, T_ref, D)
        else:
            query = x  # (B, T, D)
        
        # Decode latents to output space
        output = latents
        for decoder in self.output_decoder:
            output = decoder(output, latents)
        
        # Project to output dimension
        output = self.layer_norm(output)
        output = self.output_proj(output)  # (B, N, D_out)
        
        return output


class EfficientTemporalAttention(nn.Module):
    """Efficient temporal attention for long sequences."""
    
    def __init__(self, 
                 dim: int = 512,
                 num_heads: int = 8,
                 local_window: int = 32):
        super().__init__()
        
        self.num_heads = num_heads
        self.local_window = local_window
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Local windowed attention
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create local attention mask
        if self.local_window < T:
            mask = torch.ones(T, T, device=x.device)
            for i in range(T):
                start = max(0, i - self.local_window // 2)
                end = min(T, i + self.local_window // 2)
                mask[i, :start] = 0
                mask[i, end:] = 0
            attention = attention.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(B, T, D)
        output = self.out_proj(output)
        
        return output


class PerceiverIOBlock(nn.Module):
    """Single block of Perceiver IO with residual connections."""
    
    def __init__(self,
                 dim: int = 512,
                 num_heads: int = 8,
                 num_latents: int = 64,
                 latent_dim: int = 512):
        super().__init__()
        
        self.attention = PerceiverIOAttention(
            input_dim=dim,
            output_dim=dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_heads=num_heads
        )
        
        self.norm = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connections where applicable."""
        # Perceiver IO attention
        attn_out = self.attention(self.norm(x))
        
        # Note: Perceiver IO changes sequence length from T to num_latents
        # So we don't use residual connection here
        x = attn_out
        
        # FFN block with residual
        ffn_out = self.ffn(self.norm(x))
        x = x + self.dropout(ffn_out)
        
        return x


class PerceiverIOStack(nn.Module):
    """Stack of Perceiver IO blocks for deep processing."""
    
    def __init__(self,
                 dim: int = 512,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 num_latents: int = 64):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            PerceiverIOBlock(
                dim=dim,
                num_heads=num_heads,
                num_latents=num_latents,
                latent_dim=dim
            ) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through stack of blocks."""
        for block in self.blocks:
            x = block(x)
        return x


# Factory functions
def create_perceiver_io(input_dim: int = 512,
                        output_dim: int = 512,
                        num_latents: int = 64) -> PerceiverIOAttention:
    """Factory function for PerceiverIOAttention."""
    return PerceiverIOAttention(
        input_dim=input_dim,
        output_dim=output_dim,
        num_latents=num_latents,
        latent_dim=512,
        num_heads=8,
        num_perceiver_blocks=4,
        num_output_blocks=2
    )


def create_perceiver_io_stack(dim: int = 512,
                              num_blocks: int = 4) -> PerceiverIOStack:
    """Factory function for PerceiverIOStack."""
    return PerceiverIOStack(
        dim=dim,
        num_heads=8,
        num_blocks=num_blocks,
        num_latents=64
    )
