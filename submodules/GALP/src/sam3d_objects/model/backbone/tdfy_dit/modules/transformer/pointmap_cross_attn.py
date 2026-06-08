"""
Pointmap Cross-Attention module for MeshLayout V2.

Provides:
  - PointmapCrossAttentionLayer : per-latent cross-attention with pointmap tokens

Note: pointmap encoding is handled upstream by EmbedderFuser (via PointPatchEmbed).
The cond_pm tensor arriving here is already (B, N_pm, D) — no raw-pixel encoding needed.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class PointmapCrossAttentionLayer(nn.Module):
    """
    Applies cross-attention from every latent token sequence to pointmap tokens.

    The layer accepts the same *dict* of latent tensors used inside
    SparseStructureFlowModel and returns an updated dict of the same shape.

    Output projections are zero-initialised so the layer starts as an identity,
    allowing safe warm-starting from a V1 checkpoint.

    Args:
        channels     : latent feature dimension
        ctx_channels : pointmap token dimension (EmbedderFuser.embed_dims)
        num_heads    : attention heads
        latent_names : list of latent keys that will be processed
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        latent_names: List[str],
    ):
        super().__init__()
        self.latent_names = latent_names

        self.norm = nn.ModuleDict(
            {name: nn.LayerNorm(channels) for name in latent_names}
        )
        self.cross_attn = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    embed_dim=channels,
                    num_heads=num_heads,
                    kdim=ctx_channels,
                    vdim=ctx_channels,
                    batch_first=True,
                )
                for name in latent_names
            }
        )

        # Zero-init output projections → identity at initialisation
        for name in latent_names:
            nn.init.zeros_(self.cross_attn[name].out_proj.weight)
            nn.init.zeros_(self.cross_attn[name].out_proj.bias)

    def forward(
        self,
        h_dict: Dict[str, torch.Tensor],
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_dict  : {latent_name: (B, T, channels)}
            context : (B, S, ctx_channels) — pre-encoded pointmap tokens

        Returns:
            updated h_dict with same keys and shapes
        """
        result = {}
        for name, h in h_dict.items():
            if name in self.norm:
                normed = self.norm[name](h.float()).to(h.dtype)
                ctx = context.to(h.dtype)
                attn_out, _ = self.cross_attn[name](
                    query=normed,
                    key=ctx,
                    value=ctx,
                )
                result[name] = h + attn_out
            else:
                result[name] = h
        return result
