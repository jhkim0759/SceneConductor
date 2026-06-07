"""
MeshLayout V2: SparseStructureFlowTdfyWrapper with dual cross-attention.

Architecture difference from V1
--------------------------------
V1: each transformer block cross-attends to ONE concatenated condition tensor
    (image tokens + pointmap tokens merged into a single (B, N, D) tensor).

V2: the two modalities are kept *separate* and each transformer block
    (a) still cross-attends to image tokens  via the existing block.cross_attn
    (b) additionally cross-attends to pointmap tokens via a new
        PointmapCrossAttentionLayer inserted after every block.

How conditions are split
------------------------
train.py uses ``ConditionEmbeddingSplit`` which calls EmbedderFuser with
``return_dict=True``, groups the per-key token tensors by modality, and
concatenates within each group:
  - ``cond``    : image + mask tokens  (B, N_img, D)
  - ``cond_pm`` : pointmap tokens      (B, N_pm, D)

Both have the same feature dimension D (= EmbedderFuser.embed_dims, typically 1024).

Design goals
------------
- V1 checkpoint loads into base_model without modification.
- Only pm_layers are new; PointmapCrossAttentionLayer output projections are
  zero-initialised so training starts from the V1 fixed point.
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.utils import _pytree

from .mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
from ..modules.transformer.pointmap_cross_attn import PointmapCrossAttentionLayer


class SparseStructureFlowTdfyWrapperV2(nn.Module):
    """
    Wraps a V1 SparseStructureFlowTdfyWrapper and adds per-block
    pointmap cross-attention.

    Parameters
    ----------
    base_model   : pre-built V1 wrapper (weights can be loaded from a V1 ckpt)
    pm_cond_dim  : feature dimension of the pointmap condition tokens
                   (must match EmbedderFuser.embed_dims, default 1024)
    """

    def __init__(
        self,
        base_model: SparseStructureFlowTdfyWrapper,
        pm_cond_dim: int = 1024,
    ):
        super().__init__()
        self.base_model = base_model

        channels = base_model.model_channels
        num_heads = base_model.num_heads
        num_blocks = base_model.num_blocks

        # Exclude input-only latents (e.g. "shape") from pm_layers.
        # project_output() never uses them, so their pm_layer parameters would
        # receive no gradient in the final block → DDP unused-parameter error.
        # Pointmap cross-attention only needs to update the output (pose) latents.
        pm_latent_names = [n for n in base_model.latent_names if n != "shape"]

        self.pm_layers = nn.ModuleList(
            [
                PointmapCrossAttentionLayer(
                    channels=channels,
                    ctx_channels=pm_cond_dim,
                    num_heads=num_heads,
                    latent_names=pm_latent_names,
                )
                for _ in range(num_blocks)
            ]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        latents_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        cond_pm: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Parameters
        ----------
        latents_dict : {"shape": (N, T, C), ...}
        cond         : (N, N_img, D) image + mask condition tokens
        num_parts    : (B,) part count per scene
        cond_pm      : (N, N_pm, D) pointmap condition tokens;
                       if None the model behaves identically to V1
        """
        base = self.base_model

        # ---- project input latents (mirrors SparseStructureFlowTdfyWrapper) ----
        latent_dict = base.project_input(latents_dict)

        # ---- dtype cast (mirrors SparseStructureFlowModel.forward) ----
        input_dtype = next(iter(latent_dict.values())).dtype
        latent_dict = _pytree.tree_map(lambda t: t.type(base.dtype), latent_dict)
        if cond is not None:
            cond = cond.type(base.dtype)
        if cond_pm is not None:
            cond_pm = cond_pm.type(base.dtype)

        # ---- main blocks with interleaved pointmap cross-attention ----
        for block, pm_layer in zip(base.blocks, self.pm_layers):
            # V1 cross-attention: image tokens → latents
            latent_dict = block(latent_dict, None, cond, num_parts=num_parts)
            # V2 extra cross-attention: pointmap tokens → latents
            if cond_pm is not None:
                latent_dict = pm_layer(latent_dict, cond_pm)

        # ---- restore dtype ----
        latent_dict = _pytree.tree_map(lambda t: t.type(input_dtype), latent_dict)

        # ---- project output latents ----
        output_latents = base.project_output(latent_dict)
        return output_latents

    # ------------------------------------------------------------------
    # Transparent attribute forwarding so existing callers work unchanged
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
