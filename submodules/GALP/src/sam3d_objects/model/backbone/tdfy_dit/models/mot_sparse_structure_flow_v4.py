"""
MeshLayout V4: Shape-Only Delta Prediction.

Architecture difference from V3
--------------------------------
V3: shape + pose(RTS) tokens enter transformer → residual addition in 6D space.

V4: Only shape tokens enter transformer (pose information baked into voxel via
    init R+S+T applied before voxelization). After transformer, shape tokens are
    mean-pooled and passed through separate heads to predict delta R/T/S.
    Rotation uses matrix composition, scale uses log-scale.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mot_sparse_structure_flow import (
    SparseStructureFlowModel,
    SparseStructureFlowTdfyWrapper,
)


class SparseStructureFlowTdfyWrapperV4(nn.Module):
    """
    V4 wrapper: shape-only backbone + output heads for delta R/T/S.

    Parameters
    ----------
    base_model : SparseStructureFlowTdfyWrapper
        Shape-only backbone (instantiated from ss_generator_v4.yaml).
    model_channels : int
        Hidden dimension of the transformer (1024).
    head_hidden : int
        Hidden dimension of output heads.
    """

    def __init__(
        self,
        base_model: SparseStructureFlowTdfyWrapper,
        model_channels: int = 1024,
        head_hidden: int = 512,
    ):
        super().__init__()
        self.base_model = base_model

        # Output heads: mean-pooled shape → delta predictions
        self.pool_norm = nn.LayerNorm(model_channels)

        self.translation_head = nn.Sequential(
            nn.Linear(model_channels, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(model_channels, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 6),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(model_channels, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),  # log-scale
        )

        self._init_heads()

    def _init_heads(self):
        """Zero-init final layers so model starts near identity delta."""
        for head in (self.translation_head, self.rotation_head, self.scale_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        latents_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass: shape-only backbone → mean pool → delta heads.

        Parameters
        ----------
        latents_dict : dict
            Must contain "shape" key: [N_obj, token_len, in_ch]
        cond : torch.Tensor
            Condition embeddings [B_total?, T_cond, 1024]
        num_parts : torch.Tensor
            Number of objects per scene

        Returns
        -------
        dict with keys:
            "translation"              : [N_obj, 1, 3]   delta translation
            "6drotation_normalized"    : [N_obj, 1, 6]   delta rotation (6D)
            "scale"                    : [N_obj, 1, 1]   delta scale (log)
        """
        # 1. Project input (shape only)
        shape_input = {"shape": latents_dict["shape"]}
        latent = self.base_model.project_input(shape_input)

        # 2. Run transformer backbone (SparseStructureFlowModel.forward)
        shape_out = SparseStructureFlowModel.forward(
            self.base_model, latent, cond, num_parts=num_parts
        )
        # shape_out["shape"]: [N_obj, 512, 1024]

        # 3. Mean pool per-object shape tokens
        shape_feat = shape_out["shape"]  # [N_obj, T, C]
        shape_pooled = self.pool_norm(shape_feat.mean(dim=1))  # [N_obj, C]

        # 4. Predict deltas
        delta_t = self.translation_head(shape_pooled).unsqueeze(1)  # [N_obj, 1, 3]
        delta_r = self.rotation_head(shape_pooled).unsqueeze(1)     # [N_obj, 1, 6]
        delta_s = self.scale_head(shape_pooled).unsqueeze(1)        # [N_obj, 1, 1]

        return {
            "translation": delta_t,
            "6drotation_normalized": delta_r,
            "scale": delta_s,
        }

    # ------------------------------------------------------------------
    # Transparent attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
