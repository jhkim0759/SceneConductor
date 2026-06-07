"""
MeshLayout V1_4: Floor Rotation Prediction.

Architecture difference from V3
--------------------------------
V3: shape + pose(RTS) tokens enter transformer → residual addition in 6D space.

V1_4: V3 base + additional floor rotation token (xz2f_rot).
      The model predicts per-object R/T/S in XZ-plane coordinates AND
      a scene-level xz2f_rot (rotation from XZ plane to floor plane).
      During training, R_scene = xz2f_rot @ R_xz, T_scene = xz2f_rot @ T_xz.
"""

from typing import Optional

import torch
import torch.nn as nn

from .mot_sparse_structure_flow import (
    SparseStructureFlowModel,
    SparseStructureFlowTdfyWrapper,
)

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

class SparseStructureFlowTdfyWrapperV1_4(nn.Module):
    """
    V1_4 wrapper: V3 base model + floor rotation token.

    The base model handles shape + R/T/S tokens as usual.
    This wrapper adds an xz2f_rot token that goes through the transformer
    alongside other tokens and predicts a 6D floor rotation.

    Parameters
    ----------
    base_model : SparseStructureFlowTdfyWrapper
        V3-style backbone with shape + RTS latent mappings + xz2f_rot latent.
    """

    def __init__(
        self,
        base_model: SparseStructureFlowTdfyWrapper,
    ):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        latents_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass: base model with floor token.

        Parameters
        ----------
        latents_dict : dict
            Must contain "shape", "translation", "6drotation_normalized", "scale" keys.
            Optionally "xz2f_rot" [N_obj, 1, 6] — if not provided, uses learnable query.

        Returns
        -------
        dict with keys:
            "translation"              : [N_obj, 1, 3]
            "6drotation_normalized"    : [N_obj, 1, 6]
            "scale"                    : [N_obj, 1, 3]
            "xz2f_rot"                : [N_obj, 1, 6]   floor rotation (6D)
        """
        # Forward through base model (which now includes xz2f_rot in latent_mapping)
        outputs = self.base_model(latents_dict, cond, num_parts=num_parts, **kwargs)

        pred_t = outputs["translation"].squeeze(1)            # [N, 3]
        pred_r = outputs["6drotation_normalized"].squeeze(1)   # [N, 6]
        pred_s = outputs["scale"].squeeze(1)                  # [N, 3]
        pred_xz2f = outputs["xz2f_rot"].squeeze(1)            # [N, 6]

        # 2) XZ plane -> scene: R_scene = xz2f @ R_xz, T_scene = xz2f @ T_xz
        xz2f_mat = rotation_6d_to_matrix(pred_xz2f)       # [N, 3, 3]
        abs_r_mat = rotation_6d_to_matrix(pred_r)        # [N, 3, 3]
        scene_r_mat = xz2f_mat @ abs_r_mat                 # [N, 3, 3]
        scene_r_6d = matrix_to_rotation_6d(scene_r_mat)    # [N, 6]
        scene_t = torch.bmm(xz2f_mat, pred_t.unsqueeze(-1)).squeeze(-1)  # [N, 3]

        # 4) Floor rotation guide loss (L1)
        outputs = {
            "translation": scene_t.unsqueeze(1),  # N, 1, 3
            "6drotation_normalized": scene_r_6d.unsqueeze(1),  # N, 1, 6
            "scale": pred_s.unsqueeze(1), # N, 1, 1
            "xz2f_rot": pred_xz2f, # N, 6
            "pred_translation": pred_t.unsqueeze(1),
            "pred_6drotation_normalized": pred_r.unsqueeze(1),
        }

        return outputs

    # ------------------------------------------------------------------
    # Transparent attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
