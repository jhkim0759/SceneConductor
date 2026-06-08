"""
MeshLayout V3: Translation-Guided Attention.

Architecture difference from V1
--------------------------------
V1: global_attention uses standard SDPA across all object tokens.

V3: global_attention adds a learnable additive bias computed from
    translation token attention patterns.  The bias captures spatial
    proximity relationships between objects (via bbox centres) and
    modulates all token interactions in the global attention step.

Design goals
------------
- V1 checkpoint loads into base_model without modification.
- Only ``trans_bias_alpha`` parameters (one per global-attn block) are new;
  zero-initialised so training starts from the V1 fixed point.
- No extra condition input needed (unlike V2's cond_pm).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils import _pytree

from .mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
from ..modules.attention.modules import MOTMultiHeadSelfAttentionV3


class SparseStructureFlowTdfyWrapperV3(nn.Module):
    """
    Wraps a V1 :class:`SparseStructureFlowTdfyWrapper` and replaces the
    ``self_attn`` modules on *global-attention* blocks with
    :class:`MOTMultiHeadSelfAttentionV3`.

    The wrapper adds per-block learnable ``alpha`` scalars (zero-init) that
    control the strength of the translation-derived attention bias.

    Parameters
    ----------
    base_model               : pre-built V1 wrapper (weights already loaded)
    translation_token_offset : index of translation within the merged latent
                               group  (default 1: [rot=0, trans=1, scale=2])
    merged_latent_key        : name of the merged latent key that contains
                               translation (default ``"6drotation_normalized"``)
    """

    def __init__(
        self,
        base_model: SparseStructureFlowTdfyWrapper,
        translation_token_offset: int = 1,
        merged_latent_key: str = "6drotation_normalized",
    ):
        super().__init__()
        self.base_model = base_model
        self.translation_token_offset = translation_token_offset
        self.merged_latent_key = merged_latent_key

        # Compute translation token position (will be recomputed after set_protect_modality)
        self._trans_token_pos = self._compute_trans_token_pos()

        # Create per-global-block alpha and patch self_attn
        self._patched_attns = []  # track patched attns for recompute
        self.trans_bias_alpha = nn.ParameterList()
        for block in base_model.blocks:
            if block.self_attn.use_global_attn:
                alpha = nn.Parameter(torch.zeros(1))
                self.trans_bias_alpha.append(alpha)
                self._replace_self_attn(block, alpha)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_trans_token_pos(self) -> int:
        """
        Compute the position of the translation token in the concatenated
        per-object token sequence, **excluding protected modalities**.

        When ``set_protect_modality(["shape"])`` is active, shape tokens are
        handled via local attention and are not passed to global_attention.
        This method must return the position relative to the non-protected
        tokens that global_attention actually receives.
        """
        base = self.base_model
        # protect_modality_list is set on each block's self_attn, not on the model
        protect_list = getattr(base.blocks[0].self_attn, "protect_modality_list", [])
        pos = 0
        for name in base.latent_names:
            # Skip protected modalities (they don't enter global_attention)
            if name in protect_list:
                continue

            if name == self.merged_latent_key:
                # Within the merged group, translation is at the given offset
                pos += self.translation_token_offset
                return pos
            elif name == "translation":
                # latent_share_transformer disabled — translation is standalone
                return pos
            else:
                # Count tokens for this latent
                if name in base.latent_share_transformer:
                    token_len = sum(
                        base.latent_mapping[sub].pos_emb.shape[0]
                        for sub in base.latent_share_transformer[name]
                    )
                elif name in base.latent_mapping:
                    token_len = base.latent_mapping[name].pos_emb.shape[0]
                else:
                    raise ValueError(f"Unknown latent name: {name}")
                pos += token_len

        raise ValueError(
            f"Could not find translation token in latent_names: "
            f"{base.latent_names}"
        )

    def _replace_self_attn(self, block, alpha: nn.Parameter):
        """
        Replace a block's ``self_attn`` with V3 version by changing the
        instance's class to :class:`MOTMultiHeadSelfAttentionV3` and
        injecting the translation-guided attributes.

        This avoids parameter copying — the existing weights are kept
        in-place and only the ``global_attention`` method is overridden
        via the new class.
        """
        old_attn = block.self_attn
        old_attn.__class__ = MOTMultiHeadSelfAttentionV3
        old_attn.trans_token_pos = self._trans_token_pos
        old_attn._alpha = alpha
        self._patched_attns.append(old_attn)

    def set_protect_modality(self, protect_list: list):
        """Override to recompute trans_token_pos after protect list changes."""
        self.base_model.set_protect_modality(protect_list)
        self._trans_token_pos = self._compute_trans_token_pos()
        for attn in self._patched_attns:
            attn.trans_token_pos = self._trans_token_pos

    # ------------------------------------------------------------------
    # Forward — identical interface to V1
    # ------------------------------------------------------------------

    def forward(
        self,
        latents_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Same forward interface as V1.  No additional condition inputs.
        The translation-guided bias is computed internally from the
        translation tokens' Q and K.
        """
        return self.base_model(latents_dict, cond, num_parts=num_parts)

    # ------------------------------------------------------------------
    # Transparent attribute forwarding (mirrors V2 pattern)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
