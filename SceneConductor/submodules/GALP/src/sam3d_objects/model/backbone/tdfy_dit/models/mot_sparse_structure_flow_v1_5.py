"""
MeshLayout V1_5: Unified transformer norm/linear (shape + pose concat single stream).

Architecture difference from V1_4
---------------------------------
V1_4: Inside each transformer block, norm1/norm2/norm3/cross_attn/mlp are
      per-modality ``nn.ModuleDict`` keyed by latent name ("shape",
      "6drotation_normalized"). self_attn is already shared (MOT variant).

V1_5: Those per-modality modules are merged into **single** unified modules.
      Shape tokens and merged-pose tokens are concatenated along the sequence
      dimension and flow through the transformer as a single stream.

      Input/output projections (``latent_mapping``) remain per-modality because
      raw dims differ (shape:8, 6drot:6, trans:3, scale:3, xz2f:6).  The
      unification is specifically inside the transformer blocks.

      The floor-rotation wrapper (xz2f @ R_xz, xz2f @ T_xz) is identical to
      V1_4 and composed on top of the base V1_5 model.
"""

from functools import partial
from typing import *

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.sam3d_objects.data.utils import tree_reduce_unique

from ..modules.attention import MOTMultiHeadSelfAttention, MultiHeadAttention
from ..modules.norm import LayerNorm32
from ..modules.transformer.blocks import FeedForwardNet
from ..modules.utils import FP16_TYPE, convert_module_to_f16, convert_module_to_f32
from .mot_sparse_structure_flow import _identity_condition_embedder


# --------------------------------------------------------------------------- #
#  Unified transformer block
# --------------------------------------------------------------------------- #
class UnifiedModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm
    conditioning. Single unified norm/cross_attn/mlp modules operating on a
    single concatenated [B, T, C] stream.latent_dict["shape"].sh

    Self-attention keeps the MOT variant so that the ``use_global_attn`` /
    ``num_parts`` reshape-by-scene logic is preserved; it is constructed with
    a single ``latent_names=["concat"]`` key so it effectively operates on the
    single concatenated stream.

    Note: the ``is_last_block`` optimization in V1_4 (dropping shape's
    norm2/cross_attn/mlp at the last block) does not apply here because there
    is no per-modality split inside the block — all unified modules are
    present at every block.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        latent_names: List = None,  # unused — kept for API compatibility
        freeze_shared_parameters: bool = False,
        use_global_attn: bool = False,
        is_last_block: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.is_last_block = is_last_block

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)

        # Self-attention: keep MOT variant with a single "concat" key so the
        # num_parts / global_attention path is preserved verbatim.
        self.self_attn = MOTMultiHeadSelfAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            latent_names=["concat"],
            protect_modality_list=[],
            use_global_attn=use_global_attn,
            is_last_block=False,  # force all modules present
        )

        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )

        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
            if freeze_shared_parameters:
                self.adaLN_modulation.eval()
                self.adaLN_modulation.requires_grad_(False)

        self.learnable_mod = nn.Parameter(torch.randn(1, channels))
        self.store_attn_map = False

    def get_attn_maps(self):
        maps = {}
        if self.cross_attn is not None and self.cross_attn.last_attn_map is not None:
            maps["cross_attn"] = {"concat": self.cross_attn.last_attn_map}
        if self.self_attn.last_attn_map is not None:
            if self.self_attn.use_global_attn:
                maps["global_self_attn"] = self.self_attn.last_attn_map
            else:
                maps["local_self_attn"] = self.self_attn.last_attn_map
        return maps

    def _forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Propagate store_attn_map flag.
        self.self_attn.store_attn_map = self.store_attn_map
        self.cross_attn.store_attn_map = self.store_attn_map

        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )

        # Self-attention branch
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h_dict = self.self_attn({"concat": h}, num_parts=num_parts)
        h = h_dict["concat"]
        h = h * gate_msa.unsqueeze(1)
        x = x + h

        # Cross-attention branch (no gate, matching V1_4 MOT block pattern)
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h

        # FFN branch
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h

        return x

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mod = self.learnable_mod.repeat(context.shape[0], 1)
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, num_parts, use_reentrant=False
            )
        else:
            return self._forward(x, mod, context, num_parts)


# --------------------------------------------------------------------------- #
#  Base V1_5 flow model (operates on a single [B, T, C] tensor)
# --------------------------------------------------------------------------- #
class SparseStructureFlowModelV1_5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        freeze_shared_parameters: bool = False,
        is_shortcut_model: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = FP16_TYPE if use_fp16 else torch.float32
        self.is_shortcut_model = is_shortcut_model

        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.blocks = nn.ModuleList(
            [
                UnifiedModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                    latent_names=self.latent_names,
                    freeze_shared_parameters=freeze_shared_parameters,
                    use_global_attn=idx % 2 == 0,
                    is_last_block=(idx == num_blocks - 1),
                )
                for idx in range(num_blocks)
            ]
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _cast_type(self, x, dtype):
        return x.type(dtype)

    def forward(
        self,
        h: torch.Tensor,
        cond: torch.Tensor,
        num_parts: Optional[torch.Tensor] = None,
        store_attn: bool = False,
    ) -> torch.Tensor:
        input_dtype = h.dtype
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        if store_attn:
            for block in self.blocks:
                block.store_attn_map = True

        for block in self.blocks:
            h = block(h, None, cond, num_parts=num_parts)

        attn_maps = None
        if store_attn:
            attn_maps = {
                "cross_attn": [],
                "global_self_attn": [],
                "local_self_attn": [],
            }
            for block in self.blocks:
                block_maps = block.get_attn_maps()
                attn_maps["cross_attn"].append(block_maps.get("cross_attn", {}))
                attn_maps["global_self_attn"].append(
                    block_maps.get("global_self_attn", None)
                )
                attn_maps["local_self_attn"].append(
                    block_maps.get("local_self_attn", None)
                )
                block.store_attn_map = False

        h = h.type(input_dtype)

        if store_attn:
            return h, attn_maps
        return h


# --------------------------------------------------------------------------- #
#  TdfyWrapper V1_5: unified transformer stream + per-modality in/out proj
# --------------------------------------------------------------------------- #
class SparseStructureFlowTdfyWrapperV1_5(SparseStructureFlowModelV1_5):
    def __init__(
        self,
        latent_mapping: dict,
        latent_share_transformer: dict = {},
        *args,
        **kwargs,
    ):
        condition_embedder = kwargs.pop("condition_embedder", None)
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        kwargs.pop("shape_attend_pose", None)

        # Single unified stream inside the transformer.
        self.latent_names = ["concat"]

        super().__init__(*args, **kwargs)

        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = _identity_condition_embedder
        self.force_zeros_cond = force_zeros_cond

        self.latent_mapping = nn.ModuleDict(latent_mapping)
        if not isinstance(latent_share_transformer, dict):
            self.latent_share_transformer = OmegaConf.to_container(
                latent_share_transformer
            )
        else:
            self.latent_share_transformer = latent_share_transformer
        self.input_latent_mappings = list(self.latent_mapping.keys())

        # Filled in during project_input; used by project_output.
        self._split_lengths: Optional[Dict[str, int]] = None

    def set_protect_modality(self, protect_list: list):
        # No-op for v1_5: there is no per-modality split inside the transformer.
        for block in self.blocks:
            block.self_attn.protect_modality_list = []

    def forward(
        self,
        input_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        store_attn: bool = False,
        **condition_kwargs,
    ) -> dict:
        latent_tensor = self.project_input(input_dict)
        result = super().forward(
            latent_tensor, cond, num_parts=num_parts, store_attn=store_attn
        )

        if store_attn:
            output, attn_maps = result
        else:
            output = result

        output_latents = self.project_output(output)

        for key in output_latents:
            if key in input_dict and not key == "num_parts":
                output_latents[key] = output_latents[key] + input_dict[key]

        if store_attn:
            return output_latents, attn_maps
        return output_latents

    # ------------------------------------------------------------------ #
    #  Projection helpers
    # ------------------------------------------------------------------ #
    def project_input(
        self,
        latents_dict: Dict,
    ) -> Dict:
        # concatenate input from multiple modalities
        latent_dict = {}
        latent_dict["shape"] = self.latent_mapping["shape"].to_input(latents_dict["shape"])

        BATCH = latent_dict["shape"].shape[0]

        for latent_name in self.input_latent_mappings:
            if latent_name == "shape":
                continue
            elif latent_name in latents_dict:
                # V3: encode provided initial RTS via Latent.to_input()
                latent_dict[latent_name] = self.latent_mapping[latent_name].to_input(latents_dict[latent_name])
            else:
                # V1: use learnable query (no RTS input provided)
                learnable_query = self.latent_mapping[latent_name].pos_emb.unsqueeze(0).expand(BATCH, -1, -1)
                latent_dict[latent_name] = learnable_query

        # Merge pose modalities (mirrors V1_4 logic).
        merged_dict = self._merge_latent_share_transformer(latent_dict)

        # Concatenate every merged entry (shape + pose-merged) along tokens.
        # Ordering: follow insertion order of merged_dict for determinism.
        split_lengths: Dict[str, int] = {}
        tensors: List[torch.Tensor] = []
        for name, tensor in merged_dict.items():
            split_lengths[name] = tensor.shape[1]
            tensors.append(tensor)
        self._split_lengths = split_lengths
        concat = torch.cat(tensors, dim=1)
        return concat

    def project_output(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Split the unified output tensor back into the merged blocks, then into
        per-modality tokens, then run each modality's ``to_output()``.
        """
        assert (
            self._split_lengths is not None
        ), "project_input must be called before project_output"

        # 1) Split unified output back into merged-dict form.
        merged_output: Dict[str, torch.Tensor] = {}
        start = 0
        for name, token_len in self._split_lengths.items():
            merged_output[name] = output[:, start : start + token_len]
            start += token_len

        # 2) Split merged pose entries back into per-modality entries.
        split_output = self._split_latent_share_transformer(merged_output)

        # 3) Run per-modality ``to_output`` (skip shape, matching V1_4).
        output_latents: Dict[str, torch.Tensor] = {}
        for latent_name in self.input_latent_mappings:
            if latent_name == "shape":
                continue
            latent = self.latent_mapping[latent_name].to_output(
                split_output[latent_name]
            )
            output_latents[latent_name] = latent

        return output_latents

    def _merge_latent_share_transformer(self, latent_dict):
        visited = set()
        return_dict = {}
        for merged_name, latent_names in self.latent_share_transformer.items():
            tensors = []
            for latent_name in latent_names:
                visited.add(latent_name)
                tensors.append(latent_dict[latent_name])
            return_dict[merged_name] = torch.cat(tensors, dim=1)

        for latent_name in latent_dict:
            if latent_name not in visited:
                return_dict[latent_name] = latent_dict[latent_name]

        return return_dict

    def _split_latent_share_transformer(self, output_latents):
        return_dict = {}
        visited = set()
        for merged_name, latent_names in self.latent_share_transformer.items():
            start = 0
            visited.add(merged_name)
            tensors = output_latents[merged_name]
            for latent_name in latent_names:
                token_len = self.latent_mapping[latent_name].pos_emb.shape[0]
                return_dict[latent_name] = tensors[:, start : start + token_len]
                start += token_len

        for latent_name in output_latents:
            if latent_name not in visited:
                return_dict[latent_name] = output_latents[latent_name]

        return return_dict


# --------------------------------------------------------------------------- #
#  Floor-rotation wrapper (mirrors V1_4)
# --------------------------------------------------------------------------- #
class SparseStructureFlowTdfyWrapperV1_5Floor(nn.Module):
    """
    V1_5 Floor wrapper: unified-transformer base model + floor rotation
    composition. Logic identical to V1_4's wrapper — only the base model type
    differs.
    """

    def __init__(self, base_model: SparseStructureFlowTdfyWrapperV1_5):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        latents_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        from pytorch3d.transforms import (
            matrix_to_rotation_6d,
            rotation_6d_to_matrix,
        )

        outputs = self.base_model(latents_dict, cond, num_parts=num_parts, **kwargs)

        pred_t = outputs["translation"].squeeze(1)
        pred_r = outputs["6drotation_normalized"].squeeze(1)
        pred_s = outputs["scale"].squeeze(1)
        pred_xz2f = outputs["xz2f_rot"].squeeze(1)

        xz2f_mat = rotation_6d_to_matrix(pred_xz2f)
        abs_r_mat = rotation_6d_to_matrix(pred_r)
        scene_r_mat = xz2f_mat @ abs_r_mat
        scene_r_6d = matrix_to_rotation_6d(scene_r_mat)
        scene_t = torch.bmm(xz2f_mat, pred_t.unsqueeze(-1)).squeeze(-1)

        return {
            "translation": scene_t.unsqueeze(1),
            "6drotation_normalized": scene_r_6d.unsqueeze(1),
            "scale": pred_s.unsqueeze(1),
            "xz2f_rot": pred_xz2f,
            "pred_translation": pred_t.unsqueeze(1),
            "pred_6drotation_normalized": pred_r.unsqueeze(1),
        }

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
