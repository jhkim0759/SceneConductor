# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import *
from torch.utils import _pytree
import torch
import torch.nn as nn
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from collections import namedtuple
from ..modules.utils import FP16_TYPE
from ..modules.transformer import (
    MOTModulatedTransformerCrossBlock,
)
from src.sam3d_objects.data.utils import (
    tree_reduce_unique,
)
from .timestep_embedder import TimestepEmbedder
from omegaconf import OmegaConf
from icecream import ic 


def _identity_condition_embedder(*args, **kwargs):
    return args[-1] if args else None


class SparseStructureFlowModel(nn.Module):
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
                MOTModulatedTransformerCrossBlock(
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
                    use_global_attn=idx%2==0,
                    is_last_block=(idx==num_blocks-1),
                )
                for idx in range(num_blocks)
            ]
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


        # zero init like controlnet, for MLP should only zero 
        # the weight of the last layer only
        if self.is_shortcut_model:
            nn.init.constant_(self.d_embedder.mlp[2].weight, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
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
        h: Dict,
        cond: torch.Tensor,
        num_parts: Optional[torch.Tensor] = None,
        store_attn: bool = False,
    ) -> torch.Tensor:

        input_dtype = tree_reduce_unique(lambda tensor: tensor.dtype, h)
        h = _pytree.tree_map(
            partial(self._cast_type, dtype=self.dtype),
            h,
        )

        if cond is None and self.force_zeros_cond:
            # Unconditional inference: create zeros matching the batch (N_obj) dimension of h
            n_obj = next(iter(h.values())).shape[0]
            dev = next(self.parameters()).device
            cond = torch.zeros(n_obj, 1, self.cond_channels, dtype=self.dtype, device=dev)
        cond = cond.type(self.dtype)

        # Propagate store_attn flag to blocks
        if store_attn:
            for block in self.blocks:
                block.store_attn_map = True

        for block in self.blocks:
            h = block(h, None, cond, num_parts=num_parts)

        # Collect attention maps
        attn_maps = None
        if store_attn:
            attn_maps = {"cross_attn": [], "global_self_attn": [], "local_self_attn": []}
            for i, block in enumerate(self.blocks):
                block_maps = block.get_attn_maps()
                attn_maps["cross_attn"].append(block_maps.get("cross_attn", {}))
                attn_maps["global_self_attn"].append(block_maps.get("global_self_attn", None))
                attn_maps["local_self_attn"].append(block_maps.get("local_self_attn", None))
                # Reset flags
                block.store_attn_map = False

        h = _pytree.tree_map(
            partial(self._cast_type, dtype=input_dtype),
            h,
        )

        if store_attn:
            return h, attn_maps
        return h


class SparseStructureFlowTdfyWrapper(SparseStructureFlowModel):
    def __init__(
        self,
        latent_mapping: dict,
        latent_share_transformer: dict = {},
        *args,
        **kwargs,
    ):
        condition_embedder = kwargs.pop("condition_embedder", None)
        # if enabled, model will record the condition_shape in one run and uses zeros for all that afterwards
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        # backward compatible to models trained before PR #87
        kwargs.pop("shape_attend_pose", None)
        merge_latent_names = [i for _, v in latent_share_transformer.items() for i in v]
        self.latent_names = [
            latent_name
            for latent_name in list(latent_mapping.keys())
            if latent_name not in merge_latent_names
        ] + list(latent_share_transformer.keys())
        super().__init__(*args, **kwargs)
        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = _identity_condition_embedder
        self.force_zeros_cond = force_zeros_cond
        self.latent_mapping = nn.ModuleDict(latent_mapping)
        if not isinstance(latent_share_transformer, dict):
            self.latent_share_transformer = OmegaConf.to_container(latent_share_transformer)
        else:
            self.latent_share_transformer = latent_share_transformer
        self.input_latent_mappings = list(self.latent_mapping.keys())

    def set_protect_modality(self, protect_list: list):
        """Set self_attn.protect_modality_list for all blocks."""
        for block in self.blocks:
            block.self_attn.protect_modality_list = protect_list

        # self.in_projs = {}
        # for latent_name in self.input_latent_mappings:
        #     if latent_name == "shape":
        #         continue6drotation_normalized'], batch['scale'], batch['translation']
        # self.in_projs["6drotation_normalized"] = nn.Linear(6, 1024)
        # self.in_projs["scale"] = nn.Linear(3, 1024)
        # self.in_projs["translation"] = nn.Linear(3, 1024)
        # self.register_parameter(f"learnable_query_{latent_name}", self.learnable_query[latent_name])

    def forward(
        self,
        input_dict: dict,
        cond: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
        store_attn: bool = False,
        **condition_kwargs,
    ) -> dict:
        # concatenate input
        latent_dict = self.project_input(input_dict)
        result = super().forward(latent_dict, cond, num_parts=num_parts, store_attn=store_attn)

        if store_attn:
            output, attn_maps = result
        else:
            output = result

        # split input to multiple output modalities
        output_latents = self.project_output(output)

        for key in output_latents:
            if key in input_dict and not key=="num_parts":
                output_latents[key] = output_latents[key] + input_dict[key]

        if store_attn:
            return output_latents, attn_maps
        return output_latents

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

        latent_dict = self.merge_latent_share_transformer(latent_dict)
        return latent_dict

    def project_output(self, output: Dict) -> Dict:
        output = self.split_latent_share_transformer(output)
        output_latents = {}
        for latent_name in self.input_latent_mappings:
            if latent_name == "shape":
                continue
            latent = self.latent_mapping[latent_name].to_output(output[latent_name])
            output_latents[latent_name] = latent

        return output_latents

    def merge_latent_share_transformer(self, latent_dict):
        visited_latent_names = set()
        return_dict = {}
        for merged_name, latent_names in self.latent_share_transformer.items():
            tensors = []
            for latent_name in latent_names:
                visited_latent_names.add(latent_name)
                tensors.append(latent_dict[latent_name])
            tensors = torch.cat(tensors, dim=1)
            return_dict[merged_name] = tensors

        for latent_name in latent_dict:
            if latent_name not in visited_latent_names:
                return_dict[latent_name] = latent_dict[latent_name]

        return return_dict

    def split_latent_share_transformer(self, output_latents):
        return_dict = {}
        visited_latent_names = set()
        for merged_name, latent_names in self.latent_share_transformer.items():
            start = 0
            visited_latent_names.add(merged_name)
            tensors = output_latents[merged_name]
            for latent_name in latent_names:
                token_len = self.latent_mapping[latent_name].pos_emb.shape[0]
                latent = tensors[:, start : start + token_len]
                return_dict[latent_name] = latent
                start += token_len

        for latent_name in output_latents:
            if latent_name not in visited_latent_names:
                return_dict[latent_name] = output_latents[latent_name]

        return return_dict
