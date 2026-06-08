# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import *
from torch.utils import _pytree
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention, MOTMultiHeadSelfAttention, MOTMultiHeadSelfAttentionORG
from ..norm import LayerNorm32
from .blocks import FeedForwardNet


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, use_reentrant=False
            )
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
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
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
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
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, use_reentrant=False
            )
        else:
            return self._forward(x, mod, context)


class MOTModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
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
        latent_names: List = None,
        freeze_shared_parameters: bool = False,
        use_global_attn: bool = False,
        is_last_block: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = torch.nn.ModuleDict(
            {
                latent_name: LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
                for latent_name in latent_names
            }
        )
        self.norm2 = torch.nn.ModuleDict(
            {
                latent_name: LayerNorm32(channels, elementwise_affine=True, eps=1e-6) if not (is_last_block and latent_name == "shape") else None
                for latent_name in latent_names 
            }
        )
        self.norm3 = torch.nn.ModuleDict(
            {
                latent_name: LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
                for latent_name in latent_names
            }
        )
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
            latent_names=latent_names,
            protect_modality_list=[],  # default: legacy behavior (v1-compatible). For attn_fix, set to ["shape"] externally
            use_global_attn=use_global_attn,
            is_last_block=is_last_block,
        ) 
        
        self.cross_attn = torch.nn.ModuleDict(
            {
                latent_name: MultiHeadAttention(
                        channels,
                        ctx_channels=ctx_channels,
                        num_heads=num_heads,
                        type="cross",
                        attn_mode="full",
                        qkv_bias=qkv_bias,
                        qk_rms_norm=qk_rms_norm_cross,
                    ) if not (is_last_block and latent_name == "shape") else None
                for latent_name in latent_names 
            }
        )
        self.mlp = torch.nn.ModuleDict(
           {
                latent_name: (
                    FeedForwardNet(
                        channels,
                        mlp_ratio=mlp_ratio,
                    ) if not (is_last_block and latent_name == "shape") else None
                )
                for latent_name in latent_names 
            }
        )
        self.is_last_block = is_last_block

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )
            if freeze_shared_parameters:
                self.adaLN_modulation.eval()
                self.adaLN_modulation.requires_grad_(False)

        self.learnable_mod = nn.Parameter(torch.randn(1, channels))
        self.store_attn_map = False

    def _apply_module(self, h, module):
        if module is None: 
            return h
        else:
            return module(h)

    def _apply_cross_attn(self, h, cross_attn, context):
        if cross_attn is not None:
            return cross_attn(h, context)
        else:
            return h


    def _apply_msa(self, h, scale_msa, shift_msa):
        return h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

    def _apply_mlp(self, h, scale_mlp, shift_mlp):
        return h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

    def _apply_add(self, x, h):
        return x + h

    def _apply_multiplication(self, h, multiplier):
        return h * multiplier.unsqueeze(1)

    # This is stupid, _pytree does not support ModuleDict
    def _moduledict_to_dict(self, module):
        return {key: module for key, module in module.items()}

    def get_attn_maps(self):
        maps = {}
        # Cross-attention maps
        cross_maps = {}
        for name, ca in self.cross_attn.items():
            if ca is not None and ca.last_attn_map is not None:
                cross_maps[name] = ca.last_attn_map
        if cross_maps:
            maps["cross_attn"] = cross_maps
        # Self-attention maps (global or local depending on use_global_attn)
        if self.self_attn.last_attn_map is not None:
            if self.self_attn.use_global_attn:
                maps["global_self_attn"] = self.self_attn.last_attn_map
            else:
                maps["local_self_attn"] = self.self_attn.last_attn_map
        return maps

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, num_parts: Optional[torch.Tensor] = None):
        # Propagate store_attn_map flag to sub-modules
        self.self_attn.store_attn_map = self.store_attn_map
        for name, ca in self.cross_attn.items():
            if ca is not None:
                ca.store_attn_map = self.store_attn_map
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm1))
        h = _pytree.tree_map(
            partial(self._apply_msa, scale_msa=scale_msa, shift_msa=shift_msa),
            h
        )

        h = self.self_attn(h, num_parts=num_parts)
        h = _pytree.tree_map(
            partial(self._apply_multiplication, multiplier=gate_msa),
            h
        )
        x = _pytree.tree_map(
            self._apply_add,
            x,
            h
        )
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm2))

        h = _pytree.tree_map(
            partial(self._apply_cross_attn, context=context),
            h,
            self._moduledict_to_dict(self.cross_attn),
        )

        x = _pytree.tree_map(
            self._apply_add,
            x,
            h
        )
        h = _pytree.tree_map(self._apply_module, x, self._moduledict_to_dict(self.norm3))
        h = _pytree.tree_map(
            partial(self._apply_mlp, scale_mlp=scale_mlp, shift_mlp=shift_mlp),
            h
        )

        h = _pytree.tree_map(self._apply_module, h, self._moduledict_to_dict(self.mlp))

        h = _pytree.tree_map(
            partial(self._apply_multiplication, multiplier=gate_mlp),
            h
        )

        x = _pytree.tree_map(
            self._apply_add,
            x,
            h
        )

        return x

    def forward(self, x: Dict, mod: torch.Tensor, context: torch.Tensor, num_parts: Optional[torch.Tensor] = None):
        mod = self.learnable_mod.repeat(context.shape[0], 1)
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, num_parts, use_reentrant=False)
        else:
            return self._forward(x, mod, context, num_parts)
