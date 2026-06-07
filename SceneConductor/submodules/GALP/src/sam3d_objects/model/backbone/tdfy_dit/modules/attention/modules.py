# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import *
from torch.utils import _pytree
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from src.sam3d_objects.data.utils import (
    tree_reduce_unique,
)
from einops import rearrange

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x, dim=-1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = (
            torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        )
        return x_embed

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))

        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(
                            *phases.shape[:-1],
                            self.hidden_size // 2 - phases.shape[1],
                            device=phases.device,
                        ),
                        torch.zeros(
                            *phases.shape[:-1],
                            self.hidden_size // 2 - phases.shape[1],
                            device=phases.device,
                        ),
                    ),
                ],
                dim=-1,
            )
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed

from icecream import ic
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.store_attn_map = False
        self.last_attn_map = None
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert (
            type == "self" or attn_mode == "full"
        ), "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=2)
                q, k = self.rope(q, k, indices)
                qkv = torch.stack([q, k, v], dim=2)
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = qkv.unbind(dim=2)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                k, v = kv.unbind(dim=2)
                h = scaled_dot_product_attention(q, k, v)
            if self.store_attn_map:
                with torch.no_grad():
                    _q = q.permute(0, 2, 1, 3).float()  # [B, H, L, C]
                    _k = k.permute(0, 2, 1, 3).float()  # [B, H, Lkv, C]
                    scale = 1.0 / (_q.shape[-1] ** 0.5)
                    attn_w = torch.softmax(_q @ _k.transpose(-2, -1) * scale, dim=-1)
                    self.last_attn_map = attn_w.cpu()  # [B, H, L, Lkv]
        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        return h


class MOTMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        latent_names: List = None,
        protect_modality_list: List = [],
        use_global_attn: bool = False,
        is_last_block: bool = False,
    ):
        super().__init__()
        self.store_attn_map = False
        self.last_attn_map = None
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert (
            type == "self" or attn_mode == "full"
        ), "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.protect_modality_list = protect_modality_list
        self.use_global_attn = use_global_attn
        self.latent_names = latent_names

        if self._type == "self":
            self.to_qkv = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(channels, channels * 3, bias=qkv_bias)
                    for latent_name in latent_names
                }
            )
        else:
            self.to_q = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(channels, channels, bias=qkv_bias)
                    for latent_name in latent_names
                }
            )
            self.to_kv = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(
                        self.ctx_channels, channels * 2, bias=qkv_bias
                    )
                    for latent_name in latent_names
                }
            )

        if self.qk_rms_norm:
            self.q_rms_norm = torch.nn.ModuleDict(
                {
                    latent_name: MultiHeadRMSNorm(self.head_dim, num_heads) if not (is_last_block and latent_name == "shape") else None            
                    for latent_name in latent_names
                }
            )
            self.k_rms_norm = torch.nn.ModuleDict(
                {
                    latent_name: MultiHeadRMSNorm(self.head_dim, num_heads)
                    for latent_name in latent_names
                }
            )

        self.to_out = torch.nn.ModuleDict(
            {
                latent_name: nn.Linear(channels, channels) if not (is_last_block and latent_name == "shape") else None            
                for latent_name in latent_names 
            }
        )

        self.is_last_block = is_last_block

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def _reshape(self, qkv, tensor_shape, num_heads):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, 3, num_heads, -1)

    def _reshape_back(self, qkv, tensor_shape):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, -1)

    def _apply_module(self, x, module):
        if module is None:
            return x
        else:
            return module(x)

    # This is stupid, _pytree does not support ModuleDict
    def _moduledict_to_dict(self, module):
        return {key: module for key, module in module.items()}

    def unbind_qkv(self, qkv):
        q, k, v = {}, {}, {}
        for latent_name, _qkv in qkv.items():
            _q, _k, _v = _qkv.unbind(dim=2)
            q[latent_name] = _q
            k[latent_name] = _k
            v[latent_name] = _v

        return q, k, v

    def _get_shape(self, x):
        return x.shape

    def concatenate_tensor(self, tensor_dict, latent_names):
        merged = []
        indicies_mapping = {}
        total_tokens = 0
        for latent_name in latent_names:
            merged.append(tensor_dict[latent_name])
            cur_token_len = tensor_dict[latent_name].shape[1]
            indicies_mapping[latent_name] = [total_tokens, cur_token_len]
            total_tokens += cur_token_len
        # merge along token dimension
        return torch.cat(merged, dim=1), indicies_mapping

    def unpack_tensors(self, h_others, indicies_mapping):
        h = {}
        for latent_name, (start, cur_token_len) in indicies_mapping.items():
            h[latent_name] = h_others[:, start : start + cur_token_len]

        return h

    def mm_scale_dot_product_attention(self, q, k, v, num_parts: Optional[torch.Tensor] = None):
        h = {}
        latent_names = list(q.keys())

        # Protected modalities: per-object local self-attention only
        for protect_modality in self.protect_modality_list:
            if protect_modality in q:
                _pq = q[protect_modality]
                _pk = k[protect_modality]
                _pv = v[protect_modality]
                h[protect_modality] = scaled_dot_product_attention(_pq, _pk, _pv)

        # Other modalities: global cross-object or local attention
        other_modalities = [
            n for n in latent_names if n not in self.protect_modality_list
        ]
        if other_modalities:
            _q, indicies_mapping = self.concatenate_tensor(q, other_modalities)
            _k, _ = self.concatenate_tensor(k, other_modalities)
            _v, _ = self.concatenate_tensor(v, other_modalities)

            if self.use_global_attn:
                h_others = self.global_attention(_q, _k, _v, num_parts)
            else:
                h_others = scaled_dot_product_attention(_q, _k, _v)
                if self.store_attn_map:
                    with torch.no_grad():
                        _q_f = _q.permute(0, 2, 1, 3).float()  # [B, H, T, C]
                        _k_f = _k.permute(0, 2, 1, 3).float()  # [B, H, T, C]
                        scale = 1.0 / (_q_f.shape[-1] ** 0.5)
                        attn_w = _q_f @ _k_f.transpose(-2, -1) * scale
                        
                        self.last_attn_map = attn_w.cpu()  # [B, H, T, T]
            h.update(self.unpack_tensors(h_others, indicies_mapping))

        return h


    def global_attention(self, query, key, value, num_parts):
        idx = 0
        hidden_states_list = []
        attn_maps_list = [] if self.store_attn_map else None
        for n_p in num_parts:
            q = query[idx : idx + n_p]
            k = key[idx : idx + n_p]
            v = value[idx : idx + n_p]
            idx += n_p

            q = rearrange(
                q, "(b ni) nt h c -> b (ni nt) h c", ni=n_p
            ) # [b, h, ni*nt, c]
            k = rearrange(
                k, "(b ni) nt h c -> b (ni nt) h c", ni=n_p
            ) # [b, h, ni*nt, c]
            v = rearrange(
                v, "(b ni) nt h c -> b (ni nt) h c", ni=n_p
            ) # [b, h, ni*nt, c]

            h_s = scaled_dot_product_attention(
                q, k, v
            )

            if self.store_attn_map:
                with torch.no_grad():
                    _q = q.permute(0, 2, 1, 3).float()  # [1, H, L, C]
                    _k = k.permute(0, 2, 1, 3).float()  # [1, H, L, C]
                    scale = 1.0 / (_q.shape[-1] ** 0.5)
                    attn_w = torch.softmax(_q @ _k.transpose(-2, -1) * scale, dim=-1)
                    attn_maps_list.append(attn_w.cpu())  # [1, H, n_p*T, n_p*T]

            h_s = rearrange(
                h_s, "b (ni nt) h c -> (b ni) nt h c", ni=n_p
            )

            h_s = h_s.to(query.dtype)
            hidden_states_list.append(h_s)
        h = torch.cat(hidden_states_list, dim=0)
        if self.store_attn_map:
            self.last_attn_map = attn_maps_list
        return h

    def forward(
        self,
        x: Dict,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shapes = _pytree.tree_map(self._get_shape, x)
        if self._type == "self":
            qkv = _pytree.tree_map(
                self._apply_module, x, self._moduledict_to_dict(self.to_qkv)
            )
            qkv = _pytree.tree_map(
                partial(self._reshape, num_heads=self.num_heads), qkv, shapes
            )
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = self.unbind_qkv(qkv)
                    q = _pytree.tree_map(
                        self._apply_module, q, self._moduledict_to_dict(self.q_rms_norm)
                    )
                    k = _pytree.tree_map(
                        self._apply_module, k, self._moduledict_to_dict(self.k_rms_norm)
                    )
                    h = self.mm_scale_dot_product_attention(q, k, v, num_parts=num_parts)
                else:
                    raise NotImplementedError
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            raise NotImplementedError

        h = _pytree.tree_map(self._reshape_back, h, shapes)

        h = _pytree.tree_map(
                self._apply_module, h, self._moduledict_to_dict(self.to_out)
            )

        return h


class MOTMultiHeadSelfAttentionV3(MOTMultiHeadSelfAttention):
    """
    V3: Translation-Guided Self-Attention.

    Overrides ``global_attention`` to add a learnable additive bias derived
    from translation token attention patterns.  The bias captures object-level
    spatial relationships (proximity via bbox centres) and modulates *all*
    token interactions in the global attention step.

    New parameters
    --------------
    trans_token_pos : int
        Position of the translation token within each object's concatenated
        token block.  Computed by the V3 wrapper from latent_names / token
        lengths (default 65 for the standard config: shape 64 + merged offset 1).
    _alpha : nn.Parameter | None
        Learnable scalar that scales the translation bias.  Injected by
        the V3 wrapper.  Zero-initialised so the model starts at the V1
        fixed point.
    """

    def __init__(self, *args, trans_token_pos: int = 65, **kwargs):
        super().__init__(*args, **kwargs)
        self.trans_token_pos = trans_token_pos
        self._alpha: Optional[torch.nn.Parameter] = None
        print("ATTN FIX MODEL")

    def set_alpha(self, alpha: torch.nn.Parameter):
        """Inject the learnable alpha from the V3 wrapper."""
        self._alpha = alpha

    def global_attention(self, query, key, value, num_parts):
        """
        Translation-guided global attention.

        For each scene in the batch:
        1. Flatten all objects' tokens: ``[n_p, T, H, C] -> [1, n_p*T, H, C]``
        2. Extract translation Q / K at known positions
        3. Compute object-level attention logits: ``[1, H, n_p, n_p]``
        4. Expand to full-token bias via index mapping
        5. Add ``alpha * bias`` to full attention logits before softmax
        """
        import math as _math

        idx = 0
        hidden_states_list = []
        T_total = query.shape[1]          # tokens per object
        trans_pos = self.trans_token_pos   # within each object's block

        for n_p in num_parts:
            q = query[idx : idx + n_p]
            k = key[idx : idx + n_p]
            v = value[idx : idx + n_p]
            idx += n_p

            # Flatten across objects: [n_p, T, H, C] -> [1, L, H, C]
            q_full = rearrange(q, "(b ni) nt h c -> b (ni nt) h c", ni=n_p)
            k_full = rearrange(k, "(b ni) nt h c -> b (ni nt) h c", ni=n_p)
            v_full = rearrange(v, "(b ni) nt h c -> b (ni nt) h c", ni=n_p)

            L = n_p * T_total
            C = q_full.shape[3]
            scale_factor = 1.0 / _math.sqrt(C)

            # --- Extract translation tokens ---
            trans_indices = torch.arange(n_p, device=q.device) * T_total + trans_pos
            q_trans = q_full[:, trans_indices, :, :]   # [1, n_p, H, C]
            k_trans = k_full[:, trans_indices, :, :]   # [1, n_p, H, C]

            # --- Object-level translation attention logits ---
            q_t = q_trans.permute(0, 2, 1, 3)         # [1, H, n_p, C]
            k_t = k_trans.permute(0, 2, 1, 3)         # [1, H, n_p, C]
            A_trans = (q_t @ k_t.transpose(-2, -1)) * scale_factor  # [1, H, n_p, n_p]

            # --- Expand to full-token bias via fancy indexing ---
            # obj_ids[t] = which object token t belongs to
            obj_ids = torch.arange(L, device=q.device) // T_total  # [L]
            A_expanded = A_trans[:, :, obj_ids][:, :, :, obj_ids]  # [1, H, L, L]

            # --- Full attention with additive translation bias ---
            q_perm = q_full.permute(0, 2, 1, 3)       # [1, H, L, C]
            k_perm = k_full.permute(0, 2, 1, 3)       # [1, H, L, C]
            v_perm = v_full.permute(0, 2, 1, 3)       # [1, H, L, C]

            attn_logits = (q_perm @ k_perm.transpose(-2, -1)) * scale_factor
            attn_logits = attn_logits + self._alpha * A_expanded

            attn_weight = torch.softmax(attn_logits, dim=-1)
            h_s = attn_weight @ v_perm                 # [1, H, L, C]
            h_s = h_s.permute(0, 2, 1, 3)             # [1, L, H, C]

            h_s = rearrange(h_s, "b (ni nt) h c -> (b ni) nt h c", ni=n_p)
            h_s = h_s.to(query.dtype)
            hidden_states_list.append(h_s)

        h = torch.cat(hidden_states_list, dim=0)
        return h




class MOTMultiHeadSelfAttentionORG(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        latent_names: List = None,
        protect_modality_list: List = ["shape"],
        use_global_attn: bool = False,
        is_last_block: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert (
            type == "self" or attn_mode == "full"
        ), "Cross-attention only supports full attention"

        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")

        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        self.protect_modality_list = protect_modality_list
        self.use_global_attn = use_global_attn
        self.latent_names = latent_names

        if self._type == "self":
            self.to_qkv = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(channels, channels * 3, bias=qkv_bias)
                    for latent_name in latent_names
                }
            )
        else:
            self.to_q = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(channels, channels, bias=qkv_bias)
                    for latent_name in latent_names
                }
            )
            self.to_kv = torch.nn.ModuleDict(
                {
                    latent_name: nn.Linear(
                        self.ctx_channels, channels * 2, bias=qkv_bias
                    )
                    for latent_name in latent_names
                }
            )

        if self.qk_rms_norm:
            self.q_rms_norm = torch.nn.ModuleDict(
                {
                    latent_name: MultiHeadRMSNorm(self.head_dim, num_heads) if not (is_last_block and latent_name == "shape") else None 
                    for latent_name in latent_names
                }
            )
            self.k_rms_norm = torch.nn.ModuleDict(
                {
                    latent_name: MultiHeadRMSNorm(self.head_dim, num_heads)
                    for latent_name in latent_names
                }
            )

        self.to_out = torch.nn.ModuleDict(
            {
                latent_name: nn.Linear(channels, channels) if not (is_last_block and latent_name == "shape") else None            
                for latent_name in latent_names 
            }
        )

        self.is_last_block = is_last_block

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)

    def _reshape(self, qkv, tensor_shape, num_heads):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, 3, num_heads, -1)

    def _reshape_back(self, qkv, tensor_shape):
        B, L, _ = tensor_shape
        return qkv.reshape(B, L, -1)

    def _apply_module(self, x, module):
        if module is None:
            return x
        else:
            return module(x)

    # This is stupid, _pytree does not support ModuleDict
    def _moduledict_to_dict(self, module):
        return {key: module for key, module in module.items()}

    def unbind_qkv(self, qkv):
        q, k, v = {}, {}, {}
        for latent_name, _qkv in qkv.items():
            _q, _k, _v = _qkv.unbind(dim=2)
            q[latent_name] = _q
            k[latent_name] = _k
            v[latent_name] = _v

        return q, k, v

    def _get_shape(self, x):
        return x.shape

    def concatenate_tensor(self, tensor_dict, latent_names):
        merged = []
        indicies_mapping = {}
        total_tokens = 0
        for latent_name in latent_names:
            merged.append(tensor_dict[latent_name])
            cur_token_len = tensor_dict[latent_name].shape[1]
            indicies_mapping[latent_name] = [total_tokens, cur_token_len]
            total_tokens += cur_token_len
        # merge along token dimension
        return torch.cat(merged, dim=1), indicies_mapping

    def unpack_tensors(self, h_others, indicies_mapping):
        h = {}
        for latent_name, (start, cur_token_len) in indicies_mapping.items():
            h[latent_name] = h_others[:, start : start + cur_token_len]

        return h

    def mm_scale_dot_product_attention(self, q, k, v, num_parts: Optional[torch.Tensor] = None):
        h = {}
        latent_names = list(q.keys())
        # for protected modality, it only attends itself
        for protect_modality in self.protect_modality_list:
            _q = q[protect_modality]
            _k = k[protect_modality]
            _v = v[protect_modality]
            h[protect_modality] = scaled_dot_product_attention(_q, _k, _v)

        # for the rest it is ok to attend each other and allow gradient
        other_modalities = [
            n for n in latent_names if n not in self.protect_modality_list
        ]
        _q, indicies_mapping = self.concatenate_tensor(q, other_modalities)
        o_k, _ = self.concatenate_tensor(k, other_modalities)
        o_v, _ = self.concatenate_tensor(v, other_modalities)
        # no gradiant flow back to protected modality (e.g. shape)
        _k, _ = self.concatenate_tensor(k, self.protect_modality_list)
        _v, _ = self.concatenate_tensor(v, self.protect_modality_list)
        # _k = _k.detach()
        # _v = _v.detach()
        _k = torch.cat([o_k, _k], dim=1)
        _v = torch.cat([o_v, _v], dim=1)
        

        if self.use_global_attn:
            h_others = self.global_attention(_q, _k, _v, num_parts)
        else:
            h_others = scaled_dot_product_attention(_q, _k, _v)
        h.update(self.unpack_tensors(h_others, indicies_mapping))

        return h


    def global_attention(self, query, key, value, num_parts):
        idx = 0
        hidden_states_list = []
        for n_p in num_parts:
            q = query[idx : idx + n_p]
            k = key[idx : idx + n_p]
            v = value[idx : idx + n_p]
            idx += n_p
            
            q = rearrange(
                q, "(b ni) nt h c -> b (ni nt) h c", ni=n_p
            ) # [b, h, ni*nt, c]
            k = rearrange(
                k, "(b ni) nt h c -> b (ni nt) h c", ni=n_p
            ) # [b, h, ni*nt, c]
            v = rearrange(
                v, "(b ni) nt h c -> b (ni nt) h c", ni=n_p
            ) # [b, h, ni*nt, c]

            h_s = scaled_dot_product_attention(
                q, k, v
            )
            
            h_s = rearrange(
                h_s, "b (ni nt) h c -> (b ni) nt h c", ni=n_p
            )

            h_s = h_s.to(query.dtype)
            hidden_states_list.append(h_s)
        h = torch.cat(hidden_states_list, dim=0)
        return h

    def forward(
        self,
        x: Dict,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        shapes = _pytree.tree_map(self._get_shape, x)
        if self._type == "self":
            qkv = _pytree.tree_map(
                self._apply_module, x, self._moduledict_to_dict(self.to_qkv)
            )
            qkv = _pytree.tree_map(
                partial(self._reshape, num_heads=self.num_heads), qkv, shapes
            )
            # if self.use_rope:
            #     raise NotImplementedError
            if self.attn_mode == "full":
                if self.qk_rms_norm:
                    q, k, v = self.unbind_qkv(qkv)
                    q = _pytree.tree_map(
                        self._apply_module, q, self._moduledict_to_dict(self.q_rms_norm)
                    )
                    k = _pytree.tree_map(
                        self._apply_module, k, self._moduledict_to_dict(self.k_rms_norm)
                    )
                    h = self.mm_scale_dot_product_attention(q, k, v, num_parts=num_parts)
                else:
                    raise NotImplementedError
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            raise NotImplementedError

        h = _pytree.tree_map(self._reshape_back, h, shapes)

        h = _pytree.tree_map(
                self._apply_module, h, self._moduledict_to_dict(self.to_out)
            )
            
        return h
