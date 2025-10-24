# SPDX-License-Identifier: Apache-2.0
# Copyright 2025

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

try:
    MIRAGE_OPS = torch.ops.mirage_ops  # type: ignore[attr-defined]
except AttributeError as exc:  # pragma: no cover - informative error
    raise ImportError(
        "Mirage custom ops are not available. "
        "Ensure `pip install -e .` has been rerun after adding the Mirage "
        "kernels so that `vllm._mirage_ops` is built.") from exc


def _ensure_cuda_tensor(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    return t.contiguous()


def _split_real_imag(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, num_heads, head_dim = x.shape
    head_pairs = head_dim // 2
    pairs = x.view(batch, seq_len, num_heads, head_pairs, 2)
    return pairs[..., 0].contiguous(), pairs[..., 1].contiguous()


def _combine_real_imag(real: torch.Tensor,
                       imag: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack((real, imag), dim=-1)
    return stacked.reshape(real.shape[0], real.shape[1], real.shape[2],
                           real.shape[3] * 2)


def _expand_rotary_tables(cos: torch.Tensor,
                          sin: torch.Tensor,
                          batch: int,
                          seq_len: int,
                          num_heads: int,
                          head_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.expand(batch, seq_len, num_heads, head_pairs).contiguous()
    sin = sin.expand(batch, seq_len, num_heads, head_pairs).contiguous()
    return cos, sin


class MirageRopeKernel:
    """Thin wrapper around the Mirage RoPE kernels."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor,
                 sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = _ensure_cuda_tensor(q, "q")
        k = _ensure_cuda_tensor(k, "k")
        cos = _ensure_cuda_tensor(cos, "cos")
        sin = _ensure_cuda_tensor(sin, "sin")

        if q.dim() != 4 or k.dim() != 4:
            raise ValueError("q and k must be [batch, seq, heads, head_dim].")
        if q.shape != k.shape:
            raise ValueError("q and k must have identical shapes.")

        batch, seq_len, num_heads, head_dim = q.shape
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE kernels.")

        head_pairs = head_dim // 2
        q_real, q_imag = _split_real_imag(q)
        k_real, k_imag = _split_real_imag(k)

        cos_q, sin_q = _expand_rotary_tables(cos, sin, batch, seq_len,
                                             num_heads, head_pairs)
        cos_k, sin_k = _expand_rotary_tables(cos, sin, batch, seq_len,
                                             k.shape[2], head_pairs)
        neg_sin_q = (-sin_q).contiguous()
        neg_sin_k = (-sin_k).contiguous()

        inputs = [
            q_real, q_imag, k_real, k_imag, cos_q, sin_q, neg_sin_q, cos_k,
            sin_k, neg_sin_k
        ]

        q_rot_real = torch.empty_like(q_real)
        q_rot_imag = torch.empty_like(q_imag)
        k_rot_real = torch.empty_like(k_real)
        k_rot_imag = torch.empty_like(k_imag)
        outputs = [q_rot_real, q_rot_imag, k_rot_real, k_rot_imag]

        MIRAGE_OPS.rope_run(batch, inputs, outputs)

        q_rot = _combine_real_imag(q_rot_real, q_rot_imag)
        k_rot = _combine_real_imag(k_rot_real, k_rot_imag)
        return q_rot, k_rot


@dataclass(frozen=True)
class QkvShape:
    num_tokens: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def total_qkv_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim


def _selection_matrix(total_dim: int, start: int, length: int,
                      dtype, device) -> torch.Tensor:
    mat = torch.zeros(total_dim, length, dtype=dtype, device=device)
    mat[start:start + length, :] = torch.eye(length, dtype=dtype, device=device)
    return mat


def _head_reduce_matrix(total_dim: int, num_heads: int, head_dim: int,
                        dtype, device) -> torch.Tensor:
    mat = torch.zeros(total_dim, num_heads, dtype=dtype, device=device)
    scale = torch.tensor(1.0 / head_dim, dtype=dtype, device=device)
    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim
        mat[start:end, head] = scale
    return mat


def _head_expand_matrix(num_heads: int, head_dim: int, dtype,
                        device) -> torch.Tensor:
    mat = torch.zeros(num_heads, num_heads * head_dim, dtype=dtype, device=device)
    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim
        mat[head, start:end] = torch.ones(head_dim, dtype=dtype, device=device)
    return mat


@lru_cache(maxsize=32)
def _prepare_static_matrices(shape: QkvShape, dtype: torch.dtype,
                             device: torch.device) -> Dict[str, torch.Tensor]:
    matrices: Dict[str, torch.Tensor] = {}

    matrices["select_q"] = _selection_matrix(shape.total_qkv_dim, 0,
                                             shape.q_dim, dtype, device)
    matrices["select_k"] = _selection_matrix(shape.total_qkv_dim,
                                             shape.q_dim,
                                             shape.kv_dim,
                                             dtype,
                                             device)
    matrices["select_v"] = _selection_matrix(shape.total_qkv_dim,
                                             shape.q_dim + shape.kv_dim,
                                             shape.kv_dim,
                                             dtype,
                                             device)
    matrices["select_q_t"] = matrices["select_q"].transpose(0,
                                                            1).contiguous()
    matrices["select_k_t"] = matrices["select_k"].transpose(0,
                                                            1).contiguous()
    matrices["select_v_t"] = matrices["select_v"].transpose(0,
                                                            1).contiguous()

    matrices["head_reduce_q"] = _head_reduce_matrix(shape.q_dim,
                                                    shape.num_q_heads,
                                                    shape.head_dim, dtype,
                                                    device)
    matrices["head_reduce_k"] = _head_reduce_matrix(shape.kv_dim,
                                                    shape.num_kv_heads,
                                                    shape.head_dim, dtype,
                                                    device)
    matrices["head_expand_q"] = _head_expand_matrix(shape.num_q_heads,
                                                    shape.head_dim, dtype,
                                                    device)
    matrices["head_expand_k"] = _head_expand_matrix(shape.num_kv_heads,
                                                    shape.head_dim, dtype,
                                                    device)
    return matrices


class MirageQKVRMSNormKernel:
    """Fused QKV projection + RMSNorm for the Mirage kernels."""

    def __call__(self,
                 hidden_flat: torch.Tensor,
                 w_qkv: torch.Tensor,
                 q_weight: torch.Tensor,
                 k_weight: torch.Tensor,
                 *,
                 batch_size: int,
                 seq_len: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 eps: float) -> torch.Tensor:
        hidden_flat = _ensure_cuda_tensor(hidden_flat, "hidden_flat")
        w_qkv = _ensure_cuda_tensor(w_qkv, "w_qkv")
        q_weight = _ensure_cuda_tensor(q_weight, "q_weight")
        k_weight = _ensure_cuda_tensor(k_weight, "k_weight")

        num_tokens = hidden_flat.shape[0]
        if num_tokens != batch_size * seq_len:
            raise ValueError(
                "hidden_flat rows must equal batch_size * seq_len.")

        shape = QkvShape(num_tokens, hidden_flat.shape[1], num_q_heads,
                         num_kv_heads, head_dim)
        dtype = hidden_flat.dtype
        device = hidden_flat.device
        matrices = _prepare_static_matrices(shape, dtype, device)

        gamma_q_head = q_weight.view(1, head_dim).expand(shape.num_q_heads,
                                                         -1).reshape(1, -1)
        gamma_k_head = k_weight.view(1,
                                     head_dim).expand(shape.num_kv_heads,
                                                      -1).reshape(1, -1)
        gamma_q = gamma_q_head.expand(shape.num_tokens, -1).contiguous()
        gamma_k = gamma_k_head.expand(shape.num_tokens, -1).contiguous()

        eps_q = torch.full((shape.num_tokens, shape.num_q_heads),
                           eps,
                           dtype=dtype,
                           device=device)
        eps_k = torch.full((shape.num_tokens, shape.num_kv_heads),
                           eps,
                           dtype=dtype,
                           device=device)

        inputs = [
            hidden_flat,
            w_qkv,
            matrices["select_q"],
            matrices["select_k"],
            matrices["select_v"],
            matrices["select_q_t"],
            matrices["select_k_t"],
            matrices["select_v_t"],
            matrices["head_reduce_q"],
            matrices["head_reduce_k"],
            matrices["head_expand_q"],
            matrices["head_expand_k"],
            eps_q,
            eps_k,
            gamma_q,
            gamma_k,
        ]

        output = torch.empty_like(hidden_flat)
        MIRAGE_OPS.qkv_rmsnorm_run(batch_size, inputs, [output])
        return output


@lru_cache(maxsize=32)
def _build_gqa_matrices(num_q_heads: int, num_kv_heads: int, head_dim: int,
                        dtype: torch.dtype, device: torch.device
                        ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor],
                                   List[torch.Tensor]]:
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    group_size = num_q_heads // max(num_kv_heads, 1)

    expand = torch.zeros(kv_dim, q_dim, dtype=dtype, device=device)
    eye = torch.eye(head_dim, dtype=dtype, device=device)
    for kv_idx in range(num_kv_heads):
        for g in range(group_size):
            q_idx = kv_idx * group_size + g
            k_start = kv_idx * head_dim
            q_start = q_idx * head_dim
            expand[k_start:k_start + head_dim,
                   q_start:q_start + head_dim] = eye
    expand_t = expand.transpose(0, 1).contiguous()

    select_mats: List[torch.Tensor] = []
    select_t_mats: List[torch.Tensor] = []
    for head_idx in range(num_q_heads):
        select = torch.zeros(q_dim, head_dim, dtype=dtype, device=device)
        start = head_idx * head_dim
        select[start:start + head_dim, :] = eye
        select_mats.append(select)
        select_t_mats.append(select.transpose(0, 1).contiguous())

    return expand, expand_t, select_mats, select_t_mats


class MirageAttentionKernel:
    """Mirage grouped-query attention kernel wrapper."""

    def __call__(self,
                 q_flat: torch.Tensor,
                 k_flat: torch.Tensor,
                 v_flat: torch.Tensor,
                 w_o: torch.Tensor,
                 *,
                 batch_size: int,
                 seq_len: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 head_dim: int) -> torch.Tensor:
        q_flat = _ensure_cuda_tensor(q_flat, "q_flat")
        k_flat = _ensure_cuda_tensor(k_flat, "k_flat")
        v_flat = _ensure_cuda_tensor(v_flat, "v_flat")
        w_o = _ensure_cuda_tensor(w_o, "w_o")

        tokens = q_flat.shape[0]
        if tokens != batch_size * seq_len:
            raise ValueError("q_flat rows must equal batch_size * seq_len.")

        dtype = q_flat.dtype
        device = q_flat.device

        expand, expand_t, select_mats, select_t_mats = _build_gqa_matrices(
            num_q_heads, num_kv_heads, head_dim, dtype, device)

        scale_value = 1.0 / (float(head_dim)**0.5)
        scale_matrix = torch.full((tokens, tokens), scale_value, dtype=dtype,
                                  device=device)
        ones_col = torch.ones(tokens, 1, dtype=dtype, device=device)
        ones_row = torch.ones(1, tokens, dtype=dtype, device=device)
        k_trans = k_flat.transpose(0, 1).contiguous()

        inputs: List[torch.Tensor] = [
            q_flat,
            k_flat,
            k_trans,
            v_flat,
            w_o,
            expand,
            expand_t,
            scale_matrix,
            ones_col,
            ones_row,
        ]
        for sel, sel_t in zip(select_mats, select_t_mats):
            inputs.append(sel)
            inputs.append(sel_t)

        hidden_size = w_o.shape[1]
        output = torch.empty(tokens, hidden_size, dtype=dtype, device=device)
        MIRAGE_OPS.attention_run(batch_size, inputs, [output])
        return output


class MirageRMSNormOnlyKernel:
    """Standalone RMSNorm (token-wise)."""

    def __call__(self,
                 inputs: torch.Tensor,
                 gamma: torch.Tensor,
                 *,
                 batch_size: int,
                 seq_len: int) -> torch.Tensor:
        inputs = _ensure_cuda_tensor(inputs, "inputs")
        gamma = _ensure_cuda_tensor(gamma, "gamma")

        tokens = inputs.shape[0]
        if tokens != batch_size * seq_len:
            raise ValueError("inputs rows must equal batch_size * seq_len.")

        weight_matrix = torch.diag(gamma).contiguous()
        outputs = torch.empty_like(inputs)
        MIRAGE_OPS.rmsnorm_only_run(batch_size, [inputs, weight_matrix],
                                    [outputs])
        return outputs


class MirageFFNKernel:
    """Mirage gated-SiLU FFN kernel wrapper."""

    def __call__(self,
                 x: torch.Tensor,
                 w_gate: torch.Tensor,
                 w_up: torch.Tensor,
                 w_down: torch.Tensor,
                 bias_gate: torch.Tensor,
                 bias_up: torch.Tensor,
                 bias_down: torch.Tensor,
                 *,
                 batch_size: int,
                 seq_len: int) -> torch.Tensor:
        tensors = [
            ("x", x),
            ("w_gate", w_gate),
            ("w_up", w_up),
            ("w_down", w_down),
            ("bias_gate", bias_gate),
            ("bias_up", bias_up),
            ("bias_down", bias_down),
        ]
        inputs: List[torch.Tensor] = []
        for name, tensor in tensors:
            inputs.append(_ensure_cuda_tensor(tensor, name))

        tokens = x.shape[0]
        if tokens != batch_size * seq_len:
            raise ValueError("x rows must equal batch_size * seq_len.")

        output = torch.empty_like(x)
        MIRAGE_OPS.ffn_run(batch_size, inputs, [output])
        return output


__all__ = [
    "MirageRopeKernel",
    "MirageQKVRMSNormKernel",
    "MirageAttentionKernel",
    "MirageRMSNormOnlyKernel",
    "MirageFFNKernel",
]
