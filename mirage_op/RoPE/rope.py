"""RoPE kernel demo tailored to the Qwen3 0.6B configuration.

This script:
  1. Loads Qwen3 0.6B hyperparameters from ``mirage_op/config/config.json``.
  2. Builds a Mirage graph that applies rotary position embeddings to query
     and key tensors using precomputed cosine/sine tables.
  3. Runs the graph on sample inputs to verify correctness against a PyTorch
     reference implementation.
  4. Exports the generated CUDA source into ``mirage_op/RoPE/output_cu``.

Run:
    python mirage_op/RoPE/rope.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import mirage as mi
import torch
from mirage.kernel import HARD_CODE  # type: ignore

from mirage_op.utils import (DEFAULT_BATCH, DEFAULT_SEED, load_model_config,
                             resolve_dtypes, resolve_output_dir)

DEFAULT_SEQ_LEN = 128


def build_rope_graph(batch: int, seq_len: int, num_q_heads: int,
                     num_kv_heads: int, head_pairs: int,
                     mi_dtype: mi.dtype) -> mi.KNGraph:
    graph = mi.new_kernel_graph()

    q_real = graph.new_input((batch, seq_len, num_q_heads, head_pairs),
                             dtype=mi_dtype)
    q_imag = graph.new_input((batch, seq_len, num_q_heads, head_pairs),
                             dtype=mi_dtype)
    k_real = graph.new_input((batch, seq_len, num_kv_heads, head_pairs),
                             dtype=mi_dtype)
    k_imag = graph.new_input((batch, seq_len, num_kv_heads, head_pairs),
                             dtype=mi_dtype)
    cos_q = graph.new_input((batch, seq_len, num_q_heads, head_pairs),
                            dtype=mi_dtype)
    sin_q = graph.new_input((batch, seq_len, num_q_heads, head_pairs),
                            dtype=mi_dtype)
    neg_sin_q = graph.new_input((batch, seq_len, num_q_heads, head_pairs),
                                dtype=mi_dtype)
    cos_k = graph.new_input((batch, seq_len, num_kv_heads, head_pairs),
                            dtype=mi_dtype)
    sin_k = graph.new_input((batch, seq_len, num_kv_heads, head_pairs),
                            dtype=mi_dtype)
    neg_sin_k = graph.new_input((batch, seq_len, num_kv_heads, head_pairs),
                                dtype=mi_dtype)

    # q_rot_real = q_real * cos_q + q_imag * (-sin_q)
    q_real_cos = graph.mul(q_real, cos_q)
    q_imag_neg_sin = graph.mul(q_imag, neg_sin_q)
    q_rot_real = graph.add(q_real_cos, q_imag_neg_sin)

    # q_rot_imag = q_imag * cos_q + q_real * sin_q
    q_imag_cos = graph.mul(q_imag, cos_q)
    q_real_sin = graph.mul(q_real, sin_q)
    q_rot_imag = graph.add(q_imag_cos, q_real_sin)

    # k_rot_real = k_real * cos_k + k_imag * (-sin_k)
    k_real_cos = graph.mul(k_real, cos_k)
    k_imag_neg_sin = graph.mul(k_imag, neg_sin_k)
    k_rot_real = graph.add(k_real_cos, k_imag_neg_sin)

    # k_rot_imag = k_imag * cos_k + k_real * sin_k
    k_imag_cos = graph.mul(k_imag, cos_k)
    k_real_sin = graph.mul(k_real, sin_k)
    k_rot_imag = graph.add(k_imag_cos, k_real_sin)

    graph.mark_output(q_rot_real)
    graph.mark_output(q_rot_imag)
    graph.mark_output(k_rot_real)
    graph.mark_output(k_rot_imag)
    return graph


def apply_rope_reference(x: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    batch, seq_len, num_heads, head_dim = x.shape
    head_pairs = head_dim // 2

    x_pairs = x.view(batch, seq_len, num_heads, head_pairs, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]

    cos_b = cos
    sin_b = sin
    if cos_b.ndim == 2:
        cos_b = cos_b.unsqueeze(0).unsqueeze(2).expand(batch, seq_len, num_heads,
                                                       head_pairs)
        sin_b = sin_b.unsqueeze(0).unsqueeze(2).expand(batch, seq_len, num_heads,
                                                       head_pairs)

    rot_real = x_real * cos_b - x_imag * sin_b
    rot_imag = x_imag * cos_b + x_real * sin_b
    return torch.stack((rot_real, rot_imag), dim=-1).reshape_as(x)


def prepare_rotary_tables(seq_len: int, head_dim: int, theta: float,
                          dtype: torch.dtype,
                          device: torch.device) -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
    half_dim = head_dim // 2
    position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
    idx = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta**(idx / half_dim))
    freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos.contiguous(), sin.contiguous()


def split_real_imag(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, num_heads, head_dim = x.shape
    head_pairs = head_dim // 2
    x_pairs = x.view(batch, seq_len, num_heads, head_pairs, 2)
    real = x_pairs[..., 0].contiguous()
    imag = x_pairs[..., 1].contiguous()
    return real, imag


def run_graph(
    graph: mi.KNGraph,
    qk_inputs: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = graph(inputs=qk_inputs)
    if len(outputs) != 4:
        raise RuntimeError(f"Expected four outputs, got {len(outputs)}")
    return tuple(outputs)  # type: ignore[return-value]


def export_cuda_code(graph: mi.KNGraph, input_tensors: list[torch.Tensor],
                     output_dir: Path) -> Path:
    device_props = torch.cuda.get_device_properties(input_tensors[0].device)
    target_cc = device_props.major * 10 + device_props.minor
    input_strides = [tensor.stride() for tensor in input_tensors]
    program = mi.generate_cuda_program(graph.cygraph,
                                       target_cc=target_cc,
                                       input_strides=input_strides)
    output_dir.mkdir(parents=True, exist_ok=True)
    cu_path = output_dir / "rope_kernel.cu"
    cu_path.write_text(program["code"] + HARD_CODE)
    return cu_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Mirage kernels.")

    model_cfg = load_model_config()
    hidden_size = int(model_cfg.get("hidden_size", 0))
    if hidden_size == 0:
        raise ValueError("Model config is missing a valid 'hidden_size'.")

    num_q_heads = int(model_cfg.get("num_attention_heads", 0))
    num_kv_heads = int(model_cfg.get("num_key_value_heads", num_q_heads))
    head_dim = int(model_cfg.get("head_dim", 0)) or hidden_size // max(
        num_q_heads, 1)
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE.")

    head_pairs = head_dim // 2
    seq_len = min(DEFAULT_SEQ_LEN,
                  int(model_cfg.get("max_position_embeddings",
                                    DEFAULT_SEQ_LEN)))
    theta = float(model_cfg.get("rope_theta", 10000.0))

    dtype_key = str(model_cfg.get("torch_dtype", "bfloat16"))
    dtype, mi_dtype = resolve_dtypes(dtype_key)

    device = torch.device("cuda")
    torch.manual_seed(DEFAULT_SEED)

    query = torch.randn(DEFAULT_BATCH,
                        seq_len,
                        num_q_heads,
                        head_dim,
                        dtype=dtype,
                        device=device)
    key = torch.randn(DEFAULT_BATCH,
                      seq_len,
                      num_kv_heads,
                      head_dim,
                      dtype=dtype,
                      device=device)

    cos_base, sin_base = prepare_rotary_tables(seq_len, head_dim, theta,
                                               dtype=dtype, device=device)
    cos_q = cos_base.unsqueeze(0).unsqueeze(2).expand(DEFAULT_BATCH, seq_len,
                                                      num_q_heads,
                                                      head_pairs).contiguous()
    sin_q = sin_base.unsqueeze(0).unsqueeze(2).expand(DEFAULT_BATCH, seq_len,
                                                      num_q_heads,
                                                      head_pairs).contiguous()
    cos_k = cos_base.unsqueeze(0).unsqueeze(2).expand(DEFAULT_BATCH, seq_len,
                                                      num_kv_heads,
                                                      head_pairs).contiguous()
    sin_k = sin_base.unsqueeze(0).unsqueeze(2).expand(DEFAULT_BATCH, seq_len,
                                                      num_kv_heads,
                                                      head_pairs).contiguous()

    q_real, q_imag = split_real_imag(query)
    k_real, k_imag = split_real_imag(key)

    graph = build_rope_graph(DEFAULT_BATCH, seq_len, num_q_heads, num_kv_heads,
                             head_pairs, mi_dtype)

    neg_sin_q = (-sin_q).contiguous()
    neg_sin_k = (-sin_k).contiguous()

    inputs = [
        q_real,
        q_imag,
        k_real,
        k_imag,
        cos_q,
        sin_q,
        neg_sin_q,
        cos_k,
        sin_k,
        neg_sin_k,
    ]
    q_rot_real, q_rot_imag, k_rot_real, k_rot_imag = run_graph(graph, inputs)

    def combine(real: torch.Tensor, imag: torch.Tensor,
                num_heads: int) -> torch.Tensor:
        stacked = torch.stack((real, imag), dim=-1)
        return stacked.reshape(DEFAULT_BATCH, seq_len, num_heads, head_dim)

    mirage_q = combine(q_rot_real, q_rot_imag, num_q_heads)
    mirage_k = combine(k_rot_real, k_rot_imag, num_kv_heads)

    torch_q = apply_rope_reference(query, cos_base, sin_base)
    torch_k = apply_rope_reference(key, cos_base, sin_base)

    max_diff_q = (mirage_q - torch_q).abs().max().item()
    max_diff_k = (mirage_k - torch_k).abs().max().item()

    output_dir = resolve_output_dir(Path(__file__).resolve().parent.name)
    cu_path = export_cuda_code(graph, inputs, output_dir)

    print("Mirage RoPE output shapes:",
          tuple(mirage_q.shape), tuple(mirage_k.shape))
    print(f"Max abs diff vs. PyTorch reference (query): {max_diff_q:.3e}")
    print(f"Max abs diff vs. PyTorch reference (key): {max_diff_k:.3e}")
    print(f"seq_len={seq_len}, head_dim={head_dim}, dtype={dtype_key}")
    print(f"Generated CUDA kernel saved to: {cu_path}")


if __name__ == "__main__":
    main()
