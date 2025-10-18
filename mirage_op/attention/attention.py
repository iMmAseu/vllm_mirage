"""Attention kernel demo tailored to the Qwen3 0.6B configuration.

This script:
  1. Loads Qwen3 0.6B hyperparameters from ``mirage_op/config/config.json``.
  2. Builds a Mirage kernel graph that performs self-attention with grouped
     query attention (GQA), including Q/K/V projections and the output
     projection.
  3. Runs the graph on sample inputs to verify correctness against a PyTorch
     reference implementation.
  4. Exports the generated CUDA source into ``mirage_op/attention/output_cu``.

Run:
    python mirage_op/attention/attention.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import mirage as mi
import torch
from mirage.kernel import HARD_CODE  # type: ignore

from mirage_op.utils import (DEFAULT_BATCH, DEFAULT_SEED, load_model_config,
                             resolve_dtypes, resolve_output_dir)

DEFAULT_SEQ_LEN = 16


def build_attention_graph(
    seq_len: int,
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mi_dtype: mi.dtype,
) -> tuple[mi.KNGraph, List[mi.DTensor]]:
    """Create a Mirage graph that performs self-attention with GQA."""
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    graph = mi.new_kernel_graph()

    inputs: List[mi.DTensor] = []
    x = graph.new_input((seq_len, hidden_size), dtype=mi_dtype)
    inputs.append(x)
    x_t = graph.new_input((hidden_size, seq_len), dtype=mi_dtype)
    inputs.append(x_t)

    w_q = graph.new_input((hidden_size, q_dim), dtype=mi_dtype)
    inputs.append(w_q)
    w_k = graph.new_input((hidden_size, kv_dim), dtype=mi_dtype)
    inputs.append(w_k)
    w_v = graph.new_input((hidden_size, kv_dim), dtype=mi_dtype)
    inputs.append(w_v)
    w_o = graph.new_input((q_dim, hidden_size), dtype=mi_dtype)
    inputs.append(w_o)
    w_k_t = graph.new_input((kv_dim, hidden_size), dtype=mi_dtype)
    inputs.append(w_k_t)

    expand = graph.new_input((kv_dim, q_dim), dtype=mi_dtype)
    inputs.append(expand)
    expand_t = graph.new_input((q_dim, kv_dim), dtype=mi_dtype)
    inputs.append(expand_t)

    scale_matrix = graph.new_input((seq_len, seq_len), dtype=mi_dtype)
    inputs.append(scale_matrix)
    ones_col = graph.new_input((seq_len, 1), dtype=mi_dtype)
    inputs.append(ones_col)
    ones_row = graph.new_input((1, seq_len), dtype=mi_dtype)
    inputs.append(ones_row)

    select_mats: List[mi.DTensor] = []
    select_t_mats: List[mi.DTensor] = []
    for _ in range(num_q_heads):
        sel = graph.new_input((q_dim, head_dim), dtype=mi_dtype)
        sel_t = graph.new_input((head_dim, q_dim), dtype=mi_dtype)
        select_mats.append(sel)
        select_t_mats.append(sel_t)
        inputs.append(sel)
        inputs.append(sel_t)

    q_flat = graph.matmul(x, w_q)
    k_flat = graph.matmul(x, w_k)
    v_flat = graph.matmul(x, w_v)

    k_trans_base = graph.matmul(w_k_t, x_t)
    k_rep = graph.matmul(k_flat, expand)
    k_rep_t = graph.matmul(expand_t, k_trans_base)
    v_rep = graph.matmul(v_flat, expand)

    combined_context = None
    for idx in range(num_q_heads):
        sel = select_mats[idx]
        sel_t = select_t_mats[idx]

        q_head = graph.matmul(q_flat, sel)
        k_head = graph.matmul(k_rep, sel)
        k_head_t = graph.matmul(sel_t, k_rep_t)
        logits = graph.matmul(q_head, k_head_t)
        scaled_logits = graph.mul(logits, scale_matrix)
        exp_logits = graph.exp(scaled_logits)
        sum_logits = graph.matmul(exp_logits, ones_col)
        sum_expand = graph.matmul(sum_logits, ones_row)
        probs = graph.div(exp_logits, sum_expand)
        v_head = graph.matmul(v_rep, sel)
        context = graph.matmul(probs, v_head)
        context_proj = graph.matmul(context, sel_t)
        combined_context = context_proj if combined_context is None else graph.add(
            combined_context, context_proj)

    attn_output = graph.matmul(combined_context, w_o)
    graph.mark_output(attn_output)
    return graph, inputs


def export_cuda_code(graph: mi.KNGraph, input_tensors: list[torch.Tensor],
                     output_dir: Path) -> Path:
    device_props = torch.cuda.get_device_properties(input_tensors[0].device)
    target_cc = device_props.major * 10 + device_props.minor
    input_strides = [tensor.stride() for tensor in input_tensors]
    program = mi.generate_cuda_program(graph.cygraph,
                                       target_cc=target_cc,
                                       input_strides=input_strides)
    output_dir.mkdir(parents=True, exist_ok=True)
    cu_path = output_dir / "attention_kernel.cu"
    cu_path.write_text(program["code"] + HARD_CODE)
    return cu_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Mirage kernels.")

    model_cfg = load_model_config()
    hidden_size = int(model_cfg.get("hidden_size", 0))
    num_q_heads = int(model_cfg.get("num_attention_heads", 0))
    num_kv_heads = int(model_cfg.get("num_key_value_heads", 0))
    head_dim = int(model_cfg.get("head_dim", 0)) or hidden_size // max(
        num_q_heads, 1)

    if min(hidden_size, num_q_heads, num_kv_heads, head_dim) == 0:
        raise ValueError("Model config is missing required attention fields.")

    group_size = num_q_heads // num_kv_heads
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")

    seq_len = min(DEFAULT_SEQ_LEN,
                  int(model_cfg.get("max_position_embeddings",
                                    DEFAULT_SEQ_LEN)))

    dtype_key = str(model_cfg.get("torch_dtype", "bfloat16"))
    dtype, mi_dtype = resolve_dtypes(dtype_key)

    torch.manual_seed(DEFAULT_SEED)
    device = torch.device("cuda")

    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    hidden_states = torch.randn(seq_len,
                                hidden_size,
                                dtype=dtype,
                                device=device)
    hidden_states_t = hidden_states.transpose(0, 1).contiguous()

    w_q = torch.randn(hidden_size, q_dim, dtype=dtype, device=device)
    w_k = torch.randn(hidden_size, kv_dim, dtype=dtype, device=device)
    w_v = torch.randn(hidden_size, kv_dim, dtype=dtype, device=device)
    w_o = torch.randn(q_dim, hidden_size, dtype=dtype, device=device)
    w_k_t = w_k.transpose(0, 1).contiguous()

    expand = torch.zeros(kv_dim, q_dim, dtype=dtype, device=device)
    for kv_idx in range(num_kv_heads):
        for g in range(group_size):
            q_idx = kv_idx * group_size + g
            k_start = kv_idx * head_dim
            q_start = q_idx * head_dim
            expand[k_start:k_start + head_dim,
                   q_start:q_start + head_dim] = torch.eye(head_dim,
                                                           dtype=dtype,
                                                           device=device)
    expand_t = expand.transpose(0, 1).contiguous()

    scale_value = 1.0 / math.sqrt(float(head_dim))
    scale_matrix = torch.full((seq_len, seq_len),
                              scale_value,
                              dtype=dtype,
                              device=device)
    ones_col = torch.ones(seq_len, 1, dtype=dtype, device=device)
    ones_row = torch.ones(1, seq_len, dtype=dtype, device=device)

    select_mats: list[torch.Tensor] = []
    select_t_mats: list[torch.Tensor] = []
    eye = torch.eye(head_dim, dtype=dtype, device=device)
    for head_idx in range(num_q_heads):
        select = torch.zeros(q_dim, head_dim, dtype=dtype, device=device)
        start = head_idx * head_dim
        select[start:start + head_dim, :] = eye
        select_mats.append(select)
        select_t_mats.append(select.transpose(0, 1).contiguous())

    graph, graph_inputs = build_attention_graph(seq_len, hidden_size,
                                                num_q_heads, num_kv_heads,
                                                head_dim, mi_dtype)

    input_tensors: list[torch.Tensor] = [
        hidden_states,
        hidden_states_t,
        w_q,
        w_k,
        w_v,
        w_o,
        w_k_t,
        expand,
        expand_t,
        scale_matrix,
        ones_col,
        ones_row,
    ]
    for sel, sel_t in zip(select_mats, select_t_mats):
        input_tensors.append(sel)
        input_tensors.append(sel_t)

    if len(graph_inputs) != len(input_tensors):
        raise RuntimeError("Mismatch between graph inputs and provided tensors.")

    mirage_output = graph(inputs=input_tensors)[0]

    # Reference PyTorch attention with GQA
    q = hidden_states @ w_q
    q = q.view(seq_len, num_q_heads, head_dim).permute(1, 0, 2)  # (num_q_heads, seq, head_dim)
    k = hidden_states @ w_k
    k = k.view(seq_len, num_kv_heads, head_dim).permute(1, 0, 2)
    v = hidden_states @ w_v
    v = v.view(seq_len, num_kv_heads, head_dim).permute(1, 0, 2)
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=0)
        v = v.repeat_interleave(group_size, dim=0)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale_value
    probs = torch.softmax(scores, dim=-1)
    context = torch.matmul(probs, v)  # (num_q_heads, seq, head_dim)
    context = context.permute(1, 0, 2).reshape(seq_len, q_dim)
    torch_output = context @ w_o

    max_diff = (mirage_output - torch_output).abs().max().item()

    output_dir = resolve_output_dir(CURRENT_DIR.name)
    cu_path = export_cuda_code(graph, input_tensors, output_dir)

    print("Mirage Attention output shape:", tuple(mirage_output.shape))
    print(f"Max abs diff vs. PyTorch reference: {max_diff:.3e}")
    print(f"seq_len={seq_len}, heads={num_q_heads}/{num_kv_heads}, dtype={dtype_key}")
    print(f"Generated CUDA kernel saved to: {cu_path}")


if __name__ == "__main__":
    main()
