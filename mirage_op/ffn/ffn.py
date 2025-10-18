"""Feed-forward (FFN) kernel demo tailored to the Qwen3 0.6B configuration.

This script:
  1. Loads Qwen3 0.6B hyperparameters from ``mirage_op/config/config.json``.
  2. Builds a Mirage kernel graph that applies the gated SiLU FFN used in the
     model (gate_proj + up_proj followed by down_proj).
  3. Runs the graph on sample inputs to verify correctness against a PyTorch
     reference implementation.
  4. Exports the generated CUDA source into ``mirage_op/ffn/output_cu``.

Run:
    python mirage_op/ffn/ffn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import mirage as mi
import torch
import torch.nn.functional as F
from mirage.kernel import HARD_CODE  # type: ignore

from mirage_op.utils import (DEFAULT_BATCH, DEFAULT_SEED, load_model_config,
                             resolve_dtypes, resolve_output_dir)

DEFAULT_SEQ_LEN = 16


def build_ffn_graph(seq_len: int, hidden_size: int, intermediate_size: int,
                    mi_dtype: mi.dtype) -> tuple[mi.KNGraph, list[mi.DTensor]]:
    """Create a Mirage graph matching Qwen3's gated SiLU FFN."""
    graph = mi.new_kernel_graph()

    inputs: list[mi.DTensor] = []

    x = graph.new_input((seq_len, hidden_size), dtype=mi_dtype)
    inputs.append(x)
    w_gate = graph.new_input((hidden_size, intermediate_size), dtype=mi_dtype)
    inputs.append(w_gate)
    w_up = graph.new_input((hidden_size, intermediate_size), dtype=mi_dtype)
    inputs.append(w_up)
    w_down = graph.new_input((intermediate_size, hidden_size), dtype=mi_dtype)
    inputs.append(w_down)
    bias_gate = graph.new_input((seq_len, intermediate_size), dtype=mi_dtype)
    inputs.append(bias_gate)
    bias_up = graph.new_input((seq_len, intermediate_size), dtype=mi_dtype)
    inputs.append(bias_up)
    bias_down = graph.new_input((seq_len, hidden_size), dtype=mi_dtype)
    inputs.append(bias_down)

    gate_lin = graph.matmul(x, w_gate)
    gate_with_bias = graph.add(gate_lin, bias_gate)
    gate_act = graph.silu(gate_with_bias)

    up_lin = graph.matmul(x, w_up)
    up_with_bias = graph.add(up_lin, bias_up)

    gated = graph.mul(gate_act, up_with_bias)
    down_lin = graph.matmul(gated, w_down)
    output = graph.add(down_lin, bias_down)

    graph.mark_output(output)
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
    cu_path = output_dir / "ffn_kernel.cu"
    cu_path.write_text(program["code"] + HARD_CODE)
    return cu_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Mirage kernels.")

    model_cfg = load_model_config()
    hidden_size = int(model_cfg.get("hidden_size", 0))
    intermediate_size = int(model_cfg.get("intermediate_size", 0))
    if min(hidden_size, intermediate_size) == 0:
        raise ValueError("Model config missing FFN dimensions.")

    dtype_key = str(model_cfg.get("torch_dtype", "bfloat16"))
    dtype, mi_dtype = resolve_dtypes(dtype_key)

    torch.manual_seed(DEFAULT_SEED)
    device = torch.device("cuda")

    seq_len = min(DEFAULT_SEQ_LEN,
                  int(model_cfg.get("max_position_embeddings",
                                    DEFAULT_SEQ_LEN)))

    graph, graph_inputs = build_ffn_graph(seq_len, hidden_size,
                                          intermediate_size, mi_dtype)

    hidden_states = torch.randn(seq_len,
                                hidden_size,
                                dtype=dtype,
                                device=device)
    w_gate = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
    w_up = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
    w_down = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)

    bias_gate_vec = torch.randn(intermediate_size, dtype=dtype, device=device)
    bias_up_vec = torch.randn(intermediate_size, dtype=dtype, device=device)
    bias_down_vec = torch.randn(hidden_size, dtype=dtype, device=device)

    bias_gate = bias_gate_vec.unsqueeze(0).expand(seq_len, -1).contiguous()
    bias_up = bias_up_vec.unsqueeze(0).expand(seq_len, -1).contiguous()
    bias_down = bias_down_vec.unsqueeze(0).expand(seq_len, -1).contiguous()

    input_tensors: list[torch.Tensor] = [
        hidden_states,
        w_gate,
        w_up,
        w_down,
        bias_gate,
        bias_up,
        bias_down,
    ]

    if len(input_tensors) != len(graph_inputs):
        raise RuntimeError("Mismatch between graph inputs and provided tensors.")

    mirage_output = graph(inputs=input_tensors)[0]

    gate_lin = hidden_states @ w_gate + bias_gate_vec
    gate_act = F.silu(gate_lin)
    up_lin = hidden_states @ w_up + bias_up_vec
    gated = gate_act * up_lin
    torch_output = gated @ w_down + bias_down_vec

    max_diff = (mirage_output - torch_output).abs().max().item()

    output_dir = resolve_output_dir(CURRENT_DIR.name)
    cu_path = export_cuda_code(graph, input_tensors, output_dir)

    print("Mirage FFN output shape:", tuple(mirage_output.shape))
    print(f"Max abs diff vs. PyTorch reference: {max_diff:.3e}")
    print(f"seq_len={seq_len}, hidden={hidden_size}, dtype={dtype_key}")
    print(f"Generated CUDA kernel saved to: {cu_path}")


if __name__ == "__main__":
    main()
