"""SiLU activation kernel demo tailored to the Qwen3 0.6B configuration.

This script:
  1. Loads Qwen3 0.6B hyperparameters from ``mirage_op/config/config.json``.
  2. Builds a Mirage kernel graph that applies the SiLU nonlinearity to the
     model's intermediate activations.
  3. Runs the graph on sample inputs to verify correctness against a PyTorch
     reference path.
  4. Exports the generated CUDA source into ``mirage_op/silu/output_cu``.

Run:
    python mirage_op/silu/silu.py
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


def build_silu_graph(batch: int, features: int,
                     mi_dtype: mi.dtype) -> mi.KNGraph:
    """Create a Mirage graph that applies SiLU elementwise."""
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(batch, features), dtype=mi_dtype)
    y = graph.silu(x)
    graph.mark_output(y)
    return graph


def run_graph(graph: mi.KNGraph, input_tensor: torch.Tensor) -> torch.Tensor:
    outputs = graph(inputs=[input_tensor])
    if len(outputs) != 1:
        raise RuntimeError(f"Expected a single output, got {len(outputs)}")
    return outputs[0]


def export_cuda_code(graph: mi.KNGraph, input_tensor: torch.Tensor,
                     output_dir: Path) -> Path:
    device_props = torch.cuda.get_device_properties(input_tensor.device)
    target_cc = device_props.major * 10 + device_props.minor
    input_strides = [input_tensor.stride()]
    program = mi.generate_cuda_program(graph.cygraph,
                                       target_cc=target_cc,
                                       input_strides=input_strides)
    output_dir.mkdir(parents=True, exist_ok=True)
    cu_path = output_dir / "silu_kernel.cu"
    cu_path.write_text(program["code"] + HARD_CODE)
    return cu_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Mirage kernels.")

    model_cfg = load_model_config()
    intermediate_size = int(model_cfg.get("intermediate_size",
                                          model_cfg.get("hidden_size", 0)))
    if intermediate_size == 0:
        raise ValueError("Model config is missing 'intermediate_size'.")

    dtype_key = str(model_cfg.get("torch_dtype", "bfloat16"))
    dtype, mi_dtype = resolve_dtypes(dtype_key)

    torch.manual_seed(DEFAULT_SEED)
    device = torch.device("cuda")

    graph = build_silu_graph(DEFAULT_BATCH, intermediate_size, mi_dtype)

    sample_input = torch.randn(DEFAULT_BATCH,
                               intermediate_size,
                               dtype=dtype,
                               device=device)

    mirage_output = run_graph(graph, sample_input)
    torch_output = F.silu(sample_input)

    max_diff = (mirage_output - torch_output).abs().max().item()

    output_dir = resolve_output_dir(CURRENT_DIR.name)
    cu_path = export_cuda_code(graph, sample_input, output_dir)

    print("Mirage SiLU output shape:", tuple(mirage_output.shape))
    print(f"Max abs diff vs. PyTorch reference: {max_diff:.3e}")
    print(f"intermediate_size={intermediate_size}, dtype={dtype_key}")
    print(f"Generated CUDA kernel saved to: {cu_path}")


if __name__ == "__main__":
    main()
