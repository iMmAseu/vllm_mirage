"""RMSNorm + Linear kernel demo tailored to the Qwen3 0.6B configuration.

This script:
  1. Loads Qwen3 0.6B hyperparameters from ``mirage_op/config/config.json``.
  2. Builds a Mirage kernel graph that applies RMSNorm over the hidden size and
     projects to the model's intermediate dimension.
  3. Compiles and runs the graph on sample inputs to verify correctness.
  4. Exports the generated CUDA source into ``mirage_op/rmsnorm/output_cu``.

Run:
    python mirage_op/rmsnorm/rmsnorm.py
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
from mirage.kernel import HARD_CODE  # type: ignore

from mirage_op.utils import (DEFAULT_BATCH, DEFAULT_EPS, DEFAULT_SEED,
                             load_model_config, resolve_dtypes,
                             resolve_output_dir)

VECTOR_TILE = 64


def build_rmsnorm_graph(batch: int, hidden: int, out_features: int,
                        mi_dtype: mi.dtype) -> mi.KNGraph:
    """Create a Mirage graph that computes matmul(RMSNorm(x), W).
    """
    if hidden % VECTOR_TILE != 0 or out_features % VECTOR_TILE != 0:
        raise ValueError(
            f"hidden ({hidden}) and out_features ({out_features}) must be "
            f"multiples of {VECTOR_TILE} for this demo.")

    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(batch, hidden), dtype=mi_dtype)
    w = graph.new_input(dims=(hidden, out_features), dtype=mi_dtype)

    grid_x = max(out_features // VECTOR_TILE, 1)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(grid_x, 1, 1),
        block_dim=(128, 1, 1),
        forloop_range=VECTOR_TILE,
        reduction_dimx=VECTOR_TILE,
    )
    t_x = tb_graph.new_input(dtensor=x,
                             input_map=(-1, -1, -1),
                             forloop_dim=1)
    t_w = tb_graph.new_input(dtensor=w,
                             input_map=(1, -1, -1),
                             forloop_dim=0)
    # Compute matmul first
    t_mat = tb_graph.matmul(t_x, t_w)
    # Compute RMS of input via forloop accumulation
    t_acc_x = tb_graph.forloop_accum(t_x, "rms")
    # Accumulate matmul result (without reduction)
    t_acc_mat = tb_graph.forloop_accum(t_mat)
    # Normalize: divide accumulated matmul by RMS
    t_out = tb_graph.div(t_acc_mat, t_acc_x)
    tb_graph.new_output(stensor=t_out, output_map=(1, -1, -1))

    outputs = graph.customized([x, w], tb_graph)
    graph.mark_output(outputs[0])
    return graph


def run_graph(graph: mi.KNGraph, input_tensor: torch.Tensor,
              weight_tensor: torch.Tensor) -> torch.Tensor:
    """Compile the Mirage graph (if needed) and execute it once."""
    outputs = graph(inputs=[input_tensor, weight_tensor])
    if len(outputs) != 1:
        raise RuntimeError(f"Expected a single output, got {len(outputs)}")
    return outputs[0]


def reference_rmsnorm_linear(x: torch.Tensor, weight: torch.Tensor,
                             eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Pure PyTorch RMSNorm + matmul for correctness checking."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    scaled = x * torch.rsqrt(variance + eps)
    return torch.matmul(scaled.to(dtype=weight.dtype), weight)


def export_cuda_code(graph: mi.KNGraph, input_tensor: torch.Tensor,
                     weight_tensor: torch.Tensor,
                     output_dir: Path) -> Path:
    """Dump the generated CUDA kernel to disk."""
    device_props = torch.cuda.get_device_properties(input_tensor.device)
    target_cc = device_props.major * 10 + device_props.minor
    input_strides = [input_tensor.stride(), weight_tensor.stride()]
    program = mi.generate_cuda_program(graph.cygraph,
                                       target_cc=target_cc,
                                       input_strides=input_strides)
    output_dir.mkdir(parents=True, exist_ok=True)
    cu_path = output_dir / "rmsnorm_kernel.cu"
    cu_path.write_text(program["code"] + HARD_CODE)
    return cu_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Mirage kernels.")

    model_cfg = load_model_config()
    hidden = int(model_cfg.get("hidden_size", 0))
    if hidden == 0:
        raise ValueError("Model config is missing a valid 'hidden_size'.")

    out_features = int(model_cfg.get("intermediate_size", hidden))
    eps = float(model_cfg.get("rms_norm_eps", DEFAULT_EPS))

    dtype_key = str(model_cfg.get("torch_dtype", "bfloat16"))
    dtype, mi_dtype = resolve_dtypes(dtype_key)

    torch.manual_seed(DEFAULT_SEED)
    device = torch.device("cuda")

    graph = build_rmsnorm_graph(DEFAULT_BATCH, hidden, out_features, mi_dtype)

    sample_input = torch.randn(DEFAULT_BATCH,
                               hidden,
                               dtype=dtype,
                               device=device)
    weight = torch.randn(hidden,
                         out_features,
                         dtype=dtype,
                         device=device)

    mirage_output = run_graph(graph, sample_input, weight)
    torch_output = reference_rmsnorm_linear(sample_input, weight, eps=eps)

    max_diff = (mirage_output - torch_output).abs().max().item()

    output_dir = resolve_output_dir(Path(__file__).resolve().parent.name)
    cu_path = export_cuda_code(graph, sample_input, weight, output_dir)

    print("Mirage RMSNorm + Linear output shape:",
          tuple(mirage_output.shape))
    print(f"Max abs diff vs. PyTorch reference (eps={eps}): "
          f"{max_diff:.3e}")
    print(f"hidden_size={hidden}, out_features={out_features}, dtype={dtype_key}")
    print(f"Generated CUDA kernel saved to: {cu_path}")


if __name__ == "__main__":
    main()
