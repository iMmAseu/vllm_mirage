"""Embedding kernel demo tailored to the Qwen3 0.6B configuration.

This script:
  1. Loads Qwen3 0.6B hyperparameters from ``mirage_op/config/config.json``.
  2. Builds a Mirage kernel graph that maps one-hot token representations to
     hidden states via the model's embedding matrix.
  3. Runs the graph on sample inputs to verify correctness against
     ``torch.nn.functional.embedding``.
  4. Exports the generated CUDA source into ``mirage_op/embedding/output_cu``.

Run:
    python mirage_op/embedding/embedding.py
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

DEFAULT_SEQ_LEN = 8


def build_embedding_graph(num_tokens: int, vocab_size: int,
                          hidden_size: int, mi_dtype: mi.dtype) -> mi.KNGraph:
    """Create a Mirage graph that performs an embedding lookup via matmul."""
    graph = mi.new_kernel_graph()
    one_hot = graph.new_input((num_tokens, vocab_size), dtype=mi_dtype)
    weight = graph.new_input((vocab_size, hidden_size), dtype=mi_dtype)
    embedded = graph.matmul(one_hot, weight)
    graph.mark_output(embedded)
    return graph


def run_graph(graph: mi.KNGraph, inputs: list[torch.Tensor]) -> torch.Tensor:
    outputs = graph(inputs=inputs)
    if len(outputs) != 1:
        raise RuntimeError(f"Expected a single output, got {len(outputs)}")
    return outputs[0]


def export_cuda_code(graph: mi.KNGraph, input_tensors: list[torch.Tensor],
                     output_dir: Path) -> Path:
    device_props = torch.cuda.get_device_properties(input_tensors[0].device)
    target_cc = device_props.major * 10 + device_props.minor
    input_strides = [tensor.stride() for tensor in input_tensors]
    program = mi.generate_cuda_program(graph.cygraph,
                                       target_cc=target_cc,
                                       input_strides=input_strides)
    output_dir.mkdir(parents=True, exist_ok=True)
    cu_path = output_dir / "embedding_kernel.cu"
    cu_path.write_text(program["code"] + HARD_CODE)
    return cu_path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Mirage kernels.")

    model_cfg = load_model_config()
    vocab_size = int(model_cfg.get("vocab_size", 0))
    hidden_size = int(model_cfg.get("hidden_size", 0))
    if vocab_size == 0 or hidden_size == 0:
        raise ValueError("Model config must define 'vocab_size' and 'hidden_size'.")

    dtype_key = str(model_cfg.get("torch_dtype", "bfloat16"))
    dtype, mi_dtype = resolve_dtypes(dtype_key)

    torch.manual_seed(DEFAULT_SEED)
    device = torch.device("cuda")

    num_tokens = DEFAULT_BATCH * DEFAULT_SEQ_LEN
    graph = build_embedding_graph(num_tokens, vocab_size, hidden_size, mi_dtype)

    token_ids = torch.randint(low=0,
                              high=vocab_size,
                              size=(DEFAULT_BATCH, DEFAULT_SEQ_LEN),
                              device=device,
                              dtype=torch.long)
    one_hot = F.one_hot(token_ids, num_classes=vocab_size).to(dtype=dtype)
    one_hot = one_hot.view(num_tokens, vocab_size).contiguous()

    weight = torch.randn(vocab_size,
                         hidden_size,
                         dtype=dtype,
                         device=device)

    mirage_output = run_graph(graph, [one_hot, weight])
    mirage_embeddings = mirage_output.view(DEFAULT_BATCH, DEFAULT_SEQ_LEN,
                                           hidden_size)

    torch_embeddings = F.embedding(token_ids, weight)

    max_diff = (mirage_embeddings - torch_embeddings).abs().max().item()

    output_dir = resolve_output_dir(CURRENT_DIR.name)
    cu_path = export_cuda_code(graph, [one_hot, weight], output_dir)

    print("Mirage Embedding output shape:", tuple(mirage_embeddings.shape))
    print(f"Max abs diff vs. PyTorch reference: {max_diff:.3e}")
    print(f"seq_len={DEFAULT_SEQ_LEN}, vocab_size={vocab_size}, dtype={dtype_key}")
    print(f"Generated CUDA kernel saved to: {cu_path}")


if __name__ == "__main__":
    main()
