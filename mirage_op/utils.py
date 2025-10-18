"""Shared helpers for Mirage operator demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import mirage as mi
import torch

ROOT_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = ROOT_DIR / "config" / "config.json"

TORCH_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

TORCH_TO_MIRAGE_DTYPE: Dict[torch.dtype, mi.dtype] = {
    torch.float16: mi.float16,
    torch.bfloat16: mi.bfloat16,
}

DEFAULT_BATCH = 1
DEFAULT_SEED = 42
DEFAULT_EPS = 1e-6


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    """Load the shared Qwen3 0.6B configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Model config not found at {path}")
    return json.loads(path.read_text())


def resolve_dtypes(torch_dtype_key: str) -> Tuple[torch.dtype, mi.dtype]:
    """Map a torch dtype string to both torch and Mirage dtypes."""
    key = torch_dtype_key.lower()
    if key not in TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype '{torch_dtype_key}'.")
    torch_dtype = TORCH_DTYPE_MAP[key]
    mi_dtype = TORCH_TO_MIRAGE_DTYPE.get(torch_dtype)
    if mi_dtype is None:
        raise ValueError(f"No Mirage dtype mapping for torch dtype {torch_dtype}.")
    return torch_dtype, mi_dtype


def resolve_output_dir(module: str) -> Path:
    """Return the standard output_cu path for a demo module."""
    return ROOT_DIR / module / "output_cu"
