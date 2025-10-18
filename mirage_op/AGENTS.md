# Repository Guidelines

## Project Structure & Module Organization
Operator demos live under `mirage_op/`, one folder per feature (for example `rmsnorm/`). Each folder should provide a runnable driver script and supporting helpers; generated CUDA lands in `output_cu/` and must stay aligned with the Mirage graph that created it. `attention/` and `RoPE/` are reserved stubs—reuse the `rmsnorm` layout when populating them and keep operator assets scoped to their folder.

## Build, Test, and Development Commands
- `python -m venv _venv && source _venv/bin/activate` sets up the shared dev environment.
- `pip install -e ../mirage` plus `pip install torch --index-url https://download.pytorch.org/whl/cu118` installs Mirage and CUDA-enabled PyTorch for local runs.
- `python rmsnorm/rmsnorm.py --batch 16 --hidden 4096` compiles and executes the RMSNorm + linear kernel, writing `rmsnorm/output_cu/rmsnorm_kernel.cu`.
- `python -m pytest tests -k rmsnorm` (once tests exist) runs targeted checks; use `CUDA_VISIBLE_DEVICES=<id>` when debugging on multi-GPU hosts.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints on public helpers, and concise docstrings describing the kernel demo. Name kernels with operator prefixes (`rmsnorm_kernel.cu`, `rope_kernel.cu`) and keep entrypoints runnable via `python <module>/<module>.py`. Shared helpers should remain inside their operator package to keep dependencies local.

## Testing Guidelines
Each demo must check results against a PyTorch reference path, mirroring `run_graph` vs. `reference_rmsnorm_linear`. Add GPU regression tests under `tests/<operator>/test_<feature>.py`, guarded with `pytest.importorskip("torch.cuda")`. Record explicit tolerances (`max_diff <= 1e-3`) and note the GPU used whenever behavior varies between architectures.

## Commit & Pull Request Guidelines
Adopt conventional commits (`feat:`, `fix:`, `refactor:`) and keep messages imperative with context on the affected operator. Group generated CUDA with the Python change that produced it so reviewers can trace cause and effect. Pull requests should describe the operator update, list validation commands (with batch/hidden sizes), note GPU model + driver, and attach any accuracy or timing deltas. Run local demos and tests before requesting review, and tag reviewers familiar with the touched folder.

## CUDA & Export Tips
Exported `.cu` files mirror the Mirage graph; if you hand-edit them, port the change back into the Python definition before regenerating. Set `MIRAGE_HOME` to the sibling `../mirage` checkout so native extensions can load. Ship small, reviewable diffs, documenting expected speedups or accuracy shifts alongside the export.
