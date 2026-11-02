# π0.5 Evaluation

This document covers environment setup, model weights and LIBERO-10 evaluation
for the baseline and memory + DTW configurations. Commands are executed from
the repository root.

## Requirements

- Linux with an NVIDIA GPU, a compatible CUDA driver and EGL/OpenGL libraries.
- Git LFS and `uv`.
- Python 3.11 for policy inference and Python 3.8 for LIBERO simulation.

OpenPI source is included at `openpi/`, commit
`215abfb217dbac7d5f1273282331b9b1866c0479`, from
[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).
LIBERO source and simulator assets are included at `openpi/third_party/libero/`,
commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, from
[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

## Installation

Policy environment:

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e .
cd ..
```

Simulator environment:

```bash
uv venv --python 3.8 openpi/examples/libero/.venv
uv pip sync --python openpi/examples/libero/.venv/bin/python \
  openpi/examples/libero/requirements.txt openpi/third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
uv pip install --python openpi/examples/libero/.venv/bin/python \
  -e openpi/packages/openpi-client -e openpi/third_party/libero
```

The launcher generates a LIBERO configuration pointing to the included simulator
assets. Dependency versions are recorded in `provenance/environment.json`.
Platform-specific simulator instructions are available in `openpi/examples/libero/README.md`.

## Model weights

Download the official `pi05_libero` checkpoint using the OpenPI downloader:

```bash
openpi/.venv/bin/python -c \
  "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero/'))"
export PI05_CKPT=/absolute/path/to/pi05_libero
```

`PI05_CKPT` must point to the checkpoint directory containing parameters and
normalization assets.

## Evaluation

Baseline:

```bash
openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method baseline --checkpoint "$PI05_CKPT" --policy-gpu 0
```

Memory + DTW:

```bash
openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method dtw --checkpoint "$PI05_CKPT" --policy-gpu 0
```

Append `--dry-run` to print commands without launching models. Append
`--tasks 0 --states 0,1` to evaluate a two-episode subset.
`--openpi-root`, `--policy-python` and `--env-python` select existing installations.

## Results and logs

| Configuration | Results | Logs |
|---|---|---|
| Baseline | `experiments/runs/pi05_baseline/` | `experiments/logs/pi05_baseline/` |
| Memory + DTW | `experiments/runs/pi05_memory_dtw/` | `experiments/logs/pi05_memory_dtw/` |

New evaluations write results to `experiments/runs/<method>_<timestamp>/` and logs
to `experiments/logs/<method>_<timestamp>/`. `--output` and `--log-root` specify
alternative locations. Existing output directories are not reused.

Video saving is disabled by default. `--save-videos` enables it.

The summary and task-level comparison are stored in `experiments/results_summary.json`
and `experiments/per_task.csv`.
