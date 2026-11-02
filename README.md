# Retrieve-then-Steer

**Online Success Memory for Test-Time Adaptation of Generative VLAs**

Retrieve-then-Steer adapts a frozen generative VLA at inference time using an
online memory of successful experience. Retrieved action chunks are filtered
by DTW consistency, aggregated into an action prior, and used to initialize
the flow-matching sampler. An optional VLAC critic controls memory admission
using predicted task progress.

This repository provides the π0.5 LIBERO-10 implementation and the files required
for baseline, memory + DTW, and VLAC evaluation.

[中文文档](README.zh-CN.md) · [Installation and evaluation](experiments/README.md) · [VLAC](experiments/vlac/README.md)

## Directory structure

```text
retrieve_then_steer_release/
├── openpi/                         OpenPI source
│   └── third_party/libero/          LIBERO source and simulator assets
├── experiments/
│   ├── scripts/                    Policy adapters, memory, DTW, evaluation
│   ├── resources/                  Evaluation protocol and memory configuration
│   ├── runs/                       Baseline and memory + DTW results
│   ├── logs/                       Evaluator and policy logs
│   ├── results_summary.json
│   ├── per_task.csv
│   └── vlac/
│       ├── VLAC/                   VLAC source
│       ├── scripts/                Critic service, progress and admission logic
│       ├── configs/                Task-specific critic configurations
│       ├── references/             24 reference frames per task
│       ├── runs/                   Memory + DTW + VLAC results
│       └── logs/                   Evaluator and policy logs
├── tools/                          Source verification
├── tests/                          CPU tests
├── provenance/                     Versions and file checksums
└── licenses/
```

## Installation and evaluation

The runtime requires Linux, an NVIDIA GPU, CUDA and EGL. Policy inference,
LIBERO simulation and VLAC inference use separate Python environments.
Model checkpoints must be downloaded separately.

1. Install the policy and simulator environments following
   [experiments/README.md](experiments/README.md).
2. Download the official `pi05_libero` checkpoint.
3. Run an evaluation from the repository root:

```bash
export PI05_CKPT=/absolute/path/to/pi05_libero

openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method baseline --checkpoint "$PI05_CKPT" --policy-gpu 0

openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method dtw --checkpoint "$PI05_CKPT" --policy-gpu 0
```

For VLAC evaluation, install the critic environment and download `VLAC-8b` as
described in [experiments/vlac/README.md](experiments/vlac/README.md):

```bash
export VLAC_CKPT=/absolute/path/to/VLAC-8b

openpi/.venv/bin/python experiments/vlac/scripts/run_vlac_reference_libero.py \
  --checkpoint "$PI05_CKPT" --critic-checkpoint "$VLAC_CKPT" \
  --critic-python experiments/vlac/env/bin/python --policy-gpu 0 --critic-gpu 1
```

`--dry-run` prints the execution plan. `--tasks 0 --states 0,1` runs a
two-episode functional check.

## Source dependencies

The following source dependencies are included in the repository. Their license
notices are retained in the corresponding directories and in `licenses/`.

| Directory | Source | Commit |
|---|---|---|
| `openpi/` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| `openpi/third_party/libero/` | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| `experiments/vlac/VLAC/` | [InternRobotics/VLAC](https://github.com/InternRobotics/VLAC) | `0f302d020555e320347c55b09431afe17ef5cdd4` |

Retrieval, sampler integration and evaluation adapters are located under
`experiments/scripts/`. Critic and progress-based admission adapters are under
`experiments/vlac/scripts/`.

## License

Project code is licensed under [Apache-2.0](LICENSE). Dependency licenses and
asset attribution are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Model weights and datasets are subject to their respective provider terms.
