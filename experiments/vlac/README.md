# VLAC Evaluation

VLAC provides pairwise task-progress scores for memory admission. This document
covers critic installation, reference images and the π0.5 memory + DTW + VLAC
evaluation. Commands are executed from the repository root.

## Source dependency

VLAC source is included at `experiments/vlac/VLAC/`, from
[InternRobotics/VLAC](https://github.com/InternRobotics/VLAC), commit
`0f302d020555e320347c55b09431afe17ef5cdd4`.

The scoring service, policy adapter, evaluator and progress accumulator are located
in `experiments/vlac/scripts/`. Task rules are in
`experiments/vlac/configs/pi05_reference24/`.

## Installation

Install the policy and simulator environments following
[experiments/README.md](../README.md). The critic uses a separate Python 3.10
environment:

```bash
uv venv --python 3.10 experiments/vlac/env
uv pip install --python experiments/vlac/env/bin/python \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python experiments/vlac/env/bin/python -r requirements-critic.txt
uv pip install --python experiments/vlac/env/bin/python \
  flash-attn==2.7.4.post1 --no-build-isolation
uv pip install --python experiments/vlac/env/bin/python -e experiments/vlac/VLAC --no-deps
```

FlashAttention requires a build compatible with the installed CUDA and Torch
versions. Building from source may require the CUDA development toolkit.
VLAC dependency details are documented in `experiments/vlac/VLAC/README.md`.

Download [InternRobotics/VLAC-8b](https://huggingface.co/InternRobotics/VLAC-8b)
separately and set its local checkpoint path.

## Evaluation

```bash
export PI05_CKPT=/absolute/path/to/pi05_libero
export VLAC_CKPT=/absolute/path/to/VLAC-8b

openpi/.venv/bin/python experiments/vlac/scripts/run_vlac_reference_libero.py \
  --checkpoint "$PI05_CKPT" --critic-checkpoint "$VLAC_CKPT" \
  --critic-python experiments/vlac/env/bin/python --policy-gpu 0 --critic-gpu 1
```

The common evaluation entry point supports the same configuration:

```bash
openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method vlac --checkpoint "$PI05_CKPT" --critic-checkpoint "$VLAC_CKPT" \
  --critic-python experiments/vlac/env/bin/python --policy-gpu 0 --critic-gpu 1
```

`--dry-run` prints the execution plan. `--tasks 0 --states 0,1` evaluates a
two-episode subset. One policy server is started per task; the critic service is
shared across tasks. Policy and critic ports must differ. The critic service binds
to the loopback interface.

## Reference images

Reference images are included in
`experiments/vlac/references/libero10_reference24/`.
Task configurations are located in `experiments/vlac/configs/pi05_reference24/`.

## Results and logs

- Results: `experiments/vlac/runs/pi05_memory_dtw_vlac/`.
- Logs: `experiments/vlac/logs/pi05_memory_dtw_vlac/`.
- New results: `experiments/vlac/runs/vlac_<timestamp>/`.
- New logs: `experiments/vlac/logs/vlac_<timestamp>/`.

Video and trajectory saving are disabled by default. Use `--save-videos` and
`--save-trajectories` to enable these outputs.
