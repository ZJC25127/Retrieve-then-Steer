# Retrieve-then-Steer

**面向生成式VLA测试时适应的在线成功记忆方法**

Retrieve-then-Steer通过在线积累的成功经验，在推理阶段引导冻结的VLA策略。
检索得到的动作块经过DTW一致性过滤和加权聚合，形成动作先验，用于初始化
flow-matching采样过程。可选的VLAC critic根据预测任务进度控制记忆入库。

本仓库提供π0.5在LIBERO-10上的基线、memory＋DTW及VLAC评测代码。

[English](README.md) · [安装与评测](experiments/README.md) · [VLAC配置](experiments/vlac/README.md)

## 目录结构

```text
retrieve_then_steer_release/
├── openpi/                         OpenPI源码
│   └── third_party/libero/          LIBERO源码及仿真资产
├── experiments/
│   ├── scripts/                    策略适配、memory、DTW及评测入口
│   ├── resources/                  评测协议和memory参数
│   ├── runs/                       基线与memory＋DTW结果
│   ├── logs/                       评测及策略日志
│   ├── results_summary.json
│   ├── per_task.csv
│   └── vlac/
│       ├── VLAC/                   VLAC源码
│       ├── scripts/                评分服务、进度累计及入库逻辑
│       ├── configs/                各任务critic参数
│       ├── references/             每任务24张参考图
│       ├── runs/                   memory＋DTW＋VLAC结果
│       └── logs/                   评测及策略日志
├── tools/                          源码校验工具
├── tests/                          CPU测试
├── provenance/                     版本及文件校验记录
└── licenses/
```

## 环境安装

运行环境为Linux、NVIDIA GPU、CUDA和EGL。策略推理、LIBERO仿真和VLAC推理
分别使用独立的Python环境：

| 组件 | Python版本 | 安装说明 |
|---|---:|---|
| OpenPI策略服务 | 3.11 | [安装与评测](experiments/README.md) |
| LIBERO仿真 | 3.8 | [安装与评测](experiments/README.md) |
| VLAC评分服务 | 3.10 | [VLAC配置](experiments/vlac/README.md) |

模型权重需单独下载。π0.5使用官方 `pi05_libero` checkpoint；VLAC评测还需
`InternRobotics/VLAC-8b`。下载路径及依赖安装命令见上述文档。

## 运行评测

以下命令在仓库根目录执行。

```bash
export PI05_CKPT=/absolute/path/to/pi05_libero

# 基线
openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method baseline --checkpoint "$PI05_CKPT" --policy-gpu 0

# memory＋DTW
openpi/.venv/bin/python experiments/scripts/run_libero.py \
  --method dtw --checkpoint "$PI05_CKPT" --policy-gpu 0
```

VLAC评测：

```bash
export VLAC_CKPT=/absolute/path/to/VLAC-8b

openpi/.venv/bin/python experiments/vlac/scripts/run_vlac_reference_libero.py \
  --checkpoint "$PI05_CKPT" --critic-checkpoint "$VLAC_CKPT" \
  --critic-python experiments/vlac/env/bin/python --policy-gpu 0 --critic-gpu 1
```

常用选项：

| 选项 | 作用 |
|---|---|
| `--dry-run` | 打印执行计划，不启动模型 |
| `--tasks 0 --states 0,1` | 运行指定任务的两个回合 |
| `--openpi-root` | 指定OpenPI源码目录 |
| `--policy-python`、`--env-python` | 指定策略和仿真环境的Python |
| `--output`、`--log-root` | 指定新结果和日志目录 |
| `--save-videos` | 保存评测视频 |
| `--save-trajectories` | 保存VLAC评测的无损NPZ轨迹 |

默认依次运行10个任务。每个memory任务从空记忆开始，在该任务的50回合内
在线累积。新结果保存至对应的 `runs/<method>_<timestamp>/`，日志保存至
`logs/<method>_<timestamp>/`。视频和无损轨迹默认不保存。

## 源码来源

以下源码已包含在仓库中，并保留对应许可证：

| 目录 | 来源 | 提交版本 |
|---|---|---|
| `openpi/` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| `openpi/third_party/libero/` | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| `experiments/vlac/VLAC/` | [InternRobotics/VLAC](https://github.com/InternRobotics/VLAC) | `0f302d020555e320347c55b09431afe17ef5cdd4` |

检索和采样适配代码位于 `experiments/scripts/`；critic与进度入库适配代码
位于 `experiments/vlac/scripts/`。

## 许可证

项目代码采用 [Apache-2.0](LICENSE)。依赖源码和资产来源见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型权重与数据集遵循各自提供方条款。
