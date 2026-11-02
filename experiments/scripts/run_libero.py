"""Reproduce the three pi05 LIBERO-10 configurations without a cluster scheduler."""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
VLAC = EXP / "vlac"


def ids(value, limit):
    try:
        values = [int(v) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or len(set(values)) != len(values) or any(v < 0 or v >= limit for v in values):
        raise argparse.ArgumentTypeError("Use distinct IDs in 0..%d" % (limit - 1))
    return values


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["baseline", "dtw", "vlac"], required=True)
    p.add_argument("--openpi-root", type=Path, default=ROOT / "openpi")
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="Downloaded official pi05_libero directory"
    )
    p.add_argument("--policy-python", type=Path)
    p.add_argument("--env-python", type=Path)
    p.add_argument("--critic-python", type=Path)
    p.add_argument(
        "--critic-checkpoint", type=Path, help="Downloaded VLAC-8b directory; required for vlac"
    )
    p.add_argument("--policy-gpu", default="0", help="One physical CUDA device ID")
    p.add_argument("--critic-gpu", default="1", help="One physical CUDA device ID")
    p.add_argument("--port", type=int, default=18785)
    p.add_argument("--critic-port", type=int, default=19891)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tasks", type=lambda v: ids(v, 10), default=list(range(10)))
    p.add_argument("--states", type=lambda v: ids(v, 50), default=list(range(50)))
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--log-root",
        type=Path,
        help="Separate log directory; defaults next to the selected runs tree",
    )
    p.add_argument("--save-videos", action="store_true")
    p.add_argument(
        "--save-trajectories",
        action="store_true",
        help="VLAC only; optional lossless NPZ collection",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without loading models or writing files",
    )
    args = p.parse_args(argv)
    if args.method == "vlac" and args.critic_checkpoint is None:
        p.error("--critic-checkpoint is required for vlac")
    if args.save_trajectories and args.method != "vlac":
        p.error("--save-trajectories applies only to vlac")
    if not 0 <= args.seed < 2**32 or not all(
        1 <= v <= 65535 for v in [args.port, args.critic_port]
    ):
        p.error("Invalid seed or port")
    if args.method == "vlac" and args.port == args.critic_port:
        p.error("Policy and critic need distinct ports")
    if not args.policy_gpu.isdigit() or not args.critic_gpu.isdigit():
        p.error("Each GPU argument must be one nonnegative device ID")
    args.openpi_root = args.openpi_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.policy_python = (
        (args.policy_python or args.openpi_root / ".venv/bin/python").expanduser().resolve()
    )
    args.env_python = (
        (args.env_python or args.openpi_root / "examples/libero/.venv/bin/python")
        .expanduser()
        .resolve()
    )
    if args.critic_python:
        args.critic_python = args.critic_python.expanduser().resolve()
    if args.method == "vlac" and args.critic_python is None:
        p.error("--critic-python is required for vlac")
    if args.critic_checkpoint:
        args.critic_checkpoint = args.critic_checkpoint.expanduser().resolve()
    args.output = (
        (
            args.output
            or (VLAC if args.method == "vlac" else EXP)
            / "runs"
            / (args.method + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        )
        .expanduser()
        .resolve()
    )
    protected = [
        EXP / "runs/pi05_baseline",
        EXP / "runs/pi05_memory_dtw",
        VLAC / "runs/pi05_memory_dtw_vlac",
        EXP / "logs/pi05_baseline",
        EXP / "logs/pi05_memory_dtw",
        VLAC / "logs/pi05_memory_dtw_vlac",
        VLAC / "references",
        EXP / "resources",
        VLAC / "configs",
        ROOT / "openpi",
        VLAC / "VLAC",
    ]
    args.log_root = (
        (args.log_root or ((VLAC if args.method == "vlac" else EXP) / "logs" / args.output.name))
        .expanduser()
        .resolve()
    )
    if any(
        candidate == p or candidate.is_relative_to(p)
        for candidate in [args.output, args.log_root]
        for p in protected
    ):
        p.error("Recorded results, reference assets, source and configs are read-only inputs")
    return args


def build_plan(args):
    """Pure command construction: safe to inspect on a machine without GPUs."""
    memory = json.loads((EXP / "resources/memory_configs/pi05_memory_dtw.json").read_text())
    env = dict(
        os.environ,
        RTS_OPENPI_ROOT=str(args.openpi_root),
        CUDA_VISIBLE_DEVICES=args.policy_gpu,
        MUJOCO_EGL_DEVICE_ID="0",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        PYTHONUNBUFFERED="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.22",
        XLA_FLAGS="--xla_gpu_autotune_level=0",
        PYTHONHASHSEED="0",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        LIBERO_EVAL_MODE="stable" if args.method == "baseline" else "memory",
        LIBERO_EVAL_SEED=str(args.seed),
        LIBERO_MEMORY_VIEWS="base",
        LIBERO_MEMORY_CONFIG=json.dumps(memory if args.method != "baseline" else {}),
        LIBERO_EVAL_STATES=",".join(map(str, args.states)),
        LIBERO_CONFIG_PATH=str(args.output / "libero_config"),
        PYTHONPATH=str(args.openpi_root / "third_party/libero"),
        TMPDIR=str(args.output / "tmp"),
        RTS_SAVE_VIDEOS="1" if args.save_videos else "0",
    )
    batches = [args.tasks] if args.method == "baseline" else [[task] for task in args.tasks]
    jobs = []
    for tasks in batches:
        name = "all_tasks" if args.method == "baseline" else "task_%02d" % tasks[0]
        directory = args.output / name
        logs = args.log_root / name
        jobenv = dict(env, LIBERO_EVAL_TASKS=",".join(map(str, tasks)))
        server = [
            str(args.policy_python),
            str(
                (VLAC if args.method == "vlac" else EXP)
                / "scripts"
                / ("openpi_vlac_server.py" if args.method == "vlac" else "stable_server.py")
            ),
            "--port",
            str(args.port),
            "policy:checkpoint",
            "--policy.config=pi05_libero",
            "--policy.dir=" + str(args.checkpoint),
        ]
        if args.method == "vlac":
            client = [
                str(args.env_python),
                str(VLAC / "scripts/openpi_vlac_collect_client.py"),
                "--port",
                str(args.port),
                "--task",
                str(tasks[0]),
                "--states",
                ",".join(map(str, args.states)),
                "--seed",
                str(args.seed),
                "--mode",
                "label",
                "--critic-url",
                "http://127.0.0.1:" + str(args.critic_port),
                "--config",
                str(VLAC / ("configs/pi05_reference24/task_%02d.json" % tasks[0])),
                "--run",
                str(directory),
            ]
            if args.save_videos:
                client.append("--save-videos")
            if args.save_trajectories:
                client.append("--save-trajectories")
        else:
            client = [
                str(args.env_python),
                str(EXP / "scripts/libero_client.py"),
                "--args.host=127.0.0.1",
                "--args.port=" + str(args.port),
                "--args.task-suite-name=libero_10",
                "--args.seed=" + str(args.seed),
                "--args.resize-size=224",
                "--args.replan-steps=5",
                "--args.num-trials-per-task=" + str(len(args.states)),
                "--args.video-out-path=" + str(directory / "videos"),
            ]
        jobs.append(
            dict(
                tasks=tasks,
                directory=str(directory),
                log_directory=str(logs),
                server=server,
                client=client,
                env=jobenv,
            )
        )
    critic = None
    if args.method == "vlac":
        critic = dict(
            command=[
                str(args.critic_python),
                str(VLAC / "scripts/vlac_server.py"),
                "--port",
                str(args.critic_port),
                "--model",
                str(args.critic_checkpoint),
                "--config",
                str(VLAC / "configs/pi05_reference24/task_00.json"),
            ],
            env=dict(env, CUDA_VISIBLE_DEVICES=args.critic_gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
        )
    return dict(
        method=args.method,
        seed=args.seed,
        tasks=args.tasks,
        states=args.states,
        expected_episodes=len(args.tasks) * len(args.states),
        memory_config=memory if args.method != "baseline" else None,
        jobs=jobs,
        critic=critic,
    )


def public_plan(plan):
    # Never serialize the caller's environment, tokens or account settings.
    keep = {
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_EGL_DEVICE_ID",
        "XLA_FLAGS",
        "LIBERO_MEMORY_VIEWS",
        "LIBERO_EVAL_TASKS",
        "LIBERO_EVAL_STATES",
        "LIBERO_EVAL_MODE",
        "LIBERO_MEMORY_CONFIG",
        "LIBERO_EVAL_SEED",
        "RTS_OPENPI_ROOT",
        "RTS_SAVE_VIDEOS",
    }
    out = {k: v for k, v in plan.items() if k not in ["jobs", "critic"]}
    out["jobs"] = [
        dict(j, env={k: v for k, v in j["env"].items() if k in keep}) for j in plan["jobs"]
    ]
    if plan["critic"]:
        out["critic"] = dict(
            command=plan["critic"]["command"], gpu=plan["critic"]["env"]["CUDA_VISIBLE_DEVICES"]
        )
    return out


def stop(p):
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def ready(url, process):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Service exited; inspect its log")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("Service readiness timed out")


def check_log(path, expected):
    text = path.read_text(errors="replace")
    values = re.findall(r"INFO:root:Success: (True|False)\s*(?:\n|$)", text)
    if len(values) != expected or "Caught exception:" in text:
        raise RuntimeError("Incomplete evaluation; inspect " + str(path))
    return [v == "True" for v in values]


def main(argv=None):
    args = arguments(argv)
    plan = build_plan(args)
    print(json.dumps(public_plan(plan), indent=2), flush=True)
    if args.dry_run:
        return
    for p in [
        args.checkpoint,
        args.openpi_root / "examples/libero/main.py",
        args.policy_python,
        args.env_python,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)
    if args.method == "vlac":
        for p in [args.critic_python, args.critic_checkpoint]:
            if not p.exists():
                raise FileNotFoundError(p)
    for port in [args.port] + ([args.critic_port] if args.method == "vlac" else []):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("Port already in use: " + str(port))
    args.output.mkdir(parents=True, exist_ok=False)
    args.log_root.mkdir(parents=True, exist_ok=False)
    (args.output / "tmp").mkdir()
    cfg = args.output / "libero_config"
    cfg.mkdir()
    base = args.openpi_root / "third_party/libero/libero/libero"
    mapping = dict(
        benchmark_root=base,
        bddl_files=base / "bddl_files",
        init_states=base / "init_files",
        assets=base / "assets",
        datasets=args.output / "datasets",
    )
    (cfg / "config.yaml").write_text(
        "\n".join(k + ": " + json.dumps(str(v)) for k, v in mapping.items()) + "\n"
    )
    (args.output / "plan.json").write_text(json.dumps(public_plan(plan), indent=2))
    critic = None
    completed = []
    try:
        if plan["critic"]:
            with (args.log_root / "critic.log").open("w") as log:
                critic = subprocess.Popen(
                    plan["critic"]["command"],
                    env=plan["critic"]["env"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            ready("http://127.0.0.1:%d/health" % args.critic_port, critic)
        for job in plan["jobs"]:
            directory = Path(job["directory"])
            directory.mkdir()
            logs = Path(job["log_directory"])
            logs.mkdir()
            server = None
            try:
                with (logs / "server.log").open("w") as log:
                    server = subprocess.Popen(
                        job["server"], env=job["env"], stdout=log, stderr=subprocess.STDOUT
                    )
                ready("http://127.0.0.1:%d/healthz" % args.port, server)
                with (logs / "eval.log").open("w") as log:
                    client = subprocess.Popen(
                        job["client"], env=job["env"], stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        while client.poll() is None:
                            if server.poll() is not None:
                                raise RuntimeError("Policy server exited during evaluation")
                            if critic is not None and critic.poll() is not None:
                                raise RuntimeError("Critic exited during evaluation")
                            time.sleep(2)
                        if client.returncode:
                            raise RuntimeError(
                                "Evaluator exited with code " + str(client.returncode)
                            )
                    finally:
                        stop(client)
                expected = len(job["tasks"]) * len(args.states)
                values = check_log(logs / "eval.log", expected)
                if args.method == "vlac":
                    episodes = json.loads((directory / "episodes.json").read_text())
                    if len(episodes) != expected or [e["env_success"] for e in episodes] != values:
                        raise RuntimeError("VLAC episode metadata differs from evaluator outcomes")
                if args.method == "dtw":
                    text = (logs / "server.log").read_text(errors="replace")
                    summaries = re.findall(r"\[memory\] summary: (\{[^\n]+\})", text)
                    summary = json.loads(summaries[-1]) if summaries else {}
                    if summary.get("episodes") != expected or summary.get(
                        "successful_episodes"
                    ) != sum(values):
                        raise RuntimeError(
                            "Memory commit summary differs from environment outcomes"
                        )
                row = dict(
                    method=args.method,
                    tasks=job["tasks"],
                    states=args.states,
                    seed=args.seed,
                    episodes=expected,
                    successes=sum(values),
                    success_rate=sum(values) / expected,
                )
                (directory / "result.json").write_text(json.dumps(row, indent=2))
                completed.append(row)
                (args.output / "completed.json").write_text(json.dumps(completed, indent=2))
            finally:
                stop(server)
        total = sum(r["episodes"] for r in completed)
        wins = sum(r["successes"] for r in completed)
        result = dict(
            method=args.method,
            seed=args.seed,
            episodes=total,
            successes=wins,
            success_rate=wins / total,
            complete_500=total == 500
            and args.tasks == list(range(10))
            and args.states == list(range(50)),
            jobs=completed,
        )
        (args.output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        stop(critic)


if __name__ == "__main__":
    main()
