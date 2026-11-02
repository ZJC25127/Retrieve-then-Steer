"""Check recorded outcomes, reference images and release checksums without models."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root=ROOT):
    summary = read(root / "experiments/results_summary.json")
    assert len(summary["runs"]) == 3
    expected = {"baseline": 463, "dtw": 476, "vlac": 476}
    for row in summary["runs"]:
        method = row["method"]
        prefix = root / ("experiments/vlac" if method == "vlac" else "experiments")
        name = {
            "baseline": "pi05_baseline",
            "dtw": "pi05_memory_dtw",
            "vlac": "pi05_memory_dtw_vlac",
        }[method]
        run = prefix / "runs" / name
        logs = prefix / "logs" / name
        outcomes = read(run / "outcomes.json")
        assert len(outcomes) == row["episodes"] == 500 and row["seed"] == 7
        assert [(r["task"], r["state"]) for r in outcomes] == [
            (t, s) for t in range(10) for s in range(50)
        ]
        assert sum(r["env_success"] for r in outcomes) == row["successes"] == expected[method]
        assert [
            sum(r["env_success"] for r in outcomes if r["task"] == t) for t in range(10)
        ] == row["per_task"]
        if method == "baseline":
            files = [logs / "eval.log"]
        else:
            files = [logs / ("task_%02d/eval.log" % t) for t in range(10)]
        values = []
        for p in files:
            values.extend(
                v == "True"
                for v in re.findall(
                    r"INFO:root:Success: (True|False)\s*(?:\n|$)", p.read_text(encoding="utf-8")
                )
            )
        assert values == [r["env_success"] for r in outcomes], method
        if method == "vlac":
            for task in range(10):
                ep = read(run / ("task_%02d/episodes.json" % task))
                assert len(ep) == 50 and [r["state"] for r in ep] == list(range(50))
                assert [r["env_success"] for r in ep] == values[task * 50 : (task + 1) * 50]
                for r in ep:
                    if r["accepted"]:
                        assert r["critic_success"] and r["kept_chunks"] > 0
    references = read(root / "experiments/vlac/references/libero10_reference24/reference_map.json")
    assert len(references) == 10
    assert {r["task_id"] for r in references.values()} == set(range(10))
    images = []
    for entry in references.values():
        assert entry["dataset_split"] == "train" and entry["reference_frames"] == 24
        assert len(entry["ref_images"]) == 24
        for name in entry["ref_images"]:
            p = root / "experiments/vlac/references/libero10_reference24" / name
            assert p.is_file() and p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            images.append(p)
    assert len(set(images)) == 240
    for task in range(10):
        rule = read(root / ("experiments/vlac/configs/pi05_reference24/task_%02d.json" % task))
        assert (
            rule["check_interval_steps"] == 8
            and rule["milestone_steps"] == 64
            and rule["reference_frames"] == 24
        )
        assert rule["completion_threshold"] == (
            80 if task in [3, 5, 6] else 99 if task == 8 else 95
        )
    forbidden = {".npz", ".mp4", ".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".pyc"}
    for p in root.rglob("*"):
        if p.is_file() and not any(
            v in p.parts
            for v in ["outputs", "third_party", "openpi", "VLAC", ".git", ".venv", "__pycache__"]
        ):
            assert p.suffix.lower() not in forbidden, p
    checksum = root / "SHA256SUMS"
    if checksum.exists():
        for line in checksum.read_text().splitlines():
            digest, rel = line.split("  ", 1)
            assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    return {
        "recorded_runs": 3,
        "recorded_episodes": 1500,
        "reference_images": 240,
        "scores": expected,
        "checksums": checksum.exists(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(p.parse_args().root), indent=2))
