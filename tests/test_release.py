"""Meaningful CPU checks: numerical retrieval, causal admission and safe launch plans."""

import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
sys.path.insert(0, str(ROOT / "experiments/vlac/scripts"))
import run_libero as evaluate
from vlac_progress import Progress, complete_chunks, prepare_commit
from verify_release import verify


class AdmissionTest(unittest.TestCase):
    def test_first_hit_is_causal_not_environment_termination(self):
        p = Progress(80, 64)
        self.assertFalse(p.update(8, 79))
        self.assertIsNone(p.first_hit)
        self.assertFalse(p.update(16, 85))
        self.assertEqual(p.first_hit, 16)
        p.update(24, -50)
        self.assertEqual(p.first_hit, 16)
        self.assertEqual(complete_chunks([0, 5, 10, 15, 20], 16), 3)

        class Memory:
            pass

        memory = Memory()
        memory.buffer = [("view", None, None, i) for i in range(5)]
        memory.chunk_index = 5
        accepted, before, kept = prepare_commit(
            memory, dict(mode="label", env_success=False, critic_success=True, complete_chunks=3)
        )
        self.assertEqual((accepted, before, kept), (True, 5, 3))
        self.assertEqual([r[3] for r in memory.buffer], [0, 1, 2])

    def test_no_hit_rejects_even_environment_success(self):
        class Memory:
            pass

        memory = Memory()
        memory.buffer = [("view", None, None, 0)]
        memory.chunk_index = 1
        accepted, _, kept = prepare_commit(
            memory, dict(mode="label", env_success=True, critic_success=False, complete_chunks=0)
        )
        self.assertFalse(accepted)
        self.assertEqual(kept, 0)

    def test_milestone_updates_reference_after_scoring(self):
        p = Progress(95, 64)
        self.assertTrue(p.update(64, 50))
        self.assertEqual(p.base, 50)
        p.update(72, 50)
        self.assertEqual(p.value, 75)
        with self.assertRaises(ValueError):
            p.update(72, 0)


class LaunchTest(unittest.TestCase):
    def test_original_project_layout_and_separate_logs(self):
        args = evaluate.arguments(
            ["--method", "dtw", "--checkpoint", "/model/pi05_libero", "--dry-run"]
        )
        plan = evaluate.build_plan(args)
        self.assertEqual(args.openpi_root, ROOT / "openpi")
        self.assertTrue(args.output.is_relative_to(ROOT / "experiments/runs"))
        self.assertTrue(args.log_root.is_relative_to(ROOT / "experiments/logs"))
        for job in plan["jobs"]:
            self.assertNotEqual(job["directory"], job["log_directory"])
            self.assertTrue(Path(job["client"][1]).is_relative_to(ROOT / "experiments/scripts"))

    def test_recorded_results_cannot_be_reused_as_output(self):
        with self.assertRaises(SystemExit):
            evaluate.arguments(
                [
                    "--method",
                    "dtw",
                    "--checkpoint",
                    "/model/pi05_libero",
                    "--dry-run",
                    "--output",
                    str(ROOT / "experiments/runs/pi05_memory_dtw"),
                ]
            )

    def test_memory_restarts_each_task_and_retains_state_order(self):
        args = evaluate.arguments(
            ["--method", "dtw", "--checkpoint", "/model/pi05_libero", "--dry-run"]
        )
        plan = evaluate.build_plan(args)
        self.assertEqual(len(plan["jobs"]), 10)
        self.assertEqual([j["tasks"] for j in plan["jobs"]], [[i] for i in range(10)])
        for j in plan["jobs"]:
            self.assertEqual(j["env"]["LIBERO_EVAL_STATES"], ",".join(map(str, range(50))))
            self.assertEqual(j["env"]["LIBERO_MEMORY_VIEWS"], "base")
            self.assertIn("--policy.config=pi05_libero", j["server"])
        self.assertEqual(plan["memory_config"]["dtw_drop_fraction"], 0.3)
        self.assertIsNone(plan["critic"])
        self.assertNotIn("PATH", evaluate.public_plan(plan)["jobs"][0]["env"])

    def test_vlac_uses_same_memory_and_explicit_label_mode(self):
        args = evaluate.arguments(
            [
                "--method",
                "vlac",
                "--checkpoint",
                "/model/pi05_libero",
                "--critic-checkpoint",
                "/model/VLAC-8b",
                "--critic-python",
                "/env/python",
                "--dry-run",
            ]
        )
        plan = evaluate.build_plan(args)
        self.assertEqual(
            plan["memory_config"],
            json.loads(
                (ROOT / "experiments/resources/memory_configs/pi05_memory_dtw.json").read_text()
            ),
        )
        for j in plan["jobs"]:
            self.assertEqual(j["client"][j["client"].index("--mode") + 1], "label")
            self.assertNotIn("--save-trajectories", j["client"])

    def test_video_suppression_keeps_episode_end_notification(self):
        source = (ROOT / "experiments/scripts/libero_client.py").read_text()
        tree = ast.parse(source)
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "write_video"
        )
        calls = []

        class Client:
            def infer(self, value):
                calls.append(value)

        class Env:
            _stable_task = 3
            _stable_state = 7
            _memory_done = True

        ns = dict(
            original_write_video=lambda *a, **k: None, current_env=Env(), memory_client=Client()
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), "callback", "exec"), ns)
        ns["write_video"]("unused.mp4", [])
        self.assertEqual(calls, [{"_episode_end": [3, 7, True]}])


class RecordedResultTest(unittest.TestCase):
    def test_all_recorded_outcomes_and_references(self):
        self.assertEqual(verify()["recorded_episodes"], 1500)


class NumericalTest(unittest.TestCase):
    def test_dtw_rejects_outlier_and_confidence_changes_guidance(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is optional for metadata-only verification")
        tree = ast.parse((ROOT / "experiments/scripts/memory_policy.py").read_text())
        names = {"dtw_consistency", "retrieval_confidence", "adaptive_t0"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = {"np": np}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "frozen_memory_functions", "exec"), ns
        )
        cfg = json.loads(
            (ROOT / "experiments/resources/memory_configs/pi05_memory_dtw.json").read_text()
        )
        actions = np.zeros((10, 5, 7), dtype=np.float32)
        actions[-1] = 100
        similarities = np.linspace(0.9993, 0.9999, 10, dtype=np.float32)
        kept, sims, info = ns["dtw_consistency"](actions, similarities, cfg)
        self.assertEqual(info["removed"], 3)
        self.assertEqual(len(kept), 7)
        self.assertTrue(np.all(kept == 0))
        low = ns["adaptive_t0"](np.array([0.9988]), cfg, 0)
        high = ns["adaptive_t0"](np.array([0.9998]), cfg, 0)
        self.assertGreater(low, high)
        self.assertGreaterEqual(
            ns["adaptive_t0"](np.array([0.9995]), cfg, 2),
            ns["adaptive_t0"](np.array([0.9995]), cfg, 0),
        )


if __name__ == "__main__":
    unittest.main()
