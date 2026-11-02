# Release adaptation: portable paths and optional media I/O; scientific method unchanged.
"""Select original task/state IDs and optionally apply deterministic environment fixes."""
import os
from pathlib import Path
import random
import runpy
import sys
from rts_paths import OPENPI, configure_video_saving

import numpy as np
from libero.libero import benchmark

configure_video_saving()

MODE = os.environ['LIBERO_EVAL_MODE']
SEED = int(os.environ['LIBERO_EVAL_SEED'])
TASKS = [int(value) for value in os.environ['LIBERO_EVAL_TASKS'].split(',')]
STATES = [int(value) for value in os.environ['LIBERO_EVAL_STATES'].split(',')]
active_task = None
current_env = None
memory_client = None
original_benchmarks = benchmark.get_benchmark_dict


class SelectedSuite:
    def __init__(self, original):
        self.original = original
        self.n_tasks = len(TASKS)

    def get_task(self, selected_id):
        global active_task
        active_task = TASKS[selected_id]
        return self.original.get_task(active_task)

    def get_task_init_states(self, selected_id):
        states = self.original.get_task_init_states(TASKS[selected_id])
        return [states[index] for index in STATES]


def selected_benchmarks():
    factories = dict(original_benchmarks())
    original_factory = factories['libero_10']
    factories['libero_10'] = lambda: SelectedSuite(original_factory())
    return factories


benchmark.get_benchmark_dict = selected_benchmarks

if MODE in ('stable', 'memory'):
    from libero.libero.envs.env_wrapper import ControlEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from robosuite.utils.binding_utils import MjRenderContext

    original_render = MjRenderContext.__init__
    original_init = ControlEnv.__init__
    original_reset = ControlEnv.reset
    original_set_state = ControlEnv.set_init_state
    original_infer = WebsocketClientPolicy.infer

    def render_init(self, sim, *args, **kwargs):
        sim.model.vis.quality.offsamples = 0
        original_render(self, sim, *args, **kwargs)

    def env_init(self, *args, **kwargs):
        np.random.seed(SEED)
        random.seed(SEED)
        original_init(self, *args, **kwargs)
        self._stable_task = active_task
        self._stable_episode_order = -1

    def reset(self):
        np.random.seed(SEED)
        random.seed(SEED)
        self.seed(SEED)
        return original_reset(self)

    def set_state(self, initial_state):
        global current_env
        result = original_set_state(self, initial_state)
        self._stable_episode_order += 1
        self._stable_state = STATES[self._stable_episode_order]
        self._stable_first_call = True
        if MODE == 'memory':
            self._memory_done = False
        current_env = self
        return result

    def infer(self, obs):
        global memory_client
        if '_episode_end' in obs:
            return original_infer(self, obs)
        if MODE == 'memory':
            memory_client = self
        env = current_env
        metadata = [SEED, env._stable_task, env._stable_state, env._stable_first_call]
        result = original_infer(self, dict(obs, _episode_key=metadata))
        env._stable_first_call = False
        return result

    MjRenderContext.__init__ = render_init
    ControlEnv.__init__ = env_init
    ControlEnv.reset = reset
    ControlEnv.set_init_state = set_state
    WebsocketClientPolicy.infer = infer

    if MODE == 'memory':
        import imageio
        original_step = ControlEnv.step
        original_write_video = imageio.mimwrite

        def step(self, action):
            result = original_step(self, action)
            self._memory_done = bool(result[2])
            return result

        def write_video(path, *args, **kwargs):
            result = original_write_video(path, *args, **kwargs)
            env = current_env
            if memory_client is not None and env is not None:
                # Official evaluator saves one video after each episode, including the last one.
                memory_client.infer({'_episode_end': [env._stable_task, env._stable_state,
                                                      bool(getattr(env, '_memory_done', False))]})
            return result

        ControlEnv.step = step
        imageio.mimwrite = write_video

sys.argv[0] = str(OPENPI / 'examples/libero/main.py')
runpy.run_path(sys.argv[0], run_name='__main__')
