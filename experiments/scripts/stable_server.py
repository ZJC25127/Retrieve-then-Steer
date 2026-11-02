# Release adaptation: portable paths and optional media I/O; scientific method unchanged.
"""Reset the official policy RNG at episode boundaries; write no key or trace files."""
from pathlib import Path
import os
import runpy
import sys
from rts_paths import OPENPI, configure_video_saving
import threading

import jax
from openpi.policies.policy import Policy

MEMORY = os.environ.get('LIBERO_EVAL_MODE') == 'memory'
if MEMORY:
    from memory_policy import enable_memory
    enable_memory()

original_infer = Policy.infer
lock = threading.Lock()


def infer(self, obs, **kwargs):
    obs = dict(obs)
    if MEMORY and '_episode_end' in obs:
        task_id, state_id, success = obs.pop('_episode_end')
        with lock:
            return self._memory.finish(task_id, state_id, bool(success))
    seed, task_id, state_id, first_call = obs.pop('_episode_key')
    with lock:
        if first_call:
            key = jax.random.key(int(seed))
            key = jax.random.fold_in(key, int(task_id))
            self._rng = jax.random.fold_in(key, int(state_id))
            if MEMORY:
                self._memory.begin(task_id, state_id)
        return original_infer(self, obs, **kwargs)


Policy.infer = infer
sys.argv[0] = str(OPENPI / 'scripts/serve_policy.py')
runpy.run_path(sys.argv[0], run_name='__main__')
