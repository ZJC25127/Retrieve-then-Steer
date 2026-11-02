# Release adaptation: portable paths and optional media I/O; scientific method unchanged.
"""New OpenPI entry point; reuse the frozen method without editing its source."""
import hashlib
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rts_paths import ROOT as EXP, VLAC_ROOT as ROOT, OPENPI
REPO = OPENPI
import jax
from openpi.policies.policy import Policy
from memory_policy import enable_memory
from vlac_progress import prepare_commit

manifest = json.loads((ROOT/'configs/install_manifest.json').read_text())
method_sha = hashlib.sha256((EXP/'scripts/memory_policy.py').read_bytes()).hexdigest()
if method_sha != manifest['before_scripts_sha256']['memory_policy.py']:
    raise RuntimeError('Frozen memory method changed; refuse mixed-version evaluation')
enable_memory()
original_infer = Policy.infer
lock = threading.Lock()

def infer(self, observation, **kwargs):
    observation = dict(observation)
    with lock:
        if '_vlac_episode_end' in observation:
            event = observation['_vlac_episode_end']
            memory = self._memory
            accepted, before, kept = prepare_commit(memory,event)
            result = memory.finish(event['task'],event['state'],accepted)
            logging.info('[vlac] commit: %s',json.dumps(dict(event,accepted=accepted,
                buffered_chunks=before,kept_chunks=kept,method_sha256=method_sha),sort_keys=True))
            return dict(result,accepted=accepted,kept_chunks=kept)
        seed,task,state,first_call = observation.pop('_episode_key')
        if first_call:
            self._rng = jax.random.fold_in(jax.random.fold_in(jax.random.key(int(seed)),int(task)),int(state))
            self._memory.begin(task,state)
        return original_infer(self,observation,**kwargs)

Policy.infer = infer
sys.argv[0] = str(REPO/'scripts/serve_policy.py')
runpy.run_path(sys.argv[0],run_name='__main__')
