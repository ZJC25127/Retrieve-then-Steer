# Release adaptation: portable paths and optional media I/O; scientific method unchanged.
"""Independent VLAC label rollout with lossless, terminal-inclusive trajectory collection."""
import argparse
import base64
from collections import deque
import importlib.util
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import random
import urllib.request

import imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from robosuite.utils.binding_utils import MjRenderContext
from libero.libero.envs.env_wrapper import ControlEnv
from vlac_progress import Progress, complete_chunks

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rts_paths import VLAC_ROOT as ROOT, OPENPI
REPO = OPENPI
spec = importlib.util.spec_from_file_location('official_libero',REPO/'examples/libero/main.py')
official = importlib.util.module_from_spec(spec)
__import__('sys').modules[spec.name] = official
spec.loader.exec_module(official)

def preprocess(obs, key='agentview_image'):
    frame = np.ascontiguousarray(obs[key][::-1,::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(frame,224,224))

def encoded(frame):
    buffer = BytesIO()
    Image.fromarray(frame).save(buffer,format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')

def get_score(url, task, ref, cur, timeout):
    data = json.dumps(dict(task=task,ref_image_b64=encoded(ref),cur_image_b64=encoded(cur))).encode()
    req = urllib.request.Request(url+'/critic_score',data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response: return json.load(response)

def deterministic_environment(seed):
    render_init = MjRenderContext.__init__
    env_init = ControlEnv.__init__
    env_reset = ControlEnv.reset
    def render(self, sim, *args, **kwargs):
        sim.model.vis.quality.offsamples = 0
        return render_init(self,sim,*args,**kwargs)
    def initialize(self,*args,**kwargs):
        np.random.seed(seed);random.seed(seed)
        return env_init(self,*args,**kwargs)
    def reset(self):
        np.random.seed(seed);random.seed(seed);self.seed(seed)
        return env_reset(self)
    MjRenderContext.__init__ = render
    ControlEnv.__init__ = initialize
    ControlEnv.reset = reset

def policy_state(obs):
    return np.concatenate((obs['robot0_eef_pos'],
        official._quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))

def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--task',type=int,default=8)
    parser.add_argument('--states',default=','.join(map(str,range(50))))
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--mode',choices=('observe','label'),default='observe')
    parser.add_argument('--critic-url',required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pi05_reference24/task_00.json')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--save-videos',action='store_true')
    parser.add_argument('--save-trajectories',action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    states = [int(s) for s in args.states.split(',')]
    np.random.seed(args.seed)
    deterministic_environment(args.seed)
    suite = benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(args.task)
    initial = suite.get_task_init_states(args.task)
    env, description = official._get_libero_env(task,256,args.seed)
    client = WebsocketClientPolicy('127.0.0.1',args.port)
    (args.run/'trajectories').mkdir(parents=True)
    logging.info('[vlac] parameters: %s',json.dumps(dict(config,mode=args.mode,task=args.task,
        seed=args.seed,initial_states=states,image_resize=224,msaa=False,policy_inputs='main+wrist',
        critic_inputs='main',step_unit='executed_actions_excluding_10_wait_steps')))
    episodes = []
    try:
        for state in states:
            env.reset()
            obs = env.set_init_state(initial[state])
            for _ in range(10): obs, _, done, _ = env.step(official.LIBERO_DUMMY_ACTION)
            reference = preprocess(obs)
            if args.smoke:
                scores = [get_score(args.critic_url,description,reference,reference,config['timeout_seconds']) for _ in range(2)]
                if scores[0]['c'] != scores[1]['c']: raise RuntimeError('VLAC greedy repeat differs')
                actions = client.infer({'observation/image':reference,
                    'observation/wrist_image':preprocess(obs,'robot0_eye_in_hand_image'),
                    'observation/state':np.concatenate((obs['robot0_eef_pos'],
                        official._quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),
                    'prompt':str(description),'_episode_key':[args.seed,args.task,state,True]})['actions']
                if actions.shape[-1] != 7 or len(actions) < 5: raise RuntimeError('Invalid OpenPI action shape')
                commit = client.infer({'_vlac_episode_end':dict(task=args.task,state=state,mode='observe',
                    env_success=False,critic_success=False,first_hit=None,complete_chunks=0)})
                (args.run/'smoke.json').write_text(json.dumps(dict(repeated_scores=scores,identical=True,
                    action_shape=list(actions.shape),commit=commit),indent=2))
                return
            progress = Progress(config['completion_threshold'],config['milestone_steps'])
            action_plan = deque()
            chunk_starts = []
            # Index i is the observation after i executed policy actions; wait steps are excluded.
            replay = [reference.copy()]
            state_trace = [policy_state(obs).copy()]
            action_trace, rewards, success_flags, checks = [], [], [], []
            first_call = True
            done = False
            score_calls = 0
            def check(step, observation):
                nonlocal reference, score_calls
                current = preprocess(observation)
                response = get_score(args.critic_url,description,reference,current,config['timeout_seconds'])
                used_reference_step = progress.reference_step
                previous_base = progress.base
                refresh = progress.update(step,float(response['c']))
                score_calls += 1
                record = dict(task=args.task,state=state,step=step,
                    c=response['c'],progress=progress.value,base=progress.base,first_hit=progress.first_hit,
                    used_reference_step=used_reference_step,base_before=previous_base,
                    reference_step=progress.reference_step,refreshed=refresh,
                    reference_frames=response['reference_frames'],latency_ms=response['latency_ms'])
                checks.append(record)
                logging.info('[vlac] check: %s',json.dumps(record,sort_keys=True))
                if refresh: reference = current
            # Same 520-action limit, preprocessing, replanning and environment stopping as official.
            for step in range(520):
                img = preprocess(obs)
                wrist = preprocess(obs,'robot0_eye_in_hand_image')
                if not action_plan:
                    element = {'observation/image':img,'observation/wrist_image':wrist,
                        'observation/state':np.concatenate((obs['robot0_eef_pos'],
                            official._quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos'])),
                        'prompt':str(description),'_episode_key':[args.seed,args.task,state,first_call]}
                    actions = client.infer(element)['actions']
                    if len(actions) < 5: raise RuntimeError('Policy chunk shorter than replan interval')
                    action_plan.extend(actions[:5]);chunk_starts.append(step);first_call = False
                action = action_plan.popleft().copy()
                obs, reward, done, _ = env.step(action.tolist())
                action_trace.append(action)
                rewards.append(float(reward))
                success_flags.append(bool(done))
                replay.append(preprocess(obs).copy())
                state_trace.append(policy_state(obs).copy())
                executed = step+1
                if executed % config['check_interval_steps'] == 0: check(executed,obs)
                if done: break
            if progress.last_checked != executed: check(executed,obs)
            critic_success = progress.first_hit is not None
            event = dict(task=args.task,state=state,mode=args.mode,env_success=bool(done),
                critic_success=critic_success,first_hit=progress.first_hit,
                complete_chunks=complete_chunks(chunk_starts,progress.first_hit),executed_steps=executed,
                final_progress=progress.value,score_calls=score_calls)
            response = client.infer({'_vlac_episode_end':event})
            event.update(accepted=response['accepted'],kept_chunks=response['kept_chunks'])
            directory = args.run/'trajectories'/f'state{state:02d}_{"success" if done else "failure"}'
            directory.mkdir()
            arrays = dict(main_frames=np.stack(replay),states=np.stack(state_trace),
                actions=np.stack(action_trace),rewards=np.asarray(rewards,dtype=np.float64),
                env_success_flags=np.asarray(success_flags,dtype=np.bool_))
            # Compressed NPZ is lossless. MP4 is a viewing copy, never used for critic calibration.
            if args.save_trajectories:
                np.savez_compressed(directory/'trajectory.tmp.npz',**arrays)
                (directory/'trajectory.tmp.npz').replace(directory/'trajectory.npz')
                with np.load(directory/'trajectory.npz',allow_pickle=False) as saved:
                    if any(not np.array_equal(saved[key],value) for key,value in arrays.items()):
                        raise RuntimeError('Trajectory save roundtrip differs')
            if args.save_videos:
                imageio.mimwrite(directory/'video.mp4',replay,fps=10)
            save_json(directory/'checks.json',checks)
            event.update(trajectory_directory=str(directory),frame_count=len(replay),
                trajectory_saved=args.save_trajectories,video_saved=args.save_videos,
                trajectory_schema=dict(version=1,main_frames='uint8 RGB 224x224, vertically and horizontally flipped then resized with pad',
                    observation_index='i = after i policy actions; initial observation at 0, terminal at executed_steps',
                    action_index='actions[i] transforms observation i to i+1; first 10 wait actions excluded',
                    states='eef position(3), axis-angle(3), gripper qpos(2)',
                    rewards='one per executed action',env_success_flags='done returned by LIBERO per executed action',
                    termination='LIBERO success or 520 executed actions',
                    checks='used_reference_step identifies the frame actually sent for this check'))
            save_json(directory/'episode.json',event)
            episodes.append(event)
            save_json(args.run/'episodes.json',episodes)
            logging.info('Success: %s',bool(done))
            logging.info('[vlac] episode: %s',json.dumps(event,sort_keys=True))
        (args.run/'episodes.json').write_text(json.dumps(episodes,indent=2))
        logging.info('Total episodes: %s',len(episodes))
    finally:
        env.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
