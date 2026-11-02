# Release adaptation: portable paths and optional media I/O; scientific method unchanged.
"""Official VLAC GAC model exposed as an isolated, stateless pairwise critic."""
import argparse
import base64
from io import BytesIO
import json
import logging
from pathlib import Path
import threading
import time

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
import torch
import uvicorn
from evo_vlac import GAC_model
from swift.llm import RequestConfig

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from rts_paths import VLAC_ROOT as ROOT
app = FastAPI()
lock = threading.Lock()
critic = None
references = {}
settings = {}

class ScoreRequest(BaseModel):
    task: str
    ref_image_b64: str
    cur_image_b64: str

def image(encoded):
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 8*1024*1024: raise ValueError('Image payload too large')
    return Image.open(BytesIO(raw)).convert('RGB')

def task_key(task): return ' '.join(task.lower().strip().rstrip('.').split())

@app.get('/health')
def health():
    return dict(ready=critic is not None, model='InternRobotics/VLAC-8b', settings=settings)

@app.post('/critic_score')
def score(request: ScoreRequest):
    start = time.monotonic()
    try:
        ref, cur = image(request.ref_image_b64), image(request.cur_image_b64)
        context = references.get(task_key(request.task))
        if references and context is None: raise ValueError('Missing reference for task')
        if context is not None and len(context) < 2: raise ValueError('Reference needs at least two frames')
        with lock, torch.inference_mode():
            scores, _ = critic.get_trajectory_critic(task=request.task, image_list=[ref,cur],
                ref_image_list=context, ref_num=len(context) if context else 0,
                batch_num=1, skip=1, rich=settings['rich'], think=False, reverse_eval=False)
        if len(scores) != 1: raise ValueError('Expected exactly one pairwise score')
        result = float(scores[0])
        if not __import__('math').isfinite(result): raise ValueError('Nonfinite model output')
        result = max(-100.,min(100.,result))
        return dict(c=result, latency_ms=(time.monotonic()-start)*1000, reference_frames=len(context or []))
    except Exception as exc:
        logging.exception('VLAC scoring failed')
        raise HTTPException(status_code=500, detail=str(exc)) from exc

def main():
    global critic, references, settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=19891)
    parser.add_argument('--model',required=True,help='Local externally downloaded VLAC-8b checkpoint')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pi05_reference24/task_00.json')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    critic = GAC_model(tag='critic')
    critic.init_model(args.model,model_type='internvl2',device_map='cuda:0')
    critic.model.eval()
    critic.do_sample = False
    critic.temperature = 0.
    critic.top_k = 1
    critic.max_tokens = 16
    critic.set_config()
    # Swift uses the request temperature to construct generation_config; force greedy there too.
    critic.request_config = RequestConfig(max_tokens=16,temperature=0.,top_k=1,
                                         logprobs=True,top_logprobs=10)
    critic.model.generation_config.do_sample = False
    critic.set_system_prompt()
    mapping = settings.get('reference_map')
    if mapping:
        map_path = Path(mapping)
        if not map_path.is_absolute(): map_path = ROOT / map_path
        for task, entry in json.loads(map_path.read_text()).items():
            paths = entry['ref_images'] if isinstance(entry,dict) else entry
            references[task_key(task)] = [Image.open(Path(p) if Path(p).is_absolute() else map_path.parent / p).convert('RGB') for p in paths]
    print('VLAC_READY '+json.dumps(dict(settings=settings,do_sample=False,temperature=0.,max_tokens=16)),flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,workers=1)

if __name__ == '__main__': main()
