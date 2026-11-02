"""Task-local successful trajectory retrieval with masked views and adaptive sampling time."""
from __future__ import annotations

import logging
import json
import os
import types

import einops
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies.policy import Policy
from openpi.shared import nnx_utils


def visual_key(self, observation, image_names, pooling=2, feature_layer='projected', wrist_weight=1.0,
               view_normalize=False):
    """Only encode explicitly selected real views, including the available wrist view."""
    observation = _model.preprocess_observation(None, observation, train=False)
    parts = []
    for name in image_names:
        tokens, auxiliary = self.PaliGemma.img(observation.images[name], train=False)
        if feature_layer == 'encoded':
            if 'encoded' not in auxiliary:
                raise ValueError('Visual encoder does not expose encoded patch tokens')
            tokens = auxiliary['encoded']
        batch, count, width = tokens.shape
        side = int(count ** 0.5)
        if side * side != count:
            side = int((count - 1) ** 0.5)
            if side * side == count - 1:
                tokens = tokens[:, 1:]
            else:
                part = jnp.tile(jnp.mean(tokens, axis=1), (1, pooling * pooling))
                if view_normalize:
                    part = part.astype(jnp.float32)
                    part = part / (jnp.linalg.norm(part, axis=-1, keepdims=True) + 1e-8)
                if name != 'base_0_rgb' and wrist_weight != 1.0:
                    part = part * wrist_weight
                parts.append(part)
                continue
        grid = tokens.reshape(batch, side, side, width)
        if side % pooling == 0:
            half = side // pooling
            pooled = grid.reshape(batch, pooling, half, pooling, half, width).mean(axis=(2, 4))
        else:
            pooled = jax.image.resize(grid, (batch, pooling, pooling, width), method='linear')
        part = pooled.reshape(batch, pooling * pooling * width)
        if view_normalize:
            part = part.astype(jnp.float32)
            part = part / (jnp.linalg.norm(part, axis=-1, keepdims=True) + 1e-8)
        if name != 'base_0_rgb' and wrist_weight != 1.0:
            part = part * wrist_weight
        parts.append(part)
    return jnp.concatenate(parts, axis=-1)


def sample_actions(self, rng, observation, *, num_steps=10, noise=None, start_time=1.0, exact_endpoint=False):
    """Official Pi0 flow sampler with the additional start_time argument used by retrieval."""
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, 'b p -> b s p', s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=positions,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        assert prefix_out is None
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        step_dt = jnp.where(exact_endpoint, jnp.maximum(dt, -time), dt)
        # Match the official weak-scalar multiplication even for BF16 velocity.
        action_dt = step_dt.astype(velocity.dtype)
        return x_t + action_dt * velocity, time + step_dt

    def cond(carry):
        return jnp.where(exact_endpoint, carry[1] > 1e-6, carry[1] >= -dt / 2)

    actions, _ = jax.lax.while_loop(cond, step, (noise, start_time))
    return actions


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)


def so3_exp(vector):
    vector = np.asarray(vector, dtype=np.float32)
    theta = float(np.linalg.norm(vector))
    matrix = skew(vector)
    if theta < 1e-8:
        return (np.eye(3, dtype=np.float32) + matrix).astype(np.float32)
    return (np.eye(3, dtype=np.float32) + np.sin(theta) / theta * matrix
            + (1 - np.cos(theta)) / theta**2 * (matrix @ matrix)).astype(np.float32)


def so3_log(matrix):
    theta = float(np.arccos(np.clip((np.trace(matrix) - 1) * 0.5, -1, 1)))
    vector = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
                       matrix[1, 0] - matrix[0, 1]], dtype=np.float32)
    return (vector * (0.5 if theta < 1e-8 else theta / (2 * np.sin(theta) + 1e-8))).astype(np.float32)


def aggregate_actions(actions, weights):
    """Preserve local policy's linear position/gripper and SO(3) rotation aggregation."""
    weights = weights / (weights.sum() + 1e-8)
    result = (weights[:, None, None] * actions).sum(axis=0).astype(np.float32)
    if actions.shape[-1] < 7 or actions.shape[0] == 1:
        return result
    for step in range(actions.shape[1]):
        vectors = actions[:, step, 3:6]
        rotations = np.stack([so3_exp(vector) for vector in vectors])
        mean = so3_exp((weights[:, None] * vectors).sum(axis=0))
        for _ in range(20):
            delta = sum((weight * so3_log(mean.T @ rotation) for weight, rotation in zip(weights, rotations)),
                        np.zeros(3, dtype=np.float32))
            if np.linalg.norm(delta) < 1e-5:
                break
            mean = mean @ so3_exp(delta)
        result[step, 3:6] = so3_log(mean)
    return result


DEFAULT_CONFIG = dict(max_entries=900, top_k=10, threshold=0.9991, tau=0.05,
    phase_filter=True, phase_window=10, phase_min_candidates=3, adaptive_t0=True, t_min=0.75, t_max=1.0,
    fixed_t0=0.75, t0_sim_ref=0.9992, t0_sim_scale=5e-4, t0_gamma=2.0,
    t0_stat='mean', pooling=2, rotation_aggregation='so3', state_weight=0.0,
    exact_endpoint=False, feature_layer='projected', wrist_weight=1.0, view_normalize=False,
    sampling_steps=10, dtw_filter=False, dtw_drop_fraction=0.5, dtw_min_candidates=3,
    t0_alpha=1.0, t0_beta=0.05)


def memory_config():
    supplied = json.loads(os.environ.get('LIBERO_MEMORY_CONFIG', '{}'))
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError('Unknown memory configuration fields')
    config = dict(DEFAULT_CONFIG, **supplied)
    for name in ('max_entries', 'top_k', 'phase_min_candidates', 'sampling_steps', 'dtw_min_candidates'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    if config['sampling_steps'] > 40:
        raise ValueError('sampling_steps must be at most 40')
    if type(config['phase_filter']) is not bool:
        raise ValueError('phase_filter must be a boolean')
    if type(config['dtw_filter']) is not bool or not 0 <= config['dtw_drop_fraction'] < 1:
        raise ValueError('Invalid DTW filter configuration')
    if config['dtw_min_candidates'] < 2:
        raise ValueError('dtw_min_candidates must be at least two')
    if not np.isfinite(config['t0_alpha']) or config['t0_alpha'] <= 0:
        raise ValueError('t0_alpha must be positive and finite')
    if not np.isfinite(config['t0_beta']) or config['t0_beta'] < 0:
        raise ValueError('t0_beta must be nonnegative and finite')
    if type(config['phase_window']) is not int or config['phase_window'] < 0:
        raise ValueError('phase_window must be a nonnegative integer')
    if type(config['adaptive_t0']) is not bool or config['pooling'] not in (1, 2, 4):
        raise ValueError('Invalid adaptive_t0 or pooling')
    for name in ('t_min', 'fixed_t0'):
        if not 0 < config[name] < 0.98:
            raise ValueError(name + ' must be between 0 and 0.98')
    if not config['t_min'] < config['t_max'] <= 1.0:
        raise ValueError('t_max must be greater than t_min and at most 1.0')
    if not -1 <= config['threshold'] <= 1 or not -1 <= config['t0_sim_ref'] <= 1:
        raise ValueError('Invalid similarity threshold/reference')
    if any(config[name] <= 0 for name in ('tau', 't0_sim_scale', 't0_gamma')):
        raise ValueError('tau, t0_sim_scale and t0_gamma must be positive')
    if config['rotation_aggregation'] not in ('so3', 'linear'):
        raise ValueError('Invalid rotation aggregation')
    if config['t0_stat'] not in ('mean', 'max'):
        raise ValueError('Invalid t0_stat')
    if not 0 <= config['state_weight'] <= 0.3 or type(config['exact_endpoint']) is not bool:
        raise ValueError('Invalid state_weight or exact_endpoint')
    if config['feature_layer'] not in ('projected', 'encoded'):
        raise ValueError('feature_layer must be projected or encoded')
    if not 0 < config['wrist_weight'] <= 4 or type(config['view_normalize']) is not bool:
        raise ValueError('Invalid wrist_weight or view_normalize')
    return config


def dtw_consistency(actions, similarities, config):
    """Pairwise multivariate DTW on the seven real LIBERO action dimensions.

    Costs are sums of Euclidean distances in model action space, with unrestricted
    monotone alignment and both endpoints included. Padding dimensions are excluded.
    Drop candidates with the highest median pairwise distance; similarity breaks ties.
    Confidence uses the standard deviation of retained inconsistency scores.
    """
    count = len(similarities)
    info = dict(before=count, after=count, removed=0, sigma=0.0, median=0.0)
    if not config['dtw_filter'] or count < 2:
        return actions, similarities, info
    real = np.asarray(actions[..., :7], dtype=np.float32)
    left, right = np.triu_indices(count, k=1)
    local = np.linalg.norm(real[left, :, None, :] - real[right, None, :, :], axis=-1)
    horizon = real.shape[1]
    costs = np.full((len(left), horizon + 1, horizon + 1), np.inf, dtype=np.float32)
    costs[:, 0, 0] = 0
    for i in range(1, horizon + 1):
        for j in range(1, horizon + 1):
            previous = np.minimum(np.minimum(costs[:, i-1, j], costs[:, i, j-1]), costs[:, i-1, j-1])
            costs[:, i, j] = local[:, i-1, j-1] + previous
    distances = np.full((count, count), np.nan, dtype=np.float32)
    distances[left, right] = distances[right, left] = costs[:, -1, -1]
    scores = np.nanmedian(distances, axis=1)
    selected = np.arange(count)
    if count >= config['dtw_min_candidates']:
        keep = max(1, int(np.ceil(count * (1.0 - config['dtw_drop_fraction']))))
        selected = np.lexsort((-similarities, scores))[:keep]
        # Keep the original similarity order for deterministic aggregation.
        selected = np.sort(selected)
    info.update(after=len(selected), removed=count-len(selected),
                sigma=float(np.std(scores[selected])), median=float(np.median(scores[selected])))
    return actions[selected], similarities[selected], info


def retrieval_confidence(similarities, config, dtw_sigma=0.0):
    statistic = np.max(similarities) if config.get('t0_stat', 'mean') == 'max' else np.mean(similarities)
    visual = np.clip((float(statistic) - config['t0_sim_ref']) / config['t0_sim_scale'], -5, 5)
    return float(config['t0_alpha'] * visual - config['t0_beta'] * dtw_sigma)


def adaptive_t0(similarities, config=None, dtw_sigma=0.0):
    config = DEFAULT_CONFIG if config is None else config
    if not config['adaptive_t0']:
        return float(config['fixed_t0'])
    confidence = retrieval_confidence(similarities, config, dtw_sigma)
    floor = config['t_min']
    ceiling = config.get('t_max', 1.0)
    return float(floor + (ceiling - floor) / (1 + np.exp(config['t0_gamma'] * confidence)))


def real_views(observation):
    """Resolve masks outside JIT so absent cameras never enter the visual encoder."""
    return tuple(name for name in sorted(observation.images)
                 if bool(np.all(np.asarray(observation.image_masks.get(name, True)))))


class SuccessfulMemory:
    def __init__(self, max_size=900, config=None):
        self.max_size = max_size
        self.config = DEFAULT_CONFIG if config is None else config
        self.keys = self.actions = self.chunk_indices = None

    def __len__(self):
        return 0 if self.keys is None else len(self.keys)

    def add(self, keys, actions, chunk_indices):
        keys = np.asarray(keys, dtype=np.float32)
        keys = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
        actions = np.asarray(actions, dtype=np.float32)
        chunk_indices = np.asarray(chunk_indices, dtype=np.int32)
        if len(self):
            keys = np.concatenate([self.keys, keys])
            actions = np.concatenate([self.actions, actions])
            chunk_indices = np.concatenate([self.chunk_indices, chunk_indices])
        self.keys = keys[-self.max_size:]
        self.actions = actions[-self.max_size:]
        self.chunk_indices = chunk_indices[-self.max_size:]

    def retrieve(self, key, chunk_idx, k=10, *, diagnostics=False):
        keys, actions = self.keys, self.actions
        scope = 'all'
        if self.config['phase_filter']:
            candidates = np.abs(self.chunk_indices - chunk_idx) <= self.config['phase_window']
            scope = 'all_fallback'
            if int(candidates.sum()) >= self.config['phase_min_candidates']:
                keys, actions = keys[candidates], actions[candidates]
                scope = 'phase'
        key = np.asarray(key, dtype=np.float32)
        similarities = keys @ (key / (np.linalg.norm(key) + 1e-8))
        k = min(k, len(similarities))
        order = np.argpartition(-similarities, k - 1)[:k]
        order = order[np.argsort(-similarities[order])]
        keep = similarities[order] >= self.config['threshold']
        if diagnostics:
            return actions[order[keep]], similarities[order[keep]], dict(
                max_sim=float(similarities[order[0]]), candidates=len(similarities),
                top_k=k, scope=scope)
        return actions[order[keep]], similarities[order[keep]]


class MemoryAdapter:
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler
        self.encoders = {}
        self.memories = {}
        self.task_id = None
        self.episode = None
        self.buffer = []
        self.chunk_index = 0
        self.successes = 0
        self.config = memory_config()
        self.view_mode = os.environ.get('LIBERO_MEMORY_VIEWS', 'all')
        if self.view_mode not in ('all', 'base'):
            raise ValueError('LIBERO_MEMORY_VIEWS must be all or base.')
        self.inferences = self.retrievals = self.injections = self.episodes = 0
        parameters = dict(view_mode=self.view_mode, task_local=True, success_only=True,
            max_entries_per_view_set=900, top_k=10, threshold=0.9991, tau=0.05,
            phase_filter=True, phase_window=10, phase_min_candidates=3, min_kept_for_prior=1,
            adaptive_t0=True, t_min=0.75, t0_alpha=1.0, t0_beta=0.05, t0_gamma=2.0,
            t0_sim_ref=0.9992, t0_sim_scale=5e-4, confidence_clip=[-5, 5], t0_fallback=0.98,
            DTW=False, negative_memory=False, component_aware_aggregation=True,
            feature_encoder='PaliGemma.img/SigLIP', pooling='2x2', similarity='cosine',
            mask_padding=True, action_horizon=model.action_horizon, action_dim=model.action_dim,
            default_sampling_steps=self.config['sampling_steps'], sampling_dt=-1.0/self.config['sampling_steps'],
            seed=os.environ.get('LIBERO_EVAL_SEED'), image_resize=224, replan_steps=5,
            xla_flags=os.environ.get('XLA_FLAGS'), msaa=False)
        parameters.update({name: self.config[name] for name in (
            'top_k', 'threshold', 'tau', 'phase_filter', 'phase_window', 'phase_min_candidates',
            'adaptive_t0', 't_min', 't_max', 't0_gamma', 't0_sim_ref', 't0_sim_scale',
            'dtw_filter', 'dtw_drop_fraction', 'dtw_min_candidates', 't0_alpha', 't0_beta')})
        parameters['DTW'] = self.config['dtw_filter']
        parameters.update(max_entries_per_view_set=self.config['max_entries'],
            fixed_t0=self.config['fixed_t0'], pooling='%dx%d' % (self.config['pooling'], self.config['pooling']),
            component_aware_aggregation=self.config['rotation_aggregation'] == 'so3',
            state_weight=self.config['state_weight'], exact_endpoint=self.config['exact_endpoint'],
            t0_stat=self.config['t0_stat'], feature_layer=self.config['feature_layer'],
            wrist_weight=self.config['wrist_weight'], view_normalize=self.config['view_normalize'])
        logging.info('[memory] parameters: %s', json.dumps(parameters, sort_keys=True))
        logging.info('[memory] effective_config: %s', json.dumps(self.config, sort_keys=True))

    def summary(self):
        return dict(episodes=self.episodes, successful_episodes=self.successes,
            inferences=self.inferences, retrievals=self.retrievals, injections=self.injections,
            injection_rate=self.injections / self.inferences if self.inferences else 0.0,
            retrieval_injection_rate=self.injections / self.retrievals if self.retrievals else 0.0)

    def begin(self, task_id, state_id):
        task_id, state_id = int(task_id), int(state_id)
        if self.task_id is not None and self.task_id != task_id:
            raise ValueError('A memory server serves one task; start another server for the next task.')
        if self.episode is not None:
            raise ValueError('Previous memory episode has not received its completion signal.')
        self.task_id = task_id
        self.episode = (task_id, state_id)
        self.buffer = []
        self.chunk_index = 0

    def finish(self, task_id, state_id, success):
        if self.episode != (int(task_id), int(state_id)):
            raise ValueError('Episode completion does not match the active task/state.')
        if success:
            for views in {record[0] for record in self.buffer}:
                records = [record for record in self.buffer if record[0] == views]
                memory = self.memories.setdefault(views, SuccessfulMemory(self.config['max_entries'], self.config))
                memory.add(np.stack([record[1] for record in records]),
                           np.stack([record[2] for record in records]),
                           [record[3] for record in records])
            self.successes += 1
        entries = sum(len(memory) for memory in self.memories.values())
        logging.info('[memory] task=%s state=%s success=%s successful_episodes=%d entries=%d',
                     task_id, state_id, bool(success), self.successes, entries)
        self.buffer = []
        self.episode = None
        self.episodes += 1
        logging.info('[memory] summary: %s', json.dumps(self.summary(), sort_keys=True))
        return dict(ok=True, successful_episodes=self.successes, memory_entries=entries)

    def sample(self, rng, observation, **kwargs):
        if self.episode is None:
            raise ValueError('Memory inference needs an episode start.')
        if self.config['sampling_steps'] != 10:
            kwargs.setdefault('num_steps', self.config['sampling_steps'])
        views = real_views(observation)
        if self.view_mode == 'base':
            views = tuple(name for name in views if name == 'base_0_rgb')
        key = None
        max_sim = sim_mean = 'NA'
        candidate_count = kept_count = memory_entries = 0
        scope = 'none'
        injected = False
        actual_t0 = 1.0
        reason = 'no_valid_views'
        dtw_info = dict(before=0, after=0, removed=0, sigma=0.0, median=0.0)
        confidence = 'NA'
        if views:
            if views not in self.encoders:
                # Each view set has fixed shape; masked cameras are never computed or concatenated.
                def compute(model, obs):
                    return visual_key(model, obs, views, self.config['pooling'],
                        self.config['feature_layer'], self.config['wrist_weight'], self.config['view_normalize'])
                self.encoders[views] = nnx_utils.module_jit(types.MethodType(compute, self.model))
                logging.info('[memory] retrieval views=%s', views)
            key = np.asarray(self.encoders[views](observation)[0], dtype=np.float32)
            if self.config['state_weight']:
                state = np.asarray(observation.state[0], dtype=np.float32)
                state_part = np.clip(state, -3, 3) * self.config['state_weight'] / np.sqrt(state.size)
                key = np.concatenate([key / (np.linalg.norm(key) + 1e-8), state_part]).astype(np.float32)
            memory = self.memories.get(views)
            memory_entries = len(memory) if memory is not None else 0
            reason = 'empty_memory' if not memory_entries else 'external_noise'
            if memory is not None and len(memory) and kwargs.get('noise') is None:
                actions, similarities, details = memory.retrieve(key, self.chunk_index, self.config['top_k'], diagnostics=True)
                self.retrievals += 1
                max_sim = '%.9f' % details['max_sim']
                candidate_count, scope = details['candidates'], details['scope']
                actions, similarities, dtw_info = dtw_consistency(actions, similarities, self.config)
                kept_count = len(similarities)
                reason = 'below_threshold'
                if len(similarities):
                    confidence = '%.6f' % retrieval_confidence(similarities, self.config, dtw_info['sigma'])
                    t0 = adaptive_t0(similarities, self.config, dtw_info['sigma'])
                    sim_mean = '%.9f' % float(np.mean(similarities))
                    reason = 't0_fallback'
                    if t0 < 0.98:
                        weights = np.exp((similarities - similarities.max()) / self.config['tau'])
                        if self.config['rotation_aggregation'] == 'so3':
                            prior = aggregate_actions(actions, weights)
                        else:
                            prior = (actions * (weights / weights.sum())[:, None, None]).sum(axis=0)
                        eps = jax.random.normal(rng, prior.shape, dtype=jnp.float32)
                        kwargs['noise'] = (t0 * eps + (1 - t0) * jnp.asarray(prior))[None]
                        kwargs['start_time'] = jnp.asarray(t0, dtype=jnp.float32)
                        if self.config['exact_endpoint']:
                            kwargs['exact_endpoint'] = True
                        actual_t0 = t0
                        injected = True
                        reason = 'injected'
                        self.injections += 1
        self.inferences += 1
        logging.info('[memory] retrieval: task=%d state=%d chunk=%d memory_entries=%d '
                     'scope=%s candidates=%d max_sim=%s kept=%d sim_mean=%s '
                     'injected=%s t0=%.6f prior_weight=%.6f reason=%s '
                     'dtw_enabled=%s gated=%d dtw_kept=%d dtw_removed=%d '
                     'dtw_sigma=%.6f dtw_median=%.6f confidence=%s',
                     *self.episode, self.chunk_index, memory_entries, scope, candidate_count,
                     max_sim, kept_count, sim_mean, injected, actual_t0,
                     1.0 - actual_t0 if injected else 0.0, reason,
                     self.config['dtw_filter'], dtw_info['before'], dtw_info['after'],
                     dtw_info['removed'], dtw_info['sigma'], dtw_info['median'], confidence)
        result = self.sampler(rng, observation, **kwargs)
        if key is not None:
            self.buffer.append((views, key, np.asarray(result[0], dtype=np.float32), self.chunk_index))
        self.chunk_index += 1
        return result


def enable_memory():
    """Install only in the memory server process; official files and baseline modes stay intact."""
    _pi0.Pi0.sample_actions = sample_actions
    original_init = Policy.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self._is_pytorch_model or not isinstance(self._model, _pi0.Pi0):
            raise ValueError('Memory mode requires a JAX pi0/pi05 checkpoint.')
        self._memory = MemoryAdapter(self._model, self._sample_actions)
        self._sample_actions = self._memory.sample

    Policy.__init__ = init
