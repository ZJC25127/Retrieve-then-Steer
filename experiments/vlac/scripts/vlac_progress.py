"""Progress accumulation and causal memory boundaries, independent of model runtimes."""
import math

class Progress:
    def __init__(self, threshold=95., milestone=64):
        self.threshold = threshold
        self.milestone = milestone
        self.base = self.value = 0.
        self.reference_step = 0
        self.last_checked = 0
        self.first_hit = None

    def update(self, step, score):
        if step <= self.last_checked:
            raise ValueError('Progress checks must advance actual executed action steps')
        if not math.isfinite(score): raise ValueError('Nonfinite critic score')
        score = max(-100., min(100., score))
        self.value = max(0., min(100., self.base+(100.-self.base)*score/100.))
        self.last_checked = step
        if self.first_hit is None and self.value >= self.threshold: self.first_hit = step
        refresh = step-self.reference_step >= self.milestone
        if refresh:
            self.base = self.value
            self.reference_step = step
        return refresh

def complete_chunks(starts, boundary, replan=5):
    """Exclude the partially executed replanning window at the first success crossing."""
    if boundary is None: return 0
    return sum(start+replan <= boundary for start in starts)

def prepare_commit(memory, event):
    """Apply the label to the existing buffer; finish remains the frozen implementation."""
    mode = event['mode']
    if mode not in ('observe','label'): raise ValueError('Invalid VLAC mode')
    before = len(memory.buffer)
    accepted = bool(event['env_success']) if mode == 'observe' else bool(event['critic_success'])
    if mode == 'label':
        limit = int(event['complete_chunks'])
        if not 0 <= limit <= memory.chunk_index: raise ValueError('Invalid execution boundary')
        memory.buffer = [record for record in memory.buffer if record[3] < limit]
        accepted = accepted and bool(memory.buffer)
    return accepted, before, len(memory.buffer) if accepted else 0
