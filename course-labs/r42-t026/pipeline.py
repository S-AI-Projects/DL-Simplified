"""
R42-T026: Bounded CPU smoke test pipeline: at most 32 samples, fixed seed, at most 1 epoch / 20 steps.
Asserts interface correctness without claiming real-world accuracy.
"""
from typing import Any, Dict, List
import random

class SimpleSmokeLinearModel:
    def __init__(self, vocab_size: int, seed: int = 42):
        random.seed(seed)
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(vocab_size + 1)]
        self.bias = 0.0

    def train_step(self, token_ids: List[int], label: int, lr: float = 0.01) -> float:
        # Simple logistic regression step
        score = self.bias + sum(self.weights[tid] for tid in token_ids if tid < len(self.weights))
        pred = 1.0 / (1.0 + 2.71828 ** (-score))
        err = label - pred
        # gradient update
        self.bias += lr * err
        for tid in token_ids:
            if tid < len(self.weights):
                self.weights[tid] += lr * err
        return (pred - label) ** 2

    def run_smoke_epoch(self, records: List[Dict[str, Any]], max_steps: int = 20) -> Dict[str, Any]:
        steps = min(len(records), max_steps)
        losses = []
        for i in range(steps):
            rec = records[i]
            loss = self.train_step(rec["token_ids"], rec["label"])
            losses.append(loss)
        return {
            "steps_run": steps,
            "final_loss": round(losses[-1], 4) if losses else 0.0,
            "bounded_cpu_smoke": True
        }
