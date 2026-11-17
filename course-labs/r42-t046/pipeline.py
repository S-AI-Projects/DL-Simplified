"""Bounded CPU smoke run for Weather images - CNN.

Implements a deterministic, bounded single-epoch / <= 20 update step linear classifier.
Requires no GPU, no remote credentials, and no external datasets.
Teaching fixture: Six synthetic 4x4 grayscale images.
This is a CPU smoke fixture, not the named real dataset.
"""

import math
import random
from typing import Any, Dict, List, Tuple


class TinyLinearClassifier:
    """Tiny linear classifier trained via stochastic gradient descent."""

    def __init__(self, input_dim: int = 16, seed: int = 42):
        random.seed(seed)
        # Small deterministic initial weights and bias
        self.weights = [(random.random() - 0.5) * 0.1 for _ in range(input_dim)]
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        # Clamp to avoid overflow
        z = max(-50.0, min(50.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def forward(self, x: List[float]) -> float:
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def train_bounded(
        self,
        features: List[List[float]],
        labels: List[int],
        max_steps: int = 10,
        lr: float = 0.1
    ) -> Dict[str, Any]:
        """Run at most max_steps updates (bounded <= 20 steps, 1 epoch)."""
        steps = 0
        total_loss = 0.0

        for x, y in zip(features, labels):
            if steps >= max_steps:
                break
            pred = self.forward(x)
            # Binary cross entropy loss
            eps = 1e-12
            loss = -(y * math.log(pred + eps) + (1 - y) * math.log(1.0 - pred + eps))
            total_loss += loss

            # Gradient update
            err = pred - y
            for i in range(len(self.weights)):
                self.weights[i] -= lr * err * x[i]
            self.bias -= lr * err
            steps += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        return {
            "steps_recorded": steps,
            "average_loss": avg_loss,
            "loss_is_finite": math.isfinite(avg_loss)
        }

    def predict(self, features: List[List[float]], threshold: float = 0.5) -> List[int]:
        return [1 if self.forward(x) >= threshold else 0 for x in features]


def flatten_record_pixels(records: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
    """Flatten 4x4 matrix into 16 normalized float features."""
    features = []
    labels = []
    for r in records:
        flat = []
        for row in r["pixels"]:
            for p in row:
                flat.append(float(p) / 255.0)
        features.append(flat)
        labels.append(r["label"])
    return features, labels
