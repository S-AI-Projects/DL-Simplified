"""Bounded CPU smoke run for R42-T036."""
import math
from typing import Any, Dict, List

class BoundedCPULogisticModel:
    def __init__(self, input_dim: int = 16, seed: int = 42):
        self.input_dim = input_dim
        self.seed = seed
        self.weights = [0.01 * ((i * seed) % 5) for i in range(input_dim)]
        self.bias = 0.0

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))

    def train_smoke_run(self, records: List[Dict[str, Any]], max_steps: int = 10, lr: float = 0.1) -> Dict[str, Any]:
        if len(records) > 32:
            records = records[:32]
        
        step_count = min(max_steps, 20)
        loss_history = []

        for step in range(step_count):
            total_loss = 0.0
            for r in records:
                x = [val / 255.0 for row in r["pixels"] for val in row]
                y = float(r["label"])
                z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
                p = self._sigmoid(z)
                loss = -(y * math.log(max(p, 1e-9)) + (1.0 - y) * math.log(max(1.0 - p, 1e-9)))
                total_loss += loss

                # Simple gradient update step
                err = p - y
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * err * x[i]
                self.bias -= lr * err

            loss_history.append(total_loss / len(records))

        return {
            "step_count": step_count,
            "final_loss": loss_history[-1],
            "finite_loss": math.isfinite(loss_history[-1])
        }

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        preds = []
        for r in records:
            x = [val / 255.0 for row in r["pixels"] for val in row]
            z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            preds.append(self._sigmoid(z))
        return preds
