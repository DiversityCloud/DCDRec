import time
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm


class MetricTracker:
    def __init__(self, k_list: Iterable[int]):
        self.k_list = list(k_list)
        self.metrics = {f"HR@{k}": 0.0 for k in self.k_list}
        self.metrics.update({f"NDCG@{k}": 0.0 for k in self.k_list})
        self.total_samples = 0

    def update(self, topk_items: torch.Tensor, targets: torch.Tensor) -> None:
        for k in self.k_list:
            topk_items_k = topk_items[:, :k]
            hits_k = (topk_items_k == targets.unsqueeze(1)).float()
            self.metrics[f"HR@{k}"] += hits_k.sum().item()

            nonzero = hits_k.nonzero(as_tuple=False)
            for row in nonzero:
                rank = row[1].item()
                self.metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 2)
        self.total_samples += targets.size(0)

    def compute(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {key: 0.0 for key in self.metrics}
        return {key: value / self.total_samples for key, value in self.metrics.items()}


class Trainer:
    def __init__(self, model, optimizer, device, k_list=(5, 10, 20, 50), num_neg=100):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.k_list = list(k_list)
        self.num_neg = num_neg

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def train_one_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        running = {
            "total_loss": 0.0,
            "contrastive_loss": 0.0,
            "diffusion_loss": 0.0,
        }
        num_batches = 0

        for batch in tqdm(train_loader, desc="Train", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch, training=True)
            loss = outputs["total_loss"]
            loss.backward()
            self.optimizer.step()

            running["total_loss"] += loss.item()
            running["contrastive_loss"] += outputs["contrastive_loss"].item()
            running["diffusion_loss"] += outputs["diffusion_loss"].item()
            num_batches += 1

        return {key: value / max(1, num_batches) for key, value in running.items()}

    def evaluate(self, data_loader, return_timing: bool = False) -> Tuple[Dict[str, float], float, float]:
        self.model.eval()
        rng = np.random.default_rng(2025)
        tracker = MetricTracker(self.k_list)
        reverse_infer_time = 0.0

        self._sync()
        t0_eval = time.perf_counter()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Eval", leave=False):
                self._sync()
                t0_model = time.perf_counter()
                logits_full = self.model(batch, training=False)
                self._sync()
                reverse_infer_time += time.perf_counter() - t0_model

                targets = batch["target_movies"].to(self.device)
                history = batch["movie_sequences"]
                batch_size = targets.size(0)
                num_items = self.model.num_items

                candidate_ids = torch.empty(batch_size, 1 + self.num_neg, dtype=torch.long)
                for i in range(batch_size):
                    exclude = set(history[i].tolist()) | {targets[i].item()}
                    negs = []
                    while len(negs) < self.num_neg:
                        samples = rng.integers(0, num_items, size=(self.num_neg - len(negs)) * 2)
                        negs.extend([sid for sid in samples if sid not in exclude])
                    candidate = [targets[i].item()] + negs[: self.num_neg]
                    candidate_ids[i] = torch.tensor(candidate, dtype=torch.long)

                candidate_ids = candidate_ids.to(self.device)
                candidate_logits = logits_full.gather(1, candidate_ids)
                _, topk_idx = torch.topk(candidate_logits, max(self.k_list), dim=1)
                topk_items = candidate_ids.gather(1, topk_idx)
                tracker.update(topk_items, targets)

        self._sync()
        eval_total_time = time.perf_counter() - t0_eval
        metrics = tracker.compute()

        if return_timing:
            return metrics, reverse_infer_time, eval_total_time
        return metrics, 0.0, 0.0


def format_metrics(metrics: Dict[str, float], k_list=(5, 10, 20, 50), prefix: str = "") -> str:
    prefix_str = f"{prefix} " if prefix else ""
    parts = []
    for k in k_list:
        parts.append(
            f"{prefix_str}HR@{k}={metrics[f'HR@{k}']:.4f} "
            f"NDCG@{k}={metrics[f'NDCG@{k}']:.4f} "
        )
    return " | ".join(parts)
