"""
Aggregate the 5-fold results for both models into overall metrics.

Two views: per-fold UAR (mean/std across the 5 independent test folds) and
pooled UAR (sum the 5 confusion matrices into one, covering all 535 clips,
then compute macro recall on that).
"""

import numpy as np
import torch

from dataset import EMOTIONS
from paths import PROCESSED_DIR


def load_fold_results(prefix: str, n_folds: int = 5) -> list[dict]:
    return [torch.load(PROCESSED_DIR / f"{prefix}_fold{i}_result.pt", weights_only=False)
            for i in range(n_folds)]


def summarize(name: str, results: list[dict]) -> None:
    test_uars = [r["test_uar"] for r in results]
    print(f"\n=== {name} ===")
    print(f"Per-fold test UAR: {[f'{u:.3f}' for u in test_uars]}")
    print(f"Mean +/- std: {np.mean(test_uars):.3f} +/- {np.std(test_uars):.3f}")

    pooled_cm = sum(r["confusion_matrix"] for r in results)
    row_sums = pooled_cm.sum(axis=1, keepdims=True)
    per_class_recall = np.diag(pooled_cm) / np.clip(row_sums.flatten(), 1, None)
    pooled_uar = per_class_recall.mean()

    print(f"Pooled UAR (all 535 clips, one confusion matrix): {pooled_uar:.3f}")
    print("Pooled per-class recall:")
    for e, r in zip(EMOTIONS, per_class_recall):
        print(f"  {e:10s}: {r:.2f}")
    print("Pooled confusion matrix (rows=true, cols=pred):")
    print("           " + "  ".join(f"{e[:4]:>4s}" for e in EMOTIONS))
    for e, row in zip(EMOTIONS, pooled_cm):
        print(f"  {e:10s}" + "  ".join(f"{v:4d}" for v in row))


if __name__ == "__main__":
    scratch_results = load_fold_results("scratch")
    transfer_results = load_fold_results("transfer")

    summarize("Model 1 (from scratch)", scratch_results)
    summarize("Model 2 (transfer learning)", transfer_results)