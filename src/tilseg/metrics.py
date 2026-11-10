"""Overlap metrics between predicted and reference binary masks.

Every non-zero pixel counts as foreground, so masks stored as 0/1 (nnU-Net
labels) and 0/255 (visual masks) can be mixed freely.

When both masks are empty the overlap is undefined and the metric is ``nan``;
summary statistics ignore those cases, which is also what nnU-Net's own
evaluation does.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def _foreground(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def confusion(pred: np.ndarray, ref: np.ndarray) -> tuple[int, int, int]:
    """True positives, false positives and false negatives (pixel counts)."""
    p, r = _foreground(pred), _foreground(ref)
    if p.shape != r.shape:
        raise ValueError(f"Shape mismatch: prediction {p.shape} vs reference {r.shape}")
    tp = int(np.count_nonzero(p & r))
    return tp, int(np.count_nonzero(p)) - tp, int(np.count_nonzero(r)) - tp


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    """Dice similarity coefficient: ``2|A n B| / (|A| + |B|)``."""
    tp, fp, fn = confusion(pred, ref)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else math.nan


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """Intersection over union (Jaccard index)."""
    tp, fp, fn = confusion(pred, ref)
    denom = tp + fp + fn
    return tp / denom if denom else math.nan


def precision(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, fp, _ = confusion(pred, ref)
    return tp / (tp + fp) if tp + fp else math.nan


def recall(pred: np.ndarray, ref: np.ndarray) -> float:
    tp, _, fn = confusion(pred, ref)
    return tp / (tp + fn) if tp + fn else math.nan


METRICS = {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
    return arr.any(axis=-1) if arr.ndim == 3 else arr


def evaluate_folder(pred_dir: str | Path, ref_dir: str | Path) -> list[dict]:
    """Per-case metrics for every PNG in ``pred_dir`` with a same-named reference.

    Raises:
        FileNotFoundError: if a prediction has no reference mask.
    """
    pred_dir, ref_dir = Path(pred_dir), Path(ref_dir)
    rows: list[dict] = []
    for pred_path in sorted(pred_dir.glob("*.png")):
        ref_path = ref_dir / pred_path.name
        if not ref_path.exists():
            raise FileNotFoundError(f"No reference mask for {pred_path.name} in {ref_dir}")
        pred, ref = _read_mask(pred_path), _read_mask(ref_path)
        rows.append({"case": pred_path.stem, **{k: fn(pred, ref) for k, fn in METRICS.items()}})
    return rows


def summarize(rows: list[dict], metric: str = "dice") -> dict:
    """Mean, population standard deviation, median, min and max over cases.

    Cases where the metric is ``nan`` (both masks empty) are excluded; ``n`` is
    the number of cases that entered the statistics.
    """
    values = np.array([r[metric] for r in rows], dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"metric": metric, "n": 0}
    return {
        "metric": metric,
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_csv(rows: list[dict], path: str | Path) -> None:
    """Write per-case metrics to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def load_nnunet_dice(summary_json: str | Path, label: str = "1") -> list[float]:
    """Per-case Dice of one label from an nnU-Net v2 evaluation ``summary.json``."""
    with open(summary_json, encoding="utf-8") as fh:
        data = json.load(fh)
    values = []
    for case in data.get("metric_per_case", []):
        value = case.get("metrics", {}).get(label, {}).get("Dice")
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return values
