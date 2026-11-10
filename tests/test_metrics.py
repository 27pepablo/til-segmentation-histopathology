import json
import math

import numpy as np
import pytest

from tilseg import metrics


def test_dice_and_iou_known_values():
    ref = np.zeros((4, 4), dtype=np.uint8)
    ref[:2] = 1  # 8 px
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[1:3] = 1  # 8 px, 4 shared
    assert metrics.dice(pred, ref) == pytest.approx(0.5)
    assert metrics.iou(pred, ref) == pytest.approx(1 / 3)
    assert metrics.precision(pred, ref) == pytest.approx(0.5)
    assert metrics.recall(pred, ref) == pytest.approx(0.5)


def test_perfect_and_disjoint():
    a = np.eye(4, dtype=np.uint8)
    assert metrics.dice(a, a) == 1.0
    assert metrics.dice(a, 1 - a) == 0.0


def test_binary_and_255_masks_are_equivalent():
    ref = np.eye(4, dtype=np.uint8)
    assert metrics.dice(ref * 255, ref) == 1.0


def test_both_empty_is_nan_and_excluded_from_summary():
    empty = np.zeros((4, 4), dtype=np.uint8)
    assert math.isnan(metrics.dice(empty, empty))
    rows = [{"dice": 0.8}, {"dice": math.nan}, {"dice": 0.4}]
    s = metrics.summarize(rows)
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.6) and s["std"] == pytest.approx(0.2)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.dice(np.zeros((2, 2)), np.zeros((3, 3)))


def test_evaluate_folder_and_csv(make_png, tmp_path):
    ref = np.zeros((4, 4))
    ref[:2] = 1
    make_png("ref/a.png", ref)
    make_png("ref/b.png", ref)
    make_png("pred/a.png", ref)
    make_png("pred/b.png", 1 - ref)

    rows = metrics.evaluate_folder(tmp_path / "pred", tmp_path / "ref")
    assert [(r["case"], r["dice"]) for r in rows] == [("a", 1.0), ("b", 0.0)]

    metrics.save_csv(rows, tmp_path / "out" / "m.csv")
    assert (tmp_path / "out" / "m.csv").read_text().splitlines()[0] == "case,dice,iou,precision,recall"


def test_evaluate_folder_requires_reference(make_png, tmp_path):
    make_png("pred/a.png", np.zeros((2, 2)))
    (tmp_path / "ref").mkdir()
    with pytest.raises(FileNotFoundError):
        metrics.evaluate_folder(tmp_path / "pred", tmp_path / "ref")


def test_load_nnunet_dice(tmp_path):
    summary = {
        "metric_per_case": [
            {"metrics": {"1": {"Dice": 0.7}}},
            {"metrics": {"1": {"Dice": float("nan")}}},
            {"metrics": {"1": {"Dice": 0.9}}},
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    assert metrics.load_nnunet_dice(path) == [0.7, 0.9]
