"""Summary figure of the Dice scores reported in the thesis (Table 3).

The per-case predictions live on the research group's server together with the
datasets, so this plot is drawn from the reported mean and standard deviation
of each model (``docs/results.json``), not from per-case values.

    python examples/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results.json"
OUT = ROOT / "docs" / "figures" / "dice_summary.png"

BASELINE, CELLVIT = "#5DADE2", "#58D68D"


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))["experiments"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    xs, ticks = [], []
    x = 0.0
    for group in ("TCGA test set (30 tiles)", "External cohort (87 ROIs)"):
        rows = [r for r in results if r["evaluation"] == group]
        for r in rows:
            color = CELLVIT if r["labels"] == "cellvit" else BASELINE
            # Dice is bounded by 1, so the upper whisker is clipped there.
            yerr = [[r["dice_std"]], [min(r["dice_std"], 1 - r["dice_mean"])]]
            ax.bar(x, r["dice_mean"], 0.8, yerr=yerr, color=color, edgecolor="black",
                   capsize=5, error_kw={"elinewidth": 1})
            ax.text(x, 0.03, f"{r['dice_mean']:.2f}", ha="center", fontsize=11, fontweight="bold")
            xs.append(x)
            x += 0.9
        ticks.append((sum(xs[-len(rows):]) / len(rows), group))
        x += 0.8

    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=11)
    ax.set_ylabel("Dice (mean ± std over cases)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE, edgecolor="black"),
            plt.Rectangle((0, 0), 1, 1, facecolor=CELLVIT, edgecolor="black"),
        ],
        labels=["nnU-Net, original annotations", "nnU-Net, CellViT++-derived labels"],
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"Figure saved to {OUT}")


if __name__ == "__main__":
    main()
