"""Draw the method overview diagram shown in the README.

    python examples/draw_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "pipeline.png"

BLUE, GREEN, GREY, DARK = "#5DADE2", "#58D68D", "#ECEFF1", "#263238"
W, H = 2.6, 1.15  # box size


def box(ax, x, y, title, subtitle, color, bold=False):
    ax.add_patch(
        FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.14",
                       facecolor=color, edgecolor=DARK, linewidth=1.6 if bold else 1.1)
    )
    ax.text(x + W / 2, y + H * 0.64, title, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=DARK)
    ax.text(x + W / 2, y + H * 0.27, subtitle, ha="center", va="center", fontsize=9, color=DARK)


def arrow(ax, start, end, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.4,
                        color=DARK, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 4.9))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    top, bottom, mid = 3.35, 0.45, 1.9
    xs = [0.2, 3.35, 6.5, 9.65, 12.8]

    box(ax, xs[0], mid, "H&E tiles", "TCGA ROIs · 0.25 µm/px", GREY)

    # Baseline branch (top)
    box(ax, xs[1], top, "Original annotations", "class map → binary TIL mask", BLUE)
    box(ax, xs[3], top, "nnU-Net 2D", "baseline model", BLUE)

    # CellViT branch (bottom)
    box(ax, xs[1], bottom, "CellViT++", "pretrained nuclei detector", GREEN)
    box(ax, xs[2], bottom, "Cells → regions", "density filter + morphology", GREEN, bold=True)
    box(ax, xs[3], bottom, "nnU-Net 2D", "same setup, new labels", GREEN)

    box(ax, xs[4], mid, "Evaluation", "Dice on held-out tiles\nand an external cohort", GREY)

    cy = lambda y: y + H / 2  # noqa: E731
    arrow(ax, (xs[0] + W, cy(mid) + 0.2), (xs[1], cy(top)), rad=-0.15)
    arrow(ax, (xs[0] + W, cy(mid) - 0.2), (xs[1], cy(bottom)), rad=0.15)
    arrow(ax, (xs[1] + W, cy(top)), (xs[3], cy(top)))
    arrow(ax, (xs[1] + W, cy(bottom)), (xs[2], cy(bottom)))
    arrow(ax, (xs[2] + W, cy(bottom)), (xs[3], cy(bottom)))
    arrow(ax, (xs[3] + W, cy(top)), (xs[4], cy(mid) + 0.2), rad=-0.15)
    arrow(ax, (xs[3] + W, cy(bottom)), (xs[4], cy(mid) - 0.2), rad=0.15)

    ax.text(xs[2] + W / 2, top + H + 0.18, "Experiment 1 · train on the labels as published",
            ha="center", fontsize=10.5, color="#1F618D", fontweight="bold")
    ax.text(xs[2] + W / 2, bottom - 0.3,
            "Experiment 2 · rebuild the labels from cell detections, retrain",
            ha="center", fontsize=10.5, color="#1E8449", fontweight="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Figure saved to {OUT}")


if __name__ == "__main__":
    main()
