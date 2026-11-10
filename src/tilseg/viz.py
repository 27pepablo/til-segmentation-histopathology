"""Figures: contour overlays, side-by-side comparisons and Dice box plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

YELLOW_BGR = (0, 255, 255)
BLACK_BGR = (0, 0, 0)


def overlay_contours(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = YELLOW_BGR,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw the outline of every mask region on a copy of the image.

    A black halo is drawn under the coloured line so the contour stays visible
    on both pale stroma and dark lymphocyte-rich areas. ``thickness`` defaults
    to ~0.25 % of the longest image side.
    """
    if thickness is None:
        thickness = max(2, round(max(image_bgr.shape[:2]) * 0.0025))
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image_bgr.copy()
    cv2.drawContours(out, contours, -1, BLACK_BGR, thickness=thickness * 2 + 2)
    cv2.drawContours(out, contours, -1, color, thickness=thickness)
    return out


def side_by_side(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_title: str,
    right_title: str,
    title_height: int = 60,
) -> np.ndarray:
    """Place two images next to each other with a title above each."""
    if left_bgr.shape[:2] != right_bgr.shape[:2]:
        right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]))
    h, w = left_bgr.shape[:2]
    canvas = np.full((h + title_height, 2 * w, 3), 255, dtype=np.uint8)
    canvas[title_height:, :w] = left_bgr
    canvas[title_height:, w:] = right_bgr
    cv2.line(canvas, (w, title_height), (w, h + title_height), BLACK_BGR, 4)

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    for text, x0 in ((left_title, 0), (right_title, w)):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        org = (x0 + (w - tw) // 2, int(title_height * 0.68))
        cv2.putText(canvas, text, org, font, scale, BLACK_BGR, thick, cv2.LINE_AA)
    return canvas


def overlay_file(image_path: str | Path, mask_path: str | Path, out_path: str | Path) -> None:
    """Read an image and a mask from disk and save the contour overlay."""
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay_contours(image, mask))


def dice_boxplot(
    groups: dict[str, list[float]],
    out_path: str | Path,
    title: str = "",
    colors: tuple[str, ...] = ("#5DADE2", "#58D68D", "#F5B041", "#AF7AC5"),
) -> None:
    """Box plot of per-case Dice for several models, annotated with mean +/- std."""
    labels, values = list(groups), [np.asarray(v, dtype=float) for v in groups.values()]
    fig, ax = plt.subplots(figsize=(2.6 * len(labels) + 2, 5))
    box = ax.boxplot(values, patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], colors, strict=False):
        patch.set_facecolor(color)
    for i, v in enumerate(values, start=1):
        ax.text(i, 0.02, f"{v.mean():.2f} ± {v.std():.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Dice coefficient")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
