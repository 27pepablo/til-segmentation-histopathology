"""Run the cell -> region conversion on synthetic data (no datasets or GPU needed).

Simulates what CellViT++ would return for one 2048x2048 tile (three lymphocyte
aggregates of different shape plus scattered single cells), writes it in the
same GeoJSON layout, and runs the exact code path used on real tiles:

    GeoJSON -> TIL centroids -> density filter -> dilation + closing -> mask

The figure it saves (``docs/figures/synthetic_demo.png``) shows each stage and
compares the result with the convex-hull alternative.

    python examples/synthetic_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tilseg import cellvit_io, metrics, til_regions  # noqa: E402

SIZE = 2048
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "synthetic_demo.png"


def simulate_cells(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Synthetic nuclei centroids per PanNuke class, as (x, y) pixels."""
    blob = rng.normal((520, 480), (110, 90), size=(420, 2))  # compact aggregate
    t = rng.uniform(0, 1, 380)  # elongated band following a curved vessel-like path
    band = np.column_stack([900 + 800 * t, 1350 + 260 * np.sin(3 * t)])
    band += rng.normal(0, 38, band.shape)
    theta = rng.uniform(0.2 * np.pi, 1.6 * np.pi, 330)  # C-shaped (concave) infiltrate
    crescent = np.column_stack([1500 + 300 * np.cos(theta), 520 + 300 * np.sin(theta)])
    crescent += rng.normal(0, 30, crescent.shape)
    scattered = rng.uniform(0, SIZE, size=(160, 2))  # isolated lymphocytes

    til = np.vstack([blob, band, crescent, scattered])
    til = til[((til >= 0) & (til < SIZE)).all(axis=1)]
    return {
        "Inflammatory": til,
        "Neoplastic": rng.uniform(0, SIZE, size=(2500, 2)),
        "Connective": rng.uniform(0, SIZE, size=(900, 2)),
    }


def write_cellvit_geojson(cells: dict[str, np.ndarray], path: Path) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "MultiPoint", "coordinates": np.round(pts, 1).tolist()},
            "properties": {"objectType": "annotation", "classification": {"name": name}},
        }
        for name, pts in cells.items()
    ]
    path.write_text(json.dumps([features]), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(7)
    cells = simulate_cells(rng)
    params = til_regions.RegionParams()

    with tempfile.TemporaryDirectory() as tmp:
        geojson = Path(tmp) / "synthetic_cell_detection.geojson"
        write_cellvit_geojson(cells, geojson)
        counts = cellvit_io.class_counts(geojson)
        til = cellvit_io.load_centroids(geojson)

    density = til_regions.local_density(til, params.radius)
    keep = density >= params.min_density
    mask = til_regions.region_mask(til, (SIZE, SIZE), params)
    hull = til_regions.hull_region_mask(til, (SIZE, SIZE))

    print("Cells per class:", dict(counts))
    print(f"TILs kept by the density filter: {keep.sum()} / {len(til)}")
    print(f"TIL-region area (morphology):  {100 * til_regions.til_area_fraction(mask):.1f} %")
    print(f"TIL-region area (convex hull): {100 * til_regions.til_area_fraction(hull):.1f} %")
    print(f"Dice between the two variants: {metrics.dice(hull, mask):.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    other = np.vstack([cells["Neoplastic"], cells["Connective"]])
    axes[0].scatter(other[:, 0], other[:, 1], s=1.5, c="#c9c9c9", label="other nuclei")
    axes[0].scatter(til[:, 0], til[:, 1], s=4, c="#1e8449", label="Inflammatory (TIL)")
    axes[0].set_title("1. CellViT++ detections")
    axes[0].legend(loc="lower left", fontsize=8, markerscale=3, framealpha=0.9)

    axes[1].scatter(til[~keep, 0], til[~keep, 1], s=4, c="#bdbdbd", label="discarded")
    sc = axes[1].scatter(til[keep, 0], til[keep, 1], s=4, c=density[keep] * 1e3, cmap="viridis")
    cax = axes[1].inset_axes([0.52, 0.06, 0.42, 0.035])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label(r"TILs / 1000 px$^2$", fontsize=8, labelpad=2)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=7)
    axes[1].set_title(f"2. Density filter (R = {params.radius:.0f} px)")
    axes[1].legend(loc="lower left", fontsize=8, markerscale=3, framealpha=0.9)

    axes[2].imshow(mask, cmap="Greys", vmin=0, vmax=1)
    axes[2].set_title("3. Dilation + closing  →  training label")

    axes[3].imshow(hull, cmap="Greys", vmin=0, vmax=1)
    axes[3].contour(mask, levels=[0.5], colors="#d35400", linewidths=1.2)
    axes[3].set_title("Alternative: DBSCAN + convex hull")

    for ax in axes:
        ax.set_xlim(0, SIZE)
        ax.set_ylim(SIZE, 0)  # image convention: origin at the top-left corner
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"Figure saved to {OUT}")


if __name__ == "__main__":
    main()
