"""Read CellViT++ outputs and prepare its inputs.

CellViT++ (``detect_cells.py --geojson``) writes, for every slide, a
``<slide>_cell_detection.geojson`` file with one ``MultiPoint`` feature per cell
class. Each coordinate is the ``[x, y]`` centroid of a detected nucleus in pixel
units of the input image, and the class name is stored under
``properties.classification.name`` (PanNuke classes: Neoplastic, Inflammatory,
Connective, Dead, Epithelial).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

TIL_CLASS = "Inflammatory"
DETECTION_SUFFIX = "_cell_detection.geojson"


def _iter_features(data) -> list[dict]:
    """Flatten the GeoJSON root, which can be a FeatureCollection or (nested) lists."""
    if isinstance(data, dict):
        if "features" in data:
            return _iter_features(data["features"])
        return [data]
    features: list[dict] = []
    for item in data:
        features.extend(_iter_features(item))
    return features


def load_features(path: str | Path) -> list[dict]:
    """Load every GeoJSON feature of a CellViT++ output file."""
    with open(path, encoding="utf-8") as fh:
        return _iter_features(json.load(fh))


def class_counts(path: str | Path) -> Counter:
    """Number of detected cells per class in a ``*_cell_detection.geojson`` file."""
    counts: Counter = Counter()
    for feat in load_features(path):
        name = feat.get("properties", {}).get("classification", {}).get("name")
        geom = feat.get("geometry", {})
        n = len(geom.get("coordinates", [])) if geom.get("type") == "MultiPoint" else 1
        counts[name] += n
    return counts


def load_centroids(path: str | Path, class_name: str = TIL_CLASS) -> np.ndarray:
    """Centroids of the cells of one class as an ``(N, 2)`` array of ``(x, y)``.

    ``x`` is the column and ``y`` the row of the input image. Returns an empty
    ``(0, 2)`` array when the class is absent.
    """
    points: list[tuple[float, float]] = []
    for feat in load_features(path):
        name = feat.get("properties", {}).get("classification", {}).get("name")
        if name != class_name:
            continue
        geom = feat.get("geometry", {})
        if geom.get("type") == "MultiPoint":
            points.extend((float(x), float(y)) for x, y, *_ in geom.get("coordinates", []))
        elif geom.get("type") == "Point":
            x, y, *_ = geom["coordinates"]
            points.append((float(x), float(y)))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def write_filelist(
    image_dir: str | Path,
    out_csv: str | Path,
    mpp: float = 0.25,
    magnification: int = 40,
    extensions: tuple[str, ...] = (".tif", ".tiff"),
) -> int:
    """Write the CSV consumed by ``detect_cells.py process_dataset --filelist``.

    Columns: ``path``, ``slide_mpp``, ``magnification``. Returns the number of
    slides listed.
    """
    image_dir, out_csv = Path(image_dir), Path(out_csv)
    slides = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "slide_mpp", "magnification"])
        for slide in slides:
            writer.writerow([str(slide), mpp, magnification])
    return len(slides)
