import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def make_png(tmp_path):
    """Write an array as PNG under tmp_path and return the path."""

    def _make(rel_path, array, palette=False):
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(np.asarray(array, dtype=np.uint8))
        if palette:
            img = img.convert("P")
            img.putdata(np.asarray(array, dtype=np.uint8).ravel())
        img.save(path)
        return path

    return _make


@pytest.fixture
def make_geojson(tmp_path):
    """Write a CellViT++-style cell_detection.geojson (one MultiPoint per class)."""

    def _make(rel_path, classes, nested=True):
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "MultiPoint", "coordinates": [list(map(float, p)) for p in pts]},
                "properties": {"objectType": "annotation", "classification": {"name": name}},
            }
            for name, pts in classes.items()
        ]
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([features] if nested else features), encoding="utf-8")
        return path

    return _make
