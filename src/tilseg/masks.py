"""Convert the multi-class annotations of the public datasets into binary TIL masks.

Both datasets ship one PNG mask per image in which every pixel stores a class
index. nnU-Net expects label maps with consecutive integers, so the TIL class is
mapped to ``1`` and everything else to ``0``.

* **CrowdsourcingDataset-Amgadetal2019** (BCSS): grayscale masks, class codes
  listed in the dataset's ``gtruth_codes.tsv``. ``3`` is *lymphocytic infiltrate*.
* **BCSS-WSSS**: palette ("P" mode) masks with indices
  ``0 = tumour, 1 = stroma, 2 = lymphocyte (TIL), 3 = necrosis, 4 = excluded``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

#: Class code of "lymphocytic infiltrate" in CrowdsourcingDataset-Amgadetal2019.
CROWDSOURCING_TIL_LABELS: tuple[int, ...] = (3,)
#: Palette index of the lymphocyte class in BCSS-WSSS.
BCSS_WSSS_TIL_LABELS: tuple[int, ...] = (2,)


def binarize(label_map: np.ndarray, til_labels: Iterable[int]) -> np.ndarray:
    """Return a ``uint8`` mask with 1 where ``label_map`` holds any TIL label."""
    return np.isin(label_map, list(til_labels)).astype(np.uint8)


def read_label_map(path: str | Path) -> np.ndarray:
    """Read a mask PNG as a 2-D array of class indices.

    Palette images are read *without* RGB conversion so the stored indices are
    preserved; any other mode is reduced to a single 8-bit channel.
    """
    with Image.open(path) as img:
        if img.mode not in ("P", "L", "I;16", "I"):
            img = img.convert("L")
        return np.array(img)


def is_binary(mask: np.ndarray) -> bool:
    """True if the mask only contains the values 0 and/or 1."""
    return bool(np.isin(np.unique(mask), (0, 1)).all())


def convert_folder(
    src_dir: str | Path,
    dst_dir: str | Path,
    til_labels: Iterable[int],
    skip_empty: bool = False,
) -> dict[str, list[str]]:
    """Binarise every PNG mask in ``src_dir`` and write it to ``dst_dir``.

    Args:
        src_dir: folder with the original multi-class masks.
        dst_dir: output folder (created if needed). File names are preserved.
        til_labels: class indices that count as TIL.
        skip_empty: do not write masks without a single TIL pixel. Used for
            BCSS-WSSS, where most 224x224 patches contain no lymphocyte region.

    Returns:
        ``{"written": [...], "empty": [...]}`` with the file names in each group
        (``empty`` lists the masks without TIL pixels, whether written or not).
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    til_labels = tuple(til_labels)

    report: dict[str, list[str]] = {"written": [], "empty": []}
    for path in sorted(src_dir.glob("*.png")):
        mask = binarize(read_label_map(path), til_labels)
        if not mask.any():
            report["empty"].append(path.name)
            if skip_empty:
                continue
        Image.fromarray(mask).save(dst_dir / path.name)
        report["written"].append(path.name)
    return report
