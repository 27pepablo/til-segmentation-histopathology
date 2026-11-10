"""Build an nnU-Net v2 raw dataset from a folder of images and a folder of masks.

nnU-Net v2 expects::

    nnUNet_raw/DatasetXXX_NAME/
        imagesTr/<case>_0000.png      training images (channel suffix _0000)
        labelsTr/<case>.png           training labels (0 = background, 1 = TIL)
        imagesTs/<case>_0000.png      held-out test images
        labelsTs/<case>.png           held-out test labels (not used by nnU-Net)
        dataset.json

For RGB PNGs nnU-Net reads the three colour channels from the single ``_0000``
file through its ``NaturalImage2DIO`` reader.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

CHANNEL_SUFFIX = "_0000"
FILE_ENDING = ".png"


def pair_cases(images_dir: str | Path, labels_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match every image with the mask that has the same file stem.

    Images may or may not already carry the ``_0000`` channel suffix.

    Raises:
        ValueError: if an image has no mask or a mask has no image.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    labels = {p.stem: p for p in labels_dir.glob(f"*{FILE_ENDING}")}

    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    missing: list[str] = []
    for img in sorted(images_dir.glob(f"*{FILE_ENDING}")):
        case = case_id(img)
        if case in labels:
            pairs.append((img, labels[case]))
            seen.add(case)
        else:
            missing.append(img.name)

    orphans = sorted(set(labels) - seen)
    if missing or orphans:
        raise ValueError(
            f"Images without mask: {missing[:5]} ({len(missing)} total); "
            f"masks without image: {orphans[:5]} ({len(orphans)} total)"
        )
    return pairs


def case_id(image_path: str | Path) -> str:
    """Case identifier of an image file: its stem without the ``_0000`` suffix."""
    stem = Path(image_path).stem
    return stem[: -len(CHANNEL_SUFFIX)] if stem.endswith(CHANNEL_SUFFIX) else stem


def split_cases(
    cases: list, test_fraction: float = 0.2, seed: int = 42
) -> tuple[list, list]:
    """Random train/test split (no stratification: no clinical metadata available).

    The number of test cases is ``int(len(cases) * test_fraction)``, matching the
    split used in the thesis (151 images -> 121 train / 30 test).
    """
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    n_test = int(len(shuffled) * test_fraction)
    return sorted(shuffled[n_test:]), sorted(shuffled[:n_test])


def dataset_json(num_training: int, name: str, description: str = "") -> dict:
    """Content of ``dataset.json`` for RGB H&E tiles with a binary TIL label."""
    return {
        "name": name,
        "description": description,
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "til": 1},
        "numTraining": num_training,
        "file_ending": FILE_ENDING,
    }


def build_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    nnunet_raw: str | Path,
    dataset_id: int,
    dataset_name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    description: str = "TIL-region segmentation in H&E breast cancer tiles",
) -> Path:
    """Copy images and masks into the nnU-Net raw layout and write ``dataset.json``.

    Source files are copied, never moved. Returns the dataset folder.
    """
    if not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be in 1..999")

    pairs = pair_cases(images_dir, labels_dir)
    train, test = split_cases(pairs, test_fraction, seed)

    root = Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}"
    for subset, img_sub, lbl_sub in ((train, "imagesTr", "labelsTr"), (test, "imagesTs", "labelsTs")):
        (root / img_sub).mkdir(parents=True, exist_ok=True)
        (root / lbl_sub).mkdir(parents=True, exist_ok=True)
        for img, lbl in subset:
            case = case_id(img)
            shutil.copyfile(img, root / img_sub / f"{case}{CHANNEL_SUFFIX}{FILE_ENDING}")
            shutil.copyfile(lbl, root / lbl_sub / f"{case}{FILE_ENDING}")

    meta = dataset_json(len(train), root.name, description)
    (root / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def replace_labels(dataset_dir: str | Path, new_labels_dir: str | Path) -> dict[str, int]:
    """Swap the labels of an existing dataset for masks with the same case names.

    Used for Experiment 2: the images and the train/test split stay identical and
    only the label definition changes (original annotation -> CellViT-derived).
    Cases without a new mask keep their current label and are counted as ``kept``.
    """
    dataset_dir, new_labels_dir = Path(dataset_dir), Path(new_labels_dir)
    new = {p.stem: p for p in new_labels_dir.glob(f"*{FILE_ENDING}")}

    counts = {"replaced": 0, "kept": 0}
    for sub in ("labelsTr", "labelsTs"):
        for old in sorted((dataset_dir / sub).glob(f"*{FILE_ENDING}")):
            if old.stem in new:
                shutil.copyfile(new[old.stem], old)
                counts["replaced"] += 1
            else:
                counts["kept"] += 1
    return counts
