import json

import numpy as np
import pytest
from PIL import Image

from tilseg import nnunet_dataset as nd


@pytest.fixture
def source(make_png, tmp_path):
    for i in range(10):
        make_png(f"images/case{i:02d}.png", np.full((4, 4, 3), i))
        make_png(f"labels/case{i:02d}.png", np.zeros((4, 4)))
    return tmp_path / "images", tmp_path / "labels"


def test_case_id_strips_channel_suffix():
    assert nd.case_id("a/TCGA-A1_MPP-0.2500_0000.png") == "TCGA-A1_MPP-0.2500"
    assert nd.case_id("TCGA-A1_MPP-0.2500.png") == "TCGA-A1_MPP-0.2500"


def test_split_is_reproducible_and_disjoint():
    cases = [f"c{i}" for i in range(151)]
    train, test = nd.split_cases(cases, 0.2, seed=42)
    assert (len(train), len(test)) == (121, 30)  # sizes reported in the thesis
    assert not set(train) & set(test)
    assert nd.split_cases(cases, 0.2, seed=42) == (train, test)
    assert nd.split_cases(cases, 0.2, seed=7) != (train, test)


def test_build_dataset_layout(source, tmp_path):
    images, labels = source
    root = nd.build_dataset(images, labels, tmp_path / "raw", 111, "TILsCrowd")

    assert root.name == "Dataset111_TILsCrowd"
    train_imgs = sorted(p.name for p in (root / "imagesTr").iterdir())
    train_lbls = sorted(p.name for p in (root / "labelsTr").iterdir())
    assert len(train_imgs) == 8 and len(list((root / "imagesTs").iterdir())) == 2
    assert all(n.endswith("_0000.png") for n in train_imgs)
    assert [n.replace("_0000", "") for n in train_imgs] == train_lbls

    meta = json.loads((root / "dataset.json").read_text())
    assert meta["numTraining"] == 8
    assert meta["labels"] == {"background": 0, "til": 1}
    assert meta["file_ending"] == ".png"
    assert len(list(images.iterdir())) == 10  # sources are copied, not moved


def test_pair_cases_reports_unmatched_files(source, make_png):
    images, labels = source
    make_png("images/orphan.png", np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="orphan.png"):
        nd.pair_cases(images, labels)


def test_replace_labels_keeps_split(source, make_png, tmp_path):
    images, labels = source
    root = nd.build_dataset(images, labels, tmp_path / "raw", 444, "TILsCellViT")
    before = sorted(p.name for p in (root / "labelsTr").iterdir())

    for i in range(9):  # case09 has no CellViT mask -> keeps its original label
        make_png(f"new/case{i:02d}.png", np.ones((4, 4)))
    counts = nd.replace_labels(root, tmp_path / "new")

    assert counts == {"replaced": 9, "kept": 1}
    assert sorted(p.name for p in (root / "labelsTr").iterdir()) == before
    replaced = [p for sub in ("labelsTr", "labelsTs") for p in (root / sub).iterdir()]
    values = {p.stem: int(np.array(Image.open(p)).max()) for p in replaced}
    assert values.pop("case09") == 0
    assert set(values.values()) == {1}
