import numpy as np
from PIL import Image

from tilseg import masks


def test_binarize_keeps_only_til_labels():
    label_map = np.array([[0, 1, 3], [3, 4, 10]])
    out = masks.binarize(label_map, (3,))
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_binarize_accepts_several_labels():
    label_map = np.array([[3, 10, 11, 2]])
    assert masks.binarize(label_map, (3, 10, 11)).tolist() == [[1, 1, 1, 0]]


def test_is_binary():
    assert masks.is_binary(np.array([[0, 1], [1, 0]]))
    assert masks.is_binary(np.zeros((2, 2)))
    assert not masks.is_binary(np.array([[0, 255]]))


def test_palette_mask_indices_are_preserved(make_png):
    # BCSS-WSSS masks are palette PNGs: converting to RGB/L would change the values.
    path = make_png("m.png", [[0, 1, 2], [2, 3, 4]], palette=True)
    assert Image.open(path).mode == "P"
    assert masks.read_label_map(path).tolist() == [[0, 1, 2], [2, 3, 4]]


def test_convert_folder_skips_empty_masks(make_png, tmp_path):
    make_png("src/with_til.png", [[0, 2], [2, 1]], palette=True)
    make_png("src/no_til.png", [[0, 1], [3, 4]], palette=True)

    report = masks.convert_folder(
        tmp_path / "src", tmp_path / "dst", masks.BCSS_WSSS_TIL_LABELS, skip_empty=True
    )

    assert report == {"written": ["with_til.png"], "empty": ["no_til.png"]}
    out = np.array(Image.open(tmp_path / "dst" / "with_til.png"))
    assert out.tolist() == [[0, 1], [1, 0]]
    assert not (tmp_path / "dst" / "no_til.png").exists()


def test_convert_folder_keeps_empty_masks_by_default(make_png, tmp_path):
    make_png("src/no_til.png", [[0, 1], [1, 2]])
    report = masks.convert_folder(tmp_path / "src", tmp_path / "dst", masks.CROWDSOURCING_TIL_LABELS)
    assert report["written"] == ["no_til.png"]
    assert not np.array(Image.open(tmp_path / "dst" / "no_til.png")).any()
