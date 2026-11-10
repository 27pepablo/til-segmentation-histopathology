"""End-to-end check of the CellViT -> region-mask -> evaluation path through the CLI."""

import numpy as np
from PIL import Image

from tilseg import cli


def test_til_masks_then_evaluate(make_png, make_geojson, tmp_path, rng, capsys):
    # CellViT++ truncates the slide name at the first dot: "..._MPP-0.2500.tif" -> "..._MPP-0"
    make_png("tiles/TCGA-XX_MPP-0.2500_0000.png", np.full((800, 900, 3), 200))
    cells = rng.normal((450, 400), 40, size=(200, 2))
    make_geojson("cellvit/TCGA-XX_MPP-0_cell_detection.geojson", {"Inflammatory": cells})

    cli.main(
        ["til-masks", "--geojson", str(tmp_path / "cellvit"), "--images", str(tmp_path / "tiles"),
         "--out", str(tmp_path / "masks")]
    )

    out = tmp_path / "masks" / "TCGA-XX_MPP-0.2500.png"  # nnU-Net label name of that tile
    mask = np.array(Image.open(out))
    assert mask.shape == (800, 900) and set(np.unique(mask)) == {0, 1}
    assert mask[400, 450] == 1

    cli.main(["evaluate", "--pred", str(tmp_path / "masks"), "--ref", str(tmp_path / "masks"),
              "--out", str(tmp_path / "metrics.csv")])
    assert "dice: 1.000" in capsys.readouterr().out
    assert (tmp_path / "metrics.json").exists()


def test_binarize_and_build_dataset(make_png, tmp_path, capsys):
    for i in range(5):
        make_png(f"img/c{i}.png", np.zeros((8, 8, 3)))
        make_png(f"raw_masks/c{i}.png", np.full((8, 8), 3 if i % 2 else 1))

    cli.main(["binarize", "--dataset", "crowdsourcing", "--src", str(tmp_path / "raw_masks"),
              "--dst", str(tmp_path / "bin")])
    assert "without TIL pixels: 3" in capsys.readouterr().out

    cli.main(["build-dataset", "--images", str(tmp_path / "img"), "--labels", str(tmp_path / "bin"),
              "--nnunet-raw", str(tmp_path / "nnUNet_raw"), "--id", "111", "--name", "TILsCrowd"])
    assert "4 training / 1 test" in capsys.readouterr().out
