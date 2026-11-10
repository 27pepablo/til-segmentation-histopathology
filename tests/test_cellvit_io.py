import csv

from tilseg import cellvit_io


def test_load_centroids_selects_inflammatory(make_geojson):
    path = make_geojson(
        "s_cell_detection.geojson",
        {"Neoplastic": [(1, 2)], "Inflammatory": [(10, 20), (30, 40)], "Connective": [(5, 5)]},
    )
    pts = cellvit_io.load_centroids(path)
    assert pts.shape == (2, 2)
    assert pts.tolist() == [[10.0, 20.0], [30.0, 40.0]]  # (x, y) order is preserved


def test_flat_and_nested_roots_are_equivalent(make_geojson):
    classes = {"Inflammatory": [(1, 1), (2, 2)]}
    nested = make_geojson("a_cell_detection.geojson", classes, nested=True)
    flat = make_geojson("b_cell_detection.geojson", classes, nested=False)
    assert cellvit_io.load_centroids(nested).tolist() == cellvit_io.load_centroids(flat).tolist()


def test_missing_class_gives_empty_array(make_geojson):
    path = make_geojson("s_cell_detection.geojson", {"Neoplastic": [(1, 2)]})
    assert cellvit_io.load_centroids(path).shape == (0, 2)


def test_class_counts(make_geojson):
    path = make_geojson(
        "s_cell_detection.geojson", {"Neoplastic": [(1, 2)] * 3, "Inflammatory": [(0, 0)] * 5}
    )
    assert cellvit_io.class_counts(path) == {"Neoplastic": 3, "Inflammatory": 5}


def test_write_filelist(tmp_path):
    (tmp_path / "tiff").mkdir()
    for name in ("b.tif", "a.tiff", "ignore.png"):
        (tmp_path / "tiff" / name).touch()

    n = cellvit_io.write_filelist(tmp_path / "tiff", tmp_path / "out" / "filelist.csv")

    assert n == 2
    with open(tmp_path / "out" / "filelist.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["path"].replace("\\", "/").rsplit("/", 1)[-1] for r in rows] == ["a.tiff", "b.tif"]
    assert rows[0]["slide_mpp"] == "0.25" and rows[0]["magnification"] == "40"
