import math

import numpy as np
import pytest

from tilseg import til_regions as tr


def cluster(rng, center, n, spread):
    return rng.normal(center, spread, size=(n, 2))


def test_local_density_counts_neighbours_including_self():
    pts = np.array([[0, 0], [3, 4], [100, 100]], dtype=float)
    dens = tr.local_density(pts, radius=5)
    area = math.pi * 25
    assert dens == pytest.approx([2 / area, 2 / area, 1 / area])


def test_local_density_empty():
    assert tr.local_density(np.zeros((0, 2)), 100).shape == (0,)


def test_min_neighbours_matches_thesis_threshold():
    # 0.0003 TILs/px^2 in a 100 px radius ~= 9.4 cells
    assert tr.RegionParams().min_neighbours == pytest.approx(9.42, abs=0.01)


def test_dense_cluster_becomes_one_region_and_sparse_cells_vanish(rng):
    dense = cluster(rng, (400, 300), 150, 40)
    sparse = rng.uniform(0, 1000, size=(8, 2)) + [0, 600]  # scattered, far from the cluster
    sparse = sparse[np.linalg.norm(sparse - [400, 300], axis=1) > 350]
    mask = tr.region_mask(np.vstack([dense, sparse]), (1600, 1000))

    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 1}
    assert mask[300, 400] == 1  # (x=400, y=300) -> row 300, col 400
    for x, y in np.rint(sparse).astype(int):
        assert mask[min(y, 1599), min(x, 999)] == 0
    n_regions = cv2_count(mask)
    assert n_regions == 1


def test_region_covers_the_dense_cells(rng):
    pts = cluster(rng, (500, 500), 300, 60)
    mask = tr.region_mask(pts, (1000, 1000))
    dense = np.rint(tr.dense_points(pts, 100, 0.0003)).astype(int)
    assert all(mask[y, x] == 1 for x, y in dense)


def test_two_distant_clusters_stay_separate(rng):
    pts = np.vstack([cluster(rng, (300, 300), 120, 30), cluster(rng, (1300, 1300), 120, 30)])
    assert cv2_count(tr.region_mask(pts, (1600, 1600))) == 2


def test_empty_or_sparse_input_gives_empty_mask(rng):
    assert not tr.region_mask(np.zeros((0, 2)), (200, 200)).any()
    lonely = np.array([[50.0, 50.0], [900.0, 900.0]])
    assert not tr.region_mask(lonely, (1000, 1000)).any()


def test_points_outside_the_image_are_ignored(rng):
    pts = cluster(rng, (5000, 5000), 100, 20)
    assert not tr.region_mask(pts, (500, 500)).any()


def test_hull_variant_is_filled_and_convex(rng):
    pts = cluster(rng, (500, 500), 200, 50)
    mask = tr.hull_region_mask(pts, (1000, 1000))
    assert mask[500, 500] == 1 and cv2_count(mask) == 1
    assert not tr.hull_region_mask(pts[:2], (1000, 1000)).any()


def test_til_area_fraction():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5] = 1
    assert tr.til_area_fraction(mask) == 0.5


def cv2_count(mask):
    import cv2

    n, _ = cv2.connectedComponents(mask)
    return n - 1
