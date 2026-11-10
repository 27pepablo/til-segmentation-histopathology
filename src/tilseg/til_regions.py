"""Turn cell-level TIL detections into region-level TIL masks.

CellViT++ answers "where is each lymphocyte?", while the segmentation network
needs "which area is infiltrated?". This module bridges the two:

1. **Density filter** - for every TIL centroid, count the TILs inside a circle
   of radius ``R`` and divide by the circle area. Isolated lymphocytes fall
   below the threshold and are discarded.
2. **Rasterise** - stamp a small disc on the mask at every remaining centroid.
3. **Morphology** - dilate with a large elliptical kernel so neighbouring cells
   merge into one region, then apply a morphological closing to fill holes and
   smooth the contour.

An alternative that was explored during the thesis (DBSCAN clustering followed
by the convex hull of each cluster) is available as :func:`hull_region_mask`.

Default parameters are the ones used in the thesis for 0.25 um/px tiles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree


@dataclass(frozen=True)
class RegionParams:
    """Parameters of the centroid -> region conversion (pixel units)."""

    radius: float = 100.0  #: neighbourhood radius R for the local density
    min_density: float = 0.0003  #: minimum TILs / px^2 inside the neighbourhood
    seed_radius: int = 2  #: radius of the disc stamped on every dense TIL
    dilate_kernel: int = 100  #: diameter of the elliptical dilation kernel
    close_kernel: int = 120  #: diameter of the elliptical closing kernel

    @property
    def min_neighbours(self) -> float:
        """Density threshold expressed as a cell count inside the neighbourhood."""
        return self.min_density * math.pi * self.radius**2


def local_density(points: np.ndarray, radius: float) -> np.ndarray:
    """TILs per px^2 around each point (the point itself is included in the count)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros(0)
    counts = KDTree(points).query_radius(points, r=radius, count_only=True)
    return counts / (math.pi * radius**2)


def dense_points(points: np.ndarray, radius: float, min_density: float) -> np.ndarray:
    """Subset of ``points`` whose local density reaches ``min_density``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points[local_density(points, radius) >= min_density]


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def region_mask(
    points: np.ndarray, shape: tuple[int, int], params: RegionParams | None = None
) -> np.ndarray:
    """Binary TIL-region mask from TIL centroids.

    Args:
        points: ``(N, 2)`` array of ``(x, y)`` centroids in pixels.
        shape: ``(height, width)`` of the image the centroids refer to.
        params: conversion parameters; thesis defaults when omitted.

    Returns:
        ``uint8`` array of ``shape`` with 1 inside TIL regions and 0 elsewhere.
    """
    params = params or RegionParams()
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    dense = dense_points(points, params.radius, params.min_density)
    for x, y in np.rint(dense).astype(int):
        if 0 <= y < height and 0 <= x < width:
            cv2.circle(mask, (int(x), int(y)), params.seed_radius, 1, thickness=-1)

    if mask.any():
        mask = cv2.dilate(mask, _ellipse(params.dilate_kernel), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(params.close_kernel))
    return mask


def hull_region_mask(
    points: np.ndarray,
    shape: tuple[int, int],
    radius: float = 100.0,
    min_density: float = 0.0003,
    min_cluster_size: int = 10,
) -> np.ndarray:
    """Alternative region definition: DBSCAN clusters filled by their convex hull.

    Dense TILs are grouped with DBSCAN (``eps = radius``) and every cluster is
    replaced by its convex hull. Hulls over-cover concave infiltrates, which is
    why the morphological variant in :func:`region_mask` was preferred.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    dense = dense_points(points, radius, min_density)
    if len(dense) < max(min_cluster_size, 3):
        return mask

    labels = DBSCAN(eps=radius, min_samples=min_cluster_size).fit(dense).labels_
    for label in set(labels) - {-1}:
        cluster = np.rint(dense[labels == label]).astype(np.int32)
        if len(cluster) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(cluster), 1)
    return mask


def til_area_fraction(mask: np.ndarray) -> float:
    """Fraction of the tile covered by TIL regions (0-1)."""
    return float(np.count_nonzero(mask)) / mask.size if mask.size else 0.0
