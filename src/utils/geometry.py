from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import open3d as o3d

LOGGER = logging.getLogger(__name__)


def load_geometry(path: Path) -> o3d.geometry.Geometry:
    suffix = path.suffix.lower()
    if suffix in {".obj", ".stl", ".off"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise ValueError("triangle mesh is empty")
        mesh.compute_vertex_normals()
        return mesh

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.is_empty() and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        return mesh

    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError("failed to read geometry as mesh/point cloud")
    return pcd


def get_points(geom: o3d.geometry.Geometry) -> np.ndarray:
    if isinstance(geom, o3d.geometry.TriangleMesh):
        pts = np.asarray(geom.vertices)
    elif isinstance(geom, o3d.geometry.PointCloud):
        pts = np.asarray(geom.points)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    if pts.size == 0:
        raise ValueError("geometry has no points")
    return pts


def normalize_geometry(geom: o3d.geometry.Geometry, target_extent: float = 1.0) -> o3d.geometry.Geometry:
    points = get_points(geom)
    min_b = points.min(axis=0)
    max_b = points.max(axis=0)
    center = (min_b + max_b) / 2.0
    extent = float(np.max(max_b - min_b))
    if extent <= 0:
        raise ValueError("invalid geometry extent")

    geom = geom.translate(-center)
    scale = target_extent / extent
    geom = geom.scale(scale, center=(0.0, 0.0, 0.0))
    return geom


def fibonacci_sphere_points(n_views: int, radius: float = 2.0) -> list[np.ndarray]:
    if n_views < 1:
        raise ValueError("n_views must be >= 1")

    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_views):
        y = 1 - (i / float(max(n_views - 1, 1))) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append(np.array([x, y, z], dtype=np.float32) * radius)
    return points
