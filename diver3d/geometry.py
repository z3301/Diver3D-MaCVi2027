"""Rotation / box helpers shared by the data pipeline, model and decoder.

Conventions (see FORMAT.md): sonar frame x forward, y left, z up.
R = [ex ey ez] maps body -> sonar. dims = (length, width, height) along (ex, ey, ez).
Quaternions are Hamilton (w, x, y, z).
"""
from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-8

CORNER_TEMPLATE = np.array(
    [[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
     [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]], dtype=np.float64)


def quat_to_rotmat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(R) -> np.ndarray:
    """3x3 -> (w, x, y, z) with w >= 0."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def box_corners(center, dims, R) -> np.ndarray:
    return (CORNER_TEMPLATE * np.asarray(dims)) @ np.asarray(R).T + np.asarray(center)


def points_in_box_mask(points_xyz: np.ndarray, center, dims, R, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside the oriented box (optionally grown by `margin` metres per side)."""
    local = (points_xyz - np.asarray(center)) @ np.asarray(R)  # (N,3) body coords
    half = np.asarray(dims) / 2.0 + margin
    return np.all(np.abs(local) <= half, axis=1)


def mirror_y(points: np.ndarray, centers: np.ndarray, Rs: np.ndarray):
    """Reflect the scene across the x-z plane (y -> -y).

    Body axes: ex -> F ex, ez -> F ez, and the shoulder axis flips sign so the
    frame stays right-handed: R' = F R diag(1,-1,1).
    """
    F = np.diag([1.0, -1.0, 1.0])
    points = points.copy()
    points[:, 1] *= -1
    if len(centers):
        centers = centers * np.array([1.0, -1.0, 1.0])
        Rs = F @ Rs @ F  # F R diag(1,-1,1) and diag(1,-1,1) == F
    return points, centers, Rs


def transform_scene(points: np.ndarray, centers: np.ndarray, Rs: np.ndarray, M: np.ndarray):
    """Apply a global rotation M (about the sonar origin) to points and boxes."""
    points = points.copy()
    points[:, :3] = points[:, :3] @ M.T
    if len(centers):
        centers = centers @ M.T
        Rs = M[None] @ Rs
    return points, centers, Rs


# ----------------------------------------------------------------------------- torch
def gram_schmidt_6d_torch(a: torch.Tensor) -> torch.Tensor:
    """(N, 6) -> (N, 3, 3). First three numbers -> ex, next three -> ey (projected)."""
    a1, a2 = a[:, :3], a[:, 3:6]
    b1 = a1 / (a1.norm(dim=1, keepdim=True) + _EPS)
    v = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = v / (v.norm(dim=1, keepdim=True) + _EPS)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two columns."""
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)
