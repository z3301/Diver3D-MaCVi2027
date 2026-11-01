"""CenterPoint-style training targets on the BEV output grid."""
from __future__ import annotations

import numpy as np

from .geometry import box_corners, rotmat_to_6d

REG_DIM = 12  # offx, offy, z, log l, log w, log h, 6D rotation


def world_to_grid(xyz: np.ndarray, grid: dict) -> np.ndarray:
    """(N, 3) sonar-frame points -> BEV grid coordinates; identity unless grid.polar_deg_per_unit > 0,
    in which case u = range [m], v = bearing [deg] / polar_deg_per_unit, z unchanged."""
    k = float(grid.get("polar_deg_per_unit", 0.0) or 0.0)
    if k <= 0:
        return xyz
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    th = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) / k
    return np.stack([r, th, xyz[:, 2]], 1)


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet radius so that any box shifted by r still overlaps the GT at >= min_overlap."""
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center_xy, radius: int):
    d = 2 * radius + 1
    sigma = d / 6
    m, n = radius, radius
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    cx, cy = int(center_xy[0]), int(center_xy[1])
    H, W = heatmap.shape
    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    hm_patch = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    g_patch = g[radius - top:radius + bottom, radius - left:radius + right]
    if hm_patch.size and g_patch.size:
        np.maximum(hm_patch, g_patch, out=hm_patch)


def zref_map(pts: np.ndarray, grid: dict, out_stride: int) -> np.ndarray:
    """(Ho, Wo) mean z of current-frame returns in the 3x3 output-cell neighbourhood of each cell;
    NaN where there are no returns. Points carry a time channel in column 4 when frames are
    stacked; only t == 0 counts. Same definition as SonarVoxNet.zref_map (torch)."""
    sx, sy = grid["vx"] * out_stride, grid["vy"] * out_stride
    Wo = int(round((grid["xmax"] - grid["xmin"]) / sx))
    Ho = int(round((grid["ymax"] - grid["ymin"]) / sy))
    if pts.shape[1] > 4:
        pts = pts[pts[:, 4] == 0]
    g = world_to_grid(pts[:, :3], grid)
    xi = np.floor((g[:, 0] - grid["xmin"]) / sx).astype(np.int64)
    yi = np.floor((g[:, 1] - grid["ymin"]) / sy).astype(np.int64)
    ok = (xi >= 0) & (xi < Wo) & (yi >= 0) & (yi < Ho)
    ssum = np.zeros((Ho, Wo)); cnt = np.zeros((Ho, Wo))
    np.add.at(ssum, (yi[ok], xi[ok]), pts[ok, 2]); np.add.at(cnt, (yi[ok], xi[ok]), 1.0)
    pad_s = np.pad(ssum, 1); pad_c = np.pad(cnt, 1)
    s3 = sum(pad_s[dy:dy + Ho, dx:dx + Wo] for dy in range(3) for dx in range(3))
    c3 = sum(pad_c[dy:dy + Ho, dx:dx + Wo] for dy in range(3) for dx in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c3 > 0, s3 / np.maximum(c3, 1), np.nan)


class TargetBuilder:
    def __init__(self, grid: dict, out_stride: int, max_objs: int = 16,
                 min_overlap: float = 0.7, min_radius: int = 1, z_relative: bool = False):
        self.z_relative = z_relative
        self.out_stride = out_stride
        self.xmin, self.ymin = grid["xmin"], grid["ymin"]
        self.xmax, self.ymax = grid["xmax"], grid["ymax"]
        self.zmin, self.zmax = grid["zmin"], grid["zmax"]
        self.sx = grid["vx"] * out_stride
        self.sy = grid["vy"] * out_stride
        self.Wo = int(round((grid["xmax"] - grid["xmin"]) / self.sx))
        self.Ho = int(round((grid["ymax"] - grid["ymin"]) / self.sy))
        self.max_objs = max_objs
        self.min_overlap = min_overlap
        self.min_radius = min_radius
        self.grid = grid

    def __call__(self, centers: np.ndarray, dims: np.ndarray, Rs: np.ndarray, pts: np.ndarray | None = None):
        zmap = zref_map(pts, self.grid, self.out_stride) if (self.z_relative and pts is not None) else None
        hm = np.zeros((1, self.Ho, self.Wo), dtype=np.float32)
        reg = np.zeros((self.max_objs, REG_DIM), dtype=np.float32)
        inds = np.zeros(self.max_objs, dtype=np.int64)
        mask = np.zeros(self.max_objs, dtype=np.float32)
        k = 0
        for c, d, R in zip(centers, dims, Rs):
            if k >= self.max_objs:
                break
            cg = world_to_grid(np.asarray(c, dtype=np.float64)[None], self.grid)[0]
            if not (self.xmin <= cg[0] < self.xmax and self.ymin <= cg[1] < self.ymax
                    and self.zmin <= cg[2] < self.zmax):
                continue
            cx = (cg[0] - self.xmin) / self.sx
            cy = (cg[1] - self.ymin) / self.sy
            ci, cj = int(cx), int(cy)
            corners = world_to_grid(box_corners(c, d, R), self.grid)
            ext = corners.max(0) - corners.min(0)          # BEV footprint extent in grid units
            r = gaussian_radius(ext[1] / self.sy, ext[0] / self.sx, self.min_overlap)
            r = max(self.min_radius, int(r))
            draw_gaussian(hm[0], (ci, cj), r)
            reg[k, 0] = cx - ci
            reg[k, 1] = cy - cj
            zref = 0.0
            if zmap is not None and not np.isnan(zmap[cj, ci]):
                zref = float(zmap[cj, ci])
            reg[k, 2] = c[2] - zref
            reg[k, 3:6] = np.log(d)
            reg[k, 6:12] = rotmat_to_6d(R)
            inds[k] = cj * self.Wo + ci
            mask[k] = 1.0
            k += 1
        return {"hm": hm, "reg": reg, "inds": inds, "mask": mask}
