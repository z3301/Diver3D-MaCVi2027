"""Dataset, augmentation (GT sampling with annealed count, flips, small global rotations)
and collate. Point clouds are preloaded into RAM (the whole dataset is ~3.5 GB)."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import (mirror_y, points_in_box_mask, quat_to_rotmat, rot_x, rot_y, rot_z,
                       transform_scene)
from .targets import TargetBuilder, world_to_grid


def load_points(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


class Frames:
    """Tokens + preloaded points (+ GT) for one split."""

    def __init__(self, root: str, split: str, limit: int | None = None, preload: bool = True,
                 exclude_scenes=None, only_scenes=None):
        self.root, self.split = root, split
        gt_path = os.path.join(root, "annotations", f"gt_{split}.json")
        if os.path.exists(gt_path):
            gt = json.load(open(gt_path))["results"]
            self.tokens = sorted(gt)
            self.gt = {}
            for t in self.tokens:
                recs = gt[t]
                self.gt[t] = (
                    np.array([r["center"] for r in recs], dtype=np.float64).reshape(-1, 3),
                    np.array([r["dimensions"] for r in recs], dtype=np.float64).reshape(-1, 3),
                    np.array([quat_to_rotmat(r["quaternion"]) for r in recs], dtype=np.float64).reshape(-1, 3, 3),
                    np.array([r.get("instance_id", -1) for r in recs], dtype=np.int64),
                )
        else:
            self.gt = None
            with open(os.path.join(root, "annotations", "test_tokens.txt")) as fh:
                self.tokens = [ln.strip() for ln in fh if ln.strip()]
        if exclude_scenes:
            self.tokens = [t for t in self.tokens if t.split("/")[0] not in set(exclude_scenes)]
        if only_scenes:
            self.tokens = [t for t in self.tokens if t.split("/")[0] in set(only_scenes)]
        if self.gt is not None:
            self.gt = {t: self.gt[t] for t in self.tokens}
        if limit:
            self.tokens = self.tokens[:limit]
        self.points: list[np.ndarray] | None = None
        if preload:
            with ThreadPoolExecutor(32) as ex:
                self.points = list(ex.map(lambda t: load_points(self.path(t)), self.tokens))

    def path(self, token: str) -> str:
        scene, frame = token.split("/")
        return os.path.join(self.root, "scenes", scene, "sonar", f"frame_{frame}.bin")

    def __len__(self):
        return len(self.tokens)

    def get_points(self, i: int) -> np.ndarray:
        if self.points is not None:
            return self.points[i]
        return load_points(self.path(self.tokens[i]))

    def get_stack(self, i: int, n_prev: int) -> np.ndarray:
        """Current frame plus up to n_prev previous frames of the same scene, with a time
        channel (0 for current, -k for k frames back). Returns (N, 5) when n_prev > 0."""
        pts = self.get_points(i)
        if n_prev <= 0:
            return pts
        scene = self.tokens[i].split("/")[0]
        stack = [np.concatenate([pts, np.zeros((len(pts), 1), np.float32)], 1)]
        for k in range(1, n_prev + 1):
            j = i - k
            if j < 0 or not self.tokens[j].startswith(scene + "/"):
                break
            pk = self.get_points(j)
            stack.append(np.concatenate([pk, np.full((len(pk), 1), -float(k), np.float32)], 1))
        return np.concatenate(stack, 0)

    def get_boxes(self, i: int):
        if self.gt is None:
            e = np.zeros((0, 3))
            return e, e.copy(), np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64)
        return self.gt[self.tokens[i]]


BEAM = {"bear_min": -45.0, "bear_step": 0.353, "n_bear": 256, "elev_min": -20.0, "elev_step": 0.635, "n_elev": 64}


def beam_keys(xyz: np.ndarray, beam: dict = BEAM):
    """Integer beam id per point (bearing idx * n_elev + elev idx; -1 outside the FOV) and range."""
    r = np.linalg.norm(xyz, axis=1)
    bear = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    elev = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    bi = np.rint((bear - beam["bear_min"]) / beam["bear_step"]).astype(np.int64)
    ei = np.rint((elev - beam["elev_min"]) / beam["elev_step"]).astype(np.int64)
    ok = (bi >= 0) & (bi < beam["n_bear"]) & (ei >= 0) & (ei < beam["n_elev"])
    key = np.where(ok, bi * beam["n_elev"] + ei, -1)
    return key, r


def local_feats(pts: np.ndarray, w: int = 2, beam: dict = BEAM) -> np.ndarray:
    """Sonar-native local structure (GBlobs-style): for every return, the mean offset (3), covariance
    (6: xx, yy, zz, xy, xz, yz) and log count of the returns within +-w beams in bearing and elevation on
    the range image. Computed per frame (callers split stacked frames by their time channel)."""
    out = np.zeros((len(pts), 10), np.float32)
    if len(pts) == 0:
        return out
    key, _ = beam_keys(pts[:, :3], beam)
    ok = key >= 0
    if not ok.any():
        return out
    nb, ne = beam["n_bear"], beam["n_elev"]
    bi, ei = key[ok] // ne, key[ok] % ne
    xyz = pts[ok, :3].astype(np.float64)
    vals = np.c_[np.ones(len(xyz)), xyz, xyz ** 2, xyz[:, 0] * xyz[:, 1], xyz[:, 0] * xyz[:, 2], xyz[:, 1] * xyz[:, 2]]
    S = np.zeros((nb, ne, 10))
    np.add.at(S, (bi, ei), vals)
    Sp = np.pad(S, ((w, w), (w, w), (0, 0)))
    C = np.zeros((nb + 2 * w + 1, ne + 2 * w + 1, 10)); C[1:, 1:] = Sp.cumsum(0).cumsum(1)
    k = 2 * w + 1
    W = C[k:, k:] - C[:-k, k:] - C[k:, :-k] + C[:-k, :-k]            # (nb, ne, 10) window sums
    Wp = W[bi, ei]
    n = np.maximum(Wp[:, 0], 1.0)
    mean = Wp[:, 1:4] / n[:, None]
    cov = Wp[:, 4:10] / n[:, None] - np.c_[mean ** 2, mean[:, 0] * mean[:, 1], mean[:, 0] * mean[:, 2], mean[:, 1] * mean[:, 2]]
    out[ok, 0:3] = mean - xyz
    out[ok, 3:9] = cov
    out[ok, 9] = np.log1p(np.maximum(Wp[:, 0] - 1.0, 0.0))
    return out


def add_local_feats(pts: np.ndarray, w: int) -> np.ndarray:
    """Append the 10 local-structure channels; stacked frames (time channel at column 4) are handled per frame."""
    if pts.shape[1] >= 5:
        out = np.zeros((len(pts), 10), np.float32)
        for t in np.unique(pts[:, 4]):
            m = pts[:, 4] == t
            out[m] = local_feats(pts[m], w)
    else:
        out = local_feats(pts, w)
    return np.concatenate([pts.astype(np.float32), out], 1)


def occlude_merge(bg: np.ndarray, fg: np.ndarray, margin: float = 0.0):
    """One return per beam: where a background point and a pasted point share a beam, keep the
    nearer one (pasted diver shadows the beams behind it, and is itself hidden behind nearer
    background). Points carry a time channel when frames are stacked; beams are resolved per
    time slice so each stacked frame stays a valid range image."""
    if len(fg) == 0:
        return bg, fg
    kb, rb = beam_keys(bg[:, :3])
    kf, rf = beam_keys(fg[:, :3])
    tb = bg[:, 4].astype(np.int64) if bg.shape[1] > 4 else np.zeros(len(bg), np.int64)
    tf = fg[:, 4].astype(np.int64) if fg.shape[1] > 4 else np.zeros(len(fg), np.int64)
    nk = BEAM["n_bear"] * BEAM["n_elev"]
    kb2 = np.where(kb >= 0, kb + (tb + 8) * nk, -1)
    kf2 = np.where(kf >= 0, kf + (tf + 8) * nk, -1)
    # nearest pasted return per beam
    order = np.argsort(rf)
    first_f = {}
    for i in order:
        k = kf2[i]
        if k >= 0 and k not in first_f:
            first_f[k] = rf[i]
    keep_bg = np.ones(len(bg), bool)
    for i in range(len(bg)):
        k = kb2[i]
        if k >= 0 and k in first_f and rb[i] > first_f[k] + margin:
            keep_bg[i] = False
    # nearest background return per beam
    order = np.argsort(rb)
    first_b = {}
    for i in order:
        k = kb2[i]
        if k >= 0 and k not in first_b:
            first_b[k] = rb[i]
    keep_fg = kf >= 0                      # pasted returns outside the sensor wedge cannot exist
    for i in range(len(fg)):
        k = kf2[i]
        if k >= 0 and k in first_b and rf[i] > first_b[k] + margin:
            keep_fg[i] = False
    return bg[keep_bg], fg[keep_fg]


class GTSampler:
    """Bank of real diver returns cut from train frames, pasted into other frames."""

    def __init__(self, frames: Frames, cfg: dict, grid: dict, n_prev: int = 0):
        self.cfg = cfg
        self.grid = grid
        self.bank = []
        margin = cfg.get("crop_margin", 0.1)
        self.group_scenes = set(cfg.get("group_scenes", []) or [])   # same_group: paste only within/outside this set
        self.same_group = bool(cfg.get("same_group", False)) and bool(self.group_scenes)
        self.bank_group = []
        for i in range(len(frames)):
            pts = frames.get_stack(i, n_prev)
            centers, dims, Rs, _ = frames.get_boxes(i)
            in_grp = frames.tokens[i].split("/")[0] in self.group_scenes
            for c, d, R in zip(centers, dims, Rs):
                m = points_in_box_mask(pts[:, :3], c, d, R, margin)
                if m.sum() >= cfg.get("min_points", 20):
                    self.bank.append((pts[m].copy(), c.copy(), d.copy(), R.copy()))
                    self.bank_group.append(in_grp)
        self.bank_group = np.array(self.bank_group, dtype=bool)
        self.idx_in = np.nonzero(self.bank_group)[0]
        self.idx_out = np.nonzero(~self.bank_group)[0]

    def __len__(self):
        return len(self.bank)

    def sample(self, rng: np.random.Generator, points: np.ndarray, centers, dims, Rs, n_insert: int,
               scene: str | None = None):
        """Returns augmented (points, centers, dims, Rs) with up to n_insert pasted divers."""
        if n_insert <= 0 or not self.bank:
            return points, centers, dims, Rs
        pool = None
        if self.same_group and scene is not None:
            pool = self.idx_in if scene in self.group_scenes else self.idx_out
            if len(pool) == 0:
                pool = None
        cfg = self.cfg
        yaw_max = np.radians(cfg.get("bearing_max_deg", 40.0))
        el_jit = np.radians(cfg.get("elev_jitter_deg", 8.0))
        rs_lo, rs_hi = cfg.get("range_scale", [0.85, 1.15])
        new_pts, new_c, new_d, new_R = [], list(centers), list(dims), list(Rs)
        added = 0
        for _ in range(n_insert * 6):
            if added >= n_insert:
                break
            spts, sc, sd, sR = self.bank[int(pool[rng.integers(len(pool))]) if pool is not None else rng.integers(len(self.bank))]
            # keep the sample on a bearing inside the sonar fan, at a similar range
            theta0 = np.arctan2(sc[1], sc[0])
            theta1 = rng.uniform(-yaw_max, yaw_max)
            M = rot_z(theta1 - theta0)
            if el_jit > 0:
                M = rot_y(rng.uniform(-el_jit, el_jit)) @ M
            s = rng.uniform(rs_lo, rs_hi)
            c2 = M @ sc
            shift = (s - 1.0) * c2                      # slide along the ray, keep shape
            c2 = c2 + shift
            cg = world_to_grid(c2[None], self.grid)[0]
            if not (self.grid["xmin"] + 0.3 <= cg[0] < self.grid["xmax"] - 0.3
                    and self.grid["ymin"] + 0.3 <= cg[1] < self.grid["ymax"] - 0.3
                    and self.grid["zmin"] + 0.3 <= cg[2] < self.grid["zmax"] - 0.3):
                continue
            R2 = M @ sR
            # collision check against everything already in the frame
            ok = True
            for c, d in zip(new_c, new_d):
                if np.linalg.norm(c - c2) < 0.5 * (np.linalg.norm(d) + np.linalg.norm(sd)) * 0.8:
                    ok = False
                    break
            if not ok:
                continue
            p2 = spts.copy()
            sd = sd.copy()
            lo_s, hi_s = cfg.get("inst_scale", [1.0, 1.0])
            if lo_s != 1.0 or hi_s != 1.0:        # scale the diver about its own centre (points + extents)
                k = rng.uniform(lo_s, hi_s)
                p2[:, :3] = sc + (p2[:, :3] - sc) * k
                sd = sd * k
            p2[:, :3] = p2[:, :3] @ M.T + shift
            # clear background returns that would sit inside the pasted diver
            keep = ~points_in_box_mask(points[:, :3], c2, sd, R2, 0.0)
            points = points[keep]
            new_pts.append(p2)
            new_c.append(c2)
            new_d.append(sd.copy())
            new_R.append(R2)
            added += 1
        if new_pts:
            fg = np.concatenate(new_pts, 0)
            if cfg.get("occlude", False):           # beam-consistent paste: one return per beam
                points, fg = occlude_merge(points, fg)
            points = np.concatenate([points, fg], 0)
        return (points, np.array(new_c).reshape(-1, 3), np.array(new_d).reshape(-1, 3),
                np.array(new_R).reshape(-1, 3, 3))


class Diver3DDataset(Dataset):
    def __init__(self, root: str, split: str, cfg: dict, training: bool,
                 limit: int | None = None, sampler: GTSampler | None = None,
                 exclude_scenes=None, only_scenes=None):
        self.frames = Frames(root, split, limit=limit, exclude_scenes=exclude_scenes, only_scenes=only_scenes)
        self.cfg = cfg
        self.training = training
        self.grid = cfg["grid"]
        self.aug = cfg.get("aug", {})
        self.sampler = sampler
        self.n_prev = cfg.get("data", {}).get("n_prev", 0)
        self.epoch = 0
        self.epochs = cfg["train"]["epochs"]
        self.seed = cfg["train"].get("seed", 0)
        self.targets = TargetBuilder(self.grid, cfg["out_stride"],
                                     max_objs=cfg.get("max_objs", 16),
                                     min_overlap=cfg["loss"].get("gaussian_min_overlap", 0.7),
                                     min_radius=cfg["loss"].get("min_radius", 1),
                                     z_relative=bool(cfg.get("data", {}).get("z_relative", False)))

    def set_epoch(self, e: int):
        self.epoch = e

    def gt_sample_target(self) -> int:
        g = self.aug.get("gt_sample", {})
        if not g.get("enable", False):
            return 0
        tmax, floor = g.get("target_max", 4), g.get("floor", 1)
        start = int(g.get("anneal_start_frac", 0.65) * self.epochs)
        if self.epoch < start:
            return tmax
        return max(floor, tmax - (self.epoch - start + 1))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        rng_key = i                       # oversampled duplicates arrive as i + k * len(frames)
        i = i % len(self.frames)
        token = self.frames.tokens[i]
        pts = self.frames.get_stack(i, self.n_prev)
        centers, dims, Rs, _ = self.frames.get_boxes(i)
        if self.training and self.aug.get("enable", True):
            rng = np.random.default_rng([self.seed, self.epoch, rng_key])
            pts, centers, dims, Rs = self.augment(rng, pts, centers, dims, Rs, scene=token.split("/")[0])
        lf = int(self.cfg.get("data", {}).get("local_feats", 0))
        if lf > 0:
            pts = add_local_feats(pts, lf)
        tg = self.targets(centers, dims, Rs, pts)
        return {"token": token, "points": pts.astype(np.float32),
                "n_gt": len(centers), **tg}

    def augment(self, rng, pts, centers, dims, Rs, scene=None):
        a = self.aug
        pmix = a.get("frame_mix", 0.0)
        if pmix > 0 and rng.random() < pmix:
            # union with the background of a random other frame (its divers removed), so the
            # scene layout stops being a reliable cue for box size / presence
            j = int(rng.integers(len(self.frames)))
            pj = self.frames.get_stack(j, self.n_prev)
            cj, dj, Rj, _ = self.frames.get_boxes(j)
            keep = np.ones(len(pj), dtype=bool)
            for c, d, R in zip(cj, dj, Rj):
                keep &= ~points_in_box_mask(pj[:, :3], c, d, R, 0.25)
            pj = pj[keep]
            if a.get("frame_mix_sub", 0.5) < 1.0:            # subsample the mixed-in background
                pj = pj[rng.random(len(pj)) < a.get("frame_mix_sub", 0.5)]
            pts = np.concatenate([pts, pj], 0)
        if self.sampler is not None:
            target = self.gt_sample_target()
            n_insert = max(0, target - len(centers))
            pts, centers, dims, Rs = self.sampler.sample(rng, pts, centers, dims, Rs, n_insert, scene=scene)
        if rng.random() < a.get("flip_y", 0.5):
            pts, centers, Rs = mirror_y(pts, centers, Rs)
        rz, ry, rx = (np.radians(a.get(k, 0.0)) for k in ("rot_z_deg", "rot_y_deg", "rot_x_deg"))
        if rz or ry or rx:
            M = rot_z(rng.uniform(-rz, rz)) @ rot_y(rng.uniform(-ry, ry)) @ rot_x(rng.uniform(-rx, rx))
            pts, centers, Rs = transform_scene(pts, centers, Rs, M)
        zs = a.get("z_shift", 0.0)                      # random vertical shift of the whole frame (m)
        if zs > 0:
            dz = rng.uniform(-zs, zs)
            pts = pts.copy(); pts[:, 2] += dz
            centers = centers + np.array([0.0, 0.0, dz])
        lo, hi = a.get("scale", [1.0, 1.0])
        if lo != 1.0 or hi != 1.0:
            s = rng.uniform(lo, hi)
            pts = pts.copy()
            pts[:, :3] *= s
            centers = centers * s
            dims = dims * s
        bd = a.get("beam_drop", 0.0)            # beam-structured dropout: drop every k-th bearing column or elevation row
        if bd > 0 and rng.random() < bd:
            key, _ = beam_keys(pts[:, :3])
            k = int(rng.choice([2, 3])); off = int(rng.integers(k)); ne = BEAM["n_elev"]
            idx = (key // ne) if rng.random() < 0.5 else (key % ne)   # bearing index or elevation index
            keep = (key < 0) | ((idx % k) != off)
            pts = pts[keep]
        pdrop = a.get("point_drop_max", 0.0)
        if pdrop > 0:
            keep = rng.random(len(pts)) >= rng.uniform(0, pdrop)
            pts = pts[keep]
        ij = a.get("intensity_jitter", 0.0)
        if ij > 0:
            pts = pts.copy()
            pts[:, 3] = np.clip(pts[:, 3] * rng.uniform(1 - ij, 1 + ij), 0, 1)
        return pts, centers, dims, Rs


def collate(batch: list[dict]) -> dict:
    pts = []
    for b, item in enumerate(batch):
        p = item["points"]
        pts.append(np.concatenate([np.full((len(p), 1), b, dtype=np.float32), p], 1))
    out = {
        "tokens": [b["token"] for b in batch],
        "points": torch.from_numpy(np.concatenate(pts, 0)),
        "n_gt": torch.tensor([b["n_gt"] for b in batch]),
    }
    for k in ("hm", "reg", "inds", "mask"):
        out[k] = torch.from_numpy(np.stack([b[k] for b in batch], 0))
    return out


class SceneWeightedSampler(torch.utils.data.Sampler):
    """Draws len(frames) indices per epoch with replacement, frame weight = scene weight (default 1).
    The k-th duplicate of frame i is yielded as i + k * N so Diver3DDataset gives it a different
    augmentation RNG. Reshuffled every epoch (seed + epoch counter)."""

    def __init__(self, tokens: list[str], scene_weights: dict, seed: int = 0):
        self.N = len(tokens)
        w = np.array([float(scene_weights.get(t.split("/")[0], 1.0)) for t in tokens])
        self.p = w / w.sum()
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.N

    def __iter__(self):
        rng = np.random.default_rng([self.seed, 7919, self.epoch])
        self.epoch += 1
        draws = rng.choice(self.N, self.N, replace=True, p=self.p)
        order = rng.permutation(self.N)
        seen = {}
        out = []
        for i in draws[order]:
            k = seen.get(i, 0)
            seen[i] = k + 1
            out.append(int(i + k * self.N))
        return iter(out)
