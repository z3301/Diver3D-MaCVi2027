"""Fusion of 9-DoF boxes across TTA passes / ensemble members (weighted box fusion in SO(3))."""
from __future__ import annotations

import numpy as np

from .evalio import server_rotation

_RZ_PI = np.diag([-1.0, -1.0, 1.0])


def _align_symmetry(R: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pick R or R·Rz(pi), whichever is closer to ref (the cuboid is the same either way)."""
    alt = R @ _RZ_PI
    return R if np.trace(ref.T @ R) >= np.trace(ref.T @ alt) else alt


def mean_rotation(Rs: np.ndarray, w: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Weighted chordal mean on SO(3) after folding the Rz(pi) ambiguity toward `ref`."""
    M = sum(wi * _align_symmetry(Ri, ref) for Ri, wi in zip(Rs, w))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def fuse_frame(ev, box_lists: list[list], iou_thr: float = 0.45, n_sources: int | None = None,
               score_mode: str = "avg") -> list:
    """Weighted box fusion over several detection lists for one frame.

    box_lists: one list of official Box objects per source (model / TTA pass).
    Boxes are clustered greedily by 3D IoU with the cluster's running fused box; centre, dims
    are score-weighted means, rotation is the weighted chordal mean. The fused score is the
    mean score over sources (absent sources count as 0), which down-weights boxes only one
    member found.
    """
    n_sources = n_sources or len(box_lists)
    cands = [(b, si) for si, bl in enumerate(box_lists) for b in bl]
    cands.sort(key=lambda t: -t[0].score)
    clusters = []   # each: dict(members=[(box, src)], fused=Box)
    for b, si in cands:
        best, best_iou = None, iou_thr
        for cl in clusters:
            iou = ev.iou_3d(cl["fused"], b)
            if iou > best_iou:
                best, best_iou = cl, iou
        if best is None:
            clusters.append({"members": [(b, si)], "fused": b})
            continue
        best["members"].append((b, si))
        ms = best["members"]
        w = np.array([m[0].score for m in ms])
        w = w / w.sum()
        c = sum(wi * m[0].center for m, wi in zip(ms, w))
        d = sum(wi * m[0].dims for m, wi in zip(ms, w))
        R = mean_rotation(np.stack([m[0].R for m in ms]), w, ms[0][0].R)
        best["fused"] = ev.Box(center=c, dims=d, R=server_rotation(R), score=best["fused"].score)
    out = []
    for cl in clusters:
        ms = cl["members"]
        scores = np.array([m[0].score for m in ms])
        if score_mode == "avg":
            s = scores.sum() / n_sources                      # missing sources contribute 0
        else:
            s = scores.max()
        b = cl["fused"]
        out.append(ev.Box(center=b.center, dims=b.dims, R=b.R, score=float(min(1.0, s))))
    out.sort(key=lambda b: -b.score)
    return out


def fuse_predictions(ev, preds: list[dict], iou_thr: float = 0.45, score_mode: str = "avg") -> dict:
    """preds: list of {token: [Box]} from different sources -> fused {token: [Box]}."""
    tokens = sorted(set().union(*[p.keys() for p in preds]))
    return {t: fuse_frame(ev, [p.get(t, []) for p in preds], iou_thr, len(preds), score_mode) for t in tokens}
