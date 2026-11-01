"""Extent snapping: set each predicted box's extents (and optionally centre) from the robust
extent of the sonar returns inside the enlarged predicted box, measured along the predicted body
axes. Motivated by the val finding that GT extents track the observed point extent (R2 0.8-0.9,
ratio ~0.95) while the detector's extents barely do (R2 0.04-0.5).

    python -m diver3d.snap --pred submissions/val/x.jsonl.gz --split val --grid
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import Frames
from .evalio import load_gt, official, score, server_rotation, write_submission


def snap_boxes(ev, frames: Frames, pred: dict, margin=0.3, ratio=0.95, blend_dims=1.0, blend_center=0.0,
               q=(2.0, 98.0), min_pts=10, max_change=0.5, axes=(0, 1, 2)) -> dict:
    idx = {t: i for i, t in enumerate(frames.tokens)}
    out = {}
    for tok, boxes in pred.items():
        pts = frames.points[idx[tok]][:, :3] if frames.points is not None else frames.get_points(idx[tok])[:, :3]
        new = []
        for b in boxes:
            R = b.R
            loc = (pts - b.center) @ R                                   # body-frame coords
            half = b.dims / 2 + margin
            m = np.all(np.abs(loc) <= half, axis=1)
            if m.sum() < min_pts:
                new.append(b)
                continue
            l = loc[m]
            lo = np.percentile(l, q[0], axis=0)
            hi = np.percentile(l, q[1], axis=0)
            ext = (hi - lo) * ratio
            dims = b.dims.copy()
            center = b.center.copy()
            for a in axes:
                d_new = float(np.clip(ext[a], b.dims[a] * (1 - max_change), b.dims[a] * (1 + max_change)))
                dims[a] = (1 - blend_dims) * b.dims[a] + blend_dims * d_new
            if blend_center > 0:
                mid = (hi + lo) / 2
                shift = np.zeros(3)
                for a in axes:
                    shift[a] = float(np.clip(mid[a], -max_change, max_change))
                center = center + blend_center * (R @ shift)
            new.append(ev.Box(center=center, dims=dims, R=server_rotation(R), score=b.score))
        out[tok] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", default="data/diver3d_v2_extracted/diver3d_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--ratio", type=float, default=0.95)
    ap.add_argument("--blend_dims", type=float, default=1.0)
    ap.add_argument("--blend_center", type=float, default=0.0)
    args = ap.parse_args()
    ev = official(args.root)
    fr = Frames(args.root, args.split)
    pred = ev.load_boxes(args.pred, require_score=True)
    gt = load_gt(args.root, args.split) if args.split != "test" else None

    def rep(name, p):
        if gt is None:
            return
        m = score(args.root, gt, p)
        print(f"{name:52s} SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} AP50 {m['AP3D@0.50']:.4f} "
              f"ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f}", flush=True)

    rep("raw", pred)
    if args.grid:
        for margin in (0.2, 0.3):
            for ratio in (0.9, 0.95, 1.0):
                for bd in (0.5, 1.0):
                    rep(f"dims margin {margin} ratio {ratio} blend {bd}", snap_boxes(ev, fr, pred, margin, ratio, bd, 0.0))
        for bc in (0.5, 1.0):
            rep(f"centre only blend {bc} (margin 0.3)", snap_boxes(ev, fr, pred, 0.3, 0.95, 0.0, bc))
            rep(f"dims 1.0 + centre {bc} (margin 0.3 ratio 0.95)", snap_boxes(ev, fr, pred, 0.3, 0.95, 1.0, bc))
        rep("h axis only (margin 0.3 ratio 0.95)", snap_boxes(ev, fr, pred, 0.3, 0.95, 1.0, 0.0, axes=(2,)))
        rep("l,w axes only (margin 0.3 ratio 0.95)", snap_boxes(ev, fr, pred, 0.3, 0.95, 1.0, 0.0, axes=(0, 1)))
        return
    p = snap_boxes(ev, fr, pred, args.margin, args.ratio, args.blend_dims, args.blend_center)
    rep(f"snap margin {args.margin} ratio {args.ratio} dims {args.blend_dims} centre {args.blend_center}", p)
    if args.out:
        write_submission(p, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
