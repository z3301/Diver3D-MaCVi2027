"""Temporal geometry refinement of per-frame detections (inference-time, no training).

Follows the spirit of SonarVoxNet Sec. III-D: associate detections into short tracks by BEV
overlap within a scene, then smooth centre / rotation / size along the track. Never adds or
removes detections and never changes scores, so recall and ranking are untouched; only box
geometry moves.

    python -m diver3d.refine --pred submissions/val/x.jsonl.gz --split val [--causal] [--window 3]
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from .evalio import load_gt, official, score, server_rotation, write_submission
from .fuse import mean_rotation


def _frames_by_scene(tokens):
    scenes = defaultdict(list)
    for t in tokens:
        s, f = t.split("/")
        scenes[s].append((int(f), t))
    return {s: [t for _, t in sorted(v)] for s, v in scenes.items()}


def build_tracks(ev, pred: dict, iou_thr: float = 0.2, max_gap: int = 1):
    """Greedy frame-to-frame association by BEV IoU. Returns list of tracks; each track is a
    list of (token, index-into-pred[token]) in time order."""
    tracks = []
    for scene, toks in _frames_by_scene(pred.keys()).items():
        active = []   # (track_id, last_box, last_frame_idx)
        for fi, tok in enumerate(toks):
            boxes = pred[tok]
            order = np.argsort([-b.score for b in boxes])
            used = set()
            new_active = []
            for i in order:
                b = boxes[i]
                best, best_iou = None, iou_thr
                for a in active:
                    if a[0] in used or fi - a[2] > max_gap:
                        continue
                    iou = ev.iou_bev(b, a[1])
                    if iou > best_iou:
                        best, best_iou = a, iou
                if best is None:
                    tracks.append([(tok, i)])
                    tid = len(tracks) - 1
                else:
                    tid = best[0]
                    tracks[tid].append((tok, i))
                used.add(tid)
                new_active.append((tid, b, fi))
            # keep recent unmatched tracks alive for max_gap frames
            for a in active:
                if a[0] not in used and fi - a[2] < max_gap:
                    new_active.append(a)
            active = new_active
    return tracks


def refine(ev, pred: dict, window: int = 3, causal: bool = False, iou_thr: float = 0.2,
           smooth_center=True, smooth_rot=True, smooth_dims=True, dims_mode: str = "mean",
           min_len: int = 2, windows=None, tracks=None) -> dict:
    """Smooth each track's geometry over a window of neighbouring frames (score-weighted).

    window: frames on each side (causal: only the past). windows=(wc, wr, wd) overrides it per
    component (0 disables that component). dims_mode: mean | min | median. `tracks` can be passed
    to reuse an association computed once.
    """
    wc, wr, wd = windows if windows is not None else (
        window if smooth_center else 0, window if smooth_rot else 0, window if smooth_dims else 0)
    out = {t: list(bs) for t, bs in pred.items()}
    if tracks is None:
        tracks = build_tracks(ev, pred, iou_thr)

    def neighbours(boxes, k, win):
        n = len(boxes)
        lo = max(0, k - win)
        hi = k if causal else min(n - 1, k + win)
        idx = list(range(lo, hi + 1))
        w = np.array([boxes[j].score for j in idx], dtype=np.float64)
        return idx, w / w.sum()

    for tr in tracks:
        if len(tr) < min_len:
            continue
        boxes = [pred[t][i] for t, i in tr]
        for k, b in enumerate(boxes):
            c, d, R = b.center, b.dims, b.R
            if wc > 0:
                idx, w = neighbours(boxes, k, wc)
                c = sum(wi * boxes[j].center for j, wi in zip(idx, w))
            if wd > 0:
                idx, w = neighbours(boxes, k, wd)
                D = np.stack([boxes[j].dims for j in idx])
                if dims_mode.startswith("hq"):          # mean for l,w; upper quantile for h (oblique views lower-bound h)
                    d = (w[:, None] * D).sum(0)
                    d[2] = np.percentile(D[:, 2], float(dims_mode[2:]))
                elif dims_mode.startswith("q"):          # upper quantile for all axes
                    d = np.percentile(D, float(dims_mode[1:]), axis=0)
                else:
                    d = D.min(0) if dims_mode == "min" else np.median(D, 0) if dims_mode == "median" else (w[:, None] * D).sum(0)
            if wr > 0:
                idx, w = neighbours(boxes, k, wr)
                R = server_rotation(mean_rotation(np.stack([boxes[j].R for j in idx]), w, b.R))
            t, i = tr[k]
            out[t][i] = ev.Box(center=c, dims=d, R=R, score=b.score)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default="data/diver3d_v2_extracted/diver3d_v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--iou", type=float, default=0.2)
    ap.add_argument("--dims_mode", default="mean")
    ap.add_argument("--grid", action="store_true", help="sweep a few settings and print a table")
    ap.add_argument("--grid2", action="store_true", help="per-component window sweep (centered)")
    ap.add_argument("--windows", type=int, nargs=3, default=None, metavar=("WC", "WR", "WD"))
    args = ap.parse_args()
    ev = official(args.root)
    pred = ev.load_boxes(args.pred, require_score=True)
    gt = load_gt(args.root, args.split) if args.split != "test" else None

    def report(name, p):
        if gt is None:
            return
        m = score(args.root, gt, p)
        print(f"{name:45s} SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} AP50 {m['AP3D@0.50']:.4f} "
              f"ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f} TILT {m['TILT_deg']:.1f}", flush=True)

    report("raw", pred)
    if args.grid:
        for causal in (False, True):
            for window in (1, 2, 3, 5):
                for parts in ("c", "r", "d", "crd"):
                    p = refine(ev, pred, window, causal, args.iou,
                               smooth_center="c" in parts, smooth_rot="r" in parts, smooth_dims="d" in parts,
                               dims_mode=args.dims_mode)
                    report(f"{'causal' if causal else 'centered'} w={window} {parts}", p)
        return
    if args.grid2:
        tracks = build_tracks(ev, pred, args.iou)
        for wc in (0, 1):
            for wr in (2, 3, 5, 8):
                for wd in (2, 3, 5, 8):
                    for dm in ("mean", "median"):
                        p = refine(ev, pred, causal=False, windows=(wc, wr, wd), dims_mode=dm, tracks=tracks)
                        report(f"wc={wc} wr={wr} wd={wd} dims={dm}", p)
        return
    p = refine(ev, pred, args.window, args.causal, args.iou, dims_mode=args.dims_mode, windows=args.windows)
    report(f"refined w={args.window} windows={args.windows} causal={args.causal}", p)
    if args.out:
        write_submission(p, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
