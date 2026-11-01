"""Pre-registered paired test for the range-view branch: per GT box at r < r_max, the best BEV IoU
(and 3D IoU) from model A vs model B on val. Reports mean/median paired difference and the
fraction of boxes improved. Also the same for all ranges.

    python -m diver3d.paired --a submissions/val/rv_only_val.jsonl.gz --b submissions/val/ref_*.jsonl.gz
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

from .evalio import load_gt, official


def per_gt_best(ev, pred, gt):
    out = {}
    for tok, g in gt.items():
        p = pred.get(tok, [])
        for j, gb in enumerate(g):
            if not p:
                out[(tok, j)] = (0.0, 0.0, np.linalg.norm(gb.center[:2]))
                continue
            Mb = ev._iou_matrix(p, g, ev.iou_bev)[:, j]
            M3 = ev._iou_matrix(p, g, ev.iou_3d)[:, j]
            out[(tok, j)] = (float(Mb.max()), float(M3.max()), np.linalg.norm(gb.center[:2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--root", default="data/diver3d_v2_extracted/diver3d_v2")
    ap.add_argument("--r_max", type=float, default=2.5)
    args = ap.parse_args()
    ev = official(args.root)
    gt = load_gt(args.root, "val")
    A = per_gt_best(ev, ev.load_boxes(args.a, require_score=True), gt)
    keys = sorted(A)
    r = np.array([A[k][2] for k in keys])
    a_bev = np.array([A[k][0] for k in keys]); a_3d = np.array([A[k][1] for k in keys])
    bs = [f for pat in args.b for f in sorted(glob.glob(pat))]
    B_bev = []; B_3d = []
    for f in bs:
        B = per_gt_best(ev, ev.load_boxes(f, require_score=True), gt)
        B_bev.append(np.array([B[k][0] for k in keys])); B_3d.append(np.array([B[k][1] for k in keys]))
    B_bev = np.mean(B_bev, 0); B_3d = np.mean(B_3d, 0)
    for name, sel in ((f"r < {args.r_max} m", r < args.r_max), ("all ranges", np.ones_like(r, dtype=bool))):
        d = a_bev[sel] - B_bev[sel]; d3 = a_3d[sel] - B_3d[sel]
        print(f"{name:12s} n={sel.sum():5d} | BEV IoU A {a_bev[sel].mean():.3f} B {B_bev[sel].mean():.3f} paired diff mean {d.mean():+.3f} "
              f"median {np.median(d):+.3f} improved {np.mean(d > 0):.2f} | 3D IoU A {a_3d[sel].mean():.3f} B {B_3d[sel].mean():.3f} diff {d3.mean():+.3f} "
              f"| found@0.35 A {np.mean(a_3d[sel] >= 0.35):.3f} B {np.mean(B_3d[sel] >= 0.35):.3f}")
    print(f"reference models averaged: {len(bs)}")


if __name__ == "__main__":
    main()
