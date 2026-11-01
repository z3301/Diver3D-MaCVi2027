"""Where does detection AP go? Recall ceiling, near-misses, false positives on empty frames,
per-scene AP, misses by range / tilt. Uses the official matching code.

    python -m diver3d.diagnose --pred submissions/val/x.jsonl.gz
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from .evalio import load_gt, official, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", default="data/diver3d_v2_extracted/diver3d_v2")
    ap.add_argument("--thr", type=float, default=0.3, help="score threshold for the FP analysis")
    args = ap.parse_args()
    ev = official(args.root)
    gt = load_gt(args.root, "val")
    pred = ev.load_boxes(args.pred, require_score=True)

    gt_best = []        # (scene, max IoU over all preds, score of that pred, range, tilt)
    fp = []             # (scene, score, max IoU to any GT, frame_empty)
    tp_scores = []
    for tok in sorted(gt):
        g, p = gt[tok], pred.get(tok, [])
        scene = tok.split("/")[0]
        if g and p:
            M = ev._iou_matrix(p, g, ev.iou_3d)          # (P, G)
        else:
            M = np.zeros((len(p), len(g)))
        for j, gb in enumerate(g):
            if len(p):
                i = int(M[:, j].argmax())
                gt_best.append((scene, float(M[i, j]), p[i].score, np.linalg.norm(gb.center[:2]),
                                np.degrees(np.arccos(np.clip(gb.R[2, 2], -1, 1)))))
            else:
                gt_best.append((scene, 0.0, 0.0, np.linalg.norm(gb.center[:2]),
                                np.degrees(np.arccos(np.clip(gb.R[2, 2], -1, 1)))))
        # greedy match at 0.35 like the scorer, on preds above thr
        order = np.argsort([-b.score for b in p])
        used = set()
        for i in order:
            if p[i].score < args.thr:
                continue
            best, bj = 0.35, -1
            for j in range(len(g)):
                if j not in used and M[i, j] >= best:
                    best, bj = M[i, j], j
            if bj >= 0:
                used.add(bj)
                tp_scores.append(p[i].score)
            else:
                fp.append((scene, p[i].score, float(M[i].max()) if len(g) else 0.0, len(g) == 0))

    G = np.array([(b[1], b[2], b[3], b[4]) for b in gt_best])
    print(f"GT boxes: {len(G)}")
    for t in (0.3, 0.35, 0.4, 0.5):
        print(f"  recall ceiling @IoU {t:.2f} (any score): {np.mean(G[:, 0] >= t):.3f}")
    print(f"  GT with best IoU in [0.2,0.35) (near miss):  {np.mean((G[:, 0] >= 0.2) & (G[:, 0] < 0.35)):.3f}")
    print(f"  GT with best IoU < 0.1 (not found at all):   {np.mean(G[:, 0] < 0.1):.3f}")
    m = G[:, 0] >= 0.35
    print(f"  matched-GT score: median {np.median(G[m, 1]):.2f}, 10th pct {np.percentile(G[m, 1], 10):.2f}")
    print("\nmisses (best IoU < 0.35) by range bin:")
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6), (6, 20)):
        sel = (G[:, 2] >= lo) & (G[:, 2] < hi)
        if sel.any():
            print(f"  {lo:2d}-{hi:2d} m: n={sel.sum():5d}  miss rate {np.mean(G[sel, 0] < 0.35):.3f}")
    print("misses by tilt bin (deg):")
    for lo, hi in ((0, 15), (15, 45), (45, 75), (75, 105), (105, 180)):
        sel = (G[:, 3] >= lo) & (G[:, 3] < hi)
        if sel.any():
            print(f"  {lo:3d}-{hi:3d}: n={sel.sum():5d}  miss rate {np.mean(G[sel, 0] < 0.35):.3f}")

    F = np.array([(f[1], f[2], f[3]) for f in fp]) if fp else np.zeros((0, 3))
    print(f"\nfalse positives with score >= {args.thr}: {len(F)}  (TPs at this thr: {len(tp_scores)})")
    if len(F):
        print(f"  on empty frames: {np.mean(F[:, 2] > 0):.3f}")
        print(f"  near-miss (IoU to some GT in [0.1,0.35)): {np.mean((F[:, 1] >= 0.1) & (F[:, 1] < 0.35)):.3f}")
        print(f"  spurious (IoU < 0.1, frame has GT): {np.mean((F[:, 1] < 0.1) & (F[:, 2] == 0)):.3f}")
        print(f"  FP score: median {np.median(F[:, 0]):.2f}, 90th pct {np.percentile(F[:, 0], 90):.2f}; "
              f"TP score median {np.median(tp_scores):.2f}")
        # empty-frame top scores
    empt = [max((b.score for b in pred.get(t, [])), default=0.0) for t in gt if not gt[t]]
    print(f"  empty frames: {len(empt)}; top score >= 0.3 on {np.mean(np.array(empt) >= 0.3):.3f}, "
          f">= 0.5 on {np.mean(np.array(empt) >= 0.5):.3f}")

    print("\nper-scene (official scorer on the scene's frames):")
    by_scene = defaultdict(list)
    for t in gt:
        by_scene[t.split("/")[0]].append(t)
    for sc in sorted(by_scene):
        toks = by_scene[sc]
        g = {t: gt[t] for t in toks}
        p = {t: pred.get(t, []) for t in toks}
        r = score(args.root, g, p)
        n_gt = sum(len(v) for v in g.values())
        tilts = [np.degrees(np.arccos(np.clip(b.R[2, 2], -1, 1))) for v in g.values() for b in v]
        up = np.mean(np.array(tilts) < 15) if tilts else 0
        print(f"  {sc}: frames {len(toks):5d} gt {n_gt:5d} upright {up:.2f} | SDS {r['SDS']:.3f} "
              f"mAP3D {r['mAP3D']:.3f} AP35 {r['AP3D@0.35']:.3f} AP50 {r['AP3D@0.50']:.3f} AOE3D {r['AOE3D_deg']:.1f}")


if __name__ == "__main__":
    main()
