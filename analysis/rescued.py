"""Counterfactual 'rescued-pair' statistic between two final outputs of the same pipeline (A = NMS 0.1,
B = NMS 0.3): per frame, a box in B (score >= 0.3) with no A box (score >= 0.2) within 0.3 m, lying
0.35-1.2 m from a confident B box that is also in A. R = fraction of frames with >= 1 rescued box.
With --gt also per-scene SDS / AP35 deltas, GT close-pair counts, and the fraction of close pairs with BOTH
divers matched one-to-one (official greedy matching at 3D IoU 0.35).

    python analysis/rescued.py --a <nms0.1 file> --b <nms0.3 file> --split train|val|test [--gt]
"""
import argparse, sys
from collections import defaultdict
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official
ROOT = "data/diver3d_v2_extracted/diver3d_v2"
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--split", default="test"); ap.add_argument("--gt", action="store_true")
a = ap.parse_args()
ev = official(ROOT)
A = ev.load_boxes(a.a, require_score=True); B = ev.load_boxes(a.b, require_score=True)
toks = sorted(set(A) | set(B))
per = defaultdict(lambda: [0, 0, 0])   # frames, frames with rescued, rescued boxes
for t in toks:
    s = t.split("/")[0]; per[s][0] += 1
    ca = np.array([b.center[:2] for b in A.get(t, []) if b.score >= 0.2]).reshape(-1, 2)
    bb = [b for b in B.get(t, []) if b.score >= 0.3]
    shared = [b for b in bb if len(ca) and np.min(np.linalg.norm(ca - b.center[:2], axis=1)) < 0.3]
    new = [b for b in bb if not len(ca) or np.min(np.linalg.norm(ca - b.center[:2], axis=1)) >= 0.3]
    k = sum(1 for b in new if any(0.35 <= np.linalg.norm(b.center[:2] - c.center[:2]) <= 1.2 for c in shared))
    per[s][1] += k > 0; per[s][2] += k
gt = load_gt(ROOT, a.split) if a.gt else None
print(f"{'scene':11s} {'frames':>6s} {'R':>7s} {'rescued':>7s}" + ("  | GT close  SDS A -> B (delta)   AP35 delta  both-matched A -> B" if gt else ""))
for s in sorted(per):
    f, fr, k = per[s]
    line = f"{s:11s} {f:6d} {fr / f:7.4f} {k:7d}"
    if gt:
        g = {t: v for t, v in gt.items() if t.startswith(s) and t in set(toks)}
        ra = ev.evaluate(g, {t: A.get(t, []) for t in g}, verbose=False); rb = ev.evaluate(g, {t: B.get(t, []) for t in g}, verbose=False)
        npair, both = 0, [0, 0]
        for t, gg in g.items():
            c = np.array([x.center[:2] for x in gg]).reshape(-1, 2)
            pairs = [(i, j) for i in range(len(c)) for j in range(i + 1, len(c)) if np.linalg.norm(c[i] - c[j]) < 1.0]
            if not pairs: continue
            for idx, P in enumerate((A, B)):
                p = sorted(P.get(t, []), key=lambda x: -x.score)
                m = ev._greedy_match(ev._iou_matrix(p, gg, ev.iou_3d), 0.35) if p else np.array([], int)
                matched = set(int(x) for x in m if x >= 0)
                both[idx] += sum(1 for i, j in pairs if i in matched and j in matched)
            npair += len(pairs)
        line += f"  | {npair:5d}  {ra['SDS']:.4f} -> {rb['SDS']:.4f} ({rb['SDS'] - ra['SDS']:+.4f})  {rb['AP3D@0.35'] - ra['AP3D@0.35']:+.4f}"
        line += f"  {both[0] / max(npair, 1):.3f} -> {both[1] / max(npair, 1):.3f}" if npair else ""
    print(line)
F = sum(v[0] for v in per.values()); FR = sum(v[1] for v in per.values())
print(f"all: frames {F}, R {FR / F:.4f}, rescued boxes {sum(v[2] for v in per.values())}")
