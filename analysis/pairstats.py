"""Label-free count of close-pair candidates that BEV NMS at 0.1 suppresses: per member pass, sort boxes by
score, run NMS 0.1, and count suppressed boxes with score >= s_min whose centre lies 0.35-1.2 m from the
suppressing box (a second diver, not a duplicate). Reported per scene as suppressed candidates per frame,
averaged over member passes. Run on test, val and out-of-fold train dumps to compare.

    python analysis/pairstats.py --split test --names vz025 vz025_s1 ...
    python analysis/pairstats.py --split train --members submissions/train/members --names oof_f0 oof_f2 --gt
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official

ROOT = "data/diver3d_v2_extracted/diver3d_v2"
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--members", default=None)
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--s_min", type=float, default=0.3)
ap.add_argument("--gt", action="store_true", help="also count GT close pairs per scene (labelled splits)")
a = ap.parse_args()
members_dir = a.members or f"submissions/{a.split}/members"
ev = official(ROOT)
per = defaultdict(lambda: [0, 0.0, 0.0])   # scene -> frames, suppressed close candidates, frames with >= 1
n_pass = 0
frames_seen = defaultdict(set)
for n in a.names:
    for p in (0, 1):
        f = os.path.join(members_dir, f"{n}_p{p}_{a.split}.jsonl.gz")
        if not os.path.exists(f):
            continue
        n_pass += 1
        pred = ev.load_boxes(f, require_score=True)
        for tok, bs in pred.items():
            s = tok.split("/")[0]
            frames_seen[s].add(tok)
            bs = sorted([b for b in bs if b.score >= 0.05], key=lambda b: -b.score)
            keep, sup = [], 0
            for b in bs:
                hit = None
                for k in keep:
                    if ev.iou_bev(b, k) > 0.1:
                        hit = k
                        break
                if hit is None:
                    keep.append(b)
                elif b.score >= a.s_min and hit.score >= a.s_min:
                    d = np.linalg.norm(b.center[:2] - hit.center[:2])
                    if 0.35 <= d <= 1.2:
                        sup += 1
            per[s][1] += sup
            per[s][2] += sup > 0
for s in per:
    per[s][0] = len(frames_seen[s])
gtc = {}
if a.gt:
    gt = load_gt(ROOT, a.split)
    for tok, g in gt.items():
        s = tok.split("/")[0]
        if s not in per:
            continue
        c = np.array([gb.center[:2] for gb in g]) if g else np.zeros((0, 2))
        k = sum(1 for i in range(len(c)) if any(np.linalg.norm(c[i] - c[j]) < 1.0 for j in range(len(c)) if j != i))
        gtc.setdefault(s, [0, 0]); gtc[s][0] += k; gtc[s][1] += len(g)
print(f"{a.split}: {n_pass} member passes; suppressed close-pair candidates (score >= {a.s_min}, 0.35-1.2 m) per frame per pass")
tot_f = tot_s = 0
for s in sorted(per):
    f, sup, fr = per[s]
    tot_f += f; tot_s += sup
    extra = f" | GT close-pair boxes {gtc[s][0]} of {gtc[s][1]}" if s in gtc else ""
    print(f"  {s}: frames {f:5d} | suppressed/frame {sup / max(f, 1) / max(n_pass, 1):.4f} | frames with any {fr / max(f, 1) / max(n_pass, 1):.4f}{extra}")
print(f"  all: suppressed/frame {tot_s / max(tot_f, 1) / max(n_pass, 1):.4f}")
