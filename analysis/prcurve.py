"""Score-scale-free FP burden: precision and FP count at matched recall (3D IoU 0.35, official greedy
matching per frame), for several prediction files.

    python analysis/prcurve.py <pred files...> [--exclude=scene_0075]
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
gt = load_gt(root, "val")
excl = [a.split("=")[1] for a in sys.argv[1:] if a.startswith("--exclude=")]
files = [a for a in sys.argv[1:] if not a.startswith("--")]
G = {t: v for t, v in gt.items() if t.split("/")[0] not in excl}
n_gt = sum(len(v) for v in G.values())
R = (0.6, 0.7, 0.8, 0.85, 0.9)
print(f"excluded scenes {excl}; n_gt {n_gt}")
print(f"{'file':44s} " + " ".join(f"P@R{r:.2f}  FP@R{r:.2f}" for r in R) + "   maxR    AP35")
for f in files:
    pred = ev.load_boxes(f, require_score=True)
    sc, tp = [], []
    for tok, g in G.items():
        p = sorted(pred.get(tok, []), key=lambda b: -b.score)
        if not p:
            continue
        m = ev._greedy_match(ev._iou_matrix(p, g, ev.iou_3d), 0.35) if g else np.full(len(p), -1)
        sc += [b.score for b in p]
        tp += list(m >= 0)
    sc, tp = np.array(sc), np.array(tp, bool)
    o = np.argsort(-sc)
    tps = tp[o]
    ctp = np.cumsum(tps)
    cfp = np.cumsum(~tps)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    ap = ev.average_precision(sc, tp, n_gt)
    row = []
    for r in R:
        k = np.searchsorted(rec, r)
        row.append(f"{prec[k]:7.3f} {cfp[k]:9d}" if k < len(rec) else f"{'-':>7s} {'-':>9s}")
    print(f"{f.split('/')[-1][:44]:44s} " + " ".join(row) + f"  {rec[-1]:.3f}  {ap:.4f}")
