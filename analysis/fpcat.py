"""False-positive accounting on val: categories of unmatched detections and the AP cost of each.

    python analysis/fpcat.py <pred files...> [--full]
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
gt = load_gt(root, "val")
full = "--full" in sys.argv
for f in [a for a in sys.argv[1:] if not a.startswith("--")]:
    pred = ev.load_boxes(f, require_score=True)
    cat = {}
    for tok, p in pred.items():
        g = gt[tok]
        if not p:
            continue
        if not g:
            for i in range(len(p)):
                cat[(tok, i)] = "empty-frame"
            continue
        order = np.argsort([-b.score for b in p])
        ps = [p[i] for i in order]
        M3 = ev._iou_matrix(ps, g, ev.iou_3d)
        Mb = ev._iou_matrix(ps, g, ev.iou_bev)
        m = ev._greedy_match(M3, 0.35)
        for k, i in enumerate(order):
            if m[k] >= 0:
                cat[(tok, i)] = "TP"
            elif M3[k].max() > 0.1 or Mb[k].max() > 0.2:
                j = int(np.argmax(Mb[k]))
                cat[(tok, i)] = "dup" if (m == j).any() else "near-miss"
            else:
                cat[(tok, i)] = "far"
    names = ["TP", "near-miss", "dup", "far", "empty-frame"]
    sc = {n: np.array([pred[t][i].score for (t, i), c in cat.items() if c == n]) for n in names}
    print(f"\n{f}")
    print("  " + " | ".join(f"{n} n={len(sc[n])} (>=0.3: {(sc[n] >= 0.3).sum()}, med {np.median(sc[n]) if len(sc[n]) else 0:.2f})" for n in names))
    if full:
        base = ev.evaluate(gt, pred, verbose=False)
        print(f"  as is: SDS {base['SDS']:.4f} AP35 {base['AP3D@0.35']:.4f} APBEV35 {base['APBEV@0.35']:.4f}")
        for drop in (["empty-frame"], ["far"], ["near-miss"], ["far", "empty-frame"]):
            p2 = {t: [b for i, b in enumerate(v) if cat.get((t, i)) not in drop] for t, v in pred.items()}
            r = ev.evaluate(gt, p2, verbose=False)
            print(f"  drop {'+'.join(drop):22s}: SDS {r['SDS']:.4f} ({r['SDS']-base['SDS']:+.4f}) AP35 {r['AP3D@0.35']:.4f} ({r['AP3D@0.35']-base['AP3D@0.35']:+.4f}) APBEV35 {r['APBEV@0.35']:.4f} ({r['APBEV@0.35']-base['APBEV@0.35']:+.4f})")
