"""Per-scene SDS deltas, paired per-GT IoU and size-bias diagnostics of an ensemble file vs a base file.

    python analysis/scalediag.py <new ensemble val file> [<base val file>]
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official
from diver3d.paired import per_gt_best

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
gt = load_gt(root, "val")
base_file = sys.argv[2] if len(sys.argv) > 2 else "submissions/val/ens_mixed7_val.jsonl.gz"
base = ev.load_boxes(base_file, require_score=True)
new = ev.load_boxes(sys.argv[1], require_score=True)
scenes = sorted({t.split("/")[0] for t in gt})


def per_scene(p):
    return {s: ev.evaluate({t: v for t, v in gt.items() if t.startswith(s)}, {t: p.get(t, []) for t in gt if t.startswith(s)}, verbose=False) for s in scenes}


rb, rn = ev.evaluate(gt, base, verbose=False), ev.evaluate(gt, new, verbose=False)
pb, pn = per_scene(base), per_scene(new)
print(f"{sys.argv[1]}: SDS {rn['SDS']:.4f} ({rn['SDS']-rb['SDS']:+.4f}) AP35 {rn['AP3D@0.35']:.4f} ({rn['AP3D@0.35']-rb['AP3D@0.35']:+.4f}) "
      f"AP30 {rn['AP3D@0.30']-rb['AP3D@0.30']:+.4f} AP40 {rn['AP3D@0.40']-rb['AP3D@0.40']:+.4f} APBEV35 {rn['APBEV@0.35']:.4f} ({rn['APBEV@0.35']-rb['APBEV@0.35']:+.4f}) "
      f"ATE {rn['ATE_m']:.3f} ASE {rn['ASE']:.3f} AOE3D {rn['AOE3D_deg']:.1f}")
print("per-scene dSDS: " + " ".join(f"{s[-4:]} {pn[s]['SDS']-pb[s]['SDS']:+.4f}" for s in scenes) + f" | up {sum(pn[s]['SDS'] > pb[s]['SDS'] for s in scenes)}/7")
print("per-scene dAP35: " + " ".join(f"{s[-4:]} {pn[s]['AP3D@0.35']-pb[s]['AP3D@0.35']:+.4f}" for s in scenes))
A, B = per_gt_best(ev, new, gt), per_gt_best(ev, base, gt)
keys = sorted(A)
r = np.array([A[k][2] for k in keys])
a3 = np.array([A[k][1] for k in keys])
b3 = np.array([B[k][1] for k in keys])


def bias(pred):
    out = {}
    for tok, g in gt.items():
        p = pred.get(tok, [])
        if not p:
            continue
        M = ev._iou_matrix(p, g, ev.iou_3d)
        for j, gb in enumerate(g):
            i = int(np.argmax(M[:, j]))
            if M[i, j] > 0.1:
                A_ = np.abs(gb.R.T @ p[i].R)
                d = np.array([p[i].dims[int(np.argmax(A_[k]))] for k in range(3)])
                out[(tok, j)] = (np.log(d / gb.dims), np.linalg.norm(p[i].center[:2]) - np.linalg.norm(gb.center[:2]), p[i].center[2] - gb.center[2])
    return out


bb, bn = bias(base), bias(new)
for name, sel in (("r<2", r < 2), ("2-3", (r >= 2) & (r < 3)), ("r>=3", r >= 3), ("all", r >= 0)):
    ks = [k for k, s in zip(keys, sel) if s and k in bb and k in bn]
    lb = np.array([bb[k][0] for k in ks])
    ln = np.array([bn[k][0] for k in ks])
    print(f"  {name:5s} n {sel.sum():5d} | 3D IoU base {b3[sel].mean():.3f} new {a3[sel].mean():.3f} ({(a3[sel]-b3[sel]).mean():+.4f}, improved {np.mean(a3[sel] > b3[sel]):.2f}) "
          f"found@0.35 {np.mean(b3[sel] >= .35):.3f} -> {np.mean(a3[sel] >= .35):.3f} | mean log(pred/GT) l,w,h base {lb.mean(0).round(3)} new {ln.mean(0).round(3)} | "
          f"radial offset base {np.mean([bb[k][1] for k in ks]):+.3f} new {np.mean([bn[k][1] for k in ks]):+.3f} | dz base {np.mean([bb[k][2] for k in ks]):+.3f} new {np.mean([bn[k][2] for k in ks]):+.3f}")
