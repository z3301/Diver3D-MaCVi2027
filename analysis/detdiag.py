"""Three inference-only diagnostics on a val prediction file (assumption audit / review round):
  1. oracle footprint: dims replaced by the nearest GT's dims (centre/rotation untouched) -> APBEV/AP35;
     and dims / 1.3 only for predictions under 2 m;
  2. score comparability: pooled APBEV vs n_gt-weighted mean of per-scene APBEV; TP score medians by
     scene and range bin;
  3. adjacent divers: found@BEV0.35 for GT boxes with another GT within 1.2 m vs isolated.

    python analysis/detdiag.py submissions/val/ens3z_ref158_val.jsonl.gz
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
gt = load_gt(root, "val")
pred = ev.load_boxes(sys.argv[1], require_score=True)
scenes = sorted({t.split("/")[0] for t in gt})
base = ev.evaluate(gt, pred, verbose=False)
print(f"base: SDS {base['SDS']:.4f} AP35 {base['AP3D@0.35']:.4f} APBEV35 {base['APBEV@0.35']:.4f} AP50 {base['AP3D@0.50']:.4f}")

# ---- 1. oracle footprint ------------------------------------------------------------------
def variant(mode):
    out = {}
    for tok, p in pred.items():
        g = gt.get(tok, [])
        bs = []
        for b in p:
            d, c = b.dims, b.center
            r = np.linalg.norm(c[:2])
            if g:
                dist = [np.linalg.norm(gg.center[:2] - c[:2]) for gg in g]
                j = int(np.argmin(dist))
                if dist[j] < 1.0:
                    if mode == "gt_dims":
                        d = g[j].dims
                    elif mode == "gt_dims_xy":                       # footprint only, keep predicted height
                        d = np.array([g[j].dims[0], g[j].dims[1], b.dims[2]])
                    elif mode == "gt_dims_z":
                        d = g[j].dims; c = np.array([c[0], c[1], g[j].center[2]])
            if mode == "shrink_near" and r < 2.0:
                d = b.dims / 1.3
            if mode == "shrink_near_xy" and r < 2.0:
                d = np.array([b.dims[0] / 1.3, b.dims[1] / 1.3, b.dims[2]])
            bs.append(ev.Box(center=c, dims=d, R=b.R, score=b.score))
        out[tok] = bs
    return out

for mode in ("gt_dims", "gt_dims_xy", "gt_dims_z", "shrink_near", "shrink_near_xy"):
    r = ev.evaluate(gt, variant(mode), verbose=False)
    print(f"  {mode:14s}: SDS {r['SDS']:.4f} ({r['SDS']-base['SDS']:+.4f}) AP35 {r['AP3D@0.35']:.4f} ({r['AP3D@0.35']-base['AP3D@0.35']:+.4f}) "
          f"APBEV35 {r['APBEV@0.35']:.4f} ({r['APBEV@0.35']-base['APBEV@0.35']:+.4f}) AP50 {r['AP3D@0.50']:.4f}", flush=True)

# ---- 2. score comparability across recordings ----------------------------------------------
per = {}
for s in scenes:
    g = {t: v for t, v in gt.items() if t.startswith(s)}
    per[s] = (ev.evaluate(g, {t: pred.get(t, []) for t in g}, verbose=False), sum(len(v) for v in g.values()))
w = np.array([per[s][1] for s in scenes], float)
mean_bev = sum(per[s][0]["APBEV@0.35"] * per[s][1] for s in scenes) / w.sum()
mean_35 = sum(per[s][0]["AP3D@0.35"] * per[s][1] for s in scenes) / w.sum()
print(f"\npooled APBEV35 {base['APBEV@0.35']:.4f} vs n_gt-weighted mean of per-scene APBEV35 {mean_bev:.4f} (gap {mean_bev-base['APBEV@0.35']:+.4f}); "
      f"AP35 pooled {base['AP3D@0.35']:.4f} vs per-scene mean {mean_35:.4f} (gap {mean_35-base['AP3D@0.35']:+.4f})")
print("per-scene APBEV35: " + " ".join(f"{s[-4:]} {per[s][0]['APBEV@0.35']:.3f}" for s in scenes))
# TP score by scene and range
tp = []
for tok, g in gt.items():
    p = sorted(pred.get(tok, []), key=lambda b: -b.score)
    if not p or not g:
        continue
    m = ev._greedy_match(ev._iou_matrix(p, g, ev.iou_3d), 0.35)
    for i in np.flatnonzero(m >= 0):
        tp.append((tok.split("/")[0], np.linalg.norm(g[m[i]].center[:2]), p[i].score))
tp = np.array([(r, s) for _, r, s in tp]); sc_ = [x for x, _, _ in []]
scs = np.array([t.split("/")[0] for t in gt for _ in gt[t]])  # placeholder not used
import collections
by_scene = collections.defaultdict(list)
for tok, g in gt.items():
    pass
print("median TP score by range bin: " + " ".join(f"{lo}-{hi}: {np.median(tp[(tp[:,0]>=lo)&(tp[:,0]<hi),1]):.2f} (n {((tp[:,0]>=lo)&(tp[:,0]<hi)).sum()})" for lo, hi in ((0, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 99))))

# ---- 3. adjacent divers --------------------------------------------------------------------
rows = []
for tok, g in gt.items():
    if not g:
        continue
    p = pred.get(tok, [])
    for j, gb in enumerate(g):
        nb = min((np.linalg.norm(gb.center[:2] - gg.center[:2]) for k, gg in enumerate(g) if k != j), default=99.0)
        bev = max((ev.iou_bev(b, gb) for b in p), default=0.0)
        i3 = max((ev.iou_3d(b, gb) for b in p), default=0.0)
        rows.append((nb, bev, i3, np.linalg.norm(gb.center[:2])))
R = np.array(rows)
adj = R[:, 0] < 1.2
print(f"\nGT with a same-frame neighbour within 1.2 m: {adj.sum()} of {len(R)} ({adj.mean():.3f}); found@BEV0.35 adjacent {np.mean(R[adj,1] >= .35):.3f} vs isolated {np.mean(R[~adj,1] >= .35):.3f}; "
      f"found@3D0.35 adjacent {np.mean(R[adj,2] >= .35):.3f} vs isolated {np.mean(R[~adj,2] >= .35):.3f}")
near = R[:, 3] < 2.5
print(f"  under 2.5 m: adjacent {np.sum(adj & near)} found@BEV0.35 {np.mean(R[adj & near, 1] >= .35):.3f} vs isolated {np.mean(R[~adj & near, 1] >= .35):.3f}")
