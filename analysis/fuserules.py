"""Cluster-level comparison of fusion rules. For every val GT, collect the member-pass boxes that overlap
it (3D IoU > 0.1, score >= 0.05), then fuse them with several rules and score the result against the GT.
This isolates the geometry rule from clustering/scoring. Rules: score-weighted mean (current), medoid
(member maximising summed 3D IoU to the others), weighted-mean centre+rotation with median dims, with
25th-percentile dims, with min dims, top-score member, and the member with the smallest volume.

    python analysis/fuserules.py vz025,vz025_s1,... submissions/val/ens_mixed7_val.jsonl.gz
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official, server_rotation
from diver3d.fuse import mean_rotation

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
gt = load_gt(root, "val")
names = sys.argv[1].split(",")
fused_file = sys.argv[2]
members = [ev.load_boxes(f"submissions/val/members/{n}_p{p}_val.jsonl.gz", require_score=True) for n in names for p in (0, 1)]
fused = ev.load_boxes(fused_file, require_score=True)
RULES = ["wmean", "medoid", "med_dims", "p25_dims", "min_dims", "top_score", "min_vol", "wmean_c_med_d_medoid_R"]


def fuse_rule(ms, rule):
    w = np.array([b.score for b in ms]); w = w / w.sum()
    C = np.array([b.center for b in ms]); D = np.array([b.dims for b in ms]); Rs = np.stack([b.R for b in ms])
    c = (w[:, None] * C).sum(0); d = (w[:, None] * D).sum(0)
    R = server_rotation(mean_rotation(Rs, w, Rs[0]))
    if rule == "wmean":
        return ev.Box(center=c, dims=d, R=R, score=1.0)
    if rule in ("medoid", "wmean_c_med_d_medoid_R"):
        n = len(ms)
        if n == 1:
            k = 0
        else:
            M = np.array([[ev.iou_3d(a, b) if i != j else 0.0 for j, b in enumerate(ms)] for i, a in enumerate(ms)])
            k = int(np.argmax(M.sum(1)))
        if rule == "medoid":
            return ms[k]
        return ev.Box(center=c, dims=np.median(D, 0), R=ms[k].R, score=1.0)
    if rule == "med_dims":
        return ev.Box(center=c, dims=np.median(D, 0), R=R, score=1.0)
    if rule == "p25_dims":
        return ev.Box(center=c, dims=np.percentile(D, 25, axis=0), R=R, score=1.0)
    if rule == "min_dims":
        return ev.Box(center=c, dims=D.min(0), R=R, score=1.0)
    if rule == "top_score":
        return ms[int(np.argmax([b.score for b in ms]))]
    if rule == "min_vol":
        return ms[int(np.argmin([b.volume for b in ms]))]
    raise ValueError(rule)


rows = []
for tok, g in gt.items():
    if not g:
        continue
    allb = [b for m in members for b in m.get(tok, []) if b.score >= 0.05]
    fb = fused.get(tok, [])
    for gb in g:
        r = np.linalg.norm(gb.center[:2])
        ms = [b for b in allb if ev.iou_3d(b, gb) > 0.1]
        f3 = max((ev.iou_3d(b, gb) for b in fb), default=0.0)
        row = [r, len(ms), f3]
        for rule in RULES:
            row.append(ev.iou_3d(fuse_rule(ms, rule), gb) if ms else 0.0)
        rows.append(row)
R = np.array(rows)
print(f"GT boxes {len(R)}; mean member boxes overlapping a GT: {R[:, 1].mean():.1f}")
hdr = f"{'range':8s} {'n':>5s} {'fused file':>11s} " + " ".join(f"{r:>12s}" for r in RULES)
print("found@0.35 (3D):"); print(hdr)
for name, sel in (("r<1.5", R[:, 0] < 1.5), ("1.5-2", (R[:, 0] >= 1.5) & (R[:, 0] < 2)), ("2-3", (R[:, 0] >= 2) & (R[:, 0] < 3)), ("r>=3", R[:, 0] >= 3), ("all", R[:, 0] >= 0)):
    s = R[sel]
    print(f"{name:8s} {len(s):5d} {np.mean(s[:, 2] >= .35):11.3f} " + " ".join(f"{np.mean(s[:, 3 + i] >= .35):12.3f}" for i in range(len(RULES))))
print("mean 3D IoU:"); print(hdr)
for name, sel in (("r<1.5", R[:, 0] < 1.5), ("1.5-2", (R[:, 0] >= 1.5) & (R[:, 0] < 2)), ("2-3", (R[:, 0] >= 2) & (R[:, 0] < 3)), ("r>=3", R[:, 0] >= 3), ("all", R[:, 0] >= 0)):
    s = R[sel]
    print(f"{name:8s} {len(s):5d} {s[:, 2].mean():11.3f} " + " ".join(f"{s[:, 3 + i].mean():12.3f}" for i in range(len(RULES))))
