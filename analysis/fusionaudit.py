"""Does fusion destroy qualifying boxes? Per val GT: best 3D/BEV IoU over the UNION of member-pass boxes
(pre-fusion) vs the fused+refined ensemble's best box, by range bin.

    python analysis/fusionaudit.py vz025,vz025_s1,... submissions/val/ens_mixed7_val.jsonl.gz
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, official

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
gt = load_gt(root, "val")
names = sys.argv[1].split(",")
fused_file = sys.argv[2]
members = [ev.load_boxes(f"submissions/val/members/{n}_p{p}_val.jsonl.gz", require_score=True) for n in names for p in (0, 1)]
fused = ev.load_boxes(fused_file, require_score=True)
rows = []
for tok, g in gt.items():
    if not g:
        continue
    allb = [b for m in members for b in m.get(tok, []) if b.score >= 0.05]
    fb = fused.get(tok, [])
    for j, gb in enumerate(g):
        r = np.linalg.norm(gb.center[:2])
        u3 = max((ev.iou_3d(b, gb) for b in allb), default=0.0)
        ub = max((ev.iou_bev(b, gb) for b in allb), default=0.0)
        f3 = max((ev.iou_3d(b, gb) for b in fb), default=0.0)
        fbv = max((ev.iou_bev(b, gb) for b in fb), default=0.0)
        rows.append((r, u3, ub, f3, fbv))
R = np.array(rows)
print(f"members: {len(members)} passes; GT boxes {len(R)}")
print(f"{'range':8s} {'n':>5s} | union best 3D IoU / found@0.35 | fused best 3D IoU / found@0.35 | union BEV found@0.35 | fused BEV found@0.35 | GTs union>=0.35 & fused<0.35")
for name, sel in (("r<1.5", R[:, 0] < 1.5), ("1.5-2", (R[:, 0] >= 1.5) & (R[:, 0] < 2)), ("2-3", (R[:, 0] >= 2) & (R[:, 0] < 3)), ("r>=3", R[:, 0] >= 3), ("all", R[:, 0] >= 0)):
    s = R[sel]
    lost = (s[:, 1] >= .35) & (s[:, 3] < .35)
    print(f"{name:8s} {len(s):5d} | {s[:, 1].mean():.3f} / {np.mean(s[:, 1] >= .35):.3f} | {s[:, 3].mean():.3f} / {np.mean(s[:, 3] >= .35):.3f} | {np.mean(s[:, 2] >= .35):.3f} | {np.mean(s[:, 4] >= .35):.3f} | {lost.sum():5d} ({lost.mean():.3f})")
