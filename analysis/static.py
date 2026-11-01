"""Per-recording causal background on the beam grid: is a detection's support 'static' (its returns match
the median range of the same beams over the previous K frames)? TP vs FP distributions on val, and a
rescoring sweep s' = s * (1 - lam * static_frac).

    python analysis/static.py submissions/val/ens3z_ref158_val.jsonl.gz [--K=15] [--split=val]
"""
import sys
sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
from collections import defaultdict, deque
import numpy as np
from diver3d.data import BEAM, Frames
from diver3d.evalio import load_gt, official
from diver3d.geometry import points_in_box_mask

root = "data/diver3d_v2_extracted/diver3d_v2"
ev = official(root)
pred_file = [a for a in sys.argv[1:] if not a.startswith("--")][0]
K = int([a.split("=")[1] for a in sys.argv if a.startswith("--K=")][0]) if any(a.startswith("--K=") for a in sys.argv) else 15
TOL = 0.15
split = "val"
gt = load_gt(root, split)
pred = ev.load_boxes(pred_file, require_score=True)
frames = Frames(root, split, preload=False)


def beam_uv(xyz):
    r = np.linalg.norm(xyz, axis=1)
    bear = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    elev = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(r, 1e-6), -1, 1)))
    u = np.clip(np.round((bear - BEAM["bear_min"]) / BEAM["bear_step"]).astype(int), 0, BEAM["n_bear"] - 1)
    v = np.clip(np.round((elev - BEAM["elev_min"]) / BEAM["elev_step"]).astype(int), 0, BEAM["n_elev"] - 1)
    return u, v, r


# per scene, frames in order; rolling buffer of range images
by_scene = defaultdict(list)
for i, t in enumerate(frames.tokens):
    by_scene[t.split("/")[0]].append((int(t.split("/")[1]), i, t))
det_static = {}   # (tok, i) -> static fraction of the points inside the box
for sc, lst in by_scene.items():
    buf = deque(maxlen=K)
    for f, idx, tok in sorted(lst):
        pts = frames.get_points(idx)
        xyz = pts[:, :3]
        u, v, r = beam_uv(xyz)
        img = np.full((BEAM["n_bear"], BEAM["n_elev"]), np.nan, np.float32)
        img[u, v] = r
        if len(buf) >= max(3, K // 2):
            stack = np.stack(buf)
            with np.errstate(all="ignore"):
                bg = np.nanmedian(stack, 0)
                n_obs = np.sum(~np.isnan(stack), 0)
            bgr = bg[u, v]
            ok = (n_obs[u, v] >= max(3, K // 2)) & ~np.isnan(bgr)
            static_pt = ok & (np.abs(r - bgr) < TOL)
        else:
            static_pt = np.zeros(len(r), bool)
        for i, b in enumerate(pred.get(tok, [])):
            m = points_in_box_mask(xyz, b.center, b.dims * 1.2, b.R)
            n = int(m.sum())
            det_static[(tok, i)] = float(static_pt[m].mean()) if n >= 5 else 0.0
        buf.append(img)

# TP / FP labels (greedy 3D IoU 0.35)
cat = {}
for tok, p in pred.items():
    g = gt[tok]
    if not p:
        continue
    order = np.argsort([-b.score for b in p])
    ps = [p[i] for i in order]
    if g:
        m = ev._greedy_match(ev._iou_matrix(ps, g, ev.iou_3d), 0.35)
        for k, i in enumerate(order):
            cat[(tok, i)] = "TP" if m[k] >= 0 else "FP"
    else:
        for i in range(len(p)):
            cat[(tok, i)] = "FP-empty"
for name in ("TP", "FP", "FP-empty"):
    v = np.array([det_static[k] for k, c in cat.items() if c == name and pred[k[0]][k[1]].score >= 0.3])
    if len(v):
        print(f"{name:9s} n {len(v):5d} (score>=0.3): static frac mean {v.mean():.3f} median {np.median(v):.3f} | share > 0.5: {np.mean(v > 0.5):.3f} | share > 0.8: {np.mean(v > 0.8):.3f}")
print("per scene, confident dets: mean static frac TP / FP(all):")
for sc in sorted(by_scene):
    tp = [det_static[k] for k, c in cat.items() if c == "TP" and k[0].startswith(sc) and pred[k[0]][k[1]].score >= 0.3]
    fp = [det_static[k] for k, c in cat.items() if c != "TP" and k[0].startswith(sc) and pred[k[0]][k[1]].score >= 0.3]
    print(f"  {sc}: TP {np.mean(tp) if tp else float('nan'):.3f} (n {len(tp)}) | FP {np.mean(fp) if fp else float('nan'):.3f} (n {len(fp)})")
base = ev.evaluate(gt, pred, verbose=False)
print(f"\nbase: SDS {base['SDS']:.4f} AP35 {base['AP3D@0.35']:.4f} APBEV35 {base['APBEV@0.35']:.4f}")
for lam in (0.3, 0.6, 1.0):
    p2 = {t: [ev.Box(center=b.center, dims=b.dims, R=b.R, score=max(0.0, b.score * (1 - lam * det_static.get((t, i), 0.0)))) for i, b in enumerate(v)] for t, v in pred.items()}
    p2 = {t: [b for b in v if b.score >= 0.01] for t, v in p2.items()}
    r = ev.evaluate(gt, p2, verbose=False)
    print(f"rescored lam {lam}: SDS {r['SDS']:.4f} ({r['SDS']-base['SDS']:+.4f}) AP35 {r['AP3D@0.35']:.4f} ({r['AP3D@0.35']-base['AP3D@0.35']:+.4f}) APBEV35 {r['APBEV@0.35']:.4f} ({r['APBEV@0.35']-base['APBEV@0.35']:+.4f})", flush=True)
