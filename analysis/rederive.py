"""Re-derive the Section 4 analyses of the MaCVi team report whose scripts were lost in the 2026-10-24 rollback,
from surviving files only (val GT + submissions/val/ens_mixed7_val.jsonl.gz, the 7-model validation ensemble).

  1. Mixed-effects recording effects on box width (covariates: range, log range, |up| and |along-beam| components
     of the body long axis): partial R^2 of recording given covariates, conditional ICC, recording-effect SDs and
     BLUPs for log labelled width, log predicted width and their log ratio.
  2. Extent oracles: predicted dims replaced by GT per-frame dims, and by the GT track-mean dims.
  3. False-positive oracle: drop detections in GT-empty frames and far from any GT.
  4. Label-to-no-label transitions with a confident detection continuing at the last labelled position.
  5. Median predicted / labelled BEV footprint area of the best-overlap prediction, by range bin.

    python analysis/rederive.py > logs/rederive.log
"""
import json
import sys
import warnings
from collections import defaultdict

sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
import pandas as pd
from diver3d.evalio import load_gt, official

ROOT = "data/diver3d_v2_extracted/diver3d_v2"
PRED = sys.argv[1] if len(sys.argv) > 1 else "submissions/val/ens_mixed7_val.jsonl.gz"
ev = official(ROOT)
gt = load_gt(ROOT, "val")
pred = ev.load_boxes(PRED, require_score=True)
raw = json.load(open(f"{ROOT}/annotations/gt_val.json"))["results"]
inst = {(t, j): r.get("instance_id", -1) for t, recs in raw.items() for j, r in enumerate(recs)}
print(f"prediction file {PRED}; val frames {len(gt)}, boxes {sum(len(v) for v in gt.values())}")
base = ev.evaluate(gt, pred, verbose=False)
print(f"base: SDS {base['SDS']:.4f} AP35 {base['AP3D@0.35']:.4f} APBEV35 {base['APBEV@0.35']:.4f}")

# ---------------------------------------------------------------- per-box table (best 3D-overlap prediction)
rows = []
for tok, g in gt.items():
    p = pred.get(tok, [])
    if not g or not p:
        continue
    M3 = ev._iou_matrix(p, g, ev.iou_3d)
    for j, gb in enumerate(g):
        k = int(np.argmax(M3[:, j]))
        if M3[k, j] <= 0:
            continue
        pb = p[k]
        c = gb.center; r = float(np.linalg.norm(c[:2])); u = np.array([c[0], c[1], 0.0]) / max(r, 1e-6)
        ez = gb.R[:, 2]
        A = np.abs(gb.R.T @ pb.R)
        pd_dims = np.array([pb.dims[int(np.argmax(A[a]))] for a in range(3)])
        foot_p = pb.dims[0] * pb.dims[1] if abs(pb.R[2, 2]) > 0.7 else None
        rows.append(dict(scene=tok.split("/")[0], tok=tok, j=j, track=f"{tok.split('/')[0]}#{inst[(tok, j)]}",
                         r=r, log_r=np.log(r), up=abs(float(ez[2])), along=abs(float(ez @ u)),
                         gt_w=gb.dims[1], pred_w=pd_dims[1], iou3=float(M3[k, j])))
df = pd.DataFrame(rows)
df["lgw"], df["lpw"] = np.log(df.gt_w), np.log(df.pred_w)
df["dw"] = df.lpw - df.lgw
print(f"\n[1] matched boxes (3D IoU > 0): {len(df)} over {df.scene.nunique()} recordings")
import statsmodels.formula.api as smf
cov = "r + log_r + up + along"
for y in ("dw", "lgw", "lpw"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_c = smf.ols(f"{y} ~ {cov}", df).fit().rsquared
        r_cs = smf.ols(f"{y} ~ {cov} + C(scene)", df).fit().rsquared
        m = smf.mixedlm(f"{y} ~ {cov}", df, groups=df["scene"]).fit(reml=True)
    vs, ve = float(m.cov_re.iloc[0, 0]), float(m.scale)
    blup = {k[-4:]: float(v.iloc[0]) for k, v in m.random_effects.items()}
    print(f"  {y:4s}: partial R2 of recording given covariates {(r_cs - r_c) / (1 - r_c):.3f}; conditional ICC {vs / (vs + ve):.3f}; "
          f"recording-effect SD (variance parameter) {np.sqrt(vs):.3f}, SD of the 7 BLUPs {np.std(list(blup.values()), ddof=1):.3f}")
    print("        BLUPs: " + " ".join(f"{k} {v:+.3f}" for k, v in sorted(blup.items())))
    if y == "lgw": b_g = blup
    if y == "lpw": b_p = blup
ks = sorted(b_g)
print(f"  correlation of labelled vs predicted recording effects across the 7 recordings: r = {np.corrcoef([b_g[k] for k in ks], [b_p[k] for k in ks])[0, 1]:+.2f}")

# ---------------------------------------------------------------- [2] extent oracles
assign = {}
for tok, p in pred.items():
    g = gt.get(tok, [])
    if not p or not g:
        continue
    M = ev._iou_matrix(p, g, ev.iou_bev)
    for i in range(len(p)):
        j = int(np.argmax(M[i]))
        if M[i, j] > 0.1:
            assign[(tok, i)] = j
tmean = defaultdict(list)
for tok, g in gt.items():
    for j, gb in enumerate(g):
        tmean[f"{tok.split('/')[0]}#{inst[(tok, j)]}"].append(gb.dims)
tmean = {k: np.mean(v, 0) for k, v in tmean.items()}
for mode in ("GT per-frame dims", "GT track-mean dims"):
    out = {}
    for tok, p in pred.items():
        bs = []
        for i, b in enumerate(p):
            d = b.dims
            if (tok, i) in assign:
                j = assign[(tok, i)]
                d = gt[tok][j].dims if mode.startswith("GT per") else tmean[f"{tok.split('/')[0]}#{inst[(tok, j)]}"]
            bs.append(ev.Box(center=b.center, dims=d, R=b.R, score=b.score))
        out[tok] = bs
    r = ev.evaluate(gt, out, verbose=False)
    print(f"\n[2] {mode}: SDS {r['SDS']:.4f} ({r['SDS'] - base['SDS']:+.4f})" if mode.startswith("GT per") else f"[2] {mode}: SDS {r['SDS']:.4f} ({r['SDS'] - base['SDS']:+.4f})")

# ---------------------------------------------------------------- [3] false-positive oracle
keep = {}
for tok, p in pred.items():
    g = gt[tok]
    if not g:
        keep[tok] = []
        continue
    M3 = ev._iou_matrix(p, g, ev.iou_3d) if p else None
    Mb = ev._iou_matrix(p, g, ev.iou_bev) if p else None
    keep[tok] = [b for i, b in enumerate(p) if M3[i].max() > 0.1 or Mb[i].max() > 0.2]
r = ev.evaluate(gt, keep, verbose=False)
print(f"\n[3] drop dets in GT-empty frames and far from any GT: AP35 {r['AP3D@0.35']:.4f} ({r['AP3D@0.35'] - base['AP3D@0.35']:+.4f}), "
      f"APBEV35 {r['APBEV@0.35'] - base['APBEV@0.35']:+.4f}, SDS {r['SDS'] - base['SDS']:+.4f}")
print(f"    GT-empty frames: val {sum(1 for v in gt.values() if not v)} of {len(gt)}")

# ---------------------------------------------------------------- [4] label-to-no-label transitions
scenes = defaultdict(dict)
for t in gt:
    scenes[t.split("/")[0]][int(t.split("/")[1])] = t
n_tr = n_cont = 0; longest = {}
for sc, fr in scenes.items():
    fs = sorted(fr)
    gap = run = 0; best = 0
    for f in fs:
        if not gt[fr[f]]:
            run += 1; best = max(best, run)
        else:
            run = 0
    longest[sc] = best
    for f in fs:
        t, tn = fr[f], fr.get(f + 1)
        if not gt[t] or tn is None or gt[tn]:
            continue
        for gb in gt[t]:
            c = gb.center[:2]
            vals = []
            for d in range(1, 6):
                tt = fr.get(f + d)
                if tt is None or gt[tt]:
                    continue
                vals.append(max((b.score for b in pred.get(tt, []) if np.linalg.norm(b.center[:2] - c) < 0.5), default=0.0))
            n_tr += 1
            n_cont += (len(vals) >= 3 and np.mean(vals) >= 0.3)
print(f"\n[4] label-to-no-label transitions (per labelled diver): {n_tr}; with a confident detection (mean score >= 0.3 over the next 5 unlabelled frames, within 0.5 m): {n_cont}")
print("    longest unlabelled run per recording: " + " ".join(f"{k[-4:]} {v}" for k, v in sorted(longest.items())))

# ---------------------------------------------------------------- [5] footprint area ratio by range
fr_rows = []
for tok, g in gt.items():
    p = pred.get(tok, [])
    if not g or not p:
        continue
    Mb = ev._iou_matrix(p, g, ev.iou_bev)
    for j, gb in enumerate(g):
        k = int(np.argmax(Mb[:, j]))
        if Mb[k, j] <= 0:
            continue
        def area(b):
            return float(ev._hull_area(ev._bev_footprint(b))) if hasattr(ev, "_hull_area") else float(np.prod(b.dims[:2]))
        fr_rows.append((np.linalg.norm(gb.center[:2]), area(p[k]) / max(area(gb), 1e-6)))
F = np.array(fr_rows)
for lo, hi in ((0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99)):
    s = F[(F[:, 0] >= lo) & (F[:, 0] < hi), 1]
    print(f"[5] range {lo}-{hi} m: n {len(s):5d}, median predicted/labelled BEV footprint area {np.median(s):.2f}")
