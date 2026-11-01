"""Do our post-processing thresholds kill close diver pairs? On out-of-fold train scenes (labelled, never
seen by the fold detector): found@BEV0.35 / found@3D0.35 for divers with a neighbour < 1.0 m vs isolated
(nearest neighbour >= 1.5 m or alone), and the official metrics on those scenes, under per-pass BEV NMS
thresholds and WBF fusion thresholds.

    python analysis/pairs.py --dumps submissions/train/members/oof_f0 submissions/train/members/oof_f2 \
        --scenes_from configs/oof_folds.json --folds 0 2
"""
import argparse
import json
import sys

sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
import numpy as np
from diver3d.evalio import load_gt, nms_bev, official
from diver3d.fuse import fuse_frame
from diver3d.refine import refine

ROOT = "data/diver3d_v2_extracted/diver3d_v2"
ap = argparse.ArgumentParser()
ap.add_argument("--dumps", nargs="+", required=True, help="dump prefixes, one per fold, e.g. submissions/train/members/oof_f0")
ap.add_argument("--folds", nargs="+", type=int, required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--combos", nargs="*", default=None, help="nms:fuse:refine triples, e.g. 0.3:0.30:1")
ap.add_argument("--save_prefix", default=None, help="write each combo output to <prefix>_nms<n>_fuse<f>[_ref].jsonl.gz")
a = ap.parse_args()
ev = official(ROOT)
gt_all = load_gt(ROOT, a.split)
folds = json.load(open("configs/oof_folds.json"))

passes = {}   # token -> [p0 boxes, p1 boxes]
for prefix, fi in zip(a.dumps, a.folds):
    sc = set(folds[fi])
    p0 = ev.load_boxes(f"{prefix}_p0_{a.split}.jsonl.gz", require_score=True)
    p1 = ev.load_boxes(f"{prefix}_p1_{a.split}.jsonl.gz", require_score=True)
    for t in p0:
        if t.split("/")[0] in sc:
            passes[t] = [p0[t], p1.get(t, [])]
gt = {t: v for t, v in gt_all.items() if t in passes}
print(f"out-of-fold frames {len(gt)}, boxes {sum(len(v) for v in gt.values())}, scenes {sorted({t.split('/')[0] for t in gt})}")

# neighbour class per GT
cls = {}
for t, g in gt.items():
    for j, gb in enumerate(g):
        d = [np.linalg.norm(gb.center[:2] - gg.center[:2]) for k, gg in enumerate(g) if k != j]
        nn = min(d) if d else 99.0
        cls[(t, j)] = "close" if nn < 1.0 else ("mid" if nn < 1.5 else "isolated")
n_cls = {c: sum(1 for v in cls.values() if v == c) for c in ("close", "mid", "isolated")}
print("GT by nearest neighbour:", n_cls)


def build(nms, fuse_iou, use_flip, do_refine):
    out = {}
    for t, (p0, p1) in passes.items():
        lists = []
        for p in ([p0, p1] if use_flip else [p0]):
            bs = sorted([b for b in p if b.score >= 0.01], key=lambda b: -b.score)
            if nms < 1.0:
                bs = nms_bev(ev, bs, nms)
            lists.append(bs[:100])
        if use_flip:
            fused = fuse_frame(ev, lists, fuse_iou, len(lists), "avg")
        else:
            fused = lists[0]
        out[t] = [b for b in fused if b.score >= 0.05][:100]
    if do_refine:
        out = refine(ev, out, windows=(1, 5, 8))
    return out


def report(name, pred):
    found_b, found_3 = {c: [] for c in n_cls}, {c: [] for c in n_cls}
    for t, g in gt.items():
        p = [b for b in pred.get(t, []) if b.score >= 0.2]
        for j, gb in enumerate(g):
            c = cls[(t, j)]
            found_b[c].append(max((ev.iou_bev(b, gb) for b in p), default=0.0) >= 0.35)
            found_3[c].append(max((ev.iou_3d(b, gb) for b in p), default=0.0) >= 0.35)
    r = ev.evaluate(gt, pred, verbose=False)
    print(f"{name:34s} SDS {r['SDS']:.4f} AP35 {r['AP3D@0.35']:.4f} APBEV35 {r['APBEV@0.35']:.4f} dets {sum(len(v) for v in pred.values()):6d} | "
          + " ".join(f"{c} BEV {np.mean(found_b[c]):.3f} 3D {np.mean(found_3[c]):.3f}" for c in ("close", "mid", "isolated")), flush=True)


if a.combos:
    for cmb in a.combos:
        nms, fi, rf = cmb.split(":")
        out = build(float(nms), float(fi), True, rf == "1")
        report(f"flip nms {nms} fuse {fi}{' +refine' if rf == '1' else ''}", out)
        if a.save_prefix:
            from diver3d.evalio import write_submission
            write_submission(out, f"{a.save_prefix}_nms{nms}_fuse{fi}{'_ref' if rf == '1' else ''}.jsonl.gz")
    sys.exit(0)
print("\nsingle pass (p0), per-pass BEV NMS sweep (score >= 0.2 for found@):")
for nms in (0.1, 0.2, 0.3, 0.5, 1.0):
    report(f"p0 nms {nms}", build(nms, 0.3, False, False))
print("\nflip fused (p0 + p1 WBF), as in the pipeline:")
for nms, fi in ((0.1, 0.30), (0.3, 0.30), (0.3, 0.45), (0.5, 0.45), (0.5, 0.60)):
    report(f"flip nms {nms} fuse {fi}", build(nms, fi, True, False))
print("\n+ track refinement (1/5/8):")
for nms, fi in ((0.1, 0.30), (0.3, 0.45), (0.5, 0.60)):
    report(f"flip nms {nms} fuse {fi} +refine", build(nms, fi, True, True))
