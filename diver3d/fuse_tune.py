"""Offline sweep of NMS / fusion / scoring settings over dumped per-pass member predictions.

    python -m diver3d.fuse_tune --members submissions/val/members --names vz025 vz025_s1 vz025_mf2
"""
from __future__ import annotations

import argparse
import itertools
import os
from multiprocessing import Pool

import numpy as np

from .evalio import load_gt, nms_bev, official, score
from .fuse import fuse_frame
from .refine import refine

ROOT = "data/diver3d_v2_extracted/diver3d_v2"
_EV = None
_MEMBERS = None
_GT = None


def _init(members_dir, names, tta):
    global _EV, _MEMBERS, _GT
    _EV = official(ROOT)
    _GT = load_gt(ROOT, "val")
    _MEMBERS = []
    for n in names:
        for p in ((0, 1) if tta else (0,)):
            _MEMBERS.append(_EV.load_boxes(os.path.join(members_dir, f"{n}_p{p}_val.jsonl.gz"), require_score=True))


def _run(cfg):
    nms, fiou, mode, beta, pre_thr, refine_on = cfg[:6]
    out_path = cfg[6] if len(cfg) > 6 else None
    ev = _EV
    out = {}
    for tok in _GT:
        lists = []
        for m in _MEMBERS:
            bs = [b for b in m.get(tok, []) if b.score >= pre_thr]
            bs.sort(key=lambda b: -b.score)
            if nms < 1.0:
                bs = nms_bev(ev, bs, nms)
            lists.append(bs[:100])
        fused = fuse_frame(ev, lists, fiou, len(lists), "max" if mode == "max" else "avg")
        if mode == "present":  # mean over members that found it, times (n_found/n)^beta
            fused2 = []
            for cl_box in fused:
                pass
        out[tok] = [b for b in fused if b.score >= 0.05][:100]
    if refine_on:
        out = refine(ev, out, windows=(1, 5, 8))
    if out_path:
        from .evalio import write_submission
        write_submission(out, out_path)
    m = score(ROOT, _GT, out)
    return cfg[:6], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="submissions/val/members")
    ap.add_argument("--names", nargs="+", default=["vz025", "vz025_s1", "vz025_mf2"])
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--nms", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--fuse_ious", type=float, nargs="+", default=[0.3, 0.45, 0.6])
    ap.add_argument("--modes", nargs="+", default=["avg", "max"])
    ap.add_argument("--pre_thr", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out", default=None, help="write the fused+refined val predictions of the (single) config here")
    ap.add_argument("--no_refine", action="store_true")
    args = ap.parse_args()
    grid = list(itertools.product(args.nms, args.fuse_ious, args.modes, (1.0,), args.pre_thr, (not args.no_refine,)))
    if args.out:
        grid = [g + (args.out,) for g in grid]
    with Pool(args.procs, initializer=_init, initargs=(args.members, args.names, not args.no_tta)) as pool:
        for cfg, m in pool.imap_unordered(_run, grid):
            nms, fiou, mode, beta, pre_thr, ref = cfg
            print(f"nms {nms:.1f} fuse_iou {fiou:.2f} score {mode:3s} pre_thr {pre_thr:.2f} refine {ref}: "
                  f"SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} AP50 {m['AP3D@0.50']:.4f} "
                  f"ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f} preds {m['_meta']['num_pred_boxes']}", flush=True)


if __name__ == "__main__":
    main()
