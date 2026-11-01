"""Dump-based test fusion mirroring diver3d.fuse_tune._run: per-pass BEV NMS, WBF (avg score over all
passes), cut 0.05, track refinement (1/5/8), then the official validator.

    python analysis/fuse_test.py --names <member dump names...> --nms 0.3 --fuse_iou 0.30 --out submissions/<tag>_test.jsonl.gz
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

sys.path.insert(0, "/raid/scratch/dzimmerman2021/Diver3D")
from diver3d.evalio import nms_bev, official, write_submission
from diver3d.fuse import fuse_frame
from diver3d.refine import refine

ROOT = "data/diver3d_v2_extracted/diver3d_v2"
ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--members", default="submissions/test/members")
ap.add_argument("--split", default="test")
ap.add_argument("--out", required=True)
ap.add_argument("--nms", type=float, default=0.1)
ap.add_argument("--fuse_iou", type=float, default=0.30)
ap.add_argument("--pre_thr", type=float, default=0.01)
ap.add_argument("--procs", type=int, default=16)
a = ap.parse_args()
ev = official(ROOT)
_M = None


def _init():
    global _M
    _M = [ev.load_boxes(os.path.join(a.members, f"{n}_p{p}_{a.split}.jsonl.gz"), require_score=True) for n in a.names for p in (0, 1)]


def _fuse(toks):
    out = {}
    for tok in toks:
        lists = []
        for m in _M:
            bs = sorted([b for b in m.get(tok, []) if b.score >= a.pre_thr], key=lambda b: -b.score)
            if a.nms < 1.0:
                bs = nms_bev(ev, bs, a.nms)
            lists.append(bs[:100])
        fused = fuse_frame(ev, lists, a.fuse_iou, len(lists), "avg")
        out[tok] = [b for b in fused if b.score >= 0.05][:100]
    return out


if __name__ == "__main__":
    if a.split == "test":
        tokens = [l.strip() for l in open(os.path.join(ROOT, "annotations", "test_tokens.txt")) if l.strip()]
    else:
        from diver3d.evalio import load_gt
        tokens = sorted(load_gt(ROOT, a.split))
    chunks = [tokens[i::a.procs] for i in range(a.procs)]
    out = {}
    with Pool(a.procs, initializer=_init) as pool:
        for part in pool.imap_unordered(_fuse, chunks):
            out.update(part)
    print(f"fused {len(out)} frames from {2 * len(a.names)} passes (nms {a.nms}, fuse {a.fuse_iou})", flush=True)
    out = refine(ev, {t: out[t] for t in tokens}, windows=(1, 5, 8))
    write_submission(out, a.out)
    print("wrote", a.out, sum(len(v) for v in out.values()), "boxes", flush=True)
    if a.split == "test":
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "validate_submission.py"), "--pred", a.out,
                        "--tokens", os.path.join(ROOT, "annotations", "test_tokens.txt")], check=False)
    else:
        from diver3d.evalio import load_gt
        r = ev.evaluate(load_gt(ROOT, a.split), out, verbose=False)
        print(f"{a.split}: SDS {r['SDS']:.4f} AP35 {r['AP3D@0.35']:.4f} APBEV35 {r['APBEV@0.35']:.4f} AP50 {r['AP3D@0.50']:.4f} ASE {r['ASE']:.3f} AOE3D {r['AOE3D_deg']:.1f}")
