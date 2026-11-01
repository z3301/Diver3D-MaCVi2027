"""Run a checkpoint on a split, write a submission, and score it (val) or validate it (test).

    python -m diver3d.predict --ckpt runs/base/best_sds.pt --split val
    python -m diver3d.predict --ckpt runs/base/best_sds.pt --split test --out submissions/base.jsonl.gz
    python -m diver3d.predict --ckpt runs/base/best_sds.pt --split val --sweep   # score threshold sweep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from .data import Diver3DDataset, collate
from .evalio import load_gt, official, score, write_submission
from .fuse import fuse_predictions
from .model import SonarVoxNet
from .train import run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, nargs="+", help="one or more checkpoints (ensembled by box fusion)")
    ap.add_argument("--tta", action="store_true", help="flip-y test-time augmentation")
    ap.add_argument("--fuse_iou", type=float, default=0.30)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--score_thr", type=float, default=None)
    ap.add_argument("--nms", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="sweep the score threshold on a labelled split")
    ap.add_argument("--scenes", nargs="*", default=None, help="restrict the split to these scene ids")
    ap.add_argument("--scale", type=float, default=1.0, help="radial scale TTA: multiply xyz by this before inference, divide boxes after")
    ap.add_argument("--pitch", type=float, default=0.0, help="pitch the cloud by this many degrees about y before inference (diagnostic TTA)")
    ap.add_argument("--dump_dir", default=None, help="save each model's raw per-pass predictions (pre-NMS, thr 0.01) here and exit")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    models = []
    for path in args.ckpt:
        ck = torch.load(path, map_location=device, weights_only=False)  # our own checkpoint
        cfg = ck["cfg"]
        m = SonarVoxNet(cfg).to(device)
        missing, unexpected = m.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if not k.startswith("head.iou.")] + list(unexpected)   # iou branch is optional
        assert not bad, bad
        models.append((m, cfg))
        print(f"loaded {path} (epoch {ck['epoch']}, val {json.dumps({k: round(v, 4) for k, v in ck.get('val', {}).items() if isinstance(v, float)})})")
    root = cfg["data_root"]
    ev = official(root)

    def infer(score_thr):
        preds = []
        for m, c in models:   # each model may use a different n_prev, so build its own loader
            ds = Diver3DDataset(root, args.split, c, training=False, only_scenes=args.scenes)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
            preds.append(run_inference(m, loader, c, device, score_thr=score_thr, nms_thr=args.nms,
                                       tta_flip=args.tta, pitch_deg=args.pitch, fuse_iou=args.fuse_iou, scale=args.scale))
        if len(preds) == 1:
            return preds[0]
        fused = fuse_predictions(ev, preds, args.fuse_iou)
        return {t: bs[:cfg["decode"]["max_dets"]] for t, bs in fused.items()}

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        for (m, c), path in zip(models, args.ckpt):
            name = os.path.basename(os.path.dirname(path))
            if args.scale != 1.0:
                name += f"_sc{int(round(args.scale * 100)):03d}"
            ds = Diver3DDataset(root, args.split, c, training=False, only_scenes=args.scenes)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
            passes = run_inference(m, loader, c, device, score_thr=0.01, nms_thr=1.0, max_dets=100,
                                   tta_flip=args.tta, pitch_deg=args.pitch, return_passes=True, scale=args.scale)
            for pi, pp in enumerate(passes):
                out = os.path.join(args.dump_dir, f"{name}_p{pi}_{args.split}.jsonl.gz")
                write_submission(pp, out)
                print("wrote", out, sum(len(v) for v in pp.values()), "boxes")
        return

    if args.sweep:
        gt = load_gt(root, args.split)
        pred = infer(0.01)
        for thr in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            p = {t: [b for b in bs if b.score >= thr] for t, bs in pred.items()}
            m = score(root, gt, p)
            print(f"thr {thr:.2f}: SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} "
                  f"AP50 {m['AP3D@0.50']:.4f} ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f} "
                  f"preds {m['_meta']['num_pred_boxes']}")
        return

    pred = infer(args.score_thr)
    tag = "+".join(os.path.basename(os.path.dirname(p)) for p in args.ckpt) + ("_tta" if args.tta else "")
    out = args.out or os.path.join("submissions", f"{tag}_{args.split}.jsonl.gz")
    write_submission(pred, out)
    print(f"wrote {out} ({sum(len(v) for v in pred.values())} boxes over {len(pred)} frames)")

    if args.split == "test":
        subprocess.run([sys.executable, os.path.join(root, "tools", "validate_submission.py"),
                        "--pred", out, "--tokens", os.path.join(root, "annotations", "test_tokens.txt")], check=False)
    else:
        gt = load_gt(root, args.split)
        if args.scenes:
            gt = {t: v for t, v in gt.items() if t.split("/")[0] in set(args.scenes)}
        m = score(root, gt, pred)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, indent=1))


if __name__ == "__main__":
    main()
