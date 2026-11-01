"""Post-hoc checkpoint averaging (SWA-style) over the saved per-epoch snapshots of a run.

    python -m diver3d.swa --run runs/vz025 --epochs 15 16 17 18 19 [--bn last|recompute] [--tta]
Writes runs/<run>/swa_<e0>-<e1>.pt (same format as the other checkpoints) and scores it on val.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .data import Diver3DDataset, collate
from .evalio import load_gt, score
from .model import SonarVoxNet
from .train import run_inference


@torch.no_grad()
def recompute_bn(model, loader, device, n_batches=200):
    """Re-estimate BatchNorm running stats with the averaged weights (cumulative average)."""
    moms = {}
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            moms[m] = m.momentum
            m.momentum = None
    model.train()
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        model(b["points"].to(device), len(b["tokens"]))
    for m, mom in moms.items():
        m.momentum = mom
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--bn", default="last", choices=["last", "recompute"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    cks = [torch.load(os.path.join(args.run, f"ep{e:02d}.pt"), map_location=device, weights_only=False) for e in args.epochs]
    cfg = cks[-1]["cfg"]
    avg = {}
    for k, v in cks[-1]["model"].items():
        if v.dtype.is_floating_point and "running" not in k:
            avg[k] = sum(c["model"][k].float() for c in cks) / len(cks)
        else:
            avg[k] = v                           # BN running stats / counters from the last snapshot
    model = SonarVoxNet(cfg).to(device)
    model.load_state_dict(avg)
    model.eval()
    root = cfg["data_root"]
    if args.bn == "recompute":
        tr = Diver3DDataset(root, "train", cfg, training=False, limit=6400)
        recompute_bn(model, DataLoader(tr, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate), device)
    val = Diver3DDataset(root, "val", cfg, training=False)
    loader = DataLoader(val, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
    pred = run_inference(model, loader, cfg, device, score_thr=0.05, tta_flip=args.tta)
    m = score(root, load_gt(root, "val"), pred)
    tag = f"swa_{args.epochs[0]:02d}-{args.epochs[-1]:02d}" + ("_bnr" if args.bn == "recompute" else "")
    print(f"{args.run} {tag} tta={args.tta}: SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} "
          f"AP50 {m['AP3D@0.50']:.4f} ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f}", flush=True)
    out = os.path.join(args.run, tag + ".pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": args.epochs[-1],
                "val": {k: v for k, v in m.items() if k != "_meta"}}, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
