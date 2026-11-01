"""Train the dense SonarVoxNet and score every epoch on val with the official evaluator.

    python -m diver3d.train --config configs/base.yaml --name base --gpu 0
    python -m diver3d.train --config configs/base.yaml --name smoke --limit 200 --set train.epochs=1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import dump_config, load_config
from .data import Diver3DDataset, GTSampler, collate
from .evalio import load_gt, official, postprocess, score, server_rotation, write_submission
from .fuse import fuse_frame
from .model import SonarVoxNet, decode, detection_loss

_F = np.diag([1.0, -1.0, 1.0])


def unmirror_boxes(ev, boxes):
    """Map boxes predicted on a y-mirrored cloud back to the original frame."""
    return [ev.Box(center=b.center * np.array([1.0, -1.0, 1.0]), dims=b.dims,
                   R=server_rotation(_F @ b.R @ _F), score=b.score) for b in boxes]


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def run_inference(model, loader, cfg, device, score_thr=None, nms_thr=None, max_dets=None,
                  tta_flip=None, fuse_iou=0.45, return_passes=False, pitch_deg: float = 0.0,
                  scale: float = 1.0):
    """-> token -> list[Box]. With tta_flip, also runs on the y-mirrored cloud and fuses.
    return_passes=True instead returns [pass0_pred, pass1_pred] unfused (for offline fusion tuning)."""
    dcfg = cfg["decode"]
    score_thr = dcfg["score_thr"] if score_thr is None else score_thr
    nms_thr = dcfg["nms_bev_iou"] if nms_thr is None else nms_thr
    max_dets = dcfg["max_dets"] if max_dets is None else max_dets
    tta_flip = dcfg.get("tta_flip", False) if tta_flip is None else tta_flip
    ev = official(cfg["data_root"])
    model.eval()
    pred = {}
    passes_out = [{}, {}]
    Mp = None
    if pitch_deg:                      # pitch the whole cloud about the sonar y axis (elevation shift)
        from .geometry import rot_y
        Mp = rot_y(np.radians(pitch_deg))
        Mp_t = torch.from_numpy(Mp.astype(np.float32)).to(device)
    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        if Mp is not None:
            pts = pts.clone()
            pts[:, 1:4] = pts[:, 1:4] @ Mp_t.T
        if scale != 1.0:                   # radial scale TTA: scale the cloud, unscale the boxes
            pts = pts.clone()
            pts[:, 1:4] *= scale
        B = len(batch["tokens"])
        passes = [pts]
        if tta_flip:
            m = pts.clone()
            m[:, 2] *= -1
            if int(cfg.get("data", {}).get("local_feats", 0)) > 0:    # mirror the local-structure channels too
                base = 5 + (1 if cfg.get("data", {}).get("n_prev", 0) > 0 else 0)
                for col in (base + 1, base + 6, base + 8):            # mean-offset y, cov xy, cov yz change sign under y -> -y
                    m[:, col] *= -1
            passes.append(m)
        per_pass = []
        for pi, p in enumerate(passes):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"].get("amp", False)):
                out = model(p, B)
            out = {k: v.float() for k, v in out.items()}
            decs = decode(out, cfg, K=dcfg.get("topk", 100))
            boxes = [postprocess(ev, dec, score_thr, nms_thr, max_dets) for dec in decs]
            if pi == 1:
                boxes = [unmirror_boxes(ev, bl) for bl in boxes]
            if scale != 1.0:
                boxes = [[ev.Box(center=b.center / scale, dims=b.dims / scale, R=b.R, score=b.score) for b in bl] for bl in boxes]
            if Mp is not None:         # undo the pitch
                boxes = [[ev.Box(center=Mp.T @ b.center, dims=b.dims, R=server_rotation(Mp.T @ b.R), score=b.score)
                          for b in bl] for bl in boxes]
            per_pass.append(boxes)
        for bi, tok in enumerate(batch["tokens"]):
            if return_passes:
                for pi, pp in enumerate(per_pass):
                    passes_out[pi][tok] = pp[bi]
            elif len(per_pass) == 1:
                pred[tok] = per_pass[0][bi]
            else:
                pred[tok] = fuse_frame(ev, [pp[bi] for pp in per_pass], fuse_iou)[:max_dets]
    if return_passes:
        return passes_out[:len(passes)]
    return pred


def evaluate_split(model, loader, cfg, device, gt) -> dict:
    pred = run_inference(model, loader, cfg, device)
    return score(cfg["data_root"], gt, pred), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=5e-4")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N frames per split (smoke test)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    open(os.path.join(run_dir, "config.yaml"), "w").write(dump_config(cfg))
    seed_all(cfg["train"]["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{args.gpu}")
    root = cfg["data_root"]

    t0 = time.time()
    excl = cfg.get("data", {}).get("exclude_scenes")       # for out-of-fold detector outputs
    train_ds = Diver3DDataset(root, "train", cfg, training=True, limit=args.limit, exclude_scenes=excl)
    if excl:
        print(f"excluding {len(excl)} train scenes: {excl}")
    if cfg["aug"].get("enable", True) and cfg["aug"].get("gt_sample", {}).get("enable", False):
        train_ds.sampler = GTSampler(train_ds.frames, cfg["aug"]["gt_sample"], cfg["grid"], train_ds.n_prev)
        print(f"GT-sampling bank: {len(train_ds.sampler)} diver instances")
    val_ds = Diver3DDataset(root, "val", cfg, training=False, limit=args.limit)
    gt_val = load_gt(root, "val")
    if args.limit:
        gt_val = {t: gt_val[t] for t in val_ds.frames.tokens}
    print(f"loaded train={len(train_ds)} val={len(val_ds)} frames in {time.time() - t0:.0f}s")

    tcfg = cfg["train"]
    sw = cfg.get("data", {}).get("scene_weights")          # {scene_id: weight}: oversample scenes
    if sw:
        from .data import SceneWeightedSampler
        sampler = SceneWeightedSampler(train_ds.frames.tokens, sw, tcfg.get("seed", 0))
        print(f"scene-weighted sampling: {sw}")
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler, drop_last=True,
                                  num_workers=tcfg["workers"], collate_fn=collate, pin_memory=True,
                                  persistent_workers=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                                  num_workers=tcfg["workers"], collate_fn=collate, pin_memory=True,
                                  persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4,
                            collate_fn=collate, pin_memory=True)

    model = SonarVoxNet(cfg).to(device)
    ema_decay = tcfg.get("ema", 0.0)
    ema = None
    if ema_decay > 0:                      # weight EMA, evaluated/saved in place of the raw weights
        import copy
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    steps = tcfg["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=max(steps, 1),
                                                pct_start=tcfg.get("pct_start", 0.3),
                                                div_factor=10.0, final_div_factor=100.0)
    start_epoch, best = 0, {"SDS": -1, "AP3D@0.35": -1}
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("raw_model", ck["model"]))
        if ema is not None:
            ema.load_state_dict(ck.get("ema", ck["model"]))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", best)

    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    for epoch in range(start_epoch, tcfg["epochs"]):
        train_ds.set_epoch(epoch)
        model.train()
        te = time.time()
        meter = {"loss": 0.0, "hm": 0.0, "geo": 0.0, "rot": 0.0, "n": 0}
        for it, batch in enumerate(train_loader):
            pts = batch["points"].to(device, non_blocking=True)
            tg = {k: batch[k].to(device, non_blocking=True) for k in ("hm", "reg", "inds", "mask")}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.get("amp", False)):
                out = model(pts, len(batch["tokens"]))
            out = {k: v.float() for k, v in out.items()}
            loss, parts = detection_loss(out, tg, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_grad"])
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            meter["loss"] += loss.item()
            for k in ("hm", "geo", "rot"):
                meter[k] += parts[k]
            meter["corner"] = meter.get("corner", 0.0) + parts.get("corner", 0.0)
            meter["iou"] = meter.get("iou", 0.0) + parts.get("iou", 0.0)
            meter["n"] += 1
            if (it + 1) % tcfg.get("log_every", 100) == 0:
                n = meter["n"]
                print(f"ep {epoch} it {it + 1}/{len(train_loader)} loss {meter['loss'] / n:.3f} "
                      f"hm {meter['hm'] / n:.3f} geo {meter['geo'] / n:.3f} rot {meter['rot'] / n:.3f} corner {meter.get('corner', 0.0) / n:.3f} iou {meter.get('iou', 0.0) / n:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} gt_target {train_ds.gt_sample_target()} "
                      f"{(time.time() - te) / (it + 1):.2f}s/it", flush=True)
        n = max(meter["n"], 1)
        train_stats = {k: meter[k] / n for k in ("loss", "hm", "geo", "rot")}

        tv = time.time()
        metrics, _ = evaluate_split(ema if ema is not None else model, val_loader, cfg, device, gt_val)
        rec = {"epoch": epoch, "train": train_stats, "val": {k: v for k, v in metrics.items() if k != "_meta"},
               "n_pred": metrics["_meta"]["num_pred_boxes"], "train_s": round(tv - te), "val_s": round(time.time() - tv)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"== ep {epoch} val SDS {metrics['SDS']:.4f} mAP3D {metrics['mAP3D']:.4f} "
              f"AP35 {metrics['AP3D@0.35']:.4f} AP50 {metrics['AP3D@0.50']:.4f} ATE {metrics['ATE_m']:.3f} "
              f"ASE {metrics['ASE']:.3f} AOE3D {metrics['AOE3D_deg']:.1f} TILT {metrics['TILT_deg']:.1f} "
              f"preds {rec['n_pred']} | train {rec['train_s']}s val {rec['val_s']}s", flush=True)

        ck = {"model": (ema if ema is not None else model).state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": epoch, "cfg": cfg, "val": rec["val"], "best": best}
        if ema is not None:
            ck["raw_model"] = model.state_dict()
            ck["ema"] = ema.state_dict()
        torch.save(ck, os.path.join(run_dir, "last.pt"))
        if epoch >= int(tcfg.get("keep_from_frac", 0.7) * tcfg["epochs"]):   # snapshots for ensembling
            torch.save({k: v for k, v in ck.items() if k not in ("opt", "sched")},
                       os.path.join(run_dir, f"ep{epoch:02d}.pt"))
        # only select checkpoints from epochs after GT-sampling has stopped pasting (if it stops),
        # otherwise the chosen model has never seen a truly empty frame
        g = cfg["aug"].get("gt_sample", {})
        if cfg["aug"].get("enable", True) and g.get("enable", False) and g.get("floor", 1) == 0:
            stop = int(g.get("anneal_start_frac", 0.65) * tcfg["epochs"]) + g.get("target_max", 4) - 1
            selectable = epoch >= stop
        else:
            selectable = True
        for key, fname in (("SDS", "best_sds.pt"), ("AP3D@0.35", "best_ap35.pt")):
            if selectable and metrics[key] > best[key]:
                best[key] = metrics[key]
                torch.save(ck, os.path.join(run_dir, fname))
    print("best:", best)


if __name__ == "__main__":
    main()
