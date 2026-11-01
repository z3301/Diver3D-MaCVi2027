"""Second-stage box refinement: a small point network that takes the sonar returns around a
proposal box (in the proposal's body frame) and regresses residual centre / extents / rotation.

Train on the train split with jittered GT boxes as proposals (no detector outputs needed):

    python -m diver3d.stage2 train --name s2a --gpu 1 [--set key=val ...]

Apply to a prediction file (fused ensemble, before or after temporal smoothing):

    python -m diver3d.stage2 apply --ckpt runs/s2a/best.pt --pred submissions/val/x.jsonl.gz --split val --out y.jsonl.gz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data import Frames
from .evalio import load_gt, official, score, server_rotation, write_submission
from .geometry import CORNER_TEMPLATE, gram_schmidt_6d_torch, rot_x, rot_y, rot_z

S2_CFG = {
    "n_points": 256,
    "enlarge": 1.6,            # proposal box scaled by this factor to collect points
    "n_prev": 0,
    "jit_center": 0.30,        # m, uniform per axis (train proposals)
    "jit_scale": [0.75, 1.3],
    "jit_rot_deg": 20.0,
    "p_flip_yaw": 0.5,         # random Rz(pi) on the proposal: the net must be symmetry-robust
    "epochs": 12,
    "batch_size": 64,
    "lr": 1e-3,
    "corner_weight": 1.0,
}


def crop_points(pts: np.ndarray, center, dims, R, enlarge: float, n_points: int, rng=None):
    """Points inside the enlarged proposal, expressed in the proposal body frame, scaled by the
    proposal extents. Returns (n_points, 5): x, y, z (body, normalised), intensity, inside-flag."""
    local = (pts[:, :3] - center) @ R                       # body coords
    half = np.asarray(dims) / 2.0
    m = np.all(np.abs(local) <= half * enlarge, axis=1)
    loc = local[m]
    inten = pts[m, 3]
    inside = np.all(np.abs(loc) <= half, axis=1).astype(np.float32)
    feats = np.concatenate([loc / np.maximum(half, 0.05), inten[:, None], inside[:, None]], 1).astype(np.float32)
    n = len(feats)
    if n == 0:
        return np.zeros((n_points, 5), np.float32), 0
    rng = rng or np.random
    idx = rng.choice(n, n_points, replace=n < n_points)
    return feats[idx], n


class ProposalDataset(Dataset):
    def __init__(self, root, split, cfg, s2, training, limit=None, proposal_files=None, min_iou=0.1, mix_jitter=0.5):
        self.frames = Frames(root, split, limit=limit)
        self.s2 = s2
        self.training = training
        self.items = [(i, j, None) for i in range(len(self.frames)) for j in range(len(self.frames.get_boxes(i)[0]))]
        self.real = []            # (frame idx, gt idx, proposal (c, d, R)) from detector outputs
        if proposal_files:
            ev = official(root)
            tok2i = {t: k for k, t in enumerate(self.frames.tokens)}
            for f in proposal_files:
                pred = ev.load_boxes(f, require_score=True)
                for tok, boxes in pred.items():
                    if tok not in tok2i:
                        continue
                    i = tok2i[tok]
                    c, d, R, _ = self.frames.get_boxes(i)
                    if len(c) == 0 or not boxes:
                        continue
                    gts = [ev.Box(center=c[j], dims=d[j], R=R[j]) for j in range(len(c))]
                    M = ev._iou_matrix(boxes, gts, ev.iou_3d)
                    for bi, b in enumerate(boxes):
                        if b.score < 0.1:
                            continue
                        j = int(M[bi].argmax())
                        if M[bi, j] >= min_iou:
                            self.real.append((i, j, (b.center.copy(), b.dims.copy(), b.R.copy())))
            print(f"real proposals from detector outputs: {len(self.real)} (jittered GT items: {len(self.items)})", flush=True)
            self.mix_jitter = mix_jitter
            if training:
                n_j = int(len(self.real) * mix_jitter / max(1e-6, 1 - mix_jitter))
                rng = np.random.default_rng(0)
                jit = [self.items[k] for k in rng.choice(len(self.items), min(n_j, len(self.items)), replace=False)]
                self.items = self.real + jit
            else:
                self.items = self.real
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, j, prop = self.items[k]
        rng = np.random.default_rng([self.epoch, k])
        pts = self.frames.get_stack(i, self.s2["n_prev"])
        c, d, R, _ = self.frames.get_boxes(i)
        c, d, R = c[j], d[j], R[j]
        s2 = self.s2
        if prop is not None:                                   # real detector proposal
            pc, pd, pR = prop
            if rng.random() < s2["p_flip_yaw"]:
                pR = pR @ np.diag([-1.0, -1.0, 1.0])
        else:                                                  # jittered GT proposal
            pc = c + rng.uniform(-s2["jit_center"], s2["jit_center"], 3)
            pd = d * rng.uniform(s2["jit_scale"][0], s2["jit_scale"][1], 3)
            a = np.radians(s2["jit_rot_deg"])
            pR = R @ rot_z(rng.uniform(-a, a)) @ rot_y(rng.uniform(-a, a)) @ rot_x(rng.uniform(-a, a))
            if rng.random() < s2["p_flip_yaw"]:
                pR = pR @ np.diag([-1.0, -1.0, 1.0])
        feats, n = crop_points(pts, pc, pd, pR, s2["enlarge"], s2["n_points"], rng)
        # targets in the proposal frame: centre offset (normalised by proposal half-extents), log scale, rotation residual (6D of R_prop^T R_gt), symmetry-folded
        dc = ((c - pc) @ pR) / (pd / 2.0)
        dl = np.log(d / pd)
        Rrel = pR.T @ R
        Rrel_alt = Rrel @ np.diag([-1.0, -1.0, 1.0])
        if np.trace(Rrel_alt) > np.trace(Rrel):
            Rrel = Rrel_alt
        r6 = np.concatenate([Rrel[:, 0], Rrel[:, 1]])
        tgt = np.concatenate([dc, dl, r6]).astype(np.float32)
        prop = np.concatenate([pc, pd, pR.reshape(-1)]).astype(np.float32)
        gtb = np.concatenate([c, d, R.reshape(-1)]).astype(np.float32)
        return feats, tgt, prop, gtb, np.float32(n > 0)


class RefineNet(nn.Module):
    def __init__(self, in_ch=5, ch=(64, 128, 256), out=12):
        super().__init__()
        layers, prev = [], in_ch
        for c in ch:
            layers += [nn.Conv1d(prev, c, 1), nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            prev = c
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(prev * 2, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
                                  nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, out))
        with torch.no_grad():                              # start at identity residual
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()
            self.head[-1].bias[6] = 1.0                    # 6D rotation identity: ex=(1,0,0), ey=(0,1,0)
            self.head[-1].bias[10] = 1.0

    def forward(self, x):                                  # x (B, N, 5)
        f = self.mlp(x.transpose(1, 2))                    # (B, C, N)
        g = torch.cat([f.max(2).values, f.mean(2)], 1)
        return self.head(g)


def apply_residual(prop: torch.Tensor, res: torch.Tensor):
    """prop (B, 15) = c(3) d(3) R(9) ; res (B, 12) -> refined c, d, R tensors."""
    pc, pd, pR = prop[:, :3], prop[:, 3:6], prop[:, 6:].view(-1, 3, 3)
    c = pc + torch.einsum("bij,bj->bi", pR, res[:, :3] * pd / 2.0)
    d = pd * res[:, 3:6].clamp(-1.0, 1.0).exp()
    R = pR @ gram_schmidt_6d_torch(res[:, 6:12])
    return c, d, R


def corners_t(c, d, R):
    tmpl = torch.tensor(CORNER_TEMPLATE, dtype=c.dtype, device=c.device)
    return torch.einsum("bij,bcj->bci", R, tmpl[None] * d[:, None, :]) + c[:, None, :]


def s2_loss(res, tgt, prop, gtb, valid, w_corner):
    l1 = ((res - tgt).abs().sum(1) * valid).sum() / valid.sum().clamp(min=1)
    c, d, R = apply_residual(prop, res)
    gc, gd, gR = gtb[:, :3], gtb[:, 3:6], gtb[:, 6:].view(-1, 3, 3)
    pc_, gc_ = corners_t(c, d, R), corners_t(gc, gd, gR)
    perm = [2, 3, 0, 1, 6, 7, 4, 5]
    cl = torch.minimum((pc_ - gc_).abs().mean((1, 2)), (pc_[:, perm] - gc_).abs().mean((1, 2)))
    corner = (cl * valid).sum() / valid.sum().clamp(min=1)
    return l1 + w_corner * corner, l1.item(), corner.item()


def train(args):
    from .config import load_config
    base = load_config("configs/base.yaml")
    s2 = dict(S2_CFG)
    for ov in args.set:
        k, _, v = ov.partition("=")
        import yaml
        s2[k] = yaml.safe_load(v)
    root = base["data_root"]
    device = torch.device(f"cuda:{args.gpu}")
    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(s2, open(os.path.join(run_dir, "s2.json"), "w"), indent=1)
    tr = ProposalDataset(root, "train", base, s2, True, limit=args.limit, proposal_files=args.proposals,
                         mix_jitter=s2.get("mix_jitter", 0.3))
    va = ProposalDataset(root, "val", base, s2, False, limit=args.limit,
                         proposal_files=[args.val_proposals] if args.val_proposals else None)
    print(f"proposals: train {len(tr)} val {len(va)}", flush=True)
    tl = DataLoader(tr, batch_size=s2["batch_size"], shuffle=True, num_workers=8, drop_last=True)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=4)
    net = RefineNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=s2["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s2["lr"], total_steps=s2["epochs"] * len(tl), pct_start=0.2)
    best = 1e9
    for ep in range(s2["epochs"]):
        tr.epoch = ep
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for feats, tgt, prop, gtb, valid in tl:
            feats, tgt, prop, gtb, valid = (x.to(device) for x in (feats, tgt, prop, gtb, valid))
            res = net(feats)
            loss, l1, cl = s2_loss(res, tgt, prop, gtb, valid, s2["corner_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        net.eval()
        vt, vn, vc0, vc1 = 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for feats, tgt, prop, gtb, valid in vl:
                feats, tgt, prop, gtb, valid = (x.to(device) for x in (feats, tgt, prop, gtb, valid))
                res = net(feats)
                loss, l1, cl = s2_loss(res, tgt, prop, gtb, valid, s2["corner_weight"])
                # corner error of the raw proposal, for reference
                zero = torch.zeros_like(res)
                zero[:, 6] = 1.0
                zero[:, 10] = 1.0
                _, _, cl0 = s2_loss(zero, tgt, prop, gtb, valid, s2["corner_weight"])
                vt += loss.item()
                vc0 += cl0
                vc1 += cl
                vn += 1
        print(f"ep {ep} train loss {tot / n:.4f} | val loss {vt / vn:.4f} corner err proposal {vc0 / vn:.4f} -> refined {vc1 / vn:.4f} | {time.time() - t0:.0f}s", flush=True)
        if vt / vn < best:
            best = vt / vn
            torch.save({"model": net.state_dict(), "s2": s2}, os.path.join(run_dir, "best.pt"))
    torch.save({"model": net.state_dict(), "s2": s2}, os.path.join(run_dir, "last.pt"))


@torch.no_grad()
def apply(args):
    from .config import load_config
    base = load_config("configs/base.yaml")
    root = base["data_root"]
    device = torch.device(f"cuda:{args.gpu}")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    s2 = ck["s2"]
    net = RefineNet().to(device).eval()
    net.load_state_dict(ck["model"])
    ev = official(root)
    pred = ev.load_boxes(args.pred, require_score=True)
    frames = Frames(root, args.split, preload=False)
    tok2i = {t: i for i, t in enumerate(frames.tokens)}
    out = {}
    cache = {}
    rng = np.random.default_rng(0)
    for tok, boxes in pred.items():
        if not boxes:
            out[tok] = []
            continue
        i = tok2i[tok]
        if i not in cache:
            cache = {i: frames.get_stack(i, s2["n_prev"])}
        pts = cache[i]
        feats, props, keep = [], [], []
        for b in boxes:
            f, n = crop_points(pts, b.center, b.dims, b.R, s2["enlarge"], s2["n_points"], rng)
            feats.append(f)
            props.append(np.concatenate([b.center, b.dims, b.R.reshape(-1)]))
            keep.append(n >= args.min_points)
        res = net(torch.from_numpy(np.stack(feats)).to(device))
        if args.n_iter > 1:                                  # iterative refinement
            for _ in range(args.n_iter - 1):
                c, d, R = apply_residual(torch.from_numpy(np.stack(props)).float().to(device), res)
                props = [np.concatenate([c[k].cpu().numpy(), d[k].cpu().numpy(), R[k].cpu().numpy().reshape(-1)]) for k in range(len(boxes))]
                feats = [crop_points(pts, props[k][:3], props[k][3:6], props[k][6:].reshape(3, 3), s2["enlarge"], s2["n_points"], rng)[0] for k in range(len(boxes))]
                res = net(torch.from_numpy(np.stack(feats)).to(device))
        c, d, R = apply_residual(torch.from_numpy(np.stack(props)).float().to(device), res)
        c, d, R = c.cpu().numpy().astype(np.float64), d.cpu().numpy().astype(np.float64), R.cpu().numpy().astype(np.float64)
        new = []
        for k, b in enumerate(boxes):
            if keep[k]:
                lam = args.blend
                lc = lam if "c" in args.parts else 0.0
                ld = lam if "d" in args.parts else 0.0
                lr = lam if "r" in args.parts else 0.0
                cc = lc * c[k] + (1 - lc) * b.center
                dd = np.exp(ld * np.log(d[k]) + (1 - ld) * np.log(b.dims))
                RR = server_rotation(R[k]) if lr >= 0.999 else (b.R if lr <= 0 else server_rotation(b.R @ _slerp(b.R.T @ R[k], lr)))
                new.append(ev.Box(center=cc, dims=dd, R=RR, score=b.score))
            else:
                new.append(b)
        out[tok] = new
    if args.split != "test":
        for name, p in (("before", pred), ("after", out)):
            m = score(root, load_gt(root, args.split), p)
            print(f"{name:7s} SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} AP50 {m['AP3D@0.50']:.4f} "
                  f"ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f}", flush=True)
    if args.out:
        write_submission(out, args.out)
        print("wrote", args.out)


def _slerp(Rrel: np.ndarray, lam: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(Rrel).as_rotvec()
    return Rotation.from_rotvec(rv * lam).as_matrix()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--set", nargs="*", default=[])
    t.add_argument("--proposals", nargs="*", default=None, help="out-of-fold detector prediction files on the train split")
    t.add_argument("--val_proposals", default=None, help="detector prediction file on val, for the val loss")
    a = sub.add_parser("apply")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--pred", required=True)
    a.add_argument("--split", default="val")
    a.add_argument("--out", default=None)
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--min_points", type=int, default=8)
    a.add_argument("--n_iter", type=int, default=1)
    a.add_argument("--blend", type=float, default=1.0, help="1 = take the refined box, 0.5 = halfway")
    a.add_argument("--parts", default="cdr", help="which components to refine: c centre, d dims, r rotation")
    args = ap.parse_args()
    (train if args.cmd == "train" else apply)(args)


if __name__ == "__main__":
    main()
