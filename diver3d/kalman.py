"""Track-level Kalman filtering / RTS smoothing of box geometry (SonarVoxNet Sec. III-D style,
without the learned residual regressors, which would need val labels).

Centre: constant-velocity Kalman in R^3 with an RTS backward pass (optionally causal only).
Rotation: the same filter on the SO(3) tangent coordinates of each rotation relative to the
track's first rotation (log map), after folding the Rz(pi) ambiguity; mapped back with exp.
Extents: score-weighted mean over a window (shrink-only hurt on val; see EXPERIMENTS.md).

    python -m diver3d.kalman --pred submissions/val/x_raw.jsonl.gz --grid
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from .evalio import load_gt, official, score, server_rotation, write_submission
from .fuse import _align_symmetry
from .refine import build_tracks


def _cv_kalman_rts(z: np.ndarray, t: np.ndarray, w: np.ndarray, q: float, r: float, causal: bool):
    """Constant-velocity Kalman filter + RTS smoother on D-dim measurements z (N, D) at integer
    times t (N,), with per-measurement weights w (N,) scaling the measurement noise (R = r / w).
    Returns smoothed positions (N, D)."""
    N, D = z.shape
    # state per axis: [p, v]; axes are independent so run them jointly with block-diagonal matrices
    x = np.zeros((N, 2 * D))
    P = np.zeros((N, 2 * D, 2 * D))
    x_pred = np.zeros_like(x)
    P_pred = np.zeros_like(P)
    Fs = []
    H = np.zeros((D, 2 * D))
    H[np.arange(D), 2 * np.arange(D)] = 1.0
    # init
    x[0, 0::2] = z[0]
    P[0] = np.eye(2 * D) * 1.0
    P[0][1::2, 1::2] = np.eye(D) * 0.25            # initial velocity uncertainty (m/frame)^2
    x_pred[0], P_pred[0] = x[0], P[0]
    Fs.append(np.eye(2 * D))
    for k in range(1, N):
        dt = float(t[k] - t[k - 1])
        F = np.eye(2 * D)
        F[0::2, 1::2] = np.eye(D) * dt
        Q = np.zeros((2 * D, 2 * D))
        qd = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        for d in range(D):
            Q[2 * d:2 * d + 2, 2 * d:2 * d + 2] = qd
        xp = F @ x[k - 1]
        Pp = F @ P[k - 1] @ F.T + Q
        R = np.eye(D) * (r / max(w[k], 1e-3))
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        x[k] = xp + K @ (z[k] - H @ xp)
        P[k] = (np.eye(2 * D) - K @ H) @ Pp
        x_pred[k], P_pred[k] = xp, Pp
        Fs.append(F)
    if causal:
        return x[:, 0::2]
    xs = x.copy()
    Ps = P.copy()
    for k in range(N - 2, -1, -1):
        C = P[k] @ Fs[k + 1].T @ np.linalg.pinv(P_pred[k + 1])
        xs[k] = x[k] + C @ (xs[k + 1] - x_pred[k + 1])
        Ps[k] = P[k] + C @ (Ps[k + 1] - P_pred[k + 1]) @ C.T
    return xs[:, 0::2]


def kalman_refine(ev, pred: dict, q_c=0.02, r_c=0.04, q_r=0.02, r_r=0.1, wd=8, causal=False,
                  iou_thr=0.2, min_len=2, tracks=None, use_rot=True, use_center=True, use_dims=True) -> dict:
    out = {t: list(bs) for t, bs in pred.items()}
    if tracks is None:
        tracks = build_tracks(ev, pred, iou_thr)
    for tr in tracks:
        if len(tr) < min_len:
            continue
        boxes = [pred[t][i] for t, i in tr]
        n = len(boxes)
        frames = np.array([int(t.split("/")[1]) for t, _ in tr], dtype=float)
        w = np.array([b.score for b in boxes])
        C = np.stack([b.center for b in boxes])
        if use_center:
            C = _cv_kalman_rts(C, frames, w, q_c, r_c, causal)
        Rs = [boxes[0].R]
        for b in boxes[1:]:
            Rs.append(_align_symmetry(b.R, Rs[-1]))
        if use_rot:
            R0 = Rs[0]
            tang = np.stack([Rotation.from_matrix(R0.T @ R).as_rotvec() for R in Rs])   # body-frame log map
            tang_s = _cv_kalman_rts(tang, frames, w, q_r, r_r, causal)
            Rs = [server_rotation(R0 @ Rotation.from_rotvec(v).as_matrix()) for v in tang_s]
        Dm = np.stack([b.dims for b in boxes])
        for k in range(n):
            d = Dm[k]
            if use_dims and wd > 0:
                lo, hi = max(0, k - wd), (k if causal else min(n - 1, k + wd))
                ww = w[lo:hi + 1] / w[lo:hi + 1].sum()
                d = (ww[:, None] * Dm[lo:hi + 1]).sum(0)
            t, i = tr[k]
            out[t][i] = ev.Box(center=C[k], dims=d, R=Rs[k], score=boxes[k].score)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--root", default="data/diver3d_v2_extracted/diver3d_v2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--q_c", type=float, default=0.02)
    ap.add_argument("--r_c", type=float, default=0.04)
    ap.add_argument("--q_r", type=float, default=0.02)
    ap.add_argument("--r_r", type=float, default=0.1)
    ap.add_argument("--wd", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ev = official(args.root)
    pred = ev.load_boxes(args.pred, require_score=True)
    gt = load_gt(args.root, args.split) if args.split != "test" else None
    tracks = build_tracks(ev, pred, 0.2)

    def rep(name, p):
        if gt is None:
            return
        m = score(args.root, gt, p)
        print(f"{name:48s} SDS {m['SDS']:.4f} mAP3D {m['mAP3D']:.4f} AP35 {m['AP3D@0.35']:.4f} AP50 {m['AP3D@0.50']:.4f} "
              f"ATE {m['ATE_m']:.3f} ASE {m['ASE']:.3f} AOE3D {m['AOE3D_deg']:.1f}", flush=True)

    rep("raw", pred)
    if args.grid:
        from .refine import refine
        rep("window ref (1,5,8) [reference]", refine(ev, pred, windows=(1, 5, 8), tracks=tracks))
        for q_c, r_c in itertools.product((0.005, 0.02, 0.08), (0.01, 0.04, 0.16)):
            rep(f"kalman centre q={q_c} r={r_c} (rot/dims off)", kalman_refine(ev, pred, q_c, r_c, tracks=tracks, use_rot=False, use_dims=False))
        for q_r, r_r in itertools.product((0.005, 0.02, 0.08), (0.03, 0.1, 0.3)):
            rep(f"kalman rot q={q_r} r={r_r} (centre/dims off)", kalman_refine(ev, pred, q_r=q_r, r_r=r_r, tracks=tracks, use_center=False, use_dims=False))
        return
    p = kalman_refine(ev, pred, args.q_c, args.r_c, args.q_r, args.r_r, args.wd, args.causal, tracks=tracks)
    rep(f"kalman q_c={args.q_c} r_c={args.r_c} q_r={args.q_r} r_r={args.r_r} wd={args.wd} causal={args.causal}", p)
    if args.out:
        write_submission(p, args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
