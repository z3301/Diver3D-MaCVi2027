"""Quick invariants: run with `python -m tests.test_pipeline` from the project root."""
import numpy as np
import torch

from diver3d.config import load_config
from diver3d.data import Frames, GTSampler, mirror_y, points_in_box_mask, transform_scene
from diver3d.evalio import load_gt, official, postprocess, score
from diver3d.geometry import rot_x, rot_z, rotmat_to_quat, quat_to_rotmat, rotmat_to_6d
from diver3d.model import decode
from diver3d.targets import TargetBuilder

import sys
cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")
root = cfg["data_root"]
fr = Frames(root, "val", limit=300)
ev = official(root)
gt = load_gt(root, "val")
gt = {t: gt[t] for t in fr.tokens}

# 1. targets -> decode round trip must reproduce the GT boxes (and score 1.0 officially)
tb = TargetBuilder(cfg["grid"], cfg["out_stride"], max_objs=16)
pred = {}
for i, tok in enumerate(fr.tokens):
    c, d, R, _ = fr.get_boxes(i)
    tg = tb(c, d, R)
    hm = torch.from_numpy(tg["hm"])[None]
    hm_logit = torch.where(hm >= 0.999, torch.tensor(10.0), torch.tensor(-10.0))
    reg = torch.zeros(1, 12, hm.shape[-2], hm.shape[-1])
    for k in range(int(tg["mask"].sum())):
        ind = tg["inds"][k]
        reg[0, :, ind // hm.shape[-1], ind % hm.shape[-1]] = torch.from_numpy(tg["reg"][k])
    dec = decode({"hm": hm_logit, "reg": reg}, cfg)[0]
    pred[tok] = postprocess(ev, dec, 0.5, 0.1, 100)
m = score(root, gt, pred)
print(f"round trip: SDS {m['SDS']:.4f} ATE {m['ATE_m']:.4f} AOE3D {m['AOE3D_deg']:.3f} ASE {m['ASE']:.4f}")
assert m["SDS"] > 0.99, m

# 2. augmentation invariants
rng = np.random.default_rng(0)
checked = 0
for i in range(len(fr)):
    pts = fr.get_points(i)
    c, d, R, _ = fr.get_boxes(i)
    if len(c) == 0:
        continue
    n0 = [points_in_box_mask(pts[:, :3], c[j], d[j], R[j]).sum() for j in range(len(c))]
    p2, c2, R2 = mirror_y(pts, c, R)
    n1 = [points_in_box_mask(p2[:, :3], c2[j], d[j], R2[j]).sum() for j in range(len(c))]
    assert n0 == n1, ("mirror", n0, n1)
    assert all(abs(np.linalg.det(R2[j]) - 1) < 1e-6 for j in range(len(c)))
    M = rot_z(rng.uniform(-0.3, 0.3)) @ rot_x(rng.uniform(-0.3, 0.3))
    p3, c3, R3 = transform_scene(pts, c, R, M)
    n2 = [points_in_box_mask(p3[:, :3], c3[j], d[j], R3[j]).sum() for j in range(len(c))]
    assert n0 == n2, ("rotate", n0, n2)
    q = rotmat_to_quat(R[0])
    assert np.allclose(quat_to_rotmat(q), R[0], atol=1e-6)
    checked += 1
    if checked >= 50:
        break
print(f"aug invariants ok on {checked} frames")

# 3. GT sampler produces boxes whose points are inside them
sm = GTSampler(fr, cfg["aug"]["gt_sample"], cfg["grid"])
pts = fr.get_points(0)
c, d, R, _ = fr.get_boxes(0)
p4, c4, d4, R4 = sm.sample(np.random.default_rng(1), pts, c, d, R, 3)
print(f"sampler bank {len(sm)}; frame0 boxes {len(c)} -> {len(c4)}, points {len(pts)} -> {len(p4)}")
for j in range(len(c), len(c4)):
    n = points_in_box_mask(p4[:, :3], c4[j], d4[j], R4[j], 0.1).sum()
    assert n >= 20, n
print("all ok")
