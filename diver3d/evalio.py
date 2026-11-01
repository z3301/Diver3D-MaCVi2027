"""Post-processing (NMS), submission writing and in-process scoring with the official tool."""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

from .geometry import quat_to_rotmat, rotmat_to_quat


def server_rotation(R: np.ndarray) -> np.ndarray:
    """Round-trip R through the 6-decimal quaternion we submit, exactly as the server parses it.
    This also restores float64 orthonormality, which the exact-cuboid IoU is sensitive to."""
    q = np.round(rotmat_to_quat(R), 6)
    return quat_to_rotmat(q)

_EV = None


def official(root: str):
    """Import tools/diver3d_eval.py from the dataset release."""
    global _EV
    if _EV is None:
        sys.path.append(os.path.join(root, "tools"))
        import diver3d_eval  # noqa: E402
        _EV = diver3d_eval
    return _EV


def nms_bev(ev, boxes: list, thr: float) -> list:
    """Greedy NMS on the true BEV footprint IoU (official iou_bev). boxes sorted by score desc."""
    keep = []
    for b in boxes:
        if all(ev.iou_bev(b, k) <= thr for k in keep):
            keep.append(b)
    return keep


def postprocess(ev, dec: dict, score_thr: float, nms_thr: float, max_dets: int) -> list:
    """Decoded tensors for one frame -> list of official Box objects."""
    sc = dec["score"].cpu().numpy()
    sel = np.where(sc >= score_thr)[0]
    if len(sel) == 0:
        return []
    c = dec["center"].cpu().numpy()[sel]
    d = dec["dims"].cpu().numpy()[sel]
    R = dec["R"].cpu().numpy()[sel]
    order = np.argsort(-sc[sel])
    boxes = [ev.Box(center=np.round(c[i].astype(np.float64), 4), dims=np.round(d[i].astype(np.float64), 4),
                    R=server_rotation(R[i].astype(np.float64)), score=round(float(sc[sel][i]), 5)) for i in order]
    if nms_thr is not None and nms_thr < 1.0:
        boxes = nms_bev(ev, boxes, nms_thr)
    return boxes[:max_dets]


def box_to_record(b) -> dict:
    return {"center": [round(float(v), 4) for v in b.center],
            "dimensions": [round(float(v), 4) for v in b.dims],
            "quaternion": [round(float(v), 6) for v in rotmat_to_quat(b.R)],
            "score": round(float(b.score), 5)}


def write_submission(pred: dict, path: str):
    """pred: token -> list[Box]. Writes .json or .jsonl(.gz) depending on extension."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".json"):
        json.dump({"results": {t: [box_to_record(b) for b in bs] for t, bs in pred.items()}}, open(path, "w"))
        return
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as fh:
        for t, bs in pred.items():
            fh.write(json.dumps({"sample_token": t, "boxes": [box_to_record(b) for b in bs]}) + "\n")


def load_gt(root: str, split: str) -> dict:
    ev = official(root)
    return ev.load_boxes(os.path.join(root, "annotations", f"gt_{split}.json"), require_score=False)


def score(root: str, gt: dict, pred: dict) -> dict:
    ev = official(root)
    res = ev.evaluate(gt, pred, verbose=False)
    return {k: v for k, v in res.items() if not k.startswith("_")} | {"_meta": res["_meta"]}
