#!/bin/bash
cd /raid/scratch/dzimmerman2021/Diver3D
until [ -f runs/a45_gscale_s0/ep19.pt ] && [ -f runs/a45_gscale_s1/ep19.pt ]; do sleep 60; done
echo "### per-epoch"
python3 - <<'PY'
import json
for n in ("a45_gscale_s0", "a45_gscale_s1"):
    L = [json.loads(l) for l in open(f"runs/{n}/log.jsonl")]
    t = [(l["epoch"], l["val"]["SDS"], l["val"]["AP3D@0.35"], l["val"]["APBEV@0.35"]) for l in L if l.get("val", {}).get("SDS") is not None]
    print(n, " ".join(f"{e}:{s:.3f}/{a:.3f}/{b:.3f}" for e, s, a, b in t) + f" | best SDS {max(x[1] for x in t):.4f}")
PY
echo "### dumps"
python3 -m diver3d.predict --ckpt runs/a45_gscale_s0/best_sds.pt runs/a45_gscale_s1/best_sds.pt --tta --split val --dump_dir submissions/val/members --gpu 0
echo "### paired near-range vs 5 reference p0 dumps"
for s in 0 1; do echo "gscale_s$s:"; python3 -m diver3d.paired --a submissions/val/members/a45_gscale_s${s}_p0_val.jsonl.gz --b "submissions/val/members/vz025_mf2_floor0_a45_s[12346]_p0_val.jsonl.gz" --r_max 2.0; done
echo "### FP at matched recall"
python3 analysis/prcurve.py submissions/val/members/a45_gscale_s0_p0_val.jsonl.gz submissions/val/members/a45_gscale_s1_p0_val.jsonl.gz submissions/val/members/vz025_mf2_floor0_a45_s2_p0_val.jsonl.gz submissions/val/members/vz025_mf2_floor0_a45_s6_p0_val.jsonl.gz
echo "### swap ensemble"
python3 -m diver3d.fuse_tune --names vz025 vz025_s1 vz025_mf2 vz025_floor0 mf2a45e12 a45_gscale_s0 a45_gscale_s1 --nms 0.1 --fuse_ious 0.3 --modes avg --pre_thr 0.01 --procs 1 --out submissions/val/ens_mixed5gs2_val.jsonl.gz
python3 analysis/scalediag.py submissions/val/ens_mixed5gs2_val.jsonl.gz
echo "### done"
