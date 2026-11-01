#!/bin/bash
cd /raid/scratch/dzimmerman2021/Diver3D
until [ -f runs/a45_local_s0/ep19.pt ] && [ -f runs/a45_local_s1/ep19.pt ]; do sleep 60; done
echo "### per-epoch"
python3 - <<'PY'
import json
for n in ("a45_local_s0", "a45_local_s1"):
    L = [json.loads(l) for l in open(f"runs/{n}/log.jsonl")]
    t = [(l["epoch"], l["val"]["SDS"], l["val"]["AP3D@0.35"], l["val"]["APBEV@0.35"]) for l in L if l.get("val", {}).get("SDS") is not None]
    print(n, " ".join(f"{e}:{s:.3f}/{a:.3f}/{b:.3f}" for e, s, a, b in t) + f" | best SDS {max(x[1] for x in t):.4f} best APBEV {max(x[3] for x in t):.4f}")
PY
echo "### dumps"
CUDA_VISIBLE_DEVICES=0 python3 -m diver3d.predict --ckpt runs/a45_local_s0/best_sds.pt runs/a45_local_s1/best_sds.pt --tta --split val --dump_dir submissions/val/members --gpu 0
for s in 0 1; do echo "local_s$s paired near-range:"; python3 -m diver3d.paired --a submissions/val/members/a45_local_s${s}_p0_val.jsonl.gz --b "submissions/val/members/vz025_mf2_floor0_a45_s[1234]_p0_val.jsonl.gz" --r_max 2.0; done
python3 analysis/prcurve.py submissions/val/members/a45_local_s0_p0_val.jsonl.gz submissions/val/members/a45_local_s1_p0_val.jsonl.gz submissions/val/members/vz025_mf2_floor0_a45_s2_p0_val.jsonl.gz
echo "### ensembles"
B5="vz025 vz025_s1 vz025_mf2 vz025_floor0 mf2a45e12"
python3 -m diver3d.fuse_tune --names $B5 a45_local_s0 a45_local_s1 --nms 0.1 --fuse_ious 0.3 --modes avg --pre_thr 0.01 --procs 1 --out submissions/val/ens_mixed5loc2_val.jsonl.gz > logs/fuse_mixed5loc2.log 2>&1 &
python3 -m diver3d.fuse_tune --names $B5 vz025_mf2_floor0_a45_s1 vz025_mf2_floor0_a45_s2 a45_local_s0 a45_local_s1 --nms 0.1 --fuse_ious 0.3 --modes avg --pre_thr 0.01 --procs 1 --out submissions/val/ens_mixed7loc2_val.jsonl.gz > logs/fuse_mixed7loc2.log 2>&1 &
wait
for f in mixed5loc2 mixed7loc2; do echo "== $f"; grep "^nms" logs/fuse_$f.log | cut -c1-170; python3 analysis/scalediag.py submissions/val/ens_${f}_val.jsonl.gz 2>&1 | grep -v Warning | head -3; done
echo "### done"
