#!/bin/bash
# after the gscale seeds: beam-drop arm (2 seeds), then the larger-radius arm (2 seeds)
cd /raid/scratch/dzimmerman2021/Diver3D
until [ -f runs/a45_gscale_s0/ep19.pt ] && [ -f runs/a45_gscale_s1/ep19.pt ]; do sleep 60; done
for arm in beamdrop rad2; do
  mkdir -p runs/a45_${arm}_s0 runs/a45_${arm}_s1
  python3 -m diver3d.train --config configs/a45_${arm}.yaml --name a45_${arm}_s0 --gpu 0 --set train.seed=0 > runs/a45_${arm}_s0/stdout.log 2>&1 &
  python3 -m diver3d.train --config configs/a45_${arm}.yaml --name a45_${arm}_s1 --gpu 1 --set train.seed=1 > runs/a45_${arm}_s1/stdout.log 2>&1 &
  wait
  echo "### $arm trained"
  python3 -m diver3d.predict --ckpt runs/a45_${arm}_s0/best_sds.pt runs/a45_${arm}_s1/best_sds.pt --tta --split val --dump_dir submissions/val/members --gpu 0
  for s in 0 1; do echo "${arm}_s$s paired near-range:"; python3 -m diver3d.paired --a submissions/val/members/a45_${arm}_s${s}_p0_val.jsonl.gz --b "submissions/val/members/vz025_mf2_floor0_a45_s[1234]_p0_val.jsonl.gz" --r_max 2.0; done
  python3 analysis/prcurve.py submissions/val/members/a45_${arm}_s0_p0_val.jsonl.gz submissions/val/members/a45_${arm}_s1_p0_val.jsonl.gz submissions/val/members/vz025_mf2_floor0_a45_s2_p0_val.jsonl.gz
  python3 - <<PY
import json
for n in ("a45_${arm}_s0", "a45_${arm}_s1"):
    L = [json.loads(l) for l in open(f"runs/{n}/log.jsonl")]
    t = [(l["epoch"], l["val"]["SDS"], l["val"]["AP3D@0.35"], l["val"]["APBEV@0.35"]) for l in L if l.get("val", {}).get("SDS") is not None]
    print(n, " ".join(f"{e}:{s:.3f}/{a:.3f}/{b:.3f}" for e, s, a, b in t) + f" | best SDS {max(x[1] for x in t):.4f} best APBEV {max(x[3] for x in t):.4f}")
PY
  echo "### $arm done"
done
