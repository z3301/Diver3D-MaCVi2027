#!/bin/bash
# round 16: label-free pair stats on test, test files with relaxed NMS, and the val equivalent with sc105 members
cd /raid/scratch/dzimmerman2021/Diver3D
R="vz025 vz025_s1 vz025_mf2 vz025_floor0 vz025_mf2_floor0_a45 vz025_mf2_floor0_a45_s1 vz025_mf2_floor0_a45_s2 a45_occl a45_occl_s1"
S=$(for n in $R; do echo -n "${n}_sc105 "; done)
until [ $(ls submissions/test/members/ 2>/dev/null | grep -c "_p1_test") -ge 18 ]; do sleep 60; done; sleep 30
echo "### test dumps complete $(date +%H:%M)"
# val sc105 dumps for the 7 mixed7 members (GPU 0) in the background while the CPU work runs
V7="runs/vz025/best_sds.pt runs/vz025_s1/best_sds.pt runs/vz025_mf2/best_sds.pt runs/vz025_floor0/best_sds.pt runs/vz025_mf2_floor0_a45/best_sds.pt runs/vz025_mf2_floor0_a45_s1/best_sds.pt runs/vz025_mf2_floor0_a45_s2/best_sds.pt"
CUDA_VISIBLE_DEVICES=0 python3 -m diver3d.predict --ckpt $V7 --tta --split val --dump_dir submissions/val/members --scale 1.05 --gpu 0 > logs/valdump_sc105.log 2>&1 &
echo "### label-free suppressed-pair statistic on test (scale 1.0 members)"
python3 analysis/pairstats.py --split test --names $R 2>&1 | grep -v Warning
echo "### test files"
python3 analysis/fuse_test.py --names $R $S --nms 0.1 --out submissions/ens9o_sc105_nms01_test.jsonl.gz 2>&1 | grep -v Warning | tail -5
python3 analysis/fuse_test.py --names $R $S --nms 0.3 --out submissions/ens9o_sc105_nms03_test.jsonl.gz 2>&1 | grep -v Warning | tail -5
python3 analysis/fuse_test.py --names $R --nms 0.3 --out submissions/ens9o_nms03_test.jsonl.gz 2>&1 | grep -v Warning | tail -5
echo "### val: mixed7 + sc105 members at nms 0.1 vs 0.3"
wait
B7="vz025 vz025_s1 vz025_mf2 vz025_floor0 mf2a45e12 vz025_mf2_floor0_a45_s1 vz025_mf2_floor0_a45_s2"
V=$(for n in vz025 vz025_s1 vz025_mf2 vz025_floor0 vz025_mf2_floor0_a45 vz025_mf2_floor0_a45_s1 vz025_mf2_floor0_a45_s2; do echo -n "${n}_sc105 "; done)
python3 analysis/fuse_test.py --split val --members submissions/val/members --names $B7 $V --nms 0.1 --out submissions/val/ens_mixed7_sc105_nms01_val.jsonl.gz 2>&1 | grep -v Warning | tail -2
python3 analysis/fuse_test.py --split val --members submissions/val/members --names $B7 $V --nms 0.3 --out submissions/val/ens_mixed7_sc105_nms03_val.jsonl.gz 2>&1 | grep -v Warning | tail -2
python3 analysis/fuse_test.py --split val --members submissions/val/members --names $B7 --nms 0.3 --out submissions/val/ens_mixed7_nms03_val.jsonl.gz 2>&1 | grep -v Warning | tail -2
python3 analysis/scalediag.py submissions/val/ens_mixed7_sc105_nms03_val.jsonl.gz submissions/val/ens_mixed7_sc105_nms01_val.jsonl.gz 2>&1 | grep -v Warning | head -3
echo "### done $(date +%H:%M)"
