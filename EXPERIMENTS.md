# Experiment log

All numbers are val SDS from the official scorer unless marked LB (leaderboard / test).
Val is noisy: two seeds of the same config differ by ~0.03 at their best epoch.

## Training runs (20 epochs, batch 8, AdamW 1e-3 one-cycle)

| run | config | best SDS (epoch) | mAP3D | ATE | ASE | AOE3D | notes |
|---|---|---|---|---|---|---|---|
| base | configs/base.yaml (full aug + GT-sampling curriculum) | 0.626 (18) | 0.571 | 0.193 | 0.344 | 37.8 | seed 0 |
| base_seed1 | same config (was meant to be no-aug; override bug) | 0.655 (15) | 0.623 | 0.186 | 0.331 | 37.9 | effectively seed 1 |
| noaug | aug.enable=false | 0.454 (9) | 0.288 | 0.217 | 0.312 | 54.9 | collapses to ~0.39 by ep 19; overfits train pose prior |
| mf2 | data.n_prev=2 (stack 2 previous frames, time channel) | 0.637 (15) | 0.599 | 0.185 | 0.333 | 41.1 | same band as single-frame seeds; +5% epoch time |
| base_s2_e30 | seed 2, 30 epochs | 0.638 (23) | 0.601 | 0.185 | 0.347 | 39.7 | 30 epochs no better than 20 |
| vz025_s1 | grid.vz=0.25, seed 1 | 0.674 (15) | 0.662 | 0.198 | 0.347 | 35.5 | confirms vz025 gain (seed noise ~0.01-0.03) |
| vz0125 | grid.vz=0.125 (40 z-bins) | 0.664 (15) | 0.628 | 0.195 | 0.328 | 33.7 | no gain over 0.25 m at 2x cost; z resolution saturated |
| vz025_mf4 | grid.vz=0.25 + data.n_prev=4 | 0.646 (15) | 0.622 | 0.185 | 0.348 | 41.1 | worse than n_prev=2; AOE3D degrades |
| vz025_ema | grid.vz=0.25 + weight EMA 0.9995 | killed | | | | | EMA model produced 0 detections for 2 epochs (weights move too fast on this 52k-step schedule); abandoned |
| vz025_mf2_s2 | grid.vz=0.25 + n_prev=2, seed 2 | 0.668 (18) | 0.648 | 0.179 | 0.340 | 37.8 | seeds of this config: 0.700 / 0.661 / 0.668 |
| vz025_wide | grid.vz=0.25, bb3d_ch=128, neck_ch=256 | 0.642 (15) | 0.595 | 0.194 | 0.338 | 36.0 | capacity hurts (overfits); 2x cost |
| vz025_rot2 | grid.vz=0.25, loss.rot_weight=2 | 0.668 (18) | 0.649 | 0.198 | 0.334 | 36.4 | orientation got worse, not better |
| vz025 | grid.vz=0.25 (20 z-bins) | **0.683 (16)** | 0.659 | 0.188 | 0.334 | 32.1 | clear win: orientation/tilt; 1.8x epoch time |
| vz025_mf2 | grid.vz=0.25 + data.n_prev=2 | **0.700 (15)** | 0.690 | 0.181 | 0.337 | 31.7 | best single model |
| vz025_mf2_s1 | same, seed 1 | 0.661 (19) | 0.623 | 0.183 | 0.342 | 33.9 | interrupted at ep 3 and resumed; seed noise ~0.04 |
| vxy005_vz025 | grid 0.05 x 0.05 x 0.25 m (240x200x20) | 0.670 (19) | 0.639 | 0.187 | 0.333 | 33.6 | no gain over 0.1 m xy at 4x cost; AP50 unchanged |
| vz025_os1 | grid.vz=0.25, out_stride=1 (0.1 m BEV output) | 0.670 (15) | 0.634 | 0.190 | 0.334 | 32.0 | within seed band; no gain |
| vz025_crop8 | grid.vz=0.25, crop [0,8] x [-4,4] m | 0.654 (19) | 0.623 | 0.192 | 0.346 | 36.7 | slightly below band; faster (2.3 min/epoch) but no gain |

Paper reference (test split): SonarVoxNet 0.673 / mAP3D 0.690 / ATE 0.183 / ASE 0.336 / AOE3D 46.2.
Our orientation is ~8 deg better than the paper (rot_sym loss folds Rz(pi)); our detection mAP is
weaker (0.57-0.62 vs 0.69). APBEV@0.35 is 0.75 vs AP3D@0.35 0.58-0.63, so the gap is out-of-plane
geometry (z, tilt, extents), not top-down detection.

## Inference-time sweeps (runs/base/best_sds.pt, val)

| setting | SDS | mAP3D | AOE3D |
|---|---|---|---|
| thr 0.30 (paper) | 0.613 | 0.542 | 37.3 |
| thr 0.10 (config default) | 0.626 | 0.571 | 37.9 |
| thr 0.05 | 0.627 | 0.573 | 38.0 |
| thr 0.05 + flip-y TTA | 0.649 | 0.610 | 35.8 |
| thr 0.05 + TTA, 2-seed ensemble (base + base_seed1 best_sds) | **0.681** | 0.666 | 35.3 |
| thr 0.05 + TTA, 4-ckpt ensemble (+ last.pt of both seeds) | 0.673 | 0.652 | 35.5 |
| thr 0.05 + TTA, 3-model ensemble (base, base_seed1, mf2 best_sds) | 0.693 | 0.688 | 35.2 |
| vz025 best_sds, thr 0.05 + TTA (single model) | 0.690 | 0.676 | 34.0 |
| thr 0.05 + TTA, 4-model ensemble (+ vz025) | 0.705 | 0.705 | 33.9 |
| thr 0.05 + TTA, 5-model ensemble (+ vz025_s1) | 0.715 | 0.722 | 33.0 |
| thr 0.05 + TTA, finer-z only: vz025 + vz025_s1 | 0.715 | 0.725 | 32.5 |
| thr 0.05 + TTA, finer-z only: vz025 + vz025_s1 + vz025_mf2 | **0.730** | 0.749 | 31.8 |
| + vz025_mf2_s1 (0.661) as 4th member | 0.728 | 0.744 | 31.7 |
| vz025 + vz025_s1 + vz025_mf2 + vz025_mf2_s2 (0.668) | 0.731 | 0.751 | 32.0 |
| same + vz025_mf2_s1 (5 members) | 0.730 | 0.747 | 31.8 |

Ensembling this family saturates at ~0.73 raw / ~0.74 refined; extra ~0.66-0.67 seeds add <= 0.001.
Further gains need a better single model (fine xy grid and wide model running overnight 10-08/09).

-> use thr 0.05, TTA, and ensemble independent seeds. Adding weaker same-run checkpoints
(last epoch, 0.62) to the 2-seed ensemble *lowered* SDS, so member quality matters: prefer more
seeds / diverse configs over more epochs of one run. Fusion: cluster at 3D IoU
0.45, score-weighted mean of centre/dims, chordal mean rotation, score = mean over sources.
Submissions written: `submissions/seed1_tta_thr05_test.jsonl.gz` (val 0.67), `submissions/ens2_tta_thr05_test.jsonl.gz` (val 0.681),
`submissions/ens3_tta_thr05_test.jsonl.gz` (val 0.693), `submissions/ens4_tta_thr05_test.jsonl.gz` (val 0.705), `submissions/ens3z_tta_thr05_test.jsonl.gz` (val 0.730).

Lesson: the 0.5 m-z models add nothing once finer-z models are in the pool (2 finer-z == 5 mixed).
Build the final ensemble from vz025-family seeds/variants only.

## Temporal refinement (diver3d/refine.py, on ens3z val predictions, raw 0.7305)

Tracks by BEV IoU >= 0.2 between consecutive frames; score-weighted smoothing of centre / rotation
(chordal mean) / dims over a centred window; scores and detection set untouched.

| setting | SDS | AP50 | ASE | AOE3D |
|---|---|---|---|---|
| centre only, +-1 frame | 0.7318 | | | |
| centre +-2 / +-3 / +-5 | 0.7289 / 0.7223 / 0.6965 | | | | 
| rotation only +-1 / +-3 / +-5 | 0.7328 / 0.7343 / 0.7348 | | | 31.1 / 30.5 / 30.3 |
| dims only +-1 / +-3 / +-5 | 0.7320 / 0.7338 / 0.7341 | | 0.331 / 0.328 / 0.326 | |
| wc=0 wr=5 wd=8 | 0.7386 | 0.301 | 0.325 | 30.3 |
| **wc=1 wr=5 wd=8 (chosen)** | **0.7405** | 0.303 | 0.325 | 30.3 |
| causal variants | ~0.002-0.004 below centred | | | |

Centre must use a tiny window (divers move at 6 Hz); rotation and size can average widely.
Median dims slightly worse than weighted mean everywhere. Applied to test:
`submissions/ens3z_tta_thr05_ref158_test.jsonl.gz` (val-equivalent 0.7405).

## Diagnosis after the first leaderboard result (2026-10-09)

Val diagnosis of the submitted ensemble (diver3d/diagnose.py):
- Recall ceiling at IoU 0.35 is 0.90 and no diver is missed outright; misses are box-fit failures,
  concentrated at < 2 m range (29% miss) and lying divers (28%).
- 53% of false positives (score >= 0.3) are on EMPTY frames; 59% of empty frames carry a detection
  >= 0.3. Cause (hypothesis): GT-sampling floor=1 pastes a diver into every empty frame in the final
  epochs, so the model never sees an empty frame late in training. -> run vz025_floor0.
- Two val scenes collapse: 0075 (89% empty, AP35 0.06) and 0080 (tiny boxes 0.63x0.40x0.96 m at
  < 2 m, AP35 0.01; our widths are 1.85x GT, median best IoU 0.30).
- The single model fits train's small-box scenes (0067/0077/0079) with dims ratio 1.00 and train
  SDS 0.889, i.e. it memorises box size per scene layout and does not transfer it to unseen scenes.
  -> generalisation across scenes is the lever: vz025_mix (frame-mix background augmentation),
  head dropout, and the fusion/NMS sweep (diver3d/fuse_tune.py) on dumped member passes.

| run | config | best SDS (epoch) | notes |
|---|---|---|---|
| vz025_floor0 | gt_sample.floor=0 | 0.675 (16); TTA 0.694 | FPs at >=0.3: 3730 -> 1634; empty frames with det >=0.5: 25% -> 6%. Val SDS +0.004 only (val has few empty frames) but should transfer to test |
| vz025_mix | aug.frame_mix=0.5 (union with another frame's background, divers removed, 50% subsampled) | 0.664 | no gain |
| vz025_mf2_floor0 | n_prev=2 + floor=0 | 0.665 (18) | adds nothing on top of floor0 in the ensemble (5-member 0.7448 = 4-member) |
| vz025_floor0_s1 | floor=0, seed 1 | running | GPU 1, 16:48 |
| vz025_floor0_s2 | floor=0, seed 2 | 0.670 (best) | floor0 seeds: 0.675 / 0.664 / 0.670 |
| vz025_mf2_floor0_s1 | n_prev=2 + floor=0, seed 1 | 0.642 | weak seed |
| vz025_mf2_floor0_iscale | n_prev=2 + floor=0 + inst_scale [0.75,1.25] | 0.6504 (ep>=16 pick) | weak |
| vz025_mf2_floor0_a45 | n_prev=2 + floor=0 + anneal_start_frac=0.45 | **0.693 (12)**; TTA **0.717** | best single model so far; ep12 = first epoch after pasting stops; FPs 1634 (2.3% of empty frames >= 0.5), recall ceiling 0.873. Later epochs decay to 0.66. Mixed ensemble legacy3 + floor0 + this = **0.7518** val. Second seed needed to tell recipe from luck. |
| vz025_mf2_floor0_a45_s1 | same, seed 1 | 0.655 | ep12 0.639: the spike did not reproduce |
| vz025_mf2_floor0_a45_s2 | same, seed 2 | 0.670 (15) | ep12 0.615 -> seed-0's 0.693 was luck, not a recipe |
| vz025_floor0_iscale | floor=0 + per-instance scaling of pasted divers [0.75,1.25] | 0.661 (15); TTA 0.674 | scene_0080 width ratio unchanged (1.86 vs 1.87): the width error is annotation convention, not a size prior. Dead end. |
| vz025_floor0_a45 | floor=0, anneal_start_frac=0.45 (pasting stops at ep 12) | 0.632 (16); TTA 0.660 | 8 clean epochs: FPs 1263 (28% on empty) but TPs at >=0.3 only 3977 (floor0 5105, legacy 5754); the model goes conservative everywhere. 4 clean epochs (plain floor0) is the better trade. Not pursued. |

Finding: with floor=0 and anneal start 0.65, pasting stops only at epoch 16, so just epochs 16-19 see
real empty frames. best_sds checkpoints from epoch <= 15 (e.g. vz025_floor0_iscale, ep15) still
hallucinate on 76% of empty frames; floor0 (ep16) does not. Checkpoint epoch, not seed, decides it.
Confirmed on one run (vz025_floor0_iscale): ep15 snapshot 76% of empty frames flagged (25% at >= 0.5),
ep18 snapshot 20% (4.5%). train.py now only selects best checkpoints from epochs >= pasting stop.
Post-hoc picks (best val SDS among ep>=16): floor0 ep16, floor0_s1 ep18, floor0_s2 ep16, mf2_floor0 ep18,
iscale ep16 (saved as best_sds_e16.pt). Ensemble notes: floor5 (floor0 x3 + mf2_floor0 + iscale@ep15) 0.7351;
legacy3 + floor0 + iscale 0.7433 (mixed family saturated at ~0.745). floor5 with iscale@ep16 (correct
epoch) 0.7310 < 0.7351 with iscale@ep15: on val the FP-prone checkpoints score HIGHER in the ensemble
(val rewards mid-score recall more than it penalises empty-frame FPs). Only test can arbitrate.
With mf2a45@ep12 added: mixed (legacy3 + floor0 + mf2a45) **0.7518**; FP-aware (floor0 x3 + mf2_floor0 +
mf2a45) 0.7440; compact FP-aware (floor0 + floor0_s2 + mf2_floor0 + mf2a45) 0.7442; FP-aware 6 (compact 4 +
a45_s1 + a45_s2) **0.7485** -> `submissions/fp6_tta_ref158_thr05_test.jsonl.gz`. Mixed 7 (legacy3 + floor0 +
mf2a45 + a45_s1 + a45_s2) **0.7550** -> `submissions/ens7m_tta_ref158_thr05_test.jsonl.gz`.

Compositions (val, fuse 0.30, refine, cut 0.05): legacy3+floor0 0.7448; +mf2_floor0 0.7448;
+floor0_s1 (5) 0.7433; all 6 0.7444; floor0+mf2_floor0 only 0.7257; floor0+floor0_s1+mf2_floor0 0.7299.

Empty-frame FP comparison of fused ensembles (val): legacy3 71% of empty frames have a det >= 0.3
(16% >= 0.5), mixed4 58% / 9%, floor3 21% / 3.7%. floor3 also loses TPs at >= 0.3 (5421 vs 6212)
so val SDS is 0.730 vs 0.745. floor4 (floor0 x3 + mf2_floor0) 0.7338 (still climbing per member);
legacy3 + floor0 + floor0_s2 0.7449 (saturated at 0.745). Submitting both mixed4 and floor3 to let test arbitrate:
`submissions/ens4f_tta_ref158_thr05_test.jsonl.gz` (mixed4) and `submissions/floor3_tta_ref158_thr05_test.jsonl.gz`.

Ensemble with floor0 (fuse 0.30, refine 1/5/8, post-cut 0.05, val): vz025+vz025_s1+vz025_mf2 = 0.7420;
+ vz025_floor0 as 4th = **0.7448**; floor0 replacing vz025_s1 = 0.7431. Test file:
`submissions/ens4f_tta_ref158_thr05_test.jsonl.gz`. Plan: move the whole ensemble to floor0-style
members so empty-frame scores stop being propped up by the legacy members.

Fusion/NMS sweep (diver3d/fuse_tune.py, 3 members x 2 passes, refined, val): fusion IoU 0.30 best
(0.742) vs 0.45 (0.740) vs 0.60 (0.711); 0.15-0.25 flat at 0.741; NMS 0.1/0.3/0.5/none within 0.002;
'max' scoring -0.015 everywhere; pre-threshold 0.01 vs 0.05 +0.0002. New default: fuse_iou 0.30.

## Round 3 (2026-10-10): box fit, after the second leaderboard result

LB 2026-10-11: ens7 (mixed) 0.7264, fp6 (FP-aware) 0.7217, leader 0.7420. FP-aware family does NOT
transfer better -> closed. APBEV@0.35 ours 0.835 vs leader 0.842, AP3D@0.35 0.724 vs 0.763: the gap is
3D box fit of near-threshold boxes, not top-down detection. Levers:
- corner loss (L1 on 8 corners, Rz(pi)-folded): vz025_mf2_corner best 0.634 (17), AP50 ~0.16, ASE 0.36
  vs plain mf2 seeds 0.661-0.700 / AP50 0.20-0.24 / ASE 0.34 -> hurts box fit. Closed.
- IoU-quality head (BCE on exact 3D IoU of decoded boxes at GT cells; score = hm^(1-a) * iou^a): run vz025_mf2_iou (GPU 1, 19:52)
- Kalman/RTS track smoother (constant-velocity, SO(3) tangent; paper Sec. III-D minus the learned residuals): diver3d/kalman.py, grid on fused_mixed7_raw val
- second-stage point refinement: diver3d/stage2.py (PointNet on points in the enlarged box, jittered-GT proposals)

Results (val, on fused_mixed7_raw 0.7460; window smoother -> 0.7550):
- Kalman/RTS centre (CV model): best 0.7472 (q=0.02 r=0.01), smoother settings monotonically worse
  (0.741 at r=0.04, 0.726 at r=0.16). Rotation Kalman in SO(3) tangent: 0.741-0.742 for every setting,
  i.e. WORSE than raw; the chordal-mean window (+0.004) stays. Kalman line closed.
- Stage-2 refiner full (c,d,r) blend 1: 0.7305 (ATE 0.178 -> 0.190): trained on jittered GT, it
  learned train-scene box style; centre correction hurts. Blend 0.5: 0.7464 but AP50 0.308 -> 0.330.
  Extents-only blend 0.5: 0.7471; + window smoother 0.7567 (AP50 0.349, ASE 0.313) vs 0.7550.
  Rotation-only 0.5 + smoother 0.7559. Extents+rotation 0.5 + smoother **0.7581** (AP35 0.803, AP50 0.357)
  -> `submissions/ens7m_s2dr_ref158_thr05_test.jsonl.gz`. Full blend worse everywhere; 2 iterations 0.6945.
  Next: retrain the refiner on out-of-fold detector proposals: oof_f0/f1/f2 (vz025 config, each excluding
  7 train scenes; configs/oof_folds.json), predict excluded scenes with TTA, train s2oof on those real
  proposals (+30% jittered GT), evaluate dr/cdr x blend 0.5/1.0 (+ window smoother). Automated chain:
  logs/oof_chain.log.
  Result (2026-10-10 23:40): fold detectors 0.636/0.687/0.605; s2oof on 27,264 real+jittered proposals
  barely learns (val corner error 0.2478 -> 0.2459). On fused_mixed7_raw + window smoother: dr/0.5 0.7575,
  dr/1.0 0.7594, cdr/0.5 0.7581, cdr/1.0 0.7593 vs 0.7581 (jittered-GT refiner). Within noise; the
  detector residual on unseen scenes is not predictable from the points in the box. Test file unchanged.
- IoU-quality head (vz025_mf2_iou): best raw 0.673 (ep 19), TTA 0.6859 at alpha 0; score blend
  hm^(1-a)*iou^a: a=0.3 0.680, worse. Rescoring by predicted IoU does not improve the ranking. Closed.
- Corner loss: 0.634, closed (above).

Round-3 summary: every box-fit lever (corner loss, IoU head, Kalman, OOF refiner) is <= +0.003 on
val. Remaining untried diversity source: a different grid geometry. Polar BEV grid (u = range 0.1 m,
v = bearing 1 deg or 0.5 deg, z 0.25 m; `configs/polar.yaml`, `grid.polar_deg_per_unit`) runs
polar_mf2_floor0 (GPU 1, 23:43) and polar_mf2_floor0_s1 (GPU 0, 00:10). 0.5 deg cells (polar05): 0.479/0.500/0.498 at ep 0-2 vs 0.496/0.560/0.578 for 1 deg, 2x slower -> killed.

## Things that did not work (2026-10-08 evening, all on val)

- Snapshot averaging of the last 3-5 epochs (SWA, with or without BN re-estimation): equal to or
  below the best single epoch for vz025 (0.675-0.683 vs 0.683), vz025_mf2 (0.681-0.695 vs 0.700),
  vz025_s1 (0.665-0.671 vs 0.674). Weight EMA during training: collapsed. Not a lever here.
- Extent scaling after refinement: global x0.97 +0.0014, per-axis x0.95 +0.001; shrink-only dims
  (paper's choice) -0.03. Not applied (val divers are smaller than train, so this would not transfer).
- Track-length rescoring (score x f(track length)): <= +0.0006. The ensemble score already encodes it.
- 4 previous frames (n_prev=4): 0.646, worse than 2. 0.125 m z-voxels: 0.664, no gain over 0.25 m.

## Leaderboard snapshots (test)

| date | leader | SDS | mAP3D | AP50 | ATE | ASE | AOE3D | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-08 | StagAI d3-ens5 | 0.7216 | 0.733 | 0.404 | 0.171 | 0.293 | 36.5 | |
| 2026-10-08 18:30 | Dev Thakkar d3-ens3-calb | 0.7386 | 0.754 | 0.417 | 0.171 | 0.309 | 31.5 | "cal" = calibration?, "d3-base-ts" single 0.714 w/ AOE3D 28.6 = temporal smoothing? |

| 2026-10-09 | Dev Thakkar d3-ens3-kalman | 0.7420 | 0.755 | 0.409 | 0.170 | 0.306 | 30.4 | |
| 2026-10-09 | **ours: FAU CAAI d3-vz-ens3-ref (rank 5)** | **0.7235** | 0.713 | 0.392 | 0.165 | 0.312 | **29.0** | = submissions/ens3z_tta_thr05_ref158 (val 0.7405) |

**Val -> test calibration (first upload):** SDS -0.017. mAP3D 0.762 -> 0.713 (-0.05) is the whole
loss; ATE/ASE/AOE3D all transferred equal or better (AOE3D 29.0 is the best on the board). Gap to
rank 1 is detection quality only (mAP3D 0.713 vs 0.755). Next work should target detection/recall
on the test pose mix, not geometry.

Gap analysis vs our val: leaders' AP3D@0.50 ~0.41 vs our 0.28 -> our boxes are looser. ASE 0.29-0.31 vs
our 0.33. Candidate levers: causal/temporal refinement (paper: +0.046 SDS), IoU-aware score calibration.

## Bugs found on the way

- Config overrides: `--set aug.enable=false` was parsed as the string "false" (truthy). Fixed to
  YAML scalar parsing on 2026-10-08. The run named noaug from the first batch is really seed 1.
- Official IoU needs float64-orthonormal R; see README.

## Round 4 (2026-10-11): grid geometry and domain balance

- Polar BEV grid (range 0.1 m x bearing 1 deg, z 0.25 m; `configs/polar.yaml`): polar_mf2_floor0 best
  0.639 (ep 18; ep 15 0.650 not selectable), seed 1 0.650 (ep 18). Single + TTA + smoother 0.681.
  Mixed7 + polar s0: 0.7545 vs mixed7 0.7550 -> no ensemble gain at equal weight. 0.5 deg cells worse
  and 2x slower (killed at ep 2). Polar is a valid but weaker member; parked.
- Domain finding: scenes split into two families by mean intensity / range. Short-range low-intensity
  scenes (inten mean 0.12-0.15, range ~3.7-4.6 m, small boxes ~0.6-0.8 x 0.6-0.8 x 1.0-1.4 m):
  train 0067/0077/0078/0079 = 3,520 of 20,861 frames (17%); val 0000/0048/0068/0080 = 55% of val boxes;
  test 0019/0021/0049 = 4,631 of 8,112 frames (57%, scene_0021 alone 43%). The val ensemble loses
  mainly on 0068 (lying, short range: SDS 0.69) and 0080 (tiny boxes: 0.37); 0000/0048 are fine (0.81/0.84).
  -> `data.scene_weights` (SceneWeightedSampler, 3x for the four short-range train scenes; duplicates get
  distinct aug RNG) and `aug.gt_sample.same_group` (paste divers only from the same scene family).
  Runs: vz025_mf2_floor0_short3 (GPU 1), vz025_mf2_floor0_short3grp (GPU 0), launched 01:35.
  Result (03:00): short3 best 0.640 (ep 17), short3grp 0.639 (ep 16) vs reference 0.665. Per scene
  (single, thr 0.05, short3 vs ref): 0000 0.700/0.666, 0048 0.749/0.818, 0068 0.575/0.597, 0080 0.362/0.374,
  0043 0.703/0.715, 0082 0.851/0.847. Oversampling the short-range family does not help the short-range
  val scenes (0048, 0068 get worse; 0080 unchanged). The 0068/0080 loss is not a lack of in-domain
  training frames. Closed.
- predict.py now loads pre-IoU-head checkpoints (strict=False, only head.iou.* may be missing).
- Idle GPUs: vz025_mf2_floor0_a45 seeds 3 and 4 (03:03) for the final ensemble refresh.
- Ensemble close-out (03:48): mixed7 + polar s0 + s1 0.7539; mixed7 + short3 0.7529; mixed7 0.7550.
  Neither polar nor the oversampled model adds to the ensemble. Mixed7 (+ s2 refiner, 0.7581) stays.
- a45 seeds 3/4: 0.673 (ep 13) / 0.657 (ep 14). Mixed9 (mixed7 + s3 + s4): raw 0.7452, + s2a dr/0.5 +
  smoother 0.7569 vs mixed7 0.7460 / 0.7581. Ensemble saturated; mixed7 + s2 remains the final candidate.
  (ens9m_s2dr_ref158_thr05_test.jsonl.gz is built for completeness but not recommended.)
- Single-model calibration upload (2026-10-12): vz025_mf2_floor0_a45 @ ep12 + flip TTA + smoother (1,5,8)
  + thr 0.05 -> val 0.7320 (raw 0.6926, TTA 0.7170, smoothed 0.7328). File
  `submissions/single_a45e12_tta_ref158_thr05_test.jsonl.gz` (valid, 18,677 dets). Purpose: measure
  val->test transfer of one config vs the ensemble (0.7550 -> 0.7264).

## Leaderboard 2026-10-13 (test)

| entry | val | test SDS | mAP3D | AP3D@0.35 | AP50 | APBEV@0.35 | ATE | ASE | AOE3D |
|---|---|---|---|---|---|---|---|---|---|
| d3-vz-ens7-ref (mixed7, no s2) | 0.7550 | **0.7264** | 0.714 | 0.724 | 0.411 | 0.835 | 0.164 | 0.308 | 28.1 |
| d3-vz-ens7-s2 (mixed7 + s2a dr/0.5) | 0.7581 | 0.7246 | 0.713 | 0.724 | 0.399 | 0.836 | 0.164 | 0.317 | 27.9 |
| d3-vz-single-ref (a45@ep12 + TTA + smoother) | 0.7320 | 0.6954 | 0.662 | 0.670 | 0.293 | 0.770 | 0.176 | 0.327 | 27.9 |

- Stage-2 refiner: +0.003 val, -0.002 test, ASE 0.308 -> 0.317: learned train box conventions. Closed.
- Single model loses 0.037 val->test vs 0.029 for the ensemble: ensembling helps transfer. Keep ensemble.
- Gap to leader (0.7420): AP3D@0.35 0.724 vs 0.763 with APBEV@0.35 0.835 vs 0.842 and equal/better
  ATE/ASE/AOE3D on matched boxes -> near-threshold boxes failing 3D overlap (vertical placement / height).

## Round 5 (2026-10-12/13): sensor physics (after re-reading the papers; PLAN.md section 8)

Base config vz025_mf2_floor0_a45; reference seeds 0.693 / 0.655 / 0.670 / 0.673 / 0.657 (mean 0.670).
- a45_vis_occl (visibility channels + beam-consistent paste): best 0.640 (ep 13). Below the band.
- a45_occl (beam-consistent paste, one return per beam, pasted returns outside the wedge dropped):
  best 0.671 (ep 13). Neutral on val (the leader credits this trick; it may matter on test only).
- a45_vis (visibility channels only): best 0.652 (ep 14). a45_occl_s1: 0.652 (ep 12).
  Physics round on val: vis 0.652, occl 0.671/0.652, vis+occl 0.640 vs reference mean 0.670. No gain.
  Caveat: the leader credits beam-space paste and val has disagreed with test before ->
  ens9o = mixed7 + occl s0 + occl s1 test file built for an optional calibration upload (logs/ens9o.log).
- a45_zcls (z classified over 20 bins of 0.25 m + in-bin residual; `model.z_bins`, the papers'
  "sonar-native elevation decoder" in its simplest form): best 0.678 (ep 12, AP50 0.254, ATE 0.178),
  selectable epochs 0.631-0.678 (mean 0.646) vs reference seeds 0.670 / 0.643. AP50 consistently a
  little higher (0.19-0.25 vs 0.18-0.21). Seed 1: best 0.651 (ep 15), selectable
  epochs 0.605-0.651 (mean 0.632). Two seeds 0.678 / 0.651 vs reference 0.693 / 0.655 / 0.670 / 0.673 /
  0.657: neutral. Sonar-native z decoding in its simplest form does not move val. Closed.

Round-5 summary: none of the physics-motivated variants (visibility channels, beam-consistent paste,
z classification) changes val beyond seed noise. Candidate files for a test-side check of the paste
trick: ens9o (val 0.7541). GPUs idle 2026-10-13 03:52.

## Leaderboard 2026-10-13 morning: d3-vz-ens9-occl (ens9o) **0.7277** vs ens7-ref 0.7264 (val 0.7541 vs 0.7550).
Beam-consistent paste is val-neutral but test-positive (+0.0013, within noise but in the leader's
direction). Acting on it: 4 more paste models launched 07:58 (a45_occl_s2/s3 2-frame; occl_sf
single-frame floor 0; occl_f1 2-frame floor-1 curriculum) for a paste-heavy final ensemble.
Results (10:50): a45_occl_s2 0.660, a45_occl_s3 0.659, occl_sf (single-frame) 0.664, occl_f1 (floor-1
curriculum) 0.659. Six paste models now: 0.671/0.652/0.660/0.659/0.664/0.659. Chain logs/occl_ens.log
builds occl6 (paste-only) and ens13o (mixed7 + 6 paste) on val and test.
Ensembles (13:40): occl6 (paste-only) val 0.7421 -> `submissions/occl6_tta_ref158_thr05_test.jsonl.gz`;
ens13o (mixed7 + 6 paste) val 0.7543 -> `submissions/ens13o_tta_ref158_thr05_test.jsonl.gz`. Both valid.
Awaiting test scores (suggested names d3-vz-ens13-occl, d3-vz-occl6). occl_sf_s1 / occl_f1_s1 training.
occl_sf_s1 0.655, occl_f1_s1 **0.684** (ep 15, 2nd-best single ever). Eight paste models:
0.671/0.652/0.660/0.659/0.664/0.659/0.655/0.684. Building occl8 (paste-only 8) val + test.

## Round 6 (2026-10-13): diagnosis first (review questions Q1/Q6), on mixed7 + smoother + thr 0.05, val

Decomposition of GT boxes that pass BEV IoU 0.35 but fail 3D IoU 0.35 (7.1% of GT, n=532):
- 77% are at r < 2.5 m. There the vertical 1-D overlap is 0.74 (fine) but BEV IoU is 0.46 (vs 0.62
  for matched boxes): the failure is a loose footprint plus a box ~0.1 m too tall at the BOTTOM edge
  (top error 0.00, bottom error -0.09..-0.11). Predicted vertical span at r<2.5 (median 1.51 m) equals
  the visible wedge (1.51) while GT span is 1.36: labels are smaller than what is visible, i.e. a
  size convention, not FOV truncation. H1 (amodal truncation) in its original form is wrong.
- 17% are at r > 4 m: whole box 0.45 m too low (top and bottom both -0.45), mostly upright divers.
Range-conditioned calibration (dz, l/w/h scale per range bin, fit on val): in-sample +0.002 SDS;
scene-wise leave-one-out CV: z-only 0.7371, dims-only 0.7401, both 0.7224 vs 0.7550 baseline.
The biases are scene-specific (dz per scene from -0.148 to +0.087), so any val-calibrated correction
only transfers to test scenes that happen to share conventions. Closed as a method; it also explains
why the train-fit stage-2 refiner hurt test ASE.
- Review round (14:30-15:00: methodology review, literature search, assumption audit, anomaly mining):
  * Label size vs observed point extent INSIDE THE GT BOX: R2 0.83/0.81/0.90 (l/w/h) vs R2 0.24/0.06/0.48 for
    our predicted dims. Looked decisive but is a selection artefact (the GT box chooses the points):
    extent snapping of predicted boxes (diver3d/snap.py) HURTS (full snap 0.667-0.725, half blend
    0.746-0.752 vs 0.7550). In a ray-aligned frame returns cover only 42/55/72% of the box (far side
    in acoustic shadow), so extent is not observable from one frame. Closed.
  * z prior shrinkage (assumption audit): pred z - GT z = -0.106*GT z - 0.048 on val; train divers
    sit below the sonar, val/test's dominant family above it. Linear z correction: in-sample 0.7545,
    scene-LOO 0.7507 (no gain): not fixable post hoc. -> a45_zrel (`data.z_relative`: z regressed
    relative to the mean z of returns in the 3x3 output cells at the centre), launched 15:05 GPU 0.
  * Failure rate vs body-axis alignment to the ray: 4% perpendicular, 21% oblique, 27% end-on, with
    predicted extents shrinking to 0.90-0.93 of GT for oblique views (assumption-audit measurement).
    Inference probe: upper-quantile h over the track window instead of mean: hq80 0.7551 (neutral).
  * Literature search, top picks: RadarGaussianDet3D box-Gaussian KL loss; LiDAR R-CNN boundary-offset refiner
    features (explains why our PointNet refiner could not learn extent); BtcDet spherical occlusion
    occupancy head; MPPNet trajectory refiner; 9D+SVD rotation (Geist et al. 2024).
  * Anomaly mining: (A) GT-centre elevation train median -3.9 deg vs val +4.2 deg (62% above +2 deg);
    per-scene z slope -0.26 on val vs -0.065 OOF train; only z regresses to a prior. (B) ALL 16
    floor-0/anneal-0.45 runs peak at the first 1-3 clean epochs and decay ~0.004 SDS per clean epoch
    (AP35 and APBEV fall together, n_pred falls, train hm loss falls): the ep-12 'luck' is timing.
    (C) corr(pred IoU, true IoU) on TPs = 0.085; residual unpredictable from input. (E) polar beats
    Cartesian at r<2 m (found@0.35 0.30 vs 0.16 at 0-1.5 m) and loses at 3-4 m.
  * Pitch-TTA test of (A) on a45@ep12: pitch +6 / -6 deg -> SDS 0.6806 / 0.6802 vs 0.6954; scene_0043
    dz -0.136 -> -0.152 / -0.141 (unchanged), other scenes shift by ~+0.03..0.08 / -0.05 with the
    pitch. Elevation extrapolation is NOT the mechanism for 0043; the model sits ~0.2 m below the
    returns there for another reason. a45_zrel (z anchored to returns) is the direct test.
  * occl8 (paste-only 8): val 0.7448 -> `submissions/occl8_tta_ref158_thr05_test.jsonl.gz` (valid).
  * (B) test: occl_f1a45 (floor 1, anneal 0.45, beam-consistent paste) launched 15:05 GPU 1.
  * (C) variance of residuals explained by scene id: dz 0.17, log l 0.27, log w 0.65, log h 0.35.
    Width is mostly convention; z is mostly within-scene scatter.
  * (A) label-free surrogate z correction (centroid z of returns in the predicted box minus predicted z,
    linear fit, scene-LOSO): 0.7549 vs 0.7550 (in-sample 0.7573). No gain on the refined ensemble. Closed.

## Leaderboard 2026-10-13 16:00: ens13-occl 0.7229 (val 0.7543), occl8 0.7145 (val 0.7448), ens9-occl 0.7277.
Val->test drop 0.030 for every ensemble; more paste models = lower test. The ens9o +0.0013 was noise.
Beam-consistent paste closed (no transfer advantage). Best entry stays ens9-occl 0.7277 (= ens7-ref
0.7264 within noise). Extra paste seeds a45_occl_s4 / occl_f1_s2 killed. Remaining live experiments:
a45_zrel (evidence-relative z), occl_f1a45 (floor 1 + anneal 0.45).
- a45_zrel (z relative to mean z of returns in the 3x3 output cells): best selectable 0.624 (ep 13),
  last epochs 0.60-0.61 vs band 0.655-0.693. Worse: the local reference includes floor/clutter returns
  and adds variance to the target. Closed. Per-scene dz: see below.
- Launched 16:25 GPU 0: a45_roty10 (rot_y +-10 deg on real frames) and a45_zshift (whole-frame vertical
  shift +-0.4 m): training-side tests of the elevation-diversity hypothesis.
- occl_f1a45 (floor 1, anneal 0.45): best 0.675 (ep 16); epochs 12-19 = 0.668 0.616 0.672 0.605 0.675
  0.658 0.660 0.660 (last-4 mean 0.663) vs floor-0 twin a45_occl last-4 mean 0.645, best 0.671.
  Floor 1 stops the post-paste decay (anomaly B confirmed) but does not raise the peak. Use floor 1 +
  anneal 0.45 for robustness; no SDS lever.
- a45_roty10 (rot_y +-10 deg): best selectable 0.649, below band -> arm dropped (seed 1 killed).
- a45_zshift (whole-frame z shift +-0.4 m): best 0.683 (ep 15), epochs 12-19 0.665 0.683 0.675 0.661
  0.669 0.644 0.642 0.653. Upper half of the band on one seed; seeds 1-3 running (19:25) for a 4-seed arm.
- Range-gated polar fusion (anomaly E): polar x1 all ranges 0.7539, x1 r<2 m 0.7465, x2 r<2 m 0.7421 vs
  0.7550. Closed.
- zshift per-scene dz (ep 15): 0043 -0.176 (vs -0.136 for a45): the whole-frame z shift does not remove
  the scene-0043 offset either; the scene bias is not a z-prior effect that augmentation can reach.
- a45_zshift, 4 seeds: 0.683 / 0.675 / 0.678 / 0.647 (mean 0.671, sd 0.016) vs a45 reference 5 seeds
  0.693 / 0.655 / 0.670 / 0.673 / 0.657 (mean 0.670). Identical. Closed.

Round-6 summary (2026-10-13 21:20): the review round produced the correct diagnosis (failures
are near-range loose footprints + scene-specific size/z conventions, 65% of width residual is scene
id; z offsets are scene-specific and not reachable by calibration, pitch, z-shift, or evidence-relative
targets; all floor-0 runs decay after pasting stops) but no lever: every intervention is within the
+-0.02 single-seed band on val and the test drop is a constant 0.030 for every ensemble. Final entry
remains ens9-occl 0.7277 (LB rank 5; leader 0.7420). GPUs idle.

## Round 7 (2026-10-15): advisor meeting follow-up, oracles before building

Literature and novelty check on George's suggestions: "3D YOLO" = anchor-based
yaw-only BEV heads, strictly less expressive than our head (no support); multi-view range+BEV fusion
(MVF +3.6/+4.5 3D AP on lidar, near range) and shadow occupancy (WYSIWYG +4.5, BtcDet learned occupancy
+2.3 vs mask +0.3) are the literature-supported changes; nothing applies them to 3D sonar.
Diagnostics on mixed7 val predictions (all before building anything):
- Along-beam extent vs alignment: side-on fail 2.5%, oblique 15.6%, end-on 23.2%. Returns cover 0.47-0.59
  of the GT box along the beam, but predicted extent is 1.02-1.07x GT (1.12x for end-on failures) and
  body-axis length 1.18x (1.29x failures): the detector OVER-extrapolates. Censored/lower-bound extent
  loss would push the wrong way. Closed.
- View-angle diversity within tracks: median |d cos| 0.02 over 1 s, 0.06 over 4 s; 11-20% change
  alignment class. A multi-frame amodal refiner has no view diversity to use. Closed.
- Footprint residual of BEV-ok/3D-fail boxes at r<2.5 m: far edge +0.18 m (too long along the beam),
  near edge -0.06, cross-beam extent 0.96 and centre |err| 0.07: the error is along-beam over-extension,
  not cross-beam.
- Flip analysis: moving failing boxes 25% of the way to GT in both centre and dims flips 76% of them.
- Oracles (component replaced by GT on matched boxes): z +0.029, xy +0.028, dims +0.067, rotation
  +0.061, z+dims +0.092 (val 0.847). Track-mean GT dims +0.028; track-mean GT z -0.094 (z moves within
  tracks). A 10-15% SDS gain requires near-perfect size AND height.
- Variance shares (scene / track-within-scene / frame-within-track): z 0.17/0.05/0.77, log w
  0.65/0.03/0.32, log h 0.35/0.05/0.60. z error lag-1 autocorrelation along tracks +0.94 (smooth,
  state-dependent bias, not noise). GT z unchanged between consecutive frames 62% of the time, GT h 87%,
  GT w 90% (labels are track-level). Predictability of the z error from label-free observables
  (range, elevation, predicted z/dims, tilt, point count, return z stats, intensity, floor estimate,
  score; scene-LOSO): linear ridge R2 -0.26, gradient boosting R2 -0.36 (worse than predicting zero).
  The smooth per-track z bias is scene-conditioned annotation behaviour, not sensor state. No-go on a
  height model.

Round-7 conclusion: the rule-respecting ceiling is within ~0.01-0.02 of the current entry. Final entry
ens9-occl 0.7277. Remaining time is better spent on the write-up (dataset/label-convention findings,
oracle ceilings, seed-count methodology) than on further detector changes.
- Range-view branch built (18:20, Dan's go): `model.rv.mode` only|fuse; 2D U-Net (32 ch) on the native
  64x256 beam image (range, intensity, x, y, z, valid), 32 features per beam gathered back to each
  current-frame point. "only" = point features are xyz + time + beam features (voxel geometry features
  removed); "fuse" = appended to the standard features. 9.88M params. Runs rv_only (GPU 0), rv_fuse
  (GPU 1), seed 0. PRE-REGISTERED kill criterion: rv_only must beat the mean of the 5 reference seeds on
  BEV IoU for GT boxes at r<2.5 m, paired per box, by >= +0.05 (diver3d/paired.py); otherwise the
  direction stops regardless of SDS. Chain logs/rv_chain.log.
  Result (19:55): rv_only best 0.610 (ep 15); paired near-range (r<2.5 m, n=3422) BEV IoU 0.511 vs
  reference-mean 0.547 (diff -0.036, improved 44% of boxes), found@0.35 0.639 vs 0.788. Criterion
  (>= +0.05) FAILS decisively. rv_fuse best 0.651; near-range BEV IoU +0.014 (55% improved), 3D IoU
  +0.004, found@0.35 0.767 vs 0.788: within noise. The beam image carries no footprint information the
  0.1 m voxels lack. Range-view direction CLOSED per pre-registration. GPUs idle. Final entry stays
  ens9-occl 0.7277; remaining time -> write-up.

## Round 8 (2026-10-16): two test-side hypotheses for the leader gap (Dan's go)

Gap decomposition on test: AP3D@0.35 0.724 vs 0.763 is the whole gap; APBEV@0.35 and AP3D@0.50 equal;
geometry terms better. Leader single model 0.714 test vs ours 0.695.
- A. Selection-bias hypothesis: ensemble of LAST-epoch checkpoints of six floor-1 models (vz025,
  vz025_s1, vz025_mf2, occl_f1, occl_f1_s1, occl_f1a45), no val selection. Val 0.7261 raw / 0.7361
  refined (vs 0.7550 for val-selected mixed7) -> `submissions/last6_tta_ref158_thr05_test.jsonl.gz`
  (valid, 18,757 dets). If test > 0.7277 the val->test drop was selection bias.
- B. Calibration-transfer hypothesis: val-fitted per-range-bin dz and l/w/h scale (fit on mixed7 val,
  in-sample 0.7550 -> 0.7571) applied to the ens9o test file ->
  `submissions/ens9o_calib_tta_ref158_thr05_test.jsonl.gz` (valid). Scene-LOSO on val says it will
  not transfer; the leaderboard decides.

## Leaderboard 2026-10-24 22:50: new team APC_TDC holds ranks 1-13 (15 uploads Oct 20-24, no description)
Top: D3D_sub016 **0.7588** (mAP3D 0.7883, AP35 0.7979, AP50 0.4179, APBEV35 **0.8906**, ATE 0.171, ASE
0.302, AOE3D 30.5, tilt 13.7); sub019 0.7579 (AP50 0.4589, ASE 0.296: a tighter-box variant); their
progression sub005 0.7376 (Oct 20) -> sub008 0.7533 -> sub010 0.7557 -> sub016 0.7588, BEV AP 0.869 ->
0.888 -> 0.891. Dev Thakkar 0.7420 now rank 14; StagAI 0.7298 rank 19; ours 0.7290 (rank ~21, 4th team).
Gap to APC_TDC (0.0298): mAP3D 0.0712 x 0.5 = 0.0356 for them; ASE +0.0012 for them; ATE 0.0013 and
AOE3D 0.0057 for us. Their box fit is ordinary (AP50 0.418, ASE 0.302 ~ ours); their edge is DETECTION:
APBEV35 0.891 vs our 0.832 (and vs our own val 0.881): they find/rank divers in BEV far better on the
test scenes than we do even on val. Nothing to copy (no description). First place is out of reach with
3 days left; the entire val oracle headroom would be needed. Recommendation: keep d3-vz-ens9-sc105.

## Round 14 (2026-10-24 23:00, Dan: "we have to try something different"; 3 days left)
Ideation round running: external review of the open problems, literature search, assumption audit.
Launched in parallel the best-supported untested input-side change:
- a45_gscale (GPUs 0/1, seeds 0/1): whole-frame radial scale augmentation aug.scale [0.85, 1.2] (existing
  code path in data.py: xyz, centres and extents scaled about the sonar origin; never enabled). Motivation:
  the 1.05x scale TTA improved near-range fit (found@0.35 0.72 -> 0.75 under 2 m), so the net is not
  scale-robust, and test is near-range dominated. Gate, pre-registered: paired vs the 5 reference p0
  dumps, near-range (< 2 m) found@0.35 >= +0.02 and BEV IoU >= +0.01 for the 2-seed mean, whole-val
  SDS >= 0.660; then the swap ensemble (mixed7 with its two a45 seeds replaced) >= 0.7580 with the near-
  range gain intact; FP burden at matched recall not worse than reference.

### INCIDENT 2026-10-24 23:10: /raid project directory rolled back to 2026-10-16 00:38
The DGX rebooted on 2026-10-19 ~18:30 (uptime 5 d at 23:05 on 10-24); /raid (btrfs) now holds the
project as of 2026-10-16 00:38 (newest surviving run files: runs/rv_only, 10-15 19:48). Lost on /raid:
EXPERIMENTS.md rounds 9-13 (10-16 .. 10-19: beam-paste control, a45x7, floor-1 branch, valres, Gaussian
loss, vertical attention, CVPR draft + its review, false-positive accounting, empty_keep,
track prior, rotation NLL, scale TTA sweep), all runs after 10-15 (a45_bs4*, a45_f1*, near*, gauss*,
vattn*, upfirst, a45_ek5*, a45_f1_ek5*, a45_rotnll*, the proper oof_f1; runs/oof_f1 is again the
all-scenes model), member dumps after 10-16 (a45_bs4, a45_f1, s6, ek5, rotnll, sc095/102/105/110),
analysis/ scripts, paper/, submissions after 10-16 (a45x7, a45x8b, legacy5, floor1x7, ens9o_sc105 and
its test/val member dumps), configs after 10-16, and the code changes after 10-16 (scale TTA, rot_nll,
empty_keep, gauss loss, vattn, near paste, z_shift ...). Survived: notes in the home directory outside /raid (round summaries with
the key numbers), the leaderboard itself (d3-vz-ens9-sc105 0.7290 stands), runs up to 10-15 incl. all
ens9o members, the reference seeds s1-s4 member dumps, paired.py (--r_max). Restored tonight from the
working notes: analysis/prcurve.py, scalediag.py, fpcat.py, fusionaudit.py; the --scale TTA in
train.py/predict.py; ens_mixed7_val.jsonl.gz regenerating via fuse_tune. ACTION FOR DAN: ask the DGX
admins whether a /raid btrfs snapshot newer than 10-16 exists (the 10-19 reboot is the likely cause).
- Diagnostics 23:15 (analysis/detdiag.py, static.py on ens3z_ref158 val 0.7405): oracle GT dims -> APBEV35
  +0.016 only (AP35 +0.042; + GT z: AP35 +0.071); shrinking near-range boxes /1.3: APBEV +0.005, AP35
  -0.042. Footprint is not the BEV gap. Staticness (per-beam causal median range over 15 frames): TPs are
  MORE static (0.40) than FPs (0.29); rescoring by staticness loses 0.025-0.13 APBEV -> background
  subtraction (CFC-style) closed. Adjacent divers: 0.9% of val GT, all found. Ranking: pooled APBEV 0.861
  vs n_gt-weighted per-scene mean 0.879 (AP35 0.769 vs 0.809); per-scene APBEV 0000 0.97, 0043 0.94,
  0048 0.92, 0082 0.95, 0068 0.84, 0080 0.38, 0075 0.18; median TP score 0.26 (< 1.5 m), 0.46 (1.5-2),
  0.54-0.58 (2-5 m). Reading: the BEV loss is cross-recording ranking driven by the sparsely labelled
  scenes' unlabelled blobs and by low near-range scores; no label-free legal fix has survived val.
- Queued behind gscale (arms2_chain.sh): a45_beamdrop (aug.beam_drop 0.5: drop every 2nd/3rd bearing
  column or elevation row, PDDA / LiDAR-Distillation-style density resampling on the beam grid; new code
  in data.py) and a45_rad2 (loss.min_radius 2, gaussian_min_overlap 0.5: larger heat-map positives,
  hotspots/CenterPoint-tau style). Gate for each, pre-registered: paired near-range found@0.35 and BEV
  IoU vs the 4 surviving reference p0 dumps (>= +0.02 / +0.01), best val APBEV35 >= reference, FP at
  matched recall not worse; then the swap ensemble >= mixed7 + 0.003.
- Fusion audit (analysis/fusionaudit.py, fuserules.py on the regenerated mixed7 val 0.7550, 14 member
  passes): some member box reaches 3D IoU 0.35 for 96.2% of GTs (union over ~14 overlapping boxes per
  GT) vs 90.1% for the fused file; under 1.5 m 0.659 vs 0.159, 1.5-2 m 0.902 vs 0.756. But no label-free
  cluster rule recovers it: found@0.35 all ranges wmean 0.888, medoid 0.885, median dims 0.886, p25 dims
  0.873 (0.232 under 1.5 m, worse elsewhere), min dims 0.779, top-score member 0.852, min volume 0.722;
  mean 3D IoU is highest for the weighted mean everywhere except under 1.5 m (+0.01-0.02 for p25 dims).
  The union figure is an oracle over the minority of qualifying boxes inside each cluster. Review hypothesis 1
  (fusion destroys capability) falsified; fusion geometry CLOSED.
- RESULT a45_gscale (2026-10-25 00:45, logs/gscale_eval.log): best val SDS 0.661 / 0.674 (reference mean
  0.670); best APBEV35 0.833 / 0.842; paired vs 4 reference p0 dumps: near-range found@0.35 0.677 / 0.663
  vs 0.723, BEV IoU -0.002 / +0.015, all-range found 0.809 / 0.837 vs 0.879; FP@R0.70 2,378 / 1,888 vs
  1,598 (s2). Swap ensemble 0.7546 (-0.0004), 3/7 scenes up, near-range found 0.720 -> 0.723. GATE
  FAILED: training-time whole-frame scale augmentation does not reproduce the inference-time scale TTA
  gain; it costs near-range recall. CLOSED. beamdrop / rad2 arms now training (arms2_chain.sh).
- RESULTS 2026-10-25 (arms2_chain.log; the chain survived a terminal restart):
  a45_beamdrop (aug.beam_drop 0.5): best val SDS 0.651 / 0.641 (ref mean 0.670); near-range found@0.35
  0.675 / 0.674 vs 0.723; FP@R0.70 2,579 / 3,406 vs 1,598 (s2). GATE FAILED, worse than reference. CLOSED.
  a45_rad2 (min_radius 2, min_overlap 0.5): best val SDS 0.656 / 0.680 (mean 0.668 = ref); dump AP35
  0.668 / 0.714 (s1 is the best single-model dump so far; ref s2 0.709), max recall 0.836 / 0.851 (ref
  0.848), FP@R0.70 2,208 / 1,653 (ref 1,598); near-range found@0.35 0.651 / 0.665 vs 0.723 (gate FAILED on
  the near-range criterion); s1 all-range BEV IoU +0.005, found 0.851. Mixed: one weak and one strong seed,
  gains at mid/far range, none at near range. Two more seeds queued to resolve it (cheapest open question).
- 10:05 Dan: resolve rad2 with two more seeds alongside his geminio jobs. Launched a45_rad2_s2 (GPU 0) and
  s3 (GPU 1) with CUDA_VISIBLE_DEVICES set (pre-launch check 09:55: GPUs 0-3 at 93-98% from geminio, 17-18
  GB; 4-7 another user; shared server idle). Gate, pre-registered: 4-seed rad2 mean best SDS >= 0.675, or
  an ensemble (mixed5 + 4 rad2 / mixed7 + rad2 s1,s3 / rad4 alone) >= 0.7580 with >= 5/7 scenes up and
  near-range found not below mixed7's 0.720; eval_rad2b.sh -> logs/rad2b_eval.log.
- RESULT a45_rad2 x4 (13:55, logs/rad2b_eval.log): best val SDS 0.656 / 0.680 / 0.673 / 0.652 (mean 0.665
  vs reference 0.670; gate >= 0.675 FAILED); dump AP35 0.668 / 0.714 / 0.686 / 0.638; near-range found
  0.651 / 0.665 / 0.696 / 0.662 vs 0.723 (all below); FP@R0.70 2,208 / 1,653 / 2,291 / 3,154 vs 1,598.
  Ensembles vs mixed7 0.7550: mixed5 + 4 rad2 0.7516 (-0.0034, APBEV -0.001); mixed7 + rad2 s1,s3 0.7533
  (-0.0017, AP35 +0.003, APBEV +0.002, 5/7 scenes up, driven by 0068/0075); rad4 alone 0.7389. Seed 1 was
  seed luck. Larger heat-map positives CLOSED. Round 14 summary: scale aug, beam dropout, heat-map radius,
  fusion rules, staticness, footprint oracle: all null or negative on val. Nothing running; GPUs returned.

## Round 15 (2026-10-25 17:35, Dan: "2 then 3"): sonar-native local-structure features, then a diagnostic upload
- a45_local (configs/a45_local.yaml): data.local_feats 2 (per return: mean offset, covariance, log count of
  the returns within +-2 beams on the range image; data.py local_feats/add_local_feats, applied after
  augmentation and per stacked frame) + model.drop_xyz (no absolute xyz in the VFE input; 6 + 12 channels).
  GBlobs (CVPR 2025) adapted to one-return-per-beam sonar. Mirror TTA flips the odd channels (train.py;
  verified exact). Audit: no blockers; W1: pasted divers without beam occlusion carry background returns
  inside the window (cov_xx ~5x that of real divers during the paste epochs; the last 8 epochs are
  paste-free), so a null result is inconclusive about the features; scale/pitch TTA must not be used
  with these checkpoints. Launched s0 (GPU 0) / s1 (GPU 1) with CUDA_VISIBLE_DEVICES, sharing with the
  geminio jobs (pre-launch check: 0-3 at 76-100%, 17-20 GB; shared server idle). Gate as for the other arms;
  eval_local.sh -> logs/local_eval.log (swap and add ensembles vs mixed7 0.7550).
- Diagnostic file for Dan's upload #3: submissions/ens9o_top1_test.jsonl.gz = the 0.7277 entry with only
  the top-scoring box per frame (7,723 boxes, VALID). Val equivalent on mixed7: top-1 costs -0.072 SDS
  (AP35 -0.145, APBEV -0.162; top-2 costs -0.020). Reading rule for the test score: a drop well below
  0.07 => the extra boxes on the multi-object test scenes are mostly false positives; a drop >= 0.07
  => they are divers. Suggested name d3-vz-ens9-top1.
- RESULT a45_local (21:10, logs/local_eval.log): best val SDS 0.649 / 0.632 (ref mean 0.670); dump AP35
  0.650 / 0.628 vs 0.709; near-range found@0.35 0.564 / 0.598 vs 0.723 (BEV IoU -0.053 / -0.018);
  all-range found 0.814 / 0.810 vs 0.879; FP@R0.70 2,398 / 3,276 vs 1,598. Ensembles: swap 0.7512
  (-0.0038), add 0.7543 (-0.0007). GATE FAILED, clearly worse at near range. CLOSED (with the audit's
  caveat: un-occluded pasted divers contaminate the local features during the paste epochs, so this does
  not cleanly refute GBlobs-style features; no time to rerun with beam occlusion). Round 15 training done;
  nothing running. Remaining step: Dan's diagnostic upload of ens9o_top1_test.jsonl.gz.

## Round 16 (2026-10-26 01:35): the top-1 diagnostic upload and close-pair suppression
- LB d3-vz-ens9-top1 (ens9o, only the top-scoring box per frame): **0.6241** (mAP3D 0.5041, AP35 0.5130,
  AP50 0.2798, APBEV35 0.6061, ATE 0.149, ASE 0.317, AOE3D 27.2) vs the full entry 0.7277: -0.104 (val -0.072).
  APBEV top-1/full: val 0.816, test 0.728 -> test has more divers per occupied frame (~1.4 vs 1.27 train /
  1.26 val, if top-1 recall ~ 1/divers-per-frame). Our extra boxes on test are mostly divers, not FPs.
- Test geometry: among confident (>= 0.3) test detections of the 0.7277 entry, NO two lie within 1.0 m
  (nearest-neighbour p1 1.24 m) in 8,112 frames; train GT has 2.6% of divers with a neighbour < 1.0 m and
  9.5% < 1.5 m (val 0.27% / 7.2%). Test scenes 0019/0021/0049 (57% of frames) emit >= 2 confident dets in
  0-2% of frames; 0042/0064 in 71-90%.
- Mechanism test on out-of-fold train (analysis/pairs.py; oof_f0 / oof_f2 fold detectors on their 14 held-out
  scenes, 14,091 boxes, 464 with a neighbour < 1 m, mostly scenes 0091/0065): per-pass BEV NMS 0.1 kills
  close pairs. found@BEV0.35 close / mid (1-1.5 m) / isolated: NMS 0.1 0.657 / 0.773 / 0.865; 0.2 0.821 /
  0.796 / 0.866; 0.3 0.843 / 0.797 / 0.867; 0.5 0.849 / 0.799 / 0.869. Single pass AP35 0.7229 -> 0.7307
  (0.3) -> 0.7334 (0.5), APBEV 0.8034 -> 0.8140 -> 0.8150. Flip-fused: nms 0.1 / fuse 0.30 SDS 0.7316 ->
  nms 0.3 / fuse 0.30 0.7334 (AP35 +0.005, APBEV +0.007); fuse 0.45-0.6 no better.
- Val cost (mixed7 members, fuse_tune sweep): nms 0.3 / fuse 0.30 = 0.7550 (unchanged; AP35 +0.0005, +667
  boxes); fuse 0.45 -0.005; fuse 0.60 -0.042. Relaxing per-pass NMS is free on val (20 close-pair boxes).
- Label-free suppressed-pair statistic (analysis/pairstats.py): OOF train baseline ~0.01 per frame per pass,
  pair-heavy 0065 0.051; val 0.002-0.018. Test values pending (test member dumps at scale 1.0 / 1.05 for
  the 9 ens9o members running on GPUs 0/1 since 02:20, shared with geminio; pre-launch check: shared server idle).
- Candidate: rebuild d3-vz-ens9-sc105 with per-pass NMS 0.3 (fusion 0.30, refine 1/5/8 unchanged) via
  analysis/fuse_test.py. One isolated change, val-neutral, held-out-positive.
- Held-out train with refinement (fusion 0.30, refine 1/5/8): NMS 0.1 0.7412 (AP35 0.7429, APBEV 0.8154,
  close found 0.638) | 0.2 0.7432 (0.7484 / 0.8210 / 0.795) | 0.3 0.7431 (0.7495 / 0.8220 / 0.830) | 0.5 0.7403
  (0.7443 / 0.8198 / 0.832). Val full sweep: nms 0.3 / fuse 0.30 0.7550, nms 0.5 / 0.30 0.7546, every fuse 0.45
  -0.005, fuse 0.60 -0.042. CHOSEN: per-pass NMS 0.3, fusion 0.30 (val-neutral, held-out +0.002 SDS /
  +0.007 AP35 / +0.007 APBEV with 3.3% close-pair boxes; gain on test scales with close-pair prevalence).
- Independent review (02:35): partially supported; endorses
  global per-pass NMS 0.30 with fusion unchanged as the candidate; rejects the top-1 upload as evidence for
  suppressed pairs (the top-1 cut only removes divers the full output already has: it shows test leans more on
  lower-ranked boxes, nothing about suppression). Asked for per-recording deltas, one-to-one pair matching, a
  same-pass double-vote check in fusion, and a calibrated counterfactual statistic with a pre-set rule.
- Per-recording, held-out fold pipeline, NMS 0.1 -> 0.3 (analysis/rescued.py on submissions/train/oof_f02_*):
  pair recordings 0065 +0.0169 SDS (AP35 +0.036; both divers of a close pair matched one-to-one 0.534 -> 0.879),
  0091 +0.0068 (+0.017; 0.419 -> 0.729), 0081 +0.0008 (+0.007; 0.053 -> 0.368); the 11 pair-free recordings
  -0.0025..+0.0024. Rescued-pair rate R (frames where NMS 0.3 adds a confident box 0.35-1.2 m from a box both
  outputs share): pair recordings 0.103 / 0.022 / 0.000, pair-free max 0.0114 (0055), others <= 0.0067.
- PRE-REGISTERED DECISION RULE (written before any test R is computed): B = 0.0114, P = 0.022. If the deployed-
  pipeline R on val (mostly pair-free) is <= ~B, the calibration transfers. Then on test: any recording with
  R >= P, or test-wide R >= P -> evidence of suppressed pairs, upload the NMS 0.3 candidate; all recordings
  R <= B -> no evidence of pairs on test, the upload would test only a val-neutral policy (expected ~0),
  hold unless an upload is otherwise unused; in between -> inconclusive, upload only as a low-risk experiment.
- TEST (02:50): test dumps reproduce the 0.7290 entry exactly (ens9o_sc105_nms01_test: 15,072 dets = uploaded
  file). Candidate ens9o_sc105_nms03_test.jsonl.gz: 15,724 dets, VALID. Rescued-pair rate R on test (candidate
  vs entry): 0019 0.0071 (4 boxes), all other recordings 0.0000; test-wide 0.0005. Raw suppressed-peak count was
  high only in 0019 (0.048/frame/pass) and those peaks re-merge in fusion. RULE OUTCOME: every test recording R
  <= B (0.0114) -> no evidence of suppressed close pairs on test; the candidate only tests a val-neutral policy
  (expected ~0 SDS). Hold unless an upload slot would otherwise go unused. The close-pair mechanism is real on
  held-out train (0065/0091) but the hidden test recordings do not contain enough close pairs for it to matter.
- Last checks (03:05-03:25): fusion double-voting at NMS 0.3 on val = 28 of 7,618 confident clusters (0.4%),
  mean score inflation 0.012 (0 at NMS 0.1) -> negligible. Val analogue of the candidate (mixed7 + sc105
  members): NMS 0.1 0.7599 (reproduces 10-19) -> NMS 0.3 0.7597 (every scene within +-0.0006). Round 16 CLOSED:
  real mechanism, no test exposure. Standing entry d3-vz-ens9-sc105 0.7290; candidate ens9o_sc105_nms03 optional.
- CORRECTION (2026-10-26, from the selected checkpoints): a45_rad2_s3 best_sds = 0.6420 (ep 15), not 0.652 (that
  was the max over all epochs incl. non-selectable ones); 4-seed mean 0.6626. a45_local_s1 best_sds = 0.6366
  (ep 19), not 0.632. a45_ek5 selectable best (ep >= 12): 0.662 / 0.646. Reference 5 seeds 0.6693 +- 0.0153.

## 2026-10-31: team report reviewed (independent review, reproducibility check, results analysis); revised
- Independent review: partially supported; no arithmetic disputed. Adopted: disclosure that validation tuned
  every inference setting, that three-frame inputs and non-causal smoothing use test-recording frames, and that
  a45x7 / ens9-top1 used test-prediction statistics; neutral wording on widths and unlabelled stretches; label-
  informed oracles labelled as sensitivities; ATE/AOE are matched-detection metrics; baseline comparison not
  like-for-like.
- Reproducibility check: every recomputable number matched; fixed 0.641 -> 0.640 (and "two runs", same seed), median
  range 1.9-2.3 m (0049 is 2.15-2.33), "4 confident boxes" wording (NMS 0.3 adds 652 boxes overall), runtime 22
  min per 18 passes (from file timestamps), "more than 80 training runs". Dropped the cross-ensemble r > 0.99
  claim (comparison files lost). Re-derived from surviving files with analysis/rederive.py (logs/rederive.log):
  partial R2 0.638, SDs 0.295/0.050, BLUPs identical, labelled-vs-predicted effect r = 0.04, oracles +0.0724 /
  +0.0323, FP oracle +0.0593 AP35, transitions 19/72, longest gap 651. CORRECTION: median predicted/labelled BEV
  footprint area is 1.52 (<1.5 m) and 1.59 (1.5-2 m), ~1.0 beyond 2 m; the earlier 1.3x / 2.2x was wrong.
- Results analysis: reference 0.6693 +- 0.0152 (95% CI 0.651-0.688); only range-view alone clearly worse; polar
  and oversampling rows were built on the anneal-0.65 configuration (seeds 0.665, 0.642), now footnoted; ten
  seeds per arm (t-test), two seeds vs five resolve ~0.04; Spearman 0.745, bootstrap CI 0.2-0.96; drop vs val
  r = -0.14; close-pair SDS block-bootstrap intervals 0.001-0.049 (0065) and 0.002-0.011 (0091). CORRECTION:
  plain-paste near-range reference is 0.68 (seeds 0.64-0.72), not 0.69; occluding-paste 0.57-0.65 overlaps it.
- Runs lost in the rollback are marked with a dagger in Table 3. Final PDF: paper/FAU_CAAI_Diver3D_MaCVi2027_report.pdf.
