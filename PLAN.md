# Diver3D challenge plan (MaCVi @ WACV 2027)

Written 2026-10-07. Challenge page: https://macvi.org/workshop/macvi27/challenges/diver3d

## 0. Hard dates (AoE = UTC-12)

| Date | What |
|---|---|
| **2026-10-09** | Start OpenReview registration if you don't have an account (activation can take 2 weeks) |
| 2026-10-23 | Author details + OpenReview profile confirmed (needed to be in the summary paper) |
| **2026-10-27 23:59 AoE** | Final submission deadline (20 days from today) |
| 2026-10-30 | Results / winner notification |
| 2026-11-20 | Camera-ready |

Submit at: https://macvi.org/upload?track=Diver3D%20Sonar%20Detection
Leaderboard: https://macvi.org/leaderboard/underwater/diver3d/diver3d-sonar-detection
Support: MaCVi Discord (https://macvi.org/discord)

## 1. The task in one paragraph

Input: one 3D sonar frame = N points `(x, y, z, intensity)`, N in [428, 11965].
Output: every diver as a **9-DoF oriented box**: centre (3), extents l/w/h (3), full rotation
R in SO(3) (given as quaternion `(w,x,y,z)`, 6D, or 3x3), plus a score in [0,1]. Single class.
27.6% of frames have no diver and are scored as true negatives (false positives there hurt).

Frame: right-handed, metres, **x forward (boresight), y left, z up**.
Body axes = columns of R: `ex` = torso forward / chest normal (extent l), `ey` = right-to-left
shoulder (extent w), `ez` = feet-to-head (extent h, median 1.74 m).
Yaw-pi symmetry only (`Rz(pi)`) is folded in AOE3D; up/down and front/back are NOT ambiguous.

## 2. Metric (what to optimise)

```
SDS = ( 3*mAP3D + (1 - min(1, ATE/1m)) + (1 - ASE) + (1 - min(1, AOE3D/90deg)) ) / 6
mAP3D = mean AP3D@{0.30, 0.35, 0.40}      (all-point interpolated, exact oriented-cuboid IoU)
```
TP errors are measured over matches at IoU >= 0.35. Ties: AP3D@0.50 then AOE3D.
Half the score is detection, the other half splits evenly across centre, scale, and full-3D
orientation. Working out the paper's final model (SDS 0.673):

| term | value | contribution to SDS |
|---|---|---|
| mAP3D | 0.690 | 0.345 |
| 1 - ATE | 0.817 | 0.136 |
| 1 - ASE | 0.664 | 0.111 |
| 1 - AOE3D/90 | 0.487 | 0.081 |

So the orientation term is the weakest (46 deg error) and scale is second. Both are also what
limits AP3D at tight IoU. Improving orientation and extents is worth more than chasing recall.

## 3. Where the bar is (leaderboard, 2026-10-08)

| Team | Model | SDS | mAP3D | ATE | ASE | AOE3D |
|---|---|---|---|---|---|---|
| StagAI | d3-ens5 | 0.7216 | 0.733 | 0.171 | 0.293 | 36.5 |
| StagAI | d3-f5 (single model?) | 0.6866 | 0.673 | 0.172 | 0.315 | 37.1 |
| hcm | g1 | 0.4774 | 0.262 | 0.214 | 0.313 | 35.4 |
| paper, SonarVoxNet | | 0.673 | 0.690 | 0.183 | 0.336 | 46.2 |
| paper, + causal refinement | | 0.719 | 0.764 | 0.169 | 0.327 | 43.7 |

Takeaways: a faithful SonarVoxNet reproduction lands around 0.67. The leader's single model is
already ~10 deg better on AOE3D than the paper, and their ensemble adds ~0.035 SDS. To be
competitive we need (a) the baseline, (b) a real orientation improvement, (c) ensembling/TTA.

## 4. What the four papers give us

### 2610.01644 SonarVoxNet (the baseline; read this one closely)
- Pipeline: crop to `[0,12] x [-5,5] x [-2.5,2.5]` m (keeps 89% of points, 99.87% of divers);
  voxelize at `(0.1, 0.1, 0.5)` m -> 120 x 100 x 10 grid, <=35 pts/voxel, <=8000 voxels;
  VoxelNet VFE -> 128-ch voxel features; sparse 3D conv middle stage collapses z 10 -> 2;
  2D conv encoder/decoder -> 768-ch 60x50 BEV map (0.2 m stride); 1x1 heads.
- Head (CenterPoint-style): heatmap (Gaussian focal, alpha=2 beta=4, min overlap 0.7),
  offset (dx,dy), z, log(l,w,h), **6D rotation** (first two columns of R, Gram-Schmidt at
  inference). All losses unit weight, L1 regression at GT centres.
- Decode: top-100 peaks after 3x3 max-pool, score > 0.30, rotated-BEV NMS at IoU 0.10.
- Train: AdamW lr 1e-3, wd 1e-2, clip 35, 2-phase cosine, 20 epochs, batch 4, FP32, 1x A100.
  Checkpoint = best 3-epoch moving average of val AP3D@0.35 excluding first 20%.
- **Key ablation**: dense 3D conv backbone == sparse backbone (SDS 0.576 vs 0.574). The grid is
  tiny, so we can start with *plain dense PyTorch 3D convs* and skip spconv entirely.
- Rotation parameterisation barely matters (6D / 9D+SVD / quat / axis-angle all 45-56 deg).
  Yaw-only -> SO(3) is the big jump (AP3D@0.5 x10). AOE3D floor ~45 deg is attributed to sensor
  resolution, but StagAI's 36 deg shows there is headroom.
- **Augmentation is huge**: GT-sampling (paste real diver returns into frames) with an
  annealed target count: 4 inserted divers for first 65% of training, then -1 per epoch down
  to a floor of 1. SDS 0.574 -> 0.673. Insertion is "aspect-preserving, constrained by sonar
  view" (keep bearing/body-aspect relation of the pasted return).
- Causal refinement (inference only, +0.046 SDS): track by BEV overlap, IMM Kalman on centre
  (R^3) and rotation (SO(3) tangent), shrink-only size. Residual regressors fit on val scenes.
  Never adds/removes detections. Legal under the rules as long as it only uses test *inputs*
  causally (worth confirming on Discord; rule 2 bans *training* on test point clouds).
- Data: 36 scenes / 37,010 frames / 34,568 boxes; 21/7/8 scene split; 1.3 divers per non-empty
  frame; 34.8% upright, 30.6% lying, rest in between. Four individuals; every split has >=3.

### 2608.27795 uScenes (sensor + data format; Diver3D is a labelled subset of this)
- Sensor: **Water Linked Sonar 3D-15**, 256 x 64 range image, 90 deg H x 40 deg V FOV, ~6 Hz.
  Point = `r * [cos(psi)cos(theta), cos(psi)sin(theta), sin(psi)]` with theta/psi linear in
  (u,v). **The point cloud is really a range image**: 0.35 deg azimuth and 0.625 deg elevation
  spacing. At 12 m that is 7 cm x 13 cm; at 3 m it is 2 cm x 3 cm. Elevation is the coarse axis.
- Intensity: range-compensated (`a * r^2`), normalised by the frame's 99th percentile, clipped
  to [0,1]. So intensity is *relative within a frame*, not calibrated.
- Vendor frame is x fwd / y right / z down; the released binaries flip y. Challenge frame is
  x fwd / y left / z up. **Check FORMAT.md and verify with a plot before trusting any axis.**
- Platform: BlueROV2, moving. Frames in a scene are 6 Hz consecutive -> temporal context is
  available and cheap (previous frames of the same scene).

### 2608.29433 FLS + 3D sonar calibration (background)
Mostly irrelevant to the challenge (camera/FLS fusion is prohibited). Useful bits: 3D sonar is
noisy with spurious returns near objects; statistical outlier removal (SOR) helped PointNet
segmentation (IoU 0.42 -> 0.63). A cheap SOR / intensity-threshold preprocessing ablation is
worth one experiment.

### 2609.32156 AquaBEV-Nav (background)
RGB -> BEV occupancy for navigation; not directly applicable. The transferable idea is the
**polar / range-causal representation** (PolarNet, PolarFormer): since our sensor is natively
polar, a polar-grid or range-image backbone matches the sampling geometry better than a
Cartesian voxel grid where near-range voxels hold 30 points and far-range voxels hold 1.

## 5. Strategy

Order of work, each step gated on a local val score from the organisers' `tools/diver3d_eval.py`.

1. **Data + eval harness (day 1-2).** Verify SHA-256, read FORMAT.md, write a loader, plot
   a few frames with GT boxes (BEV + side view) to confirm axes and the body-frame convention.
   Score the GT against itself (expect SDS 1.0) and an empty submission (expect ~0.0) to learn
   the evaluator. Run `validate_submission.py` on a dummy file.
2. **Dense SonarVoxNet reproduction (day 3-6).** Own lean PyTorch code: VFE -> dense 3D convs
   (120x100x10 is tiny; an H200 will do batch 32+) -> 2D BEV -> CenterPoint head with 6D rot.
   Target: val AP3D@0.35 ~0.5 without aug, ~0.7 with GT-sampling curriculum. Submit once to
   validate the end-to-end pipeline on the real leaderboard.
3. **Orientation and scale (day 7-12).** This is where SDS is cheapest to buy:
   - geodesic / symmetric-aware loss on R (fold Rz(pi) in the loss, not just the metric);
   - predict extents as residuals from dataset priors (h median 1.74 m; divers are near-rigid in
     size) and consider fixing extents to class means at inference as an ablation;
   - multi-frame input: concatenate the previous k frames (ego-motion compensated if pose is
     in the release, raw stacking if not) with a time channel; nuScenes-style sweeps;
   - range-image / polar branch: 2D CNN on the native 256x64 (range, intensity) image fused
     into BEV, or a polar voxel grid;
   - IoU-aware score head (predict IoU, rescore) for better AP ranking.
4. **Inference tricks (day 13-18).** Flip-y TTA (mirror the box: `R' = [F ex, -F ey, F ez]`,
   `F = diag(1,-1,1)`), multi-seed + multi-config ensemble with oriented weighted box fusion,
   score-threshold sweep against SDS (false positives on empty frames matter), causal
   refinement a la the paper (tracking + Kalman on centre / rotation).
5. **Submit the ensemble (by Oct 25 to leave slack).** Keep a submissions log with val SDS vs.
   leaderboard SDS for calibration.

## 6. Environment notes (this box)

- 8x H200, base conda has torch 2.10.0+cu128 and CUDA works (driver 550 / CUDA 12.4).
- No spconv / OpenPCDet / mmdet3d installed. Not needed for the dense path. If we want sparse
  later: `pip install spconv-cu124` (matches the driver; cu126 also exists, cu128 does not).
- Evaluator is NumPy + SciPy, CPU only.
- Dataset (3.45 GB, SHA-256 `73bcee12...9cb`) is being pulled from the Google Drive folder into
  `data/raw/`. Licence CC BY-NC 4.0; cite uScenes + SonarVoxNet.
- Baseline GitHub (era-research-lab/uScenes) is **empty** as of today, so we write our own code.

## 7. Rules checklist

- Train on `train` only; `val` labels for model selection / local scoring only.
- No training, self-supervision, or test-time adaptation on val/test point clouds.
- External data / simulation / pre-training allowed but must be disclosed.
- No camera imagery.
- <= 100 detections per frame, scores in [0,1], <= 150 MB upload, omitted frames = empty.

## 8. Findings from the data (2026-10-07, after download)

Data lives at `data/diver3d_v2_extracted/diver3d_v2/` (SHA-256 verified). Scorer verified:
GT-as-prediction -> SDS 1.0, empty -> 0.0, ~5 s per val run, so every experiment can be scored.

- **Each frame is literally a 256 x 64 range image.** Azimuth step = 90/255 deg, elevation step
  = 40/63 deg, zero duplicate (u,v) cells over 209 sampled frames, 19-66% of beams filled
  (mean 42%). So `(u, v) -> (range, intensity)` is a lossless 2-channel image. A 2D range-image
  detector (RangeDet / LaserNet style) or a polar voxel grid is a first-class option, not a hack.
- **Divers are close.** Centre x: median 2.9 m (train) / 2.5 m (val), 99th pct 6.5-6.8 m.
  A `[0, 8]` m crop with 0.05 m voxels would still cover ~99% of divers. Worth an ablation
  against the paper's `[0,12]` @ 0.1 m.
- **Train and val differ in pose mix** (tilt = angle between sonar z and body ez):

  | split | frames | boxes | empty | upright (<15 deg) | lying (75-105) | median l,w,h |
  |---|---|---|---|---|---|---|
  | train | 20,861 | 19,482 | 26.6% | 19.6% | 40.8% | 0.90, 0.82, 1.76 |
  | val | 8,037 | 7,509 | 25.9% | 52.1% | 15.2% | 0.85, 0.88, 1.58 |

  The whole-dataset numbers quoted on the site (34.8% / 30.7%) imply test is somewhere in between
  or more upright again. Do not let the model learn train's "mostly lying" prior: rotation
  augmentation about the sonar x-axis / body-pose resampling in GT-sampling matters, and val SDS
  is only a rough proxy for test SDS.
- Extents differ across splits too (val height median 1.58 vs train 1.76), so predicted extents
  should come from the points, not a memorised prior; ASE is 1/6 of the score.
- No head-down instances (tilt > 165 deg: 0%), so the `Rx(pi)` / `Ry(pi)` ambiguity the metric
  refuses to fold is rarely exercised, but a wrong head/feet flip costs 180 deg of AOE3D.
- `instance_id` is stable within a scene: usable for building tracks for the causal refinement
  stage and for diver-wise GT-sampling banks.

## 8. Paper re-read (2026-10-12) and the physics-aware direction

What the four papers say that bears on our remaining gap (AP3D@0.35 on test; near-range, lying,
box too low):
- SonarVoxNet limitations (p.8): z collapsed 10->2 before BEV ("future work: sonar domain's own
  decoder"); no acoustic-sensor physics or diver-specific priors; non-rigid full-rotation labels are
  inexact. p.6: orientation near-random at close range (sensor resolution floor). p.7: residual
  centre error is "irreducible vertical scatter set by range-elevation resolution, not a correctable
  bias" (consistent with our failed global z shift). IoU-aware rescoring and promoting sub-threshold
  boxes "do not generalize across seeds"; learned size refiner eroded loose-IoU AP (consistent with
  our stage-2 result on test).
- Sensor (uScenes p.4): 256x64 beams on a regular grid, 90x40 deg, one return per beam, intensity
  is range-compensated and per-frame normalised (not comparable across frames). Vertical FOV +-20
  deg: at 2 m the visible window is ~1.5 m tall, so near-range divers are truncated by the wedge.
- Annotation (SonarVoxNet p.4): hand-placed boxes of the "complete perceived body extent"; partially
  observed divers get the "plausible full body extent". So near-range boxes extend beyond the returns.
- Shadowing is real and modelled synthetically elsewhere (2608.29433 p.4); AquaBEV-Nav (p.6) found
  observability must be predicted, not assumed, behind occluders.
- No paper ablates multi-frame input or visibility/occupancy features.

Direction chosen (novel w.r.t. the leader, who adds beam-occluding paste but no sensor features):
1. Visibility-aware detector: ray-march every beam through the voxel grid and feed free / shadow /
   observed fractions as three dense channels into the 3D CNN (`model.visibility`), 7 ms per
   batch of 8 on an H200 (diver3d/model.py visibility_volume).
2. Beam-consistent paste (`aug.gt_sample.occlude`): one return per beam after pasting (pasted
   divers shadow the beams behind them, are hidden behind nearer background) and pasted returns
   outside the +-45 x +-20 deg wedge are dropped while the label keeps the full extent
   (FOV-truncation-aware training).
3. Next if 1-2 help: sonar-native elevation decoding (keep z resolution to the head / classify z
   over bins), range-conditioned uncertainty on z, size and rotation fed into box fusion.
Runs launched 2026-10-12 21:42: a45_vis, a45_occl, a45_vis_occl (base = vz025_mf2_floor0_a45).
