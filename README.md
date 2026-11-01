# Diver3D — 9-DoF diver detection from 3D sonar

Code for the MaCVi @ WACV 2027 Diver3D challenge. See `PLAN.md` for the strategy and
the paper/dataset notes.

## Layout

```
configs/base.yaml      paper-faithful dense SonarVoxNet config (grid, model, loss, aug, decode)
diver3d/
  geometry.py          quaternion / 6D / box helpers, mirror and rotation of scenes
  targets.py           CenterPoint heatmap + regression targets on the BEV grid
  data.py              preloaded dataset, GT-sampling bank, augmentation, collate
  model.py             dynamic VFE -> dense 3D convs -> BEV neck -> center head (6D rot); loss; decode
  evalio.py            NMS on true BEV footprint, submission writer, in-process official scoring
  train.py             training loop; scores val every epoch with tools/diver3d_eval.py
  predict.py           checkpoint -> submission (.jsonl.gz) / val score / threshold sweep
tests/test_pipeline.py target->decode round trip (must score SDS 1.0), augmentation invariants
data/                  dataset (git-ignored); release at data/diver3d_v2_extracted/diver3d_v2
runs/<name>/           config.yaml, log.jsonl, last.pt, best_sds.pt, best_ap35.pt
```

## Commands

```bash
python -m tests.test_pipeline                                   # sanity checks, ~1 min
python -m diver3d.train --config configs/base.yaml --name base --gpu 0
python -m diver3d.train --config configs/base.yaml --name noaug --gpu 1 --set aug.enable=false
python -m diver3d.predict --ckpt runs/base/best_sds.pt --split val --sweep       # score-threshold sweep
python -m diver3d.predict --ckpt runs/base/best_sds.pt --split test \
       --out submissions/base.jsonl.gz                          # writes + validates a submission
```

`--set a.b=c` overrides any config key. Logs go to `runs/<name>/log.jsonl` (one JSON per epoch
with train losses and every official val metric).

## Gotchas learned the hard way

- The official exact-cuboid IoU uses a 1e-9 tolerance. Feed it rotation matrices that are
  orthonormal in float64 (we round-trip through the submitted quaternion, see
  `evalio.server_rotation`); a float32 Gram-Schmidt output loses vertices on near-parallel faces
  and the IoU collapses.
- `tools/` in the dataset release ships a module called `diver3d.py` that shadows this package
  if it is prepended to `sys.path`. We append it.
- Train and val have very different pose mixes (train mostly lying, val mostly upright). Val
  SDS is a rough proxy for test.

## License

Code is released under the MIT License (see `LICENSE`). The WACV author-kit files in
`paper/macvi_report/` (`wacv.sty`, `ieeenat_fullname.bst`) keep their own terms. The Diver3D
dataset is not included and is distributed by its authors under CC BY-NC.
