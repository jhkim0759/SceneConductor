# GALP — Geometry-Aware Layout Prediction

GALP predicts a coarse room layout from a set of per-object masks and their textured GLBs. Given the segmented objects produced earlier in Stage 1, it outputs a scene pointmap, a floor polygon, and an initial per-object placement (position + orientation). In SceneConductor this is the **final step of Stage 1** — its `layout_prediction.json` (plus the accompanying `.glb`) seeds the Stage 2 environment build and Stage 3 refinement.

## Vendored, not a submodule

This directory is committed to the repository as plain files. It is **not** a git submodule — there is no entry for it in `<repo>/.gitmodules`, and `git submodule update` does nothing for it. After a plain `git clone`, the code here is already present.

## Layout

```
GALP/
├── demo.py            # entry script driven by the pipeline
├── configs/           # inference configs (e.g. mp8_nt512.yaml)
├── src/               # datasets, sam3d_objects model backbone, utils
├── scripts/           # helper scripts
├── assets/            # 0000000 — a bundled sample
└── checkpoints/       # placeholder dir only (real weights live elsewhere)
```

## How it is invoked

Users do **not** run `demo.py` directly. The pipeline calls GALP through
`.claude/skills/stage1-initialize-scene/src/run_galp.py`, which resolves this
directory via the `galp_repo: ./submodules/GALP` key in `DIRECTORYS.yaml` and
executes the model inside the `scenegen` conda env.

## Checkpoints

The `checkpoints/` folder here is an empty placeholder. The real GALP weights
live under `<repo>/checkpoints/galp/`:

```
checkpoints/galp/
├── checkpoint.pt
├── pipeline.yaml
├── galp.yaml
└── condition_embedder.ckpt
```

See the **Model Checkpoints** section of the main `<repo>/README.md` for the
full download layout and total checkpoint footprint.
