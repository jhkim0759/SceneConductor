# submodules — external model repositories

This directory holds the external model code SceneConductor depends on. Most are real **git submodules** (pointers into `<repo>/.gitmodules`), populated on clone. One — `GALP` — is **vendored** as plain committed files and needs no submodule step.

## After cloning SceneConductor

```bash
# Fetch every git-submodule's content
git submodule update --init --recursive
```

`submodules/GALP` is committed directly, so it is already present after a plain `git clone` regardless of the command above.

## When cloning fresh with submodules (collaborator workflow)

```bash
git clone --recurse-submodules <SceneConductor.git>
```

## Layout

| Path | Kind | Upstream | Purpose |
|---|---|---|---|
| `submodules/GALP` | vendored (plain files) | — | Geometry-Aware Layout Prediction. Required by `stage1-initialize-scene/src/run_galp.py` (`galp_repo: ./submodules/GALP`). Weights go in `<repo>/checkpoints/galp/`. See `submodules/GALP/README.md`. |
| `submodules/Grounded-SAM` | git submodule | https://github.com/IDEA-Research/Grounded-Segment-Anything.git | GroundingDINO + SAM wrapper. Loaded as a library by `stage1-initialize-scene/src/grounded-sam/run_inference.py`. |
| `submodules/SAM3D` | git submodule | https://github.com/facebookresearch/sam-3d-objects.git | Textured GLB generator. Required by `stage1-initialize-scene/src/run_sam3d.py`. |
| `submodules/Qwen3.6` | git submodule | https://github.com/QwenLM/Qwen3.6.git | Qwen3.5-VL attribute extractor used in Stage 1. |

The three git submodules ship as empty placeholder folders until you run `git submodule update --init --recursive`. `GALP` is always populated because it is checked into the repo.

## Why this lives at the repo root and not under `.claude/`

These are large external code dependencies, not Claude skills. Putting them at the repo root makes the submodule contract obvious in `.gitmodules` and keeps `.claude/` focused on the skill/agent specs Claude reads.
