# submodules — external model repositories

This directory holds the external model code SceneConductor depends on. **All four are real git submodules** (pointers recorded in `<repo>/.gitmodules`), populated on clone.

## After cloning SceneConductor

```bash
# Fetch every submodule's content (all four, GALP included)
git submodule update --init --recursive
```

## When cloning fresh with submodules (collaborator workflow)

```bash
git clone --recurse-submodules <SceneConductor.git>
```

## Layout

| Path | Kind | Upstream | Purpose |
|---|---|---|---|
| `submodules/GALP` | git submodule | https://github.com/jhkim0759/GALP.git | Geometry-Aware Layout Prediction. Required by `stage1-initialize-scene/src/run_galp.py` (`galp_repo: ./submodules/GALP`). Weights go in `<repo>/checkpoints/galp/` (read directly by `run_galp.py` — no symlink needed). See `submodules/GALP/README.md`. |
| `submodules/Grounded-SAM` | git submodule | https://github.com/IDEA-Research/Grounded-Segment-Anything.git | GroundingDINO + SAM wrapper. Loaded as a library by `stage1-initialize-scene/src/grounded-sam/run_inference.py`. `setup.sh` patches its `ms_deform_attn_cuda.cu` for modern PyTorch before building the `_C` CUDA extension. |
| `submodules/SAM3D` | git submodule | https://github.com/facebookresearch/sam-3d-objects.git | Textured GLB generator. Required by `stage1-initialize-scene/src/run_sam3d.py`. Weights are **gated** — see `submodules/SAM3D/doc/setup.md`. |
| `submodules/Qwen3.6` | git submodule | https://github.com/QwenLM/Qwen3.6.git | Qwen3.5-VL attribute extractor used in Stage 1. |

All four ship as empty placeholder folders until you run `git submodule update --init --recursive` (or clone with `--recurse-submodules`).

> The easiest way to populate these (plus Blender, the conda envs, and the model
> checkpoints) in one step is the `sceneconductor-setup` skill — see the repo
> `README.md` / `INSTALLATION.md`.

## Why this lives at the repo root and not under `.claude/`

These are large external code dependencies, not Claude skills. Putting them at the repo root makes the submodule contract obvious in `.gitmodules` and keeps `.claude/` focused on the skill/agent specs Claude reads.
