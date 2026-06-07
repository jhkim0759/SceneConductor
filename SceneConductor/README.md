# SceneConductor

Single indoor RGB → fully populated, look-dev'd Blender 3D scene via Claude Code skills.

SceneConductor is a Claude Code skills sandbox that turns one indoor photograph into a complete Blender scene: textured object meshes, a separable Floor/Wall/Ceiling stage, refined per-object placement, and 5-view renders. The pipeline is exposed as slash commands and orchestrates open-source models (GroundedSAM, SAM3D Objects, GALP, Qwen3.5-VL) plus several Claude Opus planning/validation passes.

## Pipeline

| Stage | Skill name | Trigger | What it does | Output |
|---|---|---|---|---|
| 0 (orchestrator) | `scene-orchestration` | `/scene-orchestration <scene_dir>` | Runs all three stages sequentially with resume checks | All outputs below |
| 1 | `stage1-initialize-scene` | `/stage1-initialize-scene <scene_dir>` | GroundedSAM masks → Opus mask-evaluator merge → SAM3D textured GLBs → GALP layout prediction | `inputs/object_class.json`, `inputs/mask_attribute.json`, `inputs/layout_prediction.json` + `.glb`, `inputs/object/*.glb`, `inputs/object_state.json`, `inputs/pointmap_xz.ply`, `inputs/floor.obj`, `inputs/thumbnails/` |
| 2 | `stage2-environment-construction` | `/stage2-environment-construction <scene_dir>` | Inspect → vision director (Opus) → polygon designer → blend build → separable stage → env enhance → 5-view render | `json/blender_scene.json`, `blend/blender_scene.blend`, `blend/stage2-scene.blend`, `render/blender_scene_view_*.png` |
| 3 | `stage3-scene-refinement` | `/stage3-scene-refinement <scene_dir>` | Auto-prep relation graph → heuristic ops → planner review (Opus) → apply + render → validation (Opus) → per-group Opus island refinement (N iters, default 20, configurable via `num_max_iter`) → merge back → final render | `inputs/relation_graph.json`, `json/operation_plan_revised.json`, `json/island_groups.json`, `blend/stage3-scene.blend`, `render/final/blender_scene_view_*.png` |

**Stage 1 — Initialize Scene.** Extracts per-object masks and class labels from the photo, lifts each mask to a textured GLB via SAM3D Objects, and runs GALP to predict a coarse room layout (pointmap, floor polygon, initial object placements).

**Stage 2 — Environment Construction.** A vision director (Opus) reads the image, designs a rectilinear floor polygon, and builds a separable Blender stage (Floor, Wall_NN, Ceiling). It then runs a look-dev pass on lights and stage materials to match the photo, and produces 5 reference renders.

**Stage 3 — Scene Refinement.** A pre-step extracts a relation graph between objects. A heuristic auto-pass produces a draft operation plan; an Opus planner reviews and revises it, including object deletions. After applying the plan and rendering, an Opus validator flags problem relation-groups; each such group is refined by a dedicated Opus island-refiner agent (default 20 iterations) that adjusts per-member translations and yaw targets. Refined groups are merged back and a final 5-view render is produced.

## Quickstart

1. Clone the repository.

   ```bash
   git clone --recursive <repo-url> SceneConductor
   cd SceneConductor
   ```

2. If you cloned without `--recursive`, initialize the real submodules.

   ```bash
   git submodule update --init --recursive
   ```

   Note: `submodules/GALP` is vendored (committed as plain files) and does not require this step. `Grounded-SAM`, `SAM3D`, and `Qwen3.6` do.

3. Download Blender 4.2 and extract it next to the repo (Linux x86_64):

   ```bash
   wget https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
   tar -xf blender-4.2.1-linux-x64.tar.xz
   ```

   On macOS/Windows, install Blender 4.2 manually and update `blender_bin_macos` or `blender_bin_windows` in `DIRECTORYS.yaml`.

4. Create the five conda environments (see the **Environment Setup** section for details).

   ```bash
   conda create -n sceneconductor python=3.11 -y
   conda activate sceneconductor
   pip install pyyaml numpy pillow trimesh
   # Then create scenegen / grounded-sam / sam3d-objects / qwen-vl following each submodule README.
   ```

5. Download model checkpoints into `./checkpoints/` (see the **Model Checkpoints** section). Total ~21 GB.

6. Sanity-check the Blender binary the pipeline will use.

   ```bash
   ./blender-4.2.1-linux-x64/blender --version
   ```

7. Run the full pipeline on a scene directory containing a single `image.png`.

   ```bash
   mkdir -p scenes/my_room
   cp /path/to/photo.png scenes/my_room/image.png
   # Inside Claude Code:
   /scene-orchestration scenes/my_room
   ```

## Prerequisites

- **OS:** Linux x86_64 (the vendored Blender path targets Linux; macOS/Windows are untested for the Stage 1 SAM3D/GALP GPU paths).
- **GPU:** NVIDIA, CUDA 11.8+, ~30 GiB VRAM peak (SAM3D Stage 1 post-process is the bottleneck).
- **Disk:** ~50 GB free (Blender ~4 GB + checkpoints ~21 GB + per-scene outputs).
- **Claude Code CLI** — the pipeline is driven by slash commands. Install per https://github.com/anthropics/claude-code.
- **conda / miniconda** — five environments are referenced via `conda run -n <name>`.
- **Git LFS** is not required.

## Environment Setup

| Env name | Purpose | Where to follow detailed install |
|---|---|---|
| `sceneconductor` | Python 3.11, stdlib + Blender drivers (pyyaml, numpy, pillow, trimesh) | This README, step 4 |
| `scenegen` | GALP layout prediction (torch + pytorch3d) | `submodules/GALP/README.md` |
| `grounded-sam` | GroundedSAM inference | `submodules/Grounded-SAM/README.md` (after submodule init) |
| `sam3d-objects` | SAM3D textured GLB extraction | `submodules/SAM3D/README.md` (after submodule init) |
| `qwen-vl` | Qwen3.5-VL attribute extractor | `submodules/Qwen3.6/README.md` (after submodule init) |

The Qwen-VL env can grow to ~10 GB. If `/home` is tight, install conda envs under an external prefix before creating them:

```bash
conda config --append envs_dirs /path/to/large/disk/sceneconductor_envs
```

See the comment block at the top of `DIRECTORYS.yaml` for the same tip.

## Model Checkpoints

Checkpoints are not committed. Target layout:

```
checkpoints/
├── grounded-sam/        # GroundingDINO + SAM weights         (~3 GB)
├── galp/                # checkpoint.pt, pipeline.yaml,
│                        # galp.yaml, condition_embedder.ckpt   (~2 GB)
├── sam3d/               # SAM3D Objects weights                (~6 GB)
└── qwen/Qwen3.5-27B/    # local copy of Qwen3.5-VL             (~10 GB)
```

Total: ~21 GB.

### GroundedSAM

```bash
git submodule update --init submodules/Grounded-SAM
# Follow the weight-download instructions in submodules/Grounded-SAM/README.md
# (GroundingDINO checkpoint + SAM ViT-H .pth). Place the files under
# ./checkpoints/grounded-sam/ so they are discoverable by the Stage 1 mask runner.
```

### GALP

```bash
# submodules/GALP is vendored — no submodule init needed.
# Follow submodules/GALP/README.md to obtain checkpoint.pt, pipeline.yaml,
# galp.yaml, and condition_embedder.ckpt.
# Place all four files under ./checkpoints/galp/.
```

### SAM3D Objects

```bash
git submodule update --init submodules/SAM3D
# Then follow the upstream weight-download instructions in
# submodules/SAM3D/README.md (look for "Download Pretrained Weights").
# Place the resulting files under ./checkpoints/sam3d/ so they are
# discoverable by .claude/skills/stage1-initialize-scene/src/run_sam3d.py
```

### Qwen3.5-VL

```bash
git submodule update --init submodules/Qwen3.6
# Option A — local copy: download the model into ./checkpoints/qwen/Qwen3.5-27B/.
# Option B — HuggingFace cache: leave qwen_vl_model_id: Qwen/Qwen3.5-27B in
# DIRECTORYS.yaml and let the Transformers cache resolve it on first use.
```

## Configuration

`DIRECTORYS.yaml` is the single source of truth for machine-specific paths. It has 11 keys:

```yaml
blender_bin:         ./blender-4.2.1-linux-x64/blender
blender_bin_linux:   ./blender-4.2.1-linux-x64/blender
blender_bin_windows: C:/Program Files/Blender Foundation/Blender 4.2/blender.exe
blender_bin_macos:   /Applications/Blender.app/Contents/MacOS/Blender

checkpoints_grounded_sam: ./checkpoints/grounded-sam
checkpoints_galp:         ./checkpoints/galp
checkpoints_qwen_vl:      ./checkpoints/qwen/Qwen3.5-27B

galp_repo:  ./submodules/GALP
sam3d_repo: ./submodules/SAM3D

conda_envs:
  sceneconductor: sceneconductor
  scenegen:       scenegen
  grounded_sam:   grounded-sam
  sam3d_objects:  sam3d-objects
  qwen_vl:        qwen-vl

qwen_vl_model_id: Qwen/Qwen3.5-27B
```

macOS and Windows users should edit the `blender_bin_macos` / `blender_bin_windows` key to point at their local Blender 4.2 install.

A runtime override is available: set `BLENDER=/path/to/blender` to bypass all `blender_bin*` keys.

## Usage

```bash
/scene-orchestration scenes/my_room                  # end-to-end
/stage1-initialize-scene scenes/my_room              # per-stage
/stage2-environment-construction scenes/my_room
/stage3-scene-refinement scenes/my_room
```

Flags:

- `--gpu N` — Stage 1 only; pin SAM3D and GALP to a specific CUDA device.
- `--island-refine-iter N` — Stage 3 only; per-group Opus island refinement iteration count (default 20). Equivalent to `num_max_iter`.
- `--force` — re-run from Stage 1, bypassing all resume checks.

All stages are resumable: re-invoking a stage skips work whose outputs already exist on disk. Use `--force` to override.

## Tests

Skills are invoked from inside the Claude Code CLI: type the slash command followed by a scene directory. The four entry points are:

```text
/stage1-initialize-scene scenes/my_room
/stage2-environment-construction scenes/my_room
/stage3-scene-refinement scenes/my_room
/scene-orchestration scenes/my_room          # end-to-end (runs all three)
```

The Stage 1 pipeline has a pytest suite under `tests/stage1/`. Run it with:

```bash
conda run -n sceneconductor python -m pytest tests/stage1 -v
# or, with the sceneconductor env active:
pytest tests/stage1
```

## Repository Layout

```
SceneConductor/
├── .claude/
│   ├── agents/                    # subagent definitions (Haiku/Opus per agent)
│   ├── skills/                    # per-stage skill folders (SKILL.md + src/)
│   ├── rules/                     # shared norm files
│   └── settings.json
├── submodules/
│   ├── GALP/                      # vendored (no upstream)
│   ├── Grounded-SAM/              # git submodule
│   ├── SAM3D/                     # git submodule
│   └── Qwen3.6/                   # git submodule
├── scripts/                       # batch runners (build_all_scenes.sh, etc.)
├── tests/                         # stage1 pytest
├── DIRECTORYS.yaml                # machine-specific paths
├── CLAUDE.md / AGENTS.md          # project rules
├── README.md
├── checkpoints/                   # gitignored — user downloads
├── blender-4.2.1-linux-x64/       # gitignored — user downloads
├── data/                          # gitignored — external datasets
├── tmp/                           # gitignored — throwaway scripts (per CLAUDE.md)
├── arxiv/ + .archive/             # gitignored — code snapshots
└── tasks/                         # gitignored — planning docs
```

## Outputs per scene_dir

```
<scene_dir>/
├── image.png                          # INPUT — the only required file
├── inputs/                            # Stage 1 outputs (masks, GLBs, layout)
├── json/                              # Stage 2 + 3 JSON state
├── blend/                             # Stage 2 + 3 .blend files
├── render/blender_scene_view_*.png    # Stage 2 5-view renders
├── render/final/                      # Stage 3 final 5-view renders
├── relation_groups/                   # Stage 3 per-group islands
└── logs/                              # Stage 1 logs
```

`inputs/relation_graph.json` is produced by Stage 3's auto pre-step (`stage3-sub-scene-analyze-prepare`), not by Stage 1.

## Troubleshooting

1. **"Blender not found"** — verify the `blender_bin_*` key in `DIRECTORYS.yaml` matches your platform, or `export BLENDER=/path/to/blender` to override.
2. **"Conda env not found"** — confirm all five environments exist with the exact names declared under `conda_envs:` in `DIRECTORYS.yaml`. Mismatched names are the most common Stage 1 failure.
3. **CUDA OOM during Stage 1 SAM3D** — the SAM3D post-process peaks at ~30 GiB. Close other GPU processes, pin a larger device via `--gpu N`, or run on a higher-VRAM card.
4. **"Stage skipped — already complete"** — the orchestrator caches per-stage completion. Re-run with `--force` to bypass resume and rebuild from Stage 1.
5. **Submodule directory empty after clone** — you cloned without `--recursive`. Run `git submodule update --init --recursive`. The `submodules/GALP` directory is vendored and will be populated regardless.

## Roadmap

- [x] Code release
- [x] Checkpoint release
- [ ] Codex version — an OpenAI Codex / `codex-cli` compatible variant of the pipeline

## Acknowledgements

- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) — IDEA Research
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Meta AI / FAIR
- [Qwen3.5-VL](https://github.com/QwenLM/Qwen3.6) — Alibaba
- [Blender](https://www.blender.org/) — Blender Foundation
- Claude Code — Anthropic
