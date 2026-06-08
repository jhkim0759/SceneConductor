<h1 align="center">SceneConductor: 3D Scene Generation from Single Image with Multi-Agent Orchestration</h1>

<h4 align="center">

[Jeonghwan Kim](https://jhkim0759.github.io/)<sup>1</sup>,
[Yushi Lan](https://nirvanalan.github.io/)<sup>2</sup>,
[Yongwei Chen](https://cyw-3d.github.io/)<sup>1</sup>,
[Hieu Trung Nguyen](https://hieu1999210.github.io/)<sup>3</sup>,
[Chuanyu Pan](https://pptrick.github.io/)<sup>3</sup>,
[Xingang Pan](https://xingangpan.github.io/)<sup>1</sup>

<sup>1</sup>Nanyang Technological University &nbsp;&middot;&nbsp;
<sup>2</sup>University of Oxford &nbsp;&middot;&nbsp;
<sup>3</sup>Meshy AI

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg?logo=arXiv)](#)
[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://jhkim0759.github.io/projects/SceneConductor/)
[![Model](https://img.shields.io/badge/🤗%20Model-SceneConductor-yellow.svg)](https://huggingface.co/WopperSet/SceneConductor)

## 🔭 Pipeline

<p align="center">
    <img width="95%" alt="pipeline" src="./assets/pipeline.png">
</p>
</h4>

The pipeline runs in three resumable stages, illustrated above:

- **(a) Stage 1 — Initialize Scene.** GroundedSAM masks + Opus mask-evaluator merge → SAM 3D textured GLBs → GALP layout prediction (pointmap, floor polygon, coarse placements).
- **(b) Stage 2 — Environment Construction.** An Opus vision director designs a rectilinear floor plan, builds a separable Floor/Wall/Ceiling stage, runs a look-dev pass to match the photo, and renders 5 reference views.
- **(c) Stage 3 — Scene Refinement.** A relation graph drives a heuristic + Opus planner pass (attach-to-floor/wall, align, remove); an Opus validator flags problem groups, each refined by a dedicated island-refiner agent before a final 5-view render.

## Quickstart

1. Clone the repository.

   ```bash
   git clone --recursive https://github.com/example/SceneConductor.git SceneConductor
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

Checkpoints are not committed (~25 GB total). The GALP weights live on Hugging Face at [`WopperSet/SceneConductor`](https://huggingface.co/WopperSet/SceneConductor); GroundedSAM, SAM 3D Objects, and Qwen3.5-VL come from their official sources. See **[Installation → Model Checkpoints](./INSTALLATION.md#5-model-checkpoints)** for the exact target layout and per-model download commands.

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

## 😊 Acknowledgements

We thank all the authors who made their code public, which tremendously accelerated this project.

- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) — IDEA Research
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) — Meta AI / FAIR
- [Qwen3.5-VL](https://github.com/QwenLM/Qwen3.6) — Alibaba
- [Blender](https://www.blender.org/) — Blender Foundation
- Claude Code — Anthropic

## 📚 Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{sceneconductor2026,
  title     = {SceneConductor: 3D Scene Generation from Single Image with Multi-Agent Orchestration},
  author    = {Jeonghwan Kim and Yushi Lan and Yongwei Chen and Hieu Trung Nguyen and Chuanyu Pan and Xingang Pan},
  booktitle = {Arxiv},
  year      = {2026}
}
```
