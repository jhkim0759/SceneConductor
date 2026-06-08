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

## 🚀 Quickstart

> **🧭 Where do I type this?** Two kinds of commands appear throughout this README:
> - 💻 **Terminal** — run in your normal shell (e.g. `./setup.sh`, `git`, `conda`).
> - 💬 **Claude Code prompt** — type *inside* the Claude Code CLI after you run `claude`. These are the `/slash-commands`.

### 💻 Terminal — set everything up

```bash
# 1️⃣  Clone with submodules
git clone --recursive https://github.com/example/SceneConductor.git SceneConductor
cd SceneConductor
git submodule update --init --recursive   # only if you forgot --recursive
# (GALP is vendored; Grounded-SAM / SAM3D / Qwen3.6 are the real submodules)

# 2️⃣  Get Blender 4.2.1 (Linux x86_64)
wget https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
tar -xf blender-4.2.1-linux-x64.tar.xz
./blender-4.2.1-linux-x64/blender --version   # sanity check

# 3️⃣  Create ALL FIVE conda envs with ONE command (~30–60 min; builds CUDA extensions)
./setup.sh

# 4️⃣  Download model checkpoints (~25 GB) into ./checkpoints/   (see 📦 Model Checkpoints)

# 5️⃣  Stage a scene — a folder whose only file is image.png
mkdir -p scenes/my_room
cp /path/to/photo.png scenes/my_room/image.png

# 6️⃣  Launch Claude Code from the repo root
claude
```

### 💬 Claude Code prompt — run the pipeline

Once you're inside the `claude` session, type:

```text
/scene-orchestration scenes/my_room
```

✨ **That's the whole flow** — `./setup.sh` builds every conda env in the terminal, and the rest is a single slash command in the prompt.

> 🍎 On macOS/Windows, install Blender 4.2 manually and update `blender_bin_macos` / `blender_bin_windows` in `DIRECTORYS.yaml`.

## 📋 Prerequisites

- 🐧 **OS:** Linux x86_64 (the vendored Blender path targets Linux; macOS/Windows are untested for the Stage 1 SAM3D/GALP GPU paths).
- 🎮 **GPU:** NVIDIA, CUDA 11.8+, ~30 GiB VRAM peak (SAM3D Stage 1 post-process is the bottleneck).
- 💾 **Disk:** ~50 GB free (Blender ~4 GB + checkpoints ~21 GB + per-scene outputs).
- 💬 **Claude Code CLI** — the pipeline is driven by slash commands. Install per https://github.com/anthropics/claude-code.
- 🐍 **conda / miniconda** — the five environments are created for you by 💻 `./setup.sh` and invoked via `conda run -n <name>`.
- 🚫 **Git LFS** is not required.

## 🐍 Environment Setup

You don't build these by hand — **💻 `./setup.sh` creates all five** with the exact library versions each stage needs, reads the names straight from `DIRECTORYS.yaml`, skips envs that already exist, and prints a summary.

```bash
./setup.sh                    # 💻 all five envs (skip existing)
./setup.sh --all --force      # 💻 delete + rebuild everything
./setup.sh --scenegen --qwen-vl   # 💻 only specific env(s)   ·   ./setup.sh --help
```

| Env name | 🐍 Py | Purpose |
|---|---|---|
| `sceneconductor` | 3.11 | Driver + Blender orchestration (pyyaml, numpy, pillow, trimesh, scipy, opencv, shapely…) |
| `scenegen` | 3.10 | GALP layout prediction (torch cu128 + pytorch3d + TRELLIS deps) |
| `grounded-sam` | 3.10 | GroundedSAM inference (GroundingDINO + Segment-Anything CUDA build) |
| `sam3d-objects` | 3.11 | SAM3D textured GLB extraction (official recipe) |
| `qwen-vl` | 3.11 | Qwen3.5-VL attribute extractor (transformers ≥5.5) |

> 💾 The `qwen-vl` env alone is ~10 GB. To put the envs on a bigger disk than `/home`, just prefix the command — `setup.sh` handles the rest:
> ```bash
> SC_ENVS_DIR=/path/to/large/disk/sceneconductor_envs ./setup.sh
> ```
> 🔧 CUDA source builds (`scenegen`, `grounded-sam`) use `CUDA_HOME` (default `/usr/local/cuda`).

📖 Full per-env breakdown and manual fallback: **[Installation → Conda Environments](./INSTALLATION.md#4-conda-environments--one-command-setupsh)**.

## 📦 Model Checkpoints

Checkpoints are not committed (~25 GB total). The GALP weights live on Hugging Face at [`WopperSet/SceneConductor`](https://huggingface.co/WopperSet/SceneConductor); GroundedSAM, SAM 3D Objects, and Qwen3.5-VL come from their official sources. See **[Installation → Model Checkpoints](./INSTALLATION.md#5-model-checkpoints)** for the exact target layout and per-model download commands.

## 🎬 Usage

**💬 Claude Code prompt** — run the whole pipeline, or one stage at a time:

```text
/scene-orchestration scenes/my_room              # end-to-end (runs all three)

/stage1-initialize-scene scenes/my_room          # …or per-stage
/stage2-environment-construction scenes/my_room
/stage3-scene-refinement scenes/my_room
```

**💻 Terminal** — batch a scene without the interactive prompt:

```bash
SCENE_DIR=/path/to/scene FORCE=1 bash build_one_scene_seq.sh
```

🔁 All stages are resumable: re-invoking a stage skips work whose outputs already exist on disk. Use `--force` to override.

## 🧪 Tests

The Stage 1 pipeline has a pytest suite under `tests/stage1/`.

**💻 Terminal**

```bash
conda run -n sceneconductor python -m pytest tests/stage1 -v
# or, with the sceneconductor env active:
pytest tests/stage1
```

## 📁 Repository Layout

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
├── setup.sh                       # one-shot conda env provisioner (all 5 envs)
├── DIRECTORYS.yaml                # machine-specific paths
├── INSTALLATION.md                # full install guide
├── CLAUDE.md / AGENTS.md          # project rules
├── README.md
├── checkpoints/                   # gitignored — user downloads
└── blender-4.2.1-linux-x64/       # gitignored — user downloads
```

## 📂 Outputs per scene_dir

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

## 🛠️ Troubleshooting

1. **🔍 "Blender not found"** — verify the `blender_bin_*` key in `DIRECTORYS.yaml` matches your platform, or `export BLENDER=/path/to/blender` to override.
2. **🐍 "Conda env not found"** — re-run 💻 `./setup.sh` (it builds any missing env, skips existing). Names must exactly match `conda_envs:` in `DIRECTORYS.yaml` — mismatches are the most common Stage 1 failure.
3. **🧱 An env failed to build in `./setup.sh`** — the summary marks it; rebuild just that one, e.g. 💻 `./setup.sh --grounded-sam --force`. A CUDA build error usually means `CUDA_HOME` isn't pointing at a real toolkit.
4. **💥 CUDA OOM during Stage 1 SAM3D** — the SAM3D post-process peaks at ~30 GiB. Close other GPU processes, pin a larger device via `--gpu N`, or run on a higher-VRAM card.
5. **⏭️ "Stage skipped — already complete"** — the orchestrator caches per-stage completion. Re-run with `--force` to bypass resume and rebuild from Stage 1.
6. **📂 Submodule directory empty after clone** — you cloned without `--recursive`. Run 💻 `git submodule update --init --recursive`. The `submodules/GALP` directory is vendored and will be populated regardless.

## 🗺️ Roadmap

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
