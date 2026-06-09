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

[![arXiv](https://img.shields.io/badge/arXiv-2606.08402-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.08402)
[![Project Page](https://img.shields.io/badge/🏠-Project%20Page-blue.svg)](https://jhkim0759.github.io/projects/SceneConductor/)
[![Model](https://img.shields.io/badge/🤗%20Model-SceneConductor-yellow.svg)](https://huggingface.co/WopperSet/SceneConductor)

## 🔭 Pipeline

<p align="center">
    <img width="95%" alt="pipeline" src="./assets/pipeline.png">
</p>
</h4>

The pipeline runs in three stages, shown above.

- **(a) Stage 1 — Initialize Scene.** GroundedSAM produces masks. An Opus mask-evaluator merges them. SAM 3D turns each object into a textured GLB. GALP predicts the layout (pointmap, floor polygon, coarse placements).
- **(b) Stage 2 — Environment Construction.** An Opus vision director designs a rectilinear floor plan. It builds a separable Floor/Wall/Ceiling stage. A look-dev pass matches the photo. Finally it renders 5 reference views.
- **(c) Stage 3 — Scene Refinement.** A relation graph drives a heuristic + Opus planner pass (attach-to-floor/wall, align, remove). An Opus validator flags problem groups. A dedicated island-refiner agent fixes each group. Then it renders the final 5 views.

## 🚀 Quickstart

> **🧭 Where do I type each command?** Two kinds appear in this README:
> - 💻 **Terminal** — run in your normal shell (e.g. `./setup.sh`, `git`, `conda`).
> - 💬 **Claude Code prompt** — type *inside* the Claude Code CLI, after you run `claude`. These are the `/slash-commands`.

### ✅ Step 1 (recommended) — one-skill setup

Clone, launch Claude Code, and run the **`sceneconductor-setup`** skill. It provisions the whole pipeline end to end — submodules → Blender 4.2.1 → all five conda envs → all model checkpoints → a PASS/FAIL verification — so you go straight from a bare clone to a runnable scene.

```bash
# Terminal
git clone --recursive https://github.com/jhkim0759/SceneConductor.git SceneConductor
cd SceneConductor
claude
```
```text
# Claude Code prompt — provisions AND validates everything
/sceneconductor-setup
```

Under the hood it patches the GroundingDINO CUDA source for the pinned PyTorch (so `groundingdino._C` actually builds — the #1 fresh-clone failure), downloads GroundedSAM / GALP / SAM3D / Qwen weights, and wires the SAM3D symlink. **SAM 3D Objects is a gated Hugging Face repo** — if its download fails, request access at https://huggingface.co/facebook/sam-3d-objects, create a token, then re-run with `HF_TOKEN=hf_xxx`. When the audit prints `RESULT: PASS`, jump to Step 2.

### 💻 Step 1 (manual alternative) — set everything up in the Terminal

Prefer the one-skill setup above. These are the equivalent manual steps if you'd rather drive it yourself.

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/jhkim0759/SceneConductor.git SceneConductor
cd SceneConductor
```

```bash
# If you forgot --recursive, fetch the submodules now
git submodule update --init --recursive
```

```bash
# 2. Download Blender 4.2.1 (Linux x86_64) and verify it
wget https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
tar -xf blender-4.2.1-linux-x64.tar.xz
./blender-4.2.1-linux-x64/blender --version
```

```bash
# 3. Create all five conda envs with one command (~30-60 min; builds CUDA extensions)
./setup.sh
```

```bash
# 4. Download model checkpoints (~25 GB) into ./checkpoints/  (see 📦 Model Checkpoints)
```

```bash
# 5. Stage a scene. A scene is a folder whose only file is image.png
mkdir -p <scene_dir>
cp /path/to/photo.png <scene_dir>/image.png
```

```bash
# 6. Launch Claude Code from the repo root
claude
```

### 💬 Step 2 — Run the pipeline (Claude Code prompt)

You are now inside the `claude` session. Run all three stages with one command:

```text
/scene-orchestration <scene_dir>
```

Or run each stage yourself, in order:

```text
/stage1-initialize-scene <scene_dir>
```
```text
/stage2-environment-construction <scene_dir>
```
```text
/stage3-scene-refinement <scene_dir>
```

✨ **That's the whole flow.** `./setup.sh` builds every conda env. The rest is a single slash command.

> 🍎 On macOS/Windows, install Blender 4.2 by hand. Then set `blender_bin_macos` / `blender_bin_windows` in `DIRECTORYS.yaml`.

## 📋 Prerequisites

- 🐧 **OS:** Linux x86_64. The vendored Blender path targets Linux. macOS/Windows are untested for the Stage 1 SAM3D/GALP GPU paths.
- 🎮 **GPU:** NVIDIA, CUDA 11.8+, ~30 GiB VRAM peak. The SAM3D Stage 1 post-process is the bottleneck.
- 💾 **Disk:** ~50 GB free (Blender ~4 GB + checkpoints ~21 GB + per-scene outputs).
- 💬 **Claude Code CLI** — the pipeline runs on slash commands. Install it from https://github.com/anthropics/claude-code.
- 🐍 **conda / miniconda** — `./setup.sh` builds the five envs. Each is invoked via `conda run -n <name>`.
- 🚫 **Git LFS** is not required.

## 🐍 Environment Setup

You don't build the envs by hand. **💻 `./setup.sh` creates all five.** It pins the exact library versions each stage needs, reads the names from `DIRECTORYS.yaml`, skips envs that already exist, and prints a summary.

```bash
# Create all five envs (skips any that already exist)
./setup.sh
```
```bash
# Delete and rebuild everything
./setup.sh --all --force
```
```bash
# Build only specific envs
./setup.sh --scenegen --qwen-vl
```
```bash
# See all options
./setup.sh --help
```

| Env name | 🐍 Py | Purpose |
|---|---|---|
| `sceneconductor` | 3.11 | Driver + Blender orchestration (pyyaml, numpy, pillow, trimesh, scipy, opencv, shapely…) |
| `scenegen` | 3.10 | GALP layout prediction (torch cu128 + pytorch3d + TRELLIS deps) |
| `grounded-sam` | 3.10 | GroundedSAM inference (GroundingDINO + Segment-Anything CUDA build) |
| `sam3d-objects` | 3.11 | SAM3D textured GLB extraction (official recipe) |
| `qwen-vl` | 3.11 | Qwen3.5-VL attribute extractor (transformers ≥5.5) |

> 💾 The `qwen-vl` env alone is ~10 GB. To put the envs on a larger disk than `/home`, prefix the command. `setup.sh` handles the rest:
> ```bash
> SC_ENVS_DIR=/path/to/large/disk/sceneconductor_envs ./setup.sh
> ```
> 🔧 CUDA source builds (`scenegen`, `grounded-sam`) use `CUDA_HOME` (default `/usr/local/cuda`).

📖 Full per-env breakdown and manual fallback: **[Installation → Conda Environments](./INSTALLATION.md#4-conda-environments--one-command-setupsh)**.

## 📦 Model Checkpoints

Checkpoints are not committed (~25 GB total). The GALP weights live on Hugging Face at [`WopperSet/SceneConductor`](https://huggingface.co/WopperSet/SceneConductor). GroundedSAM, SAM 3D Objects, and Qwen3.5-VL come from their official sources. See **[Installation → Model Checkpoints](./INSTALLATION.md#5-model-checkpoints)** for the target layout and per-model download commands.

**💻 Terminal** — batch a scene without the interactive prompt:

```bash
SCENE_DIR=/path/to/scene FORCE=1 bash build_one_scene_seq.sh
```

🔁 Every stage is resumable. Re-running a stage skips work whose outputs already exist. Use `--force` to override.

## 🧪 Tests

The Stage 1 pipeline has a pytest suite under `tests/stage1/`.

**💻 Terminal**

```bash
# Run the suite
conda run -n sceneconductor python -m pytest tests/stage1 -v
```
```bash
# Or, with the sceneconductor env already active
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
│   ├── GALP/                      # git submodule (jhkim0759/GALP)
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

1. **🔍 "Blender not found"** — check that the `blender_bin_*` key in `DIRECTORYS.yaml` matches your platform. Or run `export BLENDER=/path/to/blender` to override it.
2. **🐍 "Conda env not found"** — re-run 💻 `./setup.sh`. It builds any missing env and skips existing ones. Names must exactly match `conda_envs:` in `DIRECTORYS.yaml`. A mismatch is the most common Stage 1 failure.
3. **🧱 An env failed to build in `./setup.sh`** — the summary marks it. Rebuild just that one, e.g. 💻 `./setup.sh --grounded-sam --force`. A CUDA build error usually means `CUDA_HOME` isn't pointing at a real toolkit.
4. **💥 CUDA OOM during Stage 1 SAM3D** — the SAM3D post-process peaks at ~30 GiB. Close other GPU processes, pin a larger device with `--gpu N`, or use a higher-VRAM card.
5. **⏭️ "Stage skipped — already complete"** — the orchestrator caches per-stage completion. Re-run with `--force` to rebuild from Stage 1.
6. **📂 Submodule directory empty after clone** — you cloned without `--recursive`. Run 💻 `git submodule update --init --recursive` (this populates all four submodules, GALP included).

## 🗺️ Roadmap

- [x] Code release
- [x] Checkpoint release
- [ ] Codex version — an OpenAI Codex / `codex-cli` compatible variant of the pipeline

## 😊 Acknowledgements

We thank all the authors who made their code public. It tremendously accelerated this project.

- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Qwen3.5-VL](https://github.com/QwenLM/Qwen3.6)
- [Blender](https://www.blender.org/)
- Claude Code

## 📚 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{kim2026sceneconductor3dscenegeneration,
      title={SceneConductor: 3D Scene Generation from Single Image with Multi-Agent Orchestration},
      author={Jeonghwan Kim and Yushi Lan and Yongwei Chen and Hieu Trung Nguyen and Chuanyu Pan and Xingang Pan},
      year={2026},
      eprint={2606.08402},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.08402},
}
```
