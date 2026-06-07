# Installation

This guide walks through setting up **SceneConductor** — a Claude Code skills sandbox that turns a single indoor RGB photograph into a fully populated, look-dev'd Blender 3D scene. The pipeline is driven by Claude Code slash commands and orchestrates several open-source models (GroundedSAM, SAM 3D Objects, GALP, Qwen3.5-VL) plus Claude Opus planning/validation passes.

Throughout this document, `$PROJECT_ROOT` refers to the directory where you clone the repository (e.g. `~/SceneConductor`). Replace it with your own path.

---

## 1. General Environment Setup

### 1.1 Operating system

- **Linux x86_64** is the primary, tested platform. The default vendored Blender path and the Stage 1 SAM3D/GALP GPU paths target Linux.
- **macOS / Windows** are partially supported: the Blender binary path is configurable (see `DIRECTORYS.yaml`), but the Stage 1 GPU models (SAM3D, GALP, Qwen-VL) are untested off Linux.

### 1.2 GPU and CUDA

- **NVIDIA GPU required.** CUDA **11.8+**.
- Peak VRAM: **~30 GiB** during the Stage 1 SAM3D post-process (the pipeline bottleneck). A 32 GB+ card is recommended. Qwen-VL needs **> 16 GiB** free on its assigned device.
- Stage 1 lets you pin a CUDA device with `--gpu N`.

### 1.3 Python and conda

- **conda / miniconda** is required. The pipeline invokes models through `conda run -n <env-name> python ...`, so the environment **names** must match what `DIRECTORYS.yaml` declares (see Section 4).
- The base driver environment uses **Python 3.11**.
- Five conda environments are used in total (one shared driver env + one per model). See Section 4.

> Tip: the Qwen-VL env alone can reach ~10 GB. If `/home` is tight, point conda at an external prefix **before** creating the envs:
> ```bash
> conda config --append envs_dirs /path/to/large/disk/sceneconductor_envs
> ```

### 1.4 Blender

- **Blender 4.2.x** (default vendored build: `blender-4.2.1-linux-x64`).
- Download and extract it inside `$PROJECT_ROOT` (Linux x86_64):
  ```bash
  cd "$PROJECT_ROOT"
  wget https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
  tar -xf blender-4.2.1-linux-x64.tar.xz
  ./blender-4.2.1-linux-x64/blender --version   # sanity check
  ```
- On macOS / Windows, install Blender 4.2 manually and edit the `blender_bin_macos` / `blender_bin_windows` keys in `DIRECTORYS.yaml`.
- A runtime override is available: `export BLENDER=/path/to/blender` takes precedence over every `blender_bin*` key in `DIRECTORYS.yaml`.

### 1.5 Disk

- Roughly **~50 GB** free: Blender (~4 GB) + model checkpoints (~21 GB) + per-scene outputs.

---

## 2. Claude Code

SceneConductor is **driven by Claude Code**. The entire pipeline is exposed as skills and subagents under `$PROJECT_ROOT/.claude/`, invoked as slash commands from inside the Claude Code CLI.

### 2.1 Install the Claude Code CLI

Follow the official instructions at https://github.com/anthropics/claude-code, then launch it from the repo root:

```bash
cd "$PROJECT_ROOT"
claude
```

### 2.2 How the repo is structured for Claude Code

- `.claude/skills/` — per-stage skill folders. Each has a `SKILL.md` (trigger conditions + usage) plus a `src/` directory of Python wrappers.
- `.claude/agents/` — subagent definitions (Opus/Haiku planning and validation agents).
- `.claude/rules/` — shared norm files; `.claude/settings.json` — harness config.

### 2.3 The pipeline skills

The user-facing entry points (slash commands) are:

| Stage | Slash command | Role |
|---|---|---|
| 0 (orchestrator) | `/scene-orchestration <scene_dir>` | Runs all three stages sequentially with resume checks |
| 1 | `/stage1-initialize-scene <scene_dir>` | GroundedSAM masks → Opus mask-evaluator merge → SAM3D textured GLBs → GALP layout prediction |
| 2 | `/stage2-environment-construction <scene_dir>` | Vision director (Opus) → floor polygon → separable Floor/Wall/Ceiling stage → env look-dev → 5-view render |
| 3 | `/stage3-scene-refinement <scene_dir>` | Relation graph → heuristic ops → Opus planner review → apply + render → Opus validation → per-group Opus island refinement → final render |

A scene directory needs only one input file: `<scene_dir>/image.png`. Everything else is produced by the pipeline.

```bash
mkdir -p "$PROJECT_ROOT/scenes/my_room"
cp /path/to/photo.png "$PROJECT_ROOT/scenes/my_room/image.png"
# Then, inside Claude Code:
#   /scene-orchestration scenes/my_room
```

---

## 3. Cloning with Submodules

The external model code lives under `$PROJECT_ROOT/submodules/`. Three of these are **real git submodules**; one (GALP) is **vendored** (committed as plain files).

### 3.1 Clone with submodules

```bash
git clone --recurse-submodules <repo-url> SceneConductor
cd SceneConductor
```

### 3.2 If you already cloned without `--recurse-submodules`

```bash
git submodule update --init --recursive
```

### 3.3 The submodules

| Path | Kind | Upstream |
|---|---|---|
| `submodules/Grounded-SAM` | git submodule | https://github.com/IDEA-Research/Grounded-Segment-Anything.git |
| `submodules/SAM3D` | git submodule | https://github.com/facebookresearch/sam-3d-objects.git |
| `submodules/Qwen3.6` | git submodule | https://github.com/QwenLM/Qwen3.6.git |
| `submodules/GALP` | **vendored (not a submodule)** | — (no `.gitmodules` entry; already present after a plain clone) |

> **Note:** `submodules/GALP` is committed directly into the repo. `git submodule update` does nothing for it — it is always populated after a plain `git clone`. The three git submodules appear as empty placeholder folders until you run `git submodule update --init --recursive`.

---

## 4. Per-Submodule Environment Setup

Each model runs in its **own conda environment**, invoked by the pipeline as `conda run -n <env-name> python ...`. The environment **names** are the single source of truth in `DIRECTORYS.yaml`:

```yaml
conda_envs:
  sceneconductor: sceneconductor    # default — stdlib + Blender drivers
  galp:           scenegen          # GALP inference (torch + pytorch3d)
  grounded-sam:   grounded-sam      # GroundedSAM inference
  sam3d-objects:  sam3d-objects     # SAM3D textured GLB
  qwen-vl:        qwen-vl           # Qwen3.5-VL attribute extractor
```

If you change any name here, update `DIRECTORYS.yaml` to match — mismatched env names are the most common Stage 1 failure.

### 4.1 `sceneconductor` — driver / Blender env (Python 3.11)

Lightweight env that runs the stdlib + Blender driver scripts and the Stage 2/3 orchestration Python.

```bash
conda create -n sceneconductor python=3.11 -y
conda activate sceneconductor
pip install pyyaml numpy pillow trimesh
```

### 4.2 `scenegen` — GALP env (torch + pytorch3d)

GALP needs **PyTorch (CUDA build)** and **PyTorch3D**. The Stage 1 wrapper `.claude/skills/stage1-initialize-scene/src/run_galp.py` imports `pytorch3d.transforms` and `pytorch3d.renderer`, and resolves the GALP repo via `galp_repo: ./submodules/GALP`.

```bash
conda create -n scenegen python=3.11 -y
conda activate scenegen
# Install a CUDA-matched PyTorch build, then PyTorch3D, then GALP deps.
# Follow submodules/GALP/README.md for the exact dependency list.
```

> **Note:** GALP does not ship a standalone `requirements.txt`. Install PyTorch (CUDA 11.8+), PyTorch3D, OmegaConf, and the remaining imports referenced by `run_galp.py`. Consult `submodules/GALP/README.md` and `submodules/GALP/src/` for the precise versions.
>
> `DIRECTORYS.yaml` maps the GALP role to the `scenegen` env (`conda_envs.galp: scenegen`), which is authoritative.

After the env exists, run the one-time GALP runtime bundling helper, which symlinks the weight files and copies the small configs into the runtime dir:

```bash
bash .claude/skills/stage1-initialize-scene/src/galp_runtime/bundle.sh
```

### 4.3 `grounded-sam` — GroundedSAM env

GroundedSAM (GroundingDINO + SAM) is loaded as a library by `.claude/skills/stage1-initialize-scene/src/grounded-sam/run_inference.py`, and the wrapper `run_grounded_sam.py` dispatches into the `grounded-sam` env.

```bash
conda create -n grounded-sam python=3.10 -y
conda activate grounded-sam
pip install -r submodules/Grounded-SAM/requirements.txt
# Build/install GroundingDINO + Segment-Anything per the upstream README:
#   submodules/Grounded-SAM/README.md
pip install yapf   # required by the GroundingDINO import path used in Stage 1
```

> **Note:** Stage 1 expects `submodules/Grounded-SAM/Grounded-Segment-Anything` to resolve to the populated submodule. If you hit `ModuleNotFoundError: GroundingDINO`, verify that path is a valid (non-stale) link/dir into the initialized submodule.

### 4.4 `sam3d-objects` — SAM 3D Objects env

SAM3D generates the textured GLBs. The wrapper `.claude/skills/stage1-initialize-scene/src/run_sam3d.py` resolves the repo via `sam3d_repo: ./submodules/SAM3D` and reads its config from `submodules/SAM3D/checkpoints/hf/pipeline.yaml`.

```bash
conda create -n sam3d-objects python=3.10 -y
conda activate sam3d-objects
pip install -r submodules/SAM3D/requirements.txt
pip install -r submodules/SAM3D/requirements.inference.txt
pip install -r submodules/SAM3D/requirements.p3d.txt   # PyTorch3D-related deps
# requirements.dev.txt is optional (development only).
```

Follow `submodules/SAM3D/README.md` for any build steps the upstream repo requires.

### 4.5 `qwen-vl` — Qwen3.5-VL env

Qwen3.5-VL extracts per-object attributes (used by Stage 1's `extract_object_state.py` and Stage 3's analyze-prepare step). Default model id: `Qwen/Qwen3.5-27B`.

```bash
conda create -n qwen-vl python=3.10 -y
conda activate qwen-vl
# Install transformers (Qwen-VL compatible), accelerate, torch (CUDA), and the
# Qwen-VL utility deps per submodules/Qwen3.6/README.md.
```

> This env can grow to ~10 GB. Consider the external `envs_dirs` tip from Section 1.3.

---

## 5. Model Checkpoints

Checkpoints are **not committed**. The pipeline expects them under `$PROJECT_ROOT/checkpoints/`, with locations registered in `DIRECTORYS.yaml`:

```
checkpoints/
├── grounded-sam/        # GroundingDINO + SAM weights        (~3 GB)
├── galp/                # GALP weights + configs             (~6 GB)
├── sam3d/               # SAM 3D Objects weights             (~6 GB)
└── qwen/Qwen3.5-27B/    # local copy of Qwen3.5-VL           (~10 GB)
```

Total: **~25 GB**.

### 5.1 GroundedSAM (official sources)

The Stage 1 wrapper looks for these exact filenames under `checkpoints/grounded-sam/`:

- `groundingdino_swint_ogc.pth` — GroundingDINO Swin-T OGC checkpoint
- `sam_vit_h_4b8939.pth` — Segment Anything ViT-H checkpoint

Download both from the upstream project and place them under `checkpoints/grounded-sam/`. See the download instructions in:
- https://github.com/IDEA-Research/Grounded-Segment-Anything
- https://github.com/facebookresearch/segment-anything (SAM ViT-H weights)

```bash
mkdir -p "$PROJECT_ROOT/checkpoints/grounded-sam"
# Place groundingdino_swint_ogc.pth and sam_vit_h_4b8939.pth here.
```

### 5.2 GALP (Hugging Face)

The GALP checkpoints are published on the Hugging Face Hub at
[`WopperSet/SceneConductor`](https://huggingface.co/WopperSet/SceneConductor),
already laid out under `checkpoints/galp/`. Download them straight into place
with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"          # if hf is not installed
hf download WopperSet/SceneConductor \
  --include "checkpoints/galp/*" \
  --local-dir "$PROJECT_ROOT"
```

This populates `$PROJECT_ROOT/checkpoints/galp/` with:

```
checkpoints/galp/
├── checkpoint.pt                    # trained GALP checkpoint   (~3.6 GB)
├── condition_embedder.ckpt          # frozen condition embedder (~2.4 GB)
├── ss_enc_conv3d_16l8_fp16.safetensors  # sparse-structure encoder (~114 MB)
├── ss_enc_conv3d_16l8_fp16.json     # encoder config
├── galp.yaml                        # generator config
└── pipeline.yaml                    # pipeline config
```

The vendored `submodules/GALP/checkpoints/` is a symlink to
`$PROJECT_ROOT/checkpoints/galp/`, so once the download completes the GALP repo
resolves its weights automatically. `run_galp.py` expects exactly this layout.

### 5.3 SAM 3D Objects (official source)

Download the SAM 3D Objects pretrained weights from the upstream repo and follow its "Download Pretrained Weights" instructions:
- https://github.com/facebookresearch/sam-3d-objects

```bash
mkdir -p "$PROJECT_ROOT/checkpoints/sam3d"
# Download per submodules/SAM3D/README.md.
```

> **Note:** the documented target layout is `checkpoints/sam3d/`, while the Stage 1 wrapper reads its config from `submodules/SAM3D/checkpoints/hf/pipeline.yaml`. Place the HuggingFace-format SAM3D weights so that `submodules/SAM3D/checkpoints/hf/pipeline.yaml` resolves — symlinking `submodules/SAM3D/checkpoints/hf` to your downloaded weights is the safest approach. Confirm against `submodules/SAM3D/README.md`.

### 5.4 Qwen3.5-VL (official source)

Two options:

- **Local copy** — download the model into `checkpoints/qwen/Qwen3.5-27B/` (matches `checkpoints_qwen_vl` in `DIRECTORYS.yaml`).
- **HuggingFace cache** — leave `qwen_vl_model_id: Qwen/Qwen3.5-27B` in `DIRECTORYS.yaml` and let the Transformers cache resolve it on first use.

Official source:
- https://github.com/QwenLM/Qwen3.6 (and the corresponding Qwen3.5-VL weights on HuggingFace)

```bash
mkdir -p "$PROJECT_ROOT/checkpoints/qwen/Qwen3.5-27B"
# Download the Qwen3.5-VL weights here, or rely on the HF cache.
```

---

## 6. Configuration Reference

`DIRECTORYS.yaml` at the repo root is the single source of truth for machine-specific paths. Key entries:

```yaml
blender_bin:         ./blender-4.2.1-linux-x64/blender   # default fallback
blender_bin_linux:   ./blender-4.2.1-linux-x64/blender
blender_bin_windows: C:/Program Files/Blender Foundation/Blender 4.4/blender.exe
blender_bin_macos:   /Applications/Blender.app/Contents/MacOS/Blender

checkpoints_grounded_sam: ./checkpoints/grounded-sam
checkpoints_galp:         ./checkpoints/galp
checkpoints_qwen_vl:      ./checkpoints/qwen/Qwen3.5-27B

galp_repo:  ./submodules/GALP
sam3d_repo: ./submodules/SAM3D

qwen_vl_model_id: Qwen/Qwen3.5-27B
```

Blender resolution order: `$BLENDER` env var → `blender_bin_<os>` for the current platform → `blender_bin` fallback. Paths starting with `./` are relative to the repo root; paths starting with `/` are absolute host paths.

---

## 7. Verify the Installation

1. Blender resolves:
   ```bash
   ./blender-4.2.1-linux-x64/blender --version
   ```
2. All five conda envs exist with the exact names from `DIRECTORYS.yaml::conda_envs`:
   ```bash
   conda env list
   ```
3. Checkpoints are in place under `checkpoints/` (including `checkpoints/galp/` from Hugging Face).
4. Run the pipeline on a scene with a single `image.png` from inside Claude Code:
   ```text
   /scene-orchestration scenes/my_room
   ```

---

## 8. Troubleshooting

1. **"Blender not found"** — check the `blender_bin_*` key for your platform in `DIRECTORYS.yaml`, or `export BLENDER=/path/to/blender`.
2. **"Conda env not found"** — the env names must exactly match `conda_envs:` in `DIRECTORYS.yaml`. This is the most common Stage 1 failure.
3. **`ModuleNotFoundError: GroundingDINO` (Stage 1)** — ensure `submodules/Grounded-SAM` is initialized and the `Grounded-Segment-Anything` path resolves into it; `pip install yapf` in the `grounded-sam` env.
4. **CUDA OOM during Stage 1 SAM3D** — the post-process peaks at ~30 GiB. Close other GPU processes or pin a larger device with `--gpu N`.
5. **Submodule directory empty after clone** — you cloned without `--recurse-submodules`. Run `git submodule update --init --recursive`. `submodules/GALP` is vendored and is populated regardless.
6. **"Stage skipped — already complete"** — all stages are resumable and cache completion. Re-run with `--force` to rebuild from Stage 1.
