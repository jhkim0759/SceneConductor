# Installation

This guide walks through setting up **SceneConductor** — a Claude Code skills sandbox that turns a single indoor RGB photograph into a fully populated, look-dev'd Blender 3D scene. The pipeline is driven by Claude Code slash commands and orchestrates several open-source models (GroundedSAM, SAM 3D Objects, GALP, Qwen3.5-VL) plus Claude Opus planning/validation passes.

Throughout this document, `$PROJECT_ROOT` refers to the directory where you clone the repository (e.g. `~/SceneConductor`). Replace it with your own path.

---

## 0. Quick Start (TL;DR)

**Easiest — one skill does everything.** Clone, launch Claude Code, and run the `sceneconductor-setup` skill; it provisions submodules → Blender → all five conda envs → all checkpoints → a PASS/FAIL verification, so you go from a bare clone to a runnable pipeline in one step:

```bash
git clone --recurse-submodules https://github.com/jhkim0759/SceneConductor.git SceneConductor
cd SceneConductor
claude
```
```text
/sceneconductor-setup        # inside Claude Code — provisions AND validates everything
```

> SAM 3D Objects is a **gated** Hugging Face repo. If its download fails, request access at
> https://huggingface.co/facebook/sam-3d-objects, create a token, and re-run with `HF_TOKEN=hf_xxx`.

---

Prefer to drive it manually? Copy‑paste this. It assumes Linux x86_64, an NVIDIA GPU (CUDA 11.8+), and `conda`/`miniconda` already installed. Each step is explained in the sections below.

```bash
# 1. Clone (with submodules) and enter the repo
git clone --recurse-submodules https://github.com/jhkim0759/SceneConductor.git SceneConductor
cd SceneConductor
export PROJECT_ROOT="$PWD"

# 2. Get Blender 4.2.1 (vendored path)
wget https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
tar -xf blender-4.2.1-linux-x64.tar.xz

# 3. Create ALL FIVE conda envs with one command (~30–60 min; builds CUDA extensions)
./setup.sh

# 4. Download the model checkpoints (~25 GB) — see Section 5
#    (GroundedSAM, GALP, SAM3D, Qwen3.5-VL)

# 5. Launch Claude Code from the repo root and run the pipeline
claude
#   then, inside Claude Code:
#   /scene-orchestration scenes/my_room      (a folder containing only image.png)
```

That's the whole install: **one clone, one Blender download, one `./setup.sh`, the checkpoints, done.** `setup.sh` reads the env names from `DIRECTORYS.yaml`, skips any env that already exists, and prints a summary at the end. If a step fails, the matching section below has the details.

> Tip: the `qwen-vl` env alone is ~10 GB. To put the envs on a bigger disk than `/home`, prefix the command:
> ```bash
> SC_ENVS_DIR=/path/to/large/disk/sc_envs ./setup.sh
> ```

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
- Five conda environments are used in total (one shared driver env + one per model). You don't create them by hand — **`./setup.sh` builds all five** (see Section 4).

> Tip: the Qwen-VL env alone can reach ~10 GB. If `/home` is tight, just prefix the setup command with `SC_ENVS_DIR` and `setup.sh` puts the envs on the bigger disk for you:
> ```bash
> SC_ENVS_DIR=/path/to/large/disk/sceneconductor_envs ./setup.sh
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

The external model code lives under `$PROJECT_ROOT/submodules/`. All four are **real git submodules** — populate them with `--recurse-submodules` (or `git submodule update --init --recursive`).

### 3.1 Clone with submodules

```bash
git clone --recurse-submodules https://github.com/jhkim0759/SceneConductor.git SceneConductor
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
| `submodules/GALP` | git submodule | https://github.com/jhkim0759/GALP.git |

> **Note:** all four are listed in `.gitmodules`. After a plain `git clone` (without `--recurse-submodules`) the submodule folders are empty placeholders until you run `git submodule update --init --recursive`.

---

## 4. Conda Environments — one command: `./setup.sh`

Each model runs in its **own conda environment**, invoked by the pipeline as `conda run -n <env-name> python ...`. You don't create these by hand — **`./setup.sh` builds all five for you.** The environment **names** are the single source of truth in `DIRECTORYS.yaml`, and `setup.sh` reads them straight from there:

```yaml
conda_envs:
  sceneconductor: sceneconductor    # py3.11  driver + Blender orchestration (CPU libs)
  galp:           scenegen          # py3.10  GALP inference (torch cu128 + pytorch3d)
  grounded-sam:   grounded-sam      # py3.10  GroundingDINO + Segment-Anything (CUDA build)
  sam3d-objects:  sam3d-objects     # py3.11  SAM 3D Objects (official recipe)
  qwen-vl:        qwen-vl           # py3.11  Qwen3.5-VL attribute extractor
```

### 4.1 Run it

```bash
cd "$PROJECT_ROOT"
./setup.sh
```

That single command:

- creates **all five** conda envs with the exact, known‑good library versions each stage needs (PyTorch CUDA builds, PyTorch3D, GroundingDINO/SAM CUDA extensions, SAM3D, transformers, …);
- **skips** any env that already exists (so it's safe to re‑run);
- runs the post‑steps automatically — the GroundingDINO `.cu` torch-compat patch, the `groundingdino._C` build check, and the `Grounded-Segment-Anything` symlink;
- prints a per‑env **summary** at the end.

Expect **~30–60 min** the first time, mostly compiling CUDA extensions (PyTorch3D, flash‑attn, GroundingDINO).

### 4.2 Options

```bash
./setup.sh --help                  # show all flags
./setup.sh --all --force           # delete + rebuild every env from scratch
./setup.sh --scenegen --qwen-vl    # build only specific env(s)
```

| Flag | Effect |
|---|---|
| *(no args)* / `--all` | Create all five envs (skip existing) |
| `--sceneconductor` | Driver / Blender env only |
| `--scenegen` (`--galp`) | GALP inference env only |
| `--grounded-sam` | GroundedSAM env only |
| `--sam3d` (`--sam3d-objects`) | SAM 3D Objects env only |
| `--qwen-vl` | Qwen3.5-VL env only |
| `--force` | Remove and rebuild the selected env(s) |

**Environment variables it honors:**

- `CUDA_HOME` — system CUDA toolkit used to compile the CUDA extensions (default `/usr/local/cuda`). Needed for the `scenegen` and `grounded-sam` source builds.
- `SC_ENVS_DIR` — if set, appended to conda's `envs_dirs` so the big envs (esp. the ~10 GB `qwen-vl`) land off `/home`:
  ```bash
  SC_ENVS_DIR=/path/to/large/disk/sc_envs ./setup.sh
  ```

### 4.3 What each env is (reference)

You normally never touch these — `setup.sh` handles them. They're listed here only so you know what failed if an env errors out.

| Env | Python | Role | Heavy deps `setup.sh` installs |
|---|---|---|---|
| `sceneconductor` | 3.11 | Stage 2/3 orchestration + Blender drivers | pyyaml, numpy, pillow, trimesh, scipy, opencv, shapely, matplotlib |
| `scenegen` | 3.10 | GALP inference (`run_galp.py`) | torch (cu128), pytorch3d, spconv, xformers, flash-attn, utils3d, moge → `bundle.sh` |
| `grounded-sam` | 3.10 | GroundingDINO + SAM (`run_inference.py`) | torch (cu128), editable `GroundingDINO`/`segment_anything`, yapf |
| `sam3d-objects` | 3.11 | SAM 3D Objects textured GLBs (`run_sam3d.py`) | official `environments/default.yml` + `pip install -e '.[dev/p3d/inference]'` + hydra patch |
| `qwen-vl` | 3.11 | Qwen3.5-VL attribute extractor | torch (cu128), transformers ≥5.5, accelerate, qwen-vl-utils |

> The pinned versions in `setup.sh` reproduce the currently‑working reference host (CUDA 12.x). If you need to change an env name, edit `DIRECTORYS.yaml` — `setup.sh` and the pipeline both read it from there. Mismatched env names are the most common Stage 1 failure.

> **Manual fallback.** If you'd rather build one env by hand (e.g. to debug a build), open `setup.sh` — each env has its own clearly‑labelled `setup_<env>()` function you can read or copy step‑by‑step. The upstream recipes it follows live at `submodules/Grounded-SAM/README.md`, `submodules/SAM3D/doc/setup.md`, and `submodules/GALP/README.md`.

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

`run_galp.py` reads these weights directly from `$PROJECT_ROOT/checkpoints/galp/`
(via the `checkpoints_galp` key in `DIRECTORYS.yaml`), so once the download
completes they resolve automatically — no symlink into `submodules/GALP/` is
required.

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
2. All five conda envs exist with the exact names from `DIRECTORYS.yaml::conda_envs` (the `./setup.sh` summary lists them; double‑check with):
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
2. **"Conda env not found"** — re‑run `./setup.sh` (it skips envs that already exist and builds any that are missing). The env names must exactly match `conda_envs:` in `DIRECTORYS.yaml` — this is the most common Stage 1 failure.
3. **One env failed to build in `./setup.sh`** — the end‑of‑run summary marks it. Rebuild just that one, e.g. `./setup.sh --grounded-sam --force`. A specific env failing does not abort the others.
4. **CUDA extension build errors (`scenegen` / `grounded-sam`)** — the PyTorch3D / flash‑attn / GroundingDINO source builds need a real CUDA toolkit. Set `CUDA_HOME` to it and re‑run, e.g. `CUDA_HOME=/usr/local/cuda-12.1 ./setup.sh --grounded-sam --force`.
5. **No space on `/home` while creating envs** — point conda at a bigger disk: `SC_ENVS_DIR=/path/to/large/disk/sc_envs ./setup.sh`.
6. **`ModuleNotFoundError: GroundingDINO` (Stage 1)** — ensure `submodules/Grounded-SAM` is initialized and the `Grounded-Segment-Anything` path resolves into it; `pip install yapf` in the `grounded-sam` env. (`./setup.sh` creates the symlink and installs `yapf` automatically.)
7. **CUDA OOM during Stage 1 SAM3D** — the post-process peaks at ~30 GiB. Close other GPU processes or pin a larger device with `--gpu N`.
8. **Submodule directory empty after clone** — you cloned without `--recurse-submodules`. Run `git submodule update --init --recursive` (this populates all four, GALP included).
9. **"Stage skipped — already complete"** — all stages are resumable and cache completion. Re-run with `--force` to rebuild from Stage 1.
