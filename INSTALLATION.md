# Installation

Manual, step-by-step setup for **SceneConductor**. Prefer the one-command
`sceneconductor-setup` skill (see the README Quickstart) — it does everything
below automatically. This guide is the manual alternative and the per-piece
reference.

Throughout, `$PROJECT_ROOT` is your clone directory (e.g. `~/SceneConductor`).

**Requirements:** Linux x86_64 · NVIDIA GPU, CUDA 11.8+, ~30 GiB peak VRAM ·
`conda`/`miniconda` · ~50 GB free disk.

> ⚠️ **SAM 3D Objects is a gated Hugging Face repo.** If its download fails,
> request access at https://huggingface.co/facebook/sam-3d-objects, create a
> token at https://huggingface.co/settings/tokens, then re-run that step with
> `HF_TOKEN=hf_xxx`.

---

## 1. Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/jhkim0759/SceneConductor.git SceneConductor
cd SceneConductor
export PROJECT_ROOT="$PWD"
```

Already cloned without `--recurse-submodules`? Populate the four submodules
(Grounded-SAM, SAM3D, Qwen3.6, GALP) now:

```bash
git submodule update --init --recursive
```

---

## 2. Blender 4.2.1

Linux x86_64 — download and extract inside `$PROJECT_ROOT`:

```bash
cd "$PROJECT_ROOT"
wget https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
tar -xf blender-4.2.1-linux-x64.tar.xz
./blender-4.2.1-linux-x64/blender --version   # sanity check
```

macOS / Windows: install Blender 4.2 manually, then set `blender_bin_macos` /
`blender_bin_windows` in `DIRECTORYS.yaml`. A runtime override
`export BLENDER=/path/to/blender` takes precedence over every `blender_bin*` key.

---

## 3. Claude Code CLI

The pipeline is driven entirely by Claude Code slash commands. Install the CLI
from https://github.com/anthropics/claude-code, then launch it from the repo
root:

```bash
cd "$PROJECT_ROOT"
claude
```

---

## 4. Conda environments — `./setup.sh`

Each model runs in its own conda env, invoked as `conda run -n <env> python ...`.
`./setup.sh` builds all five (reading their names from `DIRECTORYS.yaml`), skips
any that already exist, applies the GroundingDINO `.cu` torch-compat patch +
`groundingdino._C` build check, and prints a per-env summary.

```bash
cd "$PROJECT_ROOT"
./setup.sh                         # all five (skips existing)
./setup.sh --all --force           # delete + rebuild every env
./setup.sh --scenegen --qwen-vl    # only specific env(s)
./setup.sh --help                  # all flags
```

Expect **~30–60 min** the first time (compiling CUDA extensions). Honors:

- `CUDA_HOME` — system CUDA toolkit for the `scenegen` / `grounded-sam` source
  builds (default `/usr/local/cuda`; a CUDA 12.x toolkit matches the pinned
  cu128 torch).
- `SC_ENVS_DIR` — put the big envs (the ~10 GB `qwen-vl`) off `/home`:
  `SC_ENVS_DIR=/big/disk/sc_envs ./setup.sh`.

| Env name | Py | Purpose |
|---|---|---|
| `sceneconductor` | 3.11 | driver + Blender orchestration |
| `scenegen` | 3.10 | GALP layout prediction (torch cu128 + pytorch3d) |
| `grounded-sam` | 3.10 | GroundingDINO + Segment-Anything (CUDA build) |
| `sam3d-objects` | 3.11 | SAM 3D Objects textured GLB extraction |
| `qwen-vl` | 3.11 | Qwen3.5-VL attribute extractor |

> Env names are the single source of truth in `DIRECTORYS.yaml::conda_envs`.
> Mismatched names are the most common Stage 1 failure.

---

## 5. Model checkpoints (~25 GB)

The pipeline expects weights under `$PROJECT_ROOT/checkpoints/`:

```
checkpoints/
├── grounded-sam/        # GroundingDINO + SAM weights   (~3 GB)
├── galp/                # GALP weights + configs        (~6 GB)
├── sam3d/hf/            # SAM 3D Objects weights         (~6 GB)
└── qwen/Qwen3.5-27B/    # local copy of Qwen3.5-VL       (~10 GB)
```

The `hf` CLI ships with the `sceneconductor` env
(`pip install -U "huggingface_hub[cli]"` if you need it standalone).

### 5.1 GroundedSAM (public direct URLs)

```bash
mkdir -p "$PROJECT_ROOT/checkpoints/grounded-sam"
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -O "$PROJECT_ROOT/checkpoints/grounded-sam/groundingdino_swint_ogc.pth"
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -O "$PROJECT_ROOT/checkpoints/grounded-sam/sam_vit_h_4b8939.pth"
```

### 5.2 GALP (Hugging Face, public)

The repo [`WopperSet/SceneConductor`](https://huggingface.co/WopperSet/SceneConductor)
stores the GALP weights directly under `checkpoints/` (no `galp/` subdir), while the
pipeline expects them at `checkpoints/galp/`. Download to a temp dir, then move them
into place:

```bash
hf download WopperSet/SceneConductor --include "checkpoints/*" --local-dir /tmp/galp-dl
mkdir -p "$PROJECT_ROOT/checkpoints/galp"
mv /tmp/galp-dl/checkpoints/* "$PROJECT_ROOT/checkpoints/galp/"
rm -rf /tmp/galp-dl
```

### 5.3 SAM 3D Objects (Hugging Face, **gated**)

```bash
HF_TOKEN=hf_xxx hf auth login --token "$HF_TOKEN"   # if you haven't already
hf download facebook/sam-3d-objects --repo-type model --local-dir "$PROJECT_ROOT/checkpoints/.sam3d-dl"
# Move the hf-format weights (pipeline.yaml + *.ckpt) into place, then wire the symlink
# the Stage 1 wrapper resolves:
mkdir -p "$PROJECT_ROOT/checkpoints/sam3d"
mv "$PROJECT_ROOT/checkpoints/.sam3d-dl/checkpoints" "$PROJECT_ROOT/checkpoints/sam3d/hf"
ln -sfn "$PROJECT_ROOT/checkpoints/sam3d/hf" "$PROJECT_ROOT/submodules/SAM3D/checkpoints/hf"
```

### 5.4 Qwen3.5-VL (Hugging Face, public, ~10 GB)

A local copy, or rely on the HF cache:

```bash
hf download Qwen/Qwen3.5-27B --local-dir "$PROJECT_ROOT/checkpoints/qwen/Qwen3.5-27B"
```

Or leave `qwen_vl_model_id: Qwen/Qwen3.5-27B` in `DIRECTORYS.yaml` and let the
Transformers cache resolve it on first use.

---

Once envs and checkpoints are in place, launch `claude` from the repo root and
run the pipeline — see **[README → Quickstart, Step 2](./README.md#-step-2--run-the-pipeline-claude-code-prompt)**.
