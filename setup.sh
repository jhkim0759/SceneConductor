#!/usr/bin/env bash
# ============================================================================
# SceneConductor — one-shot conda environment provisioner
# ============================================================================
# Creates every conda environment declared in DIRECTORYS.yaml::conda_envs with
# the exact libraries each pipeline stage needs. Running ONLY this script is
# enough to provision the whole pipeline.
#
#   ./setup.sh                 # create ALL five envs (skip ones that exist)
#   ./setup.sh --all --force   # recreate ALL five envs from scratch
#   ./setup.sh --scenegen      # (re)build a single env
#   ./setup.sh --grounded-sam --sam3d
#   ./setup.sh --help
#
# The five environments (names resolved from DIRECTORYS.yaml):
#   sceneconductor  py3.11  driver / Blender orchestration (CPU libs only)
#   scenegen        py3.10  GALP inference  (torch cu128 + pytorch3d + TRELLIS)
#   grounded-sam    py3.10  GroundingDINO + Segment-Anything (CUDA editable build)
#   sam3d-objects   py3.11  SAM 3D Objects  (official default.yml recipe)
#   qwen-vl         py3.11  Qwen3.5-VL attribute extractor (transformers 5.x)
#
# Notes
#   * Versions below reproduce the known-good, currently-working envs on the
#     reference host (CUDA 12.x / H200). They are intentionally newer than the
#     generic numbers in INSTALLATION.md, which were aspirational.
#   * Heavy CUDA builds (pytorch3d, flash-attn, GroundingDINO, kaolin, spconv)
#     require a system CUDA toolkit. Point CUDA_HOME at it (default /usr/local/cuda).
#   * The ~10 GB qwen-vl env can be placed off /home: set SC_ENVS_DIR before
#     running, e.g.  SC_ENVS_DIR=/mnt/disk/sc_envs ./setup.sh --qwen-vl
#   * Model CHECKPOINTS are NOT downloaded here — see INSTALLATION.md §5.
# ============================================================================

set -uo pipefail

# ----------------------------------------------------------------------------
# Locate repo root + DIRECTORYS.yaml
# ----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRS_YAML="$REPO_ROOT/DIRECTORYS.yaml"
cd "$REPO_ROOT"

# CUDA toolkit used by source builds (pytorch3d / flash-attn / GroundingDINO).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ----------------------------------------------------------------------------
# Resolve a dotted key out of DIRECTORYS.yaml (mirrors the project's _dirs_get).
# Falls back to the supplied default if yaml/key is unavailable.
# ----------------------------------------------------------------------------
_dirs_get() {
    local key="$1" default="${2:-}"
    local val
    val="$(python3 - "$DIRS_YAML" "$key" <<'PY' 2>/dev/null
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):
    d = d[k]
print(d)
PY
)"
    if [ -n "$val" ] && [ "$val" != "None" ]; then echo "$val"; else echo "$default"; fi
}

# Env names — DIRECTORYS.yaml is the single source of truth (with fallbacks).
ENV_SCENECONDUCTOR="$(_dirs_get conda_envs.sceneconductor sceneconductor)"
ENV_SCENEGEN="$(_dirs_get conda_envs.galp scenegen)"
ENV_GROUNDED_SAM="$(_dirs_get conda_envs.grounded-sam grounded-sam)"
ENV_SAM3D="$(_dirs_get conda_envs.sam3d-objects sam3d-objects)"
ENV_QWEN="$(_dirs_get conda_envs.qwen-vl qwen-vl)"

SAM3D_REPO="$REPO_ROOT/$(_dirs_get sam3d_repo ./submodules/SAM3D | sed 's#^\./##')"
GSAM_REPO="$REPO_ROOT/submodules/Grounded-SAM"
GALP_BUNDLE="$REPO_ROOT/.claude/skills/stage1-initialize-scene/src/galp_runtime/bundle.sh"
GSAM_SYMLINK="$REPO_ROOT/.claude/skills/stage1-initialize-scene/src/grounded-sam/Grounded-Segment-Anything"

# PyTorch CUDA wheel channel for the cu128 envs (scenegen / grounded-sam / qwen-vl).
TORCH_CU128="https://download.pytorch.org/whl/cu128"

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
DO_SCENECONDUCTOR=false
DO_SCENEGEN=false
DO_GROUNDED_SAM=false
DO_SAM3D=false
DO_QWEN=false
FORCE=false
HELP=false
ANY_ENV=false

print_help() {
    cat <<EOF
SceneConductor environment provisioner

Usage: ./setup.sh [OPTIONS]

Env selectors (choose one or more; default is --all):
  --all                 Create all five conda environments
  --sceneconductor      Driver / Blender orchestration env (py3.11)
  --scenegen, --galp    GALP inference env (py3.10, torch cu128 + pytorch3d)
  --grounded-sam        GroundingDINO + Segment-Anything env (py3.10)
  --sam3d, --sam3d-objects
                        SAM 3D Objects env (py3.11, official recipe)
  --qwen-vl             Qwen3.5-VL extractor env (py3.11)

Modifiers:
  --force               Remove and recreate the selected env(s) from scratch
  -h, --help            Show this message

Environment variables:
  CUDA_HOME             System CUDA toolkit for source builds (default: /usr/local/cuda)
  SC_ENVS_DIR           If set, appended to conda envs_dirs (place big envs off /home)

Examples:
  ./setup.sh                       # all envs, skip ones that already exist
  ./setup.sh --all --force         # rebuild everything
  ./setup.sh --scenegen --qwen-vl  # two specific envs
  SC_ENVS_DIR=/mnt/disk/sc_envs ./setup.sh --qwen-vl
EOF
}

if [ "$#" -eq 0 ]; then
    DO_SCENECONDUCTOR=true; DO_SCENEGEN=true; DO_GROUNDED_SAM=true
    DO_SAM3D=true; DO_QWEN=true; ANY_ENV=true
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --all) DO_SCENECONDUCTOR=true; DO_SCENEGEN=true; DO_GROUNDED_SAM=true
               DO_SAM3D=true; DO_QWEN=true; ANY_ENV=true ;;
        --sceneconductor) DO_SCENECONDUCTOR=true; ANY_ENV=true ;;
        --scenegen|--galp) DO_SCENEGEN=true; ANY_ENV=true ;;
        --grounded-sam) DO_GROUNDED_SAM=true; ANY_ENV=true ;;
        --sam3d|--sam3d-objects) DO_SAM3D=true; ANY_ENV=true ;;
        --qwen-vl) DO_QWEN=true; ANY_ENV=true ;;
        --force) FORCE=true ;;
        -h|--help) HELP=true ;;
        *) echo "[ERROR] Unknown argument: $1"; HELP=true; break ;;
    esac
    shift
done

if [ "$HELP" = true ]; then print_help; exit 0; fi
if [ "$ANY_ENV" = false ]; then
    DO_SCENECONDUCTOR=true; DO_SCENEGEN=true; DO_GROUNDED_SAM=true
    DO_SAM3D=true; DO_QWEN=true
fi

# ----------------------------------------------------------------------------
# Bootstrap conda so `conda activate` works inside this non-interactive shell.
# ----------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[FATAL] conda not found on PATH. Install miniconda/mamba first." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ -n "${SC_ENVS_DIR:-}" ]; then
    mkdir -p "$SC_ENVS_DIR"
    conda config --append envs_dirs "$SC_ENVS_DIR" 2>/dev/null || true
    echo "[INFO] Added envs_dir: $SC_ENVS_DIR"
fi

# Track outcomes for the final summary.
declare -A RESULTS

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# Create (or, with --force, recreate) a named env at a python version.
# Returns 0 if the env is ready to install into, 1 if it should be skipped.
ensure_env() {
    local name="$1" pyver="$2"
    if env_exists "$name"; then
        if [ "$FORCE" = true ]; then
            echo "[INFO] --force: removing existing env '$name'"
            conda env remove -n "$name" -y >/dev/null 2>&1 || true
        else
            echo "[SKIP] env '$name' already exists (use --force to rebuild)"
            return 1
        fi
    fi
    echo "[INFO] creating env '$name' (python=$pyver)"
    conda create -n "$name" "python=$pyver" -y
}

# ============================================================================
# 1. sceneconductor — driver / Blender orchestration (CPU-only libs, py3.11)
# ============================================================================
setup_sceneconductor() {
    local env="$ENV_SCENECONDUCTOR"
    echo ""; echo "========== [1/5] $env =========="
    if ensure_env "$env" 3.11; then
        conda activate "$env"
        python -m pip install --upgrade pip
        # Pure-python libs imported by the Stage 2/3 orchestration + Blender drivers.
        pip install \
            "pyyaml" "numpy" "pillow" "trimesh" "scipy" \
            "opencv-python" "shapely" "matplotlib" "scikit-image" \
            "loguru" "hydra-core" "omegaconf" "tqdm" "imageio" "imageio-ffmpeg"
        conda deactivate
        RESULTS["$env"]="OK"
    else
        RESULTS["$env"]="skipped"
    fi
}

# ============================================================================
# 2. scenegen — GALP inference (py3.10, torch cu128 + pytorch3d + TRELLIS deps)
# ============================================================================
setup_scenegen() {
    local env="$ENV_SCENEGEN"
    echo ""; echo "========== [2/5] $env (GALP) =========="
    if ensure_env "$env" 3.10; then
        conda activate "$env"
        python -m pip install --upgrade pip

        # --- PyTorch (CUDA 12.8 wheels) ---
        pip install torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_CU128"

        # --- Core scientific / IO stack ---
        pip install \
            "numpy" "pillow" "trimesh" "open3d" "scipy" "scikit-image" \
            "opencv-python-headless" "einops" "tqdm" "easydict" "loguru" \
            "pyyaml" "safetensors" "omegaconf" "hydra-core" \
            "transformers" "accelerate" "diffusers" "timm" "qwen-vl-utils"

        # --- 3D utils (pinned git rev used by the GALP/TRELLIS code path) ---
        pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

        # --- MoGe depth model (used by GALP inference_utils) ---
        pip install moge

        # --- Sparse conv (TRELLIS sparse-structure encoder) ---
        pip install spconv-cu120

        # --- Efficient attention (CUDA 12.8 wheels) ---
        pip install xformers==0.0.35 --index-url "$TORCH_CU128" \
            || echo "[WARN][$env] xformers install failed (GALP falls back to SDPA)"
        FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE \
            pip install flash-attn==2.8.3 --no-build-isolation \
            || echo "[WARN][$env] flash-attn build failed (optional; SDPA fallback)"

        # --- PyTorch3D (built from source against the installed torch) ---
        FORCE_CUDA=1 pip install --no-build-isolation \
            "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
            || echo "[ERROR][$env] pytorch3d build failed — GALP needs this; see CUDA_HOME"

        # --- Kaolin (optional for GALP; non-fatal) ---
        pip install kaolin \
            -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.10.0_cu128.html \
            || pip install kaolin \
            || echo "[WARN][$env] kaolin install failed (optional for GALP inference)"

        # NumPy can get bumped by transitive deps — keep a sane modern pin.
        pip install "numpy<3" --upgrade

        conda deactivate

        # --- One-time GALP runtime bundle (symlinks weights + copies configs) ---
        if [ -f "$GALP_BUNDLE" ]; then
            echo "[INFO][$env] running GALP runtime bundle.sh"
            bash "$GALP_BUNDLE" || echo "[WARN][$env] bundle.sh reported missing checkpoints (download per INSTALLATION.md §5.2)"
        else
            echo "[WARN][$env] bundle.sh not found at $GALP_BUNDLE"
        fi
        RESULTS["$env"]="OK"
    else
        RESULTS["$env"]="skipped"
    fi
}

# ============================================================================
# 3. grounded-sam — GroundingDINO + Segment-Anything (py3.10, CUDA editable build)
# ============================================================================
setup_grounded_sam() {
    local env="$ENV_GROUNDED_SAM"
    echo ""; echo "========== [3/5] $env =========="
    if [ ! -d "$GSAM_REPO/GroundingDINO" ]; then
        echo "[ERROR][$env] $GSAM_REPO not populated. Run: git submodule update --init --recursive"
        RESULTS["$env"]="FAILED (submodule missing)"; return
    fi
    if ensure_env "$env" 3.10; then
        conda activate "$env"
        python -m pip install --upgrade pip

        # PyTorch (CUDA 12.8 wheels) — pin before the requirements pull a CPU build.
        pip install torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_CU128"

        # Upstream Python deps (includes yapf, timm, supervision, transformers, …).
        pip install -r "$GSAM_REPO/requirements.txt"
        pip install yapf   # explicit: required by the Stage 1 GroundingDINO import path

        # CUDA extension builds need these flags + a real nvcc on CUDA_HOME.
        export AM_I_DOCKER=False
        export BUILD_WITH_CUDA=True
        echo "[INFO][$env] building CUDA extensions with CUDA_HOME=$CUDA_HOME"
        python -m pip install -e "$GSAM_REPO/segment_anything"
        pip install --no-build-isolation -e "$GSAM_REPO/GroundingDINO" \
            || echo "[ERROR][$env] GroundingDINO CUDA build failed — check CUDA_HOME/nvcc"

        conda deactivate
        RESULTS["$env"]="OK"
    else
        RESULTS["$env"]="skipped"
    fi

    # Ensure the Grounded-Segment-Anything symlink the Stage 1 wrapper expects.
    if [ ! -e "$GSAM_SYMLINK" ]; then
        ln -sfn "$GSAM_REPO" "$GSAM_SYMLINK"
        echo "[INFO] linked $GSAM_SYMLINK -> $GSAM_REPO"
    fi
}

# ============================================================================
# 4. sam3d-objects — SAM 3D Objects (py3.11, official environments/default.yml)
# ============================================================================
setup_sam3d() {
    local env="$ENV_SAM3D"
    echo ""; echo "========== [4/5] $env =========="
    local def_yml="$SAM3D_REPO/environments/default.yml"
    if [ ! -f "$def_yml" ]; then
        echo "[ERROR][$env] $def_yml missing. Run: git submodule update --init --recursive"
        RESULTS["$env"]="FAILED (submodule missing)"; return
    fi

    if env_exists "$env" && [ "$FORCE" = false ]; then
        echo "[SKIP] env '$env' already exists (use --force to rebuild)"
        RESULTS["$env"]="skipped"; return
    fi
    [ "$FORCE" = true ] && conda env remove -n "$env" -y >/dev/null 2>&1 || true

    # The env name is baked into default.yml (name: sam3d-objects). Create from it.
    echo "[INFO][$env] creating env from $def_yml (brings cuda-toolkit 12.1 + gcc-12)"
    conda env create -f "$def_yml"
    conda activate "$env"

    # Official install recipe (submodules/SAM3D/doc/setup.md).
    export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
    pip install -e "$SAM3D_REPO[dev]"
    pip install -e "$SAM3D_REPO[p3d]"        # 2-step: resolves pytorch3d/torch conflict
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e "$SAM3D_REPO[inference]"

    # Patch hydra 1.3.2 (upstream PR not yet released).
    if [ -x "$SAM3D_REPO/patching/hydra" ]; then
        ( cd "$SAM3D_REPO" && ./patching/hydra ) || echo "[WARN][$env] hydra patch failed"
    fi

    conda deactivate
    RESULTS["$env"]="OK"
}

# ============================================================================
# 5. qwen-vl — Qwen3.5-VL attribute extractor (py3.11, transformers 5.x)
# ============================================================================
setup_qwen() {
    local env="$ENV_QWEN"
    echo ""; echo "========== [5/5] $env =========="
    if ensure_env "$env" 3.11; then
        conda activate "$env"
        python -m pip install --upgrade pip

        # PyTorch (CUDA 12.8 wheels).
        pip install torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_CU128"

        # transformers must be new enough for Qwen3.5-VL (AutoModelForImageTextToText).
        pip install \
            "transformers>=5.5" "accelerate" "qwen-vl-utils" \
            "pillow" "numpy" "einops" "safetensors" "scipy" "pyyaml"

        conda deactivate
        RESULTS["$env"]="OK"
    else
        RESULTS["$env"]="skipped"
    fi
}

# ----------------------------------------------------------------------------
# Run selected envs
# ----------------------------------------------------------------------------
[ "$DO_SCENECONDUCTOR" = true ] && setup_sceneconductor
[ "$DO_SCENEGEN" = true ]       && setup_scenegen
[ "$DO_GROUNDED_SAM" = true ]   && setup_grounded_sam
[ "$DO_SAM3D" = true ]          && setup_sam3d
[ "$DO_QWEN" = true ]           && setup_qwen

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " SceneConductor setup summary"
echo "============================================================"
for env in "$ENV_SCENECONDUCTOR" "$ENV_SCENEGEN" "$ENV_GROUNDED_SAM" "$ENV_SAM3D" "$ENV_QWEN"; do
    printf "  %-18s %s\n" "$env" "${RESULTS[$env]:-not selected}"
done
echo ""
echo "Next steps:"
echo "  * Download model checkpoints   -> INSTALLATION.md §5"
echo "  * Verify envs                  -> conda env list"
echo "  * Run the pipeline (Claude Code) -> /scene-orchestration scenes/my_room"
echo "============================================================"
