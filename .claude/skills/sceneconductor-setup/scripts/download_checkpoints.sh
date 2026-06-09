#!/usr/bin/env bash
# ============================================================================
# sceneconductor-setup :: download_checkpoints.sh
# ----------------------------------------------------------------------------
# Fetch the ~25 GB of model weights into <repo>/checkpoints/ in the exact
# layout the Stage 1 wrappers expect, then wire the SAM3D symlink. Idempotent:
# each set is skipped if its sentinel file already exists.
#
#   grounded-sam : groundingdino_swint_ogc.pth + sam_vit_h_4b8939.pth  (public)
#   galp         : HF  WopperSet/SceneConductor  -> checkpoints/galp/   (public)
#   sam3d        : HF  facebook/sam-3d-objects   -> checkpoints/sam3d/hf (GATED)
#   qwen         : HF  Qwen/Qwen3.5-27B          -> checkpoints/qwen/... (public, ~10GB)
#
# Usage: bash download_checkpoints.sh [--skip-qwen] [--force]
#   --skip-qwen   skip the large Qwen weights (rely on the HF transformers cache)
#   --force       re-download even if sentinel files already exist
#
# Env: HF_TOKEN   if set, used for `hf auth` (needed for the gated SAM3D repo)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DIRS_YAML="$REPO_ROOT/DIRECTORYS.yaml"
CKPT="$REPO_ROOT/checkpoints"

SKIP_QWEN=false
FORCE=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-qwen) SKIP_QWEN=true ;;
        --force) FORCE=true ;;
        *) echo "[ERROR] unknown arg: $1"; exit 2 ;;
    esac
    shift
done

# Resolve the driver env name (for a pip/hf that is guaranteed to exist).
_dirs_get() { python3 -c "import yaml,sys;d=yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):d=d[k]
print(d)" "$DIRS_YAML" "$1" 2>/dev/null; }
ENV_DRIVER="$(_dirs_get conda_envs.sceneconductor)"; ENV_DRIVER="${ENV_DRIVER:-sceneconductor}"

# `hf` runner — prefer the new `hf` CLI, fall back to `huggingface-cli`, run it
# inside the driver conda env so it is always available.
have_conda() { command -v conda >/dev/null 2>&1; }
HF() {
    if have_conda && conda env list | awk '{print $1}' | grep -qx "$ENV_DRIVER"; then
        conda run -n "$ENV_DRIVER" bash -lc 'command -v hf >/dev/null 2>&1 && hf "$@" || huggingface-cli "$@"' _ "$@"
    else
        ( command -v hf >/dev/null 2>&1 && hf "$@" ) || huggingface-cli "$@"
    fi
}
ensure_hf() {
    if have_conda && conda env list | awk '{print $1}' | grep -qx "$ENV_DRIVER"; then
        conda run -n "$ENV_DRIVER" python -m pip install -q -U "huggingface_hub[cli]" 2>/dev/null || true
    else
        python3 -m pip install -q -U "huggingface_hub[cli]" 2>/dev/null || true
    fi
}

step() { echo ""; echo ">>> $*"; }

# ---------------------------------------------------------------------------
# 1. GroundedSAM (public direct URLs)
# ---------------------------------------------------------------------------
step "GroundedSAM checkpoints -> checkpoints/grounded-sam/"
mkdir -p "$CKPT/grounded-sam"
get() { # url dest
    local url="$1" dest="$2"
    if [ -s "$dest" ] && [ "$FORCE" = false ]; then echo "[skip] $(basename "$dest") exists"; return; fi
    wget -q --show-progress "$url" -O "$dest" && echo "[OK] $(basename "$dest")" \
        || echo "[ERROR] failed: $url"
}
get "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
    "$CKPT/grounded-sam/groundingdino_swint_ogc.pth"
get "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
    "$CKPT/grounded-sam/sam_vit_h_4b8939.pth"

# ---------------------------------------------------------------------------
# 2. GALP (public HF repo). NOTE: the repo stores the weights directly under
#    `checkpoints/` (no `galp/` subdir), but the pipeline expects them at
#    checkpoints/galp/. Download to a temp dir, then move them into place.
# ---------------------------------------------------------------------------
step "GALP checkpoints -> checkpoints/galp/  (HF: WopperSet/SceneConductor)"
if [ -s "$CKPT/galp/checkpoint.pt" ] && [ "$FORCE" = false ]; then
    echo "[skip] checkpoints/galp/checkpoint.pt exists"
else
    ensure_hf
    GALP_TMP="$CKPT/.galp-download"
    rm -rf "$GALP_TMP"
    if HF download WopperSet/SceneConductor --include "checkpoints/*" --local-dir "$GALP_TMP"; then
        mkdir -p "$CKPT/galp"
        mv "$GALP_TMP/checkpoints/"* "$CKPT/galp/" 2>/dev/null
        rm -rf "$GALP_TMP"
        [ -s "$CKPT/galp/checkpoint.pt" ] \
            && echo "[OK] GALP weights -> checkpoints/galp/" \
            || echo "[ERROR] GALP download landed but checkpoints/galp/checkpoint.pt is missing"
    else
        echo "[ERROR] GALP download failed"
    fi
fi

# ---------------------------------------------------------------------------
# 3. SAM 3D Objects (GATED HF repo) -> checkpoints/sam3d/hf/
# ---------------------------------------------------------------------------
step "SAM 3D Objects checkpoints -> checkpoints/sam3d/hf/  (HF: facebook/sam-3d-objects, GATED)"
if [ -s "$CKPT/sam3d/hf/pipeline.yaml" ] && [ "$FORCE" = false ]; then
    echo "[skip] checkpoints/sam3d/hf/pipeline.yaml exists"
else
    ensure_hf
    [ -n "${HF_TOKEN:-}" ] && HF auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
    TMP="$CKPT/.sam3d-download"
    rm -rf "$TMP"
    if HF download --repo-type model --local-dir "$TMP" --max-workers 1 facebook/sam-3d-objects; then
        mkdir -p "$CKPT/sam3d"
        # Per submodules/SAM3D/doc/setup.md the repo ships a top-level checkpoints/
        # dir holding the hf-format weights (pipeline.yaml + *.ckpt/*.safetensors)
        # directly; their recipe does `mv <dl>/checkpoints checkpoints/hf`. We mirror
        # that into checkpoints/sam3d/hf/. Probe a few layouts to stay robust.
        SRC=""
        if   [ -f "$TMP/checkpoints/pipeline.yaml" ]; then SRC="$TMP/checkpoints"
        elif [ -d "$TMP/checkpoints/hf" ];            then SRC="$TMP/checkpoints/hf"
        elif [ -f "$TMP/pipeline.yaml" ];             then SRC="$TMP"
        elif [ -d "$TMP/hf" ];                        then SRC="$TMP/hf"
        fi
        if [ -n "$SRC" ]; then
            rm -rf "$CKPT/sam3d/hf"; mv "$SRC" "$CKPT/sam3d/hf"
            rm -rf "$TMP"
            echo "[OK] SAM3D weights -> checkpoints/sam3d/hf/"
        else
            echo "[WARN] unexpected SAM3D repo layout under $TMP — inspect and move the"
            echo "       weights (pipeline.yaml + *.ckpt) to checkpoints/sam3d/hf/ manually"
        fi
    else
        echo "[ERROR] SAM3D download failed. This repo is GATED:"
        echo "   1) Request access:  https://huggingface.co/facebook/sam-3d-objects"
        echo "   2) Create a token:  https://huggingface.co/settings/tokens"
        echo "   3) Re-run with:     HF_TOKEN=hf_xxx bash $(basename "${BASH_SOURCE[0]}")"
    fi
fi

# SAM3D symlink the Stage 1 wrapper resolves: submodules/SAM3D/checkpoints/hf
SAM3D_REPO="$REPO_ROOT/$(_dirs_get sam3d_repo | sed 's#^\./##')"
[ -z "$SAM3D_REPO" ] && SAM3D_REPO="$REPO_ROOT/submodules/SAM3D"
if [ -d "$CKPT/sam3d/hf" ]; then
    mkdir -p "$SAM3D_REPO/checkpoints"
    ln -sfn "$CKPT/sam3d/hf" "$SAM3D_REPO/checkpoints/hf"
    echo "[OK] linked $SAM3D_REPO/checkpoints/hf -> checkpoints/sam3d/hf"
fi

# ---------------------------------------------------------------------------
# 4. Qwen3.5-VL (public HF repo, large)
# ---------------------------------------------------------------------------
QWEN_DIR="$REPO_ROOT/$(_dirs_get checkpoints_qwen_vl | sed 's#^\./##')"
[ -z "$QWEN_DIR" ] && QWEN_DIR="$CKPT/qwen/Qwen3.5-27B"
if $SKIP_QWEN; then
    step "Qwen3.5-VL  -> SKIPPED (--skip-qwen; will resolve from the HF cache at first use)"
elif [ -s "$QWEN_DIR/config.json" ] && [ "$FORCE" = false ]; then
    step "Qwen3.5-VL  -> [skip] $QWEN_DIR/config.json exists"
else
    step "Qwen3.5-VL -> $QWEN_DIR  (HF: Qwen/Qwen3.5-27B, ~10GB)"
    ensure_hf
    mkdir -p "$QWEN_DIR"
    HF download Qwen/Qwen3.5-27B --local-dir "$QWEN_DIR" \
        && echo "[OK] Qwen weights" \
        || echo "[WARN] Qwen download failed — the pipeline can also resolve Qwen/Qwen3.5-27B from the HF cache"
fi

echo ""
echo ">>> Checkpoint download pass complete. Run verify_install.sh to confirm."
