#!/usr/bin/env bash
# ============================================================================
# sceneconductor-setup :: verify_install.sh
# ----------------------------------------------------------------------------
# Non-destructive post-install audit. Confirms every prerequisite the pipeline
# assumes is actually in place, so failures surface HERE (in seconds) instead
# of mid-Stage-1 (after minutes of GPU work). Prints a PASS/FAIL table and
# exits non-zero if any REQUIRED check fails.
#
# Usage: bash verify_install.sh [--gpu N]
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DIRS_YAML="$REPO_ROOT/DIRECTORYS.yaml"
GPU="0"
[ "${1:-}" = "--gpu" ] && GPU="${2:-0}"

_dirs_get() { python3 -c "import yaml,sys;d=yaml.safe_load(open(sys.argv[1]))
for k in sys.argv[2].split('.'):d=d[k]
print(d)" "$DIRS_YAML" "$1" 2>/dev/null; }

FAIL=0; WARN=0
ok()   { printf "  [ OK ] %s\n" "$1"; }
bad()  { printf "  [FAIL] %s\n" "$1"; FAIL=$((FAIL+1)); }
warn() { printf "  [WARN] %s\n" "$1"; WARN=$((WARN+1)); }

echo "============================================================"
echo " SceneConductor install verification  (repo: $REPO_ROOT)"
echo "============================================================"

# --- Blender ---------------------------------------------------------------
echo "[1] Blender"
BLENDER="${BLENDER:-$REPO_ROOT/$(_dirs_get blender_bin | sed 's#^\./##')}"
if [ -x "$BLENDER" ] && "$BLENDER" --version >/dev/null 2>&1; then
    ok "blender: $("$BLENDER" --version 2>/dev/null | head -1)"
else
    bad "blender not runnable at $BLENDER (set \$BLENDER or re-run provision.sh)"
fi

# --- Conda envs ------------------------------------------------------------
echo "[2] Conda environments"
if command -v conda >/dev/null 2>&1; then
    ENV_LIST="$(conda env list | awk '{print $1}')"
    for key in sceneconductor galp grounded-sam sam3d-objects qwen-vl; do
        name="$(_dirs_get "conda_envs.$key")"; name="${name:-$key}"
        echo "$ENV_LIST" | grep -qx "$name" && ok "env '$name'" || bad "env '$name' MISSING (./setup.sh --$key)"
    done
else
    bad "conda not on PATH"
fi

# --- GroundingDINO _C (the #1 Stage 1 failure mode) ------------------------
echo "[3] GroundedSAM CUDA extension"
GS_ENV="$(_dirs_get conda_envs.grounded-sam)"; GS_ENV="${GS_ENV:-grounded-sam}"
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$GS_ENV"; then
    if conda run -n "$GS_ENV" python -c "from groundingdino import _C; import segment_anything" >/dev/null 2>&1; then
        ok "groundingdino._C + segment_anything import"
    else
        bad "groundingdino._C / segment_anything import FAILED (./setup.sh --grounded-sam --force)"
    fi
else
    warn "grounded-sam env absent — skipped _C check"
fi

# --- Checkpoints -----------------------------------------------------------
echo "[4] Checkpoints"
ckpt() { [ -s "$REPO_ROOT/$1" ] && ok "$1" || bad "$1 MISSING"; }
ckpt "checkpoints/grounded-sam/groundingdino_swint_ogc.pth"
ckpt "checkpoints/grounded-sam/sam_vit_h_4b8939.pth"
ckpt "checkpoints/galp/checkpoint.pt"
ckpt "checkpoints/galp/condition_embedder.ckpt"
ckpt "checkpoints/galp/pipeline.yaml"
# SAM3D resolves through the wrapper path (symlink) — gated, so a miss is a WARN.
SAM3D_REPO="$REPO_ROOT/$(_dirs_get sam3d_repo | sed 's#^\./##')"; SAM3D_REPO="${SAM3D_REPO:-$REPO_ROOT/submodules/SAM3D}"
if [ -s "$SAM3D_REPO/checkpoints/hf/pipeline.yaml" ]; then
    ok "submodules/SAM3D/checkpoints/hf/pipeline.yaml (SAM3D weights wired)"
else
    warn "SAM3D weights not found (gated repo: request access at huggingface.co/facebook/sam-3d-objects)"
fi
# Qwen: local copy OR HF cache both acceptable.
QWEN_DIR="$REPO_ROOT/$(_dirs_get checkpoints_qwen_vl | sed 's#^\./##')"
[ -s "$QWEN_DIR/config.json" ] && ok "Qwen local weights" || warn "Qwen local weights absent (will resolve from HF cache: Qwen/Qwen3.5-27B)"

# --- Submodules populated --------------------------------------------------
echo "[5] Submodules populated"
for d in submodules/GALP/src submodules/Grounded-SAM/GroundingDINO submodules/SAM3D/environments; do
    [ -d "$REPO_ROOT/$d" ] && ok "$d" || bad "$d MISSING (git submodule update --init --recursive)"
done

echo "============================================================"
if [ "$FAIL" -eq 0 ]; then
    echo " RESULT: PASS  ($WARN warning(s))"
    echo " Ready:  /scene-orchestration scenes/my_room"
else
    echo " RESULT: $FAIL required check(s) FAILED, $WARN warning(s)"
    echo " Fix the [FAIL] lines above, then re-run this script."
fi
echo "============================================================"
[ "$FAIL" -eq 0 ]
