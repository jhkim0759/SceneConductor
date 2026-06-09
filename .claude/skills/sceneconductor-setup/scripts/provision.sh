#!/usr/bin/env bash
# ============================================================================
# sceneconductor-setup :: provision.sh
# ----------------------------------------------------------------------------
# One command that takes a FRESH clone of SceneConductor to a fully runnable
# pipeline. It is the single orchestrator the `sceneconductor-setup` skill runs:
#
#   submodules  ->  Blender  ->  conda envs (./setup.sh)  ->  checkpoints  ->  verify
#
# Every step is idempotent and individually skippable, so re-running after a
# partial failure only does the missing work. Heavy lifting is delegated:
#   * conda envs  -> repo-root ./setup.sh   (also patches GroundingDINO .cu +
#                                             verifies groundingdino._C builds)
#   * checkpoints -> scripts/download_checkpoints.sh
#   * validation  -> scripts/verify_install.sh
#
# Usage:
#   bash provision.sh [options]
#
# Options:
#   --force                Pass --force to ./setup.sh (rebuild every conda env)
#   --gpu N                GPU index used by the optional verify smoke (default 0)
#   --skip-submodules      Don't run `git submodule update --init --recursive`
#   --skip-blender         Don't download/extract Blender
#   --skip-envs            Don't run ./setup.sh
#   --skip-checkpoints     Don't download model checkpoints
#   --skip-qwen            Download all checkpoints EXCEPT the ~10GB Qwen weights
#   --skip-verify          Don't run the final verification
#   -h | --help            Show this message
#
# Environment passthrough:
#   CUDA_HOME    forwarded to ./setup.sh for the CUDA source builds
#   SC_ENVS_DIR  forwarded to ./setup.sh to place big envs off /home
#   HF_TOKEN     forwarded to the checkpoint downloader (SAM3D is gated)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BLENDER_TARBALL="blender-4.2.1-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender4.2/$BLENDER_TARBALL"
BLENDER_DIR="$REPO_ROOT/blender-4.2.1-linux-x64"

FORCE=""
GPU="0"
DO_SUBMODULES=true
DO_BLENDER=true
DO_ENVS=true
DO_CHECKPOINTS=true
DO_VERIFY=true
CKPT_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force) FORCE="--force" ;;
        --gpu) GPU="${2:-0}"; shift ;;
        --skip-submodules) DO_SUBMODULES=false ;;
        --skip-blender) DO_BLENDER=false ;;
        --skip-envs) DO_ENVS=false ;;
        --skip-checkpoints) DO_CHECKPOINTS=false ;;
        --skip-qwen) CKPT_ARGS+=(--skip-qwen) ;;
        --skip-verify) DO_VERIFY=false ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 2 ;;
    esac
    shift
done

cd "$REPO_ROOT"
echo "============================================================"
echo " SceneConductor provisioning"
echo " repo_root: $REPO_ROOT"
echo "============================================================"

step() { echo ""; echo "------------------------------------------------------------"; echo ">>> $*"; echo "------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# 1. Submodules
# ---------------------------------------------------------------------------
if $DO_SUBMODULES; then
    step "[1/5] Submodules"
    if [ -d "$REPO_ROOT/.git" ] || [ -f "$REPO_ROOT/.git" ]; then
        git -C "$REPO_ROOT" submodule update --init --recursive \
            && echo "[OK] submodules populated" \
            || echo "[WARN] submodule update reported an issue — check network/access"
    else
        echo "[WARN] $REPO_ROOT is not a git checkout — skipping submodule init"
    fi
else
    echo "[skip] submodules"
fi

# ---------------------------------------------------------------------------
# 2. Blender 4.2.1 (vendored under repo root)
# ---------------------------------------------------------------------------
if $DO_BLENDER; then
    step "[2/5] Blender 4.2.1"
    if [ -x "$BLENDER_DIR/blender" ]; then
        echo "[skip] Blender already present: $BLENDER_DIR"
    else
        ( cd "$REPO_ROOT" \
          && wget -q --show-progress "$BLENDER_URL" -O "$BLENDER_TARBALL" \
          && tar -xf "$BLENDER_TARBALL" \
          && rm -f "$BLENDER_TARBALL" ) \
            && echo "[OK] Blender extracted to $BLENDER_DIR" \
            || echo "[ERROR] Blender download/extract failed — fetch $BLENDER_URL manually"
    fi
    "$BLENDER_DIR/blender" --version 2>/dev/null | head -1 || true
else
    echo "[skip] Blender"
fi

# ---------------------------------------------------------------------------
# 3. Conda environments (./setup.sh builds all five + patches .cu + verifies _C)
# ---------------------------------------------------------------------------
if $DO_ENVS; then
    step "[3/5] Conda environments (./setup.sh)"
    if [ -x "$REPO_ROOT/setup.sh" ] || [ -f "$REPO_ROOT/setup.sh" ]; then
        bash "$REPO_ROOT/setup.sh" $FORCE \
            || echo "[WARN] setup.sh returned non-zero — inspect its per-env summary above"
    else
        echo "[ERROR] $REPO_ROOT/setup.sh not found"
    fi
else
    echo "[skip] conda envs"
fi

# ---------------------------------------------------------------------------
# 4. Model checkpoints
# ---------------------------------------------------------------------------
if $DO_CHECKPOINTS; then
    step "[4/5] Model checkpoints"
    bash "$SCRIPT_DIR/download_checkpoints.sh" "${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}" \
        || echo "[WARN] some checkpoints did not download — see messages above (SAM3D is HF-gated)"
else
    echo "[skip] checkpoints"
fi

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
if $DO_VERIFY; then
    step "[5/5] Verify installation"
    bash "$SCRIPT_DIR/verify_install.sh" --gpu "$GPU"
    rc=$?
else
    echo "[skip] verify"
    rc=0
fi

echo ""
echo "============================================================"
echo " Provisioning finished. Verify exit code: ${rc:-0}"
echo "   Next: run the pipeline ->  /scene-orchestration scenes/my_room"
echo "============================================================"
exit "${rc:-0}"
