#!/usr/bin/env bash
# bundle.sh — Set up the galp_runtime directory from the GALP repo.
#
# Run this once from the galp_runtime/ directory, or from anywhere by
# passing the correct RUNTIME_DIR.
#
# Large files (safetensors, .ckpt, .pt) are symlinked; small configs are copied.
# Re-running is safe: existing symlinks/files are not overwritten unless -f is passed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}"
# Checkpoints now live outside .claude/ — at <repo>/checkpoints/galp/
PROJECT_ROOT="$(cd "$RUNTIME_DIR/../../../../.." && pwd)"
DIRS_YAML="$PROJECT_ROOT/DIRECTORYS.yaml"

# Helper: extract a single value from DIRECTORYS.yaml by dotted key path.
_dirs_get() {
    python3 -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); v=d
for k in sys.argv[2].split('.'):
    v=v[k]
print(v)" "$DIRS_YAML" "$1"
}

# Helper: resolve a (possibly relative ./...) registry value against PROJECT_ROOT.
_resolve() { case "$1" in /*) echo "$1";; ./*) echo "$PROJECT_ROOT/${1#./}";; *) echo "$PROJECT_ROOT/$1";; esac; }

CKPT_DIR="$(_resolve "$(_dirs_get checkpoints_galp)")"
REPO_DIR="$(_resolve "$(_dirs_get galp_repo)")"
FORCE="${1:-}"   # pass -f to overwrite existing symlinks

# ---------------------------------------------------------------------------
# Helper: create symlink, optionally forcing
# ---------------------------------------------------------------------------
symlink() {
    local target="$1"
    local link="$2"
    if [ ! -e "$target" ] && [ ! -L "$target" ]; then
        echo "  [WARN] target does not exist: $target"
        return
    fi
    if [ -L "$link" ] && [ "$FORCE" != "-f" ]; then
        echo "  [skip] symlink already exists: $link"
    else
        ln -sf "$target" "$link"
        echo "  [link] $link -> $target"
    fi
}

# Helper: copy file if not already present
copyfile() {
    local src="$1"
    local dst="$2"
    if [ ! -e "$src" ]; then
        echo "  [WARN] source does not exist: $src"
        return
    fi
    if [ -e "$dst" ] && [ "$FORCE" != "-f" ]; then
        echo "  [skip] file already exists: $dst"
    else
        cp -f "$src" "$dst"
        echo "  [copy] $dst"
    fi
}

echo "=== GALP runtime bundle setup ==="
echo "  REPO_DIR:    $REPO_DIR"
echo "  RUNTIME_DIR: $RUNTIME_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. src/ — symlink the entire tree (read-only reference, avoids duplication)
# ---------------------------------------------------------------------------
echo "[1/5] src/ tree"
mkdir -p "$RUNTIME_DIR/src"
symlink "$REPO_DIR/src" "$RUNTIME_DIR/src_link"
# Prefer full tree symlink under src/ if src/ is empty
if [ -d "$REPO_DIR/src" ]; then
    # Replace src/ with a direct symlink if it's currently an empty dir
    if [ -z "$(ls -A "$RUNTIME_DIR/src" 2>/dev/null)" ]; then
        rmdir "$RUNTIME_DIR/src"
        symlink "$REPO_DIR/src" "$RUNTIME_DIR/src"
    else
        echo "  [info] $RUNTIME_DIR/src already populated — keeping as-is"
    fi
fi

# ---------------------------------------------------------------------------
# 2. inference_utils.py — symlink
# ---------------------------------------------------------------------------
echo "[2/5] inference_utils.py"
symlink "$REPO_DIR/inference_utils.py" "$RUNTIME_DIR/inference_utils.py"

# ---------------------------------------------------------------------------
# 3. configs/mp8_nt512.yaml — copy (small file)
# ---------------------------------------------------------------------------
echo "[3/5] configs/"
mkdir -p "$RUNTIME_DIR/configs"
copyfile "$REPO_DIR/configs/mp8_nt512.yaml" "$RUNTIME_DIR/configs/mp8_nt512.yaml"

# ---------------------------------------------------------------------------
# 4. checkpoints/hf/ — copy YAMLs, symlink large .ckpt
# ---------------------------------------------------------------------------
echo "[4/5] checkpoints/hf/"
mkdir -p "$CKPT_DIR/hf"
HF_SRC="$REPO_DIR/checkpoints/hf"
HF_DST="$CKPT_DIR/hf"

for yaml_name in pipeline.yaml ss_generator.yaml ss_generator_v1_4.yaml; do
    src_yaml="$HF_SRC/$yaml_name"
    dst_yaml="$HF_DST/$yaml_name"
    if [ -f "$src_yaml" ]; then
        copyfile "$src_yaml" "$dst_yaml"
    else
        echo "  [WARN] $yaml_name not found at $src_yaml — skipping"
    fi
done

# Large ckpt — symlink
symlink "$HF_SRC/ss_generator.ckpt" "$HF_DST/ss_generator.ckpt"

# ---------------------------------------------------------------------------
# 5. checkpoints/ckpt/ckpts/ — copy JSON, symlink safetensors
# ---------------------------------------------------------------------------
echo "[5/5] checkpoints/ckpt/ckpts/"
mkdir -p "$CKPT_DIR/ckpt/ckpts"
CKPT_SRC="$REPO_DIR/checkpoints/ckpt/ckpts"
CKPT_DST="$CKPT_DIR/ckpt/ckpts"

for base in ss_enc_conv3d_16l8_fp16 ss_dec_conv3d_16l8_fp16; do
    copyfile "$CKPT_SRC/${base}.json"         "$CKPT_DST/${base}.json"
    symlink  "$CKPT_SRC/${base}.safetensors"  "$CKPT_DST/${base}.safetensors"
done

# ---------------------------------------------------------------------------
# 6. Trained checkpoint — symlink into runtime checkpoints/trained/
# ---------------------------------------------------------------------------
echo "[+] checkpoints/trained/"
mkdir -p "$CKPT_DIR/trained"
symlink \
    "$REPO_DIR/checkpoints/important_ckpt/v1_4_coco.pt" \
    "$CKPT_DIR/trained/v1_4_coco.pt"

# ---------------------------------------------------------------------------
# Summary / validation
# ---------------------------------------------------------------------------
echo ""
echo "=== Verification ==="
MISSING=0
check_path() {
    local p="$1"
    if [ -e "$p" ]; then
        echo "  [OK]   $p"
    else
        echo "  [FAIL] $p  <-- MISSING"
        MISSING=$((MISSING + 1))
    fi
}

check_path "$RUNTIME_DIR/src"
check_path "$RUNTIME_DIR/inference_utils.py"
check_path "$RUNTIME_DIR/configs/mp8_nt512.yaml"
check_path "$CKPT_DIR/hf/pipeline.yaml"
check_path "$CKPT_DIR/hf/ss_generator.yaml"
check_path "$CKPT_DIR/hf/ss_generator_v1_4.yaml"
check_path "$CKPT_DIR/hf/ss_generator.ckpt"
check_path "$CKPT_DIR/ckpt/ckpts/ss_enc_conv3d_16l8_fp16.json"
check_path "$CKPT_DIR/ckpt/ckpts/ss_enc_conv3d_16l8_fp16.safetensors"
check_path "$CKPT_DIR/ckpt/ckpts/ss_dec_conv3d_16l8_fp16.json"
check_path "$CKPT_DIR/ckpt/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
check_path "$CKPT_DIR/trained/v1_4_coco.pt"

echo ""
if [ "$MISSING" -eq 0 ]; then
    echo "All paths verified. Runtime bundle is ready."
else
    echo "$MISSING path(s) MISSING. See warnings above."
    exit 1
fi
