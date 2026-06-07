#!/usr/bin/env bash
#
# run_stage1.sh — deterministic driver for the stage1-initialize-scene pipeline.
#
# The Mask-Evaluator vision call is embedded in --phase eval via run_mask_evaluator.py.
# The orchestrator must call the three phases in order:
#
#     run_stage1.sh --phase pre   → GroundedSAM outputs
#     run_stage1.sh --phase eval  → Opus vision API call (run_mask_evaluator.py)
#     run_stage1.sh --phase post  → validate _evaluator_meta, then apply plan
#
# Usage:
#   run_stage1.sh --scene_dir <dir> --phase pre  [--gpu N] [--force]
#   run_stage1.sh --scene_dir <dir> --phase eval [--gpu N] [--force]
#   run_stage1.sh --scene_dir <dir> --phase post [--gpu N] [--force]
#
#   pre  : Step 1 object-class prompt → Step 2 GroundedSAM → Step 3 init mask attrs
#   eval : Calls run_mask_evaluator.py (Opus vision); writes merge_plan.json with _evaluator_meta
#   post : Validates _evaluator_meta signature → Step 4-apply (merge_masks + optional remask)
#          → Step 5 SAM3D → Step 6 GALP → Step 7 finalize_layout
#
#   --force : delete this phase's prior outputs before running, and bypass the
#             eval cache hit. Has no effect on image.png or logs/.
#
# Fail-fast: any non-zero exit or missing expected output aborts immediately.

set -euo pipefail

# ---- Fixed paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$SCRIPT_DIR"
FINALIZE="${SCRIPT_DIR}/../../stage1-initialize-scene/src/finalize_layout.py"
# This file lives at <repo>/.claude/skills/stage1-initialize-scene/src/run_stage1.sh,
# so the repo root is 4 levels up.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DIRS_YAML="$REPO_ROOT/DIRECTORYS.yaml"

# Helper: extract a single value from DIRECTORYS.yaml by dotted key path.
_dirs_get() {
    python3 -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); v=d
for k in sys.argv[2].split('.'):
    v=v[k]
print(v)" "$DIRS_YAML" "$1"
}

# Conda env NAMES (not paths). Invoked via `conda run -n <env> python ...`.
# Override via env vars; canonical names live in DIRECTORYS.yaml::conda_envs.
SCENECONDUCTOR_ENV="${SCENECONDUCTOR_ENV:-$(_dirs_get conda_envs.sceneconductor)}"
GALP_ENV="${GALP_ENV:-$(_dirs_get conda_envs.galp)}"
GROUNDED_SAM_ENV="${GROUNDED_SAM_ENV:-$(_dirs_get conda_envs.grounded-sam)}"
SAM3D_ENV="${SAM3D_ENV:-$(_dirs_get conda_envs.sam3d-objects)}"
QWEN_ENV="${QWEN_ENV:-$(_dirs_get conda_envs.qwen-vl)}"

# ---- Args ----
SCENE_DIR=""
PHASE=""
GPU=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene_dir) SCENE_DIR="$2"; shift 2 ;;
    --phase)     PHASE="$2";     shift 2 ;;
    --gpu)       GPU="$2";       shift 2 ;;
    --force)     FORCE=1;        shift 1 ;;
    -h|--help)
      sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "[run_stage1] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SCENE_DIR" ]] || { echo "[run_stage1] --scene_dir is required" >&2; exit 2; }
[[ -d "$SCENE_DIR" ]] || { echo "[run_stage1] scene_dir not found: $SCENE_DIR" >&2; exit 2; }
[[ "$PHASE" == "pre" || "$PHASE" == "eval" || "$PHASE" == "post" ]] || \
  { echo "[run_stage1] --phase must be 'pre', 'eval', or 'post'" >&2; exit 2; }

need() { [[ -e "$1" ]] || { echo "[run_stage1] MISSING expected output: $1" >&2; exit 1; }; }
step() { echo; echo "=== [run_stage1] $* ==="; }

# ============================ PHASE: pre ============================
if [[ "$PHASE" == "pre" ]]; then
  need "$SCENE_DIR/image.png"

  if [[ "$FORCE" == "1" ]]; then
    step "Step 0 — --force: wiping prior pre-phase outputs"
    rm -f  "$SCENE_DIR/object_class_prompt.json"
    rm -f  "$SCENE_DIR/object_class.json"
    rm -f  "$SCENE_DIR/mask_attribute.json"
    rm -f  "$SCENE_DIR/overlap_pairs.json"
    rm -f  "$SCENE_DIR/small_mask_candidates.json"
    rm -f  "$SCENE_DIR/object_state_annotated_mask.png"
    rm -rf "$SCENE_DIR/masks"
  fi

  step "Step 1 — Object-class prompt"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/generate_object_classes.py" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/object_class_prompt.json"

  step "Step 2 — GroundedSAM segmentation (gpu $GPU)"
  PROMPT=$(conda run -n "$SCENECONDUCTOR_ENV" python -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['prompt'])" \
    "$SCENE_DIR/object_class_prompt.json")
  [[ -n "$PROMPT" ]] || { echo "[run_stage1] empty prompt extracted" >&2; exit 1; }
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/run_grounded_sam.py" \
    --scene_dir "$SCENE_DIR" --prompt "$PROMPT" --gpu "$GPU"
  need "$SCENE_DIR/masks"
  need "$SCENE_DIR/object_class.json"

  step "Step 3 — Initialize mask attributes"
  conda run -n "$SCENECONDUCTOR_ENV" python -c "
import sys
sys.path.insert(0, '$SCRIPTS')
from mask_attribute import init_attributes
init_attributes('$SCENE_DIR')
"
  need "$SCENE_DIR/mask_attribute.json"

  step "Step 3.5 — Annotated mask (pre-merge)"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/make_annotated_mask.py" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/object_state_annotated_mask.png"

  step "Step 3.6 — Pairwise mask-overlap pre-filter (deterministic over-seg detector)"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/compute_overlap_pairs.py" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/overlap_pairs.json"

  step "Step 3.7 — Small-mask candidates (delete / merge-into hints)"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/compute_small_mask_candidates.py" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/small_mask_candidates.json"

  step "Step 3.8 — Enrich mask attributes (shape + VLM class-consistency check)"
  QWEN_MODEL_PATH="$REPO_ROOT/$(_dirs_get checkpoints_qwen_vl)"
  conda run -n "$QWEN_ENV" python "$SCRIPTS/enrich_mask_attributes.py" \
      --scene_dir "$SCENE_DIR" --gpu "$GPU" --model "$QWEN_MODEL_PATH"
  need "$SCENE_DIR/mask_attribute.json"

  echo
  echo "=== [run_stage1] PHASE pre complete ==="
  echo "    Next: run_stage1.sh --scene_dir $SCENE_DIR --phase eval --gpu $GPU"
  echo "    then: run_stage1.sh --scene_dir $SCENE_DIR --phase post --gpu $GPU"
  exit 0
fi

# ============================ PHASE: eval ===========================
if [[ "$PHASE" == "eval" ]]; then
  # Assert pre-phase outputs exist
  need "$SCENE_DIR/overlap_pairs.json"
  need "$SCENE_DIR/small_mask_candidates.json"
  need "$SCENE_DIR/mask_attribute.json"
  need "$SCENE_DIR/masks"

  # Idempotent cache check: HIT only if image, prompt, script, and schema all match.
  MERGE_PLAN="$SCENE_DIR/merge_plan.json"
  EVAL_PROMPT_PATH="$REPO_ROOT/.claude/agents/stage1-mask-evaluator.md"
  EVAL_SCRIPT_PATH="$SCRIPTS/run_mask_evaluator.py"
  EXPECTED_SCHEMA="stage1-mask-evaluator-v2-lowest-keep-id"

  IMG_SHA=$(sha256sum "$SCENE_DIR/image.png" | awk '{print $1}')
  PROMPT_SHA=$(sha256sum "$EVAL_PROMPT_PATH" | awk '{print $1}')
  SCRIPT_SHA=$(sha256sum "$EVAL_SCRIPT_PATH" | awk '{print $1}')
  SCENE_BASENAME=$(basename "$SCENE_DIR")

  CACHE_STATUS="miss"
  CACHE_REASON="missing"
  if [[ "$FORCE" == "1" ]]; then
    rm -f "$MERGE_PLAN" "$SCENE_DIR/remask_plan.json"
    CACHE_REASON="force"
    echo "[run_stage1] --force: bypassing eval cache, deleted merge_plan.json + remask_plan.json"
  elif [[ -f "$MERGE_PLAN" ]]; then
    CACHE_REASON=$(conda run -n "$SCENECONDUCTOR_ENV" python -c "
import json, sys
try:
    plan = json.load(open(sys.argv[1]))
except Exception:
    print('malformed'); sys.exit(0)
meta = plan.get('_evaluator_meta') or {}
if meta.get('generated_by') != 'run_mask_evaluator.py':
    print('malformed'); sys.exit(0)
if meta.get('image_sha256') != sys.argv[2]:
    print('image'); sys.exit(0)
if meta.get('evaluator_prompt_sha256') != sys.argv[3]:
    print('prompt'); sys.exit(0)
if meta.get('evaluator_script_sha256') != sys.argv[4]:
    print('script'); sys.exit(0)
if meta.get('schema_version') != sys.argv[5]:
    print('schema'); sys.exit(0)
print('hit')
" "$MERGE_PLAN" "$IMG_SHA" "$PROMPT_SHA" "$SCRIPT_SHA" "$EXPECTED_SCHEMA" 2>/dev/null || echo "malformed")
    if [[ "$CACHE_REASON" == "hit" ]]; then
      CACHE_STATUS="hit"
    fi
  fi

  if [[ "$CACHE_STATUS" == "hit" ]]; then
    conda run -n "$SCENECONDUCTOR_ENV" python -c "
import json, sys
p = sys.argv[1]
plan = json.load(open(p))
meta = plan.setdefault('_evaluator_meta', {})
meta['cache_hit'] = True
open(p, 'w').write(json.dumps(plan, indent=2, ensure_ascii=False))
" "$MERGE_PLAN"
    echo "[run_stage1] eval CACHE_HIT scene=$SCENE_BASENAME image_sha=${IMG_SHA:0:12} prompt_sha=${PROMPT_SHA:0:12} script_sha=${SCRIPT_SHA:0:12}"
    exit 0
  fi

  echo "[run_stage1] eval CACHE_MISS reason=$CACHE_REASON scene=$SCENE_BASENAME image_sha=${IMG_SHA:0:12} prompt_sha=${PROMPT_SHA:0:12} script_sha=${SCRIPT_SHA:0:12}"

  step "Phase eval — Mask-Evaluator (Opus vision API call)"
  EVAL_T0=$(date +%s.%N)
  conda run -n "$SCENECONDUCTOR_ENV" python "$EVAL_SCRIPT_PATH" --scene_dir "$SCENE_DIR"
  need "$MERGE_PLAN"
  EVAL_WALL=$(awk -v t0="$EVAL_T0" -v t1="$(date +%s.%N)" 'BEGIN{printf "%.2f", t1-t0}')
  RAW_BYTES=$(conda run -n "$SCENECONDUCTOR_ENV" python -c "
import json, sys
plan = json.load(open(sys.argv[1]))
print((plan.get('_evaluator_meta') or {}).get('response_byte_size', -1))
" "$MERGE_PLAN" 2>/dev/null || echo "?")
  echo "[run_stage1] eval DONE scene=$SCENE_BASENAME wall_sec=$EVAL_WALL raw_bytes=$RAW_BYTES merge_plan=$MERGE_PLAN"

  echo
  echo "=== [run_stage1] PHASE eval complete ==="
  echo "    Next: run_stage1.sh --scene_dir $SCENE_DIR --phase post --gpu $GPU"
  exit 0
fi

# ============================ PHASE: post ===========================
if [[ "$PHASE" == "post" ]]; then
  need "$SCENE_DIR/mask_attribute.json"
  need "$SCENE_DIR/merge_plan.json"

  if [[ "$FORCE" == "1" ]]; then
    step "Step 0 — --force: wiping prior post-phase outputs"
    rm -rf "$SCENE_DIR/inputs"
    rm -rf "$SCENE_DIR/thumbnails"
    rm -f  "$SCENE_DIR/object_state.json"
    rm -f  "$SCENE_DIR/verification_overlay.png"
    rm -f  "$SCENE_DIR/layout_prediction.json"
    rm -f  "$SCENE_DIR/layout-prediction.glb"
    rm -f  "$SCENE_DIR/pointmap_xz.ply"
    rm -f  "$SCENE_DIR/floor.obj"
  fi

  step "Signature check — validate _evaluator_meta in merge_plan.json"
  POST_PROMPT_SHA=$(sha256sum "$REPO_ROOT/.claude/agents/stage1-mask-evaluator.md" | awk '{print $1}')
  POST_SCRIPT_SHA=$(sha256sum "$SCRIPTS/run_mask_evaluator.py" | awk '{print $1}')
  POST_IMG_SHA=$(sha256sum "$SCENE_DIR/image.png" | awk '{print $1}')
  POST_SCHEMA="stage1-mask-evaluator-v2-lowest-keep-id"
  conda run -n "$SCENECONDUCTOR_ENV" python -c "
import json, sys
plan_path, img_sha, prompt_sha, script_sha, schema = sys.argv[1:6]
plan = json.load(open(plan_path))
meta = plan.get('_evaluator_meta') or {}
def fail(field, got, want):
    print(f'[run_stage1] SIGNATURE FAIL: {field} mismatch (got={got!r}, want={want!r}). '
          f'Delete merge_plan.json and rerun --phase eval.', file=sys.stderr)
    sys.exit(1)
if meta.get('generated_by') != 'run_mask_evaluator.py':
    fail('generated_by', meta.get('generated_by'), 'run_mask_evaluator.py')
if meta.get('image_sha256') != img_sha:
    fail('image_sha256', meta.get('image_sha256'), img_sha)
if meta.get('evaluator_prompt_sha256') != prompt_sha:
    fail('evaluator_prompt_sha256', meta.get('evaluator_prompt_sha256'), prompt_sha)
if meta.get('evaluator_script_sha256') != script_sha:
    fail('evaluator_script_sha256', meta.get('evaluator_script_sha256'), script_sha)
if meta.get('schema_version') != schema:
    fail('schema_version', meta.get('schema_version'), schema)
print('[run_stage1] merge_plan.json signature OK (image, prompt, script, schema all match)')
" "$SCENE_DIR/merge_plan.json" "$POST_IMG_SHA" "$POST_PROMPT_SHA" "$POST_SCRIPT_SHA" "$POST_SCHEMA" || exit 1

  step "Step 4a — Apply Mask-Evaluator merge plan"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/merge_masks.py" \
    --scene_dir "$SCENE_DIR" --merge_plan "$SCENE_DIR/merge_plan.json"

  REMASK="$SCENE_DIR/remask_plan.json"
  if [[ -s "$REMASK" ]] && \
     [[ "$(conda run -n "$SCENECONDUCTOR_ENV" python -c \
          "import json,sys; print(len(json.load(open(sys.argv[1])).get('new_objects',[])))" \
          "$REMASK")" -gt 0 ]]; then
    step "Step 4b — Remask missing objects (gpu $GPU)"
    conda run -n "$GROUNDED_SAM_ENV" python "$SCRIPTS/remask_region.py" \
      --scene_dir "$SCENE_DIR" --remask_plan "$REMASK" --gpu "$GPU"
  else
    echo "[run_stage1] No remask_plan.json with new_objects — skipping Step 4b"
  fi

  step "Step 4c — Annotated mask (post-merge)"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/make_annotated_mask.py" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/object_state_annotated_mask.png"

  step "Step 5 — SAM3D textured GLB (gpu $GPU)"
  conda run -n "$SAM3D_ENV" python "$SCRIPTS/run_sam3d.py" --scene_dir "$SCENE_DIR" --gpu "$GPU"
  need "$SCENE_DIR/inputs/object"

  step "Step 6 — GALP (gpu $GPU)"
  conda run -n "$GALP_ENV" python "$SCRIPTS/run_galp.py" --scene_dir "$SCENE_DIR" --gpu "$GPU"
  need "$SCENE_DIR/layout_prediction.json"
  need "$SCENE_DIR/layout-prediction.glb"

  step "Step 6.5 — Thumbnails"
  conda run -n "$SCENECONDUCTOR_ENV" python "$SCRIPTS/make_thumbnails.py" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/thumbnails"

  step "Step 6.6 — Object state (Qwen-VL)"
  QWEN_MODEL_PATH="$REPO_ROOT/$(_dirs_get checkpoints_qwen_vl)"
  conda run -n "$QWEN_ENV" python "$SCRIPTS/extract_object_state.py" \
      --scene_dir "$SCENE_DIR" --gpu "$GPU" --model "$QWEN_MODEL_PATH" \
    || echo "[WARN] Step 6.6 object_state skipped (Qwen weights missing or env incomplete) — continuing"

  step "Step 7 — Finalize layout (move outputs into inputs/)"
  conda run -n "$SCENECONDUCTOR_ENV" python "$FINALIZE" --scene_dir "$SCENE_DIR"
  need "$SCENE_DIR/inputs"

  echo
  echo "=== [run_stage1] PHASE post complete — scene initialized ==="
  exit 0
fi
