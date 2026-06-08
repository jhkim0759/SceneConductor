#!/usr/bin/env bash
# build_all_scenes_seq.sh
# Drive Stage 1 → Stage 2 → Stage 3 sequentially on every scene in DATASET_DIR,
# one stage per fresh `claude -p` invocation. Unlike build_all_scenes.sh (which
# delegates the whole pipeline to /scene-orchestration in a single claude
# session), this script spawns THREE separate sessions per scene — one for each
# stage skill — so each stage gets its own empty context window
# (== /clear between stages).
#
# Per-stage resume / skip logic mirrors what /scene-orchestration does
# internally (see .claude/skills/scene-orchestration/SKILL.md "Resume / skip
# table"). If a stage fails (non-zero exit OR missing post-check artifacts),
# the remaining stages for THAT scene are skipped and the loop moves to the
# next scene.
#
# Usage:
#   DATASET_DIR=/path/to/dataset bash build_all_scenes_seq.sh
#   DATASET_DIR=/path/to/dataset ITERS=5 bash build_all_scenes_seq.sh
#   DATASET_DIR=/path/to/dataset FROM=bedroom_b1 bash build_all_scenes_seq.sh
#   DATASET_DIR=/path/to/dataset ONLY="scene_a scene_b" bash build_all_scenes_seq.sh
#   DATASET_DIR=/path/to/dataset SPLITS=4 SPLIT=2 bash build_all_scenes_seq.sh
#
# Each stage's full transcript goes to $LOG_DIR/<scene>.stage{1,2,3}.log.
# A single overall progress log goes to $LOG_DIR/_progress.log.
#
# Env overrides:
#   DATASET_DIR=…  REQUIRED — parent directory containing per-scene subdirs
#   GPU=N          GPU index (default 0, Stage 1 only — Stage 2/3 have no GPU surface)
#   ITERS=N        island-refiner iters in [1,10] (default 10, Stage 3 only)
#   MODEL=…        claude model id (default: empty → CLI default)
#   SKIP_DONE=0|1  skip each stage if its completion files exist (default 1)
#                    Stage 1 done = inputs/layout_prediction.json
#                                   AND inputs/object_class.json
#                    Stage 2 done = blend/blender_scene.blend
#                    Stage 3 done = blend/stage3-scene.blend
#                                   AND render/final/blender_scene_view_perspective.png
#   FORCE=0|1      force re-run from Stage 1 (default 0).
#                  When 1: implies SKIP_DONE=0 (ignores existing outputs) AND
#                          appends `--force` to the /stage1-initialize-scene prompt
#                          so Stage 1 wipes inputs/ and re-runs from scratch.
#                          Stage 2 and Stage 3 have no --force surface; with
#                          fresh Stage-1 outputs they will naturally re-run.
#   STAGE3_FORCE=0|1   force re-run of Stage 3 only (default 0).
#                  When 1: Stage 1 and Stage 2 still respect SKIP_DONE (so they
#                          are skipped if their outputs exist). Stage 3 outputs
#                          (blend/stage3-*.blend, render/planned, render/final,
#                          relation_groups, scene-refine-loop, json/stage3_*,
#                          operation_plan*, heuristic_ops, graph_ops, llm_ops,
#                          island_groups, relation_pairs/solve_ops) are MOVED to
#                          <scene>/.stage3_backup_<UTC>/ before re-invoking
#                          /stage3-scene-refinement. scene-analyze-prepare outputs
#                          (json/object_state.json, json/blend_info.json,
#                          inputs/relation_graph.json) are LEFT IN PLACE.
#                          Ignored when FORCE=1 (FORCE is stronger).
#   FROM=…         resume from this scene name; skip everything before it alphabetically
#                  (ignored when ONLY is set)
#   ONLY=…         space-separated scene-name subset; if set, FROM is ignored
#   SPLITS=N       split the (post-ONLY/FROM) scene list into N contiguous chunks
#                  (default 1 = no split). Useful for running multiple shards in parallel
#                  on different GPUs/hosts.
#   SPLIT=K        1-indexed chunk number to run, must be in [1, SPLITS] (default 1).
#   LOG_DIR=…      where to write logs (default <DATASET_DIR>/_build_logs_seq/)
# Example) DATASET_DIR=test_dataset SPLITS=4 SPLIT=1 GPU=1 sh scripts/build_all_scenes_seq.sh

set -u
set -o pipefail

# -------- config --------
DATASET_DIR="${DATASET_DIR:-}"
if [ -z "$DATASET_DIR" ]; then
  echo "ERROR: DATASET_DIR is required but not set." >&2
  echo "Usage: DATASET_DIR=/path/to/dataset bash $0" >&2
  exit 2
fi

# Strip trailing slash
DATASET_DIR="${DATASET_DIR%/}"

if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: DATASET_DIR not found or not a directory: $DATASET_DIR" >&2
  exit 2
fi

GPU="${GPU:-0}"
ITERS="${ITERS:-10}"
MODEL="${MODEL:-}"
FROM="${FROM:-}"              # resume from this scene (skip everything before it)
ONLY="${ONLY:-}"              # space-separated subset; if set, FROM is ignored
SPLITS="${SPLITS:-1}"         # total number of shards to divide the scene list into
SPLIT="${SPLIT:-1}"           # 1-indexed shard number to actually run
SKIP_DONE="${SKIP_DONE:-1}"   # 1 = skip each stage that already has its outputs
FORCE="${FORCE:-0}"           # 1 = force re-run from Stage 1 (overrides SKIP_DONE, appends --force to stage1)
STAGE3_FORCE="${STAGE3_FORCE:-0}"   # 1 = force re-run of Stage 3 only (keeps Stage 1/2)
LOG_DIR="${LOG_DIR:-$DATASET_DIR/_build_logs_seq}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -------- validate FORCE --------
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "ERROR: FORCE must be 0 or 1, got: '$FORCE'" >&2
  exit 2
fi
if [[ "$STAGE3_FORCE" != "0" && "$STAGE3_FORCE" != "1" ]]; then
  echo "ERROR: STAGE3_FORCE must be 0 or 1, got: '$STAGE3_FORCE'" >&2
  exit 2
fi

# FORCE implies SKIP_DONE=0
if [ "$FORCE" = "1" ]; then
  SKIP_DONE=0
fi

# FORCE is stronger — when FORCE=1, ignore STAGE3_FORCE (Stage 1 wipes everything anyway)
if [ "$FORCE" = "1" ] && [ "$STAGE3_FORCE" = "1" ]; then
  STAGE3_FORCE=0
fi

# -------- validate GPU --------
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU must be a non-negative integer, got: '$GPU'" >&2
  exit 2
fi

# -------- validate ITERS --------
if ! [[ "$ITERS" =~ ^[0-9]+$ ]] || [ "$ITERS" -lt 1 ] || [ "$ITERS" -gt 10 ]; then
  echo "ERROR: ITERS must be an integer in [1, 10], got: '$ITERS'" >&2
  exit 2
fi

# -------- validate SPLITS / SPLIT --------
if ! [[ "$SPLITS" =~ ^[0-9]+$ ]] || [ "$SPLITS" -lt 1 ]; then
  echo "ERROR: SPLITS must be a positive integer, got: '$SPLITS'" >&2
  exit 2
fi
if ! [[ "$SPLIT" =~ ^[0-9]+$ ]] || [ "$SPLIT" -lt 1 ] || [ "$SPLIT" -gt "$SPLITS" ]; then
  echo "ERROR: SPLIT must be an integer in [1, $SPLITS], got: '$SPLIT'" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
PROGRESS_LOG="$LOG_DIR/_progress.log"

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$PROGRESS_LOG"
}

# -------- enumerate scenes --------
# Alphabetical order, only directories that actually contain image.png.
ALL_SCENES=()
for d in "$DATASET_DIR"/*/; do
  name=$(basename "$d")
  [ -f "$d/image.png" ] || continue
  ALL_SCENES+=("$name")
done

if [ ${#ALL_SCENES[@]} -eq 0 ]; then
  log "no scenes found in $DATASET_DIR (no subdirs with image.png)"
  exit 1
fi

# Apply ONLY filter (highest priority)
if [ -n "$ONLY" ]; then
  SCENES=()
  for want in $ONLY; do
    for have in "${ALL_SCENES[@]}"; do
      if [ "$want" = "$have" ]; then
        SCENES+=("$have")
        break
      fi
    done
  done
elif [ -n "$FROM" ]; then
  SCENES=()
  started=0
  for s in "${ALL_SCENES[@]}"; do
    [ "$s" = "$FROM" ] && started=1
    [ $started -eq 1 ] && SCENES+=("$s")
  done
else
  SCENES=("${ALL_SCENES[@]}")
fi

if [ ${#SCENES[@]} -eq 0 ]; then
  log "no scenes matched — exiting"
  exit 1
fi

# -------- apply SPLITS/SPLIT slicing --------
# Contiguous-chunk shard. Boundaries use floor((K-1)*N/S) .. floor(K*N/S),
# which keeps shard sizes within ±1 of each other.
PRE_SPLIT_TOTAL=${#SCENES[@]}
if [ "$SPLITS" -gt 1 ]; then
  start=$(( (SPLIT - 1) * PRE_SPLIT_TOTAL / SPLITS ))
  end=$(( SPLIT * PRE_SPLIT_TOTAL / SPLITS ))
  count=$(( end - start ))
  if [ "$count" -le 0 ]; then
    SCENES=()
  else
    SCENES=("${SCENES[@]:start:count}")
  fi
fi

if [ ${#SCENES[@]} -eq 0 ]; then
  log "split $SPLIT/$SPLITS is empty (pre-split total=$PRE_SPLIT_TOTAL) — exiting"
  exit 1
fi

log "===================================================================="
log "build_all_scenes_seq start  total=${#SCENES[@]}  gpu=$GPU  iters=$ITERS  model=${MODEL:-<default>}  force=$FORCE  stage3_force=$STAGE3_FORCE"
log "split=$SPLIT/$SPLITS  pre_split_total=$PRE_SPLIT_TOTAL"
log "dataset_dir=$DATASET_DIR"
log "log_dir=$LOG_DIR"
log "===================================================================="

# -------- helper: run one stage in its own `claude -p` session --------
# Args: scene, scene_dir, stage_num, prompt, stage_log
# Returns: 0 on success, non-zero on claude failure
run_stage() {
  local scene="$1"
  local scene_dir="$2"
  local stage_num="$3"
  local prompt="$4"
  local stage_log="$5"

  local start_ts end_ts elapsed hh mm ss
  start_ts=$(date +%s)
  log "  [stage$stage_num start] $scene  prompt=\"$prompt\""

  local cmd=(claude -p "$prompt"
       --permission-mode bypassPermissions
       --add-dir "$scene_dir"
       --add-dir "$REPO_ROOT")
  [ -n "$MODEL" ] && cmd+=(--model "$MODEL")

  "${cmd[@]}" > "$stage_log" 2>&1
  local rc=$?

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  hh=$((elapsed / 3600)); mm=$(((elapsed % 3600) / 60)); ss=$((elapsed % 60))

  if [ $rc -eq 0 ]; then
    log "  [stage$stage_num done]  $scene  exit=0  elapsed=${hh}h${mm}m${ss}s  log=$stage_log"
  else
    log "  [stage$stage_num FAIL] $scene  exit=$rc  elapsed=${hh}h${mm}m${ss}s  log=$stage_log"
  fi
  return $rc
}

# -------- helper: stage completion checks (mirrors scene-orchestration) --------
stage1_done() {
  local scene_dir="$1"
  [ -f "$scene_dir/inputs/layout_prediction.json" ] \
    && [ -f "$scene_dir/inputs/object_class.json" ]
}
stage2_done() {
  local scene_dir="$1"
  [ -f "$scene_dir/blend/blender_scene.blend" ]
}
stage3_done() {
  local scene_dir="$1"
  [ -f "$scene_dir/blend/stage3-scene.blend" ] \
    && [ -f "$scene_dir/render/final/blender_scene_view_perspective.png" ]
}

# Move all Stage 3 outputs to <scene>/.stage3_backup_<UTC>/ — mirrors orchestrate.py's
# _backup_and_reset_stage3 with skip_prepare=True (scene-analyze-prepare outputs
# stay in place: json/object_state.json, json/blend_info.json,
# inputs/relation_graph.json — so prepare is not re-run).
# Stage 1 / Stage 2 outputs (image.png, inputs/, blend/blender_scene.blend,
# json/blender_scene.json, etc.) are NEVER touched.
stage3_backup_and_clean() {
  local scene_dir="$1"
  local ts
  ts=$(date -u +%Y%m%d_%H%M%SZ)
  local backup_dir="$scene_dir/.stage3_backup_$ts"
  local items=(
    "json/stage3_state.json"
    "json/operation_plan.json"
    "json/operation_plan_revised.json"
    "json/heuristic_ops.json"
    "json/graph_ops.json"
    "json/llm_ops.json"
    "json/island_groups.json"
    "json/relation_pairs.json"
    "json/relation_solve_ops.json"
    "blend/stage3-sub-planned.blend"
    "blend/stage3-scene.blend"
    "render/planned"
    "render/final"
    "relation_groups"
    "scene-refine-loop"
  )
  local moved=0
  for item in "${items[@]}"; do
    if [ -e "$scene_dir/$item" ]; then
      if [ "$moved" = "0" ]; then
        mkdir -p "$backup_dir"
        moved=1
      fi
      mkdir -p "$backup_dir/$(dirname "$item")"
      mv "$scene_dir/$item" "$backup_dir/$item"
    fi
  done
  if [ "$moved" = "1" ]; then
    echo "$backup_dir"
  else
    # nothing existed — remove the (empty) backup dir if it got created
    [ -d "$backup_dir" ] && rmdir "$backup_dir" 2>/dev/null
    echo ""
  fi
}

# -------- main loop --------
for scene in "${SCENES[@]}"; do
  scene_dir="$DATASET_DIR/$scene"
  scene_log_prefix="$LOG_DIR/$scene"

  if [ ! -f "$scene_dir/image.png" ]; then
    log "[skip] $scene  reason=no_image_png"
    continue
  fi

  # Whole-scene short-circuit: skip if Stage 3 is already done
  # (but NOT when STAGE3_FORCE=1 — caller explicitly wants a Stage-3 redo).
  if [ "$SKIP_DONE" = "1" ] && [ "$STAGE3_FORCE" != "1" ] && stage3_done "$scene_dir"; then
    log "[skip] $scene  reason=already_built (stage3 outputs exist)"
    continue
  fi

  scene_start_ts=$(date +%s)
  log "[scene start] $scene"

  # ---------- Stage 1 ----------
  if [ "$SKIP_DONE" = "1" ] && stage1_done "$scene_dir"; then
    log "  [stage1 skip] $scene  reason=stage1 outputs exist"
  else
    s1_prompt="/stage1-initialize-scene $scene_dir --gpu $GPU"
    if [ "$FORCE" = "1" ]; then
      s1_prompt="$s1_prompt --force"
    fi
    if ! run_stage "$scene" "$scene_dir" 1 "$s1_prompt" "$scene_log_prefix.stage1.log"; then
      log "[scene FAIL] $scene  failed_stage=1  skipping remaining stages"
      continue
    fi
    if ! stage1_done "$scene_dir"; then
      log "  [stage1 POSTCHECK FAIL] $scene  missing: inputs/layout_prediction.json or inputs/object_class.json"
      log "[scene FAIL] $scene  failed_stage=1  skipping remaining stages"
      continue
    fi
  fi

  # ---------- Stage 2 ----------
  if [ "$SKIP_DONE" = "1" ] && stage2_done "$scene_dir"; then
    log "  [stage2 skip] $scene  reason=stage2 outputs exist"
  else
    s2_prompt="/stage2-environment-construction $scene_dir"
    if ! run_stage "$scene" "$scene_dir" 2 "$s2_prompt" "$scene_log_prefix.stage2.log"; then
      log "[scene FAIL] $scene  failed_stage=2  skipping remaining stages"
      continue
    fi
    if ! stage2_done "$scene_dir"; then
      log "  [stage2 POSTCHECK FAIL] $scene  missing: blend/blender_scene.blend"
      log "[scene FAIL] $scene  failed_stage=2  skipping remaining stages"
      continue
    fi
  fi

  # ---------- Stage 3 ----------
  if [ "$STAGE3_FORCE" = "1" ]; then
    bak=$(stage3_backup_and_clean "$scene_dir")
    if [ -n "$bak" ]; then
      log "  [stage3 force-clean] $scene  backup=$bak"
    else
      log "  [stage3 force-clean] $scene  no prior stage3 outputs to back up"
    fi
  fi

  if [ "$SKIP_DONE" = "1" ] && [ "$STAGE3_FORCE" != "1" ] && stage3_done "$scene_dir"; then
    log "  [stage3 skip] $scene  reason=stage3 outputs exist"
  else
    s3_prompt="/stage3-scene-refinement $scene_dir --num-max-iter $ITERS"
    if ! run_stage "$scene" "$scene_dir" 3 "$s3_prompt" "$scene_log_prefix.stage3.log"; then
      log "[scene FAIL] $scene  failed_stage=3"
      continue
    fi
    if ! stage3_done "$scene_dir"; then
      log "  [stage3 POSTCHECK FAIL] $scene  missing: blend/stage3-scene.blend or render/final/blender_scene_view_perspective.png"
      log "[scene FAIL] $scene  failed_stage=3"
      continue
    fi
  fi

  scene_end_ts=$(date +%s)
  scene_elapsed=$((scene_end_ts - scene_start_ts))
  hh=$((scene_elapsed / 3600)); mm=$(((scene_elapsed % 3600) / 60)); ss=$((scene_elapsed % 60))
  log "[scene done] $scene  total_elapsed=${hh}h${mm}m${ss}s"
done

log "===================================================================="
log "build_all_scenes_seq finished"
log "===================================================================="
