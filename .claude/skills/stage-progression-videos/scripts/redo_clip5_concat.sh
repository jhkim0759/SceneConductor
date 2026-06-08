#!/usr/bin/env bash
# Render clip 5 (clean full-scene near-BEV turntable) and SIMPLE-concat the final
# tour 1+2+3+5 -> tour_full.mp4. Reuses existing clips 1,2,3. (Tour has no clip 4.)
#   redo_clip5_concat.sh <scene_dir> [samples]
set -u
SCENE="${1%/}"; SAMPLES="${2:-48}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SK="$ROOT/.claude/skills/stage-progression-videos/scripts"
BL="${BLENDER:-$ROOT/blender-4.2.1-linux-x64/blender}"
SV="$SCENE/report/stage_videos"; LOG="$SCENE/report/_spv_logs"; mkdir -p "$LOG"
TENV="SPV_ENGINE=cycles SPV_ORIGINAL_LIGHTS=1 SPV_LIGHT_SCALE=1.0 SPV_CROP_PX=0 SPV_SAVE_BLEND=0 CUDA_VISIBLE_DEVICES=0"
echo "[clip5] $SCENE turntable"
env $TENV "$BL" -b "$SCENE/blend/blender_scene.blend" --python "$SK/build_stage_sync.py" -- \
  "$LOG/clip5_turntable.mp4" --view turntable --samples "$SAMPLES" > "$LOG/c5.log" 2>&1
conda run -n sceneconductor python "$SK/concat_tour.py" "$SV/tour_full.mp4" \
  "$SV/1_stage1_popup.mp4" "$SV/2_env_build_interior.mp4" "$SV/3_stage3_first_refine.mp4" \
  "$LOG/clip5_turntable.mp4" > "$LOG/concat.log" 2>&1
echo "CLIP5_CONCAT_DONE $SCENE -> $SV/tour_full.mp4"
