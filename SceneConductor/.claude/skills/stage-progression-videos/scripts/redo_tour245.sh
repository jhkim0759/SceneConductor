#!/usr/bin/env bash
# Re-render clips 2 (env build, lights-after-build), 4 (static group view) and
# 5 (near-BEV turntable) for a scene, then SIMPLE-concat 1+2+3+4+5 -> tour_full.
# Reuses the existing clips 1,3 and the prep (g_init/g_final).
#   redo_tour245.sh <scene_dir> [samples]
set -u
SCENE="${1%/}"; SAMPLES="${2:-48}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SK="$ROOT/.claude/skills/stage-progression-videos/scripts"
BL="${BLENDER:-$ROOT/blender-4.2.1-linux-x64/blender}"
SV="$SCENE/report/stage_videos"; W="$SCENE/report/_stage_videos_work"; LOG="$SCENE/report/_spv_logs"
TENV="SPV_ENGINE=cycles SPV_ORIGINAL_LIGHTS=1 SPV_LIGHT_SCALE=1.0 SPV_CROP_PX=0 SPV_SAVE_BLEND=0 CUDA_VISIBLE_DEVICES=0"
GINIT=$(ls "$W"/*_init.json 2>/dev/null | head -1)
if [ -z "$GINIT" ]; then echo "REDO245_SKIP $SCENE (no group json)"; exit 2; fi
GBASE=$(basename "$GINIT" _init.json); GFINAL="$W/${GBASE}_final.json"
GRP=$(python3 -c "import json;d=json.load(open('$GINIT'));print(','.join(k for k in d['objects'] if k.startswith('obj_')))")
echo "[redo245] $SCENE group=$GBASE"
env $TENV "$BL" -b "$SCENE/blend/blender_scene.blend" --python "$SK/build_stage_sync.py" -- \
  "$SV/2_env_build_interior.mp4" --view interior --samples "$SAMPLES" > "$LOG/c2.log" 2>&1
env $TENV "$BL" -b "$SCENE/blend/blender_scene.blend" --python "$SK/build_tour.py" -- \
  "$LOG/tour_4_zoom.mp4" --phase zoom --group "$GRP" --ginit "$GINIT" --gfinal "$GFINAL" --samples "$SAMPLES" > "$LOG/tour4.log" 2>&1
env $TENV "$BL" -b "$SCENE/blend/blender_scene.blend" --python "$SK/build_tour.py" -- \
  "$LOG/tour_5_turntable.mp4" --phase turntable --group "$GRP" --gfinal "$GFINAL" --samples "$SAMPLES" > "$LOG/tour5.log" 2>&1
conda run -n sceneconductor python "$SK/concat_tour.py" "$SV/tour_full.mp4" \
  "$SV/1_stage1_popup.mp4" "$SV/2_env_build_interior.mp4" "$SV/3_stage3_first_refine.mp4" \
  "$LOG/tour_4_zoom.mp4" "$LOG/tour_5_turntable.mp4" > "$LOG/concat.log" 2>&1
echo "REDO245_DONE $SCENE -> $SV/tour_full.mp4"
