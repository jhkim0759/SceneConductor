#!/usr/bin/env bash
# Full continuous-tour video for ONE scene, the TEASER recipe:
#   clips 1,2,3 (driver, original lights / original camera) + 4 zoom & 5 near-BEV
#   turntable (build_tour, boundary-aligned) -> concat 1+2+3+4+5 = tour_full.mp4
#
#   make_tour.sh <scene_dir> [samples]
# Assumes the scene already has demo_animated.blend + stage_data (run make_demo
# first if missing). Renders on GPU 0, Cycles, original lights.
set -u
SCENE="${1%/}"
SAMPLES="${2:-48}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SK="$ROOT/.claude/skills/stage-progression-videos/scripts"
BL="${BLENDER:-$ROOT/blender-4.2.1-linux-x64/blender}"
SV="$SCENE/report/stage_videos"
W="$SCENE/report/_stage_videos_work"
LOG="$SCENE/report/_spv_logs"; mkdir -p "$LOG"
TENV="SPV_ENGINE=cycles SPV_ORIGINAL_LIGHTS=1 SPV_LIGHT_SCALE=1.0 SPV_CROP_PX=0 SPV_SAVE_BLEND=0 CUDA_VISIBLE_DEVICES=0"

echo "[tour] $SCENE : clips 1,2,3 (driver)"
conda run -n sceneconductor python "$SK/make_stage_videos.py" "$SCENE" --gpu 0 \
  --engine cycles --refine-engine cycles --samples "$SAMPLES" \
  --videos 1_popup,2_env_interior,3_first_refine > "$LOG/driver_123.log" 2>&1
rc=$?
echo "[tour] driver rc=$rc"

GINIT=$(ls "$W"/*_init.json 2>/dev/null | head -1)
if [ -z "$GINIT" ]; then echo "[tour] NO island group json -> abort $SCENE"; exit 2; fi
GBASE=$(basename "$GINIT" _init.json)
GFINAL="$W/${GBASE}_final.json"
GRP=$(python3 -c "import json;d=json.load(open('$GINIT'));print(','.join(k for k in d['objects'] if k.startswith('obj_')))")
echo "[tour] island group=$GBASE objs=$GRP"

echo "[tour] clip 4 (zoom)"
env $TENV "$BL" -b "$SCENE/blend/blender_scene.blend" --python "$SK/build_tour.py" -- \
  "$LOG/tour_4_zoom.mp4" --phase zoom --group "$GRP" --ginit "$GINIT" --gfinal "$GFINAL" \
  --samples "$SAMPLES" > "$LOG/tour4.log" 2>&1
echo "[tour] clip 5 (turntable)"
env $TENV "$BL" -b "$SCENE/blend/blender_scene.blend" --python "$SK/build_tour.py" -- \
  "$LOG/tour_5_turntable.mp4" --phase turntable --group "$GRP" --gfinal "$GFINAL" \
  --samples "$SAMPLES" > "$LOG/tour5.log" 2>&1

echo "[tour] concat 1+2+3+4+5"
conda run -n sceneconductor python "$SK/concat_tour.py" "$SV/tour_full.mp4" \
  "$SV/1_stage1_popup.mp4" "$SV/2_env_build_interior.mp4" "$SV/3_stage3_first_refine.mp4" \
  "$LOG/tour_4_zoom.mp4" "$LOG/tour_5_turntable.mp4" > "$LOG/concat.log" 2>&1
echo "TOUR_DONE $SCENE -> $SV/tour_full.mp4"
