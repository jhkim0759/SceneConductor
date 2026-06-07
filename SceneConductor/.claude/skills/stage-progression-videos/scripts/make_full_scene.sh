#!/usr/bin/env bash
# Render the FULL-video clip set (1,2,3,3-1,5) for ONE scene on ONE GPU with the
# batch recipe: focal 0.7 (wider), clip-5 walls visible, Cycles, original lights
# scaled by <light_scale> uniformly (so the clips stay light-matched/connected),
# 1120x840 crop. Then simple-concat the present clips -> report/stage_videos/tour_full.mp4
#   make_full_scene.sh <scene_dir> <gpu> <light_scale> [videos_csv]
set -u
SCENE="${1%/}"; GPU="$2"; LS="$3"
VIDEOS="${4:-1_popup,2_env_interior,3_first_refine,3-1_to_final,5_turntable}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SK="$ROOT/.claude/skills/stage-progression-videos/scripts"
DRV="$SK/make_stage_videos.py"
SV="$SCENE/report/stage_videos"; LOG="$SCENE/report/_spv_logs"; mkdir -p "$LOG"

echo "[full] $SCENE gpu=$GPU light=$LS videos=$VIDEOS"
env SPV_FOCAL_SCALE=0.7 SPV_TT_WALLS=1 \
  conda run -n sceneconductor python "$DRV" "$SCENE" --gpu "$GPU" \
    --videos "$VIDEOS" --engine cycles --refine-engine cycles \
    --cycles-light-scale "$LS" --samples 48 > "$LOG/full_driver.log" 2>&1

# concat tour_full from whichever of the 5 clips exist (keeps order)
EX=""
for c in 1_stage1_popup 2_env_build_interior 3_stage3_first_refine 3-1_stage3_to_final 5_stage3_final_turntable; do
  [ -f "$SV/$c.mp4" ] && EX="$EX $SV/$c.mp4"
done
conda run -n sceneconductor python "$SK/concat_tour.py" "$SV/tour_full.mp4" $EX > "$LOG/concat_full.log" 2>&1
echo "FULL_DONE $SCENE -> $SV/tour_full.mp4 (clips:$EX)"
