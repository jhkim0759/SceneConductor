#!/usr/bin/env bash
# Sequential tour batch over scene dirs: build the demo (if missing) then the full
# continuous tour, one scene at a time (shared GPU). Skips scenes whose demo build
# fails or that lack relation groups.
#
#   batch_tour.sh <scene_dir1> <scene_dir2> ...
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SK="$ROOT/.claude/skills/stage-progression-videos/scripts"
DEMO="$HOME/.claude/skills/scene-conductor-demo-video/scripts/make_demo.py"
BLENDER="${BLENDER:-$ROOT/blender-4.2.1-linux-x64/blender}"
LOGD="$ROOT/ignored/_tour_batch_logs"; mkdir -p "$LOGD"
echo "=== tour batch: $# scenes ==="
for SCENE in "$@"; do
  SCENE="${SCENE%/}"; s=$(basename "$SCENE")
  echo ">>> [$(date +%H:%M:%S)] $s START"
  if [ ! -f "$SCENE/report/demo/demo_animated.blend" ]; then
    echo "    demo build (--skip-render)..."
    conda run -n sceneconductor python "$DEMO" --blend-folder "$SCENE/blend" \
      --out-folder "$SCENE/report/demo" --skip-render --engine cycles --blender "$BLENDER" \
      > "$LOGD/${s}_demo.log" 2>&1
    echo "    demo rc=$?"
  fi
  if [ ! -f "$SCENE/report/demo/demo_animated.blend" ]; then
    echo ">>> [$(date +%H:%M:%S)] $s : demo_animated.blend missing -> SKIP"
    continue
  fi
  bash "$SK/make_tour.sh" "$SCENE" 48 > "$LOGD/${s}_tour.log" 2>&1
  out="$SCENE/report/stage_videos/tour_full.mp4"
  echo ">>> [$(date +%H:%M:%S)] $s : DONE ($([ -f "$out" ] && echo "OK -> $out" || echo NO-OUTPUT))"
done
echo "=== tour batch complete ==="
