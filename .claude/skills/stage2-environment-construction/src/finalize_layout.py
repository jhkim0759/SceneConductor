"""finalize_layout.py — stragglers safety-net for stage2-environment-construction.

Sub-skills now write directly to their canonical subfolders:
  json/     — polygon.json, alignment_metrics.json, polygon_v2.json,
               alignment_metrics_v2.json, brightness_align_log.json,
               stage_refine_summary.json, blender_scene_pre_refine.json,
               stage_critique_iter*.json, stage_planner_decision_iter*.json,
               executor_log_iter*.json
  render/   — *_env_preview.png, blender_scene_view_*.png, stage_refine_iter*_*.png,
               blender_scene_bev_overlay.png, blender_scene_v2_bev_overlay.png
  inputs/   — layout_prediction.json, layout-prediction.glb, mask_attribute.json,
               object/, pointmap_xz.ply
  scene-pipeline/ — blender_scene_stage*.blend, blender_scene_pre_refine.blend,
                    blender_scene.blend1

This script's only job is to catch any stragglers that were written to the
top level by an older or out-of-date sub-skill and move them into the correct
subfolder.  It does NOT move canonical top-level files (blender_scene.json,
blender_scene.blend, image.*) or the inputs/ directory itself.

Idempotent — safe to re-run.  Files already in the target subfolder are left
untouched.  CLI signature is unchanged from the previous version.
"""

import argparse
import glob
import shutil
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so the em-dash and other non-ASCII bytes in our
# log lines don't trip Windows' cp949/cp1252 default encoding.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Stragglers map — (destination_subfolder, [exact_names_at_top_level])
# ---------------------------------------------------------------------------

_RENDER_EXACT = [
    "blender_scene_env_preview.png",
    "blender_scene_env_preview0001.png",
    "blender_scene_view_perspective.png",
    "blender_scene_view_bev.png",
    "blender_scene_view_wide.png",
    "blender_scene_view_topcorner.png",
    "blender_scene_view_topcorner_opposite.png",
    "blender_scene_bev_overlay.png",
    "blender_scene_v2_bev_overlay.png",
]

_RENDER_GLOBS = [
    "stage_refine_iter*_main.png",
    "stage_refine_iter*_bev.png",
]

_JSON_EXACT = [
    "polygon.json",
    "polygon_v2.json",
    "alignment_metrics.json",
    "alignment_metrics_v2.json",
    "brightness_align_log.json",
    "stage_refine_summary.json",
    "blender_scene_pre_refine.json",
]

_JSON_GLOBS = [
    "stage_critique_iter*.json",
    "stage_planner_decision_iter*.json",
    "executor_log_iter*.json",
]

_PIPELINE_EXACT = [
    "blender_scene_pre_refine.blend",
    "blender_scene.blend1",
]

_PIPELINE_GLOBS = [
    "blender_scene_stage*.blend",
]

_INPUTS_EXACT = [
    "layout_prediction.json",
    "layout-prediction.glb",
    "mask_attribute.json",
    "pointmap_xz.ply",
]


# ---------------------------------------------------------------------------
# Core move helper
# ---------------------------------------------------------------------------

def _move(src: Path, dest_dir: Path) -> None:
    """Move src into dest_dir, creating dest_dir if needed; overwrite on collision."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src.name
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    subfolder = dest_dir.relative_to(dest_dir.parent)
    print(f"[finalize/straggler] {src.name} → {subfolder}/{src.name}", flush=True)


def _sweep_exact(scene_dir: Path, names: list, dest_dir: Path) -> int:
    moved = 0
    for name in names:
        src = scene_dir / name
        if src.exists():
            _move(src, dest_dir)
            moved += 1
    return moved


def _sweep_globs(scene_dir: Path, patterns: list, dest_dir: Path) -> int:
    moved = 0
    for pattern in patterns:
        for match in sorted(glob.glob(str(scene_dir / pattern))):
            src = Path(match)
            if src.parent == scene_dir:  # top-level only
                _move(src, dest_dir)
                moved += 1
    return moved


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def finalize(scene_dir: Path) -> None:
    """Scan top-level of scene_dir for straggler artifacts and move to correct subfolders.

    This is a safety-net only.  In a correctly configured pipeline all outputs
    are written directly to their canonical subfolders and this function is
    effectively a no-op.
    """
    render_dir   = scene_dir / "render"
    json_dir     = scene_dir / "json"
    pipeline_dir = scene_dir / "scene-pipeline"
    inputs_dir   = scene_dir / "inputs"

    total = 0
    total += _sweep_exact(scene_dir, _RENDER_EXACT,   render_dir)
    total += _sweep_globs(scene_dir, _RENDER_GLOBS,   render_dir)
    total += _sweep_exact(scene_dir, _JSON_EXACT,     json_dir)
    total += _sweep_globs(scene_dir, _JSON_GLOBS,     json_dir)
    total += _sweep_exact(scene_dir, _PIPELINE_EXACT, pipeline_dir)
    total += _sweep_globs(scene_dir, _PIPELINE_GLOBS, pipeline_dir)
    total += _sweep_exact(scene_dir, _INPUTS_EXACT,   inputs_dir)

    if total == 0:
        print("[finalize] no stragglers found — all artifacts already in correct subfolders.", flush=True)
    else:
        print(f"[finalize] swept {total} straggler(s) into subfolders.", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Safety-net stragglers sweep: moves any top-level pipeline artifacts "
            "that were not written directly to their canonical subfolders "
            "(render/, json/, inputs/, scene-pipeline/) into the correct location. "
            "In a correctly configured pipeline this is effectively a no-op."
        )
    )
    ap.add_argument("scene_dir", help="Path to scene folder")
    args = ap.parse_args()
    finalize(Path(args.scene_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
