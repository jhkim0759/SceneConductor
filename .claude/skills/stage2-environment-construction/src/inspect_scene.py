#!/usr/bin/env python3
"""Scene-folder inspector for the stage2-environment-construction orchestrator.

Given a scene directory, prints a JSON plan describing which of the 6
pipeline stages (0–5) are done, ready, or blocked — based purely on which
files exist in the folder. No Blender, no heavy deps. Pure stdlib.

Status rules for each stage:
  - done:    its primary "done" output file(s) / conditions are met.
  - ready:   its required inputs exist AND its done-signal does not.
  - blocked: at least one required input is missing (and not done yet).

Stage dependencies (soft, via inputs):
  S0   stage2-environment-planner (agent)      needs: image.*
                                    produces: json/stage2_plan.json
                                    (advisory hints consumed by S1, S3, S4)
  S1   stage2-sub-pointmap-to-separable-stage       needs: layout_prediction.json + image.*
                                    produces: blender_scene.json
  S2   stage2-sub-pointmap-to-separable-stage        needs: blender_scene.json
                                    produces: blend/blender_scene.blend
  S3   stage2-sub-pointmap-to-separable-stage  needs: image.* + pointmap_xz.ply
                                    produces: polygon_v2.json + alignment_metrics_v2.json
                                              + blender_scene.json["stage"] block
  S4   stage2-sub-env-enhance            needs: image.* + a .blend containing stage + objects
                                    produces: *_env_preview.png
                                              + lighting/world/stage_materials/render blocks in JSON
  S5   multi-view-render            needs: S4 done (blender_scene.blend with env blocks)
                                    produces: blender_scene_view_perspective.png

Forcing:
  --force all            : mark every stage as ready if inputs allow (ignore done signals).
  --force-from N         : re-plan stages >= N (their done signals ignored; N+ cascade).
                           N must be an integer 0–5.
  --until N              : stages > N are marked as "skipped_by_until".
                           N must be an integer 0–5.

Usage:
  python3 inspect_scene.py <scene_dir> [--force all|off] [--force-from N] [--until N]

Exit code is always 0 unless the scene_dir itself is missing. The caller
(the orchestrator skill) reads the JSON on stdout and makes decisions.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional


IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# Canonical stage ordering for sorting / comparison purposes.
# Stage 0 = stage2-environment-planner (vision agent producing json/stage2_plan.json).
STAGE_ORDER: list[float] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
STAGE_KEY_TO_FLOAT: Dict[str, float] = {
    "0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0,
}


def _parse_stage_arg(value: str) -> float:
    """Parse a stage argument that must be an integer string 0–5.

    Raises argparse.ArgumentTypeError on invalid input.
    """
    normalized = value.strip()
    if normalized in STAGE_KEY_TO_FLOAT:
        return STAGE_KEY_TO_FLOAT[normalized]
    raise argparse.ArgumentTypeError(
        f"Invalid stage value {value!r}. Valid values: 0, 1, 2, 3, 4, 5"
    )


def find_image(scene_dir: str) -> Optional[str]:
    """Return the first image file matching 'image.{png,jpg,jpeg}' (case-insensitive)."""
    for name in sorted(os.listdir(scene_dir)):
        lower = name.lower()
        if lower == "image.png" or lower == "image.jpg" or lower == "image.jpeg":
            return name
    # Fallback: any image.* with an allowed extension
    for name in sorted(os.listdir(scene_dir)):
        if name.lower().startswith("image.") and name.lower().endswith(IMAGE_EXTS):
            return name
    return None


def has_file(scene_dir: str, name: str) -> bool:
    return os.path.isfile(os.path.join(scene_dir, name))


def has_any_glob(scene_dir: str, pattern: str) -> bool:
    return len(glob.glob(os.path.join(scene_dir, pattern))) > 0


# Mapping from well-known artifact names to their canonical subfolder.
# "" means top-level (canonical); a subfolder string means that subfolder is canonical.
# The inspector checks the canonical location first, then falls back to legacy
# locations with a [legacy-path] warning.
_CANONICAL_SUBFOLDER: Dict[str, str] = {
    "stage2_plan.json":           "json",
    "polygon_v2.json":            "json",
    "alignment_metrics_v2.json":  "json",
    # blender_scene.json is canonical at json/; top-level is legacy.
    "blender_scene.json":         "json",
    # blender_scene.blend is canonical at blend/; top-level is legacy.
    "blender_scene.blend":        "blend",
    "blender_scene_env_preview.png": "render",
    "blender_scene_view_perspective.png": "render",
    "layout_prediction.json":     "inputs",
    "layout-prediction.glb":      "inputs",
    "mask_attribute.json":        "inputs",
    "pointmap_xz.ply":            "inputs",
}

# Canonical subfolder for glob-matched artifacts (matched by prefix/suffix).
_RENDER_GLOB_SUBFOLDER = "render"
_JSON_GLOB_SUBFOLDER   = "json"


def has_file_either(scene_dir: str, name: str) -> bool:
    """Check if file exists at its canonical location first; fall back to
    legacy locations (with a [legacy-path] print) if only found there.

    Canonical location is determined by _CANONICAL_SUBFOLDER:
      - "" (empty string) → top-level (e.g. blender_scene.blend)
      - a subfolder name  → that subfolder (e.g. "json", "render", "inputs")

    New-layout writes go directly to the canonical location; this function
    remains aware of both so that partially-migrated scenes still work.
    """
    canonical_sub = _CANONICAL_SUBFOLDER.get(name)
    if canonical_sub is not None:
        if canonical_sub:
            canonical_path = os.path.join(scene_dir, canonical_sub, name)
        else:
            canonical_path = os.path.join(scene_dir, name)  # top-level canonical
        if os.path.isfile(canonical_path):
            return True

    # Legacy: check all remaining locations (top-level + old pipeline subfolders).
    for sub in ("", "scene-pipeline", "json", "render", "inputs", "blend"):
        p = os.path.join(scene_dir, sub, name) if sub else os.path.join(scene_dir, name)
        if os.path.isfile(p):
            canonical_label = canonical_sub if canonical_sub else "top-level"
            if sub != (canonical_sub or ""):
                # File found at a non-canonical location — warn once.
                print(
                    f"[legacy-path] {name} found at {p!r}; "
                    f"expected under {canonical_label}",
                    file=sys.stderr, flush=True,
                )
            return True
    return False


def has_any_glob_either(scene_dir: str, pattern: str) -> bool:
    """Check if any file matching pattern exists at canonical location first,
    then fall back to legacy top-level and old pipeline subfolders. When the
    match is only found at a non-canonical location, emit a [legacy-path]
    warning to stderr (FILE_DIRECTORY.md "When reading" rule).
    """
    # Canonical subfolder for the most common patterns we care about.
    if pattern.endswith("_env_preview.png") or "view_" in pattern or "_overlay.png" in pattern:
        canonical_sub = "render"
    elif pattern.endswith(".json"):
        canonical_sub = "json"
    else:
        canonical_sub = ""
    canonical_base = os.path.join(scene_dir, canonical_sub) if canonical_sub else scene_dir
    if len(glob.glob(os.path.join(canonical_base, pattern))) > 0:
        return True
    # Fall back to legacy locations.
    for sub in ("render", "json", "inputs", "", "scene-pipeline"):
        if sub == canonical_sub:
            continue
        base = os.path.join(scene_dir, sub) if sub else scene_dir
        matches = glob.glob(os.path.join(base, pattern))
        if matches:
            label = canonical_sub if canonical_sub else "top-level"
            print(
                f"[legacy-path] {pattern} matched at {matches[0]!r}; "
                f"expected under {label}",
                file=sys.stderr, flush=True,
            )
            return True
    return False


def _stage2_plan_has_valid_meta(scene_dir: str) -> bool:
    """Return True iff json/stage2_plan.json exists, is valid JSON, and contains a
    _director_meta block whose image_sha256 matches the current image.png.

    A file without _director_meta (e.g. written by an old agent dispatch that
    bypassed run_stage2_director.py) is treated as NOT done so the director
    subprocess is re-invoked and injects the meta block.
    """
    import hashlib
    plan_path = os.path.join(scene_dir, "json", "stage2_plan.json")
    # Also accept legacy top-level location for the sha check; canonical is json/.
    if not os.path.isfile(plan_path):
        plan_path_top = os.path.join(scene_dir, "stage2_plan.json")
        if not os.path.isfile(plan_path_top):
            return False
        plan_path = plan_path_top
    try:
        with open(plan_path, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
        meta = plan.get("_director_meta")
        if not isinstance(meta, dict):
            return False
        cached_sha = meta.get("image_sha256")
        if not cached_sha:
            return False
        # Compute sha256 of current image.png
        image_path = os.path.join(scene_dir, "image.png")
        if not os.path.isfile(image_path):
            return False
        h = hashlib.sha256()
        with open(image_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest() == cached_sha
    except Exception:
        return False


def _json_has_stage_block(path: str) -> bool:
    """Return True iff the JSON at `path` contains a top-level 'stage' key."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return isinstance(data.get("stage"), dict)
    except Exception:
        return False


def _json_has_env_blocks(path: str) -> bool:
    """Return True iff the JSON at `path` contains all four env blocks:
    'lighting', 'world', 'stage_materials', 'render'.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return (
            isinstance(data.get("lighting"), list)
            and isinstance(data.get("world"), dict)
            and isinstance(data.get("stage_materials"), dict)
            and isinstance(data.get("render"), dict)
        )
    except Exception:
        return False


def inspect(
    scene_dir: str,
    force: str,
    force_from: Optional[float],
    until: Optional[float],
) -> dict:
    if not os.path.isdir(scene_dir):
        return {
            "scene_dir": scene_dir,
            "error": f"scene_dir does not exist or is not a directory: {scene_dir}",
        }

    image_file = find_image(scene_dir)

    # blender_scene.json canonical location is json/.
    # Fall back to top-level for legacy scenes with a [legacy-path] warning.
    _json_sub = os.path.join(scene_dir, "json", "blender_scene.json")
    _json_top = os.path.join(scene_dir, "blender_scene.json")
    if os.path.isfile(_json_sub):
        json_path = _json_sub
    elif os.path.isfile(_json_top):
        print(
            f"[legacy-path] reading {_json_top}; canonical is {_json_sub}",
            file=sys.stderr, flush=True,
        )
        json_path = _json_top
    else:
        json_path = _json_sub  # let downstream fail with clearer error

    def _check_with_legacy_warning(
        canonical: str, legacy: str, name: str, *, is_dir: bool = False
    ) -> bool:
        """Probe canonical path first; fall back to legacy with [legacy-path] stderr warning.
        Returns True if the artifact exists at either location.
        """
        check = os.path.isdir if is_dir else os.path.isfile
        if check(canonical):
            return True
        if check(legacy):
            print(
                f"[legacy-path] reading {legacy}; canonical is {canonical}",
                file=sys.stderr, flush=True,
            )
            return True
        return False

    # --- File-level detection ---
    has_stage2_plan_json = has_file_either(scene_dir, "stage2_plan.json")
    has_blender_scene_json = os.path.isfile(json_path)
    # blender_scene.blend canonical location is blend/; top-level is legacy.
    has_blender_scene_blend = _check_with_legacy_warning(
        os.path.join(scene_dir, "blend", "blender_scene.blend"),
        os.path.join(scene_dir, "blender_scene.blend"),
        "blender_scene.blend",
    )
    has_legacy_aligned_scene_blend = _check_with_legacy_warning(
        os.path.join(scene_dir, "blend", "aligned_scene.blend"),
        os.path.join(scene_dir, "aligned_scene.blend"),
        "aligned_scene.blend",
    )
    # Stage-3 output files: check canonical subfolder first, then legacy locations.
    has_polygon_v2_json = has_file_either(scene_dir, "polygon_v2.json")
    has_alignment_metrics_v2_json = has_file_either(scene_dir, "alignment_metrics_v2.json")
    has_env_preview_png = has_any_glob_either(scene_dir, "*_env_preview.png")
    # pointmap_xz.ply — canonical location is inputs/; fall back to top-level.
    has_pointmap_ply = _check_with_legacy_warning(
        os.path.join(scene_dir, "inputs", "pointmap_xz.ply"),
        os.path.join(scene_dir, "pointmap_xz.ply"),
        "pointmap_xz.ply",
    )
    # Stage 5 done-signal: primary multi-view PNG in render/ (canonical) or top-level.
    has_multiview_perspective = has_file_either(scene_dir, "blender_scene_view_perspective.png")
    # layout_prediction.json — canonical: inputs/; fall back to top-level.
    has_layout_prediction_json = _check_with_legacy_warning(
        os.path.join(scene_dir, "inputs", "layout_prediction.json"),
        os.path.join(scene_dir, "layout_prediction.json"),
        "layout_prediction.json",
    )
    # object/ — canonical: inputs/object/; fall back to top-level object/.
    has_object_dir = _check_with_legacy_warning(
        os.path.join(scene_dir, "inputs", "object"),
        os.path.join(scene_dir, "object"),
        "object/",
        is_dir=True,
    )
    # mask_attribute.json — canonical: inputs/; fall back to top-level.
    has_mask_attribute_json = _check_with_legacy_warning(
        os.path.join(scene_dir, "inputs", "mask_attribute.json"),
        os.path.join(scene_dir, "mask_attribute.json"),
        "mask_attribute.json",
    )
    # masks/ directory — canonical: inputs/masks/; fall back to top-level masks/.
    has_masks_dir = _check_with_legacy_warning(
        os.path.join(scene_dir, "inputs", "masks"),
        os.path.join(scene_dir, "masks"),
        "masks/",
        is_dir=True,
    )

    # --- Content-level detection (lazy — only read if JSON exists) ---
    has_stage_block = (
        _json_has_stage_block(json_path) if has_blender_scene_json else False
    )
    has_env_blocks = (
        _json_has_env_blocks(json_path) if has_blender_scene_json else False
    )

    # "Any .blend at all" — convenient for S4's gate.
    any_blend = has_blender_scene_blend or has_legacy_aligned_scene_blend

    detected: Dict[str, Any] = {
        "stage2_plan_json":       has_stage2_plan_json,
        "layout_prediction_json": has_layout_prediction_json,
        "image": image_file,
        "pointmap_ply": has_pointmap_ply,
        "mask_attribute_json": has_mask_attribute_json,
        "object_dir": has_object_dir,
        "masks_dir": has_masks_dir,
        "blender_scene_json": has_blender_scene_json,
        # Stage 2 outputs: builder writes blender_scene.blend at blend/ (canonical)
        "blender_scene_blend": has_blender_scene_blend,
        "has_legacy_aligned_scene_blend": has_legacy_aligned_scene_blend,
        # Stage 3 outputs
        "polygon_v2_json": has_polygon_v2_json,
        "alignment_metrics_v2_json": has_alignment_metrics_v2_json,
        "blender_scene_json_has_stage_block": has_stage_block,
        # Stage 4 outputs
        "env_preview_png": has_env_preview_png,
        "blender_scene_json_has_env_blocks": has_env_blocks,
        # Stage 5 outputs
        "multiview_perspective_png": has_multiview_perspective,
    }

    # ---------------------------------------------------------------------------
    # Stage done-signals
    # ---------------------------------------------------------------------------
    # Stage 2 done = blender_scene.blend exists at blend/ (canonical).
    # If only aligned_scene.blend exists (legacy), Stage 2 is treated as "ready"
    # (the new builder will write blender_scene.blend at blend/ on rerun).
    stage2_done = has_blender_scene_blend

    # Stage 3 done = polygon_v2.json + stage block in JSON.
    # alignment_metrics_v2.json is reported in `detected` for diagnostics but
    # the current stage2-sub-pointmap-to-separable-stage skill does not write it; it was
    # required by the v1 fitter only.
    stage3_done = has_polygon_v2_json and has_stage_block

    # Stage 4 done = any *_env_preview.png AND all four env blocks in JSON
    stage4_done = has_env_preview_png and has_env_blocks

    # Stage 5 done = blender_scene_view_perspective.png exists
    stage5_done = has_multiview_perspective

    # Stage 0 done = stage2_plan.json exists AND has a valid _director_meta block
    # with image_sha256 matching current image.png. A file without the meta block
    # (written by a legacy Agent dispatch) counts as NOT done so run_stage2_director.py
    # is invoked and injects the meta.
    stage0_done = _stage2_plan_has_valid_meta(scene_dir)

    done: Dict[float, bool] = {
        0.0: stage0_done,
        1.0: detected["blender_scene_json"],
        2.0: stage2_done,
        3.0: stage3_done,
        4.0: stage4_done,
        5.0: stage5_done,
    }

    # ---------------------------------------------------------------------------
    # Ready requirements (inputs must exist; independent of prior-stage done status)
    # ---------------------------------------------------------------------------
    def ready_reason(stage: float) -> Optional[str]:
        """Return a string describing why the stage cannot start, or None if ready."""
        if stage == 0.0:
            # Stage 0 (director) is advisory — it can run as soon as we have an
            # image. A scene with no image has bigger problems; stages 1/3/4
            # already report that as a blocker.
            if image_file is None:
                return "missing image.{png,jpg,jpeg}"
            return None

        if stage == 1.0:
            if not detected["layout_prediction_json"]:
                return "missing layout_prediction.json"
            if image_file is None:
                return "missing image.{png,jpg,jpeg}"
            return None

        if stage == 2.0:
            if not detected["blender_scene_json"]:
                return "missing blender_scene.json (run stage 1 first)"
            return None

        if stage == 3.0:
            if image_file is None:
                return "missing image.{png,jpg,jpeg}"
            if not has_pointmap_ply:
                return "missing pointmap_xz.ply"
            return None

        if stage == 4.0:
            if image_file is None:
                return "missing image.{png,jpg,jpeg}"
            if not any_blend:
                return "no .blend in scene_dir (run stages 1-2 at minimum)"
            return None

        if stage == 5.0:
            if not stage4_done:
                return (
                    "stage 4 has not completed "
                    "(*_env_preview.png or env blocks in blender_scene.json missing)"
                )
            if not has_blender_scene_blend:
                return "blender_scene.blend missing (required for Blender render)"
            return None

        return f"unknown stage {stage}"

    stage_names: Dict[float, str] = {
        0.0: "run_stage2_director.py",  # subprocess wrapper; orchestrator must NOT use Agent tool for Stage 0
        1.0: "stage2-sub-pointmap-to-separable-stage",
        2.0: "stage2-sub-pointmap-to-separable-stage",
        3.0: "stage2-sub-pointmap-to-separable-stage",
        4.0: "stage2-sub-env-enhance",
        5.0: "multi-view-render",
    }
    # String keys used in the plan JSON (human-readable)
    stage_display: Dict[float, str] = {
        0.0: "0", 1.0: "1", 2.0: "2", 3.0: "3", 4.0: "4", 5.0: "5",
    }

    force_all = force == "all"
    effective_force_from: float = force_from if force_from is not None else 999.0
    effective_until: float = until if until is not None else 5.0

    plan = []
    for s in STAGE_ORDER:
        if s > effective_until:
            plan.append({
                "stage": stage_display[s],
                "name": stage_names[s],
                "status": "skipped_by_until",
                "reason": f"--until={stage_display[effective_until]} excludes this stage",
            })
            continue

        reason = ready_reason(s)
        # Forcing: treat "done" as ignorable when --force all or --force-from N applies.
        treat_done_as_done = done[s] and not force_all and s < effective_force_from

        if reason is None and treat_done_as_done:
            plan.append({
                "stage": stage_display[s],
                "name": stage_names[s],
                "status": "done",
                "reason": "output signal(s) already present in scene_dir",
            })
        elif reason is None:
            plan.append({
                "stage": stage_display[s],
                "name": stage_names[s],
                "status": "ready",
                "reason": "inputs available; will dispatch",
            })
        else:
            # Stage is blocked. If it happens to be "done" from an earlier run
            # and nothing forced re-plan, still call it done — a blocked-but-done
            # state is possible if inputs were deleted after the run.
            if treat_done_as_done:
                plan.append({
                    "stage": stage_display[s],
                    "name": stage_names[s],
                    "status": "done",
                    "reason": (
                        f"output signal(s) present; inputs missing but stage already "
                        f"complete ({reason})"
                    ),
                })
            else:
                plan.append({
                    "stage": stage_display[s],
                    "name": stage_names[s],
                    "status": "blocked",
                    "reason": reason,
                })

    # ---------------------------------------------------------------------------
    # Warnings
    # ---------------------------------------------------------------------------
    warnings = []
    if image_file is None:
        warnings.append(
            "no image.{png,jpg,jpeg} in scene_dir — stages 1, 3, 4 cannot run"
        )
    if not has_pointmap_ply:
        warnings.append("no pointmap_xz.ply — stage 3 cannot run")
    if not detected["layout_prediction_json"] and not detected["blender_scene_json"]:
        warnings.append(
            "no layout_prediction.json and no blender_scene.json — stages 1 and 2 cannot run"
        )
    # Legacy aligned_scene.blend found but canonical blender_scene.blend absent
    if has_legacy_aligned_scene_blend and not has_blender_scene_blend:
        warnings.append(
            "aligned_scene.blend found at legacy path but blender_scene.blend is absent at blend/ — "
            "stage 2 will rerun and write blender_scene.blend to blend/ (canonical path)"
        )

    counts: Dict[str, int] = {
        "done": 0, "ready": 0, "blocked": 0, "skipped_by_until": 0,
    }
    for p in plan:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v > 0)

    return {
        "scene_dir": os.path.abspath(scene_dir),
        "detected": detected,
        "plan": plan,
        "warnings": warnings,
        "summary": summary,
        "forcing": {
            "force": force,
            "force_from": force_from,
            "until": until,
        },
    }


def _delete_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _delete_all_locations(scene_dir: str, name: str) -> None:
    """Delete a file from top-level AND all output subfolders (inputs/, json/, render/)."""
    _delete_if_exists(os.path.join(scene_dir, name))
    _delete_if_exists(os.path.join(scene_dir, "inputs", name))
    _delete_if_exists(os.path.join(scene_dir, "json", name))
    _delete_if_exists(os.path.join(scene_dir, "render", name))


# Keep the old name as an alias for any code still calling it.
_delete_both = _delete_all_locations


def apply_force_from(scene_dir: str, force_from: float) -> None:
    """Delete done-signal files for stages >= force_from so they replan as ready.

    Note: this function is a helper for orchestrators that want to physically
    remove done signals before calling inspect(). The inspect() function
    itself never mutates the filesystem; it only reads.
    """
    if force_from <= 0.0:
        _delete_both(scene_dir, "stage2_plan.json")
    if force_from <= 2.0:
        _delete_if_exists(os.path.join(scene_dir, "blend", "blender_scene.blend"))  # canonical subfolder
        _delete_if_exists(os.path.join(scene_dir, "blender_scene.blend"))  # legacy top-level
        _delete_if_exists(os.path.join(scene_dir, "blend", "aligned_scene.blend"))  # old-name legacy
    if force_from <= 3.0:
        _delete_both(scene_dir, "polygon_v2.json")
        _delete_both(scene_dir, "alignment_metrics_v2.json")
        # stage block is inside blender_scene.json — cannot delete in isolation;
        # the orchestrator must rerun stage 3 which overwrites the JSON.
    if force_from <= 4.0:
        # env_preview PNGs at top level AND in render/ (post-finalize location).
        for f in glob.glob(os.path.join(scene_dir, "*_env_preview.png")):
            _delete_if_exists(f)
        for f in glob.glob(os.path.join(scene_dir, "render", "*_env_preview.png")):
            _delete_if_exists(f)
        # env blocks live in blender_scene.json — cannot delete in isolation;
        # stage 4 overwrites the whole JSON env section when it runs.
    if force_from <= 5.0:
        # Stage 5 done-signal: all five multi-view PNGs at top level AND in render/.
        # New names: perspective, bev, wide, topcorner, topcorner_opposite.
        # Old names (corner, side) are also deleted for backward compatibility
        # so a --force-from 5 on a previously-run scene cleans up legacy files.
        for name in (
            "blender_scene_view_perspective.png",
            "blender_scene_view_bev.png",
            "blender_scene_view_wide.png",
            "blender_scene_view_topcorner.png",
            "blender_scene_view_topcorner_opposite.png",
            # Legacy filenames (pre-rename) — delete if present
            "aligned_scene_view_perspective.png",
            "aligned_scene_view_bev.png",
            "aligned_scene_view_wide.png",
            "aligned_scene_view_topcorner.png",
            "aligned_scene_view_topcorner_opposite.png",
            "aligned_scene_view_corner.png",
            "aligned_scene_view_side.png",
        ):
            _delete_if_exists(os.path.join(scene_dir, name))
            _delete_if_exists(os.path.join(scene_dir, "render", name))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene_dir", help="Path to scene folder")
    ap.add_argument(
        "--force",
        choices=("off", "all"),
        default="off",
        help="off (default): honor done-signals. all: mark every stage as ready if inputs allow.",
    )
    ap.add_argument(
        "--force-from",
        type=_parse_stage_arg,
        default=None,
        metavar="N",
        help=(
            "Re-plan stages >= N (their done-signals ignored). "
            "Valid values: 1, 2, 3, 4, 5."
        ),
    )
    ap.add_argument(
        "--until",
        type=_parse_stage_arg,
        default=None,
        metavar="N",
        help=(
            "Stages > N are skipped_by_until. "
            "Valid values: 1, 2, 3, 4, 5."
        ),
    )
    args = ap.parse_args()

    result = inspect(args.scene_dir, args.force, args.force_from, args.until)
    print(json.dumps(result, indent=2))
    return 0 if "error" not in result else 2


if __name__ == "__main__":
    sys.exit(main())
