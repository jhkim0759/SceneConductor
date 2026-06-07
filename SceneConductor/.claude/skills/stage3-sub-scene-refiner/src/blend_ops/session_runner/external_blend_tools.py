"""Python tool wrappers around external_blend_runner.py.

Each function spawns a Blender subprocess, applies one op, returns the result
dict. Suitable for an LLM agent to invoke as tools.

Configuration:
- BLENDER binary: env var SCENE_EVAL_BLENDER, otherwise default path below.
- RUNNER script: resolved relative to this file (sibling external_blend_runner.py).
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

# ── DIRECTORYS.yaml (canonical machine-specific paths) ──────────────────────
# This file lives at <repo>/.claude/skills/scene-refiner/src/blend_ops/session_runner/,
# so the repo root is 6 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[6]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())

_DEFAULT_BLENDER = os.environ.get("BLENDER", _DIRS["blender_bin"])
BLENDER = os.environ.get("SCENE_EVAL_BLENDER", _DEFAULT_BLENDER)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(_SCRIPT_DIR, "external_blend_runner.py")
MULTI_VIEW_SCRIPT = os.path.join(_SCRIPT_DIR, "render_multi_view.py")


def _run(blend_in, blend_out, op, timeout=180):
    fd, op_path = tempfile.mkstemp(prefix="blendop_", suffix=".json")
    os.close(fd)
    with open(op_path, "w") as f:
        json.dump(op, f)
    result_path = op_path + ".result"
    try:
        proc = subprocess.run(
            [BLENDER, "--background", blend_in, "--python", RUNNER, "--", op_path, blend_out],
            capture_output=True, text=True, timeout=timeout,
        )
        if not os.path.exists(result_path):
            return {
                "success": False,
                "message": f"no result file. blender exit={proc.returncode}. stderr tail: {proc.stderr[-500:]}",
            }
        with open(result_path) as f:
            return json.load(f)
    finally:
        for p in (op_path, result_path):
            if os.path.exists(p):
                os.unlink(p)


def list_objects(blend_in, name_prefix="obj_"):
    return _run(blend_in, "/tmp/__noop.blend", {"action": "list_objects", "name_prefix": name_prefix})


def inspect_object(blend_in, obj_name):
    return _run(blend_in, "/tmp/__noop.blend", {"action": "inspect_object", "obj_name": obj_name})


def update_layout(blend_in, blend_out, obj_name, location):
    return _run(blend_in, blend_out, {"action": "update_layout", "obj_name": obj_name, "location": list(location)})


def update_rotation(blend_in, blend_out, obj_name, rotation_euler):
    return _run(blend_in, blend_out, {"action": "update_rotation", "obj_name": obj_name, "rotation_euler": list(rotation_euler)})


def flip_yaw_180(blend_in, blend_out, obj_name):
    return _run(blend_in, blend_out, {"action": "flip_yaw_180", "obj_name": obj_name})


def update_size(blend_in, blend_out, obj_name, scale):
    return _run(blend_in, blend_out, {"action": "update_size", "obj_name": obj_name, "scale": list(scale)})


def remove_object(blend_in, blend_out, obj_name):
    return _run(blend_in, blend_out, {"action": "remove_object", "obj_name": obj_name})


def delete_object(blend_in, blend_out, obj_name):
    return _run(blend_in, blend_out, {"action": "delete_object", "obj_name": obj_name})


def metrics(blend_in, name_prefix="obj_", room_bbox=None):
    op = {"action": "metrics", "name_prefix": name_prefix}
    if room_bbox is not None:
        op["room_bbox"] = room_bbox
    return _run(blend_in, "/tmp/__noop.blend", op)


def render(blend_in, output_png, view="top", resolution=(800, 600)):
    return _run(
        blend_in,
        "/tmp/__noop.blend",
        {"action": "render", "output_png": output_png, "view": view, "resolution": list(resolution)},
        timeout=300,
    )


def render_multi_view(blend_in, scene_dir, output_dir, samples=64,
                      resolution=(1024, 768), brightness_log=None, timeout=600,
                      engine=None):
    """5-view render via the bundled render_multi_view.py.

    Produces (in `output_dir`):
      blender_scene_view_perspective.png    — reference vantage from layout_prediction.json
      blender_scene_view_bev.png            — top-down ortho with overhead Area light
      blender_scene_view_wide.png           — same vantage as scene Camera, 20mm lens
      blender_scene_view_topcorner.png      — far-corner 3/4 (anchored to polygon vertex)
      blender_scene_view_topcorner_opposite.png — second-best far-corner 3/4

    Requires `scene_dir` to contain (any of these is fine — script falls back gracefully):
      json/blender_scene.json     for polygon_vertices + wall_objects (blocking-wall hide)
      inputs/layout_prediction.json   for camera rotation + focal length
      brightness_align_log.json   optional, for neutral lighting scaling
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        BLENDER, "--background", blend_in,
        "--python", MULTI_VIEW_SCRIPT, "--",
        "--scene-dir", str(scene_dir),
        "--output-dir", str(output_dir),
        "--samples", str(samples),
        "--resolution-x", str(resolution[0]),
        "--resolution-y", str(resolution[1]),
    ]
    if engine is not None:
        cmd += ["--engine", str(engine)]
    if brightness_log is not None:
        cmd += ["--brightness-log", str(brightness_log)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    expected = [
        os.path.join(output_dir, name) for name in (
            "blender_scene_view_perspective.png",
            "blender_scene_view_bev.png",
            "blender_scene_view_wide.png",
            "blender_scene_view_topcorner.png",
            "blender_scene_view_topcorner_opposite.png",
        )
    ]
    produced = [p for p in expected if os.path.exists(p)]
    return {
        "success": proc.returncode == 0 and len(produced) > 0,
        "exit_code": proc.returncode,
        "produced": produced,
        "expected": expected,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-500:],
    }


def attach(blend_in, blend_out, anchor_obj, moving_obj, relation="attached_to", n_samples=2000):
    """Chamfer-based attach: snap moving_obj to anchor_obj per relation.

    relation: "on" | "attached_to" | "next_to" | "+x"|"-x"|"+y"|"-y"|"+z"|"-z" | [vx,vy,vz]
    """
    return _run(blend_in, blend_out, {
        "action": "attach",
        "anchor_obj": anchor_obj, "moving_obj": moving_obj,
        "relation": relation, "n_samples": n_samples,
    })


def attach_to_wall(blend_in, blend_out, wall_obj, moving_obj, polygon_path=None,
                   clearance=0.02, t_along_m=None, preserve_rotation=False):
    """Polygon-aware wall attach: rotate moving to match wall tangent, then translate
    moving's XY to project onto the wall edge with proper inward offset.

    Z is preserved. rx/ry are preserved. By default rz is overridden to the wall tangent angle; pass preserve_rotation=True to keep the object's current rz (escape hatch, rarely needed).
    Requires polygon_v2.json (auto-discovered as <blend_dir>/../json/polygon_v2.json).
    """
    op = {"action": "attach_to_wall",
          "wall_obj": wall_obj, "moving_obj": moving_obj,
          "clearance": clearance,
          "preserve_rotation": preserve_rotation}
    if polygon_path is not None:
        op["polygon_path"] = polygon_path
    if t_along_m is not None:
        op["t_along_m"] = t_along_m
    return _run(blend_in, blend_out, op)


def apply_ops(blend_in, blend_out, ops, timeout=600):
    """Apply a list of ops in a single Blender session.

    `ops` may be a list of op dicts or a single op dict. All ops execute in
    one Blender startup; on first failure execution stops and the .blend is
    only saved if at least one mutating op succeeded.
    """
    return _run(blend_in, blend_out, ops, timeout=timeout)


TOOL_REGISTRY = {
    "update_layout": update_layout,
    "update_rotation": update_rotation,
    "flip_yaw_180": flip_yaw_180,
    "update_size": update_size,
    "remove_object": remove_object,
    "delete_object": delete_object,
    "attach": attach,
    "attach_to_wall": attach_to_wall,
}
