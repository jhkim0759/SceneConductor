"""Python tool wrappers around external_blend_runner.py.

Each function spawns a Blender subprocess, applies one op, returns the result
dict. Suitable for an LLM agent to invoke as tools.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

# ── DIRECTORYS.yaml (canonical machine-specific paths) ──────────────────────
# This file lives at <repo>/.claude/skills/scene-refiner/src/blend_ops/core/,
# so the repo root is 6 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[6]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())

BLENDER = os.environ.get(
    "SCENE_EVAL_BLENDER",
    os.environ.get("BLENDER", _DIRS["blender_bin"]),
)
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_blend_runner.py")


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
