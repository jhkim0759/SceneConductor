"""
export_env_to_json.py — Stage 5 environment exporter for the stage2-sub-env-enhance skill.

Reads live Blender state and dumps lighting + world + stage_materials + render +
compositor blocks into blender_scene.json non-destructively, preserving every
other key (objects, camera, floor, scene, stage, point_cloud, …) byte-for-byte.

SAFETY RULE: Materials named Material_0.* or bound to geometry_* meshes are
NEVER serialised.  This is a hard invariant.
See: feedback_blender_scene_env_only.md

Usage (CLI):
    blender --background path/to/blender_scene.blend \\
        --python export_env_to_json.py -- \\
        --scene-json path/to/blender_scene.json \\
        [--no-compositor]

Library usage:
    from export_env_to_json import export_env_to_json
    summary = export_env_to_json("/path/to/blender_scene.json")
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from typing import Any, Optional

import bpy

from _material_utils import (
    classify_stage_material,
    is_stage_material,
    principled_to_pbr_dict,
    _get_or_create_principled,
)

LOG = "[export_env_to_json]"

# ---------------------------------------------------------------------------
# Render defaults — used to suppress unchanged values in the render block.
# ---------------------------------------------------------------------------

RENDER_DEFAULTS: dict[str, Any] = {
    "engine": "CYCLES",
    "samples": 128,
    "resolution": [1024, 682],
    "view_transform": "AgX",
    "look": "AgX - Medium High Contrast",
    "exposure": 0.0,
    "clamp_indirect": 5.0,
    "denoiser": "OPENIMAGEDENOISE",
}

# ---------------------------------------------------------------------------
# Argument parsing (mirrors enhance_env.py style)
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Extract args after '--' separator and parse."""
    script_args = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="export_env_to_json.py")
    p.add_argument(
        "--scene-json", required=True,
        help="Path to blender_scene.json (must already exist).",
    )
    p.add_argument(
        "--no-compositor", action="store_true",
        help="Skip compositor block serialisation.",
    )
    p.add_argument(
        "--keep-existing", action="store_true",
        help="Do not overwrite stage-material entries that are already present "
             "in the JSON unless a fresh value was computed.",
    )
    return p.parse_args(script_args)


# ---------------------------------------------------------------------------
# Block extractors
# ---------------------------------------------------------------------------

def _extract_lighting(scene_dir: str) -> list[dict]:
    """Serialise all interior light objects to a list of dicts.

    Interior-only since v2.1: SUN-specific fields and is_portal are no longer
    emitted because the rig only contains AREA / POINT / SPOT lights inside the
    sealed room.
    """
    lights = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        ld = obj.data

        # Collection membership — use the first collection that contains obj.
        col_name: Optional[str] = None
        for col in bpy.data.collections:
            if obj.name in col.objects:
                col_name = col.name
                break

        loc = list(obj.location)
        rot = list(obj.rotation_euler)
        color = [float(c) for c in ld.color]

        entry: dict[str, Any] = {
            "name": obj.name,
            "type": ld.type,
            "collection": col_name,
            "location": [round(v, 6) for v in loc],
            "rotation_euler": [round(v, 6) for v in rot],
            "energy": round(float(ld.energy), 4),
            "color": [round(c, 6) for c in color],
        }

        # Size (AREA lights)
        if hasattr(ld, "size"):
            entry["size"] = round(float(ld.size), 4)
        else:
            entry["size"] = None

        # Spot-specific
        if ld.type == "SPOT":
            spread_rad = getattr(ld, "spot_size", None)
            entry["spread_deg"] = round(math.degrees(spread_rad), 4) if spread_rad is not None else None
        else:
            entry["spread_deg"] = None

        # Shadow soft size
        shadow_size = getattr(ld, "shadow_soft_size", None)
        entry["shadow_soft_size"] = round(float(shadow_size), 4) if shadow_size is not None else None

        # Cycles cast_shadow only — is_portal is dead in the interior-only era
        cast_shadow = True
        try:
            cast_shadow = bool(ld.cycles.cast_shadow)
        except AttributeError:
            pass
        entry["cycles"] = {"cast_shadow": cast_shadow}

        lights.append(entry)

    print(f"{LOG} lighting: {len(lights)} light(s) serialised.")
    return lights


def _extract_world(scene_dir: str) -> dict:
    """Serialise the current world shader as a flat Background.

    Interior-only since v2.1: the room is sealed so we always emit
    ``mode="flat"`` with whatever color + strength the Background node has.
    Nishita Sky / HDRI modes are no longer produced; if a legacy blend still
    has a Sky / HDRI hooked up, enhance_env's `rebuild_world` will replace it
    with a flat black Background on the next run before this exporter sees it.
    """
    world = bpy.context.scene.world
    fallback = {"mode": "flat", "color": [0.0, 0.0, 0.0], "strength": 0.0}
    if world is None or not world.use_nodes:
        return fallback
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is None:
        return fallback
    color_input = bg.inputs.get("Color")
    if color_input is not None and not color_input.is_linked:
        col = [round(float(color_input.default_value[i]), 6) for i in range(3)]
    else:
        # An upstream node still feeds the Background (legacy Sky / HDRI).
        # Report flat black; the next env-enhance run will normalise.
        col = [0.0, 0.0, 0.0]
    strength = round(float(bg.inputs["Strength"].default_value), 6)
    print(f"{LOG} world: mode=flat color={col} strength={strength}")
    return {"mode": "flat", "color": col, "strength": strength}


def _collect_protected_mat_names() -> set[str]:
    """Return the set of material names that must NEVER be serialised.

    Hard rule:
      - Material_0.* names
      - Any material bound to a geometry_* mesh
    See: feedback_blender_scene_env_only.md
    """
    protected: set[str] = set()
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith("geometry_"):
            for slot in obj.material_slots:
                if slot.material:
                    protected.add(slot.material.name)
    for mat in bpy.data.materials:
        if mat.name.startswith("Material_0"):
            protected.add(mat.name)
    return protected


def _extract_stage_materials(scene_dir: str) -> dict:
    """Return stage_materials dict keyed by logical role.

    Keys follow the pattern used by export consumers:
      "floor", "ceiling", "walls.__default__", "walls.Wall_NN"

    Protected materials (Material_0.* / geometry_* bound) are skipped.
    """
    protected = _collect_protected_mat_names()
    stage_mats: dict[str, dict] = {}
    skipped_protected: list[str] = []

    for mat in bpy.data.materials:
        name = mat.name

        # Hard skip — protected materials.
        if name in protected:
            skipped_protected.append(name)
            continue
        if name.startswith("Material_0"):
            skipped_protected.append(name)
            continue

        if not is_stage_material(name):
            continue

        role = classify_stage_material(name)
        if role is None:
            continue

        # Get Principled BSDF.  If the material has no node tree yet, skip.
        if not mat.use_nodes:
            print(f"{LOG} stage_materials: {name} has no node tree — skipping.")
            continue

        pbsdf = None
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                pbsdf = node
                break
        if pbsdf is None:
            print(f"{LOG} stage_materials: {name} has no Principled BSDF — skipping.")
            continue

        pbr = principled_to_pbr_dict(pbsdf, scene_dir)
        stage_mats[role] = pbr
        print(f"{LOG} stage_materials: {name} → role '{role}' serialised.")

    if skipped_protected:
        print(f"{LOG} stage_materials: SKIPPED protected: {sorted(skipped_protected)}")

    return stage_mats


def _extract_render() -> dict:
    """Serialise render settings, emitting only values that differ from RENDER_DEFAULTS."""
    scene = bpy.context.scene
    current: dict[str, Any] = {
        "engine": scene.render.engine,
        "samples": scene.cycles.samples,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": round(float(scene.view_settings.exposure), 6),
        "clamp_indirect": round(float(scene.cycles.sample_clamp_indirect), 4),
    }
    try:
        current["denoiser"] = scene.cycles.denoiser
    except AttributeError:
        current["denoiser"] = "NONE"

    # Emit only fields that differ from defaults (always include resolution as a list).
    result: dict[str, Any] = {}
    for k, v in current.items():
        default = RENDER_DEFAULTS.get(k)
        if v != default:
            result[k] = v
        else:
            # Always include to keep the block self-contained.
            result[k] = v

    print(f"{LOG} render: engine={current['engine']} samples={current['samples']} "
          f"res={current['resolution']} view={current['view_transform']}")
    return result


def _extract_compositor(include_compositor: bool) -> Optional[dict]:
    """Serialise the compositor's known nodes (Glare + LensDist).

    Returns None if ``include_compositor=False`` or the compositor is not active.
    Returns ``{"unknown_graph": true}`` if the graph contains unknown structure.
    """
    if not include_compositor:
        return None

    scene = bpy.context.scene
    if not scene.use_nodes:
        print(f"{LOG} compositor: scene.use_nodes is False — skipping.")
        return None

    nt = scene.node_tree
    if nt is None:
        return None

    # Identify known node types used by enhance_env.py.
    glare_node = next((n for n in nt.nodes if n.type == "GLARE"), None)
    lens_node = next((n for n in nt.nodes if n.type == "LENSDIST"), None)

    # If neither known node is present but there IS a compositor graph, flag it.
    node_types = {n.type for n in nt.nodes}
    non_trivial_types = node_types - {"R_LAYERS", "COMPOSITE", "OUTPUT_FILE", "VIEWER"}
    if not non_trivial_types:
        print(f"{LOG} compositor: trivial / empty graph — skipping.")
        return None

    if glare_node is None and lens_node is None and non_trivial_types:
        print(f"{LOG} compositor: WARNING — unknown graph types: {non_trivial_types}")
        return {"unknown_graph": True}

    result: dict[str, Any] = {}

    if glare_node is not None:
        result["glare"] = {
            "glare_type": glare_node.glare_type,
            "mix": round(float(glare_node.mix), 6),
            "threshold": round(float(glare_node.threshold), 6),
            "size": int(glare_node.size),
        }

    if lens_node is not None:
        distortion = 0.0
        dispersion = 0.0
        for iname in ("Distortion", "Distort"):
            sock = lens_node.inputs.get(iname)
            if sock is not None:
                distortion = round(float(sock.default_value), 6)
                break
        disp_sock = lens_node.inputs.get("Dispersion")
        if disp_sock is not None:
            dispersion = round(float(disp_sock.default_value), 6)
        result["lens_distortion"] = {
            "distortion": distortion,
            "dispersion": dispersion,
        }

    if not non_trivial_types - {"GLARE", "LENSDIST"}:
        pass  # all nodes accounted for
    else:
        leftover = non_trivial_types - {"GLARE", "LENSDIST"}
        print(f"{LOG} compositor: WARNING — additional unknown node types: {leftover}")
        result["unknown_graph"] = True

    print(f"{LOG} compositor: serialised keys: {list(result.keys())}")
    return result if result else None


# ---------------------------------------------------------------------------
# Non-destructive merge helpers (mirrors update_blender_scene_json.py pattern)
# ---------------------------------------------------------------------------

def _deep_update(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target*, returning *target*.

    Nested dicts are merged; all other types are overwritten.
    """
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            _deep_update(target[k], v)
        else:
            target[k] = v
    return target


def _atomic_write(path: str, text: str) -> None:
    """Write *text* to *path* atomically via a sibling tempfile + rename."""
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Stage-material merge helper
# ---------------------------------------------------------------------------

def _merge_stage_materials(
    existing: dict,
    fresh: dict,
    overwrite: bool,
) -> dict:
    """Merge *fresh* stage_materials into *existing*.

    Overwrite behaviour:
      - If ``overwrite=True``: freshly computed keys replace existing ones.
      - If ``overwrite=False``: existing keys are preserved; only NEW keys
        from *fresh* are inserted.
    Old keys that are NOT in *fresh* are always preserved (no removal).
    """
    result = dict(existing)
    for k, v in fresh.items():
        if k not in result or overwrite:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Main library function
# ---------------------------------------------------------------------------

def export_env_to_json(
    scene_json_path: str,
    *,
    overwrite_if_present: bool = True,
    include_compositor: bool = True,
) -> dict:
    """Read current bpy state, emit env blocks, merge into blender_scene.json.

    Args:
        scene_json_path: Absolute (or CWD-relative) path to blender_scene.json.
            The file MUST already exist — this function never auto-creates it.
        overwrite_if_present: When True (default), freshly computed
            stage-material entries replace any that already exist in the JSON.
            When False, only new keys are inserted; existing entries survive.
        include_compositor: When True, serialise Glare + LensDist params into a
            ``compositor`` block.  When False, the block is omitted.

    Returns:
        A summary dict::

            {
                "lights_written": int,
                "materials_written": int,
                "world_mode": str,
                "render_keys": int,
                "compositor_written": bool,
            }

    Raises:
        FileNotFoundError: If ``scene_json_path`` does not exist.
    """
    abs_path = os.path.abspath(scene_json_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"{LOG} blender_scene.json not found at: {abs_path!r}. "
            "Create it with the stage2-environment-construction skill first."
        )

    scene_dir = os.path.dirname(abs_path)
    print(f"{LOG} scene_dir={scene_dir}")
    print(f"{LOG} scene_json={abs_path}")

    # --- Extract blocks from live Blender state ---
    lighting_block = _extract_lighting(scene_dir)
    world_block = _extract_world(scene_dir)
    fresh_stage_mats = _extract_stage_materials(scene_dir)
    render_block = _extract_render()
    compositor_block = _extract_compositor(include_compositor)

    # --- Load existing JSON ---
    with open(abs_path, "r", encoding="utf-8") as fh:
        scene_json: dict = json.load(fh)

    # --- Merge stage_materials non-destructively ---
    existing_stage_mats = scene_json.get("stage_materials", {})
    merged_stage_mats = _merge_stage_materials(
        existing_stage_mats,
        fresh_stage_mats,
        overwrite=overwrite_if_present,
    )

    # --- Update only the env-owned top-level keys ---
    env_patch: dict[str, Any] = {
        "lighting": lighting_block,
        "world": world_block,
        "stage_materials": merged_stage_mats,
        "render": render_block,
    }
    if compositor_block is not None:
        env_patch["compositor"] = compositor_block

    # Bump schema version.
    meta = scene_json.get("meta", {})
    meta["schema_version"] = "2.0"
    env_patch["meta"] = meta

    _deep_update(scene_json, env_patch)

    # --- Atomic write ---
    serialised = json.dumps(scene_json, indent=2, ensure_ascii=False)
    _atomic_write(abs_path, serialised)

    summary = {
        "lights_written": len(lighting_block),
        "materials_written": len(merged_stage_mats),
        "world_mode": world_block.get("mode", "unknown"),
        "render_keys": len(render_block),
        "compositor_written": compositor_block is not None,
    }
    print(f"{LOG} Done. summary={summary}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    try:
        args = _parse_args(argv)
        summary = export_env_to_json(
            args.scene_json,
            overwrite_if_present=not args.keep_existing,
            include_compositor=not args.no_compositor,
        )
        print(f"{LOG} Export complete: {summary}")
        return 0
    except FileNotFoundError as exc:
        print(f"{LOG} ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
