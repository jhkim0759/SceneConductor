"""
ply_import.py — Standalone PLY import helper for the stage2-sub-pointmap-to-separable-stage skill.

Public API:
    import_ply(path, *, name, axis_forward, axis_up, world_scale, visible)
        -> bpy.types.Object

Axis remap convention for PLY import into Blender Z-up space:
    bpy.ops.wm.ply_import(forward_axis=Z, up_axis=Y)   -- Blender 4.x
    bpy.ops.import_mesh.ply(axis_forward=Z, axis_up=Y) -- Blender 3.x fallback
"""

from __future__ import annotations

import os
from typing import Optional

import bpy  # type: ignore


def import_ply(
    path: str,
    *,
    name: str = "PointCloud_XZ",
    axis_forward: str = "Z",
    axis_up: str = "Y",
    world_scale: float = 1.0,
    visible: bool = False,
) -> bpy.types.Object:
    """Import a PLY file into the current Blender scene with axis remap and scale.

    Parameters
    ----------
    path:
        Absolute path to the PLY file.
    name:
        Blender object name assigned after import. Default ``"PointCloud_XZ"``.
    axis_forward:
        Blender import axis that maps to the source PLY forward axis.
        Default ``"Z"`` (matches the XZ-pointmap convention).
    axis_up:
        Blender import axis that maps to the source PLY up axis.
        Default ``"Y"``.
    world_scale:
        Uniform scale to apply after the axis remap bake.  When the PLY was
        exported with a world_scale_factor baked in, pass that factor here so
        the cloud aligns with other scene objects.  ``1.0`` means no scaling.
    visible:
        Whether the imported object is visible in the viewport and render.
        Default ``False`` (point clouds are hidden by convention).

    Returns
    -------
    bpy.types.Object
        The imported Blender object, renamed to *name* with transforms baked.
    """
    print(f"[ply_import] Importing PLY: {path} "
          f"(axis_forward={axis_forward}, axis_up={axis_up}, "
          f"world_scale={world_scale}, visible={visible})")

    before = set(bpy.data.objects)

    # Blender 4.x: wm.ply_import with forward_axis / up_axis keyword names.
    # Blender 3.x fallback: import_mesh.ply with axis_forward / axis_up.
    try:
        bpy.ops.wm.ply_import(
            filepath=path,
            forward_axis=axis_forward,
            up_axis=axis_up,
        )
    except TypeError:
        bpy.ops.import_mesh.ply(
            filepath=path,
            axis_forward=axis_forward,
            axis_up=axis_up,
        )

    new_objs = [o for o in bpy.data.objects if o not in before]
    if not new_objs:
        raise RuntimeError(f"[ply_import] PLY import produced no new object: {path}")

    obj = new_objs[0]
    obj.name = name

    # Bake any live rotation produced by the axis remap so mesh vertex
    # coordinates already encode the remapped positions.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    has_rotation = any(abs(a) > 1e-6 for a in obj.rotation_euler)
    has_scale    = any(abs(c - 1.0) > 1e-6 for c in obj.scale)
    if has_rotation or has_scale:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Apply world_scale so the cloud aligns with rescaled layout objects.
    if abs(world_scale - 1.0) > 1e-6:
        obj.scale = (world_scale, world_scale, world_scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.ops.object.select_all(action="DESELECT")

    # Vertex count for the summary log.
    n_verts = len(obj.data.vertices) if obj.type == "MESH" else 0
    print(f"[ply_import] Imported point cloud: {n_verts} verts from {path}")

    # Visibility — both viewport and render.
    obj.hide_viewport = not visible
    obj.hide_render   = not visible

    return obj


def parse_axis_remap(remap_str: str) -> tuple[str, str]:
    """Parse the ``axis_remap`` string from the ``point_cloud`` JSON block.

    Expected format: ``"forward=Z,up=Y"`` (case-insensitive, order-independent).
    Returns ``(axis_forward, axis_up)`` as upper-case strings.
    Defaults: forward=Z, up=Y if a token is absent or malformed.
    """
    forward = "Z"
    up = "Y"
    for token in remap_str.replace(" ", "").split(","):
        if "=" not in token:
            continue
        key, val = token.split("=", 1)
        key = key.strip().lower()
        val = val.strip().upper()
        if key == "forward":
            forward = val
        elif key == "up":
            up = val
    return forward, up
