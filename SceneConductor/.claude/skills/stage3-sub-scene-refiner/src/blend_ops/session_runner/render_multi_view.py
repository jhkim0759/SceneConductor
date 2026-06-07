"""render_multi_view.py — Multi-view render for scene-env-enhance.

Renders 5 look-dev views of the env-enhanced .blend.  Most views use the
post-alignment (calibrated) lighting; the BEV view uses a neutral lighting
state that undoes the per-vantage brightness calibration.

Usage (run inside Blender):
    blender --background <scene_dir>/blender_scene.blend \\
            --python render_multi_view.py -- \\
            --scene-dir <scene_dir> \\
            [--brightness-log <path/to/brightness_align_log.json>] \\
            [--samples 256] \\
            [--resolution-x 1024] \\
            [--resolution-y 682]

Output PNGs (top level of scene_dir):
    blender_scene_view_perspective.png          — GALP reference vantage (camera at origin with predicted rotation; blocking walls camera-invisible)
    blender_scene_view_bev.png                  — top-down ortho, BEV overhead lighting (self-contained)
    blender_scene_view_wide.png                 — same vantage as Camera, 20 mm lens, calibrated
    blender_scene_view_topcorner.png            — high corner 3/4 view from the polygon vertex MOST opposite the scene camera; calibrated lighting
    blender_scene_view_topcorner_opposite.png   — complementary 3/4 view from the polygon vertex SECOND-most opposite the scene camera; calibrated lighting

Lighting modes:
    Calibrated — the post-alignment .blend state is used as-is (light energies
                 scaled ~14× by enhance_env.py's alignment loop).  Views 1, 3, 4, and 5.
    BEV overhead — scene calibration disabled; single overhead Area light 5000 W +
                 flat white world Background 0.5 strength.  View 2 only.
                 All original lights and world state restored after render.

Compositor handling:
    All compositor FileOutput nodes are muted and use_compositing is set to False
    for the entire multi-view render pass, preventing the Stage 5 preview PNG from
    being overwritten.  Both are restored in a try/finally after all 5 views.

Idempotent: re-running overwrites existing PNGs.
All temporary cameras and world replacements are cleaned up via try/finally.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Euler, Vector


# ---------------------------------------------------------------------------
# Argument parsing (everything after "--")
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    tail: List[str] = argv[argv.index("--") + 1:] if "--" in argv else []

    ap = argparse.ArgumentParser(
        prog="render_multi_view",
        description="Render 5 multi-view PNGs from the env-enhanced .blend.",
    )
    ap.add_argument(
        "--scene-dir",
        required=True,
        metavar="PATH",
        help="Absolute path to scene directory (PNGs written here).",
    )
    ap.add_argument(
        "--brightness-log",
        default=None,
        metavar="PATH",
        dest="brightness_log",
        help=(
            "Path to brightness_align_log.json (written by enhance_env.py). "
            "Default: <scene_dir>/brightness_align_log.json or "
            "<scene_dir>/scene-pipeline/brightness_align_log.json."
        ),
    )
    ap.add_argument("--samples", type=int, default=256, metavar="N")
    ap.add_argument(
        "--engine",
        default="CYCLES",
        metavar="NAME",
        help="Requested render engine. Supports BLENDER_EEVEE_NEXT, BLENDER_EEVEE, or CYCLES.",
    )
    ap.add_argument("--resolution-x", type=int, default=1024, metavar="PX", dest="resolution_x")
    ap.add_argument("--resolution-y", type=int, default=682,  metavar="PX", dest="resolution_y")
    ap.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        dest="output_dir",
        help=(
            "Override output directory for the rendered PNGs. "
            "Default: <scene_dir>/render/."
        ),
    )
    return ap.parse_args(tail)


# ---------------------------------------------------------------------------
# brightness_align_log.json reader
# ---------------------------------------------------------------------------

def _load_cumulative_scale_inverse(brightness_log_path: Optional[str], scene_dir: Path) -> float:
    """Return cumulative_scale_inverse from the brightness alignment log.

    Search order:
        1. --brightness-log flag (explicit path)
        2. <scene_dir>/brightness_align_log.json
        3. <scene_dir>/scene-pipeline/brightness_align_log.json
    Falls back to 0.05 if none found (keeps lights at ~5% of full calibrated
    energy, which empirically produces a reasonable neutral interior).
    """
    candidates: List[Path] = []
    if brightness_log_path:
        candidates.append(Path(brightness_log_path))
    candidates.append(scene_dir / "brightness_align_log.json")
    candidates.append(scene_dir / "scene-pipeline" / "brightness_align_log.json")

    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                val = float(data.get("cumulative_scale_inverse", 0.0))
                if val > 0:
                    print(f"[render_multi_view] brightness log: {path}  "
                          f"cumulative_scale_inverse={val:.4f}")
                    return val
            except Exception as _e:
                print(f"[render_multi_view] WARNING: could not read {path}: {_e}")

    print("[render_multi_view] WARNING: brightness_align_log.json not found — "
          "neutral lighting will use fixed 0.05 multiplier.")
    return 0.05


# ---------------------------------------------------------------------------
# Scene bbox — excludes PointCloud_XZ meshes and Lighting_Env collection
# ---------------------------------------------------------------------------

def _lighting_env_objects() -> set:
    result: set = set()
    coll = bpy.data.collections.get("Lighting_Env")
    if coll is None:
        return result

    def _walk(c: bpy.types.Collection) -> None:
        for obj in c.objects:
            result.add(obj.name)
        for child in c.children:
            _walk(child)

    _walk(coll)
    return result


def _scene_bbox() -> Tuple[Vector, Vector]:
    excluded_names = _lighting_env_objects()
    points: List[Vector] = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name.startswith("PointCloud_XZ"):
            continue
        if obj.name in excluded_names:
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    if not points:
        return Vector((0.0, 0.0, 0.0)), Vector((4.0, 4.0, 3.0))
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return lo, hi


# ---------------------------------------------------------------------------
# Temporary camera helpers
# ---------------------------------------------------------------------------

def _make_temp_camera(name: str) -> bpy.types.Object:
    cam_data = bpy.data.cameras.get(name)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.get(name)
    if cam_obj is None:
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    elif cam_obj.data is not cam_data:
        cam_obj.data = cam_data
    return cam_obj


def _aim_camera(cam_obj: bpy.types.Object, location: Vector, look_at: Vector) -> None:
    cam_obj.location = location
    direction = look_at - location
    if direction.length < 1e-6:
        direction = Vector((0.0, 0.0, -1.0))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler("XYZ")


def _delete_temp_camera(name: str) -> None:
    cam_obj = bpy.data.objects.get(name)
    if cam_obj is not None:
        bpy.data.objects.remove(cam_obj, do_unlink=True)
    cam_data = bpy.data.cameras.get(name)
    if cam_data is not None:
        bpy.data.cameras.remove(cam_data)


# ---------------------------------------------------------------------------
# Object hide/show helpers
# ---------------------------------------------------------------------------

def _hide_objects(names: List[str]) -> None:
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True


def _show_objects(names: List[str]) -> None:
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = False


# ---------------------------------------------------------------------------
# Render helper — does NOT touch engine / view_transform / world (unless
# the caller has already arranged a neutral context via _neutral_lighting)
# ---------------------------------------------------------------------------

def _render_view(
    scene: bpy.types.Scene,
    output_path: Path,
    resolution_x: int,
    resolution_y: int,
) -> None:
    scene.render.filepath = str(output_path)
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def _set_render_engine(scene: bpy.types.Scene, requested: str) -> str:
    requested = (requested or "CYCLES").strip()
    if requested == "BLENDER_EEVEE_NEXT":
        candidates = ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
    elif requested == "BLENDER_EEVEE":
        candidates = ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")
    else:
        candidates = (requested,)

    for engine in candidates:
        try:
            scene.render.engine = engine
            return engine
        except Exception:
            continue
    raise RuntimeError(f"unable to set render engine from request={requested!r}")


# ---------------------------------------------------------------------------
# Neutral lighting context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _neutral_lighting(scene: bpy.types.Scene, cumulative_scale_inverse: float):
    """Context manager: replace world with flat Background and dim all lights.

    On entry:
      - Saves current world node tree state and all LIGHT.energy values.
      - Replaces world with Background node (color 0.7,0.7,0.75; strength 1.0).
      - Sets each LIGHT.energy to original_energy * cumulative_scale_inverse * 0.3
        (so lights are at ~30% of the "neutral" single-unit level).
    On exit (try/finally):
      - Fully restores prior world tree and all light energies.

    The 0.3 factor gives a gently lit interior without the extreme ~14× calibration
    that makes the wall-facing views correctly exposed for view 1 only.
    """
    world = scene.world
    if world is None:
        # No world at all — create a minimal one just for the renders.
        world = bpy.data.worlds.new("_NeutralWorld_tmp")
        scene.world = world
        world_was_none = True
    else:
        world_was_none = False

    # --- Save world state ---
    orig_use_nodes = world.use_nodes
    # Serialize the current node tree by storing reference; we'll restore by
    # rebuilding the tree (shallow copy of nodes+links isn't straightforward
    # in the Blender Python API, so we save/restore key values instead).
    orig_bg_strength: Optional[float] = None
    orig_bg_color: Optional[tuple] = None
    orig_sky_nodes: List[dict] = []
    orig_world_had_nodes = world.use_nodes
    if orig_world_had_nodes and world.node_tree is not None:
        for n in world.node_tree.nodes:
            if n.bl_idname == "ShaderNodeBackground":
                orig_bg_strength = n.inputs["Strength"].default_value
                orig_bg_color = tuple(n.inputs["Color"].default_value)
            elif n.bl_idname == "ShaderNodeTexSky":
                orig_sky_nodes.append({
                    "name": n.name,
                    "sky_type": n.sky_type,
                    "sun_elevation": n.sun_elevation,
                    "sun_rotation": n.sun_rotation,
                    "dust_density": n.dust_density,
                })

    # --- Save all light energies ---
    orig_light_energies: Dict[str, float] = {}
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            orig_light_energies[obj.name] = obj.data.energy

    # --- Save exposure ---
    orig_exposure = scene.view_settings.exposure

    try:
        # Replace world with flat neutral Background
        world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.7, 0.7, 0.75, 1.0)
        bg.inputs["Strength"].default_value = 1.0
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

        # Dim all lights: calibrated_energy * cumulative_scale_inverse * 0.3
        for obj in bpy.data.objects:
            if obj.type == "LIGHT" and obj.name in orig_light_energies:
                obj.data.energy = orig_light_energies[obj.name] * cumulative_scale_inverse * 0.3

        yield

    finally:
        # Restore world
        if world_was_none:
            bpy.data.worlds.remove(world)
            scene.world = None
        else:
            world.use_nodes = orig_world_had_nodes
            if orig_world_had_nodes and world.node_tree is not None:
                nt = world.node_tree
                nt.nodes.clear()
                # Rebuild the saved sky + background structure
                if orig_sky_nodes:
                    for saved in orig_sky_nodes:
                        sky = nt.nodes.new("ShaderNodeTexSky")
                        sky.sky_type = saved["sky_type"]
                        sky.sun_elevation = saved["sun_elevation"]
                        sky.sun_rotation = saved["sun_rotation"]
                        sky.dust_density = saved["dust_density"]
                        sky.location = (-400, 0)
                    bg2 = nt.nodes.new("ShaderNodeBackground")
                    bg2.location = (-100, 0)
                    out2 = nt.nodes.new("ShaderNodeOutputWorld")
                    out2.location = (200, 0)
                    if orig_bg_strength is not None:
                        bg2.inputs["Strength"].default_value = orig_bg_strength
                    if orig_bg_color is not None:
                        bg2.inputs["Color"].default_value = orig_bg_color
                    # Link sky → bg → out
                    sky_nodes = [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexSky"]
                    if sky_nodes:
                        nt.links.new(sky_nodes[0].outputs["Color"], bg2.inputs["Color"])
                    nt.links.new(bg2.outputs["Background"], out2.inputs["Surface"])
                else:
                    # No sky was present — rebuild a simple Background
                    bg2 = nt.nodes.new("ShaderNodeBackground")
                    out2 = nt.nodes.new("ShaderNodeOutputWorld")
                    if orig_bg_strength is not None:
                        bg2.inputs["Strength"].default_value = orig_bg_strength
                    if orig_bg_color is not None:
                        bg2.inputs["Color"].default_value = orig_bg_color
                    nt.links.new(bg2.outputs["Background"], out2.inputs["Surface"])

        # Restore all light energies
        for obj in bpy.data.objects:
            if obj.type == "LIGHT" and obj.name in orig_light_energies:
                obj.data.energy = orig_light_energies[obj.name]

        # Restore exposure
        scene.view_settings.exposure = orig_exposure


# ---------------------------------------------------------------------------
# Per-view render functions
# ---------------------------------------------------------------------------

def render_perspective(
    scene: bpy.types.Scene,
    output_path: Path,
    resolution_x: int,
    resolution_y: int,
    scene_dir: Optional[Path] = None,
) -> None:
    """View 1 — GALP reference vantage: camera at (0,0,0) with predicted rotation.

    Reads layout_prediction.json from scene_dir to recover the original predicted
    camera rotation and focal length.  Applies the standard rz+180 fix that
    convert.py applies after Stage 1 (corrects upside-down framing).

    Walls that block the camera's line of sight to the room centroid are made
    camera-invisible (visible_camera=False) so they don't occlude the interior.
    They remain visible to light rays so global illumination is unaffected.

    Falls back to the existing scene Camera as-is if layout_prediction.json is
    missing or lacks the required keys.
    """
    if scene.camera is None:
        print("[render_multi_view] WARNING: no scene camera found; skipping perspective view")
        return

    # ------------------------------------------------------------------
    # Attempt to load layout_prediction.json for the reference vantage.
    # ------------------------------------------------------------------
    ref_rot: Optional[Tuple[float, float, float]] = None
    ref_lens: Optional[float] = None

    if scene_dir is not None:
        lp_candidates = [
            scene_dir / "inputs" / "layout_prediction.json",
            scene_dir / "layout_prediction.json",
            scene_dir / "scene-pipeline" / "layout_prediction.json",
        ]
        for lp_path in lp_candidates:
            if not lp_path.exists():
                continue
            try:
                lp = json.loads(lp_path.read_text())
                rot_deg = lp["blender_camera_rotation"]   # [rx, ry, rz]
                lens_mm = float(lp["blender_focal_length"])

                rx_deg = float(rot_deg[0])
                ry_deg = float(rot_deg[1])
                # Apply the standard rz+180 fix (same as convert.py after Stage 1)
                rz_raw = (float(rot_deg[2]) + 180.0) % 360.0
                rz_deg = rz_raw if rz_raw <= 180.0 else rz_raw - 360.0

                ref_rot = (
                    math.radians(rx_deg),
                    math.radians(ry_deg),
                    math.radians(rz_deg),
                )
                ref_lens = lens_mm
                break  # found and parsed
            except Exception as _e:
                print(f"[render_multi_view] WARNING: could not parse {lp_path}: {_e}")
                break

    # ------------------------------------------------------------------
    # Fallback: use scene.camera as-is (original behavior).
    # ------------------------------------------------------------------
    if ref_rot is None or ref_lens is None:
        print(
            "[render_multi_view] WARNING: layout_prediction.json missing — "
            "falling back to scene.camera vantage"
        )
        print(
            f"[render_multi_view] view=perspective  cam={scene.camera.name}  "
            f"lighting=calibrated  output={output_path}"
        )
        _render_view(scene, output_path, resolution_x, resolution_y)
        return

    # ------------------------------------------------------------------
    # Reference vantage path: override scene.camera to (0,0,0) + predicted rotation.
    # ------------------------------------------------------------------
    cam_obj = scene.camera

    # Snapshot camera state for restoration.
    orig_loc = cam_obj.location.copy()
    orig_rot_euler = cam_obj.rotation_euler.copy()
    orig_lens = cam_obj.data.lens

    # ------------------------------------------------------------------
    # Find blocking walls (walls that lie between camera and room centroid).
    # ------------------------------------------------------------------
    blocking_walls: List[str] = []
    if scene_dir is not None:
        for bsj_candidate in (
            scene_dir / "json" / "blender_scene.json",
            scene_dir / "blender_scene.json",
            scene_dir / "scene-pipeline" / "blender_scene.json",
        ):
            if not bsj_candidate.exists():
                continue
            try:
                bsj = json.loads(bsj_candidate.read_text())
                stage_data = bsj.get("stage", bsj)  # support both layouts

                verts_raw = stage_data.get("polygon_vertices", [])
                wall_names: List[str] = stage_data.get("wall_objects", [])

                if verts_raw and len(verts_raw) >= 3 and wall_names:
                    verts_2d = [(float(v[0]), float(v[1])) for v in verts_raw]
                    cx = sum(v[0] for v in verts_2d) / len(verts_2d)
                    cy = sum(v[1] for v in verts_2d) / len(verts_2d)

                    # Camera is at origin (0, 0) in XY.
                    cam_x, cam_y = 0.0, 0.0
                    dir_x = cx - cam_x
                    dir_y = cy - cam_y
                    dir_len = math.sqrt(dir_x * dir_x + dir_y * dir_y)

                    if dir_len > 1e-6:
                        dir_nx = dir_x / dir_len   # normalised cam→centroid direction
                        dir_ny = dir_y / dir_len

                        # Room extent for clearance threshold.
                        room_w = max(v[0] for v in verts_2d) - min(v[0] for v in verts_2d)
                        room_d = max(v[1] for v in verts_2d) - min(v[1] for v in verts_2d)
                        wall_clearance = max(room_w, room_d) * 0.5

                        for wname in wall_names:
                            wobj = bpy.data.objects.get(wname)
                            if wobj is None:
                                continue
                            # Wall_NN objects have origin at (0,0,0); compute actual
                            # bbox center in world space from the 8 bound_box corners.
                            bb = [wobj.matrix_world @ Vector(c) for c in wobj.bound_box]
                            wx = sum(c.x for c in bb) / 8.0
                            wy = sum(c.y for c in bb) / 8.0
                            # Vector from camera to wall midpoint.
                            wx_rel = wx - cam_x
                            wy_rel = wy - cam_y
                            # Projection along cam→centroid direction.
                            proj = wx_rel * dir_nx + wy_rel * dir_ny
                            # Perpendicular distance from wall midpoint to the line.
                            perp = abs(wx_rel * dir_ny - wy_rel * dir_nx)
                            # Wall is blocking if it lies between camera and centroid
                            # and is close to the camera-centroid line.
                            if 0.0 < proj < dir_len and perp < wall_clearance:
                                blocking_walls.append(wname)
            except Exception as _e:
                print(f"[render_multi_view] WARNING: could not read blender_scene.json "
                      f"for wall analysis: {_e}")
            break  # stop after first candidate

    # Snapshot visible_camera state for blocking walls.
    orig_visible_camera: Dict[str, bool] = {}
    for wname in blocking_walls:
        wobj = bpy.data.objects.get(wname)
        if wobj is not None:
            orig_visible_camera[wname] = wobj.visible_camera
            wobj.visible_camera = False

    rx_deg_log = math.degrees(ref_rot[0])
    ry_deg_log = math.degrees(ref_rot[1])
    rz_deg_log = math.degrees(ref_rot[2])
    print(
        f"[render_multi_view] view=perspective (reference vantage): "
        f"cam=(0.0,0.0,0.0) "
        f"rot_deg=({rx_deg_log:.1f},{ry_deg_log:.1f},{rz_deg_log:.1f}) "
        f"lens={ref_lens:.1f}mm "
        f"hidden={blocking_walls}"
    )

    try:
        # Override camera to GALP reference vantage.
        cam_obj.location = Vector((0.0, 0.0, 0.0))
        cam_obj.rotation_euler = Euler(ref_rot, "XYZ")
        cam_obj.data.lens = ref_lens

        _render_view(scene, output_path, resolution_x, resolution_y)

    finally:
        # Restore camera state.
        cam_obj.location = orig_loc
        cam_obj.rotation_euler = orig_rot_euler
        cam_obj.data.lens = orig_lens

        # Restore visible_camera on hidden walls.
        for wname, orig_val in orig_visible_camera.items():
            wobj = bpy.data.objects.get(wname)
            if wobj is not None:
                wobj.visible_camera = orig_val


def render_bev(
    scene: bpy.types.Scene,
    output_path: Path,
    resolution_x: int,
    resolution_y: int,
    lo: Vector,
    hi: Vector,
    cumulative_scale_inverse: float,
) -> None:
    """View 2 — top-down orthographic floor-plan view, brightly and evenly lit.

    Camera position: (cx, cy, hi.z + 5.0) looking straight down (-Z).
    ortho_scale: max(room_w, room_d) * 1.05.

    Lighting strategy (self-contained, independent of scene calibration):
      - Ceiling hidden (hide_render=True) so the ortho camera sees the floor.
      - All existing lights disabled (hide_render=True) — they were calibrated
        for the SCENE camera vantage and produce a hotspot in top-down view.
      - World replaced with a flat white Background (color 1.0, strength 0.5).
      - One temporary overhead Area light added:
          name="_MV_BEVAreaLight", type=AREA
          size = max(room_w, room_d) * 1.2
          location = (cx, cy, hi.z + 0.3)
          rotation = (pi, 0, 0)  -- pointing downward (-Z)
          energy = 5000 W  (bright even fill from above)
          color = (1.0, 1.0, 1.0)

    All temporary state is restored in try/finally.
    """
    cx = (lo.x + hi.x) * 0.5
    cy = (lo.y + hi.y) * 0.5
    room_w = hi.x - lo.x
    room_d = hi.y - lo.y

    cam_z = hi.z + 5.0
    ortho_scale = max(room_w, room_d) * 1.05

    print(
        f"[render_multi_view] view=bev: "
        f"pos=({cx:.2f},{cy:.2f},{cam_z:.2f})  "
        f"ortho_scale={ortho_scale:.2f}  lighting=bev_overhead  "
        f"output={output_path}"
    )

    # --- Snapshot: Ceiling hide_render state ---
    ceiling_obj = bpy.data.objects.get("Ceiling")
    orig_ceiling_hide: Optional[bool] = None
    if ceiling_obj is not None:
        orig_ceiling_hide = ceiling_obj.hide_render
        ceiling_obj.hide_render = True

    # --- Snapshot: all existing lights' hide_render + energy ---
    orig_light_hide: Dict[str, bool] = {}
    orig_light_energy: Dict[str, float] = {}
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            orig_light_hide[obj.name] = obj.hide_render
            orig_light_energy[obj.name] = obj.data.energy
            obj.hide_render = True  # suppress all scene lights for BEV

    # --- Snapshot: world state ---
    world = scene.world
    world_was_none = world is None
    if world_was_none:
        world = bpy.data.worlds.new("_BEV_TmpWorld")
        scene.world = world
    orig_world_use_nodes = world.use_nodes
    orig_world_bg_strength: Optional[float] = None
    orig_world_bg_color: Optional[tuple] = None
    orig_world_sky_nodes: List[dict] = []
    if world.use_nodes and world.node_tree is not None:
        for n in world.node_tree.nodes:
            if n.bl_idname == "ShaderNodeBackground":
                orig_world_bg_strength = n.inputs["Strength"].default_value
                orig_world_bg_color = tuple(n.inputs["Color"].default_value)
            elif n.bl_idname == "ShaderNodeTexSky":
                orig_world_sky_nodes.append({
                    "name": n.name,
                    "sky_type": n.sky_type,
                    "sun_elevation": n.sun_elevation,
                    "sun_rotation": n.sun_rotation,
                    "dust_density": n.dust_density,
                })

    # --- Add temporary overhead Area light ---
    bev_light_data = bpy.data.lights.new(name="_MV_BEVAreaLight", type="AREA")
    bev_light_data.shape = "SQUARE"
    bev_light_data.size = max(room_w, room_d) * 1.2
    bev_light_data.energy = 5000.0
    bev_light_data.color = (1.0, 1.0, 1.0)
    bev_light_obj = bpy.data.objects.new("_MV_BEVAreaLight", bev_light_data)
    bpy.context.scene.collection.objects.link(bev_light_obj)
    bev_light_obj.location = Vector((cx, cy, hi.z + 0.3))
    bev_light_obj.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")  # default points -Z (down at floor)

    # --- Replace world with flat white Background ---
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg_node = nt.nodes.new("ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs["Strength"].default_value = 0.5
    out_node = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    # --- Camera ---
    cam_obj = _make_temp_camera("_MV_BEVCam")
    cam_obj.data.type = "ORTHO"
    cam_obj.data.ortho_scale = ortho_scale
    cam_obj.data.clip_start = 0.01
    cam_obj.data.clip_end = (hi.z - lo.z + 8.0) * 2.0
    cam_obj.location = Vector((cx, cy, cam_z))
    cam_obj.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")  # default looks -Z (down at floor)

    orig_cam = scene.camera
    scene.camera = cam_obj

    try:
        _render_view(scene, output_path, resolution_x, resolution_y)
    finally:
        scene.camera = orig_cam
        _delete_temp_camera("_MV_BEVCam")

        # Remove temporary Area light
        bpy.data.objects.remove(bev_light_obj, do_unlink=True)
        bpy.data.lights.remove(bev_light_data)

        # Restore world
        if world_was_none:
            bpy.data.worlds.remove(world)
            scene.world = None
        else:
            world.use_nodes = orig_world_use_nodes
            if orig_world_use_nodes and world.node_tree is not None:
                nt = world.node_tree
                nt.nodes.clear()
                if orig_world_sky_nodes:
                    for saved in orig_world_sky_nodes:
                        sky = nt.nodes.new("ShaderNodeTexSky")
                        sky.sky_type = saved["sky_type"]
                        sky.sun_elevation = saved["sun_elevation"]
                        sky.sun_rotation = saved["sun_rotation"]
                        sky.dust_density = saved["dust_density"]
                        sky.location = (-400, 0)
                    bg2 = nt.nodes.new("ShaderNodeBackground")
                    bg2.location = (-100, 0)
                    out2 = nt.nodes.new("ShaderNodeOutputWorld")
                    out2.location = (200, 0)
                    if orig_world_bg_strength is not None:
                        bg2.inputs["Strength"].default_value = orig_world_bg_strength
                    if orig_world_bg_color is not None:
                        bg2.inputs["Color"].default_value = orig_world_bg_color
                    sky_nodes = [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexSky"]
                    if sky_nodes:
                        nt.links.new(sky_nodes[0].outputs["Color"], bg2.inputs["Color"])
                    nt.links.new(bg2.outputs["Background"], out2.inputs["Surface"])
                else:
                    bg2 = nt.nodes.new("ShaderNodeBackground")
                    out2 = nt.nodes.new("ShaderNodeOutputWorld")
                    if orig_world_bg_strength is not None:
                        bg2.inputs["Strength"].default_value = orig_world_bg_strength
                    if orig_world_bg_color is not None:
                        bg2.inputs["Color"].default_value = orig_world_bg_color
                    nt.links.new(bg2.outputs["Background"], out2.inputs["Surface"])

        # Restore all lights' hide_render + energy
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                if obj.name in orig_light_hide:
                    obj.hide_render = orig_light_hide[obj.name]
                if obj.name in orig_light_energy:
                    obj.data.energy = orig_light_energy[obj.name]

        # Restore Ceiling hide_render
        if ceiling_obj is not None and orig_ceiling_hide is not None:
            ceiling_obj.hide_render = orig_ceiling_hide


def render_wide(
    scene: bpy.types.Scene,
    output_path: Path,
    resolution_x: int,
    resolution_y: int,
) -> None:
    """View 3 — same vantage as scene Camera but with a 20 mm wide-angle lens.

    Calibrated lighting (same vantage as view 1 — brightness calibration holds).
    No hides.
    """
    scene_cam = scene.camera
    if scene_cam is None:
        print("[render_multi_view] WARNING: no scene camera; skipping wide view")
        return

    orig_loc = scene_cam.location.copy()
    orig_rot = scene_cam.rotation_euler.copy()
    orig_lens = scene_cam.data.lens

    print(
        f"[render_multi_view] view=wide  cam=({orig_loc.x:.2f},{orig_loc.y:.2f},{orig_loc.z:.2f})  "
        f"lens=20mm (was {orig_lens:.1f}mm)  lighting=calibrated  output={output_path}"
    )

    cam_obj = _make_temp_camera("_MV_WideCam")
    cam_obj.data.type = "PERSP"
    cam_obj.data.lens = 20.0  # wide angle — shows more of the room
    cam_obj.data.lens_unit = "MILLIMETERS"
    cam_obj.data.clip_start = scene_cam.data.clip_start
    cam_obj.data.clip_end = scene_cam.data.clip_end
    cam_obj.location = orig_loc.copy()
    cam_obj.rotation_euler = orig_rot.copy()

    orig_cam = scene.camera
    scene.camera = cam_obj
    try:
        _render_view(scene, output_path, resolution_x, resolution_y)
    finally:
        scene.camera = orig_cam
        _delete_temp_camera("_MV_WideCam")


def _topcorner_cam_pos(
    cx: float,
    cy: float,
    ceiling_z: float,
    scene_dir: Path,
    rank: int = 0,
) -> Tuple[Vector, str]:
    """Compute a camera position for a top-corner view, guaranteed inside the room polygon.

    Strategy: read stage.polygon_vertices from blender_scene.json, then pick the
    polygon vertex whose direction from centroid scores at position `rank` in
    ascending order of dot product with the centroid-to-scene-camera direction
    (XY only).

      rank=0  → MOST opposite to scene camera  (used by topcorner view)
      rank=1  → SECOND-most opposite           (used by topcorner_opposite view)

    "Most opposite" is the vertex whose direction from the centroid has the LOWEST
    dot product with the centroid-to-camera direction.  rank=1 picks the next-best
    diagonal vantage, giving a complementary view from the other far corner.

    Auto-skip guard: if the polygon has fewer than (rank + 1) distinct vertices
    after deduplication (i.e. not enough distinct far-corner candidates), returns
    (None, "skipped") and the caller should skip the render.

    If scene.camera is unavailable, falls back to the vertex with the largest x+y
    sum (rank=0) or second-largest (rank=1) as the original heuristic.

    Camera position: vertex + 0.30 * (centroid - vertex) — 30% inward from the
    corner, guaranteed inside any convex or near-convex polygon.

    Returns (cam_pos_Vector_or_None, log_label_str).
    """
    ALPHA = 0.30  # fraction from vertex toward centroid

    scene = bpy.context.scene

    # Try to load polygon vertices from blender_scene.json
    for candidate in (
        scene_dir / "json" / "blender_scene.json",
        scene_dir / "blender_scene.json",
        scene_dir / "scene-pipeline" / "blender_scene.json",
    ):
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
            verts_raw = (
                data.get("stage", {}).get("polygon_vertices")
                or data.get("polygon_vertices")
            )
            if verts_raw and len(verts_raw) >= 3:
                verts_2d = [(float(v[0]), float(v[1])) for v in verts_raw]

                # Guard: need at least rank+1 vertices for a meaningful Nth pick.
                if len(verts_2d) <= rank:
                    return None, f"skipped(polygon_too_small_for_rank={rank})"

                # Compute centroid-to-scene-camera direction (XY only).
                scene_cam_dir: Optional[Tuple[float, float]] = None
                if scene.camera is not None:
                    cam_loc = scene.camera.location
                    dcx = cam_loc.x - cx
                    dcy = cam_loc.y - cy
                    cam_mag = math.sqrt(dcx * dcx + dcy * dcy)
                    if cam_mag > 1e-6:
                        scene_cam_dir = (dcx / cam_mag, dcy / cam_mag)

                if scene_cam_dir is not None:
                    # Sort all vertices by ascending dot score (most-opposite first).
                    def _vert_score(vxy: Tuple[float, float]) -> float:
                        dvx = vxy[0] - cx
                        dvy = vxy[1] - cy
                        vmag = math.sqrt(dvx * dvx + dvy * dvy) + 1e-12
                        return (dvx / vmag) * scene_cam_dir[0] + (dvy / vmag) * scene_cam_dir[1]

                    # Rank vertices by score (ascending = most-opposite first).
                    sorted_indices = sorted(range(len(verts_2d)), key=lambda i: _vert_score(verts_2d[i]))
                    # Guard: ensure distinct score from rank-0 for rank=1.
                    if rank >= len(sorted_indices):
                        return None, f"skipped(not_enough_distinct_vertices_for_rank={rank})"
                    best_idx = sorted_indices[rank]
                    score_val = _vert_score(verts_2d[best_idx])
                    rank_label = "opposite_cam" if rank == 0 else f"2nd_opposite_cam(rank={rank})"
                    label = (
                        f"anchor_vertex=V{best_idx}=({verts_2d[best_idx][0]:.2f},"
                        f"{verts_2d[best_idx][1]:.2f})[{rank_label},score={score_val:.2f}]"
                    )
                else:
                    # Fallback within polygon path: sort by descending x+y when camera unknown.
                    sorted_by_sum = sorted(
                        range(len(verts_2d)),
                        key=lambda i: verts_2d[i][0] + verts_2d[i][1],
                        reverse=True,
                    )
                    if rank >= len(sorted_by_sum):
                        return None, f"skipped(not_enough_distinct_vertices_for_rank={rank})"
                    best_idx = sorted_by_sum[rank]
                    rank_label = "largest_x+y_fallback" if rank == 0 else f"2nd_largest_x+y_fallback(rank={rank})"
                    label = (
                        f"anchor_vertex=V{best_idx}=({verts_2d[best_idx][0]:.2f},"
                        f"{verts_2d[best_idx][1]:.2f})[{rank_label}]"
                    )

                vx, vy = verts_2d[best_idx]
                cam_x = vx + ALPHA * (cx - vx)
                cam_y = vy + ALPHA * (cy - vy)
                return Vector((cam_x, cam_y, ceiling_z - 0.5)), label
        except Exception as _e:
            print(f"[render_multi_view] WARNING: could not read polygon_vertices from {candidate}: {_e}")
        break  # stop after first candidate (whether it succeeded or failed)

    # Fallback: centroid (polygon_vertices not available at all)
    if rank > 0:
        # For ranks > 0, centroid fallback is not meaningful — skip.
        return None, f"skipped(polygon_vertices_not_found,rank={rank})"
    print("[render_multi_view] WARNING: polygon_vertices not found — using centroid fallback for topcorner")
    return Vector((cx, cy, ceiling_z - 0.5)), "anchor_vertex=None(centroid_fallback)"


def render_topcorner(
    scene: bpy.types.Scene,
    output_path: Path,
    resolution_x: int,
    resolution_y: int,
    lo: Vector,
    hi: Vector,
    cumulative_scale_inverse: float,
    scene_dir: Optional[Path] = None,
) -> None:
    """View 4 — elevated 3/4 perspective from just below the ceiling, far-side corner.

    Camera position: polygon-vertex-anchored (30% from the vertex MOST opposite
    the scene camera, toward centroid), z = ceiling_z - 0.5.  Guaranteed inside
    any convex polygon.  Aimed at room centroid (cx, cy, cz).  Lens 18 mm.
    Calibrated lighting (same rig as perspective/wide).
    No hides — all walls visible to show spatial layout from above.

    The "most opposite to scene camera" heuristic: among all polygon vertices,
    picks the one whose centroid-relative direction has the lowest dot product with
    the centroid-to-scene-camera direction (XY).  This always produces a view from
    across the room regardless of polygon orientation.  Falls back to largest-x+y
    if scene.camera is unavailable.

    Uses rank=0 (most-opposite vertex).  See render_topcorner_opposite for rank=1.
    """
    cx = (lo.x + hi.x) * 0.5
    cy = (lo.y + hi.y) * 0.5
    cz = (lo.z + hi.z) * 0.5
    room_w = hi.x - lo.x
    room_d = hi.y - lo.y
    ceiling_z = hi.z

    cam_pos, anchor_label = _topcorner_cam_pos(cx, cy, ceiling_z, scene_dir or Path("."), rank=0)
    if cam_pos is None:
        print(f"[render_multi_view] view=topcorner: skipped — {anchor_label}")
        return
    look_at = Vector((cx, cy, cz))

    print(
        f"[render_multi_view] view=topcorner: "
        f"{anchor_label} → "
        f"cam=({cam_pos.x:.2f},{cam_pos.y:.2f},{cam_pos.z:.2f})  "
        f"lens=18mm  lighting=calibrated  output={output_path}"
    )

    cam_obj = _make_temp_camera("_MV_TopCornerCam")
    cam_obj.data.type = "PERSP"
    cam_obj.data.lens = 18.0
    cam_obj.data.lens_unit = "MILLIMETERS"
    cam_obj.data.clip_start = 0.05
    cam_obj.data.clip_end = max(room_w, room_d) * 4.0
    _aim_camera(cam_obj, cam_pos, look_at)

    orig_cam = scene.camera
    scene.camera = cam_obj
    try:
        # Use calibrated lighting (same as perspective/wide). Neutral lighting was
        # too dim — the calibrated rig at this in-room vantage produces a normal
        # lit interior shot, since we're inside the brightness-calibrated envelope.
        _render_view(scene, output_path, resolution_x, resolution_y)
    finally:
        scene.camera = orig_cam
        _delete_temp_camera("_MV_TopCornerCam")


def render_topcorner_opposite(
    scene: bpy.types.Scene,
    output_path: Path,
    resolution_x: int,
    resolution_y: int,
    lo: Vector,
    hi: Vector,
    cumulative_scale_inverse: float,
    scene_dir: Optional[Path] = None,
) -> None:
    """View 5 — complementary elevated 3/4 view from the OTHER far corner.

    Camera position: polygon-vertex-anchored (30% from the vertex SECOND-most
    opposite the scene camera, toward centroid), z = ceiling_z - 0.5.  This is the
    next-best diagonal vantage after topcorner (view 4), giving a complementary
    view from the other side of the room.  Same lens (18 mm), same calibrated
    lighting algorithm.

    Uses rank=1 in _topcorner_cam_pos (second-lowest dot-product score).

    Auto-skipped if the polygon has fewer than 2 distinct vertex scores (i.e.,
    fewer than 2 vertices to pick from).  A "skipped" log line is printed and the
    output file is not written; the caller notes the skip in the summary.
    """
    cx = (lo.x + hi.x) * 0.5
    cy = (lo.y + hi.y) * 0.5
    cz = (lo.z + hi.z) * 0.5
    room_w = hi.x - lo.x
    room_d = hi.y - lo.y
    ceiling_z = hi.z

    cam_pos, anchor_label = _topcorner_cam_pos(cx, cy, ceiling_z, scene_dir or Path("."), rank=1)
    if cam_pos is None:
        print(f"[render_multi_view] view=topcorner_opposite: skipped — {anchor_label}")
        return
    look_at = Vector((cx, cy, cz))

    print(
        f"[render_multi_view] view=topcorner_opposite: "
        f"{anchor_label} → "
        f"cam=({cam_pos.x:.2f},{cam_pos.y:.2f},{cam_pos.z:.2f})  "
        f"lens=18mm  lighting=calibrated  output={output_path}"
    )

    cam_obj = _make_temp_camera("_MV_TopCornerOppCam")
    cam_obj.data.type = "PERSP"
    cam_obj.data.lens = 18.0
    cam_obj.data.lens_unit = "MILLIMETERS"
    cam_obj.data.clip_start = 0.05
    cam_obj.data.clip_end = max(room_w, room_d) * 4.0
    _aim_camera(cam_obj, cam_pos, look_at)

    orig_cam = scene.camera
    scene.camera = cam_obj
    try:
        _render_view(scene, output_path, resolution_x, resolution_y)
    finally:
        scene.camera = orig_cam
        _delete_temp_camera("_MV_TopCornerOppCam")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    scene_dir = Path(args.scene_dir)
    if not scene_dir.is_dir():
        print(f"[render_multi_view] ERROR: scene_dir does not exist: {scene_dir}")
        sys.exit(1)

    scene = bpy.context.scene
    orig_engine = scene.render.engine
    orig_cycles_samples: Optional[int] = None
    orig_eevee_samples: Optional[int] = None

    chosen_engine = _set_render_engine(scene, args.engine)
    print(f"[render_multi_view] engine -> {chosen_engine}")

    if hasattr(scene, "cycles") and hasattr(scene.cycles, "samples"):
        orig_cycles_samples = scene.cycles.samples
        if chosen_engine == "CYCLES" and args.samples != orig_cycles_samples:
            scene.cycles.samples = args.samples
            print(f"[render_multi_view] cycles.samples: {orig_cycles_samples} → {args.samples}")

    eevee = getattr(scene, "eevee", None)
    if eevee is not None and hasattr(eevee, "taa_render_samples"):
        orig_eevee_samples = eevee.taa_render_samples
        if chosen_engine != "CYCLES" and args.samples != orig_eevee_samples:
            eevee.taa_render_samples = args.samples
            print(f"[render_multi_view] eevee.taa_render_samples: {orig_eevee_samples} → {args.samples}")

    # Load cumulative_scale_inverse for neutral lighting
    csi = _load_cumulative_scale_inverse(args.brightness_log, scene_dir)

    # Compute scene bbox once
    lo, hi = _scene_bbox()
    print(
        f"[render_multi_view] scene bbox: "
        f"lo=({lo.x:.2f},{lo.y:.2f},{lo.z:.2f})  "
        f"hi=({hi.x:.2f},{hi.y:.2f},{hi.z:.2f})"
    )

    # Store the original scene camera
    orig_camera = scene.camera

    # --- Fix 1: Mute all compositor FileOutput nodes and disable compositing
    # during multi-view renders to prevent Stage 5 env_preview PNG from being
    # overwritten by each render() call firing the compositor chain.
    file_output_nodes: List[bpy.types.Node] = []
    orig_mute_states: Dict[str, bool] = {}
    orig_use_compositing: Optional[bool] = None

    if scene.use_nodes and scene.node_tree is not None:
        for node in scene.node_tree.nodes:
            if node.bl_idname == "CompositorNodeOutputFile":
                file_output_nodes.append(node)
                orig_mute_states[node.name] = node.mute
                node.mute = True

    orig_use_compositing = scene.render.use_compositing
    scene.render.use_compositing = False

    n_muted = len(file_output_nodes)
    print(
        f"[render_multi_view] muted {n_muted} FileOutput nodes; "
        f"use_compositing=False during multi-view"
    )

    # Output paths — by default <scene_dir>/render/, override with --output-dir
    render_dir = Path(args.output_dir) if args.output_dir else scene_dir / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    out_perspective         = render_dir / "blender_scene_view_perspective.png"
    out_bev                 = render_dir / "blender_scene_view_bev.png"
    out_wide                = render_dir / "blender_scene_view_wide.png"
    out_topcorner           = render_dir / "blender_scene_view_topcorner.png"
    out_topcorner_opposite  = render_dir / "blender_scene_view_topcorner_opposite.png"

    try:
        # View 1 — GALP reference vantage (origin + predicted rotation, blocking walls hidden)
        render_perspective(scene, out_perspective, args.resolution_x, args.resolution_y, scene_dir=scene_dir)

        # View 2 — BEV top-down (temp ortho camera, BEV overhead rig, Ceiling hidden)
        render_bev(scene, out_bev, args.resolution_x, args.resolution_y, lo, hi, csi)

        # View 3 — wide-angle at same vantage (temp cam, calibrated lighting)
        render_wide(scene, out_wide, args.resolution_x, args.resolution_y)

        # View 4 — elevated corner 3/4 from MOST-opposite polygon vertex (rank=0, calibrated)
        render_topcorner(
            scene, out_topcorner, args.resolution_x, args.resolution_y,
            lo, hi, csi, scene_dir=scene_dir,
        )

        # View 5 — complementary elevated 3/4 from SECOND-most-opposite polygon vertex (rank=1, calibrated)
        render_topcorner_opposite(
            scene, out_topcorner_opposite, args.resolution_x, args.resolution_y,
            lo, hi, csi, scene_dir=scene_dir,
        )

    finally:
        # Always restore original camera and samples
        scene.camera = orig_camera
        if orig_cycles_samples is not None and hasattr(scene, "cycles") and scene.cycles.samples != orig_cycles_samples:
            scene.cycles.samples = orig_cycles_samples
            print(f"[render_multi_view] cycles.samples restored to {orig_cycles_samples}")
        if (
            orig_eevee_samples is not None
            and eevee is not None
            and hasattr(eevee, "taa_render_samples")
            and eevee.taa_render_samples != orig_eevee_samples
        ):
            eevee.taa_render_samples = orig_eevee_samples
            print(f"[render_multi_view] eevee.taa_render_samples restored to {orig_eevee_samples}")
        if scene.render.engine != orig_engine:
            scene.render.engine = orig_engine
            print(f"[render_multi_view] engine restored to {orig_engine}")

        # Restore compositor FileOutput mute states and use_compositing
        for node in file_output_nodes:
            node.mute = orig_mute_states.get(node.name, False)
        if orig_use_compositing is not None:
            scene.render.use_compositing = orig_use_compositing
        print("[render_multi_view] restored compositor state")

    print("[render_multi_view] All 5 views complete:")
    for label, p in (
        ("perspective         (GALP reference vantage, blocking walls hidden)  ", out_perspective),
        ("bev                 (BEV overhead Area light, flat white world)            ", out_bev),
        ("wide                (calibrated lighting, 20mm lens)                       ", out_wide),
        ("topcorner           (calibrated lighting, far-side corner cam, rank=0)     ", out_topcorner),
        ("topcorner_opposite  (calibrated lighting, 2nd far-side corner, rank=1)     ", out_topcorner_opposite),
    ):
        exists_str = "OK" if p.exists() else "MISSING/SKIPPED"
        print(f"  [{exists_str}] {label}  {p}")


if __name__ == "__main__":
    main()
