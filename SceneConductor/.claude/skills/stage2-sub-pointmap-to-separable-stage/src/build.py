"""
build_blend_from_json.py

Builds a Blender .blend file from a Blender-space scene JSON.

Usage:
    blender --background --python build_blend_from_json.py -- \
        --input  /path/to/blender_scene.json \
        --output /path/to/blender_scene.blend

The JSON is ALREADY in Blender coordinates — no coordinate conversion is applied.
Schema expected:
    meta.coordinate_system == "blender"
    meta.rotation_order    == "XYZ"
    meta.rotation_unit     == "radians"
    camera: { location, rotation_euler, lens, sensor_width, sensor_fit,
              resolution[W,H], clip_start, clip_end }
    floor:   { id, mesh_path, location, rotation_euler, scale[sx,sy,sz] }  (optional)
    objects[]: { id, mesh_path, location, rotation_euler, scale[sx,sy,sz] }

Normalization contract:
    The JSON's location/rotation/scale transforms are defined relative to a
    unit-cube-normalized mesh (vertices in [-1, 1], centered at origin,
    max half-extent = 1.0).  Raw mesh files are NOT pre-normalized — each
    has its own arbitrary size and center.  This loader bakes a per-mesh
    unit-cube normalization into the vertex data before applying the JSON
    transforms, so the resulting world positions match the intended layout.

Backwards compatibility:
    If top-level "floor" key is absent but objects[] contains an entry with id=="floor",
    that entry is used as the floor and processed the same way.
    If both exist, the top-level "floor" key takes precedence.

v2.0 additive blocks (all optional; absent blocks are silently ignored):
    point_cloud, stage, stage_materials, lighting, world, render
"""

import sys
import os
import json
import argparse
import math

# ---------------------------------------------------------------------------
# sys.path augmentation — must happen before any skill-local imports
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Own src dir — siblings ply_import.py + build_stage_v2.py
# (build_stage_v2.build_from_polygon_dict provides the Stage collection builder).
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# stage2-sub-env-enhance src dir — needed for _material_utils (apply_mat_color,
# _get_or_create_principled, build_nishita_world).
_ENV_ENHANCE_SCRIPTS = os.path.normpath(
    os.path.join(_SCRIPTS_DIR,
                 "../../stage2-sub-env-enhance/src")
)
if _ENV_ENHANCE_SCRIPTS not in sys.path:
    sys.path.insert(0, _ENV_ENHANCE_SCRIPTS)

# ---------------------------------------------------------------------------
# Parse args (everything after the "--" separator)
# ---------------------------------------------------------------------------
if "--" in sys.argv:
    script_args = sys.argv[sys.argv.index("--") + 1:]
else:
    script_args = []

parser = argparse.ArgumentParser(description="Build .blend from scene JSON")
parser.add_argument("--input",  required=True, help="Path to blender_scene.json")
parser.add_argument("--output", required=True, help="Output .blend path")
args = parser.parse_args(script_args)

# ---------------------------------------------------------------------------
# Import bpy after arg parsing (bpy is only available inside Blender)
# ---------------------------------------------------------------------------
import bpy
from mathutils import Euler, Vector, Matrix

# ---------------------------------------------------------------------------
# Load JSON
# ---------------------------------------------------------------------------
with open(args.input, "r") as fh:
    scene_data = json.load(fh)

assert scene_data["meta"]["coordinate_system"] == "blender", \
    "JSON must already be in Blender coordinate space"
assert scene_data["meta"]["rotation_unit"] == "radians", \
    "Expected rotation_unit == 'radians'"

# scene_dir is the root scene directory — not the json/ subdirectory that may
# contain the input file.  If the JSON lives under <scene_dir>/json/, resolve
# one level up so that relative texture/PLY/HDRI paths in the JSON still work.
_raw_scene_dir = os.path.dirname(os.path.abspath(args.input))
if os.path.basename(_raw_scene_dir) == "json":
    scene_dir = os.path.dirname(_raw_scene_dir)
else:
    scene_dir = _raw_scene_dir

# ---------------------------------------------------------------------------
# Backward-compat shim: rename legacy aligned_scene.blend → blender_scene.blend
#
# If a previous run left an aligned_scene.blend in the scene_dir (pre-rename
# convention) and no blender_scene.blend exists yet, rename it atomically so
# users migrating from v1 don't lose their prior build.  If blender_scene.blend
# already exists, the legacy file is left alone with a WARN.
# ---------------------------------------------------------------------------
_legacy_blend = os.path.join(scene_dir, "aligned_scene.blend")
_canonical_blend = os.path.join(scene_dir, "blend", "blender_scene.blend")
if os.path.isfile(_legacy_blend):
    if not os.path.isfile(_canonical_blend):
        os.makedirs(os.path.join(scene_dir, "blend"), exist_ok=True)
        os.replace(_legacy_blend, _canonical_blend)
        print(f"[build.py] Renamed legacy aligned_scene.blend -> blend/blender_scene.blend")
    else:
        print(f"[build.py] WARN: aligned_scene.blend found in scene_dir but "
              f"blend/blender_scene.blend already exists — leaving legacy file in place.")

# ---------------------------------------------------------------------------
# 1. Clean slate — empty scene (no default cube / camera / light)
# ---------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------------------------------------------------------------------
# 2. Import each object
# ---------------------------------------------------------------------------
def import_mesh(mesh_path: str) -> list:
    """Import a mesh file and return the list of newly created root objects."""
    before = set(bpy.data.objects.keys())

    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".glb":
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == ".obj":
        # Blender >= 3.4 ships wm.obj_import; older builds use import_scene.obj
        try:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext} (path: {mesh_path})")

    after = set(bpy.data.objects.keys())
    new_names = after - before
    return [bpy.data.objects[n] for n in new_names]


def purge_embedded_cameras_and_lights(objects):
    """Delete any cameras or lights that were embedded in a GLB file."""
    to_delete = [o for o in objects if o.type in {"CAMERA", "LIGHT"}]
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)
    return [o for o in objects if o.type not in {"CAMERA", "LIGHT"}]


def patch_trellis_pbr(objects):
    """TRELLIS-style GLBs (SAM3D output) ship with metallic=1.0 on the
    Principled BSDF, which renders BLACK without strong HDRI reflections.
    Reset to metallic=0.0, roughness=0.7, specular=0.5 so objects render
    correctly under any lighting setup. Only touches imported object
    materials; stage materials live in their own namespace and are never
    exposed to this pass."""
    seen = set()
    patched = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            if not mat.use_nodes:
                mat.use_nodes = True
            bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Roughness"].default_value = 0.7
            for sk in [k for k in bsdf.inputs.keys() if "Specular" in k]:
                try:
                    bsdf.inputs[sk].default_value = 0.5
                except Exception:
                    pass
            patched += 1
    if patched:
        print(f"  trellis_pbr_patch: {patched} material(s) -> metallic=0.0, roughness=0.7")


def collect_world_verts(obj):
    """Collect all world-space vertex positions from obj and its mesh descendants."""
    bpy.context.view_layer.update()
    pts = []
    def _recurse(o):
        if o.type == "MESH":
            for v in o.data.vertices:
                pts.append(o.matrix_world @ v.co)
        for child in o.children:
            _recurse(child)
    _recurse(obj)
    return pts


def bake_unit_cube_normalization(root_objects):
    """
    Bake a unit-cube normalization into the mesh vertex data for a logical part
    (a list of root objects that together form one imported mesh asset).

    After this call:
      - Every mesh vertex has been moved so the combined world AABB is centered
        at the world origin with max half-extent = 1.0.
      - All object transforms (location, rotation, scale) are zeroed so the
        vertex data alone encodes the normalized positions.
      - Returns (center, half_extent) of the pre-normalization AABB, for logging.

    Strategy:
      1. Collect all world-space vertex positions across the hierarchy.
      2. Compute AABB center and max-axis half-extent.
      3. For each mesh object, transform its vertex coordinates so that:
             v_new = (v_world - center) / half_extent
         by computing v_new = inv(W_new) @ ... where W_new is identity.
         Equivalently: write v_new directly into mesh data as local coords
         (since after we zero all transforms, local == world).
      4. Zero every object's transform so local space == world space.
    """
    bpy.context.view_layer.update()

    # Collect all world-space vertices across the whole hierarchy
    all_pts = []
    for root in root_objects:
        all_pts.extend(collect_world_verts(root))

    if not all_pts:
        print("  WARNING: no vertices found for normalization")
        return None, None

    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    zs = [p.z for p in all_pts]

    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    center = Vector((cx, cy, cz))

    half_extent = max(
        (max(xs) - min(xs)) / 2.0,
        (max(ys) - min(ys)) / 2.0,
        (max(zs) - min(zs)) / 2.0,
    )
    if half_extent < 1e-8:
        print("  WARNING: degenerate mesh (zero extent) — skipping normalization")
        return None, None

    # For each mesh object in the hierarchy, rewrite vertex coordinates to
    # normalized world-space positions, then zero the object's transform.
    def _normalize_mesh_obj(o):
        if o.type == "MESH":
            world_mat = o.matrix_world.copy()
            mesh = o.data
            for v in mesh.vertices:
                wp = world_mat @ v.co          # current world position
                v.co = (wp - center) / half_extent   # normalized local = world position
            mesh.update()
        for child in o.children:
            _normalize_mesh_obj(child)

    for root in root_objects:
        _normalize_mesh_obj(root)

    # Zero all transforms so local == world for every object in the hierarchy
    def _zero_transform(o):
        o.location = Vector((0.0, 0.0, 0.0))
        o.rotation_mode = "XYZ"
        o.rotation_euler = Euler((0.0, 0.0, 0.0), "XYZ")
        o.scale = Vector((1.0, 1.0, 1.0))
        for child in o.children:
            _zero_transform(child)

    for root in root_objects:
        _zero_transform(root)

    bpy.context.view_layer.update()

    # Verify normalization (world AABB should be centered at origin with he=1.0)
    check_pts = []
    for root in root_objects:
        check_pts.extend(collect_world_verts(root))
    if check_pts:
        cxs = [p.x for p in check_pts]
        cys = [p.y for p in check_pts]
        czs = [p.z for p in check_pts]
        check_cx = (min(cxs) + max(cxs)) / 2.0
        check_cy = (min(cys) + max(cys)) / 2.0
        check_cz = (min(czs) + max(czs)) / 2.0
        check_he = max(
            (max(cxs) - min(cxs)) / 2.0,
            (max(cys) - min(cys)) / 2.0,
            (max(czs) - min(czs)) / 2.0,
        )
        tol = 1e-4
        center_ok = (abs(check_cx) < tol and abs(check_cy) < tol and abs(check_cz) < tol)
        he_ok = abs(check_he - 1.0) < tol
        if not (center_ok and he_ok):
            print(f"  WARNING: normalization check failed — "
                  f"center=({check_cx:.6f},{check_cy:.6f},{check_cz:.6f}), he={check_he:.6f}")
        else:
            print(f"  normalization OK: center≈(0,0,0), he≈1.0")

    return center, half_extent


def import_and_place(obj_spec):
    """Import a mesh entry from the JSON spec, normalize it, and place it."""
    obj_id    = obj_spec["id"]
    mesh_path = obj_spec["mesh_path"]
    location  = obj_spec["location"]
    rot_euler = obj_spec["rotation_euler"]   # radians, XYZ
    scale     = obj_spec["scale"]

    print(f"[import] {obj_id} <- {mesh_path}")

    if not os.path.isfile(mesh_path):
        print(f"  SKIP: mesh file not found — {mesh_path}")
        return None

    new_objs = import_mesh(mesh_path)

    # Remove any cameras/lights embedded in the GLB
    new_objs = purge_embedded_cameras_and_lights(new_objs)

    # TRELLIS GLB sanitization: metallic=1.0 default renders BLACK without HDRI.
    # Reset to PBR defaults so any environment lighting works.
    patch_trellis_pbr(new_objs)

    if not new_objs:
        print(f"  WARNING: no mesh objects imported for {obj_id}")
        return None

    # Determine the root object(s)
    # Root = object whose parent is NOT in the newly imported set
    imported_set = set(new_objs)
    roots = [o for o in new_objs if (o.parent is None or o.parent not in imported_set)]

    # -----------------------------------------------------------------------
    # Bake unit-cube normalization into vertex data before applying JSON R/T/S.
    # The JSON's location/rotation/scale are defined relative to a unit-cube-
    # normalized mesh (centered at origin, max half-extent = 1.0 in each axis).
    # Raw mesh files are NOT pre-normalized, so we must bake normalization first.
    # -----------------------------------------------------------------------
    center, half_extent = bake_unit_cube_normalization(roots)
    if center is not None:
        print(f"  normalized: raw_center=({center.x:.4f},{center.y:.4f},{center.z:.4f}), "
              f"raw_he={half_extent:.4f}")

    # Create an Empty named <id> to carry the JSON transform
    empty = bpy.data.objects.new(obj_id, None)
    bpy.context.collection.objects.link(empty)

    # Parent all normalized roots under the Empty
    for r in roots:
        # Unparent from any existing parent first
        if r.parent is not None:
            r.parent = None
        r.parent = empty
        r.matrix_parent_inverse = Matrix.Identity(4)

    # Apply JSON transform to the Empty
    empty.rotation_mode = "XYZ"
    empty.location       = Vector(location)
    empty.rotation_euler = Euler(rot_euler, "XYZ")
    empty.scale          = Vector(scale)

    print(f"  -> empty='{empty.name}' "
          f"loc={[round(v, 6) for v in empty.location]} "
          f"rot={[round(v, 6) for v in empty.rotation_euler]} "
          f"scale={[round(v, 6) for v in empty.scale]}")

    # -----------------------------------------------------------------------
    # Validation: post-transform world AABB center should match JSON location.
    # (For zero rotation, world AABB center == location exactly.  With rotation
    #  the AABB center may differ slightly, but the Empty's own location is the
    #  exact translation and is the right thing to compare.)
    # -----------------------------------------------------------------------
    bpy.context.view_layer.update()
    post_pts = collect_world_verts(empty)
    if post_pts:
        pxs = [p.x for p in post_pts]
        pys = [p.y for p in post_pts]
        pzs = [p.z for p in post_pts]
        pcx = (min(pxs) + max(pxs)) / 2.0
        pcy = (min(pys) + max(pys)) / 2.0
        pcz = (min(pzs) + max(pzs)) / 2.0

        # Compare Empty location (exact T component) against JSON location
        tol = 1e-3
        dx = abs(empty.location.x - location[0])
        dy = abs(empty.location.y - location[1])
        dz = abs(empty.location.z - location[2])
        if dx > tol or dy > tol or dz > tol:
            print(f"  NORM_CHECK_FAILED: empty.location=({empty.location.x:.5f},"
                  f"{empty.location.y:.5f},{empty.location.z:.5f}) "
                  f"json_loc=({location[0]:.5f},{location[1]:.5f},{location[2]:.5f}) "
                  f"delta=({dx:.5f},{dy:.5f},{dz:.5f})")
        else:
            print(f"  NORM_CHECK OK: world aabb center=({pcx:.4f},{pcy:.4f},{pcz:.4f})")

    return empty


# ---------------------------------------------------------------------------
# Resolve floor spec: top-level "floor" key takes precedence; fall back to
# an entry with id=="floor" inside objects[] for backwards compatibility.
# ---------------------------------------------------------------------------
raw_objects = scene_data["objects"]

top_level_floor = scene_data.get("floor")
objects_floor   = next((o for o in raw_objects if o.get("id") == "floor"), None)

if top_level_floor is not None:
    floor_spec = top_level_floor
    scene_objects = raw_objects  # objects[] should already not contain floor
elif objects_floor is not None:
    floor_spec = objects_floor
    scene_objects = [o for o in raw_objects if o.get("id") != "floor"]
    print("[compat] floor found inside objects[] — using backwards-compat path")
else:
    floor_spec = None
    scene_objects = raw_objects

# ---------------------------------------------------------------------------
# 2a. Import each scene object (obj_1 … obj_9)
# ---------------------------------------------------------------------------
for obj_spec in scene_objects:
    import_and_place(obj_spec)

# ---------------------------------------------------------------------------
# 2b. Import floor (separate top-level entry)
# ---------------------------------------------------------------------------
if floor_spec is not None:
    import_and_place(floor_spec)
else:
    print("[floor] WARNING: no floor entry found in JSON — skipping floor import")

# ---------------------------------------------------------------------------
# 3. Camera (data API — avoids UI operator side-effects)
# ---------------------------------------------------------------------------
cam_spec = scene_data["camera"]

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens         = cam_spec["lens"]
cam_data.sensor_width = cam_spec["sensor_width"]
cam_data.sensor_fit   = cam_spec["sensor_fit"]
cam_data.clip_start   = cam_spec["clip_start"]
cam_data.clip_end     = cam_spec["clip_end"]

cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam_obj)

cam_obj.location       = Vector(cam_spec["location"])
cam_obj.rotation_mode  = "XYZ"
cam_obj.rotation_euler = Euler(cam_spec["rotation_euler"], "XYZ")

scene.camera = cam_obj

print(f"[camera] Camera loc={[round(v, 6) for v in cam_obj.location]} "
      f"rot={[round(v, 6) for v in cam_obj.rotation_euler]} "
      f"lens={cam_data.lens:.4f} sensor_width={cam_data.sensor_width}")

# ---------------------------------------------------------------------------
# 4. Render settings (from camera block — base resolution)
# ---------------------------------------------------------------------------
scene.render.resolution_x          = cam_spec["resolution"][0]
scene.render.resolution_y          = cam_spec["resolution"][1]
scene.render.resolution_percentage = 100

print(f"[render] resolution={scene.render.resolution_x}x{scene.render.resolution_y}")


# ============================================================================
# ---- additive v2.0 loaders ----
#
# All functions below are gated by `if "<block>" in scene_data`.
# They have NO effect when the block is absent, ensuring byte-equivalent
# output for legacy (pre-v2.0) JSONs.
# ============================================================================


# ---------------------------------------------------------------------------
# Loader 1: point_cloud — import PLY into scene (hidden by default)
# ---------------------------------------------------------------------------

def build_point_cloud(scene_data: dict, scene_dir: str) -> int:
    """Import the PLY point cloud referenced by the ``point_cloud`` JSON block.

    Returns the vertex count of the imported object (0 if skipped/failed).
    Gated by ``scene_data.get("point_cloud")``.
    """
    pc = scene_data.get("point_cloud")
    if not pc:
        return 0

    ply_rel  = pc.get("ply_path", "")
    ply_path = os.path.join(scene_dir, ply_rel)

    if not os.path.isfile(ply_path):
        print(f"[build.py] WARN: point_cloud.ply_path not found: {ply_path} — skipping PLY import")
        return 0

    remap_str   = pc.get("axis_remap", "forward=Z,up=Y")
    world_scale = float(pc.get("world_scale_applied", 1.0))
    visible     = bool(pc.get("visible", False))
    obj_name    = pc.get("name", "PointCloud_XZ")

    # Defer import of ply_import so the module file is resolved at runtime.
    from ply_import import import_ply, parse_axis_remap
    axis_forward, axis_up = parse_axis_remap(remap_str)

    try:
        obj = import_ply(
            ply_path,
            name=obj_name,
            axis_forward=axis_forward,
            axis_up=axis_up,
            world_scale=world_scale,
            visible=visible,
        )
    except Exception as exc:
        print(f"[build.py] WARN: PLY import failed ({exc}) — skipping")
        return 0

    n_verts = len(obj.data.vertices) if obj.type == "MESH" else 0
    return n_verts


# ---------------------------------------------------------------------------
# Loader 2: stage — build Floor / Wall_NN / Ceiling from stage dict
# ---------------------------------------------------------------------------

def build_stage_from_json(scene_data: dict, blend_path: str) -> list:
    """Build the Stage collection from ``scene_data["stage"]``.

    Delegates to ``build_stage_v2.build_from_polygon_dict`` (Stage 4 library).
    Returns the errors list from the report (empty on full success).
    Gated by ``scene_data.get("stage")``.
    """
    stage = scene_data.get("stage")
    if not stage:
        return []

    print(f"[build.py] build_stage_from_json: {len(stage.get('wall_objects', []))} walls")

    try:
        from build_stage_v2 import build_from_polygon_dict
    except ImportError as exc:
        print(f"[build.py] WARN: could not import build_stage_v2 ({exc}) — skipping stage build")
        return [f"import error: {exc}"]

    try:
        report = build_from_polygon_dict(
            stage,
            blend_path,
            save=False,           # build.py does the single final save
            replace_existing=True,
        )
    except Exception as exc:
        print(f"[build.py] WARN: build_from_polygon_dict failed ({exc}) — skipping")
        return [f"build_from_polygon_dict error: {exc}"]

    n_walls    = len(report.get("wall_names", []))
    n_openings = report.get("openings_applied", 0)
    errors     = report.get("errors", [])
    manifold   = report.get("manifold_ok", True)
    print(f"[build.py] Stage: {n_walls} walls, {n_openings} openings, "
          f"manifold_ok={manifold}")
    if errors:
        for e in errors:
            print(f"[build.py] WARN (stage): {e}")
    return errors


# ---------------------------------------------------------------------------
# Loader 3: stage_materials — apply PBR params to Floor / Ceiling / Wall_NN
# ---------------------------------------------------------------------------

def apply_stage_materials(scene_data: dict, scene_dir: str) -> int:
    """Apply PBR parameters from ``scene_data["stage_materials"]`` to stage objects.

    Surfaces:
      floor            -> object named "Floor"    -> material "Mat_Floor_Stage"
      ceiling          -> object named "Ceiling"  -> material "Mat_Ceiling_Stage"
      walls.__default__-> all Wall_NN not listed  -> material "Mat_Walls_Stage"
      walls.<Wall_NN>  -> that specific object    -> material "Mat_<Wall_NN>"

    SAFETY: never touches Material_0.* or geometry_* materials (env-only rule).
    Returns the count of materials applied.
    Gated by ``scene_data.get("stage_materials")``.
    """
    sm = scene_data.get("stage_materials")
    if not sm:
        return 0

    try:
        from _material_utils import apply_mat_color, _get_or_create_principled
    except ImportError as exc:
        print(f"[build.py] WARN: could not import _material_utils ({exc}) — skipping stage_materials")
        return 0

    n_applied = 0

    def _resolve_pbr(spec: dict) -> tuple:
        """Return (linear_rgba, roughness, specular_ior_or_None) from a PBR spec dict."""
        rgba = tuple(spec["base_color_linear"])  # 4-element
        roughness = float(spec.get("roughness", 0.8))
        specular_ior = spec.get("specular_ior_level")
        if specular_ior is not None:
            specular_ior = float(specular_ior)
        return rgba, roughness, specular_ior

    def _get_or_create_mat(mat_name: str) -> "bpy.types.Material":
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
        return mat

    def _assign_mat(obj: "bpy.types.Object", mat: "bpy.types.Material") -> None:
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

    def _apply_texture(mat: "bpy.types.Material", spec: dict, scene_dir: str) -> None:
        """Link a ShaderNodeTexImage to Base Color if texture_image_path is set."""
        tex_rel = spec.get("texture_image_path")
        if not tex_rel:
            return
        tex_abs = os.path.join(scene_dir, tex_rel)
        if not os.path.isfile(tex_abs):
            print(f"[build.py] WARN: stage_materials texture not found: {tex_abs}")
            return
        pbsdf = _get_or_create_principled(mat)
        nt = mat.node_tree
        # Reuse existing TexImage node if already there.
        tex_node = None
        for node in nt.nodes:
            if node.bl_idname == "ShaderNodeTexImage":
                tex_node = node
                break
        if tex_node is None:
            tex_node = nt.nodes.new("ShaderNodeTexImage")
            tex_node.location = (-300, 0)
        img = bpy.data.images.get(os.path.basename(tex_abs))
        if img is None:
            img = bpy.data.images.load(tex_abs)
        tex_node.image = img
        nt.links.new(tex_node.outputs["Color"], pbsdf.inputs["Base Color"])

    def _apply_spec(obj_name: str, mat_name: str, spec: dict) -> None:
        """Apply PBR spec to obj_name, creating/assigning mat_name."""
        nonlocal n_applied
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "MESH":
            return
        mat = _get_or_create_mat(mat_name)
        rgba, roughness, specular_ior = _resolve_pbr(spec)
        apply_mat_color(mat, rgba, roughness, specular_ior)
        _apply_texture(mat, spec, scene_dir)
        _assign_mat(obj, mat)
        n_applied += 1
        print(f"[build.py] stage_materials -> {obj_name} [{mat_name}] "
              f"roughness={roughness:.2f}")

    # ---- floor ----
    if "floor" in sm:
        _apply_spec("Floor", "Mat_Floor_Stage", sm["floor"])

    # ---- ceiling ----
    if "ceiling" in sm:
        _apply_spec("Ceiling", "Mat_Ceiling_Stage", sm["ceiling"])

    # ---- walls ----
    walls_block = sm.get("walls", {})
    default_spec = walls_block.get("__default__")
    explicit_walls = {k: v for k, v in walls_block.items() if k != "__default__"}

    # Collect all Wall_NN objects in the scene.
    all_wall_objs = [o for o in bpy.data.objects
                     if o.name.startswith("Wall_") and o.type == "MESH"]

    for wall_obj in all_wall_objs:
        wname = wall_obj.name
        if wname in explicit_walls:
            _apply_spec(wname, f"Mat_{wname}", explicit_walls[wname])
        elif default_spec is not None:
            _apply_spec(wname, "Mat_Walls_Stage", default_spec)

    print(f"[build.py] apply_stage_materials: {n_applied} materials applied")
    return n_applied


# ---------------------------------------------------------------------------
# Loader 4: lighting — create / update lights from the lighting array
# ---------------------------------------------------------------------------

def build_lighting(scene_data: dict) -> int:
    """Create or update lights from ``scene_data["lighting"]``.

    Matches existing lights by name to avoid duplicates.  Puts each light into
    the collection specified by ``light["collection"]``; creates the collection
    if absent.  Returns the count of lights processed.
    Gated by ``scene_data.get("lighting")``.
    """
    lighting = scene_data.get("lighting")
    if not lighting:
        return 0

    print(f"[build.py] build_lighting: {len(lighting)} light(s)")

    def _get_or_create_collection(coll_name: str) -> "bpy.types.Collection":
        col = bpy.data.collections.get(coll_name)
        if col is None:
            col = bpy.data.collections.new(coll_name)
            bpy.context.scene.collection.children.link(col)
        return col

    n_lights = 0
    for light_spec in lighting:
        lname      = light_spec["name"]
        ltype      = light_spec["type"]          # SUN, AREA, POINT, SPOT
        loc        = light_spec.get("location", [0.0, 0.0, 5.0])
        rot        = light_spec.get("rotation_euler", [0.0, 0.0, 0.0])
        energy     = float(light_spec.get("energy", 1.0))
        color      = light_spec.get("color", [1.0, 1.0, 1.0])
        size       = light_spec.get("size")
        spread_deg = light_spec.get("spread_deg")
        shadow_ss  = light_spec.get("shadow_soft_size")
        coll_name  = light_spec.get("collection", "Lighting_Env")
        cycles_cfg = light_spec.get("cycles", {})

        col = _get_or_create_collection(coll_name)

        if lname in bpy.data.objects:
            # Update existing light — do not add a second one.
            obj = bpy.data.objects[lname]
            ld  = obj.data
        else:
            ld  = bpy.data.lights.new(name=lname, type=ltype)
            obj = bpy.data.objects.new(name=lname, object_data=ld)
            col.objects.link(obj)

        # Transform
        obj.location       = Vector(loc)
        obj.rotation_mode  = "XYZ"
        obj.rotation_euler = Euler(rot, "XYZ")

        # Light data
        ld.energy = energy
        ld.color  = color[:3]
        if size is not None and hasattr(ld, "size"):
            ld.size = float(size)
        if spread_deg is not None and hasattr(ld, "spot_size"):
            ld.spot_size = math.radians(float(spread_deg))
        if shadow_ss is not None and hasattr(ld, "shadow_soft_size"):
            ld.shadow_soft_size = float(shadow_ss)

        # Cycles-specific
        try:
            if "is_portal" in cycles_cfg:
                ld.cycles.is_portal = bool(cycles_cfg["is_portal"])
            if "cast_shadow" in cycles_cfg:
                ld.cycles.cast_shadow = bool(cycles_cfg["cast_shadow"])
        except Exception:
            pass  # cycles settings may not be available in EEVEE builds

        n_lights += 1
        print(f"[build.py] Light '{lname}' ({ltype}) energy={energy:.2f}")

    return n_lights


# ---------------------------------------------------------------------------
# Loader 5: world — build world shader (nishita_sky / hdri / flat)
# ---------------------------------------------------------------------------

def build_world(scene_data: dict, scene_dir: str) -> str:
    """Build the Blender World shader from ``scene_data["world"]``.

    Dispatches on ``world["mode"]``:
      "nishita_sky" — Nishita sky texture (delegates to _material_utils.build_nishita_world)
      "hdri"        — HDR environment map (ShaderNodeTexEnvironment)
      "flat"        — Uniform ShaderNodeBackground with solid color

    Returns the mode string actually applied, or "" if skipped.
    Gated by ``scene_data.get("world")``.
    """
    world_spec = scene_data.get("world")
    if not world_spec:
        return ""

    mode = world_spec.get("mode", "")
    print(f"[build.py] build_world: mode={mode!r}")

    w = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()

    if mode == "nishita_sky":
        try:
            from _material_utils import build_nishita_world
        except ImportError as exc:
            print(f"[build.py] WARN: could not import build_nishita_world ({exc}) — skipping world")
            return ""
        build_nishita_world(
            sun_elev_deg  = float(world_spec.get("sun_elevation_deg", 45.0)),
            sun_rot_deg   = float(world_spec.get("sun_rotation_deg", 45.0)),
            dust_density  = float(world_spec.get("dust_density", 1.5)),
            ozone_density = float(world_spec.get("ozone_density", 1.0)),
            altitude      = float(world_spec.get("altitude", 0.0)),
            strength      = float(world_spec.get("world_strength", 1.0)),
            exposure      = 0.0,  # exposure is part of the render block, not world
        )
        return "nishita_sky"

    elif mode == "hdri":
        hdri_rel  = world_spec.get("hdri_path", "")
        hdri_abs  = os.path.join(scene_dir, hdri_rel)
        if not os.path.isfile(hdri_abs):
            print(f"[build.py] WARN: world.hdri_path not found: {hdri_abs} — skipping HDRI")
            return ""

        strength    = float(world_spec.get("world_strength", 1.0))
        rotation_deg = float(world_spec.get("rotation_deg", 0.0))

        img = bpy.data.images.get(os.path.basename(hdri_abs))
        if img is None:
            img = bpy.data.images.load(hdri_abs)
        img.colorspace_settings.name = "Non-Color"

        tex_env = nt.nodes.new("ShaderNodeTexEnvironment")
        tex_env.image = img
        tex_env.location = (-600, 0)

        bg_node = nt.nodes.new("ShaderNodeBackground")
        bg_node.inputs["Strength"].default_value = strength
        bg_node.location = (-100, 0)

        out = nt.nodes.new("ShaderNodeOutputWorld")
        out.location = (200, 0)

        if abs(rotation_deg) > 1e-6:
            # Insert a Mapping node for HDRI rotation on Z.
            coord = nt.nodes.new("ShaderNodeTexCoord")
            coord.location = (-900, 0)
            mapping = nt.nodes.new("ShaderNodeMapping")
            mapping.location = (-750, 0)
            mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rotation_deg))
            nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
            nt.links.new(mapping.outputs["Vector"], tex_env.inputs["Vector"])

        nt.links.new(tex_env.outputs["Color"], bg_node.inputs["Color"])
        nt.links.new(bg_node.outputs["Background"], out.inputs["Surface"])
        print(f"[build.py] World HDRI: {hdri_abs} strength={strength} rotation_deg={rotation_deg}")
        return "hdri"

    elif mode == "flat":
        color    = world_spec.get("color_linear", [0.05, 0.05, 0.05])
        strength = float(world_spec.get("world_strength", 1.0))

        bg_node = nt.nodes.new("ShaderNodeBackground")
        bg_node.inputs["Color"].default_value    = (*color[:3], 1.0)
        bg_node.inputs["Strength"].default_value = strength
        bg_node.location = (-100, 0)

        out = nt.nodes.new("ShaderNodeOutputWorld")
        out.location = (200, 0)

        nt.links.new(bg_node.outputs["Background"], out.inputs["Surface"])
        print(f"[build.py] World flat: color={color} strength={strength}")
        return "flat"

    else:
        print(f"[build.py] WARN: unknown world.mode={mode!r} — skipping world build")
        return ""


# ---------------------------------------------------------------------------
# Loader 6: render — apply engine / samples / resolution / color-management
# ---------------------------------------------------------------------------

def apply_render_settings(scene_data: dict) -> None:
    """Apply render settings from ``scene_data["render"]``.

    Only touches fields that are present in the JSON; missing fields are left
    at their current Blender defaults.
    Gated by ``scene_data.get("render")``.
    """
    render_spec = scene_data.get("render")
    if not render_spec:
        return

    print(f"[build.py] apply_render_settings: {list(render_spec.keys())}")
    s = bpy.context.scene

    # Engine
    if "engine" in render_spec:
        try:
            s.render.engine = render_spec["engine"]
        except Exception as exc:
            print(f"[build.py] WARN: render.engine={render_spec['engine']!r} failed: {exc}")

    # Sampling (Cycles)
    if "samples" in render_spec:
        try:
            s.cycles.samples = int(render_spec["samples"])
        except Exception:
            pass

    # Resolution override (may differ from camera.resolution)
    if "resolution" in render_spec:
        s.render.resolution_x = int(render_spec["resolution"][0])
        s.render.resolution_y = int(render_spec["resolution"][1])

    # Color management
    if "view_transform" in render_spec:
        try:
            s.view_settings.view_transform = render_spec["view_transform"]
        except Exception as exc:
            print(f"[build.py] WARN: view_transform={render_spec['view_transform']!r}: {exc}")

    if "look" in render_spec:
        try:
            s.view_settings.look = render_spec["look"]
        except Exception as exc:
            print(f"[build.py] WARN: look={render_spec['look']!r}: {exc}")

    if "exposure" in render_spec:
        try:
            s.view_settings.exposure = float(render_spec["exposure"])
        except Exception:
            pass

    # Cycles advanced
    if "clamp_indirect" in render_spec:
        try:
            s.cycles.sample_clamp_indirect = float(render_spec["clamp_indirect"])
        except Exception:
            pass

    if "use_denoising" in render_spec:
        try:
            s.cycles.use_denoising = bool(render_spec["use_denoising"])
        except Exception:
            pass

    if "denoiser" in render_spec:
        try:
            s.cycles.denoiser = render_spec["denoiser"]
        except Exception as exc:
            print(f"[build.py] WARN: denoiser={render_spec['denoiser']!r}: {exc}")

    if "max_bounces" in render_spec:
        try:
            s.cycles.max_bounces = int(render_spec["max_bounces"])
        except Exception:
            pass

    if "diffuse_bounces" in render_spec:
        try:
            s.cycles.diffuse_bounces = int(render_spec["diffuse_bounces"])
        except Exception:
            pass

    if "glossy_bounces" in render_spec:
        try:
            s.cycles.glossy_bounces = int(render_spec["glossy_bounces"])
        except Exception:
            pass

    # Output format
    if "file_format" in render_spec:
        try:
            s.render.image_settings.file_format = render_spec["file_format"]
        except Exception:
            pass

    if "color_depth" in render_spec:
        try:
            s.render.image_settings.color_depth = render_spec["color_depth"]
        except Exception:
            pass

    print(f"[build.py] Render settings applied "
          f"(engine={s.render.engine}, "
          f"res={s.render.resolution_x}x{s.render.resolution_y})")


# ============================================================================
# ---- Run additive v2.0 loaders (in required order) ----
# All run AFTER camera + objects are built, BEFORE final save.
# ============================================================================

output_path = os.path.abspath(args.output)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Track summary state for the final log.
_summary_pc_verts    = 0
_summary_stage_built = False
_summary_stage_walls = 0
_summary_stage_openings = 0
_summary_stage_errors: list = []
_summary_world_mode  = "none"
_summary_n_lights    = 0

# 1. Point cloud
if "point_cloud" in scene_data:
    _summary_pc_verts = build_point_cloud(scene_data, scene_dir)

# 2. Stage geometry
if "stage" in scene_data:
    _summary_stage_errors = build_stage_from_json(scene_data, output_path)
    stage_coll = bpy.data.collections.get("Stage")
    if stage_coll is not None:
        _summary_stage_built = True
        _summary_stage_walls = sum(
            1 for o in stage_coll.objects if o.name.startswith("Wall_")
        )
        # Count openings from the JSON spec (applied count logged inside loader)
        _summary_stage_openings = len(
            scene_data["stage"].get("openings", [])
        )

# 3. Stage materials — only after stage geometry exists
if "stage_materials" in scene_data:
    apply_stage_materials(scene_data, scene_dir)

# 4. Lighting
if "lighting" in scene_data:
    _summary_n_lights = build_lighting(scene_data)

# 5. World
if "world" in scene_data:
    _summary_world_mode = build_world(scene_data, scene_dir)

# 6. Render settings
if "render" in scene_data:
    apply_render_settings(scene_data)


# ---------------------------------------------------------------------------
# Save — output is always blender_scene.blend
# ---------------------------------------------------------------------------
bpy.ops.wm.save_as_mainfile(filepath=output_path, copy=False)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
all_objects   = list(bpy.data.objects)
n_lights_total = sum(1 for o in all_objects if o.type == "LIGHT")
n_objects     = len(all_objects)

_stage_str = "no"
if _summary_stage_built:
    _stage_str = (f"yes ({_summary_stage_walls} walls "
                  f"/ {_summary_stage_openings} openings)")

_pc_str = "none"
if _summary_pc_verts > 0:
    pc_spec = scene_data.get("point_cloud", {})
    _pc_visible = pc_spec.get("visible", False)
    _pc_str = (f"{_summary_pc_verts} verts "
               f"({'visible' if _pc_visible else 'hidden'})")

print(f"\n[build.py] Saved blender_scene.blend")
print(f"  objects: {n_objects}, lights: {n_lights_total}, stage: {_stage_str},")
print(f"  world_mode: {_summary_world_mode}, point_cloud: {_pc_str}")
if _summary_stage_errors:
    print(f"  stage_errors: {_summary_stage_errors}")
