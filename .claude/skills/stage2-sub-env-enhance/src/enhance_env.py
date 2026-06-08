"""
enhance_env.py — stage2-sub-env-enhance skill main script (Blender 4.2 LTS, Cycles, AgX).

Usage:
    blender --background <input.blend> --python enhance_env.py -- \
        --output <output.blend> --preview <preview.png> --mood neutral_balanced \
        --wall-hex C2A8A6 --wall-lower-hex 9A807C \
        --floor-hex 6F4A2C --ceiling-hex E8DECF

Modifies ONLY: world/sky, stage material colors (Mat_Walls_Stage, Mat_Floor_Stage,
Mat_Ceiling_Stage, Mat_Wainscot*), light rig, render/compositor settings.
NEVER touches: Material_0* materials, geometry_* mesh materials, camera, object transforms.
"""

import sys
import os
import argparse
import math
import traceback
import json
from pathlib import Path

import bpy
import mathutils

# ---------------------------------------------------------------------------
# Mood presets — interior-only. Each preset shifts the color temperature and
# energy of the indoor light rig (Area_Fill + Practical_L/R). The room is a
# sealed box (Stage 3 builds Floor + Walls + Ceiling) so external light
# (Sun/Sky/HDRI/window portals) is never modeled. Names are color-semantic
# rather than time-of-day to avoid confusion with exterior lighting.
# ---------------------------------------------------------------------------
MOOD_PRESETS = {
    "neutral_balanced": {  # default — cool fill + warm practicals
        "fill_energy": 150.0, "fill_color": (0.95, 0.97, 1.0),
        "practical_energy": 15.0, "practical_color": (1.0, 0.88, 0.70),
        "exposure": 0.0,
    },
    "cool_fresh": {        # bluish, airy
        "fill_energy": 150.0, "fill_color": (0.92, 0.97, 1.0),
        "practical_energy": 15.0, "practical_color": (1.0, 0.92, 0.80),
        "exposure": 0.0,
    },
    "soft_diffuse": {      # near-neutral, low-contrast
        "fill_energy": 200.0, "fill_color": (0.96, 0.98, 1.0),
        "practical_energy": 8.0, "practical_color": (1.0, 0.95, 0.90),
        "exposure": 0.2,
    },
    "warm_amber": {        # strong warm cast
        "fill_energy": 150.0, "fill_color": (1.0, 0.92, 0.78),
        "practical_energy": 20.0, "practical_color": (1.0, 0.78, 0.55),
        "exposure": -0.3,
    },
}

# Back-compat aliases for old time-of-day names — accepted on input, mapped
# to the canonical key. Emits a warning so callers can update.
_MOOD_ALIASES = {
    "afternoon": "neutral_balanced",
    "morning": "cool_fresh",
    "overcast": "soft_diffuse",
    "golden_hour": "warm_amber",
}

# ---------------------------------------------------------------------------
# Hard-rule constants (enforced by SKILL.md)
# ---------------------------------------------------------------------------
# Rule 1: every light object newly created by this script must use 200 W,
# regardless of mood preset, scene scale, or downstream brightness alignment.
# Tagged via the custom property "env_enhance_fixed_energy" so
# _scale_all_light_energies can skip them.
FIXED_NEW_LIGHT_ENERGY = 200.0
FIXED_ENERGY_TAG = "env_enhance_fixed_energy"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def srgb_hex_to_linear_rgba(hex_str):
    """Parse #RRGGBB or RRGGBB sRGB hex and return linear (r, g, b, 1.0)."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str!r}")
    def to_lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (to_lin(int(h[0:2], 16)), to_lin(int(h[2:4], 16)), to_lin(int(h[4:6], 16)), 1.0)


def parse_color_tuple(s):
    """Parse 'R,G,B' float string into a (r, g, b) tuple."""
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Color must be R,G,B — got {s!r}")
    return tuple(float(p.strip()) for p in parts)


def get_scene_bbox():
    """Return (Vector min_xyz, Vector max_xyz) over all mesh objects in world space."""
    INF = float("inf")
    lo, hi = [INF, INF, INF], [-INF, -INF, -INF]
    found = False
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            wp = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                lo[i] = min(lo[i], wp[i])
                hi[i] = max(hi[i], wp[i])
            found = True
    if not found:
        return mathutils.Vector((0, 0, 0)), mathutils.Vector((5, 5, 3))
    return mathutils.Vector(lo), mathutils.Vector(hi)


def get_or_create_collection(name):
    """Return named collection, creating and linking to scene if absent."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv):
    """Extract args after '--' separator and parse with argparse."""
    script_args = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(prog="enhance_env.py")
    p.add_argument("--output", default=None)
    p.add_argument("--preview", default=None)
    p.add_argument("--mood", default="neutral_balanced",
                   choices=list(MOOD_PRESETS) + list(_MOOD_ALIASES))
    p.add_argument("--scale", default="auto", choices=["auto", "full", "1_10"])
    p.add_argument("--no-compositor", action="store_true")
    # Environment planner hints (from stage2-environment-planner). When --plan-file is None
    # we auto-detect <scene_dir>/json/stage2_plan.json next to --output.
    p.add_argument("--plan-file", default=None,
                   help="Path to stage2_plan.json (default: auto-detect at <scene_dir>/json/stage2_plan.json).")
    p.add_argument("--ignore-plan", action="store_true",
                   help="Ignore stage2_plan.json entirely; CLI flags + mood preset only.")
    # End of mood / plan flags — alias resolution happens after parse below.
    # Stage colors
    p.add_argument("--wall-hex", default=None)
    p.add_argument("--wall-lower-hex", default=None)
    p.add_argument("--floor-hex", default=None)
    p.add_argument("--ceiling-hex", default=None)
    # Fine-grained interior-light overrides
    p.add_argument("--fill-energy", type=float, default=None)
    p.add_argument("--fill-color", type=parse_color_tuple, default=None)
    p.add_argument("--practical-energy", type=float, default=None)
    p.add_argument("--practical-color", type=parse_color_tuple, default=None)
    p.add_argument("--exposure", type=float, default=None)
    # Render settings
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--resolution-x", type=int, default=1024)
    p.add_argument("--resolution-y", type=int, default=682)
    p.add_argument("--clamp-indirect", type=float, default=5.0)
    # Brightness alignment (off by default — existing callers unaffected)
    p.add_argument("--align_brightness", action="store_true",
                   help="If set, run a brightness alignment loop matching the reference image luminance.")
    p.add_argument("--reference_image", type=str, default=None,
                   help="Path to reference image (defaults to <scene_dir>/image.png if --align_brightness is set).")
    p.add_argument("--brightness_tolerance", type=float, default=0.05)
    p.add_argument("--brightness_max_iters", type=int, default=8)
    # brightness_log is written here; render_multi_view.py reads it via --brightness-log
    p.add_argument("--brightness-log", type=str, default=None,
                   help="Explicit path for brightness_align_log.json output "
                        "(default: <preview_dir>/brightness_align_log.json).")
    args = p.parse_args(script_args)
    # Map legacy time-of-day mood names to current color-semantic names.
    if args.mood in _MOOD_ALIASES:
        canonical = _MOOD_ALIASES[args.mood]
        print(f"[MOOD] '{args.mood}' is a legacy alias for '{canonical}' — using '{canonical}'.")
        args.mood = canonical
    return args


# ---------------------------------------------------------------------------
# Phase 1 — resolve paths
# ---------------------------------------------------------------------------

def resolve_paths(args):
    if args.output:
        out = os.path.abspath(args.output)
    else:
        fp = bpy.data.filepath
        out = os.path.abspath(fp) if fp else os.path.abspath("scene_env_enhanced.blend")
    if args.preview:
        preview = os.path.abspath(args.preview)
    else:
        # Default preview goes into <scene_dir>/render/ next to the .blend stem.
        # out is typically <scene_dir>/blend/<name>.blend; parent of parent = scene_dir.
        out_path = Path(out)
        render_dir = out_path.parent.parent / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        preview = str(render_dir / (out_path.stem + "_env_preview.png"))
    return out, preview


# ---------------------------------------------------------------------------
# Phase 1.5 — load stage2_plan.json hints (advisory, never overrides CLI)
# ---------------------------------------------------------------------------

_PLAN_MIN_CONFIDENCE = 0.5  # below this, the hint is ignored (consumer fall-back rule)


def _load_plan(args, output_path):
    """Locate + load stage2_plan.json and return it as a dict, or {} if absent.

    Auto-detect: <scene_dir>/json/stage2_plan.json where scene_dir is the
    parent of the parent of the .blend output (matches the project's canonical
    layout: <scene_dir>/blend/<name>.blend). Explicit --plan-file overrides.
    --ignore-plan short-circuits to {}.
    """
    if args.ignore_plan:
        print("[PLAN] --ignore-plan set; using CLI defaults only.")
        return {}

    if args.plan_file:
        plan_path = Path(args.plan_file)
    else:
        plan_path = Path(output_path).parent.parent / "json" / "stage2_plan.json"

    if not plan_path.is_file():
        # Defense-in-depth: if --ignore-plan is not set and the file is missing entirely,
        # fail loudly rather than silently falling back to defaults. The director MUST
        # have been run via run_stage2_director.py before enhance_env.py can proceed.
        print(
            f"[PLAN] FATAL: stage2_plan.json not found at {plan_path}.\n"
            "stage2_plan.json missing or unsigned — run Stage 0 director first via run_stage2_director.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(plan_path, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
    except Exception as e:
        print(
            f"[PLAN] FATAL: could not parse {plan_path}: {e}\n"
            "stage2_plan.json missing or unsigned — run Stage 0 director first via run_stage2_director.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Defense-in-depth: require _director_meta injected by run_stage2_director.py.
    # A file without this block was written by a legacy Agent dispatch that bypassed
    # the mandatory subprocess, and must be rejected.
    meta = plan.get("_director_meta")
    if not isinstance(meta, dict) or meta.get("generated_by") != "run_stage2_director.py":
        print(
            f"[PLAN] FATAL: stage2_plan.json at {plan_path} is missing or has invalid _director_meta.\n"
            "stage2_plan.json missing or unsigned — run Stage 0 director first via run_stage2_director.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify image_sha256 matches current image.png (scene_dir is parent of parent of .blend output).
    import hashlib
    image_path = Path(output_path).parent.parent / "image.png"
    if image_path.exists():
        h = hashlib.sha256()
        with open(image_path, "rb") as fh_img:
            for chunk in iter(lambda: fh_img.read(65536), b""):
                h.update(chunk)
        current_sha = h.hexdigest()
        cached_sha = meta.get("image_sha256", "")
        if cached_sha and cached_sha != current_sha:
            print(
                f"[PLAN] FATAL: stage2_plan.json _director_meta.image_sha256 does not match current image.png.\n"
                "stage2_plan.json missing or unsigned — run Stage 0 director first via run_stage2_director.py",
                file=sys.stderr,
            )
            sys.exit(1)

    sv = plan.get("schema_version")
    if sv not in ("1.0", "1.1"):
        print(f"[PLAN] WARNING: unknown schema_version={sv!r} (expected '1.0' or '1.1') — using hints anyway.")
    print(f"[PLAN] loaded {plan_path}  summary={plan.get('scene_summary', '<no summary>')[:80]!r}")
    return plan


def _apply_plan_to_args(args, plan):
    """Fill in missing CLI args from the plan. CLI values always win.

    Confidence-gated: when plan['confidence'][section] < 0.5, that section is
    skipped (per stage2_plan_schema.md consumer fall-back behavior).
    """
    if not plan:
        return

    conf = plan.get("confidence", {}) or {}
    materials_conf = float(conf.get("materials", 1.0))
    lighting_conf  = float(conf.get("lighting", 1.0))

    # Materials hints — only apply when each CLI flag is absent AND confidence ≥ threshold.
    if materials_conf >= _PLAN_MIN_CONFIDENCE:
        m = plan.get("materials_hint", {}) or {}
        for cli_attr, hint_key in (
            ("wall_hex",        "wall_hex"),
            ("wall_lower_hex",  "wall_lower_hex"),
            ("floor_hex",       "floor_hex"),
            ("ceiling_hex",     "ceiling_hex"),
        ):
            if getattr(args, cli_attr) is None and m.get(hint_key):
                setattr(args, cli_attr, m[hint_key])
                print(f"[PLAN]   applied {cli_attr} = {m[hint_key]} (from materials_hint)")
    else:
        print(f"[PLAN]   materials_hint skipped (confidence={materials_conf} < {_PLAN_MIN_CONFIDENCE})")

    # Lighting hints. Only `mood` is wired today; practicals_count / extra_central_light
    # are reserved for future use.
    if lighting_conf >= _PLAN_MIN_CONFIDENCE:
        l = plan.get("lighting_hint", {}) or {}
        mood_hint = l.get("mood")
        # Map legacy alias names in the plan to canonical mood names too.
        if mood_hint in _MOOD_ALIASES:
            mood_hint = _MOOD_ALIASES[mood_hint]
        if mood_hint and mood_hint in MOOD_PRESETS and args.mood == "neutral_balanced":
            # `neutral_balanced` is the argparse default; treat it as "user didn't choose".
            args.mood = mood_hint
            print(f"[PLAN]   applied mood = {mood_hint} (from lighting_hint)")
    else:
        print(f"[PLAN]   lighting_hint skipped (confidence={lighting_conf} < {_PLAN_MIN_CONFIDENCE})")


# ---------------------------------------------------------------------------
# Phase 2 — scale detection
# ---------------------------------------------------------------------------

def detect_scale(args):
    if args.scale != "auto":
        return args.scale
    lo, hi = get_scene_bbox()
    dx, dy = hi.x - lo.x, hi.y - lo.y
    diag = math.sqrt(dx * dx + dy * dy)
    print(f"[SCALE] XY diagonal = {diag:.4f} m -> {'1_10' if diag < 2.0 else 'full'}")
    return "1_10" if diag < 2.0 else "full"


# ---------------------------------------------------------------------------
# Phase 3 — resolve mood params
# ---------------------------------------------------------------------------

def resolve_mood_params(args, scale):
    """Start from preset, apply CLI overrides, then apply scale energy factor."""
    params = dict(MOOD_PRESETS[args.mood])
    override_map = {
        "fill_energy": args.fill_energy,
        "fill_color": args.fill_color,
        "practical_energy": args.practical_energy,
        "practical_color": args.practical_color,
        "exposure": args.exposure,
    }
    for k, v in override_map.items():
        if v is not None:
            params[k] = v
    if scale == "1_10":
        for ek in ("fill_energy", "practical_energy"):
            params[ek] /= 10.0
        print("[SCALE] Applied x0.1 energy factor.")
    return params


# ---------------------------------------------------------------------------
# Phase 4 — protection scan
# ---------------------------------------------------------------------------

def scan_materials():
    """Classify all materials. Returns (protected_names, touchable_names)."""
    # Collect materials used on geometry_* meshes
    geo_mats = set()
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith("geometry_"):
            for slot in obj.material_slots:
                if slot.material:
                    geo_mats.add(slot.material.name)

    STAGE_KW = ("Wall", "Floor", "Ceiling", "Wainscot")
    protected, touchable, unknown = set(), set(), set()
    for mat in bpy.data.materials:
        n = mat.name
        if n.startswith("Material_0") or n in geo_mats:
            protected.add(n)
        elif n.startswith("Mat_") and any(kw in n for kw in STAGE_KW):
            touchable.add(n)
        else:
            unknown.add(n)

    print("\n=== MATERIAL PROTECTION REPORT ===")
    for n in sorted(protected):  print(f"  [PROTECTED]    {n}")
    for n in sorted(touchable):  print(f"  [TOUCHABLE]    {n}")
    for n in sorted(unknown):    print(f"  [SKIP-UNKNOWN] {n}")
    print("==================================\n")
    return protected, touchable


# ---------------------------------------------------------------------------
# Phase 5 — world rebuild
# ---------------------------------------------------------------------------

def rebuild_world(params):
    """Build a sealed-interior world: no sky, no sun, no exterior contribution.

    Rationale: every room emitted by Stage 3 is a closed box (walls + floor +
    ceiling, openings authored only by the optional Stage 4.5 boolean-cut pass).
    Sealed walls block 100% of world emission from reaching the interior, so
    any non-zero world strength is wasted Cycles samples. We set the world to
    a flat black Background at strength 0; downstream lighting is owned
    entirely by interior light objects.
    """
    scene = bpy.context.scene
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    bg.location = (-100, 0)

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (200, 0)

    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    print("[WORLD] Sealed-interior world: flat black, strength=0 "
          "(exterior light disabled — room is a closed box)")


# ---------------------------------------------------------------------------
# Phase 6 — stage material rewrite
# ---------------------------------------------------------------------------

def _get_or_create_principled(mat):
    """Return Principled BSDF node in mat, wiring to output if newly created."""
    mat.use_nodes = True
    nt = mat.node_tree
    for node in nt.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    pbsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    pbsdf.location = (0, 0)
    for node in nt.nodes:
        if node.type == "OUTPUT_MATERIAL":
            nt.links.new(pbsdf.outputs["BSDF"], node.inputs["Surface"])
            break
    return pbsdf


def _apply_mat_color(mat, linear_rgba, roughness, specular_ior=None):
    pbsdf = _get_or_create_principled(mat)
    pbsdf.inputs["Base Color"].default_value = linear_rgba
    pbsdf.inputs["Roughness"].default_value = roughness
    if specular_ior is not None:
        for iname in ("Specular IOR Level", "Specular"):
            if iname in pbsdf.inputs:
                pbsdf.inputs[iname].default_value = specular_ior
                break
    print(f"  [MAT] {mat.name}: roughness={roughness}")


def rewrite_stage_materials(args, touchable, protected):
    """Apply hex colors to matching touchable stage materials."""
    # Each rule: (predicate, hex, roughness, specular_ior)
    rules = []
    if args.wall_hex:
        rules.append((
            lambda n: "Wainscot" not in n and ("Mat_Walls_Stage" in n or ("Mat_" in n and "Wall" in n)),
            args.wall_hex, 0.85, None))
    if args.wall_lower_hex:
        rules.append((lambda n: "Wainscot" in n, args.wall_lower_hex, 0.75, None))
    if args.floor_hex:
        rules.append((
            lambda n: "Mat_Floor_Stage" in n or ("Mat_" in n and "Floor" in n),
            args.floor_hex, 0.55, 0.5))
    if args.ceiling_hex:
        rules.append((
            lambda n: "Mat_Ceiling_Stage" in n or ("Mat_" in n and "Ceiling" in n),
            args.ceiling_hex, 0.9, None))

    modified, skipped = [], []
    for mat in bpy.data.materials:
        n = mat.name
        if n in protected:
            skipped.append(n)
            continue
        if n not in touchable:
            continue
        for pred, hex_val, roughness, specular in rules:
            if pred(n):
                _apply_mat_color(mat, srgb_hex_to_linear_rgba(hex_val), roughness, specular)
                modified.append(n)
                break

    print(f"[STAGE MATS] Modified: {modified}")
    print(f"[STAGE MATS] Protected/skipped: {sorted(set(skipped))}\n")
    return modified, skipped


# ---------------------------------------------------------------------------
# Phase 7 — light rig
# ---------------------------------------------------------------------------

def _compute_image_mean_luminance(png_path):
    """Rec.709 mean luminance over all pixels, [0..1]. Uses PIL."""
    from PIL import Image
    import numpy as np
    img = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32) / 255.0
    lum = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    return float(lum.mean())


def _scale_all_light_energies(factor):
    """Multiply data.energy of every Blender LIGHT object by `factor` (in place).

    Skips:
    - lights whose name starts with 'Pendant_user_' (user-added pendants preserved)
    - lights tagged with FIXED_ENERGY_TAG (Rule 1: fixed at 200 W on creation —
      brightness alignment must not move them off 200 W)
    """
    skipped_fixed = 0
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        if obj.name.startswith("Pendant_user_"):
            continue
        if obj.get(FIXED_ENERGY_TAG):
            skipped_fixed += 1
            continue
        obj.data.energy *= float(factor)
    if skipped_fixed:
        print(f"[BRIGHTNESS_ALIGN] skipped {skipped_fixed} fixed-energy light(s) (Rule 1)")


def _scale_world_background_strength(factor):
    """Multiply the Background node's Strength input on the active world by `factor`.

    No-op if the world has no node tree or no Background node — returns False then.
    World contributes via Nishita sky / sun in this skill, so it must scale together
    with lights for the brightness-alignment loop to actually move the needle.
    """
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return False
    scaled = False
    for n in world.node_tree.nodes:
        if n.bl_idname == "ShaderNodeBackground":
            n.inputs["Strength"].default_value *= float(factor)
            scaled = True
    return scaled


def _check_camera_interior_position(scene_json_path=None):
    """Log a WARNING if the scene camera is outside the room polygon or <1 m from any wall.

    This is a read-only safety check — it never moves the camera (Problem 1's fix in
    build_stage_v2.py is responsible for placement).  The check fires before the
    brightness-alignment loop so that a badly-placed camera does not silently calibrate
    lighting for a wall-surface view.

    Returns True if the camera is in a good interior position; False with a WARNING log
    otherwise.  Either way execution continues — this is advisory only.
    """
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.context.scene.camera
    if cam is None:
        print("[enhance_env] ALIGN-CHECK: no Camera found — skipping interior position check.")
        return True

    cx, cy = cam.location.x, cam.location.y

    # Try to load polygon vertices from blender_scene.json["stage"]["polygon_vertices"]
    poly_verts = None
    if scene_json_path is not None:
        try:
            import json as _json
            data = _json.loads(open(scene_json_path).read())
            poly_verts = data.get("stage", {}).get("polygon_vertices")
        except Exception as _e:
            print(f"[enhance_env] ALIGN-CHECK: could not read polygon from JSON: {_e}")

    if poly_verts is None or len(poly_verts) < 3:
        print("[enhance_env] ALIGN-CHECK: no polygon_vertices available — skipping check.")
        return True

    try:
        from shapely.geometry import Polygon as _Poly, Point as _Pt
        poly = _Poly(poly_verts)
        pt = _Pt(cx, cy)
        wall_dist = float(poly.exterior.distance(pt))
        inside = poly.contains(pt)
        if not inside:
            print(
                f"[enhance_env] WARNING: scene Camera at ({cx:.3f}, {cy:.3f}) is OUTSIDE "
                f"the room polygon (wall dist={wall_dist:.3f} m). "
                f"Brightness calibration may target a wall surface instead of the interior. "
                f"Re-run stage2-sub-pointmap-to-separable-stage to relocate the camera."
            )
            return False
        if wall_dist < 1.0:
            print(
                f"[enhance_env] WARNING: scene Camera at ({cx:.3f}, {cy:.3f}) is only "
                f"{wall_dist:.3f} m from the nearest wall (< 1 m minimum). "
                f"Brightness calibration may be dominated by a close wall surface."
            )
            return False
        print(
            f"[enhance_env] ALIGN-CHECK: Camera at ({cx:.3f}, {cy:.3f}) is inside polygon, "
            f"{wall_dist:.3f} m from nearest wall — OK."
        )
        return True
    except ImportError:
        print("[enhance_env] ALIGN-CHECK: shapely not available — skipping polygon check.")
        return True
    except Exception as _e:
        print(f"[enhance_env] ALIGN-CHECK: polygon check error: {_e} — proceeding anyway.")
        return True


def align_brightness_to_reference(scene, render_path, reference_path,
                                   tolerance=0.05, max_iters=3):
    """Iteratively scale light energies until render mean luminance matches reference.

    Args:
        scene: bpy.context.scene already configured with Cycles/AgX.
        render_path: pathlib.Path where preview render is saved each iter.
        reference_path: pathlib.Path of reference image (image.png).
        tolerance: |render_lum - ref_lum| below which we declare convergence.
        max_iters: hard cap on iterations.

    Returns:
        dict with keys: target, iterations (list of {iter,lum,scale_applied,delta}),
                        final_lum, final_delta, converged (bool),
                        cumulative_scale (float), cumulative_scale_inverse (float).
        cumulative_scale_inverse = 1.0 / cumulative_scale.  Downstream code (e.g.
        render_multi_view.py) uses this to undo the per-primary-vantage calibration
        and produce a neutral lighting state for non-primary views.
    """
    target = _compute_image_mean_luminance(reference_path)
    log = []
    cumulative_scale = 1.0
    converged = False
    render_lum = 0.0
    for i in range(1, max_iters + 1):
        scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        render_lum = _compute_image_mean_luminance(render_path)
        delta = abs(render_lum - target)
        if delta <= tolerance:
            log.append({"iter": i, "lum": render_lum, "scale_applied": 1.0, "delta": delta})
            converged = True
            break
        # Scale ratio (clamp to avoid runaway: 0.25..4.0 per iteration)
        ratio = max(0.25, min(4.0, target / max(render_lum, 1e-4)))
        _scale_all_light_energies(ratio)
        _scale_world_background_strength(ratio)
        cumulative_scale *= ratio
        log.append({"iter": i, "lum": render_lum, "scale_applied": ratio, "delta": delta})
    return {
        "target": target,
        "iterations": log,
        "cumulative_scale": cumulative_scale,
        "cumulative_scale_inverse": 1.0 / max(cumulative_scale, 1e-9),
        "final_lum": render_lum,
        "final_delta": abs(render_lum - target),
        "converged": converged,
    }


def _set_light(obj, energy, color, size=None):
    ld = obj.data
    ld.energy = energy
    ld.color = color
    if size is not None and hasattr(ld, "size"):
        ld.size = size


def _ensure_light(name, ltype, energy, color, loc, size=None, col=None):
    """Update existing light by name or create it and add to collection.

    Newly created lights are pinned to FIXED_NEW_LIGHT_ENERGY (Rule 1) and
    tagged with FIXED_ENERGY_TAG so the brightness-align loop skips them.
    Existing same-named lights are rebalanced via the caller's `energy` arg.
    """
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        _set_light(obj, energy, color, size)
        return obj
    ld = bpy.data.lights.new(name=name, type=ltype)
    ld.energy = FIXED_NEW_LIGHT_ENERGY
    ld.color = color
    if size is not None and hasattr(ld, "size"):
        ld.size = size
    obj = bpy.data.objects.new(name=name, object_data=ld)
    obj.location = loc
    obj[FIXED_ENERGY_TAG] = True
    target = col if col is not None else bpy.context.scene.collection
    target.objects.link(obj)
    print(f"[LIGHT] {name} created at fixed energy {FIXED_NEW_LIGHT_ENERGY} W (Rule 1)")
    return obj


def build_light_rig(params, scale):
    """Interior-only light rig. No Sun / no Sky / no window portals.

    The room is a sealed box (Stage 3 builds Floor+Walls+Ceiling), so exterior
    light contributes zero to the render. Everything visible must come from
    interior emitters:

      - `Area_Fill`         — overhead cool bounce (Lambert ceiling diffuser)
      - `Practical_L/R`     — two warm point lights for warm/cool contrast
      - class-driven lights — Blender LIGHTs spawned by `build_class_lights`
                              at every detected lamp/fluorescent/pendant
      - `Fallback_Light`    — central POINT light, only created if the rig
                              would otherwise have zero LIGHT objects
                              (Rule 2 guarantee — see SKILL.md hard rules)

    Stragglers from older runs (Sun, Area_Window, Portal_Window_*) are
    actively stripped so re-running on a previously-enhanced .blend never
    leaves dead exterior-light objects floating around.
    """
    col = get_or_create_collection("Lighting_Env")
    lo, hi = get_scene_bbox()
    cx = (lo.x + hi.x) / 2.0
    cy = (lo.y + hi.y) / 2.0
    cz_center = (lo.z + hi.z) / 2.0
    room_h = lo.z + (hi.z - lo.z) * 0.7
    fill_size = 3.0 if scale == "full" else 1.0

    # Strip any external/portal lights left over from previous enhance_env runs.
    _EXTERNAL_LIGHT_NAMES = (
        "Sun", "Area_Window", "Ambient_Fill",
        "Portal_Window_1", "Portal_Window_2",
    )
    stripped = []
    for name in _EXTERNAL_LIGHT_NAMES:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        ld = obj.data if obj.type == "LIGHT" else None
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
        if ld is not None and ld.users == 0:
            bpy.data.lights.remove(ld)
        stripped.append(name)
    if stripped:
        print(f"[LIGHT] stripped external lights from prior runs: {stripped}")

    # Area_Fill — overhead cool bounce (interior, not exterior).
    af = _ensure_light("Area_Fill", "AREA",
                       params["fill_energy"], params["fill_color"],
                       mathutils.Vector((cx, cy, hi.z - 0.1)),
                       size=fill_size, col=col)
    af.rotation_euler = mathutils.Euler((math.pi, 0.0, 0.0), "XYZ")
    print(f"[LIGHT] Area_Fill: energy={params['fill_energy']}, size={fill_size}")

    # Practical_L / Practical_R — warm interior point lights for cool/warm contrast.
    ox = (hi.x - lo.x) * 0.3
    oy = (hi.y - lo.y) * 0.1
    _ensure_light("Practical_L", "POINT",
                  params["practical_energy"], params["practical_color"],
                  mathutils.Vector((cx - ox, cy + oy, room_h)), col=col)
    _ensure_light("Practical_R", "POINT",
                  params["practical_energy"], params["practical_color"],
                  mathutils.Vector((cx + ox, cy - oy, room_h)), col=col)
    print(f"[LIGHT] Practical_L/R: energy={params['practical_energy']}")

    # Rule 2 guarantee: re-check after the rig is built. If somehow no LIGHT
    # objects exist (every _ensure_light no-op'd, or an external hook stripped
    # them), drop one POINT light at the true 3D centroid at fixed 200 W. This
    # block has no early-return / no exception path — the guarantee always runs.
    final_lights = [o for o in bpy.data.objects if o.type == "LIGHT"]
    if not final_lights:
        guarantee_loc = mathutils.Vector((cx, cy, cz_center))
        _ensure_light("Fallback_Light", "POINT",
                      FIXED_NEW_LIGHT_ENERGY,
                      params.get("practical_color", (1.0, 1.0, 1.0)),
                      guarantee_loc, col=col)
        print(f"[LIGHT] Rule 2 fallback fired: rig produced 0 lights — "
              f"force-created Fallback_Light at room-centroid "
              f"({cx:.2f}, {cy:.2f}, {cz_center:.2f})")
    else:
        print(f"[LIGHT] Rule 2 check: rig finished with {len(final_lights)} "
              f"light(s) — fallback not needed")


# Class-driven light fixtures: spawn a Blender LIGHT at the world location of
# every blender_scene.json object whose `class` field matches a known light/lamp
# keyword. Lets enhance_env reproduce the visible light emitters from the input
# image (fluorescents on the ceiling, lamps on tables, …) without depending on
# the Phase-B scene-enricher pass that scene-orchestra-run skips by default.
#
# (keyword substring, light_type, color_rgb, energy, size_m_or_None)
# Order matters: first match wins, so put specific keywords above generic ones.
_CLASS_LIGHT_PRESETS = (
    ("fluorescent", "AREA",  (0.96, 0.98, 1.00), 60.0, 0.6),
    ("ceiling_lig", "AREA",  (0.96, 0.98, 1.00), 60.0, 0.6),
    ("panel_light", "AREA",  (0.96, 0.98, 1.00), 60.0, 0.6),
    ("recessed",    "AREA",  (1.00, 0.95, 0.85), 40.0, 0.3),
    ("pendant",     "POINT", (1.00, 0.94, 0.85), 20.0, None),
    ("chandelier",  "POINT", (1.00, 0.92, 0.80), 25.0, None),
    ("desk_lamp",   "POINT", (1.00, 0.90, 0.75), 10.0, None),
    ("table_lamp",  "POINT", (1.00, 0.88, 0.72), 15.0, None),
    ("floor_lamp",  "POINT", (1.00, 0.88, 0.72), 20.0, None),
    ("lamp",        "POINT", (1.00, 0.90, 0.75), 15.0, None),
    ("sconce",      "POINT", (1.00, 0.85, 0.65), 10.0, None),
    ("light",       "AREA",  (0.96, 0.98, 1.00), 50.0, 0.5),
)


def _load_mesh_groups_map(scene_dir):
    """Read inputs/mask_attribute.json mesh_groups → return {instance_int_id: canonical_int_id}.

    Returns an empty dict if the file is missing or has no mesh_groups. Used by
    build_class_lights to share a single bpy.data.lights datablock across SAM3D
    dedup instances (so editing the canonical light edits all copies in one go).
    """
    import json as _json
    p = scene_dir / "inputs" / "mask_attribute.json"
    if not p.exists():
        return {}
    try:
        groups = _json.loads(p.read_text()).get("mesh_groups", {})
    except (_json.JSONDecodeError, OSError):
        return {}
    mapping = {}
    for grp in groups.values():
        canon = grp.get("canonical_id") or grp.get("canonical")
        if canon is None:
            continue
        for inst in grp.get("instance_ids", []):
            mapping[int(inst)] = int(canon)
    return mapping


def _disable_self_shadow_on_mesh(obj_id_str):
    """Mark the mesh corresponding to `obj_id_str` (e.g. 'obj_10') and its children
    as non-shadow-casting. Prevents the fixture's own GLB body from occluding the
    Blender light we just placed at its location.
    """
    targets = []
    parent = bpy.data.objects.get(obj_id_str)
    if parent is not None:
        targets.append(parent)
        targets.extend(parent.children_recursive)
    for t in targets:
        if t.type == "MESH":
            try:
                t.visible_shadow = False
            except AttributeError:
                pass


_CLASS_PARENT_CON_NAME = "Class_Parent_PosOnly"


def _attach_position_only_parent(blight, parent_empty):
    """Bind `blight` to follow only the LOCATION of `parent_empty`.

    Implemented via Child Of constraint with rotation/scale channels disabled,
    because regular Blender Object parenting does not expose
    inherit_rotation / inherit_scale toggles for non-bone parents. The
    `inverse_matrix` is set so the light stays at its current world location
    the moment the constraint is added (no jump).

    Idempotent: any prior Class_Parent_PosOnly constraint on `blight` is
    removed first, so re-runs of build_class_lights produce a clean state.
    """
    for c in list(blight.constraints):
        if c.name == _CLASS_PARENT_CON_NAME:
            blight.constraints.remove(c)

    con = blight.constraints.new(type="CHILD_OF")
    con.name = _CLASS_PARENT_CON_NAME
    con.target = parent_empty

    # Position-only follow: enable location channels, disable rotation+scale.
    con.use_location_x = True
    con.use_location_y = True
    con.use_location_z = True
    con.use_rotation_x = False
    con.use_rotation_y = False
    con.use_rotation_z = False
    con.use_scale_x = False
    con.use_scale_y = False
    con.use_scale_z = False

    # inverse_matrix cancels the parent's current contribution so the light
    # holds its existing world position. Because only translation channels are
    # active, the effective parent matrix the constraint applies is
    # Translation(parent_world_translation). Its inverse is the negative
    # translation; cancelling that keeps `blight` exactly where it is now.
    parent_world_loc = parent_empty.matrix_world.translation
    con.inverse_matrix = mathutils.Matrix.Translation(-parent_world_loc)


def build_class_lights(blend_path):
    """Add a Blender light at every object whose class matches a light/lamp keyword.

    Three behaviours layered on top of a basic class-keyword match:

    1. **Data-block sharing (Option A grouping).** SAM3D dedup instances belong
       to a single `mesh_groups` entry in inputs/mask_attribute.json (e.g. 4
       fluorescent_light copies share canonical=10). We create the
       `bpy.data.lights` datablock ONCE per canonical and link every instance
       Object to that shared datablock — editing one tweaks all copies.

    2. **Light-type-aware positioning.** AREA lights are pushed to *just below*
       the GLB's bottom face (`loc.z - scale.z/2 - 0.05`) so the panel mesh
       does not occlude its own downward-emitted light. POINT lights stay at
       the object centroid (lamps emit omnidirectionally).

    3. **Self-shadow disable.** For AREA fixtures the corresponding Empty's
       child mesh is set to `visible_shadow=False` so even residual occlusion
       inside the panel body doesn't shadow the room.
    """
    import json as _json
    blend_dir = Path(blend_path).parent
    scene_dir = blend_dir.parent
    candidates = [
        scene_dir / "json" / "blender_scene.json",
        scene_dir / "blender_scene.json",
    ]
    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        print("[CLASS_LIGHT] blender_scene.json not found — skipping class-driven lights.")
        return 0

    objects = _json.loads(json_path.read_text()).get("objects", [])
    inst_to_canon = _load_mesh_groups_map(scene_dir)
    col = get_or_create_collection("Lighting_Env")

    shared_data = {}  # canonical_int_id → bpy.data.lights datablock
    added = 0
    for obj in objects:
        cls = (obj.get("class") or "").lower()
        if not cls:
            continue
        match = next(((t, c, e, s) for kw, t, c, e, s in _CLASS_LIGHT_PRESETS if kw in cls), None)
        if match is None:
            continue
        ltype, color, energy, size = match

        try:
            obj_int_id = int(str(obj["id"]).split("_", 1)[1])
        except (KeyError, ValueError, IndexError):
            obj_int_id = -1
        canon_id = inst_to_canon.get(obj_int_id, obj_int_id)

        # 1. Get-or-create the shared LIGHT datablock for this canonical.
        ld = shared_data.get(canon_id)
        if ld is None:
            data_name = f"ClassLightData_{canon_id}"
            ld = bpy.data.lights.get(data_name)
            if ld is None or ld.type != ltype:
                if ld is not None:
                    bpy.data.lights.remove(ld, do_unlink=True)
                ld = bpy.data.lights.new(name=data_name, type=ltype)
            ld.energy = FIXED_NEW_LIGHT_ENERGY
            ld.color  = color
            if size is not None and hasattr(ld, "size"):
                ld.size = size
            shared_data[canon_id] = ld

        # 2. Build the per-instance Object linked to the shared datablock.
        loc = list(obj.get("location", [0.0, 0.0, 0.0]))
        scale = obj.get("scale", [0.0, 0.0, 0.0])
        if ltype == "AREA":
            # Push the light just below the panel's bottom face so the GLB body
            # doesn't occlude the downward-emitted light.
            sz = float(scale[2]) if len(scale) >= 3 else 0.0
            loc[2] = loc[2] - sz * 0.5 - 0.05

        obj_name = f"Class_Light_{obj['id']}"
        existing = bpy.data.objects.get(obj_name)
        if existing is not None:
            # Idempotent re-run: rebind to the (possibly new) shared datablock.
            old_data = existing.data
            existing.data = ld
            if old_data is not None and old_data.users == 0 and old_data is not ld:
                bpy.data.lights.remove(old_data, do_unlink=True)
            existing.location = mathutils.Vector(tuple(loc))
            blight = existing
        else:
            blight = bpy.data.objects.new(name=obj_name, object_data=ld)
            blight.location = mathutils.Vector(tuple(loc))
            blight[FIXED_ENERGY_TAG] = True
            col.objects.link(blight)

        if ltype == "AREA":
            blight.rotation_euler = mathutils.Euler((math.pi, 0.0, 0.0), "XYZ")
            # 3. Disable self-shadow on the GLB body so it can't shadow its own light.
            _disable_self_shadow_on_mesh(obj["id"])

        # 4. Position-only parent: when scene-orchestra-run translates obj_<N>
        # via op-executor (ground_to_floor / snap_to_wall / move_object), the
        # light fixture follows. Rotation/scale stay independent so a panel
        # light always faces down and keeps its emitter size.
        parent_empty = bpy.data.objects.get(obj["id"])
        if parent_empty is not None:
            _attach_position_only_parent(blight, parent_empty)

        added += 1
        loc_round = tuple(round(c, 2) for c in loc)
        sharing = "" if canon_id == obj_int_id else f"  (shares ClassLightData_{canon_id})"
        print(f"[CLASS_LIGHT] {obj_name}  cls='{cls}'  type={ltype}  energy={FIXED_NEW_LIGHT_ENERGY} W (Rule 1)  loc={loc_round}{sharing}")

    if added == 0:
        print("[CLASS_LIGHT] no objects matched a light/lamp class.")
    else:
        n_unique_data = len(shared_data)
        print(f"[CLASS_LIGHT] added {added} fixture(s) backed by {n_unique_data} shared LIGHT datablock(s).")
    return added


# ---------------------------------------------------------------------------
# Phase 8 — render settings
# ---------------------------------------------------------------------------

def apply_render_settings(args, params):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # Sampling + denoise
    scene.cycles.samples = args.samples
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    # Bounces
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 6
    scene.cycles.glossy_bounces = 6
    scene.cycles.sample_clamp_indirect = args.clamp_indirect
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    # AgX
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        try:
            scene.view_settings.look = "Medium High Contrast"
        except Exception as e:
            print(f"[WARN] AgX look not set: {e}")
    scene.view_settings.exposure = params["exposure"]
    # Resolution + format
    scene.render.resolution_x = args.resolution_x
    scene.render.resolution_y = args.resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    print(f"[RENDER] Cycles {args.samples}spp, AgX, OIDN, "
          f"{args.resolution_x}x{args.resolution_y}, clamp={args.clamp_indirect}")


# ---------------------------------------------------------------------------
# Phase 9 — compositor
# ---------------------------------------------------------------------------

def build_compositor(preview_path):
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()

    rl    = nt.nodes.new("CompositorNodeRLayers");   rl.location    = (-400, 0)
    glare = nt.nodes.new("CompositorNodeGlare");     glare.location = (-150, 0)
    lens  = nt.nodes.new("CompositorNodeLensdist");  lens.location  = (100, 0)
    comp  = nt.nodes.new("CompositorNodeComposite"); comp.location  = (350, 0)
    fout  = nt.nodes.new("CompositorNodeOutputFile"); fout.location = (350, -180)

    glare.glare_type = "FOG_GLOW"
    glare.mix = -0.92
    glare.threshold = 1.0
    glare.size = 6

    # Use correct input name: "Distortion" not "Distort"
    for iname in ("Distortion", "Distort"):
        if iname in lens.inputs:
            lens.inputs[iname].default_value = 0.006
            break
    if "Dispersion" in lens.inputs:
        lens.inputs["Dispersion"].default_value = 0.002

    fout.base_path = os.path.dirname(preview_path) or "."
    fout.file_slots[0].path = os.path.splitext(os.path.basename(preview_path))[0]
    fout.format.file_format = "PNG"
    fout.format.color_mode = "RGBA"
    fout.format.color_depth = "16"

    nt.links.new(rl.outputs["Image"],    glare.inputs["Image"])
    nt.links.new(glare.outputs["Image"], lens.inputs["Image"])
    nt.links.new(lens.outputs["Image"],  comp.inputs["Image"])
    nt.links.new(lens.outputs["Image"],  fout.inputs["Image"])
    print(f"[COMPOSITOR] RenderLayers -> Glare(FOG_GLOW) -> LensDist -> Composite")
    print(f"[COMPOSITOR] FileOutput -> {preview_path}")


# ---------------------------------------------------------------------------
# Phase 10 — post-config verify
# ---------------------------------------------------------------------------

def print_verify(args, params, modified, skipped, output_path, preview_path):
    scene = bpy.context.scene
    print("\n" + "=" * 60)
    print("POST-CONFIG VERIFICATION")
    print("=" * 60)

    # World nodes
    w = scene.world
    if w and w.use_nodes:
        print("\n[WORLD NODES]")
        for n in w.node_tree.nodes:
            print(f"  {n.name} ({n.type})")
        for lk in w.node_tree.links:
            print(f"    {lk.from_node.name}.{lk.from_socket.name} -> "
                  f"{lk.to_node.name}.{lk.to_socket.name}")

    print(f"\n[VIEW]  view_transform={scene.view_settings.view_transform}  "
          f"look={scene.view_settings.look}  exposure={scene.view_settings.exposure}")
    print(f"[CYCLES] samples={scene.cycles.samples}  "
          f"bounces(max/diff/gls)={scene.cycles.max_bounces}/"
          f"{scene.cycles.diffuse_bounces}/{scene.cycles.glossy_bounces}  "
          f"clamp_indirect={scene.cycles.sample_clamp_indirect}")

    print("\n[LIGHTS]")
    # Interior-only roster. Show every LIGHT in the scene so class-driven
    # fixtures and the Fallback_Light (Rule 2) are visible in the verify block.
    light_objs = sorted(
        (o for o in bpy.data.objects if o.type == "LIGHT"),
        key=lambda o: o.name,
    )
    if not light_objs:
        print("  (no LIGHT objects in scene — Rule 2 fallback should have fired)")
    for lo in light_objs:
        ld = lo.data
        print(f"  {lo.name:24s} type={ld.type:6s} energy={ld.energy:.3f} "
              f"color={tuple(round(c, 3) for c in ld.color)}")

    print(f"\n[OUTPUTS]  blend={output_path}  preview={preview_path}")
    print(f"[MATERIALS] modified={sorted(modified)}")
    print(f"[MATERIALS] protected/skipped={sorted(set(skipped))}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    try:
        args = parse_args(argv)
        print(f"\n[INIT] mood={args.mood}, scale={args.scale}, samples={args.samples}")

        output_path, preview_path = resolve_paths(args)
        print(f"[INIT] output={output_path}  preview={preview_path}")

        # Apply director's stage2_plan.json hints to args (CLI flags win).
        plan = _load_plan(args, output_path)
        _apply_plan_to_args(args, plan)

        scale = detect_scale(args)
        params = resolve_mood_params(args, scale)
        protected, touchable = scan_materials()
        rebuild_world(params)
        modified, skipped = rewrite_stage_materials(args, touchable, protected)
        build_light_rig(params, scale)
        build_class_lights(output_path)
        apply_render_settings(args, params)

        if not args.no_compositor:
            build_compositor(preview_path)
        else:
            print("[COMPOSITOR] Skipped (--no-compositor).")

        print(f"[SAVE] {output_path}")
        bpy.ops.wm.save_as_mainfile(filepath=output_path)

        bpy.context.scene.render.filepath = preview_path
        print(f"[RENDER] {preview_path}")
        bpy.ops.render.render(write_still=True)

        if getattr(args, "align_brightness", False):
            ref_path = args.reference_image
            if ref_path is None:
                # Default: <scene_dir>/image.png.
                # The .blend now lives in <scene_dir>/blend/; go up one level to scene_dir.
                blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
                candidate = blend_dir / "image.png"
                if not candidate.exists():
                    candidate = blend_dir.parent / "image.png"
                ref_path = candidate
            ref_path = Path(ref_path)
            if not ref_path.exists():
                print(f"[ALIGN] reference image not found: {ref_path} — skipping alignment")
            else:
                # Safety check: warn if the scene camera is not in a viable interior
                # position.  This is advisory — alignment proceeds regardless.
                # blender_scene.json lives at <scene_dir>/json/ (canonical) or
                # <scene_dir>/ (legacy top-level).
                blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
                _json_canonical = blend_dir.parent / "json" / "blender_scene.json"
                _json_legacy = blend_dir.parent / "blender_scene.json"
                if _json_canonical.exists():
                    _scene_json_for_check = str(_json_canonical)
                elif _json_legacy.exists():
                    print("[enhance_env] [legacy-path] reading blender_scene.json from top-level")
                    _scene_json_for_check = str(_json_legacy)
                else:
                    _scene_json_for_check = str(_json_canonical)  # will simply be missing; check handles that
                _check_camera_interior_position(
                    _scene_json_for_check if _scene_json_for_check else None
                )

                print(f"[ALIGN] Reference: {ref_path}")
                result = align_brightness_to_reference(
                    bpy.context.scene, Path(preview_path), ref_path,
                    tolerance=args.brightness_tolerance,
                    max_iters=args.brightness_max_iters,
                )
                print(f"[ALIGN] target_lum={result['target']:.4f} "
                      f"final_lum={result['final_lum']:.4f} delta={result['final_delta']:.4f} "
                      f"converged={result['converged']} cumulative_scale={result['cumulative_scale']:.3f} "
                      f"cumulative_scale_inverse={result['cumulative_scale_inverse']:.4f}")
                # Persist log: prefer --brightness-log if given, else next to the preview.
                if getattr(args, "brightness_log", None):
                    log_path = Path(args.brightness_log)
                else:
                    log_path = Path(preview_path).with_name("brightness_align_log.json")
                log_path.write_text(json.dumps(result, indent=2))
                print(f"[ALIGN] log: {log_path}")

                # Re-save .blend so post-alignment light energies are persisted.
                # Without this, subsequent renders (e.g. render_multi_view.py) would
                # re-open the .blend and see the pre-alignment energies.
                try:
                    bpy.ops.wm.save_as_mainfile(filepath=output_path)
                    print(f"[SAVE-POST-ALIGN] {output_path}")
                except Exception as _save_exc:
                    print(f"[SAVE-POST-ALIGN] WARNING: re-save failed: {_save_exc}")

        print_verify(args, params, modified, skipped, output_path, preview_path)

        # Export env state to blender_scene.json so JSON <-> .blend remains 1:1.
        # blender_scene.json lives at <scene_dir>/json/ (canonical) or
        # <scene_dir>/ (legacy top-level).
        try:
            import sys as _sys
            import os as _os
            _scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
            if _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            from export_env_to_json import export_env_to_json
            _blend_parent = _os.path.dirname(output_path)
            _scene_dir = _os.path.dirname(_blend_parent)
            _scene_json_canonical = _os.path.join(_scene_dir, "json", "blender_scene.json")
            _scene_json_legacy = _os.path.join(_scene_dir, "blender_scene.json")
            if _os.path.exists(_scene_json_canonical):
                scene_json = _scene_json_canonical
            elif _os.path.exists(_scene_json_legacy):
                print(f"[enhance_env] [legacy-path] reading blender_scene.json from top-level")
                scene_json = _scene_json_legacy
            else:
                scene_json = _scene_json_canonical  # will be reported as not found below
            if _os.path.exists(scene_json):
                summary = export_env_to_json(scene_json, overwrite_if_present=True, include_compositor=True)
                print(f"[enhance_env] Exported env to JSON: {summary}")
            else:
                print(f"[enhance_env] WARN: blender_scene.json not found at {scene_json} — skipping env export to JSON")
        except Exception as e:
            print(f"[enhance_env] WARN: env-to-JSON export failed: {e}")

        print("[DONE] enhance_env completed successfully.")
        sys.exit(0)

    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
