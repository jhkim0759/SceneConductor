#!/usr/bin/env python3
"""
layout_to_blender_json.py

Converts a trimesh-coordinate scene layout (layout_prediction.json) into a
Blender-ready scene descriptor (blender_scene.json).

Coordinate conversion: trimesh (x, y, z) -> Blender (-x, z, y)
Permutation matrix P = [[-1,0,0],[0,0,1],[0,1,0]], which is self-inverse.
  - vector: v_blender = P @ v_trimesh
  - rotation: R_blender = P @ R_trimesh @ P
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# ── DIRECTORYS.yaml (canonical machine-specific paths) ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text(encoding="utf-8"))

# Defaults for the optional chain trigger (--build-blend).
# OS-aware blender binary lookup: $BLENDER env var > DIRECTORYS.yaml platform key
# > DIRECTORYS.yaml legacy blender_bin.
def _pick_blender_bin(d):
    env = os.environ.get("BLENDER")
    if env:
        return env
    if sys.platform.startswith("win") and d.get("blender_bin_windows"):
        return d["blender_bin_windows"]
    if sys.platform == "darwin" and d.get("blender_bin_macos"):
        return d["blender_bin_macos"]
    return d.get("blender_bin", "blender")
DEFAULT_BLENDER_BIN = _pick_blender_bin(_DIRS)
# build.py lives next to this file; project root is 4 levels up from src/.
DEFAULT_BUILD_PY = str(Path(__file__).resolve().parent / "build.py")

# -----------------------------------------------------------------------------
# World-scale priors: typical real-world longest-side dimension in METERS per
# object class. Drives --world-scale auto. Values are intentionally rough —
# the median across classified objects smooths out individual errors.
# -----------------------------------------------------------------------------
CLASS_SIZE_PRIORS_M = {
    "sofa": 1.9, "armchair": 0.9, "chair": 0.5, "dining_chair": 0.5, "stool": 0.45,
    "coffee_table": 1.0, "table": 1.5, "desk": 1.4, "dining_table": 1.8,
    "pool_table": 2.4,
    "cushion": 0.5, "pillow": 0.5,
    "curtain": 1.8, "window": 1.2,
    "picture_frame": 0.6, "wall_art": 0.6, "mirror": 0.8,
    "bed": 2.0, "nightstand": 0.5,
    "dresser": 1.3, "wardrobe": 1.8,
    "lamp": 0.4, "floor_lamp": 1.5, "table_lamp": 0.45,
    "plant": 0.6, "vase": 0.3,
    "rug": 2.0, "carpet": 2.5,
    "tv": 1.2, "monitor": 0.6, "laptop": 0.35,
    "bookshelf": 1.8, "cabinet": 1.0, "shelf": 1.0,
    "refrigerator": 1.8, "stove": 0.7, "sink": 0.6,
    "toilet": 0.7, "bathtub": 1.7, "shower": 1.8,
    "door": 2.1, "clock": 0.4,
}


def load_class_map(path):
    """Load a class mapping into the canonical {"obj_<N>": "<class_name>"} form.

    Supports the two on-disk schemas the Stage 1 pipeline actually produces:

      1) mask_attribute.json — {"objects": {"<id>": {"class": "<name>", ...}, ...}, ...}
         Canonical output of mask_attribute.init_attributes; "<id>" is a 1-indexed
         integer string with possible gaps after merges.

      2) object_class.json — flat {"<id>": "<name>", ...} with the same id semantics.
         Legacy obj_<N> string keys are also accepted.

    Output keys are always normalised to "obj_<int_id>" so downstream lookups
    against object ids from layout_prediction.json (also "obj_<N>") match. The
    previous dict-branch returned raw keys ("1", "2", …) which silently failed
    `class_map.get("obj_1")` lookups in compute_world_scale → world_scale_factor
    fell through to "no_classes" → k=1.0. This implementation fixes that.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Unrecognized class map format (expected dict): {path}")

    # Case 1: mask_attribute.json — nested under "objects" with per-id "class".
    objs = data.get("objects")
    if isinstance(objs, dict) and objs and any(
        isinstance(v, dict) and "class" in v for v in objs.values()
    ):
        out: dict[str, str] = {}
        for k, v in objs.items():
            if isinstance(v, dict) and "class" in v:
                out[f"obj_{int(k)}"] = str(v["class"])
        return out

    # Case 2: object_class.json — flat {"<id>": "<class>"} (or legacy obj_<N> keys).
    out = {}
    for k, v in data.items():
        if not isinstance(v, str):
            continue  # skip nested / non-string-valued entries (defensive)
        key = str(k)
        if key.startswith("obj_"):
            out[key] = v
        else:
            try:
                out[f"obj_{int(key)}"] = v
            except (TypeError, ValueError):
                continue  # non-integer key in a class file — skip rather than fail
    if out:
        return out

    raise ValueError(f"Unrecognized class map format: {path}")


def find_class_file(input_path, override=None):
    """Locate a class mapping file. Preference order:
       1) explicit --object-classes override
       2) sibling mask_attribute.json (list of {mask_idx, object_class})
       3) sibling object_class.json  (flat dict)
    Returns the path or None.
    """
    if override:
        return override if os.path.isfile(override) else None
    scene_dir = os.path.dirname(input_path)
    # Prefer the plural list-format / obj_N-keyed files (the ones load_class_map
    # parses into usable obj_N entries); fall back to the singular variants.
    for name in ("mask_attributes.json", "object_classes.json",
                 "mask_attribute.json", "object_class.json"):
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            return p
    return None


def compute_world_scale(object_ids, scales, class_map):
    """Pick a uniform world-scale factor `k` so that multiplying every
    per-object scale by k produces approximately real-world-sized objects.

    For each object whose class appears in CLASS_SIZE_PRIORS_M, take the
    ratio `target_m / predicted_scale` and return the median. Robust to
    individual mispredictions — priors only need to be roughly right.

    Returns (k, detail_str, per_object_factors).
    Returns (None, reason, []) when no matches are found.
    """
    factors = []
    for oid, s in zip(object_ids, scales):
        if oid == "floor" or s <= 0:
            continue
        cls = class_map.get(oid)
        if not cls:
            continue
        target_m = CLASS_SIZE_PRIORS_M.get(cls.lower())
        if target_m is None:
            continue
        factors.append(target_m / s)

    if not factors:
        return None, "no class priors matched", []

    # A4: outlier rejection. The class prior dictionary is small and
    # frequently mismatches niche/kindergarten/small-furniture classes
    # (e.g., "toy_shelf" -> "bookshelf" prior 1.8 m gives ratios up to 20×
    # while a correctly-matched "rug" gives 5×). A plain median is fooled
    # when ≥ half of the matches are biased the same way. We use MAD-based
    # filtering: drop ratios whose distance from the median exceeds 3·MAD,
    # then take the median of the survivors. Falls back to plain median if
    # MAD is degenerate (all factors equal).
    factors_arr = np.array(factors, dtype=float)
    med = float(np.median(factors_arr))
    mad = float(np.median(np.abs(factors_arr - med)))
    if mad > 1e-6:
        keep_mask = np.abs(factors_arr - med) <= 3.0 * mad
        kept = factors_arr[keep_mask].tolist()
    else:
        kept = factors_arr.tolist()
    if not kept:
        kept = factors  # fallback: should never trigger but be safe
    k = float(np.median(kept))
    detail = (
        f"MAD-trimmed median of {len(kept)}/{len(factors)} class priors "
        f"(raw_med={med:.3f}, mad={mad:.3f})"
    )
    return k, detail, factors


def normalize_deg_180(d):
    """Normalize an angle in degrees to (-180, 180]."""
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d

# ---------------------------------------------------------------------------
# Permutation matrix (trimesh <-> Blender, self-inverse)
# ---------------------------------------------------------------------------
P = np.array([
    [-1,  0,  0],
    [ 0,  0,  1],
    [ 0,  1,  0],
], dtype=float)


def convert_vec(v):
    """Apply P to a 3-vector."""
    return (P @ np.array(v, dtype=float)).tolist()


def convert_rotation(R_tm):
    """Convert a rotation (either 3x3 matrix or 4-element quaternion) to Blender space.
    
    Quaternions are in (x, y, z, w) format (scipy convention).
    Rotation matrices are 3x3.
    """
    R_tm = np.array(R_tm, dtype=float)
    
    # If it's a quaternion (4 elements), convert to rotation matrix first
    if R_tm.shape == (4,) or (isinstance(R_tm, (list, tuple)) and len(R_tm) == 4):
        try:
            from scipy.spatial.transform import Rotation
            # Quaternion in (x, y, z, w) format (scipy convention)
            quat = Rotation.from_quat(R_tm)
            R = quat.as_matrix()
        except ImportError:
            # Fallback: manual quaternion to matrix conversion
            x, y, z, w = R_tm[0], R_tm[1], R_tm[2], R_tm[3]
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=float)
    else:
        R = R_tm.reshape(3, 3)
    
    return P @ R @ P


def rotation_matrix_to_euler_xyz(R):
    """
    Extract Euler XYZ (intrinsic) angles in radians from a 3x3 rotation matrix.
    Uses scipy.spatial.transform.Rotation for robustness. Falls back to a
    hand-rolled extraction (with gimbal guard) if scipy is unavailable.

    Returns (rx, ry, rz) in radians.
    """
    try:
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(R)
        return r.as_euler('xyz', degrees=False).tolist()
    except ImportError:
        pass

    # Hand-rolled XYZ Euler extraction (R = Rx @ Ry @ Rz)
    # R[0,2] = sin(ry)
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    ry = math.asin(sy)

    if abs(sy) < 0.9999:
        rx = math.atan2(-R[1, 2], R[2, 2])
        rz = math.atan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock: ry = ±90°
        rx = math.atan2(R[2, 1], R[1, 1])
        rz = 0.0

    return [rx, ry, rz]


def euler_deg(euler_rad):
    return [math.degrees(a) for a in euler_rad]


def parse_scale(raw):
    """
    Scale entries can be:
      - a bare scalar (int or float)
      - a 1-element list [v]
      - a nested list [[v]]
      - a 3-element list [x, y, z] (uniform scale, all equal)
    Always returns a single float.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], list):
            return float(raw[0][0])
        if len(raw) == 1:
            return float(raw[0])
        # 3-element scale [x, y, z] — assume uniform (all equal)
        if len(raw) == 3:
            return float(raw[0])
    raise ValueError(f"Unexpected scale format: {raw!r}")


def main():
    default_input = (
        "./datas/scenes/coco_val2017_000000000139/layout_prediction.json"
    )
    default_output = (
        "./datas/scenes/coco_val2017_000000000139/blender_scene.json"
    )

    parser = argparse.ArgumentParser(
        description="Convert layout_prediction.json (trimesh) to blender_scene.json"
    )
    parser.add_argument(
        "--scene-dir",
        default=None,
        help=(
            "Unified entry point. When set, --input is derived as "
            "<scene_dir>/inputs/layout_prediction.json, --output as "
            "<scene_dir>/json/blender_scene.json, and --build-blend as "
            "<scene_dir>/blend/blender_scene.blend (auto-chains the build). "
            "Requires layout_prediction.json and either image.png or image.jpg "
            "to exist inside the scene directory."
        ),
    )
    parser.add_argument("--input",  default=default_input,  help="Path to layout_prediction.json")
    parser.add_argument("--output", default=default_output, help="Path to output blender_scene.json")
    parser.add_argument(
        "--object-classes",
        default=None,
        help=(
            "Explicit path to a class-mapping file. Supported formats: "
            "mask_attribute.json (list of {mask_idx, object_class, ...}) or "
            "object_class.json (flat dict of {obj_N: class_name}). "
            "If omitted, looks for mask_attribute.json then object_class.json "
            "next to --input."
        ),
    )
    parser.add_argument(
        "--build-blend",
        default=None,
        help=(
            "Optional .blend output path. When set, after writing the JSON "
            "this script invokes the stage2-sub-pointmap-to-separable-stage skill's build.py "
            "to produce the .blend in one pass. Requires --blender-bin and "
            "--build-py to be correct (defaults point at the installed skill)."
        ),
    )
    parser.add_argument("--blender-bin", default=DEFAULT_BLENDER_BIN, help="Path to the Blender binary.")
    parser.add_argument("--build-py",    default=DEFAULT_BUILD_PY,    help="Path to stage2-sub-pointmap-to-separable-stage/src/build.py.")
    parser.add_argument(
        "--world-scale",
        default="auto",
        metavar="{auto|<float>|none}",
        help=(
            "World-scale mode. "
            "'auto' (default): use class priors to compute a uniform factor k; "
            "if no classes are provided or no class matches a prior, k is set "
            "to 1.0 (no rescale). "
            "a float: use that value directly as k. "
            "'none': k=1 (explicit opt-out)."
        ),
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=0.1,
        help=(
            "Lower sanity bound for k in auto mode (default 0.1). Only rejects "
            "absurd estimates (object predicted >10x too large); legitimate "
            "estimates pass through. If priors compute a k below this, it is "
            "clamped up and a warning is logged. "
            "Manual --world-scale <float> bypasses this clamp."
        ),
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=20.0,
        help=(
            "Upper sanity bound for k in auto mode (default 20.0). Wide enough "
            "to admit legitimate large estimates (e.g. child-scale furniture "
            "scenes yield k~10); only rejects absurd values (object predicted "
            ">20x too small, almost certainly a unit error). If priors compute "
            "a k above this, it is clamped down and a warning is logged. "
            "Manual --world-scale <float> bypasses this clamp."
        ),
    )
    args = parser.parse_args()

    # --scene-dir: single-input mode. Derives all three paths from the folder
    # and asserts the image precondition so downstream skills (env-enhance,
    # stage2-sub-pointmap-to-separable-stage, etc.) can rely on its presence.
    if args.scene_dir:
        scene_dir = os.path.abspath(args.scene_dir)
        if not os.path.isdir(scene_dir):
            raise SystemExit(f"--scene-dir is not a directory: {scene_dir}")
        # Try inputs/ first, then top-level for legacy scenes.
        _lp_new = os.path.join(scene_dir, "inputs", "layout_prediction.json")
        _lp_legacy = os.path.join(scene_dir, "layout_prediction.json")
        if os.path.isfile(_lp_new):
            layout_path = _lp_new
        elif os.path.isfile(_lp_legacy):
            print(f"[legacy-path] reading from {_lp_legacy}, expected {_lp_new}", file=sys.stderr)
            layout_path = _lp_legacy
        else:
            raise SystemExit(
                f"Missing layout_prediction.json inside --scene-dir: tried {_lp_new} and {_lp_legacy}"
            )
        image_candidates = [
            os.path.join(scene_dir, "image.png"),
            os.path.join(scene_dir, "image.jpg"),
            os.path.join(scene_dir, "image.jpeg"),
        ]
        image_found = next((p for p in image_candidates if os.path.isfile(p)), None)
        if image_found is None:
            raise SystemExit(
                f"--scene-dir must contain image.png or image.jpg (looked for "
                f"image.png, image.jpg, image.jpeg) in {scene_dir}"
            )
        print(f"[scene-dir] layout={layout_path}", file=sys.stderr)
        print(f"[scene-dir] image={image_found}", file=sys.stderr)

        args.input = layout_path
        # blender_scene.json is canonical at json/; blender_scene.blend at blend/.
        args.output = os.path.join(scene_dir, "json", "blender_scene.json")
        if not args.build_blend:
            args.build_blend = os.path.join(scene_dir, "blend", "blender_scene.blend")

    input_path  = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        src = json.load(f)

    # Anchor for scene-relative path rewrites: the scene_dir explicitly passed
    # via --scene-dir, else the parent of inputs/ derived from --input.
    scene_dir_anchor = None
    if args.scene_dir:
        scene_dir_anchor = os.path.abspath(args.scene_dir)
    else:
        parent = os.path.dirname(input_path)
        if os.path.basename(parent) == "inputs":
            scene_dir_anchor = os.path.dirname(parent)

    # ------------------------------------------------------------------
    # scene-level fields
    # ------------------------------------------------------------------
    shifted_center_blender = convert_vec(src["shifted_center"])
    shifted_scale = float(src["shifted_scale"])

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    # Camera location is ALWAYS (0, 0, 0): GLB objects are authored relative
    # to camera origin, so the Blender camera is always placed at the origin.
    # c2w_extrinsic translation is intentionally ignored here.
    cam_loc_bl = [0.0, 0.0, 0.0]

    # blender_camera_rotation is already in Blender degrees.
    # Apply one user-authored orientation fix:
    #   rz += 180  (roll 180° to correct upside-down camera), normalized to (-180, 180]
    # (rx is left as-is; negating rx composes with rz+180 to point at the void,
    #  see /tmp/cam_{A,B,C,D}.png diagnostic — only rz+180 yields a correctly
    #  oriented render.)
    _raw = src["blender_camera_rotation"]
    _rx =  float(_raw[0])
    _ry =  float(_raw[1])
    _rz =  normalize_deg_180(float(_raw[2]) + 180.0)
    cam_rot_deg = [_rx, _ry, _rz]
    cam_rot_rad = [math.radians(d) for d in cam_rot_deg]

    # Infer resolution from principal point: W = 2*cx, H = 2*cy
    K = src["intrinsics"]
    cx = K[0][2]
    cy = K[1][2]
    resolution = [int(round(2 * cx)), int(round(2 * cy))]

    camera = {
        "location":           cam_loc_bl,
        "rotation_euler":     cam_rot_rad,
        "rotation_euler_deg": cam_rot_deg,
        "lens":               float(src["blender_focal_length"]),
        "sensor_width":       36.0,
        "sensor_fit":         "HORIZONTAL",
        "resolution":         resolution,
        "clip_start":         0.01,
        "clip_end":           100.0,
    }

    # ------------------------------------------------------------------
    # objects + floor (separated)
    # ------------------------------------------------------------------
    translations = src["translation"]
    rotations    = src["rotation"]
    scales_raw   = src["scale"]
    meshes       = src["meshes"]
    object_ids   = src["object_id"]

    # Load class map once, up front. Used both to (a) attach a `class` field to
    # every object entry and (b) drive auto world-scale below.  Always attempt
    # the load regardless of --world-scale mode so classes survive even when
    # the user opts out of class-driven rescaling.
    class_map = {}
    class_map_source = None
    cls_path = find_class_file(input_path, override=args.object_classes)
    if cls_path:
        class_map = load_class_map(cls_path)
        class_map_source = cls_path

    # Rewrite mesh paths so they point at local files under <scene_dir>/inputs/.
    # The layout-predictor records absolute paths from whichever machine ran it
    # (e.g. /home/user/.../inputs/object/1.glb) — those won't exist on another
    # host. We prefer the path as-given when it's a real file; otherwise we
    # take just the basename and look for it under <scene_dir>/inputs/.
    def _remap_mesh_path(p: str) -> str:
        from pathlib import Path as _P
        pp = _P(p)
        if pp.is_file():
            return p
        if scene_dir_anchor is not None:
            inputs_root = _P(scene_dir_anchor) / "inputs"
            # Search the canonical sub-locations for this basename.
            for sub in ("object", "", "floor"):
                cand = inputs_root / sub / pp.name if sub else inputs_root / pp.name
                if cand.is_file():
                    return str(cand)
            # Last resort: rglob for the basename (cheap — inputs/ is small).
            try:
                hit = next(inputs_root.rglob(pp.name))
                return str(hit)
            except (StopIteration, OSError):
                pass
        return p
    meshes = [_remap_mesh_path(m) for m in meshes]

    objects = []
    for i, oid in enumerate(object_ids):
        loc_bl  = convert_vec(translations[i])
        R_bl    = convert_rotation(rotations[i])
        euler   = rotation_matrix_to_euler_xyz(R_bl)
        e_deg   = euler_deg(euler)
        s       = parse_scale(scales_raw[i])
        scale   = [s, s, s]

        # Per-object Z+180 fix: every scene object is rotated 180° about
        # world Z. Floor is exempt (it's flat — a Z rotation is a no-op).
        # Matches the camera's rz+180 fix so object facings align with the
        # corrected camera framing.
        if oid != "floor":
            e_deg[2] = normalize_deg_180(e_deg[2] + 180.0)
            euler[2] = math.radians(e_deg[2])

        entry = {
            "id":                oid,
            "mesh_path":         meshes[i],
            "location":          loc_bl,
            "rotation_euler":    euler,
            "rotation_euler_deg": e_deg,
            "scale":             scale,
        }

        if oid == "floor":
            # Legacy: layout_prediction.json carries a procedural floor.obj that
            # Stage 3 supersedes with the polygon-driven `Stage/Floor` mesh.
            # We deliberately drop it here so build.py doesn't import a second
            # floor (which would Z-fight with the stage Floor). See schema §2.
            continue
        entry["class"] = class_map.get(oid)
        objects.append(entry)

    # ------------------------------------------------------------------
    # world scaling — multiply every world-space length by k so the scene
    # is in real-world meters. Rotations are invariant and untouched.
    # Rule: if no class info is available or no class matches a prior,
    # k = 1.0 (no rescale). Only class_priors or an explicit override
    # can produce a non-identity k.
    # ------------------------------------------------------------------
    ws_arg = args.world_scale.strip()

    # Defaults for both auto and non-auto branches so meta-block references
    # are always defined.
    k_raw = None
    k_clamped_flag = False

    if ws_arg == "none":
        k = 1.0
        world_scale_method = "identity"
        k_raw = 1.0
    elif ws_arg == "auto":
        all_ids    = [e["id"]      for e in objects]
        all_scales = [e["scale"][0] for e in objects]

        # class_map already loaded above (used for per-object .class field).
        k_result, k_detail, _ = compute_world_scale(all_ids, all_scales, class_map)
        if k_result is not None:
            k = k_result
            world_scale_method = "class_priors"
        else:
            k = 1.0
            world_scale_method = "no_classes"

        # Clamp k to [k_min, k_max] in auto mode to prevent runaway shrink/grow
        # when priors disagree wildly with the layout-predictor's raw scale.
        # Manual --world-scale bypasses this clamp (handled in the else branch below).
        k_raw = k
        k_clamped_flag = False
        if world_scale_method == "class_priors":
            if k < args.k_min:
                k = args.k_min
                k_clamped_flag = True
                k_detail = f"{k_detail}; CLAMPED UP from {k_raw:.4f} (below k_min={args.k_min})"
            elif k > args.k_max:
                k = args.k_max
                k_clamped_flag = True
                k_detail = f"{k_detail}; CLAMPED DOWN from {k_raw:.4f} (above k_max={args.k_max})"
            if k_clamped_flag:
                world_scale_method = "class_priors_clamped"
                print(
                    f"[world-scale] WARNING: raw k={k_raw:.4f} clamped to {k:.4f} "
                    f"(bounds=[{args.k_min}, {args.k_max}]). The layout predictor's "
                    f"raw scale disagreed strongly with class priors — review the "
                    f"scene before downstream stages.",
                    file=sys.stderr,
                )

        src_msg = f" source={class_map_source}" if class_map_source else " source=<none>"
        print(
            f"[world-scale] method={world_scale_method} k={k:.6f} k_raw={k_raw:.6f} "
            f"bounds=[{args.k_min},{args.k_max}] ({k_detail}){src_msg}",
            file=sys.stderr,
        )
    else:
        # Manual float override (bypasses k_min/k_max clamp by design).
        try:
            k = float(ws_arg)
        except ValueError:
            raise SystemExit(
                f"--world-scale must be 'auto', 'none', or a float; got: {ws_arg!r}"
            )
        world_scale_method = "manual"
        k_raw = k
        print(f"[world-scale] method={world_scale_method} k={k:.6f}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Vision scale gate — only in auto mode (class_priors / class_priors_clamped).
    # A vision agent may write <scene_dir>/json/stage2_plan.json with a
    # `scale_prior` block giving an expected horizontal room footprint range.
    # If the current (class-prior) k would imply a footprint outside that range,
    # we GATE k so the scene lands at a plausible size. This runs AFTER k is
    # computed + outer-clamped but BEFORE the apply block below, because it
    # needs the RAW (pre-scale) object locations.
    # ------------------------------------------------------------------
    world_scale_vision_gate_applied = False
    world_scale_vision_prior = None
    world_scale_before_gate = None
    if world_scale_method in ("class_priors", "class_priors_clamped") and scene_dir_anchor is not None:
        plan_path = os.path.join(scene_dir_anchor, "json", "stage2_plan.json")
        scale_prior = None
        if os.path.isfile(plan_path):
            try:
                with open(plan_path, "r", encoding="utf-8") as f:
                    _plan = json.load(f)
                if isinstance(_plan, dict):
                    scale_prior = _plan.get("scale_prior")
            except (OSError, ValueError):
                scale_prior = None  # missing/unreadable/not JSON -> skip gate silently

        # Validate the prior. Skip gate (fallback) on any malformed field.
        footprint = None
        if isinstance(scale_prior, dict):
            conf = scale_prior.get("confidence")
            fp = scale_prior.get("expected_room_footprint_m")
            if (
                isinstance(conf, (int, float))
                and conf >= 0.5
                and isinstance(fp, (list, tuple))
                and len(fp) == 2
                and all(isinstance(v, (int, float)) for v in fp)
                and 0 < fp[0] <= fp[1]
            ):
                footprint = (float(fp[0]), float(fp[1]))

        if footprint is not None and objects:
            lo, hi = footprint
            xs = [e["location"][0] for e in objects]
            ys = [e["location"][1] for e in objects]
            ext_x = max(xs) - min(xs)
            ext_y = max(ys) - min(ys)
            ext_raw = max(ext_x, ext_y)
            if ext_raw > 1e-6:
                implied = k * ext_raw
                if implied < lo:
                    k_gate = lo / ext_raw
                elif implied > hi:
                    k_gate = hi / ext_raw
                else:
                    k_gate = k
                # Re-apply the outer sanity clamp to the gated k.
                if k_gate < args.k_min:
                    k_gate = args.k_min
                elif k_gate > args.k_max:
                    k_gate = args.k_max
                # Did the gate actually move k?
                if abs(k_gate - k) > 1e-6 * max(abs(k), 1e-9):
                    world_scale_before_gate = k
                    print(
                        f"[world-scale] vision-gate: k {k:.4f} -> {k_gate:.4f} "
                        f"(implied footprint {implied:.2f}m vs prior [{lo},{hi}]m, "
                        f"scene_scale_class={scale_prior.get('scene_scale_class')}, "
                        f"confidence={scale_prior.get('confidence')})",
                        file=sys.stderr,
                    )
                    k = k_gate
                    world_scale_method = "class_priors_vision_gated"
                    world_scale_vision_gate_applied = True
                else:
                    print(
                        "[world-scale] vision-gate: k within prior range, unchanged",
                        file=sys.stderr,
                    )
                world_scale_vision_prior = scale_prior

    if k != 1.0:
        def _scale_loc(loc):
            return [v * k for v in loc]

        def _scale_scale(sc):
            return [v * k for v in sc]

        for e in objects:
            e["location"] = _scale_loc(e["location"])
            e["scale"]    = _scale_scale(e["scale"])

        shifted_center_blender = _scale_loc(shifted_center_blender)
        shifted_scale          = shifted_scale * k

    # ------------------------------------------------------------------
    # Post-rescale scene-extent sanity warning
    # Independent of which k path we took. Catches the "no class priors
    # matched, k=1, raw predictor scale was tiny" case where the clamp
    # above never fired but the scene is still unreasonably small/large.
    # The bounds here are deliberately wider than the per-object reasoning
    # above, since they refer to the entire scene extent (in meters).
    # ------------------------------------------------------------------
    EXTENT_MIN_M = 2.0    # smaller than this is implausible for any indoor room
    EXTENT_MAX_M = 25.0   # larger than this is implausible for a single room
    if objects:
        xs = [e["location"][0] for e in objects]
        ys = [e["location"][1] for e in objects]
        zs = [e["location"][2] for e in objects]
        ext_x = max(xs) - min(xs)
        ext_y = max(ys) - min(ys)
        ext_z = max(zs) - min(zs)
        scene_extent_m = max(ext_x, ext_y, ext_z)
        if scene_extent_m < EXTENT_MIN_M or scene_extent_m > EXTENT_MAX_M:
            print(
                f"[world-scale] WARNING: post-rescale scene extent = "
                f"{scene_extent_m:.2f} m (object-loc bbox: {ext_x:.2f} x {ext_y:.2f} x {ext_z:.2f}). "
                f"Outside plausible indoor range [{EXTENT_MIN_M}, {EXTENT_MAX_M}] m. "
                f"Method={world_scale_method} k={k:.4f}. "
                f"Consider re-running with --world-scale <float> based on a "
                f"manual size measurement.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # point_cloud block — required output (see schema §10). Locates
    # inputs/pointmap_xz.ply under scene_dir_anchor and emits the import
    # metadata so build.py imports `PointCloud_XZ` for Stage 3.
    # ------------------------------------------------------------------
    point_cloud_block = None
    if scene_dir_anchor is not None:
        for cand_rel in ("inputs/pointmap_xz.ply", "pointmap_xz.ply"):
            cand_abs = os.path.join(scene_dir_anchor, cand_rel)
            if os.path.isfile(cand_abs):
                point_cloud_block = {
                    "ply_path":            cand_rel.replace("\\", "/"),
                    "axis_remap":          "forward=Z,up=Y",
                    "decimate_ratio":      1.0,
                    "visible":             False,
                    "world_scale_applied": k,
                    "name":                "PointCloud_XZ",
                }
                break
        if point_cloud_block is None:
            print(
                "[point_cloud] WARNING: pointmap_xz.ply not found under "
                f"{scene_dir_anchor}/inputs/ — Stage 3 will fail until it is "
                "present. point_cloud block omitted.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # assemble output
    # ------------------------------------------------------------------
    out = {
        "meta": {
            "schema_version":     "2.0",
            "source":             input_path,
            "coordinate_system":  "blender",
            "conversion":         "trimesh(x,y,z) -> blender(-x,z,y)",
            "units":              "meters",
            "rotation_order":     "XYZ",
            "rotation_unit":      "radians",
            "camera_rotation_fix":  "rz -> rz + 180 (normalized to (-180,180])",
            "object_rotation_fix":  "objects[].rotation_euler[z] -> rz + 180 (normalized to (-180,180])",
            "world_scale_factor": k,
            "world_scale_method": world_scale_method,
            "world_scale_factor_raw":  k_raw,
            "world_scale_clamp_bounds": [args.k_min, args.k_max] if ws_arg == "auto" else None,
            "world_scale_clamped":     k_clamped_flag,
            "world_scale_vision_gate_applied": world_scale_vision_gate_applied,
            "world_scale_vision_prior":        world_scale_vision_prior,
            "world_scale_before_gate":         world_scale_before_gate,
            "class_map_source":   class_map_source,
        },
        "scene": {
            "shifted_center_blender": shifted_center_blender,
            "shifted_scale":          shifted_scale,
        },
        "camera": camera,
        "objects": objects,
    }
    if point_cloud_block is not None:
        out["point_cloud"] = point_cloud_block

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    pc_msg = ", point_cloud=yes" if point_cloud_block else ", point_cloud=no"
    print(
        f"Wrote {len(objects)} objects{pc_msg}, "
        f"camera lens={camera['lens']:.4f} mm -> {output_path}"
    )

    # ------------------------------------------------------------------
    # Chain trigger: invoke stage2-sub-pointmap-to-separable-stage if --build-blend given.
    # Runs Blender headless on the skill's build.py and exits non-zero
    # if the build fails, so upstream callers can detect failure.
    # ------------------------------------------------------------------
    if args.build_blend:
        blend_out = os.path.abspath(args.build_blend)
        os.makedirs(os.path.dirname(blend_out), exist_ok=True)
        # blender_bin may be either a full path or a bare PATH name like "blender".
        # Only verify existence when it looks like an absolute path.
        import shutil as _shutil
        if os.path.isabs(args.blender_bin):
            if not os.path.isfile(args.blender_bin):
                raise SystemExit(f"[chain] Blender binary not found: {args.blender_bin}")
        elif _shutil.which(args.blender_bin) is None:
            raise SystemExit(f"[chain] Blender binary not on PATH: {args.blender_bin}")
        if not os.path.isfile(args.build_py):
            raise SystemExit(f"[chain] build.py not found: {args.build_py}")
        cmd = [
            args.blender_bin, "--background",
            "--python", args.build_py, "--",
            "--input",  output_path,
            "--output", blend_out,
        ]
        print(f"[chain] -> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"[chain] build.py exited with code {rc}")
        print(f"[chain] built .blend -> {blend_out}")


if __name__ == "__main__":
    main()
