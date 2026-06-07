"""render_island_views.py — Blender -P script that renders a canonical-frame island.blend.

Produces a front perspective + top-down BEV view. For perspective, uses the
canonical-frame scene camera (M_inv @ C_world) to match the original photo's
viewpoint. For BEV, uses a computed orthographic top-down camera.

Usage (run with Blender, NOT plain python):

    blender -b island.blend -P render_island_views.py -- \\
        --out-dir <dir>            # required; output directory
        [--samples N]              # default 128
        [--persp-size W H]         # default 1024 768
        [--bev-size W H]           # default 1024 1024
        [--metadata <path>]        # optional; path to metadata.json for M_inv

Outputs (written flat to --out-dir):
    render_persp.png
    render_bev.png

This script does NOT save the .blend. Read-only from Blender's perspective.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import bpy
import numpy as np
from mathutils import Vector, Matrix


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser(prog="render_island_views.py")
    p.add_argument("--out-dir", required=True,
                   help="Output directory; render_persp.png and render_bev.png land here.")
    p.add_argument("--samples", type=int, default=128,
                   help="Cycles sample count (default: 128).")
    p.add_argument("--persp-size", type=int, nargs=2, default=[1024, 768],
                   metavar=("W", "H"))
    p.add_argument("--bev-size", type=int, nargs=2, default=[1024, 1024],
                   metavar=("W", "H"))
    p.add_argument("--metadata", default=None,
                   help="Path to metadata.json for M_inv transformation (optional).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def matrix_to_np(mat) -> np.ndarray:
    """Convert a Blender matrix or list of lists to numpy array."""
    if isinstance(mat, (Matrix, list)):
        return np.array(mat, dtype=np.float64)
    return np.array(list(mat), dtype=np.float64)

def np_to_matrix(arr: np.ndarray) -> Matrix:
    """Convert numpy array to Blender Matrix."""
    return Matrix([arr[i, :] for i in range(4)])

def compute_canonical_camera(scene_camera_matrix: Matrix,
                             M_inv_4x4: List[List[float]]) -> Matrix:
    """Compute canonical-frame camera = M_inv @ C_world."""
    C_world_np = matrix_to_np(scene_camera_matrix)
    M_inv_np = matrix_to_np(M_inv_4x4)
    C_canonical_np = M_inv_np @ C_world_np
    return np_to_matrix(C_canonical_np)

def load_metadata(metadata_path: str | None) -> dict | None:
    """Load metadata.json if provided; return None if not found."""
    if not metadata_path or not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[render_island_views] WARNING: failed to load metadata: {e}")
        return None

def get_scene_camera_matrix() -> Matrix | None:
    """Get the current scene camera's matrix_world."""
    if bpy.context.scene.camera:
        return bpy.context.scene.camera.matrix_world.copy()
    return None

def get_scene_camera_focal_length() -> float:
    """Get the focal length (lens) of the scene camera."""
    if bpy.context.scene.camera and bpy.context.scene.camera.data:
        return bpy.context.scene.camera.data.lens
    return 50.0  # fallback default

# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------

def world_bbox_for(objs: List[bpy.types.Object]) -> Tuple[Vector, Vector]:
    """Return world-space (min, max) bounding box over all mesh objects rooted at objs."""
    pts: List[Vector] = []
    for o in objs:
        for child in [o] + list(o.children_recursive):
            if child.type == "MESH" and len(child.data.vertices) > 0:
                pts.extend(child.matrix_world @ Vector(c) for c in child.bound_box)
    if not pts:
        return Vector((-1.0, -1.0, 0.0)), Vector((1.0, 1.0, 1.0))
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def make_camera(name: str, location: Vector, target: Vector,
                ortho: bool = False, ortho_scale: float = 1.0,
                focal: float = 50.0) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = focal
    if ortho:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = location
    direction = (target - location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam_obj


# ---------------------------------------------------------------------------
# Scene setup helpers
# ---------------------------------------------------------------------------

def setup_world() -> None:
    """Set a neutral gray background for Cycles."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new("ShaderNodeBackground")
    bg_node.inputs[0].default_value = (0.8, 0.8, 0.8, 1.0)
    bg_node.inputs[1].default_value = 0.6


def ensure_lighting() -> None:
    """Add a simple sun light if the scene is unlit so renders are not black."""
    has_light = any(o.type == "LIGHT" for o in bpy.data.objects)
    if has_light:
        return
    light_data = bpy.data.lights.new("IslandSun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("IslandSun", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (3.0, -3.0, 3.5)
    light.rotation_euler = (math.radians(50), math.radians(20), math.radians(30))


def configure_cycles(samples: int, size_x: int, size_y: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = size_x
    scene.render.resolution_y = size_y
    scene.render.resolution_percentage = 100


def render_to(filepath: str) -> None:
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

def check_render_emptiness(filepath: str) -> Tuple[bool, float]:
    """Check if a render is essentially empty (mostly background).

    Returns: (is_empty, non_bg_ratio)
      is_empty: True if <2% of pixels are non-background
      non_bg_ratio: fraction of non-background pixels (0.0 to 1.0)
    """
    try:
        from PIL import Image
        img = Image.open(filepath).convert("RGB")
        pixels = np.array(img, dtype=np.float32) / 255.0

        # Background is neutral gray (0.8, 0.8, 0.8)
        # Consider a pixel "background" if it's close to gray (tolerance ~0.05)
        bg_color = np.array([0.8, 0.8, 0.8])
        tolerance = 0.05

        # Distance from pixel to background color
        distances = np.min(np.abs(pixels - bg_color), axis=2)
        non_bg = distances > tolerance
        non_bg_ratio = np.mean(non_bg)

        is_empty = non_bg_ratio < 0.02
        return is_empty, non_bg_ratio
    except Exception as e:
        print(f"[render_island_views] WARNING: failed to check emptiness: {e}")
        return False, 0.5  # assume not empty on error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[render_island_views] out-dir  : {out_dir}")
    print(f"[render_island_views] samples  : {args.samples}")
    print(f"[render_island_views] persp-size: {args.persp_size[0]}x{args.persp_size[1]}")
    print(f"[render_island_views] bev-size  : {args.bev_size[0]}x{args.bev_size[1]}")

    # ------------------------------------------------------------------ #
    # 1. World + lighting (shared by both renders)                         #
    # ------------------------------------------------------------------ #
    setup_world()
    ensure_lighting()

    # ------------------------------------------------------------------ #
    # 2. Compute island bbox over ALL mesh objects in the file.            #
    #    Include both top-level and parented objects so every member       #
    #    contributes to the framing, not just roots.                       #
    # ------------------------------------------------------------------ #
    all_mesh_roots = [o for o in bpy.data.objects
                      if o.type in ("EMPTY", "MESH") and not o.parent]
    mn, mx = world_bbox_for(all_mesh_roots)

    extent = mx - mn
    center = (mn + mx) * 0.5
    long_axis = max(extent.x, extent.y)
    height = max(extent.z, 0.5)

    print(f"[render_island_views] island bbox  min={tuple(round(v, 4) for v in mn)}"
          f"  max={tuple(round(v, 4) for v in mx)}")
    print(f"[render_island_views] center={tuple(round(v, 4) for v in center)}"
          f"  long_axis={long_axis:.4f}  height={height:.4f}")

    # ------------------------------------------------------------------ #
    # 3. Perspective render — use canonical-frame scene camera if          #
    #    available, with fallback to computed bbox camera if the canonical #
    #    camera produces an empty render.                                  #
    # ------------------------------------------------------------------ #
    use_fallback = False
    fallback_cam = None

    # Try to use the canonical-frame scene camera
    metadata = load_metadata(args.metadata)
    if metadata and "M_inv_4x4" in metadata:
        try:
            scene_cam_matrix = get_scene_camera_matrix()
            if scene_cam_matrix:
                # Compute C_canonical = M_inv @ C_world
                canonical_cam_matrix = compute_canonical_camera(
                    scene_cam_matrix, metadata["M_inv_4x4"]
                )
                # Create a camera at the canonical position
                persp_cam = bpy.data.cameras.new("IslandPerspCamCanonical")
                persp_cam.lens = get_scene_camera_focal_length()
                persp_cam_obj = bpy.data.objects.new("IslandPerspCamCanonical", persp_cam)
                bpy.context.scene.collection.objects.link(persp_cam_obj)
                persp_cam_obj.matrix_world = canonical_cam_matrix

                print(f"[render_island_views] using canonical-frame camera")
                print(f"[render_island_views] persp cam focal_length={persp_cam.lens}")

                # Render and check for emptiness
                bpy.context.scene.camera = persp_cam_obj
                configure_cycles(args.samples, args.persp_size[0], args.persp_size[1])
                persp_path = os.path.join(out_dir, "render_persp.png")
                render_to(persp_path)

                is_empty, non_bg_ratio = check_render_emptiness(persp_path)
                print(f"[render_island_views] persp render done → {persp_path}")
                print(f"[render_island_views] non-bg pixel ratio: {non_bg_ratio:.4f}")

                if is_empty:
                    print(f"[render_island_views] WARNING: canonical camera produced empty render; "
                          f"falling back to bbox camera")
                    use_fallback = True
                else:
                    # Success! Keep the render
                    bpy.data.objects.remove(fallback_cam) if fallback_cam else None
            else:
                print(f"[render_island_views] no scene camera found; using fallback")
                use_fallback = True
        except Exception as e:
            print(f"[render_island_views] WARNING: canonical camera setup failed: {e}")
            use_fallback = True
    else:
        print(f"[render_island_views] metadata not provided or missing M_inv_4x4; "
              f"using fallback bbox camera")
        use_fallback = True

    # Fallback: compute synthetic camera from bbox
    if use_fallback:
        cam_loc = Vector((center.x,
                          center.y - max(long_axis * 2.5, 4.5),
                          center.z + height * 0.3))
        cam_target = Vector((center.x, center.y, center.z + height * 0.2))
        fallback_cam = make_camera("IslandPerspCamFallback", cam_loc, cam_target,
                                   ortho=False, focal=50.0)
        print(f"[render_island_views] fallback cam loc=("
              f"{cam_loc.x:+.4f}, {cam_loc.y:+.4f}, {cam_loc.z:+.4f})  "
              f"target=({cam_target.x:+.4f}, {cam_target.y:+.4f}, {cam_target.z:+.4f})")

        bpy.context.scene.camera = fallback_cam
        configure_cycles(args.samples, args.persp_size[0], args.persp_size[1])
        persp_path = os.path.join(out_dir, "render_persp.png")
        render_to(persp_path)
        print(f"[render_island_views] fallback persp render done → {persp_path}")

    # ------------------------------------------------------------------ #
    # 4. BEV render — orthographic top-down.                              #
    # ------------------------------------------------------------------ #
    bev_height = center.z + max(extent.z, long_axis) * 4.0
    bev_cam_loc = Vector((center.x, center.y, bev_height))
    bev_cam_target = Vector((center.x, center.y, center.z))
    ortho_scale = max(extent.x, extent.y) + 0.6
    bev_cam = make_camera("IslandBEVCam", bev_cam_loc, bev_cam_target,
                          ortho=True, ortho_scale=ortho_scale)
    print(f"[render_island_views] bev cam loc=({bev_cam_loc.x:+.4f}, "
          f"{bev_cam_loc.y:+.4f}, {bev_cam_loc.z:+.4f})  "
          f"ortho_scale={ortho_scale:.4f}")

    bpy.context.scene.camera = bev_cam
    # BEV needs fewer samples — cap at 64 to keep iteration fast.
    bev_w, bev_h = args.bev_size[0], args.bev_size[1]
    configure_cycles(min(args.samples, 64), bev_w, bev_h)
    bev_path = os.path.join(out_dir, "render_bev.png")
    render_to(bev_path)
    print(f"[render_island_views] BEV render done → {bev_path}")

    print(f"[render_island_views] DONE persp={persp_path} bev={bev_path}")


if __name__ == "__main__":
    main()
