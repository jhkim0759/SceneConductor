"""Render a 2D bird's-eye-view (top-down) image of a Blender-coordinate scene from a JSON file."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Rotation helpers (Blender XYZ intrinsic Euler: R = Rz @ Ry @ Rx)
# ---------------------------------------------------------------------------

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def euler_to_matrix_xyz(rx, ry, rz):
    """Build rotation matrix from XYZ intrinsic Euler angles (Blender convention).

    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


# ---------------------------------------------------------------------------
# Mesh loading and transformation
# ---------------------------------------------------------------------------

def load_mesh(path: str):
    """Load a mesh from disk; handles .glb Scenes by concatenating geometries."""
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")

    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        parts = loaded.dump()
        if len(parts) == 0:
            raise ValueError(f"Empty scene in {path}")
        loaded = trimesh.util.concatenate(parts)

    return loaded


def trimesh_to_blender_local(vertices: np.ndarray) -> np.ndarray:
    """Apply coordinate conversion trimesh(x,y,z) -> blender(-x, z, y) to local vertices."""
    v = vertices.copy()
    # new_x = -old_x, new_y = old_z, new_z = old_y
    result = np.empty_like(v)
    result[:, 0] = -v[:, 0]
    result[:, 1] = v[:, 2]
    result[:, 2] = v[:, 1]
    return result


def normalize_to_unit_cube(vertices: np.ndarray):
    """Center and scale vertices so the largest axis spans [-1, 1].

    Returns (normalized_vertices, applied_successfully).
    """
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    max_ext = extent.max()

    if max_ext < 1e-12:
        return vertices, False

    centroid = (vmin + vmax) / 2.0
    normalized = (vertices - centroid) / (max_ext / 2.0)
    return normalized, True


def aabb_corners(vmin, vmax):
    """Return all 8 corners of an axis-aligned bounding box."""
    corners = []
    for xi in (vmin[0], vmax[0]):
        for yi in (vmin[1], vmax[1]):
            for zi in (vmin[2], vmax[2]):
                corners.append([xi, yi, zi])
    return np.array(corners)


def apply_srt(vertices, scale, rotation_euler, location):
    """Apply Scale -> Rotate -> Translate to a set of 3D points."""
    rx, ry, rz = rotation_euler
    R = euler_to_matrix_xyz(rx, ry, rz)
    s = np.array(scale)
    # Scale
    v = vertices * s
    # Rotate
    v = (R @ v.T).T
    # Translate
    v = v + np.array(location)
    return v


def convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """Return the XY convex hull vertices in order, falling back to the raw points."""
    if HAS_SCIPY and len(points_xy) >= 3:
        try:
            hull = ConvexHull(points_xy)
            return points_xy[hull.vertices]
        except Exception:
            pass
    return points_xy


def process_mesh_entry(entry: dict, label: str):
    """
    Full pipeline for one mesh entry (floor or object):
    1. Load mesh
    2. Coord convert (trimesh -> blender local)
    3. Normalize to unit cube
    4. Compute local AABB corners
    5. Apply SRT to both vertices and AABB corners -> world space
    Returns (world_vertices, world_aabb_corners, summary_str) or raises.
    """
    path = entry["mesh_path"]
    mesh = load_mesh(path)

    # Step 2: coordinate conversion on local vertices
    verts_local = trimesh_to_blender_local(np.array(mesh.vertices, dtype=float))

    # Step 3: normalize
    verts_norm, ok = normalize_to_unit_cube(verts_local)
    if not ok:
        print(f"  WARNING [{label}]: zero-extent mesh, skipping normalization")
        verts_norm = verts_local

    # Step 4: local AABB (after normalization)
    vmin = verts_norm.min(axis=0)
    vmax = verts_norm.max(axis=0)
    local_corners = aabb_corners(vmin, vmax)

    # Step 5: apply SRT
    scale = entry["scale"]
    rot = entry["rotation_euler"]
    loc = entry["location"]

    world_verts = apply_srt(verts_norm, scale, rot, loc)
    world_corners = apply_srt(local_corners, scale, rot, loc)

    # Summary
    wmin = world_verts.min(axis=0)
    wmax = world_verts.max(axis=0)
    ext = wmax - wmin
    summary = (
        f"[{label}] verts={len(world_verts):,}  "
        f"world-AABB extents: X={ext[0]:.3f}m  Y={ext[1]:.3f}m  Z={ext[2]:.3f}m"
    )

    return world_verts, world_corners, summary


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def camera_forward_xy(cam: dict):
    """Return the camera forward direction projected onto XY (Blender frame).

    In Blender a camera at identity looks down -Z; forward_world = R_cam @ (0,0,-1).
    """
    rx, ry, rz = cam["rotation_euler"]
    R = euler_to_matrix_xyz(rx, ry, rz)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    return fwd  # full 3-vector; caller uses [0] and [1]


def camera_fov_rays(cam: dict, length: float):
    """Return two XY endpoint pairs for the horizontal FOV cone lines.

    FOV formula (HORIZONTAL fit): fov_h = 2 * atan(sensor_width / (2 * lens))
    """
    fov_h = 2.0 * math.atan(cam["sensor_width"] / (2.0 * cam["lens"]))
    half = fov_h / 2.0

    rx, ry, rz = cam["rotation_euler"]
    R = euler_to_matrix_xyz(rx, ry, rz)

    # Two boundary rays in camera space (in the XZ plane of the camera)
    rays = []
    for sign in (-1, 1):
        # ray in camera local frame: rotate fwd (-Z) by +/-half around Y
        local_ray = np.array([math.sin(sign * half), 0.0, -math.cos(sign * half)])
        world_ray = R @ local_ray
        rays.append(world_ray)

    return rays, fov_h


# ---------------------------------------------------------------------------
# BEV rendering
# ---------------------------------------------------------------------------

def render_bev(scene_json: dict, out_path: str):
    cam = scene_json["camera"]
    objects = scene_json["objects"]

    all_entries = [(obj["id"], obj) for obj in objects]

    cmap = plt.get_cmap("tab20")
    n_objects = len(objects)
    obj_colors = [cmap(i / max(n_objects, 1)) for i in range(n_objects)]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    source_name = Path(scene_json.get("meta", {}).get("source", "scene")).stem
    ax.set_title(f"Bird's-Eye View — {source_name}")

    legend_handles = []
    all_xy_points = []  # accumulate for auto-limits

    # Draw floor and objects
    for i, (label, entry) in enumerate(all_entries):
        try:
            world_verts, world_corners, summary = process_mesh_entry(entry, label)
        except FileNotFoundError as e:
            print(f"  ERROR: {e} — skipping {label}")
            continue
        except Exception as e:
            print(f"  ERROR [{label}]: {e} — skipping")
            continue

        print(summary)

        # Project to XY
        corners_xy = world_corners[:, :2]  # (8, 2)
        hull_pts = convex_hull_xy(corners_xy)

        all_xy_points.append(corners_xy)

        color = obj_colors[i]
        facecolor = color[:3]
        edgecolor = tuple(max(c - 0.2, 0) for c in color[:3])
        alpha_fill = 0.4
        alpha_edge = 0.85
        zorder = 2

        polygon_pts = np.vstack([hull_pts, hull_pts[0]])  # close the polygon
        ax.fill(polygon_pts[:, 0], polygon_pts[:, 1],
                color=facecolor, alpha=alpha_fill, zorder=zorder)
        ax.plot(polygon_pts[:, 0], polygon_pts[:, 1],
                color=edgecolor, linewidth=1.2, alpha=alpha_edge, zorder=zorder + 1)

        # Centroid label
        cx, cy = hull_pts[:, 0].mean(), hull_pts[:, 1].mean()
        ax.text(cx, cy, label, fontsize=6.5, ha="center", va="center",
                zorder=zorder + 2,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"))

        handle = mpatches.Patch(facecolor=facecolor, edgecolor=edgecolor,
                                label=label, alpha=0.7)
        legend_handles.append(handle)

    # Camera
    cam_x, cam_y = cam["location"][0], cam["location"][1]
    all_xy_points.append(np.array([[cam_x, cam_y]]))

    # Compute scene extent for arrow scaling
    if all_xy_points:
        all_pts = np.vstack(all_xy_points)
        scene_extent = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]))
    else:
        scene_extent = 1.0

    arrow_len = scene_extent * 0.12

    fwd = camera_forward_xy(cam)
    fwd_xy = np.array([fwd[0], fwd[1]])
    fwd_norm = np.linalg.norm(fwd_xy)
    if fwd_norm > 1e-9:
        fwd_xy = fwd_xy / fwd_norm

    # Camera forward arrow
    ax.annotate("", xy=(cam_x + fwd_xy[0] * arrow_len, cam_y + fwd_xy[1] * arrow_len),
                xytext=(cam_x, cam_y),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8),
                zorder=10)

    # Camera marker (triangle)
    ax.plot(cam_x, cam_y, marker="^", markersize=10, color="red", zorder=11,
            markeredgecolor="darkred", markeredgewidth=1.0)
    ax.text(cam_x, cam_y - scene_extent * 0.025, "camera",
            fontsize=7, ha="center", va="top", color="red", zorder=12)

    # FOV cone lines
    fov_rays, fov_h = camera_fov_rays(cam, arrow_len * 2)
    fov_len = arrow_len * 2
    for ray in fov_rays:
        ray_xy = np.array([ray[0], ray[1]])
        rn = np.linalg.norm(ray_xy)
        if rn > 1e-9:
            ray_xy = ray_xy / rn
        ax.plot([cam_x, cam_x + ray_xy[0] * fov_len],
                [cam_y, cam_y + ray_xy[1] * fov_len],
                color="red", linewidth=0.8, linestyle="--", alpha=0.45, zorder=9)

    cam_patch = mpatches.Patch(color="red", label=f"camera (FOV {math.degrees(fov_h):.0f}°)")
    legend_handles.append(cam_patch)

    # Auto-fit limits with 10% padding
    if all_xy_points:
        all_pts = np.vstack(all_xy_points)
        xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
        ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
        pad_x = (xmax - xmin) * 0.10 + 0.05
        pad_y = (ymax - ymin) * 0.10 + 0.05
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, framealpha=0.8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved BEV image -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    default_scene = "./sample/output/blender_scene.json"
    default_out = "./sample/output/bev.png"

    parser = argparse.ArgumentParser(
        description="Render a 2D bird's-eye-view image of a Blender scene JSON."
    )
    parser.add_argument("--scene", default=default_scene,
                        help="Path to blender_scene.json")
    parser.add_argument("--out", default=default_out,
                        help="Output PNG path")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"ERROR: Scene JSON not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(scene_path) as f:
        scene_json = json.load(f)

    print(f"Scene: {scene_path}")
    print(f"Output: {out_path}")
    print(f"Coordinate system: {scene_json.get('meta', {}).get('coordinate_system', 'unknown')}")
    print()

    render_bev(scene_json, str(out_path))


if __name__ == "__main__":
    main()
