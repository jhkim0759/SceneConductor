"""BEV of pointmap_xz.ply convex hull; writes bev_pointmap.png + bev_pointmap.json."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh

# render_bev is a sibling module inside this skill's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_bev import convex_hull_xy, euler_to_matrix_xyz  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAX_SCATTER = 20_000
SCATTER_SEED = 42


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    return ap.parse_args()


def _camera_fwd_xy(cam: dict) -> np.ndarray:
    rx, ry, rz = cam["rotation_euler"]
    R = euler_to_matrix_xyz(rx, ry, rz)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    xy = np.array([fwd[0], fwd[1]])
    n = np.linalg.norm(xy)
    return xy / n if n > 1e-9 else xy


def _draw_camera_glyph(ax, cam_x: float, cam_y: float, fwd_xy: np.ndarray,
                        fov_deg: float, scene_extent: float) -> None:
    arrow_len = max(scene_extent * 0.12, 0.3)
    fov_rad = math.radians(fov_deg)
    half = fov_rad / 2.0

    ax.plot(cam_x, cam_y, marker="^", markersize=10, color="red", zorder=11,
            markeredgecolor="darkred", markeredgewidth=1.0)
    ax.annotate("", xy=(cam_x + fwd_xy[0] * arrow_len, cam_y + fwd_xy[1] * arrow_len),
                xytext=(cam_x, cam_y),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8), zorder=10)

    fov_len = arrow_len * 2.0
    yaw = math.atan2(float(fwd_xy[1]), float(fwd_xy[0]))
    for sign in (-1, 1):
        angle = yaw + sign * half
        rx_r, ry_r = math.cos(angle) * fov_len, math.sin(angle) * fov_len
        ax.plot([cam_x, cam_x + rx_r], [cam_y, cam_y + ry_r],
                color="red", linewidth=0.8, linestyle="--", alpha=0.45, zorder=9)


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    json_dir = scene_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    ply_path = scene_dir / "inputs" / "pointmap_xz.ply"
    loaded = trimesh.load(str(ply_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        parts = loaded.dump()
        loaded = trimesh.util.concatenate(parts) if parts else None
    if loaded is None or len(loaded.vertices) == 0:
        raise RuntimeError(f"Empty pointmap PLY: {ply_path}")

    scene_json = json.loads((scene_dir / "json" / "blender_scene.json").read_text(encoding="utf-8"))
    scale_factor = float(scene_json.get("meta", {}).get("world_scale_factor", 1.0))

    raw = np.array(loaded.vertices, dtype=np.float64)
    # Pointmap is in mesh frame; convert to Blender frame trimesh->blender: (x,y,z) -> (-x, z, y),
    # then scale by meta.world_scale_factor to match the rest of the scene's units.
    verts = raw[:, [0, 2, 1]] * np.array([-1.0, 1.0, 1.0]) * scale_factor
    n_points = len(verts)
    xy_all = verts[:, :2]
    z_range = [float(verts[:, 2].min()), float(verts[:, 2].max())]

    hull_pts = convex_hull_xy(xy_all)
    centroid_xy = [float(hull_pts[:, 0].mean()), float(hull_pts[:, 1].mean())]

    # Camera glyph
    cam = json.loads((json_dir / "blender_scene.json").read_text(encoding="utf-8")).get("camera", {})
    cam_xy = [float(cam.get("location", [0, 0, 0])[0]),
              float(cam.get("location", [0, 0, 0])[1])]
    fwd_xy = _camera_fwd_xy(cam) if cam else np.array([0.0, 1.0])
    fov_deg = float(math.degrees(2.0 * math.atan(
        cam.get("sensor_width", 36.0) / (2.0 * cam.get("lens", 26.0))
    )))

    # Downsample scatter for speed
    rng = np.random.default_rng(SCATTER_SEED)
    idx = rng.choice(n_points, min(MAX_SCATTER, n_points), replace=False)
    scatter_xy = xy_all[idx]

    all_pts_arr = np.vstack([xy_all, [cam_xy]])
    scene_extent = max(np.ptp(all_pts_arr[:, 0]), np.ptp(all_pts_arr[:, 1]), 1.0)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV — pointmap XY")

    ax.scatter(scatter_xy[:, 0], scatter_xy[:, 1], s=1, c="lightgrey", alpha=0.05, zorder=1, rasterized=True)

    closed = np.vstack([hull_pts, hull_pts[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="steelblue", linewidth=2.0, zorder=3, label="pointmap hull")

    _draw_camera_glyph(ax, cam_xy[0], cam_xy[1], fwd_xy, fov_deg, scene_extent)

    xmin, xmax = all_pts_arr[:, 0].min(), all_pts_arr[:, 0].max()
    ymin, ymax = all_pts_arr[:, 1].min(), all_pts_arr[:, 1].max()
    pad_x = (xmax - xmin) * 0.10 + 0.05
    pad_y = (ymax - ymin) * 0.10 + 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.legend(fontsize=8, framealpha=0.8)

    out_png = json_dir / "bev_pointmap.png"
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_json = {
        "hull_xy": hull_pts.tolist(),
        "centroid_xy": centroid_xy,
        "z_range": z_range,
        "n_points": n_points,
        "frame": "blend_world",
    }
    (json_dir / "bev_pointmap.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"[bev_pointmap] n_points={n_points}  hull_verts={len(hull_pts)}  z=[{z_range[0]:.3f},{z_range[1]:.3f}]")


if __name__ == "__main__":
    main()
