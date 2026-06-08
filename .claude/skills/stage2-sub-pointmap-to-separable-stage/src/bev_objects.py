"""BEV of object footprints from blender_scene.json; writes bev_objects.png + bev_objects.json."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# render_bev is a sibling module inside this skill's src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_bev import (  # noqa: E402
    process_mesh_entry,
    convex_hull_xy,
    euler_to_matrix_xyz,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    return ap.parse_args()


def _camera_forward_xy(cam: dict) -> np.ndarray:
    rx, ry, rz = cam["rotation_euler"]
    R = euler_to_matrix_xyz(rx, ry, rz)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    return fwd


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
    yaw = math.atan2(fwd_xy[1], fwd_xy[0])
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

    scene_json = json.loads((json_dir / "blender_scene.json").read_text(encoding="utf-8"))
    cam = scene_json["camera"]
    objects = scene_json.get("objects", [])

    cam_xy = [float(cam["location"][0]), float(cam["location"][1])]
    fwd = _camera_forward_xy(cam)
    fwd_xy = np.array([fwd[0], fwd[1]])
    fwd_norm = np.linalg.norm(fwd_xy)
    if fwd_norm > 1e-9:
        fwd_xy = fwd_xy / fwd_norm
    camera_yaw_deg = float(math.degrees(math.atan2(float(fwd_xy[1]), float(fwd_xy[0]))))
    fov_deg = float(math.degrees(2.0 * math.atan(cam["sensor_width"] / (2.0 * cam["lens"]))))

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV — object footprints")

    hull_records = []
    all_pts = [np.array([cam_xy])]
    legend_handles = []

    obj_idx = 0
    for entry in objects:
        obj_id = entry.get("id", "")
        mesh_path = entry.get("mesh_path", "")
        # Skip the floor entry
        if obj_id == "floor" or "floor" in Path(mesh_path).name.lower():
            continue
        try:
            world_verts, world_corners, _ = process_mesh_entry(entry, obj_id)
        except Exception as exc:
            print(f"[bev_objects] WARNING: skipping {obj_id}: {exc}")
            continue

        corners_xy = world_corners[:, :2]
        hull_pts = convex_hull_xy(corners_xy)
        all_pts.append(corners_xy)

        color = cmap(obj_idx / max(len(objects), 1))
        obj_idx += 1
        face_c = color[:3]
        edge_c = tuple(max(c - 0.2, 0) for c in face_c)

        closed = np.vstack([hull_pts, hull_pts[0]])
        ax.fill(closed[:, 0], closed[:, 1], color=face_c, alpha=0.4, zorder=2)
        ax.plot(closed[:, 0], closed[:, 1], color=edge_c, linewidth=1.0, alpha=0.85, zorder=3)
        cx_h, cy_h = hull_pts[:, 0].mean(), hull_pts[:, 1].mean()
        ax.text(cx_h, cy_h, obj_id, fontsize=6, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, ec="none"))
        legend_handles.append(mpatches.Patch(facecolor=face_c, edgecolor=edge_c, label=obj_id, alpha=0.7))
        hull_records.append({"id": obj_id, "hull_xy": hull_pts.tolist()})

    all_pts_arr = np.vstack(all_pts)
    scene_extent = max(np.ptp(all_pts_arr[:, 0]), np.ptp(all_pts_arr[:, 1]), 1.0)
    _draw_camera_glyph(ax, cam_xy[0], cam_xy[1], fwd_xy, fov_deg, scene_extent)

    xmin, xmax = all_pts_arr[:, 0].min(), all_pts_arr[:, 0].max()
    ymin, ymax = all_pts_arr[:, 1].min(), all_pts_arr[:, 1].max()
    pad_x = (xmax - xmin) * 0.10 + 0.05
    pad_y = (ymax - ymin) * 0.10 + 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    legend_handles.append(mpatches.Patch(color="red", label=f"camera (FOV {fov_deg:.0f}°)"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.8, ncol=2)

    out_png = json_dir / "bev_objects.png"
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)

    out_json = {
        "objects": hull_records,
        "camera_xy": cam_xy,
        "camera_yaw_deg": camera_yaw_deg,
        "fov_deg": fov_deg,
        "frame": "blend_world",
    }
    (json_dir / "bev_objects.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"[bev_objects] {len(hull_records)} objects  camera_yaw={camera_yaw_deg:.1f}°  fov={fov_deg:.1f}°")


if __name__ == "__main__":
    main()
