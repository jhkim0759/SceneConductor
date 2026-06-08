"""Annotated floor-plan plot from polygon_v2.json; solid=WALL, dotted=OPEN; writes floor_plan.png."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    return ap.parse_args()


def _edge_midpoint(verts: list, i_from: int, i_to: int) -> tuple[float, float]:
    a = verts[i_from]
    b = verts[i_to]
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _draw_camera_glyph(ax, cam_xy: list, fov_deg: float, yaw_deg: float,
                        scene_extent: float) -> None:
    cam_x, cam_y = cam_xy
    arrow_len = max(scene_extent * 0.12, 0.3)
    fov_rad = math.radians(fov_deg)
    half = fov_rad / 2.0
    fov_len = arrow_len * 2.0

    yaw_rad = math.radians(yaw_deg)
    fwd_x, fwd_y = math.cos(yaw_rad), math.sin(yaw_rad)

    ax.plot(cam_x, cam_y, marker="^", markersize=10, color="red", zorder=11,
            markeredgecolor="darkred", markeredgewidth=1.0)
    ax.annotate("", xy=(cam_x + fwd_x * arrow_len, cam_y + fwd_y * arrow_len),
                xytext=(cam_x, cam_y),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5), zorder=10)

    for sign in (-1, 1):
        angle = yaw_rad + sign * half
        ax.plot([cam_x, cam_x + math.cos(angle) * fov_len],
                [cam_y, cam_y + math.sin(angle) * fov_len],
                color="red", linewidth=0.7, linestyle="--", alpha=0.4, zorder=9)


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    json_dir = scene_dir / "json"

    poly_data = json.loads((json_dir / "polygon_v2.json").read_text(encoding='utf-8'))
    verts = poly_data["polygon_vertices"]
    n = len(verts)
    wall_edges = poly_data.get("wall_edges", [])
    open_edges = poly_data.get("open_edges", [])
    floor_z = poly_data.get("floor_z", 0.0)
    ceiling_z = poly_data.get("ceiling_z", 2.8)
    yaw_deg = poly_data.get("yaw_deg", 0.0)
    camera_xy = poly_data.get("camera_xy", [0.0, 0.0])

    # Mark edge types
    open_set = set()
    for e in open_edges:
        open_set.add((int(e["from"]), int(e["to"])))
        open_set.add((int(e["to"]), int(e["from"])))

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    all_pts = [np.array(verts), np.array([camera_xy])]

    # Optional context layers
    bev_obj_path = json_dir / "bev_objects.json"
    if bev_obj_path.exists():
        bev_obj = json.loads(bev_obj_path.read_text(encoding='utf-8'))
        fov_deg = bev_obj.get("fov_deg", 60.0)
        cam_yaw = bev_obj.get("camera_yaw_deg", yaw_deg)
        for obj in bev_obj.get("objects", []):
            hull = np.array(obj["hull_xy"])
            all_pts.append(hull)
            closed = np.vstack([hull, hull[0]])
            ax.fill(closed[:, 0], closed[:, 1], color="lightgrey", alpha=0.25, zorder=1)
    else:
        fov_deg = 60.0
        cam_yaw = yaw_deg

    bev_pm_path = json_dir / "bev_pointmap.json"
    if bev_pm_path.exists():
        pm = json.loads(bev_pm_path.read_text(encoding='utf-8'))
        pm_hull = np.array(pm["hull_xy"])
        all_pts.append(pm_hull)
        closed_pm = np.vstack([pm_hull, pm_hull[0]])
        ax.plot(closed_pm[:, 0], closed_pm[:, 1], color="lightsteelblue",
                linewidth=0.9, linestyle=":", zorder=2)

    # Draw polygon edges
    wall_label_map = {(e["from"], e["to"]): e["object"] for e in wall_edges}
    wall_num = 0
    for i in range(n):
        v_from = i
        v_to = (i + 1) % n
        xa, ya = verts[v_from]
        xb, yb = verts[v_to]
        is_open = (v_from, v_to) in open_set

        if is_open:
            ax.plot([xa, xb], [ya, yb], color="black", linewidth=2.0,
                    linestyle=(0, (1, 3)), zorder=5)
            mx, my = _edge_midpoint(verts, v_from, v_to)
            ax.text(mx, my, "OPEN", fontsize=7, color="red", ha="center", va="bottom",
                    zorder=6, fontweight="bold")
        else:
            ax.plot([xa, xb], [ya, yb], color="black", linewidth=2.0, zorder=5)
            wall_label = wall_label_map.get((v_from, v_to), "")
            if not wall_label:
                wall_num += 1
                wall_label = f"W{wall_num}"
            else:
                # Short label from object name e.g. "Wall_02" → "W2"
                wall_label = wall_label.replace("Wall_", "W").lstrip("W0") or "W"
                wall_label = f"W{int(wall_label)}" if wall_label.isdigit() else wall_label
            mx, my = _edge_midpoint(verts, v_from, v_to)
            ax.text(mx, my, wall_label, fontsize=7, color="black", ha="center", va="bottom",
                    zorder=6, bbox=dict(fc="white", alpha=0.6, ec="none", pad=0.1))

    # Vertex dots + labels
    for i, (x, y) in enumerate(verts):
        ax.plot(x, y, "o", color="black", markersize=5, zorder=7)
        ax.text(x, y, f" v{i}", fontsize=6.5, color="#333333", va="bottom", zorder=8)

    # Camera glyph
    _draw_camera_glyph(ax, camera_xy, fov_deg, cam_yaw, scene_extent=2.0)

    # Auto-limits
    all_pts_arr = np.vstack(all_pts)
    xmin, xmax = all_pts_arr[:, 0].min(), all_pts_arr[:, 0].max()
    ymin, ymax = all_pts_arr[:, 1].min(), all_pts_arr[:, 1].max()
    pad_x = (xmax - xmin) * 0.10 + 0.1
    pad_y = (ymax - ymin) * 0.10 + 0.1
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    n_walls = len(wall_edges)
    n_open  = len(open_edges)
    ax.set_title(f"Floor plan — {n_walls} walls, {n_open} openings, yaw={yaw_deg:.1f}°  "
                 f"[floor_z={floor_z:.2f}, ceiling_z={ceiling_z:.2f}]")

    wall_handle  = mlines.Line2D([], [], color="black", linewidth=2, label="WALL")
    open_handle  = mlines.Line2D([], [], color="black", linewidth=2, linestyle=(0, (1, 3)), label="OPEN")
    cam_handle   = mlines.Line2D([], [], marker="^", color="red", linewidth=0, label="camera")
    ax.legend(handles=[wall_handle, open_handle, cam_handle], fontsize=8, framealpha=0.85)

    out_png = json_dir / "floor_plan.png"
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_floor_plan] saved {out_png}  n_verts={n} walls={n_walls} open={n_open}")


if __name__ == "__main__":
    main()
