"""Overlay bev_objects.json + bev_pointmap.json into bev_compare.png AND
also render bev_combined_hull.png — the convex hull of (pointmap ∪ objects ∪
camera disk). The combined hull is treated as a *reference draft* for the
floor plan; the compute_polygon fitter does NOT require the polygon to
contain it, but it's useful context for the floor-plan agent."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    return ap.parse_args()


def _draw_camera_glyph(ax, cam_xy: list, fwd_xy: np.ndarray, fov_deg: float,
                        scene_extent: float) -> None:
    cam_x, cam_y = cam_xy
    arrow_len = max(scene_extent * 0.12, 0.3)
    fov_rad = math.radians(fov_deg)
    half = fov_rad / 2.0
    fov_len = arrow_len * 2.0

    ax.plot(cam_x, cam_y, marker="^", markersize=10, color="red", zorder=11,
            markeredgecolor="darkred", markeredgewidth=1.0)
    ax.annotate("", xy=(cam_x + fwd_xy[0] * arrow_len, cam_y + fwd_xy[1] * arrow_len),
                xytext=(cam_x, cam_y),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=1.8), zorder=10)

    yaw = math.atan2(float(fwd_xy[1]), float(fwd_xy[0]))
    for sign in (-1, 1):
        angle = yaw + sign * half
        ax.plot([cam_x, cam_x + math.cos(angle) * fov_len],
                [cam_y, cam_y + math.sin(angle) * fov_len],
                color="red", linewidth=0.8, linestyle="--", alpha=0.45, zorder=9)


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    json_dir = scene_dir / "json"

    bev_obj = json.loads((json_dir / "bev_objects.json").read_text(encoding='utf-8'))
    bev_pm = json.loads((json_dir / "bev_pointmap.json").read_text(encoding='utf-8'))

    cam_xy = bev_obj["camera_xy"]
    cam_yaw_deg = bev_obj["camera_yaw_deg"]
    fov_deg = bev_obj["fov_deg"]
    yaw_rad = math.radians(cam_yaw_deg)
    fwd_xy = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])

    cmap = plt.get_cmap("tab20")
    n_objs = max(len(bev_obj["objects"]), 1)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("BEV — objects (filled) + pointmap hull (dashed)")

    all_pts = [np.array([cam_xy])]
    legend_handles = []

    # Object hulls — filled polygons
    for i, obj in enumerate(bev_obj["objects"]):
        hull = np.array(obj["hull_xy"])
        all_pts.append(hull)
        color = cmap(i / n_objs)
        face_c = color[:3]
        edge_c = tuple(max(c - 0.15, 0) for c in face_c)
        closed = np.vstack([hull, hull[0]])
        ax.fill(closed[:, 0], closed[:, 1], color=face_c, alpha=0.45, zorder=2)
        ax.plot(closed[:, 0], closed[:, 1], color=edge_c, linewidth=0.8, zorder=3)
        cx, cy = hull[:, 0].mean(), hull[:, 1].mean()
        ax.text(cx, cy, obj["id"], fontsize=6, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, ec="none"))
        legend_handles.append(mpatches.Patch(facecolor=face_c, edgecolor=edge_c,
                                              label=obj["id"], alpha=0.7))

    # Pointmap hull — thick blue dashed outline, no fill
    pm_hull = np.array(bev_pm["hull_xy"])
    all_pts.append(pm_hull)
    closed_pm = np.vstack([pm_hull, pm_hull[0]])
    ax.plot(closed_pm[:, 0], closed_pm[:, 1], color="steelblue",
            linewidth=2.2, linestyle="--", zorder=5, label="pointmap hull")
    legend_handles.append(mpatches.Patch(facecolor="none", edgecolor="steelblue",
                                          label="pointmap hull", linestyle="--"))

    all_pts_arr = np.vstack(all_pts)
    scene_extent = max(np.ptp(all_pts_arr[:, 0]), np.ptp(all_pts_arr[:, 1]), 1.0)
    _draw_camera_glyph(ax, cam_xy, fwd_xy, fov_deg, scene_extent)
    legend_handles.append(mpatches.Patch(color="red", label=f"camera (FOV {fov_deg:.0f}°)"))

    # Axis limits with 5% pad
    xmin, xmax = all_pts_arr[:, 0].min(), all_pts_arr[:, 0].max()
    ymin, ymax = all_pts_arr[:, 1].min(), all_pts_arr[:, 1].max()
    pad_x = (xmax - xmin) * 0.05 + 0.05
    pad_y = (ymax - ymin) * 0.05 + 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.85,
              ncol=2, bbox_to_anchor=(1.0, 1.0))

    out_png = json_dir / "bev_compare.png"
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[bev_overlay] saved {out_png}  ({len(bev_obj['objects'])} objects, pointmap hull {len(pm_hull)} pts)")

    _render_combined_hull(scene_dir, json_dir, bev_obj, fwd_xy, fov_deg)


# ---------------------------------------------------------------------------
# Combined-hull rendering — the convex hull of (pointmap ∪ objects ∪ camera).
# Useful as a *reference draft* for the floor plan agent: it shows the full
# extent of "places we have any signal about," even though the polygon does
# NOT need to enclose every pointmap point.
# ---------------------------------------------------------------------------

def _load_pointmap_xy(scene_dir: Path) -> np.ndarray:
    """Load pointmap_xz.ply and apply the same mesh→Blender frame conversion
    used in bev_pointmap.py: (x, y, z) -> (-x, z, y) * world_scale_factor."""
    ply_path = scene_dir / "inputs" / "pointmap_xz.ply"
    sc_json  = scene_dir / "json" / "blender_scene.json"
    if not ply_path.exists():
        return np.empty((0, 2))
    pc = trimesh.load(str(ply_path), force="mesh", process=False)
    if isinstance(pc, trimesh.Scene):
        parts = pc.dump()
        pc = trimesh.util.concatenate(parts) if parts else None
    if pc is None or len(pc.vertices) == 0:
        return np.empty((0, 2))
    scale = 1.0
    if sc_json.exists():
        try:
            scale = float(json.loads(sc_json.read_text(encoding='utf-8'))
                          .get("meta", {}).get("world_scale_factor", 1.0))
        except Exception:
            pass
    raw = np.asarray(pc.vertices, dtype=np.float64)
    verts = raw[:, [0, 2, 1]] * np.array([-1.0, 1.0, 1.0]) * scale
    return verts[:, :2]


def _render_combined_hull(scene_dir, json_dir, bev_obj, fwd_xy, fov_deg):
    """Render bev_combined_hull.png — pointmap scatter + object hulls +
    combined convex hull + camera glyph."""
    pc_xy = _load_pointmap_xy(scene_dir)
    obj_hulls = [np.asarray(o["hull_xy"]) for o in bev_obj.get("objects", [])]
    obj_pts = (np.concatenate(obj_hulls, axis=0)
               if obj_hulls else np.empty((0, 2)))
    cam_xy = np.asarray(bev_obj["camera_xy"], dtype=float)
    cam_disk = np.array([
        [cam_xy[0] + 0.10 * math.cos(k * math.pi / 4),
         cam_xy[1] + 0.10 * math.sin(k * math.pi / 4)] for k in range(8)
    ])

    rng = np.random.default_rng(0)
    if len(pc_xy):
        ds = rng.choice(len(pc_xy), min(10_000, len(pc_xy)), replace=False)
        pc_for_hull = pc_xy[ds]
    else:
        pc_for_hull = pc_xy
    combined = np.concatenate([pc_for_hull, obj_pts, cam_disk], axis=0)
    if len(combined) < 3:
        print("[bev_overlay] combined hull skipped: too few points")
        return
    hull = combined[ConvexHull(combined).vertices]
    hull_closed = np.vstack([hull, hull[:1]])

    # Save numerical hull vertices alongside the PNG so the floor-plan agent
    # can start from exact coordinates instead of tracing pixels.
    hull_bbox_min = hull.min(axis=0)
    hull_bbox_max = hull.max(axis=0)
    xs, ys = hull[:, 0], hull[:, 1]
    hull_area = 0.5 * float(np.abs(np.dot(xs, np.roll(ys, -1)) -
                                    np.dot(np.roll(xs, -1), ys)))
    hull_json_path = json_dir / "bev_combined_hull.json"
    hull_json_path.write_text(json.dumps({
        "hull_xy": hull.tolist(),
        "n_verts": int(len(hull)),
        "area_m2": hull_area,
        "bbox_xy": [
            [float(hull_bbox_min[0]), float(hull_bbox_min[1])],
            [float(hull_bbox_max[0]), float(hull_bbox_max[1])],
        ],
        "extent_x": float(hull_bbox_max[0] - hull_bbox_min[0]),
        "extent_y": float(hull_bbox_max[1] - hull_bbox_min[1]),
        "camera_xy": [float(cam_xy[0]), float(cam_xy[1])],
        "source_counts": {
            "pointmap_pts": int(len(pc_for_hull)),
            "n_objects": int(len(obj_hulls)),
            "camera_disk_pts": int(len(cam_disk)),
        },
        "note": "CCW-ordered convex hull of (pointmap ∪ object hulls ∪ camera disk). "
                 "Reference draft — your final polygon does NOT need to enclose every "
                 "vertex of this hull (pointmap can overshoot the true wall).",
    }, indent=2), encoding="utf-8")
    print(f"[bev_overlay] saved {hull_json_path}  hull_verts={len(hull)} area={hull_area:.2f}m²")

    fig, ax = plt.subplots(figsize=(9, 11), dpi=200)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Combined convex hull — pointmap + {len(obj_hulls)} objects + camera"
    )

    if len(pc_xy):
        n_show = min(20_000, len(pc_xy))
        idx = rng.choice(len(pc_xy), n_show, replace=False)
        ax.scatter(pc_xy[idx, 0], pc_xy[idx, 1], s=1, alpha=0.10,
                   c="#1f77b4", label=f"pointmap ({len(pc_xy):,} pts)")

    cmap = plt.get_cmap("tab20")
    n_objs = max(len(obj_hulls), 1)
    for i, h in enumerate(obj_hulls):
        h_closed = np.vstack([h, h[:1]])
        ax.fill(h_closed[:, 0], h_closed[:, 1], alpha=0.40,
                color=cmap(i / n_objs), edgecolor="black", linewidth=0.4)

    ax.plot(hull_closed[:, 0], hull_closed[:, 1], color="red",
            linewidth=2.5, label=f"combined hull ({len(hull)} verts)")
    ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.06, color="red")

    arr_len = max(0.5, float(np.ptp(combined, axis=0).max()) * 0.08)
    ax.annotate("", xy=(cam_xy[0] + fwd_xy[0] * arr_len,
                        cam_xy[1] + fwd_xy[1] * arr_len),
                xytext=(cam_xy[0], cam_xy[1]),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax.plot([cam_xy[0]], [cam_xy[1]], marker="^", color="red",
            markersize=12, label="camera")

    mn = combined.min(axis=0); mx = combined.max(axis=0)
    pad = (mx - mn) * 0.05 + 0.05
    ax.set_xlim(min(mn[0], cam_xy[0]) - pad[0],
                max(mx[0], cam_xy[0]) + pad[0])
    ax.set_ylim(min(mn[1], cam_xy[1]) - pad[1],
                max(mx[1], cam_xy[1]) + pad[1])

    ax.legend(loc="upper right", fontsize=8)
    out = json_dir / "bev_combined_hull.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[bev_overlay] saved {out}  hull_verts={len(hull)}")


if __name__ == "__main__":
    main()
