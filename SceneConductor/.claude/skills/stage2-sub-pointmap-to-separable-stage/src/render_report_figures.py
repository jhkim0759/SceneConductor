"""One-off: regenerate paper-figure versions of bev_compare, bev_combined_hull,
and a new textured BEV of the pointmap.

Style overrides vs. the original skill scripts:
  * pointmap dots are visibly thicker (larger marker, higher alpha)
  * NO axis (no ticks, no labels, no frame, no title)
  * NO legend in any corner — but per-object 'obj_NN' labels in hull centres remain
  * axis limits padded so neither pointmap nor objects get clipped

Outputs land in <scene_dir>/report/.
Run with plain python3 (no Blender):

    python3 tmp/make_report_figures.py --scene-dir <scene_dir>
"""
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _strip_axis(ax) -> None:
    """Remove axis ticks, labels, frame, title — leave only the plotted artists."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def _euler_to_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _camera_fwd_xy(cam: dict) -> np.ndarray:
    rx, ry, rz = cam.get("rotation_euler", [0.0, 0.0, 0.0])
    R = _euler_to_matrix_xyz(rx, ry, rz)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    xy = np.array([fwd[0], fwd[1]])
    n = np.linalg.norm(xy)
    return xy / n if n > 1e-9 else np.array([0.0, 1.0])


def _draw_camera(ax, cam_xy, fwd_xy, fov_deg: float, scene_extent: float) -> None:
    cx, cy = cam_xy
    arrow_len = max(scene_extent * 0.10, 0.3)
    ax.plot(cx, cy, marker="^", markersize=14, color="red", zorder=11,
            markeredgecolor="darkred", markeredgewidth=1.2)
    ax.annotate("", xy=(cx + fwd_xy[0] * arrow_len, cy + fwd_xy[1] * arrow_len),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.0), zorder=10)
    half = math.radians(fov_deg) / 2.0
    yaw = math.atan2(float(fwd_xy[1]), float(fwd_xy[0]))
    fov_len = arrow_len * 2.0
    for sign in (-1, 1):
        a = yaw + sign * half
        ax.plot([cx, cx + math.cos(a) * fov_len],
                [cy, cy + math.sin(a) * fov_len],
                color="red", linewidth=1.0, linestyle="--", alpha=0.5, zorder=9)


def _load_pointmap(scene_dir: Path):
    """Return (xy [N,2], rgb [N,3] in 0-1, z [N]) in Blender world frame.

    Mesh→Blender frame: (x, y, z) → (-x, z, y), then * meta.world_scale_factor.
    """
    ply_path = scene_dir / "inputs" / "pointmap_xz.ply"
    pc = trimesh.load(str(ply_path), force="mesh", process=False)
    if isinstance(pc, trimesh.Scene):
        parts = pc.dump()
        pc = trimesh.util.concatenate(parts) if parts else None
    if pc is None or len(pc.vertices) == 0:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty((0,))

    sc_json = scene_dir / "json" / "blender_scene.json"
    scale = 1.0
    if sc_json.exists():
        try:
            scale = float(json.loads(sc_json.read_text())
                          .get("meta", {}).get("world_scale_factor", 1.0))
        except Exception:
            pass

    raw = np.asarray(pc.vertices, dtype=np.float64)
    verts = raw[:, [0, 2, 1]] * np.array([-1.0, 1.0, 1.0]) * scale

    rgb = np.full((len(verts), 3), 0.5, dtype=np.float32)
    vis = getattr(pc, "visual", None)
    if vis is not None and hasattr(vis, "vertex_colors"):
        try:
            vc = np.asarray(vis.vertex_colors)
            if vc.ndim == 2 and vc.shape[1] >= 3 and len(vc) == len(verts):
                rgb = vc[:, :3].astype(np.float32) / 255.0
        except Exception:
            pass

    return verts[:, :2], rgb, verts[:, 2]


# ---------------------------------------------------------------------------
# figure 1 — bev_compare style (objects filled + pointmap hull dashed +
#                                pointmap dots thicker, NO axis/legend)
# ---------------------------------------------------------------------------

def make_bev_compare(scene_dir: Path, out_dir: Path) -> Path:
    json_dir = scene_dir / "json"
    bev_obj = json.loads((json_dir / "bev_objects.json").read_text())
    bev_pm  = json.loads((json_dir / "bev_pointmap.json").read_text())

    cam_xy = bev_obj["camera_xy"]
    fov_deg = float(bev_obj["fov_deg"])
    yaw_rad = math.radians(float(bev_obj["camera_yaw_deg"]))
    fwd_xy = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])

    pm_xy, _pm_rgb, _pm_z = _load_pointmap(scene_dir)

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")

    cmap = plt.get_cmap("tab20")
    obj_hulls = [np.asarray(o["hull_xy"]) for o in bev_obj.get("objects", [])]
    n_objs = max(len(obj_hulls), 1)
    all_pts = [np.array([cam_xy])]

    # 1. pointmap scatter — thick dots
    if len(pm_xy):
        rng = np.random.default_rng(42)
        n_show = min(25_000, len(pm_xy))
        idx = rng.choice(len(pm_xy), n_show, replace=False)
        ax.scatter(pm_xy[idx, 0], pm_xy[idx, 1],
                   s=4.5, c="#3a78b8", alpha=0.55,
                   edgecolors="none", zorder=1, rasterized=True)
        all_pts.append(pm_xy)

    # 2. object hulls — filled + obj-id label at centre
    for i, hull in enumerate(obj_hulls):
        all_pts.append(hull)
        color = cmap(i / n_objs)
        face_c = color[:3]
        edge_c = tuple(max(c - 0.15, 0) for c in face_c)
        closed = np.vstack([hull, hull[0]])
        ax.fill(closed[:, 0], closed[:, 1], color=face_c, alpha=0.55, zorder=2)
        ax.plot(closed[:, 0], closed[:, 1], color=edge_c, linewidth=1.0, zorder=3)
        cx, cy = hull[:, 0].mean(), hull[:, 1].mean()
        ax.text(cx, cy, bev_obj["objects"][i]["id"], fontsize=7,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", alpha=0.7, ec="none"))

    # 3. pointmap convex hull — dashed outline
    pm_hull = np.array(bev_pm["hull_xy"])
    all_pts.append(pm_hull)
    closed_pm = np.vstack([pm_hull, pm_hull[0]])
    ax.plot(closed_pm[:, 0], closed_pm[:, 1],
            color="steelblue", linewidth=2.6, linestyle="--", zorder=5)

    # 4. camera glyph
    all_pts_arr = np.vstack(all_pts)
    scene_extent = max(np.ptp(all_pts_arr[:, 0]),
                       np.ptp(all_pts_arr[:, 1]), 1.0)
    _draw_camera(ax, cam_xy, fwd_xy, fov_deg, scene_extent)

    # 5. axis limits with generous pad → no clipping
    xmin, xmax = all_pts_arr[:, 0].min(), all_pts_arr[:, 0].max()
    ymin, ymax = all_pts_arr[:, 1].min(), all_pts_arr[:, 1].max()
    pad_x = (xmax - xmin) * 0.07 + 0.10
    pad_y = (ymax - ymin) * 0.07 + 0.10
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    _strip_axis(ax)
    out_png = out_dir / "bev_compare.png"
    fig.savefig(str(out_png), dpi=240, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# figure 2 — bev_combined_hull style (pointmap + objects + camera + combined
#                                      convex hull, thicker pointmap dots,
#                                      NO axis/legend, no clipping)
# ---------------------------------------------------------------------------

def make_combined_hull(scene_dir: Path, out_dir: Path) -> Path:
    json_dir = scene_dir / "json"
    bev_obj = json.loads((json_dir / "bev_objects.json").read_text())
    pm_xy, _pm_rgb, _pm_z = _load_pointmap(scene_dir)

    cam_xy = np.asarray(bev_obj["camera_xy"], dtype=float)
    yaw_rad = math.radians(float(bev_obj["camera_yaw_deg"]))
    fwd_xy = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    fov_deg = float(bev_obj["fov_deg"])

    obj_hulls = [np.asarray(o["hull_xy"]) for o in bev_obj.get("objects", [])]
    obj_pts = (np.concatenate(obj_hulls, axis=0)
               if obj_hulls else np.empty((0, 2)))
    cam_disk = np.array([
        [cam_xy[0] + 0.10 * math.cos(k * math.pi / 4),
         cam_xy[1] + 0.10 * math.sin(k * math.pi / 4)] for k in range(8)
    ])

    rng = np.random.default_rng(0)
    if len(pm_xy):
        ds = rng.choice(len(pm_xy), min(10_000, len(pm_xy)), replace=False)
        pc_for_hull = pm_xy[ds]
    else:
        pc_for_hull = pm_xy
    combined = np.concatenate([pc_for_hull, obj_pts, cam_disk], axis=0)
    if len(combined) < 3:
        raise SystemExit("not enough points for combined hull")
    hull = combined[ConvexHull(combined).vertices]
    hull_closed = np.vstack([hull, hull[:1]])

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_aspect("equal")

    # pointmap scatter — thick
    if len(pm_xy):
        n_show = min(25_000, len(pm_xy))
        idx = rng.choice(len(pm_xy), n_show, replace=False)
        ax.scatter(pm_xy[idx, 0], pm_xy[idx, 1],
                   s=4.5, c="#3a78b8", alpha=0.50,
                   edgecolors="none", zorder=1, rasterized=True)

    # object hulls — filled + obj_id label
    cmap = plt.get_cmap("tab20")
    n_objs = max(len(obj_hulls), 1)
    for i, h in enumerate(obj_hulls):
        h_closed = np.vstack([h, h[:1]])
        ax.fill(h_closed[:, 0], h_closed[:, 1], alpha=0.45,
                color=cmap(i / n_objs), edgecolor="black", linewidth=0.6,
                zorder=2)
        cx, cy = h[:, 0].mean(), h[:, 1].mean()
        ax.text(cx, cy, bev_obj["objects"][i]["id"], fontsize=7,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15",
                          fc="white", alpha=0.7, ec="none"))

    # combined hull — red outline + light fill
    ax.plot(hull_closed[:, 0], hull_closed[:, 1],
            color="red", linewidth=3.0, zorder=5)
    ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.06,
            color="red", zorder=1.5)

    # camera glyph
    scene_extent = max(np.ptp(combined[:, 0]),
                       np.ptp(combined[:, 1]), 1.0)
    _draw_camera(ax, cam_xy.tolist(), fwd_xy, fov_deg, scene_extent)

    # axis limits — generous pad so nothing clips
    all_pts = np.vstack([combined, hull, cam_xy[None, :]])
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad_x = (xmax - xmin) * 0.07 + 0.10
    pad_y = (ymax - ymin) * 0.07 + 0.10
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    _strip_axis(ax)
    out_png = out_dir / "bev_combined_hull.png"
    fig.savefig(str(out_png), dpi=240, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# figure 3 — pointmap rendered with its own per-vertex texture/colour, BEV.
# Also includes the camera glyph for context. NO axis, NO legend.
# ---------------------------------------------------------------------------

def make_pointmap_textured(scene_dir: Path, out_dir: Path) -> Path:
    pm_xy, pm_rgb, pm_z = _load_pointmap(scene_dir)
    if not len(pm_xy):
        raise SystemExit("pointmap empty")

    # camera info from blender_scene.json
    cam = json.loads((scene_dir / "json" / "blender_scene.json").read_text()).get("camera", {})
    cam_xy = [float(cam.get("location", [0, 0, 0])[0]),
              float(cam.get("location", [0, 0, 0])[1])]
    fwd_xy = _camera_fwd_xy(cam)
    fov_deg = float(math.degrees(2.0 * math.atan(
        cam.get("sensor_width", 36.0) / (2.0 * cam.get("lens", 26.0))
    )))

    # Z-sort so points with greater Z (closer to ceiling) draw on top —
    # gives a stable view that preserves the original colour information.
    order = np.argsort(pm_z)
    pm_xy = pm_xy[order]
    pm_rgb = pm_rgb[order]

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")

    # Bigger marker for paper readability
    ax.scatter(pm_xy[:, 0], pm_xy[:, 1],
               s=4.5, c=pm_rgb, alpha=0.85,
               edgecolors="none", zorder=1, rasterized=True)

    scene_extent = max(np.ptp(pm_xy[:, 0]), np.ptp(pm_xy[:, 1]), 1.0)
    _draw_camera(ax, cam_xy, fwd_xy, fov_deg, scene_extent)

    all_pts = np.vstack([pm_xy, [cam_xy]])
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad_x = (xmax - xmin) * 0.07 + 0.10
    pad_y = (ymax - ymin) * 0.07 + 0.10
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    _strip_axis(ax)
    out_png = out_dir / "bev_pointmap_textured.png"
    fig.savefig(str(out_png), dpi=240, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# figure 4 — pointmap-only BEV: thicker scatter dots in steel-blue + dashed
# convex hull + camera glyph. NO axis, NO legend.
# ---------------------------------------------------------------------------

def make_bev_pointmap(scene_dir: Path, out_dir: Path) -> Path:
    pm_xy, _pm_rgb, _pm_z = _load_pointmap(scene_dir)
    if not len(pm_xy):
        raise SystemExit("pointmap empty")

    bev_pm_json = scene_dir / "json" / "bev_pointmap.json"
    if bev_pm_json.exists():
        hull = np.asarray(json.loads(bev_pm_json.read_text())["hull_xy"])
    else:
        hull = pm_xy[ConvexHull(pm_xy).vertices]

    cam = json.loads((scene_dir / "json" / "blender_scene.json").read_text()).get("camera", {})
    cam_xy = [float(cam.get("location", [0, 0, 0])[0]),
              float(cam.get("location", [0, 0, 0])[1])]
    fwd_xy = _camera_fwd_xy(cam)
    fov_deg = float(math.degrees(2.0 * math.atan(
        cam.get("sensor_width", 36.0) / (2.0 * cam.get("lens", 26.0))
    )))

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")

    rng = np.random.default_rng(42)
    n_show = min(25_000, len(pm_xy))
    idx = rng.choice(len(pm_xy), n_show, replace=False)
    ax.scatter(pm_xy[idx, 0], pm_xy[idx, 1],
               s=4.5, c="#3a78b8", alpha=0.55,
               edgecolors="none", zorder=1, rasterized=True)

    closed = np.vstack([hull, hull[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="steelblue",
            linewidth=2.6, linestyle="--", zorder=3)

    scene_extent = max(np.ptp(pm_xy[:, 0]), np.ptp(pm_xy[:, 1]), 1.0)
    _draw_camera(ax, cam_xy, fwd_xy, fov_deg, scene_extent)

    all_pts = np.vstack([pm_xy, hull, [cam_xy]])
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad_x = (xmax - xmin) * 0.07 + 0.10
    pad_y = (ymax - ymin) * 0.07 + 0.10
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    _strip_axis(ax)
    out_png = out_dir / "bev_pointmap.png"
    fig.savefig(str(out_png), dpi=240, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# figure 5 — annotated floor-plan polygon: solid black for WALL, dotted for
# OPEN; light-grey object hulls + thick blue pointmap dots underneath for
# context; per-edge W# / OPEN labels stay (those are the figure's content);
# vertex (vN) labels removed. NO axis, NO legend.
# ---------------------------------------------------------------------------

def make_floor_plan(scene_dir: Path, out_dir: Path) -> Path:
    json_dir = scene_dir / "json"
    poly = json.loads((json_dir / "polygon_v2.json").read_text())
    verts = poly["polygon_vertices"]
    n = len(verts)
    wall_edges = poly.get("wall_edges", [])
    open_edges = poly.get("open_edges", [])
    yaw_deg = poly.get("yaw_deg", 0.0)
    cam_xy_poly = poly.get("camera_xy", [0.0, 0.0])

    open_set = set()
    for e in open_edges:
        open_set.add((int(e["from"]), int(e["to"])))
        open_set.add((int(e["to"]), int(e["from"])))
    wall_label_map = {(e["from"], e["to"]): e["object"] for e in wall_edges}

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_aspect("equal")

    all_pts = [np.array(verts), np.array([cam_xy_poly])]

    # Pointmap scatter underneath — thick dots, low alpha so polygon stays readable
    pm_xy, _pm_rgb, _pm_z = _load_pointmap(scene_dir)
    if len(pm_xy):
        rng = np.random.default_rng(42)
        n_show = min(25_000, len(pm_xy))
        idx = rng.choice(len(pm_xy), n_show, replace=False)
        ax.scatter(pm_xy[idx, 0], pm_xy[idx, 1],
                   s=4.0, c="#3a78b8", alpha=0.30,
                   edgecolors="none", zorder=0, rasterized=True)
        all_pts.append(pm_xy)

    # Object hulls — light grey fill (context)
    bev_obj_path = json_dir / "bev_objects.json"
    fov_deg, cam_yaw = 60.0, yaw_deg
    if bev_obj_path.exists():
        bev_obj = json.loads(bev_obj_path.read_text())
        fov_deg = bev_obj.get("fov_deg", 60.0)
        cam_yaw = bev_obj.get("camera_yaw_deg", yaw_deg)
        for obj in bev_obj.get("objects", []):
            hull = np.array(obj["hull_xy"])
            all_pts.append(hull)
            closed = np.vstack([hull, hull[0]])
            ax.fill(closed[:, 0], closed[:, 1], color="lightgrey",
                    alpha=0.40, zorder=1)
            ax.plot(closed[:, 0], closed[:, 1], color="#666666",
                    linewidth=0.6, zorder=1.5)

    # Polygon edges — solid for WALL, dotted for OPEN
    wall_num = 0
    for i in range(n):
        v_from, v_to = i, (i + 1) % n
        xa, ya = verts[v_from]
        xb, yb = verts[v_to]
        is_open = (v_from, v_to) in open_set
        mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        if is_open:
            ax.plot([xa, xb], [ya, yb], color="black", linewidth=2.5,
                    linestyle=(0, (1, 3)), zorder=5)
            ax.text(mx, my, "OPEN", fontsize=8, color="red",
                    ha="center", va="bottom", zorder=6, fontweight="bold")
        else:
            ax.plot([xa, xb], [ya, yb], color="black", linewidth=2.5, zorder=5)
            label = wall_label_map.get((v_from, v_to), "")
            if not label:
                wall_num += 1
                label = f"W{wall_num}"
            else:
                label = label.replace("Wall_", "W").lstrip("W0") or "W"
                label = f"W{int(label)}" if label.isdigit() else label
            ax.text(mx, my, label, fontsize=8, color="black",
                    ha="center", va="bottom", zorder=6,
                    bbox=dict(fc="white", alpha=0.7, ec="none", pad=0.15))

    # Vertex dots only — no v# labels
    for x, y in verts:
        ax.plot(x, y, "o", color="black", markersize=5, zorder=7)

    scene_extent = max(np.ptp(np.array(verts)[:, 0]),
                       np.ptp(np.array(verts)[:, 1]), 1.0)
    _draw_camera(ax, cam_xy_poly, np.array(
        [math.cos(math.radians(cam_yaw)),
         math.sin(math.radians(cam_yaw))]
    ), fov_deg, scene_extent)

    all_pts_arr = np.vstack(all_pts)
    xmin, xmax = all_pts_arr[:, 0].min(), all_pts_arr[:, 0].max()
    ymin, ymax = all_pts_arr[:, 1].min(), all_pts_arr[:, 1].max()
    pad_x = (xmax - xmin) * 0.07 + 0.10
    pad_y = (ymax - ymin) * 0.07 + 0.10
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    _strip_axis(ax)
    out_png = out_dir / "floor_plan.png"
    fig.savefig(str(out_png), dpi=240, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--out-subdir", default="report",
                    help="Subdirectory under scene_dir for the figures")
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()
    out_dir = scene_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    a = make_bev_compare(scene_dir, out_dir)
    b = make_combined_hull(scene_dir, out_dir)
    c = make_pointmap_textured(scene_dir, out_dir)
    d = make_bev_pointmap(scene_dir, out_dir)
    e = make_floor_plan(scene_dir, out_dir)

    print("[report] saved figures:")
    for p in (a, b, c, d, e):
        print(f"  {p}")


if __name__ == "__main__":
    main()
