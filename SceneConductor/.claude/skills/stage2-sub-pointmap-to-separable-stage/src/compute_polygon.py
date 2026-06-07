"""Compute the floor polygon and write polygon_v2.json.

Decision tree:
  1) If `floor_plan_draft.json` from the vision agent passes validation
     (rectilinear, contains every object hull + camera with clearance),
     use it as the source of truth.
  2) Otherwise run the algorithmic fitter:
       a. yaw = MABR yaw of (object hulls ∪ camera disk)
       b. Search rectilinear polygons with 4 / 6 / 8 vertices in the natural
          frame (AABB minus 0–2 corner cuts), pick the smallest area whose
          containment of the OBJECT hulls + camera holds.
       c. Pointmap is intentionally NOT a containment constraint — the
          pointmap covers only the visible floor patch and would over-
          constrain the polygon. It is consumed earlier (extract_inputs)
          for ceiling_z and shown in BEV plots for context.
  3) Classify each polygon edge:
       - WALL with evidence  : a wall-mounted object hull is within
         WM_NEAR_THRESH of the edge.
       - WALL (default)      : everything else.
     OPEN edges are NOT auto-detected here; they only come from the
     vision agent's `floor_plan_draft.json` when the image clearly shows
     a doorway / archway on that side.

Wall-mount classes are read from `<scene_dir>/inputs/object_class.json`
and matched against a whitelist (TV, picture, window,
mirror, radiator, etc.) with a small negative list (e.g. "tv stand" is NOT
wall-mounted despite containing "tv").
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from shapely import box as shapely_box
from shapely.geometry import LineString, MultiPoint, Point, Polygon as SPoly
from scipy.spatial import ConvexHull

# Hard constants (same as before — preserve schema)
WALL_THICKNESS       = 0.25
FLOOR_THICKNESS      = 0.30
CEILING_THICKNESS    = 0.30
BUFFER_M             = 0.10  # outward buffer around constraint hull
CAMERA_DISK_R        = 0.10
CAM_EXPAND_STEP      = 0.20
CAM_EXPAND_MAX_ITERS = 10
FLOOR_Z_CAP          = -0.05
CEILING_MIN_Z        = 0.05
ANGLE_TOL_DEG        = 2.0
CLEARANCE_M          = 0.06          # validation slack against draft polygon
MAX_VERTS_DEFAULT    = 6             # 8-vert is opt-in; the search is N⁴ on
                                       # candidate count and rarely buys much.
MAX_CANDIDATES_PER_CORNER = 6        # top-K by cut area; bounds total work
MIN_AREA_IMPROVEMENT = 0.005         # accept 6→8 only if it saves ≥0.5%
WM_NEAR_THRESH       = 0.45          # m, wall-mount adjacency

# Wall-mount class whitelist (substring + token match), with negatives.
# Any object whose class matches this list is treated as STRONG evidence that
# a wall must exist on the polygon edge nearest its hull. The classifier marks
# that edge WALL with `wall_mount_evidence` and the agent prompt is told to
# never mark such an edge OPEN.
WALL_MOUNT_CLASSES = {
    "tv", "television", "monitor", "screen", "display",
    "picture", "painting", "frame", "picture frame", "framed picture",
    "art", "artwork", "poster", "wall art",
    "mirror", "clock", "wall clock",
    "wall shelf", "wall_shelf", "wall light", "wall lamp", "wall_lamp",
    "sconce", "wall sconce",
    "window", "door", "doorway",
    "radiator", "heater",
    # Strong wall-adjacency cues — these almost always sit against / inside
    # a wall in real interiors:
    "fireplace", "fire place", "hearth", "mantel", "mantle", "fire surround",
    "curtain", "curtains", "drape", "drapes", "blind", "blinds",
    "window shade", "window blind", "roller blind",
    "shelf", "shelves", "bookshelf", "book shelf", "bookcase", "shelving",
    "shelving unit", "built-in shelf", "built in shelf",
}
WALL_MOUNT_TOKENS = {
    "tv", "television", "monitor", "picture", "painting", "mirror",
    "window", "radiator", "sconce", "poster", "frame", "framed",
    "fireplace", "hearth", "mantel", "mantle",
    "curtain", "drape", "blind",
    "shelf", "bookshelf", "bookcase", "shelving",
}
WALL_MOUNT_NEGATIVES = (
    "tv stand", "tv_stand", "tv unit", "media console", "media stand",
    "picture frame stand",
    # freestanding furniture that contains a token but is not wall-bound
    "shelf unit on wheels", "rolling cart shelf",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--max-verts", type=int, default=MAX_VERTS_DEFAULT,
                    choices=[4, 6, 8])
    ap.add_argument("--margin", type=float, default=BUFFER_M)
    ap.add_argument("--min-improvement", type=float, default=MIN_AREA_IMPROVEMENT)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def signed_area(pts) -> float:
    n = len(pts)
    a = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def ccwify(verts):
    return verts if signed_area(verts) > 0 else list(reversed(list(verts)))


def right_angle_check(verts, tol_deg: float = ANGLE_TOL_DEG) -> bool:
    n = len(verts)
    tol_rad = math.radians(tol_deg)
    for i in range(n):
        ax, ay = verts[(i - 1) % n]
        bx, by = verts[i]
        cx, cy = verts[(i + 1) % n]
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        m1 = math.hypot(*v1); m2 = math.hypot(*v2)
        if m1 < 1e-9 or m2 < 1e-9:
            return False
        cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)))
        if abs(math.acos(cos_a) - math.pi / 2.0) > tol_rad:
            return False
    return True


def contains_with_clearance(polygon: SPoly, points, clearance_m: float) -> bool:
    for pt in points:
        p = Point(float(pt[0]), float(pt[1]))
        if not polygon.contains(p):
            return False
        if polygon.exterior.distance(p) < clearance_m:
            return False
    return True


def cam_disk_points(cx=0.0, cy=0.0, r: float = CAMERA_DISK_R, n: int = 8):
    return [[cx + r * math.cos(k * 2 * math.pi / n),
             cy + r * math.sin(k * 2 * math.pi / n)] for k in range(n)]


def polygon_centroid(verts) -> list:
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


# ---------------------------------------------------------------------------
# Wall-mount class matching
# ---------------------------------------------------------------------------

def class_is_wall_mounted(name: str) -> bool:
    if not name:
        return False
    n = str(name).lower().strip()
    for neg in WALL_MOUNT_NEGATIVES:
        if neg in n:
            return False
    if n in WALL_MOUNT_CLASSES:
        return True
    for tok in n.replace("_", " ").split():
        if tok in WALL_MOUNT_TOKENS:
            return True
    return False


def load_class_map(scene_dir: Path) -> dict:
    p = scene_dir / "inputs" / "object_class.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def obj_index_from_id(obj_id: str) -> str:
    s = str(obj_id)
    return s.split("_")[-1] if "_" in s else s


# ---------------------------------------------------------------------------
# Fitter — MABR yaw + 4/6/8-vert rectilinear corner-cut search
# ---------------------------------------------------------------------------

def fit_mabr_yaw(points: np.ndarray) -> float:
    rect = MultiPoint(points.tolist()).minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]
    if signed_area(coords) < 0:
        coords = coords[::-1]
    coords = np.asarray(coords, dtype=float)
    edges = [coords[(i + 1) % 4] - coords[i] for i in range(4)]
    long_idx = int(np.argmax([np.linalg.norm(e) for e in edges]))
    long_e = edges[long_idx]
    yaw = math.degrees(math.atan2(long_e[1], long_e[0]))
    while yaw > 90:  yaw -= 180
    while yaw <= -90: yaw += 180
    return yaw


def search_polygon(constraint_pts: np.ndarray, max_verts: int, margin: float,
                    min_improvement: float):
    """Return (yaw_deg, world_verts, area, label, search_areas)."""
    yaw = fit_mabr_yaw(constraint_pts)
    yaw_rad = math.radians(yaw)
    R_in  = np.array([[math.cos(-yaw_rad), -math.sin(-yaw_rad)],
                      [math.sin(-yaw_rad),  math.cos(-yaw_rad)]])
    R_out = R_in.T
    pts_nat = constraint_pts @ R_in.T

    x_lo = float(pts_nat[:, 0].min()) - margin
    y_lo = float(pts_nat[:, 1].min()) - margin
    x_hi = float(pts_nat[:, 0].max()) + margin
    y_hi = float(pts_nat[:, 1].max()) + margin
    base_aabb = np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])

    # Convex-hull region in natural frame — tells us when a non-convex polygon
    # would slice through the interior. Validating only hull verts isn't
    # enough because rectilinear polygons can be concave.
    hull_idx = ConvexHull(pts_nat).vertices
    hull_nat_polygon = SPoly(pts_nat[hull_idx]).buffer(0)

    CORNER_INFO = {"br": (+1, -1), "bl": (-1, -1), "tr": (+1, +1), "tl": (-1, +1)}

    def cut_rect(corner, ix, iy):
        sx, sy = CORNER_INFO[corner]
        ax = x_hi if sx > 0 else x_lo
        ay = y_hi if sy > 0 else y_lo
        return shapely_box(min(ax, ix), min(ay, iy), max(ax, ix), max(ay, iy))

    x_mid = (x_lo + x_hi) / 2.0
    y_mid = (y_lo + y_hi) / 2.0

    def candidates_for_corner(corner):
        # Only points in the corner's half-plane can constrain a useful cut at
        # that corner. This drops the candidate enumeration from O(N²) over all
        # points to O(M²) with M ≈ N/4 — turning the 8-vert search from O(N⁴)
        # (minutes for ~30 hull points) into O(M⁴) (sub-second).
        sx, sy = CORNER_INFO[corner]
        ax = x_hi if sx > 0 else x_lo
        ay = y_hi if sy > 0 else y_lo
        if sx > 0:
            x_filter = pts_nat[:, 0] >= x_mid
        else:
            x_filter = pts_nat[:, 0] <= x_mid
        if sy > 0:
            y_filter = pts_nat[:, 1] >= y_mid
        else:
            y_filter = pts_nat[:, 1] <= y_mid
        x_anchors = pts_nat[x_filter]
        y_anchors = pts_nat[y_filter]
        out = []
        seen = set()
        for px_a in x_anchors:
            for py_a in y_anchors:
                ix = float(px_a[0]) + sx * margin
                iy = float(py_a[1]) + sy * margin
                if sx > 0 and ix >= ax - 1e-6: continue
                if sx < 0 and ix <= ax + 1e-6: continue
                if sy > 0 and iy >= ay - 1e-6: continue
                if sy < 0 and iy <= ay + 1e-6: continue
                key = (round(ix, 4), round(iy, 4))
                if key in seen: continue
                cr = cut_rect(corner, ix, iy)
                cr_in = cr.buffer(-margin * 0.5)
                if cr_in.is_empty:
                    seen.add(key); out.append((ix, iy)); continue
                if any(cr_in.contains(Point(float(q[0]), float(q[1]))) for q in pts_nat):
                    continue
                seen.add(key); out.append((ix, iy))
        # Prune to top-K candidates by cut area (largest cuts first). The
        # smallest-area polygon is built from the largest cuts; smaller cuts
        # rarely win and dominate the runtime in 8-vert search.
        if len(out) > MAX_CANDIDATES_PER_CORNER:
            ax_corner = ax; ay_corner = ay
            out.sort(key=lambda t: -abs(ax_corner - t[0]) * abs(ay_corner - t[1]))
            out = out[:MAX_CANDIDATES_PER_CORNER]
        return out

    def polygon_after(cuts):
        poly = SPoly(base_aabb)
        for c, ix, iy in cuts:
            poly = poly.difference(cut_rect(c, ix, iy))
        if poly.is_empty or poly.geom_type != "Polygon" or any(poly.interiors):
            return None
        return poly

    def is_valid(poly):
        if poly.is_empty: return False
        if not hull_nat_polygon.within(poly.buffer(1e-6)): return False
        for x, y in hull_nat_polygon.exterior.coords[:-1]:
            if poly.boundary.distance(Point(x, y)) < margin * 0.5:
                return False
        return True

    candidates = []  # (area, n_verts, label, world_verts)

    rect_poly = SPoly(base_aabb)
    if is_valid(rect_poly):
        verts_world = (np.asarray(list(rect_poly.exterior.coords)[:-1]) @ R_out.T)
        candidates.append((rect_poly.area, 4, "rectangle", ccwify(verts_world.tolist())))

    if max_verts >= 6:
        for c in CORNER_INFO:
            for ix, iy in candidates_for_corner(c):
                p = polygon_after([(c, ix, iy)])
                if p is None: continue
                if len(list(p.exterior.coords)) - 1 != 6: continue
                if is_valid(p):
                    vw = (np.asarray(list(p.exterior.coords)[:-1]) @ R_out.T)
                    candidates.append((p.area, 6, f"L-shape cut={c}", ccwify(vw.tolist())))

    if max_verts >= 8:
        for c1, c2 in combinations(CORNER_INFO.keys(), 2):
            cands1 = candidates_for_corner(c1)
            cands2 = candidates_for_corner(c2)
            for ix1, iy1 in cands1:
                for ix2, iy2 in cands2:
                    p = polygon_after([(c1, ix1, iy1), (c2, ix2, iy2)])
                    if p is None: continue
                    if len(list(p.exterior.coords)) - 1 != 8: continue
                    if is_valid(p):
                        vw = (np.asarray(list(p.exterior.coords)[:-1]) @ R_out.T)
                        candidates.append(
                            (p.area, 8, f"8-vert cuts={c1}+{c2}", ccwify(vw.tolist()))
                        )

    if not candidates:
        raise RuntimeError("polygon search produced no valid candidate")

    candidates.sort(key=lambda c: c[0])
    best_4 = next((c for c in candidates if c[1] == 4), None)
    best_6 = next((c for c in candidates if c[1] == 6), None)
    best_8 = next((c for c in candidates if c[1] == 8), None)

    chosen = best_4
    if best_6 is not None and best_4 is not None and \
       best_6[0] < best_4[0] * (1 - min_improvement):
        chosen = best_6
    if best_8 is not None and chosen is not None and \
       best_8[0] < chosen[0] * (1 - min_improvement):
        chosen = best_8
    area, _, label, world_verts = chosen
    return yaw, world_verts, area, label, {
        "best_4": best_4[0] if best_4 else None,
        "best_6": best_6[0] if best_6 else None,
        "best_8": best_8[0] if best_8 else None,
    }


# ---------------------------------------------------------------------------
# Edge classification (wall-mount evidence only; OPEN comes from agent draft)
# ---------------------------------------------------------------------------

def classify_edges(world_verts, wall_mount_records):
    n = len(world_verts)
    edge_meta = []
    for i in range(n):
        a = np.asarray(world_verts[i])
        b = np.asarray(world_verts[(i + 1) % n])
        seg = LineString([tuple(a), tuple(b)])
        nearby = []
        for r in wall_mount_records:
            dists = [seg.distance(Point(float(p[0]), float(p[1]))) for p in r["hull"]]
            n_close = sum(1 for d in dists if d <= WM_NEAR_THRESH)
            centroid_pt = Point(
                float(sum(p[0] for p in r["hull"]) / len(r["hull"])),
                float(sum(p[1] for p in r["hull"]) / len(r["hull"])),
            )
            d_centroid = seg.distance(centroid_pt)
            if n_close >= 2 or d_centroid <= WM_NEAR_THRESH * 2.5:
                d = min(dists)
                nearby.append({"id": r["id"], "class": r["class"],
                                "dist_m": round(float(d), 3)})
        edge_meta.append({
            "i": i, "a": a, "b": b, "nearby_wm": nearby,
        })

    types = []
    rationales = []
    evidence = []
    for m in edge_meta:
        if m["nearby_wm"]:
            types.append("WALL")
            rationales.append(
                "wall-mount: " + ", ".join(
                    f"{w['id']}({w['class']})" for w in m["nearby_wm"]
                )
            )
            evidence.append(m["nearby_wm"])
        else:
            types.append("WALL")
            rationales.append("default wall (no wall-mount evidence)")
            evidence.append([])
    return types, rationales, evidence


# ---------------------------------------------------------------------------
# Edge-list construction for polygon_v2.json schema
# ---------------------------------------------------------------------------

def build_wall_open_edges(verts, types, rationales, evidence):
    n = len(verts)
    wall_edges = []; open_edges = []
    wall_idx = 0
    for i in range(n):
        v_from = i; v_to = (i + 1) % n
        if types[i] == "OPEN":
            open_edges.append({
                "from": v_from, "to": v_to,
                "reason": rationales[i],
            })
        else:
            wall_idx += 1
            entry = {
                "from": v_from, "to": v_to,
                "object": f"Wall_{wall_idx:02d}",
                "orientation": "unspecified",
            }
            if evidence[i]:
                entry["wall_mount_evidence"] = evidence[i]
                entry["orientation"] = "wall_mount"
            wall_edges.append(entry)
    return wall_edges, open_edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    slug = scene_dir.name

    # Inputs from extract_inputs.py + bev_objects.py
    bboxes  = json.loads(Path(f"/tmp/{slug}_object_bboxes.json").read_text(encoding='utf-8'))
    pc_z    = json.loads(Path(f"/tmp/{slug}_pc_z.json").read_text(encoding='utf-8'))
    camera  = json.loads(Path(f"/tmp/{slug}_camera.json").read_text(encoding='utf-8'))
    bev_obj = json.loads((scene_dir / "json" / "bev_objects.json").read_text(encoding='utf-8'))

    # Z stats
    all_corners = [c for corners in bboxes.values() for c in corners]
    obj_zs = [c[2] for c in all_corners] if all_corners else [0.0]
    obj_min_z = min(obj_zs); obj_max_z = max(obj_zs)
    pc_max_z  = float(pc_z["max"])
    floor_z   = min(obj_min_z, FLOOR_Z_CAP)
    ceiling_z = max(pc_max_z, obj_max_z, CEILING_MIN_Z)

    # Object hulls (mesh-derived) from bev_objects.json — these are tighter
    # than raw AABB corners and are what the BEV plot shows. Camera disk is
    # added so the camera location stays inside the polygon.
    cam_x = float(bev_obj.get("camera_xy", [0.0, 0.0])[0])
    cam_y = float(bev_obj.get("camera_xy", [0.0, 0.0])[1])
    obj_records = []
    classes = load_class_map(scene_dir)
    for o in bev_obj.get("objects", []):
        cls = classes.get(obj_index_from_id(o["id"]), "")
        obj_records.append({
            "id": o["id"],
            "hull": np.asarray(o["hull_xy"]),
            "class": cls,
            "wall_mounted": class_is_wall_mounted(cls),
        })
    if not obj_records:
        # Fallback: synthesise from /tmp bboxes (object AABB corners only).
        for name, corners in bboxes.items():
            obj_records.append({
                "id": name, "hull": np.asarray([[c[0], c[1]] for c in corners]),
                "class": "", "wall_mounted": False,
            })
    obj_pts = np.concatenate([r["hull"] for r in obj_records], axis=0)
    constraint_pts = np.concatenate(
        [obj_pts, np.asarray(cam_disk_points(cam_x, cam_y))], axis=0
    )
    wall_mount_records = [r for r in obj_records if r["wall_mounted"]]

    # ----- Try agent draft first ------------------------------------------
    draft_path = scene_dir / "json" / "floor_plan_draft.json"
    draft = None
    if draft_path.exists():
        try:
            draft = json.loads(draft_path.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f"[compute_polygon] WARNING reading draft: {exc}")

    yaw_deg = 0.0
    verts = None
    label = ""
    source = "fallback"
    search_areas = {}

    if draft is not None:
        try:
            yaw_deg = float(draft.get("yaw_deg", 0.0))
            dverts = [[float(v[0]), float(v[1])] for v in draft.get("polygon_vertices", [])]
            n = len(dverts)
            ok, reason = True, ""
            if n < 4 or n % 2 != 0:
                ok, reason = False, f"vertex count {n}"
            elif not right_angle_check(dverts):
                ok, reason = False, "non-90° interior angles"
            else:
                poly = SPoly(dverts)
                if not poly.contains(Point(cam_x, cam_y)):
                    ok, reason = False, "camera not inside polygon"
                elif not contains_with_clearance(poly, obj_pts.tolist(), CLEARANCE_M):
                    ok, reason = False, "object hulls not enclosed with clearance"
            if ok:
                verts = ccwify(dverts)
                source = "draft"
                label = f"draft ({n} verts)"
                print(f"[compute_polygon] Draft accepted: n={n} yaw={yaw_deg:.2f}°")
            else:
                print(f"[compute_polygon] Draft rejected ({reason}); using fitter")
        except Exception as exc:
            print(f"[compute_polygon] Draft parse failed ({exc}); using fitter")

    # ----- Algorithmic fitter --------------------------------------------
    if verts is None:
        yaw_deg, verts, area, label, search_areas = search_polygon(
            constraint_pts, args.max_verts, args.margin, args.min_improvement
        )
        source = "fitter"
        print(
            f"[compute_polygon] Fitter chose {label}: yaw={yaw_deg:.2f}°  "
            f"area={area:.2f} m²  search={search_areas}"
        )

    # ----- Edge classification -------------------------------------------
    n_verts = len(verts)
    types, rationales, evidence = classify_edges(verts, wall_mount_records)

    # OPEN edges only come from the vision agent's draft (image context).
    # Wall-mount evidence is a hard veto: if a fireplace / window / curtain /
    # shelf / TV / picture / etc. sits within WM_NEAR_THRESH of an edge, the
    # agent cannot mark that edge OPEN — such an object can't exist without
    # a wall behind it. We log the veto for transparency.
    if source == "draft":
        try:
            agent_edges = draft.get("edges", [])
            if len(agent_edges) == n_verts:
                for i, e in enumerate(agent_edges):
                    if e.get("type", "WALL") == "OPEN":
                        if evidence[i]:
                            ev_classes = ", ".join(
                                f"{w['id']}({w['class']})" for w in evidence[i]
                            )
                            print(
                                f"[compute_polygon] vetoed agent OPEN on edge {i}: "
                                f"wall-mount evidence {ev_classes}"
                            )
                            continue
                        types[i] = "OPEN"
                        rationales[i] = (
                            f"agent draft: {e.get('rationale', 'OPEN')}"
                        )
        except Exception:
            pass

    wall_edges, open_edges = build_wall_open_edges(verts, types, rationales, evidence)
    centroid = polygon_centroid(verts)

    out = {
        "polygon_vertices":    [[round(x, 6), round(y, 6)] for x, y in verts],
        "polygon_centroid_xy": [round(centroid[0], 6), round(centroid[1], 6)],
        "floor_z":             round(floor_z, 6),
        "ceiling_z":           round(ceiling_z, 6),
        "wall_thickness":      WALL_THICKNESS,
        "floor_thickness":     FLOOR_THICKNESS,
        "ceiling_thickness":   CEILING_THICKNESS,
        "wall_edges":          wall_edges,
        "open_edges":          open_edges,
        "openings":            [],
        "camera_xy":           [round(cam_x, 6), round(cam_y, 6)],
        "camera_source":       "blender_camera",
        "source_frame":        "blend_world",
        "buffer_m":            args.margin,
        "rect_angle_deg":      round(yaw_deg, 4),
        "yaw_deg":             round(yaw_deg, 4),
        "generator":           "stage2-sub-pointmap-to-separable-stage",
        "fitter": {
            "source":          source,
            "shape":           label,
            "max_verts":       args.max_verts,
            "min_improvement": args.min_improvement,
            "search_areas":    search_areas,
        },
        "wall_mount_objects": [
            {"id": r["id"], "class": r["class"]}
            for r in wall_mount_records
        ],
    }
    out_path = scene_dir / "json" / "polygon_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(
        f"[compute_polygon] yaw={yaw_deg:.2f}° n_verts={n_verts} "
        f"walls={len(wall_edges)} openings={len(open_edges)} "
        f"floor_z={floor_z:.3f} ceiling_z={ceiling_z:.3f} source={source}"
    )


if __name__ == "__main__":
    main()
