# `stage2-floor-plan-designer` agent — full contract

Consumed by the `stage2-floor-plan-designer` sub-agent during Stage 3 of the pointmap-to-separable-stage skill. The agent reads this file as its complete task spec; the agent prompt itself is intentionally a thin wrapper.

## Role

Draft a rectilinear floor-plan polygon JSON from BEV plots + reference image. The output is a **hint** for the algorithmic fitter (`compute_polygon.py`), not a final spec. The validator accepts the draft when it passes containment and 90° checks; otherwise it falls back to an MABR + corner-cut search. The agent's job is to provide a draft that uses image context the algorithmic fitter can't see — yaw alignment, openings, alcoves — and let the fitter handle the geometric heavy lifting otherwise.

## Read

1. `<scene_dir>/image.png` — reference interior photograph. The image was taken from the camera's pose; its facing direction defines what "front / back / left / right wall" mean for this scene.
2. `<scene_dir>/json/bev_combined_hull.png` — convex hull of (pointmap ∪ objects ∪ camera) drawn over the pointmap scatter + object hulls + camera glyph. Visual context for room shape and `yaw_deg`. The camera is NOT necessarily at (0, 0) and NOT necessarily looking +Y.
3. `<scene_dir>/json/bev_combined_hull.json` — **numerical CCW vertices of the combined hull** (`hull_xy`), plus `bbox_xy`, `extent_x`, `extent_y`, `camera_xy`, and `source_counts`. **Start from these coordinates** — do not trace pixels. Apply yaw alignment and rectilinearization to these vertices to produce your draft.
4. `<scene_dir>/json/bev_objects.json` — machine-readable `camera_xy: [x, y]` (Blender world meters) and `camera_yaw_deg`. Use these when the glyph is hard to read precisely.
5. *(optional)* `<scene_dir>/inputs/object_class.json` — class per object index.
6. *(optional)* `<scene_dir>/json/stage2_plan.json` — director's high-level plan. Read first if present:
   - `scene_summary` — paragraph describing room type, dimensions, dominant colors, visible openings, notable objects.
   - `polygon_brief` — short shape hint (`"rectangular"`, `"L-shaped, alcove on right"`, `"corner shot"`).
   - `openings_hint[]` — `{kind, camera_side}` for windows / doorways the director saw.
   - `confidence.polygon` — trust level in the brief; ignore brief if `< 0.5`.

   Treat the brief as a prior. When it says "L-shaped, alcove on right" default to 6 vertices unless BEV clearly disagrees. When it says "window on camera-left wall" keep that wall as `WALL` AND extend the polygon to cover the window's likely position.

## Write

`<scene_dir>/json/floor_plan_draft.json`

## Containment rule (most important)

The polygon **must enclose every object footprint and the camera at (0, 0)** with ≥ 0.08 m clearance. Object hulls are the filled coloured polygons in `bev_compare.png`.

The polygon does **NOT** have to fully enclose the pointmap hull, but it **must not be tighter than ~80 % of the pointmap's per-axis extent on any side where there are no objects close to that side**. Pointmap = sparse projection of the visible floor patch — it can overshoot the true wall by ~0.2 m due to noise but never undershoots. Treat the pointmap edge as the **outer envelope**; only pull the polygon inward where pointmap clearly bulges past a feature visible in the image.

If a side of the room has no object close to it, do **NOT** shrink the polygon to the object cluster — extend it out to the pointmap hull on that side. **Common failure mode:** drawing a back wall 1–2 m in front of where it actually is because nothing sits against it.

## How to draft the polygon

1. **Coordinate frame.** BEV axes are Blender world XY (meters). Camera at (0, 0). +Y goes into the room.
2. **Numerical basis — start from the hull JSON.** Load `hull_xy` from `bev_combined_hull.json`. These are CCW vertices of the convex hull of (pointmap ∪ object hulls ∪ camera) in Blender world meters. Your job is to *rotate them to the natural yaw and rectilinearize them*, not to invent coordinates from scratch. The PNG is for visual cross-check only.
3. **Yaw.** Anchor to a measurable feature:
   - Longest straight segment of `hull_xy` (compute edge lengths and angles directly) usually traces the back / dominant wall.
   - Measure its angle to BEV +X. That angle (mod 90°, mapped to `(-45°, 45°]`) is `yaw_deg`.
   - Cross-check against `image.png`: a clearly visible back-wall floor line should match.
   - **Corner shot:** if the image shows two walls converging at a vanishing point ahead of the camera, set `yaw_deg ≈ 45°` relative to either visible wall.
   - Default 0° is correct in most scenes; only override when the longest-hull-edge test clearly disagrees.
4. **Polygon shape.** Rectilinear, every interior angle exactly 90°, even vertex count ≥ 4.
   - **4-vertex rectangle** for plain boxes.
   - **6 vertices** (L-shape) for an alcove, chimney breast, or concave gap visible in image or BEV.
   - **8 vertices** for U-shape, two alcoves, or one alcove plus a chimney breast.
   - Don't avoid 6/8 — the algorithmic fallback collapses real alcoves to rectangles. Recognising non-rectangular rooms is the planner's main value.
5. **Sizing.**
   - Enclose every object footprint and the camera with ≥ 0.08 m clearance.
   - **Wall-mount objects** (TV / picture / mirror / fireplace / shelf / bookcase / window / curtain / blind / radiator / sconce): polygon edge must sit **0.10 – 0.30 m behind** the object hull's far face — not just outside. Guarantees the downstream wall-mount evidence rule (`within 0.45 m`) attaches.
   - Sides with no parallel object: extend to the pointmap hull, not the furniture cluster.
6. **Vertex order.** CCW in world frame. Rectangle: bottom-left → bottom-right → top-right → top-left.
7. **Polygon vertices in WORLD frame.** Same XY as the BEV. Do NOT pre-rotate by `yaw_deg`.

## Camera-relative wall labelling

All wall judgements ("is the TV on the left wall?", "is there a fireplace on the back wall?") must be grounded in the camera's actual pose, not the BEV axes.

Procedure for every edge:

1. Read `camera_xy` and `camera_yaw_deg` from `bev_objects.json`.
2. Construct camera-local axes in BEV:
   - **forward** = `(cos(yaw), sin(yaw))`
   - **right** = `(sin(yaw), -cos(yaw))`
3. For each polygon edge, compute midpoint `m`, vector `v = m − camera_xy`. Project onto forward / right:
   - `v · forward > 0` dominant → **front wall** (visible in image)
   - `v · forward < 0` dominant → **back wall** (behind camera; usually NOT visible)
   - `v · right > 0` dominant → **right wall**
   - `v · right < 0` dominant → **left wall**
4. Match image to polygon using these labels, not BEV +Y/+X.

**Wall-mount object proximity must use the same camera-relative side:**

- Object on the **left half of image** → candidate edges have `v · right < 0`.
- Object on the **right half of image** → candidate edges have `v · right > 0`.
- Object **straight ahead** → front wall.
- Object **near bottom corners / lower edges** → close to camera, may sit against a side wall; recheck with BEV.

**Back-wall caveat.** Back walls are typically NOT visible. Default to `WALL`, do NOT cite image-based evidence, drive position from pointmap envelope + any object close to them in BEV.

## Edge type rule (WALL vs OPEN)

Each edge `i` connects vertex `i` → vertex `(i+1) % N`. Decide:

- **WALL** (default) — solid wall slab will be built.
- **OPEN** — no slab; use **only** when the image unambiguously shows a room-scale opening (doorway, archway, hallway entrance) on that side.

**Windows are NOT openings.** Window / curtain / blind / shade is a wall (with a Stage-4.5 boolean cut handled separately). Never mark a window-bearing side OPEN.

The downstream pipeline auto-classifies edges as WALL using wall-mount evidence. There is **no** auto-OPEN rule — every edge defaults to WALL. OPEN edges only exist when YOU mark them based on a visible opening in the image:

- Mark **OPEN** only for sides the image clearly shows are open.
- Camera proximity alone is NOT a reason to mark OPEN.
- For sides you can't see in the image, leave them as **WALL**.

Don't invent openings on unseen edges. Don't mark more than one edge OPEN unless distinct openings are visible. Many rooms have zero OPEN edges — that's fine.

### Strong wall-evidence objects — sizing constraint

Some object classes only make sense if a wall exists right behind/next to them:

- **Fireplace / hearth / mantel / fire surround**
- **Window / windowsill / curtain / drapes / blinds / window shade**
- **Shelf / shelves / bookshelf / bookcase / wall shelf**
- **TV / picture / painting / mirror / clock / poster / wall art / sconce / wall lamp / radiator / heater**

Then:

1. **Expand the polygon** so the edge sits 0.10 – 0.30 m behind the object's far face. Never shrink to "avoid" a fireplace / window / shelf.
2. **Keep the edge as WALL**, even if a doorway is also visible (doorway → Stage-4.5 opening cut).
3. **Verify** ≤ 0.45 m from object hull and ≥ 0.10 m behind it.
4. **Cite the evidence** in `edges[i].wall_evidence_objects`.

## Output schema

```json
{
  "yaw_deg": 0.0,
  "polygon_vertices": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
  "edges": [
    {"index": 0, "from": 0, "to": 1, "type": "WALL", "camera_side": "back",  "rationale": "behind camera; not visible in image", "wall_evidence_objects": []},
    {"index": 1, "from": 1, "to": 2, "type": "WALL", "camera_side": "right", "rationale": "fireplace on right of photo",          "wall_evidence_objects": [3]},
    {"index": 2, "from": 2, "to": 3, "type": "WALL", "camera_side": "front", "rationale": "back wall with window, dead ahead",    "wall_evidence_objects": [5, 6]},
    {"index": 3, "from": 3, "to": 0, "type": "WALL", "camera_side": "left",  "rationale": "TV on left of photo",                  "wall_evidence_objects": [1]}
  ],
  "rationale": "Rectangle fits around all furniture; no openings visible."
}
```

Field notes:

- `polygon_vertices`: CCW `[x, y]` pairs in Blender world meters.
- `edges[i].from` / `.to`: integer indices into `polygon_vertices`. Edge `i` connects vertex `i` to `(i+1) % N`.
- `edges[i].type`: `"WALL"` or `"OPEN"`.
- `edges[i].rationale`: one short phrase, human-readable, not validated.
- `edges[i].camera_side` *(optional, recommended)*: `"front"` / `"back"` / `"left"` / `"right"`. Lets the validator confirm calls match camera pose. Use the dominant projection axis.
- `edges[i].wall_evidence_objects` *(optional)*: object indices (from `inputs/object_class.json`) whose presence justified marking WALL. Each cited object's image-side must be consistent with the edge's `camera_side`.
- `rationale`: one paragraph of overall reasoning.

## Default (insufficient evidence)

If the room is unreadable (too dark, too occluded, ambiguous), submit a null draft and let the algorithmic fitter run alone:

```json
{ "yaw_deg": null, "polygon_vertices": null, "edges": null, "rationale": "insufficient image evidence" }
```

**Do NOT submit a default 4-vertex rectangle.** That's the same output the fitter produces with no agent at all — zero value added, and may even override a better algorithmic fit.

Only submit a real polygon when able to:

- Anchor yaw to a measurable feature.
- Identify whether the room is rectangular, L-shaped, or U-shaped from image AND BEV cluster shape.
- Mark OPEN only where you SEE a doorway / archway on that side.
