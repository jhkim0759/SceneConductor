---
name: stage2-sub-pointmap-to-separable-stage
description: Build a separable stage (Floor, Wall_NN, Ceiling) inside a `.blend` by committing a rectilinear floor polygon in 2D from a pointmap + image, then extruding in 3D. Also vendors the layout_prediction.json → blender_scene.json → .blend converters (`src/convert.py`, `src/build.py`). Trigger on "/stage2-sub-pointmap-to-separable-stage".
agents:
  - stage2-floor-plan-designer   # Phase 5, optional polygon draft before compute_polygon.py
---

## What this skill does

Given `<scene_dir>/` with `image.png`, `inputs/pointmap_xz.ply`, `json/blender_scene.json`, and `blend/blender_scene.blend` (containing an imported `PointCloud_XZ` mesh + `geometry_*` objects), produces a tight, axis-rectified room shell in two phases:

- **2D phase** — render BEVs (object footprints, pointmap hull, combined-hull reference draft); a vision agent drafts a rectilinear polygon with WALL/OPEN edges; an algorithmic fitter (MABR yaw + 4/6/8-vert corner-cut search) validates the draft or replaces it with the smallest-area valid polygon. Per-edge type is then auto-classified using object class info (TV / picture / window / radiator / etc. → WALL with evidence).
- **3D phase** — Floor (full polygon) → Wall_NN per WALL edge (skip OPEN edges) → Ceiling, delete `PointCloud_XZ`, merge a `stage` block into `blender_scene.json`, save the blend.

The output blend is a finished, separable stage; downstream `stage2-sub-env-enhance` owns lighting / materials / colors.

## Hard rules

These are baked into the scripts and reflect lessons learned from many test scenes.

- **Coordinate fix for the pointmap.** `pointmap_xz.ply` is in mesh frame; convert to Blender frame with `(x, y, z) → (-x, z, y)` and multiply by `meta.world_scale_factor` from `blender_scene.json`. The trimesh→Blender vertex permutation matches the rest of the SceneConductor pipeline; the scale matches whatever `src/convert.py` applied to objects. Skipping either step makes the pointmap extent wrong by orders of magnitude.
- **Containment scope: objects + camera, not pointmap.** The polygon must enclose every object hull and the camera at world (0, 0) with ≥ 0.08 m clearance. The pointmap is a sparse projection of the visible floor patch and often fans out past where walls actually are, so requiring it to be enclosed over-inflates the polygon. Pointmap is treated as a *reference draft* (visualised in `bev_combined_hull.png`) but not a hard constraint.
- **Rectilinear, ≤ 8 vertices.** Every interior angle is 90°. Vertex count is even and at most 8 (4-vert rectangle, 6-vert L, or 8-vert two-cut shape). The fitter searches all three and picks the smallest area, with a small-improvement gate (≥ 0.5 %) so 8-vert is only chosen when it meaningfully beats 6-vert.
- **Yaw via MABR of objects + camera.** The minimum-area rotated rectangle of (object hulls ∪ camera disk) gives the natural-frame yaw. Pointmap is excluded from yaw because its fan shape biases toward the camera frustum diagonal rather than the room's true axis. The vision agent can override this yaw based on image evidence.
- **Wall-mount edge classification.** `inputs/object_class.json` maps object indices to class strings. Classes matching a wall-mount whitelist tag any polygon edge within 0.45 m of that object's hull as WALL with `wall_mount_evidence` listing the contributing object IDs and classes. The whitelist covers four families:
   1. **Hung-on-wall**: TV / television / monitor / picture / painting / mirror / clock / poster / wall art / sconce / wall lamp / radiator / heater.
   2. **Wall openings & coverings**: window / door / doorway / curtain / drape / blinds / window shade.
   3. **Built-in / wall-set features**: fireplace / hearth / mantel / fire surround.
   4. **Wall-adjacent shelving**: shelf / shelves / bookshelf / bookcase / shelving / wall shelf (treated as wall-adjacent because in real interiors these almost always back onto a wall).
   Negatives like "tv stand" / "media console" are excluded so free-standing furniture isn't misclassified. OPEN edges are NOT auto-detected; every edge defaults to WALL unless the vision agent's `floor_plan_draft.json` explicitly marks it OPEN based on a doorway / archway visible in the image. Edges with wall-mount evidence are NEVER overridden to OPEN — even by the agent — because such an object can't exist without a wall behind it.
- **Floor / ceiling Z.** `floor_z = min(min_object_z, -0.05)`. `ceiling_z = max(pointmap_max_z, max_object_z, 0.05)` — no upper cap, so the ceiling is always above every object and every point cloud sample. Single-constant-Z planes — no per-vertex tilt.
- **Wall thickness + separability.** Each WALL edge → `Wall_NN` (flat quad → Solidify thickness=0.25, offset=-1.0, inward normal → Apply). Floor / Ceiling slabs thickness=0.30. All in `Stage` collection, each with its own mesh datablock.
- **Openings vs. window-cuts.** OPEN edges produce no wall slab. Per-wall window/door cuts (`openings[]`) belong to Stage 4.5 and are not authored here.
- **Three placeholder materials.** `Mat_Floor`, `Mat_Wall`, `Mat_Ceiling` (Principled BSDF, roughness=1.0, neutral grey, `use_fake_user=True`). Final colors are deferred to `stage2-sub-env-enhance`.
- **PointCloud_XZ removed at the end.** Render noise + file bloat; the original PLY remains on disk.

## Inputs

| Path | Required |
|---|---|
| `<scene_dir>/image.png` | yes |
| `<scene_dir>/inputs/pointmap_xz.ply` | yes (mesh-frame; converted on read) |
| `<scene_dir>/json/blender_scene.json` | yes (camera + per-object mesh_path, location, rotation_euler, scale; `meta.world_scale_factor` for pointmap scaling; merged at end) |
| `<scene_dir>/blend/blender_scene.blend` | yes (must contain `PointCloud_XZ` mesh + `geometry_*` objects) |
| `<scene_dir>/inputs/object_class.json` | optional but strongly recommended (enables wall-mount edge classification) |

## Outputs

| Path | Purpose |
|---|---|
| `<scene_dir>/json/bev_objects.png` + `.json` | Object footprints (BEV) + per-object hull XY |
| `<scene_dir>/json/bev_pointmap.png` + `.json` | Pointmap XY scatter + convex hull |
| `<scene_dir>/json/bev_compare.png` | Overlay of object hulls + pointmap hull + camera (report figure) |
| `<scene_dir>/json/bev_combined_hull.png` + `.json` | Convex hull of (pointmap ∪ objects ∪ camera) — reference draft for the floor shape; JSON carries CCW `hull_xy` vertices as the agent's numerical starting basis |
| `<scene_dir>/json/floor_plan.png` | Annotated polygon — solid lines = WALL, dotted = OPEN, wall-mount evidence labelled |
| `<scene_dir>/json/polygon_v2.json` | Stage-block-compatible polygon (vertices, floor_z, ceiling_z, wall_edges with `wall_mount_evidence`, open_edges with reason, fitter metadata) |
| `<scene_dir>/json/blender_scene.json` | Merged with the new `stage` block |
| `<scene_dir>/blend/blender_scene.blend` | Separable Stage collection; no PointCloud_XZ |

## Workflow (7 steps)

```
SCENE=<absolute path to scene_dir>
SKILL=<absolute path to .claude/skills/stage2-sub-pointmap-to-separable-stage>

# Step 1 — Blender: cleanup + export pointmap XY/Z (in Blender frame, scaled) + object bboxes + camera
blender --background "$SCENE/blend/blender_scene.blend" \
        --python "$SKILL/src/extract_inputs.py" -- --scene-dir "$SCENE"

# Step 2 — BEV from objects (object hulls + camera glyph)
python3 "$SKILL/src/bev_objects.py" --scene-dir "$SCENE"

# Step 3 — BEV from pointmap (XY hull + scatter)
python3 "$SKILL/src/bev_pointmap.py" --scene-dir "$SCENE"

# Step 4 — BEV overlay + combined-hull reference draft
python3 "$SKILL/src/bev_overlay.py" --scene-dir "$SCENE"

# Step 5 — Vision agent drafts floor plan (optional; fitter falls back if absent)
#          spawn .claude/agents/stage2-floor-plan-designer.md with image.png + bev_combined_hull.png + bev_combined_hull.json (+ bev_objects.json)
#          → writes <scene_dir>/json/floor_plan_draft.json

# Step 6 — Fit polygon: validate agent draft or run MABR + corner-cut search;
#          classify edge types using wall-mount classes; render annotated plan
python3 "$SKILL/src/compute_polygon.py"   --scene-dir "$SCENE"
python3 "$SKILL/src/render_floor_plan.py" --scene-dir "$SCENE"

# Step 7 — Blender: build Floor → Wall_NN per WALL edge → Ceiling, merge JSON, delete pointcloud
blender --background "$SCENE/blend/blender_scene.blend" \
        --python "$SKILL/src/build_stage_v2.py" -- --scene-dir "$SCENE"
```

The orchestrator runs these in order. There is no iteration loop — the polygon either passes the validator's rules (rectilinear, contains all objects + camera with clearance) or the fitter searches 4/6/8-vert candidates and picks the best valid one.

## Report mode (paper figures)

Run `src/render_report_figures.py --scene-dir <scene_dir>` to write paper-ready versions of `bev_compare.png`, `bev_combined_hull.png`, `bev_pointmap.png`, `floor_plan.png`, plus a new `bev_pointmap_textured.png` (pointmap drawn with its own per-vertex RGB) into `<scene_dir>/report/`. No axis, no legend, thicker pointmap dots, no clipping. Plain `python3` — no Blender required.

## When to deviate

- **Skip the agent.** If `floor_plan_draft.json` is absent, the fitter handles everything: MABR yaw + 4/6/8-vert search and wall-mount edge classification. Every edge defaults to WALL — there is no auto-OPEN rule. If your scene needs a doorway / archway, run with the agent.
- **No `object_class.json`.** Without class info, no edges get wall-mount evidence; every edge defaults to WALL. The polygon shape is unaffected.
- **No `geometry_*` objects yet.** This skill assumes objects are already placed (it fits walls around them). If the blend has only `PointCloud_XZ`, run `src/build.py` (in this skill) first.

## Files

### Scripts
- `src/extract_inputs.py` — Blender. Cleans stale stage geometry, exports object AABBs / pointcloud XY+Z (Blender-frame, scaled) / camera. Reads `meta.world_scale_factor`.
- `src/render_bev.py` — Canonical mesh-footprint helpers (`process_mesh_entry`, `convex_hull_xy`, `euler_to_matrix_xyz`, `camera_forward_xy`, plus internal helpers `load_mesh`, `trimesh_to_blender_local`, `normalize_to_unit_cube`, `apply_srt`, `rot_x/y/z`). Imported as a sibling module by `bev_objects.py` + `bev_pointmap.py`. Runnable standalone as a BEV PNG renderer from a `blender_scene.json`.
- `src/bev_objects.py` — BEV of object footprints. Imports `process_mesh_entry`, `convex_hull_xy`, `euler_to_matrix_xyz` from sibling `render_bev.py`. Writes `bev_objects.png` + `.json`.
- `src/bev_pointmap.py` — BEV of pointmap (mesh→Blender frame conversion + scaling). Imports `convex_hull_xy` + `euler_to_matrix_xyz` from sibling `render_bev.py`. Writes `bev_pointmap.png` + `.json`.
- `src/bev_overlay.py` — Overlay (`bev_compare.png`) + combined-hull reference draft (`bev_combined_hull.png`).
- `src/compute_polygon.py` — Validates agent draft OR runs MABR + corner-cut search (4/6/8 verts); classifies edge types via wall-mount whitelist; writes `polygon_v2.json`.
- `src/render_floor_plan.py` — Annotated polygon plot — solid for WALL, dotted for OPEN, with wall-mount evidence labels.
- `src/build_stage_v2.py` — Blender. Also exposes `build_from_polygon_dict(stage_dict, blend_path, …)` consumed by sibling `src/build.py` (the JSON→.blend builder) and by `stage-op-executor`.
- `src/render_report_figures.py` — Plain Python. Paper-figure variants of `bev_compare` / `bev_combined_hull` + textured pointmap BEV → `<scene_dir>/report/`.
- `src/convert.py` — Plain Python. trimesh-frame `layout_prediction.json` → Blender-frame `blender_scene.json`. Applies the `P @ R @ P` coordinate permutation, the `rz += 180°` camera + object rotation fix, the per-class world-scale prior (median across matched classes, clamped to a wide sanity bound `[0.1, 20.0]`), and the unit-meter assertion. `--scene-dir <dir>` mode auto-chains `src/build.py` to also produce the `.blend`. See `references/class_size_priors.md`.
- `src/build.py` — Blender. `blender_scene.json` → `blender_scene.blend`. Enforces the **unit-cube vertex normalization** before applying R/T/S (this is the load-bearing step that makes `floor.obj` land where the predictor placed it). Strips GLB-embedded cameras/lights and creates a single canonical `Camera` via the data API. Optional `stage` block triggers a `build_stage_v2.build_from_polygon_dict` call (sibling import) to also materialize Floor / Wall_NN / Ceiling. See `references/scene_json_schema.md`.
- `src/ply_import.py` — Blender helper used by `build.py` for PointCloud_XZ import.

### Agents
- `.claude/agents/stage2-floor-plan-designer.md` — Vision sub-agent prompt. Reads image + bev_compare + bev_combined_hull, drafts a rectilinear polygon (≤ 8 verts) with per-edge WALL/OPEN type. Output is treated as a hint that overrides the algorithmic fitter when valid.

## Cross-skill contract

`src/build_stage_v2.py` exposes:

```python
def build_from_polygon_dict(
    stage_dict: dict,
    blend_path: str,
    *,
    save: bool = True,
    replace_existing: bool = True,
) -> dict
```

`stage_dict` minimally requires `polygon_vertices`, `floor_z`, `ceiling_z`, `wall_thickness`, `floor_thickness`, `ceiling_thickness`. Optional: `open_edges[]`, `wall_edges[]`, `openings[]`. The signature is depended on by sibling `src/build.py` and by `stage-op-executor/src/run_stage_executor.py`; do not change it.

`<scene_dir>/json/polygon_v2.json` carries the same field set as the merged `stage` block, plus `yaw_deg`, `fitter` metadata (source, shape, search_areas), and `wall_mount_objects[]`. `stage2-environment-construction/src/_roundtrip_extract.py` reads this file to repopulate the `stage` block on roundtrip.
