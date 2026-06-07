# Per-stage input/output contracts

Single source of truth for filenames the orchestrator watches to decide "is this stage ready?" and "is this stage done?". If a sub-skill renames an output or adds a required input, update this table **and** the matching check in `src/inspect_scene.py`.

## Contracts

### Stage 1 — stage2-sub-pointmap-to-separable-stage

| | Files |
|---|---|
| Required inputs | `inputs/layout_prediction.json`, `image.png` OR `image.jpg` OR `image.jpeg` |
| Optional inputs | `inputs/mask_attribute.json` or `inputs/object_class.json` (auto-discovered for class-based rescale) |
| Done signal | `json/blender_scene.json` exists |
| Idempotency | Overwrites `json/blender_scene.json` on rerun. |

---

### Stage 2 — stage2-sub-pointmap-to-separable-stage

| | Files |
|---|---|
| Required inputs | `json/blender_scene.json` (with `meta.coordinate_system == "blender"`) |
| Optional inputs | Blender binary path override, build-script path override |
| Done signal | `blend/blender_scene.blend` exists |
| Idempotency | Overwrites `blend/blender_scene.blend` on rerun. |

---

### Stage 3 — stage2-sub-pointmap-to-separable-stage

| | Files |
|---|---|
| Required inputs | `image.{png,jpg,jpeg}`, `inputs/pointmap_xz.ply`, an existing `blend/blender_scene.blend` that already contains a `PointCloud_XZ` mesh |
| Optional inputs | `json/blender_scene.json` (will be merged with a `stage` block) |
| Done signal | `json/polygon_v2.json` AND `json/alignment_metrics_v2.json` AND `blender_scene.json["stage"]` block present |
| Secondary outputs | `render/blender_scene_v2_bev_overlay.png`, updated `blend/blender_scene.blend` with `Stage` collection |
| Idempotency | Deletes old `Room_Shell`/`Wall_*`/`Floor`/`Ceiling` before rebuilding. Rerun-safe. |

Note: `json/polygon.json` (v1) and `json/alignment_metrics.json` (v1) are never produced by the simplified pipeline. If a user manually places `polygon.json`, the seed-from-v1 hook in `compute_polygon_v2.py` will pick it up; otherwise the 4-candidate path is the default.

---

### Stage 4 — stage2-sub-env-enhance

| | Files |
|---|---|
| Required inputs | `image.{png,jpg,jpeg}`, an existing `blend/blender_scene.blend` with stage + objects + camera |
| Optional inputs | Output `.blend` path override, mood/color hex overrides |
| Done signal | Any `*_env_preview.png` in `render/` AND `json/blender_scene.json` contains `lighting`, `world`, `stage_materials`, and `render` blocks |
| Secondary outputs | In-place updated `.blend`, `json/blender_scene.json` env blocks written by `export_env_to_json.py`, `render/brightness_align_log.json` |
| Idempotency | Preserves `Material_0.*` and `geometry_*` meshes; modifies stage-prefixed materials only. Re-runs replace the preview PNG and overwrite the JSON env blocks. |

---

### Stage 5 — multi-view-render

| | Files |
|---|---|
| Required inputs | `blend/blender_scene.blend` (with env-enhance settings applied — Stage 4 done); `inputs/layout_prediction.json` (for perspective camera rotation/focal-length — if absent the view falls back to scene Camera with a WARNING log) |
| Script | `.claude/skills/general-multi-view-render/src/render_multi_view.py` (runs inside Blender — canonical project-wide multi-view renderer) |
| Done signal | `render/blender_scene_view_perspective.png` exists (canonical), OR `blender_scene_view_perspective.png` at top-level (legacy) |
| All 5 outputs | `render/blender_scene_view_perspective.png` (calibrated — GALP reference vantage), `render/blender_scene_view_bev.png` (self-contained BEV overhead rig), `render/blender_scene_view_wide.png` (calibrated — same vantage, 20 mm), `render/blender_scene_view_topcorner.png` (calibrated — far-corner 3/4 view, 18 mm, rank=0), `render/blender_scene_view_topcorner_opposite.png` (calibrated — 2nd far-corner 3/4 view, 18 mm, rank=1) |
| topcorner_opposite skip | Auto-skipped if polygon has fewer than 2 distinct vertices; a "skipped" log line is printed and no PNG is written |
| Orchestrator dispatch | `blender --background <scene_dir>/blend/blender_scene.blend --python .claude/skills/general-multi-view-render/src/render_multi_view.py -- --scene-dir <scene_dir> --brightness-log <scene_dir>/render/brightness_align_log.json` |
| Idempotency | Overwrites existing view PNGs on re-run. `--force-from 5` deletes all five `blender_scene_view_*.png` files from both canonical `render/` and legacy top-level so Stage 5 re-plans as ready. |

---

## Round-trip invariant

After the full pipeline completes (Stage 4 done), `json/blender_scene.json` is a **complete bidirectional serialization** of `blend/blender_scene.blend`. The `.blend` can be fully rebuilt from the JSON by running `stage2-sub-pointmap-to-separable-stage` on the final `blender_scene.json`.

What IS serialized: `meta`, `camera`, `scene`, `objects[]` (transforms + mesh references), `stage` (polygon, floor/ceiling z, wall edges, openings), `point_cloud` (PLY import config, not vertex data), `lighting[]`, `world`, `stage_materials`, `render`, `compositor`.

What is NOT serialized (by design): per-object mesh materials (`Material_0`, `Material_0.001`, … attached to `geometry_*` meshes). These are owned by the upstream layout-prediction pipeline and are protected by the env-only rule (`feedback_blender_scene_env_only.md`) — Stage 4 must never write or modify `Material_0.*` entries.

---

## How to verify round-trip fidelity

Run after Stage 4 completes on any `scene_dir`:

```bash
python .claude/skills/stage2-environment-construction/src/verify_roundtrip.py \
    /path/to/scene_dir \
    [--blender /path/to/blender] \
    [--keep-tmp]
```

The verifier (1) extracts live blend state into a fresh `blender_scene.roundtrip.json`, (2) rebuilds `roundtrip.blend` via `stage2-sub-pointmap-to-separable-stage`, (3) diffs 16+ invariant groups (mesh transforms, camera, lights, world, stage geometry, render settings, PLY vertex count) with per-invariant tolerances, and (4) writes `roundtrip_report.json` in `scene_dir`. Exit 0 = PASS, 1 = FAIL, 2 = infrastructure error. Report shape: `{ status, summary, invariants[{name, expected, observed, tolerance, delta, pass}], missing_blocks_in_original, extra_blocks_in_roundtrip, paths, timings }`.
