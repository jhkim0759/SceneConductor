---
name: general-multi-view-render
description: Stage-agnostic 5-view Cycles renderer (perspective / bev / wide / topcorner / topcorner_opposite) for any populated `.blend` — does not touch lights, world, materials, or camera. Usable from Stage 1 / Stage 2 / Stage 3 pipelines. Trigger on "/general-multi-view-render" or "just re-render the 5 views".
---

## What this skill does

Given an existing Blender `.blend` containing a populated interior scene (stage walls/floor/ceiling + objects + a fixed camera + lights), this skill runs **only the multi-view render step** and writes five PNGs to `<scene_dir>/render/`:

1. `blender_scene_view_perspective.png` — GALP reference vantage (camera at world origin with rotation/lens recovered from `layout_prediction.json`, with the standard `rz+180` upside-down fix; walls between origin and room centroid are made camera-invisible only).
2. `blender_scene_view_bev.png` — top-down orthographic floor-plan view with a self-contained overhead lighting rig (ceiling + scene lights hidden for this view only; flat white world; one temp 5000 W Area light).
3. `blender_scene_view_wide.png` — same vantage as `scene.camera`, swapped to a 20 mm lens.
4. `blender_scene_view_topcorner.png` — elevated 3/4 view from the polygon vertex MOST opposite `scene.camera`, anchored 30% inward from the corner toward the centroid, lens 18 mm.
5. `blender_scene_view_topcorner_opposite.png` — complementary 3/4 view from the SECOND-most-opposite polygon vertex; auto-skipped if the polygon has fewer than 2 distinct vertex scores.

It never touches anything else.

## Hard rules — what the skill must NOT modify

These exist because the calling pipeline already calibrated lights, materials, and camera. This skill is a renderer, not a look-dev step.

- **Never modify** any material, light energy, light color, world node tree, view transform, render engine, or compositor configuration **persistently**. Temporary swaps (BEV's overhead rig, per-view camera/lens) are restored in `try/finally` before exit.
- **Never modify** mesh geometry, object transforms, or the scene camera's permanent location/rotation/lens.
- The script DOES temporarily set `scene.cycles.samples` (default 256) and `scene.render.use_compositing = False` and mute every `CompositorNodeOutputFile` node for the duration of the 5-view render, then restores them. This is required to prevent the env-enhance compositor from overwriting the preview PNG and is documented behavior, not a violation.

If the user wants lighting / world / stage changes, route them to `stage2-sub-env-enhance` instead.

## Inputs

| Input | Purpose |
|---|---|
| `<scene_dir>` | Directory containing the populated `.blend`, `layout_prediction.json` (optional), `blender_scene.json` (optional), and `brightness_align_log.json` (optional). Five PNGs are written to `<scene_dir>/render/`. |
| `<.blend path>` | The populated Blender file to render — passed as Blender's positional argument, NOT as a `--` flag. |
| `--samples` (optional) | Cycles samples per view. Default 256. |
| `--resolution-x`, `--resolution-y` (optional) | Output PNG resolution. Default 1024 × 682. |
| `--brightness-log` (optional) | Path to `brightness_align_log.json`. Defaults to `<scene_dir>/brightness_align_log.json` then `<scene_dir>/scene-pipeline/brightness_align_log.json`. Used only by the BEV neutral-lighting context manager (which is itself only invoked if the implementation reverts to neutral mode — current BEV uses a self-contained overhead rig instead, so the log is informational). |

## Outputs

All five PNGs land in `<scene_dir>/render/`. Re-running overwrites in place.

| Filename | View |
|---|---|
| `blender_scene_view_perspective.png` | GALP reference vantage |
| `blender_scene_view_bev.png` | Top-down orthographic |
| `blender_scene_view_wide.png` | Same vantage, 20 mm lens |
| `blender_scene_view_topcorner.png` | Far-corner 3/4 (rank=0) |
| `blender_scene_view_topcorner_opposite.png` | Other far-corner 3/4 (rank=1) — skipped if polygon too small |

## How to run

```bash
blender --background <scene_dir>/blend/blender_scene.blend \
        --python src/render_multi_view.py -- \
        --scene-dir <scene_dir> \
        [--samples 256] \
        [--resolution-x 1024] \
        [--resolution-y 682] \
        [--brightness-log <path/to/brightness_align_log.json>]
```

Notes:
- The `.blend` path goes BEFORE `--python`; everything after `--` is forwarded to the script's argparse.
- The script must run inside Blender (it imports `bpy`); plain `python3` will fail.
- No `--output` flag — the `.blend` is read but not modified, and PNGs always go to `<scene_dir>/render/`.
- Render engine default is `CYCLES`. Use `--engine BLENDER_EEVEE_NEXT` or `--engine BLENDER_EEVEE` only if you explicitly need an EEVEE fallback.

## Reference vantage and blocking-wall handling (perspective view)

The perspective view is special: it positions the camera at the world origin `(0, 0, 0)` with rotation and lens recovered from `layout_prediction.json` (with the standard `rz+180` correction applied during convert.py). Because Stage 4's `_ensure_camera_inside_polygon` may have moved the actual `.blend` camera away from origin, walls between the origin and the room centroid would occlude the interior in this view.

To handle this, the script:
1. Reads `polygon_vertices` and `wall_objects` from `<scene_dir>/json/blender_scene.json` (also tries `<scene_dir>/blender_scene.json`).
2. For each wall, projects its world-space XY midpoint (computed from `bound_box` corners, not the object origin) onto the camera-to-centroid direction.
3. If the wall lies between the camera and centroid AND its perpendicular distance to that line is under `0.5 * max(room_w, room_d)`, marks it as blocking and sets `wall.visible_camera = False` for the render. Light bounces are unaffected (GI preserved).
4. Restores `visible_camera` for every modified wall in a `try/finally`.

If `layout_prediction.json` is missing or malformed, the perspective view falls back to using `scene.camera` as-is and prints `WARNING: layout_prediction.json missing`.

## BEV view lighting

The BEV view replaces the world and lighting for this view only — necessary because the calibrated lighting rig is tuned for the scene camera vantage and produces hot spots from above:

- Ceiling object hidden (`hide_render = True`).
- ALL existing scene lights hidden (`hide_render = True`) and their energies preserved for restore.
- World replaced with flat `(1.0, 1.0, 1.0)` Background, strength 0.5.
- One temporary `_MV_BEVAreaLight` Area light (5000 W, square, size = `max(room_w, room_d) * 1.2`) added at `(cx, cy, hi.z + 0.3)` aimed straight down.
- All temp state is removed in `try/finally` after the render.

## Scene bbox

Walks every `MESH` object whose name does NOT start with `PointCloud_XZ` and is NOT in the `Lighting_Env` collection. This excludes the point cloud and light rig so the bbox tightly fits the stage walls + furniture, which is what the BEV ortho frame and topcorner anchor logic both need.

## Common pitfalls

- **Black or near-black BEV**: the script auto-disables existing lights; if you accidentally also have `hide_viewport = True` on ceiling-light objects from a prior run, the temp Area is the only emitter, which is fine. If the BEV is still black, check that `.blend` actually has geometry visible from above (no `hide_render` set on floor).
- **Perspective view shows a wall right in front of the camera**: `blender_scene.json` is missing or its `wall_objects` list doesn't match actual `Wall_NN` object names. The fallback path writes the unmodified `scene.camera` view instead — open `<scene_dir>/json/blender_scene.json` and verify `stage.wall_objects` is populated.
- **`topcorner_opposite` PNG missing**: expected when the floor polygon has fewer than 2 distinct vertex scores (degenerate or quad polygons can produce ties). The script logs `view=topcorner_opposite: skipped — …` and continues; this is not an error.
- **Compositor wrote the wrong PNG**: do NOT remove the `use_compositing = False` + `FileOutput.mute = True` block. The Stage 5 compositor would otherwise overwrite `blender_scene_env_preview0001.png` on every `bpy.ops.render.render(write_still=True)` call.
- **`samples` permanently changed**: only happens if the script crashes between the `try` and `finally`. The provided code restores it; if you edit the script, keep the restore in `finally`.

## Idempotency and forcing

Re-running overwrites the five PNGs. There is no checkpoint or skip-if-exists logic — every invocation re-renders all 5 views.

## When to use this vs. `stage2-sub-env-enhance`

- **Use `stage2-sub-env-enhance`** when the user wants to change lights / sky / wall-floor-ceiling colors, OR wants the original single-view `blender_scene_env_preview.png` plus a fresh stage-look-dev pass.
- **Use this skill** when the `.blend` is already in the desired environment state (env-enhance was already run, OR the user has manually tuned lights, OR a non-`stage2-sub-env-enhance` upstream produced the lighting) and only the 5 multi-view PNGs need to be (re)generated.
