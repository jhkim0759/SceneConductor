---
name: stage-progression-videos
description: >-
  Render the SceneConductor pipeline-stage "progression" report videos (and a
  matching .blend per clip) from a processed scene_dir: object pop-in, Stage-2
  environment build (interior + external), Stage-3 first refine (raw→planned),
  Stage-3 planned→final, and an isolated relation-group island refine (BEV +
  perspective). Use this whenever the user wants per-stage / per-step report
  videos, "stage progression" or "refinement process" clips, an env-build or
  wall-build video, a planned-vs-final or first-refine animation, or an island /
  table-and-chairs alignment clip from a SceneConductor scene — even if they only
  name one of the stages or say "make the report videos" or "render the stage
  clips like before". Produces 1160×600 H.264 mp4s on a white background with the
  established camera/crop/lighting conventions. NOT for the single 30s combined
  demo (that is scene-conductor-demo-video) and NOT for the 5-view look-dev
  render (general-multi-view-render).
---

# Stage-progression videos

Turn one processed SceneConductor `scene_dir` into a set of short, clean
report clips — one per pipeline stage — each saved as an `mp4` plus the
animated `.blend` that produced it. This captures the exact look-dev recipe
developed for the TEASER report: white background, a moderate interior lens
with edge cropping to tame wide-angle "bulge", per-stage object animation, and
a per-clip `.blend` so anyone can reopen and tweak a stage.

## When to use

Trigger on requests for any of these (the user rarely names all of them):

- "make the stage videos / progression videos / per-step clips for `<scene>`"
- "env build video", "wall build video" (interior or seen from outside)
- "first refine / raw→planned animation", "planned→final clip"
- "island refine video", "table + chairs aligning", "G1 BEV/perspective clip"
- "re-render the stage clips like we did for TEASER, but on this scene"

Pick the relevant subset with `--videos` if they only want some.

## Output

`<scene_dir>/report/stage_videos/` — eight `mp4`s (**1120×840, 4:3**, 24fps,
H.264). Frames render at a **1160×880** base and every finished clip is cropped a
uniform **20 px on all four edges** in post → exactly **1120×840 (4:3)**. By
default every clip is lit by the blend's **own lights** (`--original-lights`, the
default; white backdrop kept, world lighting = 0) at native intensity
(`--cycles-light-scale 1.0`). Pass `--substitute-lights` for the old white-bg
sun+fill look. A same-named per-clip `.blend` is written **only** with
`--save-blend` (OFF by default — each is a full scene copy and fills the disk
fast):

| key | file | what it shows |
|---|---|---|
| `1_popup` | `1_stage1_popup.mp4` | Stage-1 objects pop in one by one (white bg) |
| `2_env_interior` | `2_env_build_interior.mp4` | Floor→walls→ceiling build, **original (reference) camera view** |
| `2-1_env_external` | `2-1_stage2_external_build.mp4` | SAME build, external corner orbit (frame-synced with `2`) |
| `3_first_refine` | `3_stage3_first_refine.mp4` | raw → planned (simple ops: size / rotation / floor+wall attach) |
| `3-1_to_final` | `3-1_stage3_to_final.mp4` | planned → final (island chair alignment only) |
| `4_island_persp` | `4_island_G1_perspective.mp4` | one relation group: init → final, 3/4 view |
| `4_island_bev` | `4_island_G1_bev.mp4` | same group, top-down |
| `5_turntable` | `5_stage3_final_turntable.mp4` | full 360° turntable of the finished Stage-3 scene, from the `2-1` elevated external vantage (ceiling hidden) |

## Prerequisites (inputs in `scene_dir`)

This is a *reporting* step that runs after Stages 1–3 and the demo build. It needs:

- `blend/blender_scene.blend` — walled final scene (objects + walls + camera). Used as the build base for `2`/`2-1`.
- `blend/stage3-sub-planned.blend` — walled scene used as the reset base for the planned blend.
- `operation_plan.json` — Stage-3 simple-op plan. **Watch for dropped `update_size` ops** (see Gotchas) — pass `--size-from <fuller plan>` if the canonical plan only has 1 size op.
- `relation_groups/<G>/{metadata.json, island_init.blend, island.blend, refiner/iter_*}` — per-group island data (for `3-1` merge-back math and video `4`).
- `report/demo/demo_animated.blend` + `report/demo/stage_data/05_s2_scene.json` — from the **scene-conductor-demo-video** skill (the pop-in animation base + the raw/stage-1 object transforms). Run that skill first if missing.
- Reused pipeline scripts (override paths via env if moved):
  `APPLY_PLAN` → `stage3-sub-scene-refiner/src/apply_plan.py`,
  `EXTRACT_STAGE` → `scene-conductor-demo-video/scripts/extract_stage.py`,
  `BLENDER` → the Blender 4.2 binary.

Run everything in the `sceneconductor` conda env (the post-crop needs `cv2`).

## Quick start

```bash
conda run -n sceneconductor python \
  .claude/skills/stage-progression-videos/scripts/make_stage_videos.py \
  <scene_dir> --gpu 0
```

That runs the whole flow: prep the planned blend, compute the planned→final
deltas, render all seven clips, and post-crop. Common options:

| flag | default | meaning |
|---|---|---|
| `--videos a,b,…` | `all` | subset from the key column above |
| `--focal-interior` | `24` | lens (mm) for interior perspective views (`1`,`3`,`3-1`) — smaller = whole scene fits but more bulge |
| `--crop-render-px` | `20` | render-border vertical crop on the 1280×960 (4:3) base; horizontal auto = round(px·4/3) so it stays 4:3 |
| `--crop-post-px` | `40` | extra vertical crop on the finished mp4 (horizontal auto-scaled 4:3) → final **1120×840** |
| `--cycles-light-scale` | `1.0` | lighting multiplier for the Cycles clips; 1.0 = native (matches calibrated original lights). Lower to dim |
| `--save-blend` | off | also save a full `.blend` next to each mp4 (reopen/tweak). OFF by default — 7 full-scene copies/scene fill the disk fast |
| `--original-lights` / `--substitute-lights` | **original (default)** | original = light with the blend's own lights (white backdrop kept, world lighting=0); substitute = old white-bg sun+interior fill |
| `--engine` | `cycles` | engine for `1`/`2`/`2-1`/`4`/`5` |
| `--refine-engine` | `eevee` | engine for `3`/`3-1`. NOTE: the GALP-origin camera renders **fine under Cycles** once `use_compositing=False` + a fill/own light are set, so `cycles` works |
| `--samples` | `64` | render samples |
| `--size-from <plan>` | — | merge `update_size` ops from a fuller plan into the applied plan |
| `--island-group <name>` | auto | relation group for video `4` (default: first `*table*` group) |
| `--skip-prep` | off | reuse an existing prepared planned blend / stage jsons |

## How it works (and why)

The driver (`scripts/make_stage_videos.py`) only wires paths and shells out — to
Blender (one render script per clip) and OpenCV (the post-crop). The decisions
that make the clips look right live in the render scripts + `flatcam.py`:

- **White background** (`flatcam.white_bg`): a plain white world renders *grey*
  under the default AgX view transform, so we switch to the `Standard` transform
  with a white world at strength 1.0, drop the scene's (often mis-calibrated)
  lights, and add one gentle sun. Objects stay true-coloured and well exposed.
  With `--original-lights` the blend's own lights are **kept** instead (white
  backdrop preserved via a Light-Path mix whose lighting ray = black, so the
  world adds no ambient and only the scene's lights illuminate it).
- **Shared viewpoint** for `1`/`2`/`3`/`3-1`: all four use the blend's ORIGINAL
  reference camera (the GALP/photo pose, e.g. origin 0,0,0) — popup keeps its
  native lens, the interior/refine scripts no longer reposition. One consistent
  angle across the stages.
- **`5` turntable**: a near-BEV (≈78° top-down) full 360° orbit of the finished
  scene, auto-fit to the room footprint, with the ceiling AND any ceiling-mounted
  fixtures (light panels) hidden so the camera looks straight down onto the floor
  layout.
- **Convex / wide-angle bulge**: this is perspective from a short lens, not lens
  distortion — cropping alone barely helps. We render at a moderate `--focal-interior`
  and then crop the most-distorted edges (`flatcam.crop`, render-border) plus a
  post crop on the mp4. Trade-off: shorter lens fits the whole scene but bulges
  more; longer flattens but shows less.
- **`2` ↔ `2-1` are frame-synced**: both come from one shell-build schedule in
  `build_stage_sync.py` (floor→walls→ceiling, identical keyframes), rendered from
  two cameras. The interior camera sits *inside* the room looking at the back wall
  so the front wall is naturally behind it (no hidden-wall gap) and all visible
  walls complete; an interior light keeps the closing room lit.
- **`3-1` "final" is built, not read**: the stored `stage3-scene.blend` bakes the
  island refinement into meshes (and may use an older wall-attach), so animating
  obj_ empties to it makes objects "revert". Instead the final pose of each island
  member is computed as `M_anchor @ canonical_final` (`island_to_scene.py`) and
  layered onto the planned scene (location+rotation only; planned scale kept) —
  so only the chairs move and nothing snaps back.
- **Per-clip `.blend`** (opt-in, `--save-blend`): a render script can save its
  animated scene next to the mp4 so a stage can be reopened and adjusted. OFF by
  default — each `.blend` is a full walled-scene copy and 7 of them per scene fill
  the disk fast (this once pushed /home to 100% and an external pipeline archived
  the in-flight `well_done` inputs). Turn on only when you need to tweak a stage.

Render scripts (each is `blender -b <base> --python <script> -- …`):
`render_popup_flat.py` (1), `build_stage_sync.py` (2 / 2-1, `--view`),
`build_refine_multi.py` (3 / 3-1, N stage-json args), `build_island_2view.py`
(4, `--view persp|bev`). Tunables reach them via env: `SPV_LENS`, `SPV_CROP_PX`,
`SPV_ENGINE` (set by the driver). Helpers: `flatcam.py` (engine/white-bg/crop),
`reset_empties.py`, `island_to_scene.py`.

## Gotchas

- **Dropped size ops.** A "revised" `operation_plan.json` may keep only 1
  `update_size` op while the mesh-group normalization (chairs / lights / tables)
  lives in an earlier `operation_plan.json.bak_*`. If the planned/refine clip
  looks like size & grounding weren't applied, pass `--size-from <that bak>` —
  the driver merges the missing `update_size` ops before applying.
- **Refine views under Cycles render black.** The GALP origin (0,0,0) camera in
  `corrected_planned`/`blender_scene` ray-casts into geometry from the origin and
  comes out black in Cycles (fine in EEVEE). That's why `3`/`3-1` default to
  `--refine-engine eevee`. The other clips use Cycles fine.
- **Missing `demo_animated.blend`.** Video `1` needs the pop-in animation from
  scene-conductor-demo-video. Run that skill first, or drop `1_popup` from `--videos`.
```
