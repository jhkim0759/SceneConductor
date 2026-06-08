---
name: scene-orchestration
description: Full end-to-end pipeline orchestrator — runs stage1-initialize-scene → stage2-environment-construction → stage3-scene-refinement in sequence on a single scene_dir containing only image.png. Accepts: scene-dir (required), --gpu N (default 0, Stage 1 only), --island-refine-iter N (default 10, Stage 3 only), --force (skip all resume checks and rerun from Stage 1). Trigger on "/scene-orchestration", "run the full pipeline", "process scene end-to-end", "image to final 3D scene", or any time the user provides a scene_dir with only image.png and wants the complete output.
---

> For the authoritative file-layout contract see `FILE_DIRECTORY.md` at the SceneConductor root.

# scene-orchestration

Thin sequential orchestrator. Takes one folder with `image.png` and drives all three stages in order until `render/final/*.png` exists. Each stage delegates to its own skill — this file owns only the sequencing, resume logic, and failure triage.

## Quick reference (arg guide)

When the user types `/scene-orchestration ...`, these are the only args this skill consumes. Anything not listed here is passed through verbatim (or ignored).

```
/scene-orchestration <scene-dir> [--gpu N] [--island-refine-iter N] [--force]
```

| Arg | Required? | Default | Range | Forwarded to |
|---|---|---|---|---|
| `<scene-dir>` (positional) | **yes** | — | absolute path containing `image.png` | all three stages |
| `--gpu N` | no | `0` | non-negative int | Stage 1 only (`stage1-initialize-scene`) |
| `--island-refine-iter N` | no | `10` | int in `[1, 10]` (clamped silently if outside) | Stage 3 only (forwarded as `--num-max-iter`) |
| `--force` | no | off | flag (no value) | bypasses every resume check; restarts from Stage 1 |

**Resume vs force.** By default the skill skips any stage whose completion files are already present (see resume table below). `--force` disables every skip, so Stages 1 → 2 → 3 all re-run from scratch.

### Usage examples

```text
/scene-orchestration /data/scenes/room_01
/scene-orchestration /data/scenes/room_01 --gpu 3
/scene-orchestration /data/scenes/room_01 --gpu 0 --island-refine-iter 8
/scene-orchestration /data/scenes/room_01 --force
/scene-orchestration /data/scenes/room_01 --gpu 2 --island-refine-iter 5 --force
```

Equivalent natural-language forms are also accepted (see `Parsing rules`). For example:

```text
Run the full pipeline on /data/scenes/room_01 with GPU 3 and 15 island iterations
Rerun /data/scenes/room_01 from scratch on GPU 0
```

## When to trigger

- `/scene-orchestration`
- `/scene-orchestration /path/to/scene`
- `/scene-orchestration /path/to/scene --gpu 3`
- `/scene-orchestration /path/to/scene --gpu 0 --island-refine-iter 8`
- `/scene-orchestration /path/to/scene --force`
- `/scene-orchestration /path/to/scene --gpu 2 --island-refine-iter 5 --force`
- "Run the full pipeline on `/path/to/scene/`"
- "Run the full pipeline on `/path/to/scene/` with GPU 3"
- "Run the full pipeline on `/path/to/scene/` with GPU 3 and 15 island iterations"
- "Rerun the full pipeline from scratch on `/path/to/scene/`"
- "Force-restart everything on `/path/to/scene/`"
- "Process this scene end-to-end"
- "Image to final 3D scene"
- "I have `image.png` — run everything"
- "Run stage 1 through stage 3 on `/path/to/scene/`"
- User supplies a `scene_dir` that contains only `image.png` (no `inputs/`, no `blend/`) and asks for the final result

**Scope boundary.** This skill stops at `render/final/blender_scene_view_perspective.png`. It does not render additional variants, modify materials, or do any post-processing. For those, invoke the appropriate sub-skill directly.

## Inputs required

Only `<scene_dir>/image.png` is required. If it is missing, stop immediately:

```
ERROR: <scene_dir>/image.png not found. scene-orchestration requires exactly one input file.
```

Do NOT create placeholder files or auto-download anything.

## Arguments

Parse the following from the invocation string (CLI flags or natural language):

| Arg | Default | Forwarded to | Notes |
|---|---|---|---|
| `scene-dir` (positional) | (required) | all three stages | absolute path to scene folder |
| `--gpu N` | `0` | Stage 1 only | Stage 2 and Stage 3 have no `--gpu` surface — the value is silently ignored for those stages |
| `--island-refine-iter N` | `10` | Stage 3 only (as `--num-max-iter`) | clamped to [1, 10] by `orchestrate.py` |
| `--force` | off | none — consumed locally | Boolean flag (no value). When present, skip every resume check and rerun Stage 1 → 2 → 3 from scratch. Existing artifacts in `inputs/`, `blend/`, and `render/` are left on disk; downstream stages overwrite them. |

### Parsing rules

Recognize the following forms:

- **scene-dir**: any bare absolute path token (starts with `/`)
- **gpu**: `--gpu N`, "on GPU N", "with GPU N", "GPU index N", "use GPU N"
- **island-refine-iter**: `--island-refine-iter N`, "N island iterations", "N iters for refinement", "island iterations N", "island-refine-iter N"
- **force**: `--force`, "force", "rerun from scratch", "start over", "restart from scratch", "ignore existing outputs", "redo everything", "force-restart"

Natural-language examples:
- "Run the full pipeline on /path/to/scene with GPU 3 and 15 island iterations" → `scene_dir=/path/to/scene`, `gpu=3`, `island_iters=15`, `force=false`.
- "Rerun /path/to/scene from scratch on GPU 0" → `scene_dir=/path/to/scene`, `gpu=0`, `island_iters=10`, `force=true`.

### Validation

- `gpu` must be a non-negative integer; if not → print `ERROR: --gpu must be a non-negative integer` and stop.
- `island-refine-iter` must be an integer in [1, 10]; if outside this range → clamp silently and warn: `[scene-orchestration] island-refine-iter clamped to <clamped> (requested <N>)`.
- `scene-dir` must be an absolute path containing `image.png`; if missing → stop with the error above.
- `--force` is a Boolean flag and takes no value. If the user writes `--force <something>`, treat `<something>` as the next positional/flag token (do NOT consume it as the force value).

## Resume / skip-if-complete logic (mandatory)

Before invoking each stage, check whether it is already complete using the file signals below. If complete → log one line and skip. If not complete → invoke via `Skill` tool.

| Stage | Complete when ALL of these exist |
|---|---|
| **Stage 1** | `<scene_dir>/inputs/layout_prediction.json` AND `<scene_dir>/inputs/object_class.json` |
| **Stage 2** | `<scene_dir>/blend/blender_scene.blend` |
| **Stage 3** | `<scene_dir>/blend/stage3-scene.blend` AND `<scene_dir>/render/final/blender_scene_view_perspective.png` |

Log format on skip:

```
[scene-orchestration] Stage N already complete — skipping.
```

### Force / rerun-from-scratch convention

Force mode is triggered by **either** of:

1. The CLI flag `--force` parsed from the invocation string (see Arguments table).
2. Any of the following natural-language phrases in the user's request:
   - "rerun from scratch"
   - "force"
   - "start over"
   - "restart from scratch"
   - "ignore existing outputs"
   - "redo everything"
   - "force-restart"

When force mode is on, treat every stage as incomplete — **do NOT skip any stage**, regardless of what files already exist on disk. Existing files are not deleted up-front; downstream stages overwrite them naturally.

Announce this to the user at the start of the run, BEFORE invoking Stage 1:

```
[scene-orchestration] Force mode — skipping all resume checks; rerunning Stage 1 → 2 → 3 from scratch.
```

When force mode is off (default), apply the resume table above and log per-stage skip messages.

## Flow (sequential — no parallelism)

Stage 2 depends on Stage 1 outputs. Stage 3 depends on Stage 2 outputs. Never run stages in parallel.

### Stage 1 — Initialize scene

**Resume check:** `inputs/layout_prediction.json` + `inputs/object_class.json` both present?

If not complete:

```python
# Forward --gpu; Stage 1's agent passes it to run_stage1.sh for both phases.
# Example: gpu=3 → args="<scene_dir> --gpu 3"
Skill(skill="stage1-initialize-scene", args="<scene_dir> --gpu <gpu>")
```

**Post-check** (after Skill returns): verify the two files above exist. If any are missing → stop (see Failure handling).

> `inputs/relation_graph.json` is **not** a Stage-1 artifact anymore — it is produced by the `stage3-relation-graph` agent inside Stage 3 prep (`/scene-analyze-prepare`), so it is not part of the Stage-1 done condition.

Typical runtime: 18–33 min.

### Stage 2 — Environment construction

**Resume check:** `blend/blender_scene.blend` present?

If not complete:

```python
# Stage 2 has no --gpu surface — GPU arg is not forwarded here.
Skill(skill="stage2-environment-construction", args="<scene_dir>")
```

**Post-check** (after Skill returns): verify `blend/blender_scene.blend` exists. If missing → stop.

Typical runtime: 15–30 min.

### Stage 3 — Scene refinement

**Resume check:** `blend/stage3-scene.blend` + `render/final/blender_scene_view_perspective.png` both present?

If not complete:

```python
# Forward --num-max-iter; Stage 3's orchestrate.py accepts this flag (default 10, clamped [1,10]).
# Stage 3 has no --gpu surface — GPU arg is not forwarded here.
# Example: island_iters=15 → args="<scene_dir> --num-max-iter 15"
Skill(skill="stage3-scene-refinement", args="<scene_dir> --num-max-iter <island_iters>")
```

**Post-check** (after Skill returns): verify `blend/stage3-scene.blend` and at least one file matching `render/final/blender_scene_view_*.png` exist. If missing → stop.

Typical runtime: 20–60 min.

## Failure handling

If a stage's Skill invocation returns and its post-check fails, **stop immediately**. Do NOT attempt later stages.

Produce a structured failure report:

| Field | Content |
|---|---|
| **Failed stage** | Stage N (`stage1-initialize-scene` / `stage2-environment-construction` / `stage3-scene-refinement`) |
| **Last successful stage** | Stage N-1 (or "none" if Stage 1 failed) |
| **Missing outputs** | List every expected file that was not found |
| **Where to look** | See log pointers table below |
| **Suggested action** | Re-run the failed stage skill directly: `Skill(skill="stageN-...", args="<scene_dir>")` |

### Log pointers by stage

| Stage | Log file / state file | Notes |
|---|---|---|
| Stage 1 pre-phase | `<scene_dir>/logs/stage1_pre.log` | GroundedSAM + mask attribute |
| Stage 1 post-phase | `<scene_dir>/logs/stage1_post.log` | SAM3D + GALP + finalize |
| Stage 2 | inspector JSON + sub-skill logs in `<scene_dir>/render/` | Re-run `inspect_scene.py <scene_dir>` to see which sub-stage is blocked |
| Stage 3 | `<scene_dir>/json/stage3_state.json` | `step_status` and `step` fields pinpoint where the state machine stopped |

Do NOT read multi-megabyte log files proactively. Only open a log after confirming the exit condition and confirming the specific file is relevant.

## Final report (on success)

When all three stages complete (either freshly run or resume-skipped), print a compact summary covering: object count (from `inputs/object_class.json`), mask count (`Glob inputs/masks/*.png`), GLB count (`Glob inputs/object/*.glb`), presence of `blend/blender_scene.blend`, `blend/stage3-scene.blend`, and count of PNGs in `render/final/`. Point the user to `<scene_dir>/render/final/` for the 5-view PNGs. Derive counts from small JSON files and `Glob` — do not read binary files.

## Final scene contract

Expected directory tree after a successful full pipeline run. Indent shows which stage produced each artifact.

```
<scene_dir>/
├── image.png                                         # input (required)
│
├── inputs/                                           # Stage 1 outputs
│   ├── object_class.json
│   ├── mask_attribute.json
│   ├── layout_prediction.json
│   ├── layout-prediction.glb
│   ├── pointmap_xz.ply
│   ├── floor.obj
│   ├── object_state.json
│   ├── object_state_annotated_mask.png
│   ├── merge_plan.json
│   ├── remask_plan.json                              # only if evaluator flagged remask
│   ├── relation_graph.json                           # Stage 3 prep — stage3-relation-graph agent (NOT Stage 1)
│   ├── masks/
│   │   ├── mask.png
│   │   └── 1.png .. M.png
│   ├── object/
│   │   └── 1.glb .. M.glb
│   └── thumbnails/
│       └── obj_<id>_<class>.png
│
├── logs/                                             # Stage 1 logs
│   ├── stage1_pre.log
│   └── stage1_post.log
│
├── json/                                             # Stage 2 + Stage 3 JSON
│   ├── stage2_plan.json                              # Stage 2 — director
│   ├── blender_scene.json                            # Stage 2 — source of truth for .blend rebuild
│   ├── polygon_v2.json                               # Stage 2 — floor polygon
│   ├── alignment_metrics_v2.json                     # Stage 2
│   ├── blend_info.json                               # Stage 2 / scene-analyze-prepare
│   ├── object_state.json                             # Stage 2 / scene-analyze-prepare
│   ├── operation_plan.json                           # Stage 3
│   ├── heuristic_ops.json                            # Stage 3
│   ├── llm_ops.json                                  # Stage 3
│   ├── relation_pairs.json                           # Stage 3 — relation-solve
│   ├── relation_solve_ops.json                       # Stage 3 — relation-solve
│   ├── island_groups.json                            # Stage 3 — validation
│   └── stage3_state.json                             # Stage 3 — state machine
│
├── blend/                                            # Stage 2 + Stage 3 blend files
│   ├── blender_scene.blend                           # Stage 2 — working blend
│   ├── stage2-sub-build.blend                        # Stage 2 snapshot (after stage 2)
│   ├── stage2-sub-separable.blend                    # Stage 2 snapshot (after stage 3)
│   ├── stage2-sub-env.blend                          # Stage 2 snapshot (after stage 4)
│   ├── stage2-scene.blend                            # Stage 2 FINAL snapshot
│   ├── stage3-sub-planned.blend                      # Stage 3 — auto-pass + relation-solve
│   └── stage3-scene.blend                            # Stage 3 FINAL blend
│
├── render/                                           # Stage 2 + Stage 3 renders
│   ├── blender_scene_view_perspective.png            # Stage 2 multi-view
│   ├── blender_scene_view_bev.png
│   ├── blender_scene_view_wide.png
│   ├── blender_scene_view_topcorner.png
│   ├── blender_scene_view_topcorner_opposite.png
│   ├── planned/                                      # Stage 3 — auto-pass render
│   │   └── blender_scene_view_perspective.png
│   └── final/                                        # Stage 3 FINAL renders (5-view)
│       ├── blender_scene_view_perspective.png
│       ├── blender_scene_view_bev.png
│       ├── blender_scene_view_wide.png
│       ├── blender_scene_view_topcorner.png
│       └── blender_scene_view_topcorner_opposite.png
│
└── relation_groups/                                  # Stage 3 — per-group island blends
    └── <G>/
        ├── island.blend
        ├── masked.png
        └── metadata.json
```

## Don'ts

- Do NOT run Stage 2 before Stage 1's post-check passes — it will fail on missing `inputs/layout_prediction.json`.
- Do NOT run Stage 3 before Stage 2's post-check passes — it requires `blend/blender_scene.blend`.
- Do NOT parallelize stages — GPU memory contention and sequential file dependencies make this unsafe.
- Do NOT auto-run `stage3-sub-scene-analyze-prepare` inline — Stage 3's skill handles that internally.
- Do NOT read multi-megabyte log files proactively — only open logs after a confirmed failure.
- Do NOT create new Python scripts, shell scripts, or orchestrate.py — this skill is a pure SKILL.md delegator.
- Do NOT copy or re-implement any sub-skill's internal logic here — changes to sub-skills must not require edits to this file. For internal contracts, failure modes, and manual reproduction steps, consult each stage's own SKILL.md: `stage1-initialize-scene/SKILL.md`, `stage2-environment-construction/SKILL.md`, `stage3-scene-refinement/SKILL.md`.
