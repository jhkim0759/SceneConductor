---
name: stage2-environment-construction
description: Orchestrator that inspects a `scene_dir` and runs the Stage-2 sub-skills in order, skipping completed stages and stopping on blocked inputs. Trigger on "/stage2-environment-construction".
---

> For the authoritative file-layout contract see `FILE_DIRECTORY.md` at the SceneConductor root.

## What this skill does

One input: a `scene_dir` path. One output: every pipeline stage that can run has run.

Concretely, the skill:

1. Runs `src/inspect_scene.py <scene_dir>` which prints a JSON plan describing each of the 6 stages (0–5) as `done`, `ready`, or `blocked` based on the files and JSON content present in the folder.
2. For each stage in order where status is `ready`, invokes that sub-skill / agent live with the scene_dir as its argument:
   - **Stage 0** uses a **direct Bash subprocess** calling `src/run_stage2_director.py` — the orchestrator must NOT spawn `stage2-environment-planner` via the `Agent` tool. The director call is embedded in the subprocess script and is structurally un-bypassable, matching the Stage 1 mask-evaluator pattern.
   - **Stages 1–4** use the `Skill` tool.
   - **Stage 5** is a direct Blender command (no Skill/Agent).
3. After each step returns, re-runs the inspector to confirm the stage is now `done`. If not, surfaces the error and stops.
4. Skips stages already `done` so re-runs on the same scene_dir are fast.
5. Reports which stages ran, which were skipped, and which were blocked with reasons.

## Why an orchestrator (and not one big monolithic skill)

The user explicitly asked that this skill **not** be independent. Here, "not independent" means: when `stage2-sub-pointmap-to-separable-stage` gets updated — new flags, new output files, new agent prompts — this orchestrator must pick up the change without any edit here. The only way to achieve that is to delegate: call each sub-skill by name via the `Skill` tool and trust it to do its work. This file contains **zero** copies of any sub-skill's Python scripts, agent prompts, or internal contracts. The sub-skill owns its work; this skill owns only the sequencing.

The one exception: the orchestrator needs to know each stage's "is this done?" and "is this ready?" filename signals so it can plan. Those signals are intentionally simple — just checking for the existence of a few well-known output files and JSON content blocks — and they live in `references/stage_contracts.md` so they can be adjusted quickly if a sub-skill renames an output, without rewriting this skill.

## The 6 stages

| # | Stage | Dispatched via | Primary "done" signal |
|---|---|---|---|
| **0** | **Director plan** | subprocess — `run_stage2_director.py` | **`json/stage2_plan.json` exists with valid `_director_meta` block** |
| 1 | Layout JSON | `Skill` `stage2-sub-pointmap-to-separable-stage` | `json/blender_scene.json` exists |
| 2 | Build .blend | `Skill` `stage2-sub-pointmap-to-separable-stage` | `blend/blender_scene.blend` exists |
| 3 | Separable stage | `Skill` `stage2-sub-pointmap-to-separable-stage` | `json/polygon_v2.json` + `json/alignment_metrics_v2.json` + `stage` block in `json/blender_scene.json` |
| 4 | Env-enhance | `Skill` `stage2-sub-env-enhance` | any `*_env_preview.png` in `render/` AND `lighting` + `world` + `stage_materials` + `render` blocks in `json/blender_scene.json` |
| **5** | **Multi-view render** | *(direct Blender call — see below)* | **`render/blender_scene_view_perspective.png` exists** |

### Stage 0 — director (vision plan)

The director is invoked via a **subprocess wrapper** (`src/run_stage2_director.py`), not via the `Agent` tool. It runs **once at the start of Stage 2** and writes `json/stage2_plan.json` — a structured plan containing:

- `scene_summary` (paragraph): room type, dimensions, dominant colors, openings, notable objects
- `polygon_brief` (sentence): shape hint consumed by Stage 3's `stage2-floor-plan-designer` agent
- `materials_hint` (object): wall / floor / ceiling sRGB hex + optional per-wall overrides — consumed by Stage 4's `enhance_env.py`
- `lighting_hint` (object): mood + practicals count — consumed by Stage 4's `enhance_env.py`
- `openings_hint` (array): visible windows / doorways (Stage 4.5 will consume this when it exists)
- `confidence` per section — sections with `< 0.5` are ignored by consumers

Every hint is **advisory**: the algorithmic validators downstream (polygon fitter, PBR albedo clamp, manifold check) still have final say.

The director call is embedded in a subprocess script (`run_stage2_director.py`). The orchestrator agent must not spawn `stage2-environment-planner` directly — it is invoked exclusively by `run_stage2_director.py`. This makes the vision step structurally un-bypassable, matching the Stage 1 mask-evaluator pattern.

After the call, `run_stage2_director.py` validates `json/stage2_plan.json` (required keys + valid JSON) and injects a `_director_meta` block containing `model`, `image_sha256`, `generated_by`, and `timestamp_utc`. The inspector treats Stage 0 as `done` only when this block is present and the `image_sha256` matches the current `image.png`.

Schema: `references/stage2_plan_schema.md`. Agent prompt: `.claude/agents/stage2-environment-planner.md`.

Dispatch pattern (orchestrator):

```bash
# When the inspector reports stage 0 as "ready":
conda run -n sceneconductor python <skill_dir>/src/run_stage2_director.py --scene_dir <scene_dir>
```

Full per-stage input/output contracts: `references/stage_contracts.md`. Consult that file if a sub-skill seems to have changed its filenames or done-signal conditions — update the contracts and `inspect_scene.py`, don't patch this file.

## JSON ↔ .blend round-trip invariant

After the full pipeline completes, `json/blender_scene.json` is the **single source of truth** for rebuilding `blend/blender_scene.blend`. Running `stage2-sub-pointmap-to-separable-stage` on the final `blender_scene.json` must produce a `.blend` that is functionally identical to the on-disk `blender_scene.blend`.

**What IS serialized** (bidirectional): `meta`, `camera`, `scene`, `objects[]` (transforms + mesh_path references), `stage` (polygon, floor/ceiling z, wall edges, openings from separable-stage), `point_cloud` (PLY import metadata), `lighting[]`, `world`, `stage_materials`, `render`, `compositor`.

**What is NOT serialized** (by design, per the env-only rule in `feedback_blender_scene_env_only.md`): per-object mesh materials (`Material_0`, `Material_0.001`, … on `geometry_*` meshes). These are owned by the upstream layout-prediction pipeline. No stage in this pipeline reads, modifies, or writes them.

The future `verify_roundtrip.py` CLI (planned for Wave 4) will automate verification of this invariant.

## How to run — delegation

When the user triggers `/stage2-environment-construction`, the cheapest path is to **delegate the whole flow to the matching orchestrator agent**:

```python
Agent(
    description="Stage 2 env-construction",
    subagent_type="stage2-environment-construction",
    prompt=f"Run Stage 2 on {scene_dir}",
    run_in_background=True,   # optional — long-running
)
```

The `stage2-environment-construction` agent (Haiku, `tools: Read, Write, Glob, Bash`) drives the inspect-and-dispatch loop below, spawns `stage2-environment-planner` for Stage 0 and `stage2-floor-plan-designer` inside Stage 3, and runs all Bash steps. The main conversation only sees the agent's final report — large Cycles render logs stay in the agent's throwaway context.

The remainder of this document is the **agent's contract** (the inspector, the per-stage dispatch table, the snapshot rule). Read it if you need to debug, customize, or run the pipeline manually without going through the orchestrator agent.

## Manual orchestration

### Step 1 — Inspect

Run the inspector and capture its JSON output:

```bash
python3 <this-skill-dir>/src/inspect_scene.py <scene_dir>
```

Read this plan in full. The user wants to see it — show them a short summary. If every stage is `done`, report that and stop.

### Step 2 — Dispatch ready stages in order

For each stage with `status == "ready"`, in ascending stage number order (0, 1, 2, 3, 4, 5), invoke its agent / sub-skill / direct command:

| Stage key | Dispatch via | Target |
|---|---|---|
| `"0"` | direct Bash | `conda run -n sceneconductor python <skill_dir>/src/run_stage2_director.py --scene_dir <scene_dir>` |
| `"1"` | `Skill` tool | `stage2-sub-pointmap-to-separable-stage` |
| `"2"` | `Skill` tool | `stage2-sub-pointmap-to-separable-stage` |
| `"3"` | `Skill` tool | `stage2-sub-pointmap-to-separable-stage` |
| `"4"` | `Skill` tool | `stage2-sub-env-enhance` |
| `"5"` | direct Bash | Blender `render_multi_view.py` |

### Step 3 — Verify and continue

After each sub-skill returns, re-run the inspector to confirm the stage is now `done`. If the done-signal is still absent, surface the error clearly (include the inspector's `plan` entry and any relevant log output) and stop — do not continue to the next stage. Only proceed when the inspector confirms `"status": "done"` for the completed stage.

**Stage 5 — Multi-view render** does not use the `Skill` tool. Dispatch the canonical renderer directly with a Bash command:

```bash
blender --background <scene_dir>/blend/blender_scene.blend \
        --python <general-multi-view-render-skill-dir>/src/render_multi_view.py \
        -- --scene-dir <scene_dir> \
           --brightness-log <scene_dir>/render/brightness_align_log.json
```

Where `<general-multi-view-render-skill-dir>` is `.claude/skills/general-multi-view-render` relative to the SceneConductor root (or its absolute path). This is the **single canonical multi-view renderer** for the whole project — Stage 1, 2, and 3 all route here. The `--samples` flag defaults to 256; pass `--samples 128` to go faster, or `--samples 512` for maximum quality.

The `--brightness-log` flag tells the multi-view script where to find the brightness alignment log (written by Stage 4). The canonical location is `<scene_dir>/render/brightness_align_log.json`. If absent the script falls back to a 0.05 multiplier for neutral views.

Verify Stage 5 is done by re-running the inspector and checking `detected.multiview_perspective_png == true`.

Pass the scene_dir as an argument to stages 1–4 — every sub-skill in this repo is designed to work off a single scene folder. Do not construct additional arguments; the sub-skill's own triggering logic will pick up the files it needs from the folder. If a sub-skill needs additional parameters (e.g. a non-default Blender path), let the user provide those in their original request and include them in the args string.

#### Stage snapshot rule (mandatory)

Immediately after the inspector confirms a stage is `"done"`, copy `blend/blender_scene.blend` to a stage-named snapshot in `<scene_dir>/blend/`:

| Stage confirmed done | Snapshot filename |
|---|---|
| 2 (`stage2-sub-pointmap-to-separable-stage`) | `blend/stage2-sub-build.blend` |
| 3 (`stage2-sub-pointmap-to-separable-stage`) | `blend/stage2-sub-separable.blend` |
| 4 (`stage2-sub-env-enhance`) | `blend/stage2-sub-env.blend` **and** `blend/stage2-scene.blend` |

When stage 4 (`stage2-sub-env-enhance`) is confirmed done, save **two** snapshots: the per-stage `blend/stage2-sub-env.blend`, **and** `blend/stage2-scene.blend`. The latter is the **Stage-2 FINAL output snapshot** — env-enhance is the last sub-stage that modifies geometry/materials (stage 5 only renders), so the canonical blend at that point represents the finished Stage-2 environment.

Stage 1 produces no `.blend`, so no snapshot. Use a simple `cp` (or `shutil.copy2`):

```bash
# Stage 2 done — copy, then inject floor.obj into the snapshot for visual reporting.
# floor.obj (from layout_prediction.json) is added to stage2-sub-build.blend only;
# blend/blender_scene.blend (and all other snapshots) are left untouched.
python3 <this-skill-dir>/src/inject_floor_obj_snapshot.py <scene_dir>

# Stage 4 done — produces BOTH the per-stage snapshot and the Stage-2 final:
cp <scene_dir>/blend/blender_scene.blend <scene_dir>/blend/stage2-sub-env.blend
cp <scene_dir>/blend/blender_scene.blend <scene_dir>/blend/stage2-scene.blend
```

**Idempotency**: if the snapshot already exists (stage was already done on a previous run), overwrite it — the current `blend/blender_scene.blend` is always the authoritative source. Never skip the copy to "save time"; the snapshots are what let the user inspect intermediate geometry without re-running stages.

### Step 4 — Report

When the loop finishes (either all stages done, or a stop was triggered), give the user a compact summary:

- What ran this invocation (and roughly how long each took, if known).
- What was skipped because already done.
- What was blocked, with the specific missing input that blocked it.
- Pointers to the artifacts now in `scene_dir`.

### Step 5 — Finalize layout (stragglers safety-net)

After the loop terminates (all stages done OR a stop was triggered), run the
finalizer as a safety-net to catch any files that were written to the wrong
location by an older or out-of-date sub-skill:


```bash
python3 <this-skill-dir>/src/finalize_layout.py <scene_dir>
```

**Sub-skills now write directly to their canonical subfolders** — the finalizer
is no longer responsible for the initial placement.  Its only job is to sweep
top-level stragglers into the correct destination and print a one-line summary.
In a correctly configured pipeline it is a no-op:

```
[finalize] no stragglers found — all artifacts already in correct subfolders.
```

Canonical output layout (scripts write here directly):

**`json/`** — `polygon_v2.json`, `alignment_metrics_v2.json`, `brightness_align_log.json`, `blender_scene.json`

**`render/`** — `blender_scene_env_preview.png`, `*_env_preview0001.png`, `blender_scene_view_*.png`, `blender_scene_bev_overlay.png`, `blender_scene_v2_bev_overlay.png`

**`inputs/`** — `layout_prediction.json`, `layout-prediction.glb`, `mask_attribute.json`, `pointmap_xz.ply`, `object/`

**`blend/`** — `blender_scene.blend`, `blender_scene_stage*.blend`, `blender_scene.blend1`

**Canonical top-level files that are NEVER moved:** `image.png`. See `FILE_DIRECTORY.md` for the full layout.

The inspector (`inspect_scene.py`) checks canonical subfolder locations first, then falls back to legacy top-level paths with a `[legacy-path]` warning printed to stdout.

Idempotent — safe to re-run.


#### Directory state after a full pipeline run

> For the authoritative file-layout contract see `FILE_DIRECTORY.md` at the SceneConductor root.


## Handling skips and forces

By default the skill honors idempotency — an already-`done` stage is not re-run. The user can override this in their request:

- "rerun everything" / `force=all` → treat every stage's status as `ready` if its inputs exist, ignoring "done" signals.
- "redo stage 3 onward" / `force_from=3` → delete stage 3's output signals (and everything downstream of it) from consideration, so stage 3 re-plans as `ready` and stages 4 and 5 cascade as ready.
- "redo env-enhance onward" / `force_from=4` → delete Stage 4's output signals; stages 4 and 5 cascade as ready.
- "redo multi-view only" / `force_from=5` → delete all five `blender_scene_view_*.png` files; only Stage 5 re-plans as ready.
- "stop after stage 3" / `until=3` → skip stages 4 and 5 (accept separable-stage geometry without running env-enhance).
- "stop after stage 4" / `until=4` → skip Stage 5 (run env-enhance but not multi-view).

Pass these hints into `inspect_scene.py` via CLI flags (`--force all`, `--force-from 3`, `--force-from 4`, `--force-from 5`, `--until 3`, `--until 4`). Valid stage keys are 1–5. Keeping all plan-adjustment logic inside the inspector means the orchestrator stays thin and predictable.

## Blocked-stage conventions

`blocked` means a stage's required inputs do not exist in `scene_dir`. The inspector reports a reason; relay it. Common cases:

- `inputs/pointmap_xz.ply` missing → stage 3 blocked. Stage 4 may still proceed if a stage-2 blend exists (env-enhance only needs a .blend + image), but practically stage 4 needs the polygon for wall/stage geometry, so it is also effectively blocked for full runs.
- No `image.{png,jpg,jpeg}` → stages 1, 3, 4 blocked.
- No `inputs/layout_prediction.json` → stage 1 blocked; if `json/blender_scene.json` already exists from an earlier run, stages 2+ may still proceed.

A `blocked` stage does not stop the pipeline unless a later stage depends on its output. The inspector encodes those dependencies.

## One-sentence mental model

Run `inspect_scene.py`, then for every `ready` stage in ascending order, call the matching sub-skill via `Skill` (stages 1–4) or a direct Blender command (stage 5), re-inspect after each to confirm done, repeat until the plan is fully resolved. Nothing else.

## Reference files
- `src/inspect_scene.py` — pure-Python scene-folder inspector. Produces the plan JSON. Uses only `os`, `json`, `glob`, `argparse`. No Blender. No external dependencies.
- `references/stage_contracts.md` — per-stage input/output filename table. Update this file (not the script) when a sub-skill changes its output filenames or done-signal conditions.
- `agents/` — any agent definitions used by this skill (if present).
