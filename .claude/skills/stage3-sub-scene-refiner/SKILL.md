---
name: stage3-sub-scene-refiner
description: Single-shot Stage 3 auto-pass — heuristic + graph_tool → operation_plan.json, apply, render. No LLM. Trigger ONLY on direct invocation ("/scene-refiner"), never auto-triggered from "fix the scene".
---

# scene-refiner

Single-shot: scene_dir → operation_plan.json → modified .blend → 5-view render.

## Pipeline (in order)

1. **Verify** scene_dir inputs.
2. **Backup** prior `operation_plan.json`.
3. *(optional)* **Re-extract** `json/blender_scene.json` from working blend if `--re-extract-json`.
4a. **Heuristic pre-plan** via `heuristic_planner.py` → writes `json/heuristic_ops.json` (scale normalization, floor/ceiling attach, support attach — no LLM).
4b. **Graph tool planner** (deterministic, no LLM) → emits attach/attach_to_wall ops from relation_graph → writes `json/graph_ops.json`.
4c. **Merge** via `merge_ops.py` → writes `operation_plan.json` (combined, deduplicated, priority-sorted).
5. **Refresh working copy** by copying source→working, unless `--copy-source=false`.
6. **Apply** plan to working blend in one Blender session.
7. **Render** 5 multi-views to `render/planned/`.
8. **Report** op counts, pass/fail, paths to verify.

If any step fails, stop and report. Do NOT retry.

## Parameters

Positional: `<scene_dir>` (required, absolute path).

Optional overrides (all default to current behavior — single-invocation users can ignore them):

| Flag | Default | Effect |
|---|---|---|
| `--source-blend <path>` | `<scene_dir>/blend/blender_scene.blend` | Source blend (only read, never modified) |
| `--working-blend <path>` | `<scene_dir>/blend/stage3-sub-planned.blend` | Blend that apply writes to |
| `--copy-source=true\|false` | `true` | If `true`, copy source→working before apply (refreshes working from source). If `false`, apply on top of existing working blend. |
| `--re-extract-json` | off | If set, run `extract_scene_json.py` on `--working-blend` before planning, overwriting `--scene-json`. Only meaningful when `--copy-source=false` (otherwise the working blend was just refreshed from source and the existing JSON already matches). |
| `--scene-json <path>` | `<scene_dir>/json/blender_scene.json` | JSON the planner agent reads |
| `--operation-plan <path>` | `<scene_dir>/operation_plan.json` | Plan output path |
| `--render-dir <subpath>` | `render/planned` | Render output subdir under scene_dir |

The orchestrating agent (you) MUST parse these from the invocation args, fall back to defaults for anything unspecified, and echo the resolved values in the first status message.

## Required inputs

`<scene_dir>` must contain: `image.png`, `inputs/object_state_annotated_mask.png`, `inputs/object_class.json`, `inputs/relation_graph.json`, `json/blender_scene.json`, `json/blend_info.json`, `json/object_state.json`, `json/polygon_v2.json`, the file at `--source-blend`. If any are missing, fail loudly with the list — do NOT auto-run upstream skills.

## Required environment

- `src/blend_ops/` (apply_ops + render_multi_view wrappers, vendored inside this skill)
- Blender 4.2.1 LTS (path read from `DIRECTORYS.yaml::blender_bin` or env var `BLENDER`; override per-invocation via `SCENE_EVAL_BLENDER`)

## Execution

Let `$SCENE = <scene_dir>`, `$SKILL = .../skills/scene-refiner`, `$BLENDER = ${SCENE_EVAL_BLENDER:-${BLENDER:-blender}}` (canonical path read from `DIRECTORYS.yaml::blender_bin`).

### Step 1 — Verify
Check every Required Input exists. Missing → stop with full list, message: "scene-refiner inputs missing — run scene-analyze-prepare first if upstream prep wasn't done."

### Step 2 — Backup prior plan
```bash
ts=$(date -u +%Y%m%d_%H%M%S)
[ -f "<--operation-plan>" ] && cp "<--operation-plan>" "<--operation-plan>.bak_$ts"
[ -f "$SCENE/json/heuristic_ops.json" ] && cp "$SCENE/json/heuristic_ops.json" "$SCENE/json/heuristic_ops.json.bak_$ts"
[ -f "$SCENE/json/graph_ops.json" ] && cp "$SCENE/json/graph_ops.json" "$SCENE/json/graph_ops.json.bak_$ts"
```

### Step 3 — Re-extract scene JSON (if `--re-extract-json`)
```bash
"$BLENDER" -b "<--working-blend>" --python "$SKILL/../stage3-scene-refinement/_migrated/extract_scene_json.py" -- "<--scene-json>"
```
Skip entirely if the flag is not set.

### Step 4a.0 — LLM visual ground-object selection (optional but recommended)
Before running the heuristic planner, you (the orchestrating agent) SHOULD make one multimodal pass to decide which objects visually rest on the floor. This makes floor-grounding robust when `object_state.json` has no `attached_to` signal and classes are ambiguous.

1. Read (multimodal): `image.png`, `inputs/object_state_annotated_mask.png`, and `inputs/object_class.json`.
2. For each labelled object, judge whether it visually RESTS ON THE FLOOR (chairs, tables, counters, shelves, bins on the ground) versus sits on a surface, is mounted on a wall, or hangs from the ceiling (TVs on shelves, posters/picture frames/chalkboards, fluorescent lights).
3. Write `json/ground_objects.json`:
   ```json
   {"ground_objects": ["obj_3", "obj_8"], "not_ground_objects": ["obj_1"], "reason": {"obj_1": "TV resting on bookshelf"}}
   ```
   - `ground_objects`: obj_ids you judge to be on the floor (adds to the union — include any the class taxonomy would miss).
   - `not_ground_objects`: obj_ids that must NOT be floor-attached (LLM exclusion authority — removes even FLOOR-class candidates).
   - `reason`: optional per-obj_id rationale.
4. If you skip this step, Step 4a falls back to class-based selection only (still grounds class-matched floor objects).

### Step 4a — Heuristic pre-plan
```bash
python3 "$SKILL/src/heuristic_planner.py" --scene-dir "$SCENE"
```
Reads `inputs/merge_plan.json`, `json/blender_scene.json`, `json/object_state.json`, `inputs/relation_graph.json`, `inputs/object_class.json`, and (optional) `json/ground_objects.json`. Writes `json/heuristic_ops.json`. Runs in seconds — no LLM. If a file is missing it warns and degrades gracefully (does not fail hard).

**Force floor-grounding:** the floor/ceiling pass no longer requires `attached_to=["floor"]`. It selects a ground set as
`final_ground = (FLOOR-class matches ∪ attached_to=floor ∪ ground_objects.json::ground_objects) − (relation_graph on-surface/wall/ceiling members ∪ WALL/CEILING-class objects ∪ support-covered ∪ ground_objects.json::not_ground_objects)`
and emits an `attach` (anchor=Floor, relation=on) op for EVERY object in that set (de-duplicated against the support pass). The optional `json/ground_objects.json` from Step 4a.0 takes union+exclusion authority over the class-based set.

Print the heuristic summary to the user: `scale=N, floor=N, ceiling=N, support=N`.

### Step 4b — Graph tool planner
The conda env name is NOT hardcoded — resolve it from `DIRECTORYS.yaml` key `conda_envs.sceneconductor` (single source of truth), then reuse `$ENV` for the conda commands below:
```bash
ENV=$(python3 -c "import yaml; print(yaml.safe_load(open('DIRECTORYS.yaml'))['conda_envs']['sceneconductor'])")
```
```bash
conda run -n "$ENV" python3 "$SKILL/src/graph_tool_planner.py" --scene-dir "$SCENE"
```
Reads `inputs/relation_graph.json` + `json/blend_info.json`. For `mounted_on_same_wall` groups emits `attach_to_wall` ops; for `on_top_of` groups emits `attach` ops. `seated_around` groups are skipped (handled by island refinement after validation). Writes `json/graph_ops.json`. No LLM — runs in under a second.

Print: `wall_attach=N surface_attach=N skipped_seated_around=N`.

### Step 4c — Merge ops
```bash
python3 "$SKILL/src/merge_ops.py" \
    --scene-dir "$SCENE" \
    --graph-ops "$SCENE/json/graph_ops.json"
```
Combines `json/heuristic_ops.json` + `json/graph_ops.json` → `operation_plan.json`. Priority order: heuristic (1) < graph_tool/llm (3) < polygon_clamp (4). Prints merge summary.

### Step 5 — Refresh working copy (if `--copy-source=true`, the default)
```bash
cp -f "<--source-blend>" "<--working-blend>"
```
Source mtime MUST stay unchanged. If `--copy-source=false`, skip — caller is preserving prior working-blend state.

### Step 6 — Apply
```bash
python "$SKILL/src/apply_plan.py" "<--operation-plan>" "<--working-blend>" "<--working-blend>"
```
On non-zero exit, report `[apply_plan]` summary + first failed op. Do NOT retry.

### Step 7 — Render
```bash
python "$SKILL/src/render_planned.py" "$SCENE" "<--working-blend>" --render-dir "<--render-dir>"
```
(Note: `render_planned.py` currently writes to `render/planned/` by default; if `--render-dir` differs, the orchestrator may need to invoke directly via `render_multi_view` or extend the script.)

5 outputs in `<scene_dir>/<--render-dir>/`: `blender_scene_view_{perspective,bev,wide,topcorner,topcorner_opposite}.png`.

### Step 8 — Report
- resolved parameter values
- op counts by action (`update_size`, `update_rotation`, `update_layout`, `attach`, `attach_to_wall`)
- apply pass/fail (failed-op messages if any)
- render pass/fail
- paths: `<--operation-plan>`, `<--working-blend>`, perspective render
- source-blend mtime unchanged confirmation
- `json/heuristic_ops.json` — heuristic op counts (scale/floor/support)
- `json/graph_ops.json` — graph tool op counts (wall_attach/surface_attach)
- merge summary (heuristic=N graph_tool=N final=N)

## Files in this skill

| File | Role |
|---|---|
| `src/heuristic_planner.py` | Step 4a — scale/floor/ceiling/support ops, no LLM |
| `src/graph_tool_planner.py` | Step 4b — deterministic attach/attach_to_wall ops from relation_graph |
| `src/merge_ops.py` | Step 4c — merges heuristic + graph_tool op streams |
| `src/apply_plan.py` | Step 6 — applies `operation_plan.json` to working blend |
| `src/render_planned.py` | Step 7 — renders 5 multi-views from working blend |
| `src/blend_ops/` | Vendored apply_ops + render_multi_view wrappers |

## Out of scope

Iteration, island-task resolution, multi-scene batching, upstream pipeline. Use `scene-analyze-prepare` + earlier stages before this; use `/stage3-scene-refinement` or `/island-refiner` after.

## Failure modes

| What broke | Report |
|---|---|
| Required input missing | exact missing paths + "run scene-analyze-prepare first" |
| Planner emitted disallowed action / invalid JSON | which action / which JSON key + vocabulary reminder |
| `apply_plan.py` non-zero | `[apply_plan]` summary + first failed op message |
| Render non-zero | apply success reported separately; render error surfaced |
| Source-blend mtime changed | CRITICAL — copy step was bypassed; investigate |
