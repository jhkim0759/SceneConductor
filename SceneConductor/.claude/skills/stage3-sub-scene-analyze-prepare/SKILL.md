---
name: stage3-sub-scene-analyze-prepare
description: Given a scene_dir (image + .blend), extract three planning JSONs — `json/object_state.json`, `json/blend_info.json`, and `inputs/relation_graph.json` — without producing any operation plan or Blender edits. Trigger on "/scene-analyze-prepare".
---

## What this skill does

Given a Blender scene already built by the Stage-1+ pipeline (so `scene_dir/` contains `image.png`, `inputs/`, and `blend/blender_scene.blend`), this skill produces three per-scene extractor outputs:

```
image.png  +  blend/blender_scene.blend  +  inputs/{merge_plan, mask_attribute, object_class}.json
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
extract_object_state.py   extract_blend_info.py   Step 3 (inline)
(Qwen-VL on image+masks)  (Blender headless ops)  (vision Agent for groups)
        │                       │                       │
        ▼                       ▼                       ▼
json/object_state.json    json/blend_info.json    inputs/relation_graph.json
```

Downstream consumers (operation planning, environment edits, etc.) live in **other** skills and read these three JSONs as input. This skill is intentionally extractor-only.

## When to skip / when to fail loudly

Required inputs in `<scene_dir>`:

| Path | Required | Why |
|---|---|---|
| `image.png` | yes | reference photo |
| `blend/blender_scene.blend` | yes | current scene to extract from |
| `inputs/masks/*.png` | yes | per-object PNGs (Stage-1 output) |
| `inputs/mask_attribute.json` | yes | post-merge mesh_groups |
| `inputs/object_class.json` | yes | per-id class names (Qwen registry) |
| `inputs/merge_plan.json` | recommended | pre-merge mesh_groups |
| `inputs/object_state_annotated_mask.png` | yes (Step 1 generates it) | required by Step 3 (stage3-relation-graph agent) |

If `image.png` or the .blend is missing, fail with a clear message — do NOT try to invoke an upstream pipeline. The user is expected to have run `/stage1-initialize-scene` and `/stage2-environment-construction` already.

## How to run

The skill is orchestrated by the main agent (you). The flow is **3 deterministic preprocessors**, run in order.

Outputs:
- Steps 1 and 2 (this skill's own scripts) write to `<scene_dir>/json/`.
- Step 3 (inline `stage3-relation-graph` agent) writes `inputs/relation_graph.json`. **This is the SOLE producer of `relation_graph.json`** in the entire pipeline — Stage 1 no longer builds it.

Inputs (`<scene_dir>/inputs/`) are produced upstream by `/stage1-initialize-scene` + `/stage2-environment-construction` and are read but never written here (except `inputs/relation_graph.json`, which this skill owns).

### Step 1 — Extract object_state.json (Qwen-VL)

> **Cache reuse**: Step 1 (object_state.json) and Step 3.2 (thumbnails)
> are auto-skipped if Stage 1 already produced `inputs/object_state.json` and `inputs/thumbnails/`.
> This avoids redoing ~1–2 min of Qwen-VL inference plus a few seconds of thumbnail cropping on
> every refinement pass. Pass `--force` to make_thumbnails to override.

```bash
# MUST run in the `scenegen` conda env — it has torch + transformers.
# The default `sceneconductor` env lacks torch; using it makes the Qwen load
# silently fail and every object falls back to attached_to=["none"]/category="unknown".
conda run -n scenegen python3 .claude/skills/stage3-sub-scene-analyze-prepare/src/extract_object_state.py \
  --scene_dir <scene_dir> \
  --gpu 0 \
  --model Qwen/Qwen3.5-27B \
  --local_files_only
```

Wraps `src/generate_object_state_json.py` and forces its `--output` to land in the `json/` folder. Output: `<scene_dir>/json/object_state.json` (attachment per object, alignment groups, stacking).

GPU pick: default 0. The 27B-dense model needs ≳60 GiB of GPU memory at bf16 (or use FP8/AWQ variants). For a single-GPU box without that headroom, rely on `device_map="auto"` to offload across CPU+GPU.

Model: `Qwen/Qwen3.5-27B` is the dense multimodal model documented at `DIRECTORYS.yaml:qwen_vl_model_id`. Local weights live at `./checkpoints/qwen/Qwen3.5-27B` (see `checkpoints/README.md`); pass that path as `--model` to load from the vendored copy instead of the HuggingFace cache.

### Step 2 — Extract blend_info.json (headless Blender)

```bash
python3 .claude/skills/stage3-sub-scene-analyze-prepare/src/extract_blend_info.py \
  --scene_dir <scene_dir>
```

Internally invokes `src/external_blend_runner.py` twice (once with `list_objects` for everything, once with `metrics` for OOB+collisions), then aggregates into a single JSON. Output: `<scene_dir>/json/blend_info.json`.

The metrics op intentionally tries to save the .blend back to `/dev/null` and fails on save — that's expected and harmless; the script post-patches `metrics.success = true` once it confirms the analysis fields are present.

### Step 3 — Build relation_graph.json (inline workflow)

Semantic groups (which chairs surround WHICH table, which TV sits on WHICH shelf, which posters share WHICH wall) are produced inline in three deterministic sub-steps + one vision Agent call. This used to live in a separate `/scene-relation-graph` skill that has been absorbed here.

If `inputs/relation_graph.json` already exists from a previous run, you can skip this step — but if the .blend has changed materially since then (objects added/removed/renamed), regenerate it.

#### Step 3.1 — Validate inputs
```bash
python3 .claude/skills/stage3-sub-scene-analyze-prepare/src/validate_inputs.py \
  --scene_dir <scene_dir>
```
Exits non-zero with a list of missing files. Required: `image.png`, `inputs/blend_info.json` (or `json/blend_info.json` from Step 2), `inputs/object_class.json`, `inputs/object_state_annotated_mask.png`, `inputs/masks/*.png`.

#### Step 3.2 — Generate per-object thumbnails
```bash
python3 .claude/skills/stage3-sub-scene-analyze-prepare/src/make_thumbnails.py \
  --scene_dir <scene_dir>
```
For each `inputs/masks/<id>.png`, crops `image.png` around the mask bbox and writes `inputs/thumbnails/obj_<id>_<class>.png`. Skips existing; pass `--force` to regenerate.

#### Step 3.3 — Run stage3-relation-graph agent
Spawn `Agent(subagent_type="stage3-relation-graph", prompt="Build the relation graph at <scene_dir>")`. Provide absolute paths to:
- `image.png`
- `inputs/object_state_annotated_mask.png`
- `json/blend_info.json` (authoritative for chair↔table collision assignments — the agent uses `metrics.collisions` first, vision second)
- `inputs/object_class.json`

> Note: the per-object thumbnails generated in Step 3.2 are still produced for caching / other consumers, but are **not** fed to the relation_graph agent — the annotated overview + JSONs carry everything its structural output needs.

Agent writes `<scene_dir>/inputs/relation_graph.json`.

#### Step 3.4 — Sanity-check JSON
- JSON loads.
- Every `obj_id` referenced exists in `object_class.json`.
- Every object↔object group's `anchor`/`members` also appears in `edges` (skip for `mounted_on_same_wall` / `co_illuminates` whose anchors are abstract).
- Warn (don't fail) if any object is in zero groups.

If validation fails, re-spawn the Agent with failure message appended to its prompt.

**Edge vocabulary** (the only allowed edge types — Agent must respect):
| type | meaning | anchor / members |
|---|---|---|
| `seated_around` | chairs around a table | anchor=table |
| `on_top_of` | small object on a surface | anchor=surface |
| `adjacent_to` | floor objects abutting | symmetric |
| `mounted_on_same_wall` | wall decor sharing a wall | anchor=wall_id |
| `co_illuminates` | ceiling lights jointly lighting a region | anchor=region |

Custom edge type allowed only as exception with `custom_type` field.

**Output schema** (`inputs/relation_graph.json`):
```json
{
  "scene_dir": "<absolute>",
  "groups": [{"group_id":"G1","name":"...","edge_type":"seated_around","anchor":"obj_6","members":["obj_2",...],"evidence":"..."}],
  "edges": [{"source":"obj_2","target":"obj_6","type":"seated_around","confidence":0.95,"evidence":"..."}],
  "cross_group_edges": [{"source_group":"G7","target_group":"G1","type":"co_illuminates","evidence":"..."}]
}
```

Output that downstream consumers read:
- `<scene_dir>/inputs/relation_graph.json` — `groups`, `edges`, `cross_group_edges`.

## Files in this skill

| Path | Purpose |
|---|---|
| `SKILL.md` | This file — orchestration doc. |
| `src/extract_object_state.py` | Step 1 — Qwen-VL → `object_state.json`. |
| `src/extract_blend_info.py` | Step 2 — Blender headless → `blend_info.json`. |
| `src/validate_inputs.py` | Step 3.1 — validate required inputs before `stage3-relation-graph` agent. |
| `src/make_thumbnails.py` | Step 3.2 — crop per-object thumbnails from image.png + masks. |
| `.claude/agents/stage3-relation-graph.md` | Step 3.3 — vision + geometric agent that builds `inputs/relation_graph.json`. |
