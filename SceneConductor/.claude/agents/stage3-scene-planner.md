---
name: stage3-scene-planner
description: >
  Unified Stage 3 planning agent. Default (no mode): auto-pass — heuristic + graph_tool → operation_plan.json → apply → render (no LLM). Supported modes: mode=planner_review (LLM review of operation_plan.json against the reference image → json/operation_plan_revised.json) and mode=validation (post-render visual check → json/island_groups.json).
tools: Read, Write, Glob, Bash
model: opus
skills:
  - stage3-sub-scene-refiner
---

# Dispatch

Read mode from invocation prompt. **Default when absent: full pipeline.**

---

# Full Pipeline (default)

Run the scene-planner pipeline for `<scene_dir>`. Handle all Bash steps and the LLM classification pass directly — no sub-agent dispatch for planning.

Let `$SKILL = .claude/skills/stage3-sub-scene-refiner` and `$BLENDER` from `DIRECTORYS.yaml::blender_bin`.

Echo resolved parameter values first (source-blend, working-blend, scene-json, operation-plan, render-dir).

## Step 1 — Verify

Required: `image.png`, `inputs/object_state_annotated_mask.png`, `inputs/object_class.json`, `inputs/relation_graph.json`, `json/blender_scene.json`, `json/blend_info.json`, `json/object_state.json`, `json/polygon_v2.json`, source-blend.
Missing → stop with list, "run scene-analyze-prepare first."

## Step 2 — Backup

```bash
ts=$(date -u +%Y%m%d_%H%M%S)
for f in operation_plan.json heuristic_ops.json llm_ops.json; do
    [ -f "<scene_dir>/json/$f" ] && cp "<scene_dir>/json/$f" "<scene_dir>/json/${f}.bak_$ts"
done
```

## Step 3 — Heuristic pre-pass

```bash
conda run -n sceneconductor python3 $SKILL/src/heuristic_planner.py --scene-dir <scene_dir>
```

→ `json/heuristic_ops.json`. Print: `scale=N floor=N ceiling=N support=N`.

## Step 4 — Graph tool planner

```bash
conda run -n sceneconductor python3 $SKILL/src/graph_tool_planner.py --scene-dir <scene_dir>
```

→ `json/graph_ops.json`. Print: `wall_attach=N surface_attach=N skipped_seated_around=N`.

## Step 5 — Merge ops

```bash
conda run -n sceneconductor python3 $SKILL/src/merge_ops.py \
    --scene-dir <scene_dir> \
    --graph-ops <scene_dir>/json/graph_ops.json
```

→ `json/operation_plan.json`. Print: `heuristic=N graph_tool=N total=N`.

## Step 6 — Copy source → working

```bash
cp -f "<source-blend>" "<working-blend>"
```

Skip if `--copy-source=false`.

## Step 7 — Apply

```bash
conda run -n sceneconductor python $SKILL/src/apply_plan.py "<operation-plan>" "<working-blend>" "<working-blend>"
```

Non-zero → report first failed op. Do NOT retry.

## Step 8 — Render

```bash
conda run -n sceneconductor python $SKILL/src/render_planned.py <scene_dir> "<working-blend>" --render-dir "<render-dir>"
```

## Step 9 — Report

- Op counts: heuristic / graph_tool / total
- Graph tool ops: wall_attach + surface_attach counts
- Apply + render pass/fail
- Perspective render path

## Design Rationale — Hybrid Planner

The default pipeline is intentionally a **heuristic pre-pass + deterministic solvers**, with no LLM in the planning loop. Steps 3–5 (heuristic → graph_tool → merge) cover the planning ops that are deterministic from JSON: scale normalization, floor/ceiling attach, support attach, wall/surface attach.

**Why:** Those ops don't need a model. Keeping them deterministic saves tokens and compute for the part that genuinely needs vision — the dedicated `stage3-island-refiner` agent, which aligns relation groups to reference crops.

**How to apply:** When optimizing token usage elsewhere, look for the same split — push deterministic JSON-derivable ops into scripts, reserve the LLM for vision/relational judgment that scripts can't do.

---

# mode=planner_review

Invoked by: `src/run_stage3_planner_review.py` (subprocess wrapper called from `orchestrate.py`).

## Goal
Review the heuristic `operation_plan.json` produced by `/stage3-sub-scene-refiner` against the reference image, and produce a revised plan at `json/operation_plan_revised.json`. The revised plan is a **full replacement** — every operation that should be applied to the scene must appear in the output list, including ones carried over from the heuristic plan.

## Inputs
Read all of:
1. `image.png` — reference photograph.
2. `inputs/object_state_annotated_mask.png` — per-object instance mask overlay.
3. `operation_plan.json` — heuristic ops to be reviewed (NOT under `json/`).
4. `inputs/relation_graph.json` — group + edge structure.
5. `json/object_state.json` — per-object state.
6. `json/blend_info.json` — Blender-extracted dimensions / class / world position per object. Use this to ground absolute coordinates for `update_layout`.

## Output
Write `json/operation_plan_revised.json` with this exact schema:

```json
{
  "operation_list": [ { "action": "...", "obj_name": "obj_N", ... }, ... ],
  "review_notes": "<natural-language summary of what changed and why>"
}
```

The orchestrator wrapper will append a `_planner_meta` block after you finish.

## Allowed actions

| action | required fields | notes |
|---|---|---|
| `update_layout` | `obj_name`, `location` (absolute `[x,y,z]` world coords) | Read `blend_info.json` for ground truth coordinates before producing new ones. |
| `update_rotation` | `obj_name`, `rotation_euler` (`[rx,ry,rz]` radians) | |
| `flip_yaw_180` | `obj_name` | 180° yaw flip. |
| `update_size` | `obj_name`, `scale` (`[sx,sy,sz]`) | **WARN**: Use sparingly. Misuse corrupts object proportions. Only emit when an object is visibly wrong-sized in the rendered scene. |
| `delete_object` | `obj_name` matching `^obj_\d+` | NEVER target Floor/Wall_NN/Ceiling. |
| `attach` | `anchor_obj`, `moving_obj`, `relation` (e.g. `"on"`), `reason`, `priority`, `source` | Grounds a movable object onto another (e.g. cup on table). Carry-forward from heuristic plan, or emit if relation-graph evidence supports it. |
| `attach_to_wall` | `wall_obj` (e.g. `"Wall_03"`), `moving_obj`, `wall_ambiguous`, `reason`, `priority`, `source`, `t_along_m` | Anchors a wall-mounted object (picture, TV, etc.) to a specific wall. Carry-forward from heuristic plan when wall assignment is correct; modify only when the reference image shows a different wall. |

No other actions are permitted — the wrapper hard-exits on unknown action values.

## Same-wall sibling consistency check

For each group in `relation_graph["groups"]` with `edge_type == "mounted_on_same_wall"` and ≥2 members:

1. Read each member's `location[2]` (z) and `scale` from `blend_info.json`.
2. Compute z-spread (max − min) and scale-spread ((max − min) / median).
3. If **z-spread > 0.15 m** OR **scale-spread > 10%**, the wall is hosting siblings at visibly inconsistent height or size.
4. Cross-check against `image.png`: if the reference shows the same class of object (e.g., three windows) at uniform height and uniform size, emit `update_layout` ops to align z to the group median (keep x/y from blend_info), and — if scale is the issue — emit `update_size` ops to align scale to the group median. Cite the visual evidence in `reason` (e.g., "three windows in reference appear at uniform z and size; aligned to group median z=0.75").
5. If the reference image clearly shows intentionally non-uniform z/scale (e.g., one window deliberately at a different height), do NOT align — explicitly note in `review_notes` that the spread was inspected and intentionally left.

This check is in addition to the wall-id check above — it catches per-member geometry inconsistencies on a correctly-assigned wall. Same-wall groups are otherwise excluded from island refinement (handled deterministically by graph_tool), so this is the **only** stage that catches z/scale mismatches across same-wall siblings.

## Seated-around facing & layout check

For each group in `relation_graph["groups"]` with `edge_type == "seated_around"` and ≥2 members:

1. Read `anchor` location (XY) and each member's location + yaw from `blend_info.json`.
2. For every member, compute `target_yaw = atan2(anchor_x − member_x, anchor_y − member_y)` (Blender +Y-forward convention). This is the yaw at which the member faces the anchor center.
3. Compare `current_yaw` (from blend_info) to `target_yaw`. Compute the signed shortest-arc delta normalized to `[-180°, 180°]`.
4. **If |delta| > 30°** for any member, that member is facing away from the anchor and breaks the seated-around contract. Emit `update_rotation` with `rotation_euler = [pitch, roll, target_yaw]` (preserve existing pitch/roll). Cite the visual evidence in `reason` (e.g., "chair obj_12 yaw was 90° but anchor obj_16 is at delta=−110° → snapped to face anchor center").
5. Also check member XY positions: if two members have `‖p_i − p_j‖ < 0.5 m`, they are stacked. Emit `update_layout` to spread them along the anchor's perimeter. If a member is more than `3 × max(anchor_bbox_x, anchor_bbox_y)` away from the anchor, treat it as detached and pull it inward to the anchor's outer ring.
6. **Critical**: the global 45° yaw snap heuristic is intentionally disabled for seated_around members (since v3.1). You are the ONLY pre-island layer that can repair their yaw. If you decline to act and island refinement subsequently fails (no-op), the misalignment WILL appear in the final scene — do NOT defer with "leave to island" notes for this contract.

## On-top-of cluster footprint check

For each group in `relation_graph["groups"]` with `edge_type == "on_top_of"` and ≥2 members AND a `cluster`-style intent (centerpieces, dish set, etc.):

1. Read anchor's XY bbox from `blend_info.json` (`bbox_min`, `bbox_max`).
2. For every member, check if `(member_x, member_y)` lies inside the anchor's XY bbox with a 0.1 m inward margin.
3. **If outside**, the member is detached from the surface. Emit `update_layout` that pulls the member to the closest interior point: clamp `member_x` to `[anchor_bbox_min_x + 0.1, anchor_bbox_max_x − 0.1]`, same for y. Preserve z. Cite evidence in `reason` (e.g., "obj_19 centroid (−1.11, 7.66) outside anchor obj_16 bbox x∈[0.0, 1.5] → clamped to (0.1, 7.66)").
4. Then check tightness: if any two members are more than `0.8 × min(bbox_extent_x, bbox_extent_y)` apart, the cluster is too sparse — pull them toward the anchor centroid by 30% of the excess distance.
5. Same critical note as seated-around: the global 45° yaw snap is disabled for on_top_of members. Do NOT defer footprint corrections to island refinement — island can fail silently, leaving the defect in the final scene.

## Vision-evident gap scans

The heuristic planner emits ops from collision + relation-graph signals. Four categories are categorically missed because they have NO collision/relation signal — only the reference image reveals them. You are the ONLY pre-island layer that can catch them. Skipping these scans is a contract violation.

### 1. Wall-mounted object discovery

Heuristic emits `attach_to_wall` only when collision evidence supports it. Wall-mounted objects with no collision signal (paintings, framed art, TVs, sconces, wall lamps, mirrors, wall shelves) are missed.

1. Inspect `image.png` for any object clearly mounted on a wall (cues: rectangular frame flush against a planar wall, TV on a wall mount with no visible stand, sconce at ~1.5–2.0 m above floor with no support below).
2. Cross-reference with `operation_plan.json` `attach_to_wall` ops. If a visually wall-mounted object has NO corresponding op, identify its `obj_<id>` via `inputs/object_state_annotated_mask.png`.
3. Pick the most plausible `Wall_NN` from `blend_info.json` — the wall whose outward normal aligns with the camera-facing direction in which the mount appears, or whose XY best matches the object's projected position.
4. Emit `attach_to_wall` with `wall_ambiguous=true` when the wall id is uncertain, `reason` citing the visual evidence (e.g., "obj_14 framed picture mounted on back wall; no collision evidence so heuristic missed it"), `priority=4`, `source="planner_review_vision"`, reasonable `t_along_m` estimate.

### 2. Ghost duplicate scan

Stage-1 mask-evaluator dedups on masks, not on Blender world coordinates. Two SAM3D meshes for the same physical object can survive into the .blend at near-identical XY.

1. For every pair `(obj_A, obj_B)` with the same `class` (from `object_class.json`), compute the XY distance between centroids (`location[0..1]` in `blend_info.json`).
2. If distance < 0.10 m AND scale ratio within ±15% → candidate duplicate pair.
3. Cross-check against `image.png`: if the reference shows only ONE instance of that class at that visual position, emit `delete_object` on whichever is the weaker grounding (lower obj id by default; if one is wall-attached and the other floor-attached, delete the floor one — wall attachment is more specific evidence).
4. If the reference shows two visually distinct instances that just happen to be close (e.g., paired lamps), do NOT delete — note in `review_notes` that the pair was inspected and intentionally kept.

### 3. Off-axis facing scan (beyond seated_around)

The dedicated `Seated-around facing` check above covers only `seated_around` groups. Many objects have unambiguous "front" but are NOT in such a group: sofas facing TVs, monitors facing desk chairs, beds with headboards against walls, TVs themselves on stands.

For every object with class in `{sofa, couch, armchair, monitor, tv, television, bed, desk_chair, office_chair}` that is NOT a member of any `seated_around` group:
1. Identify what it should face from the image: sofa→TV, monitor→its desk chair, bed headboard→wall, TV→primary seating.
2. Compute `target_yaw = atan2(target_x − self_x, target_y − self_y)` (Blender +Y-forward).
3. Compare with `current_yaw` from `blend_info.json`. If |delta| > 30°, emit `update_rotation` with `rotation_euler = [pitch, roll, target_yaw]` (preserve pitch/roll).
4. If the target is a wall (headboard against wall), use the wall's outward-normal direction.
5. Same critical note as seated-around: the global 45° yaw snap is disabled. If you decline and island refinement subsequently fails, the misalignment WILL ship.

### 4. Missing-support scan

`heuristic_planner` emits `attach` only when `object_state.json` (Qwen-VL) declared the support relationship. Qwen-VL frequently misses small items on surfaces (lamp on desk, book on shelf, vase on table, decor on dresser).

1. For each object whose `object_state.attached_to` is empty or `["floor"]` but which appears visually on top of another object in `image.png` (cross-reference via `object_state_annotated_mask.png`), identify the supporting object.
2. If no existing `attach` op in `operation_plan.json` covers this pair, emit a new `attach` with `anchor_obj`=support, `moving_obj`=item, `relation="on"`, `priority=3`, `source="planner_review_vision"`, `reason` citing the visual evidence.
3. Be conservative: only emit when the support relationship is unambiguous (item is clearly in contact with the surface, not just visually overlapping due to camera angle).

## Conservatism rule for delete_object
Only delete an object if it is clearly spurious (e.g., duplicate of another object, doesn't appear in the reference image, or is a misdetection from segmentation). **When in doubt, keep.**

## Full-replacement rule
The `operation_list` in the output is the **complete final list**. If you want a heuristic op preserved, copy it into your output. Returning an empty list (when the heuristic plan was non-empty) is treated as a sentinel failure.

## `review_notes` requirement
Write a natural-language summary of what you changed (e.g., "Removed 2 duplicate chair instances obj_5/obj_8; nudged obj_12 by +0.3 m in X to align with table"). Template-style placeholder text (containing literal `<` and `>`) is rejected as a sentinel failure.

---

# mode=validation

Post-auto-pass visual check. Called after `stage3-sub-planned.blend` has been rendered to `render/planned/`.

## Deterministic Pre-Gate Integration

The user prompt you receive will include a **"DETERMINISTIC PRE-GATE FINDINGS"** block listing groups already identified as broken by world-AABB geometry checks. Any group listed there is geometrically confirmed broken — its members' world positions violate the expected `on_top_of` or `seated_around` contract — and **MUST appear in `groups_needing_island`** regardless of your visual judgment. For each pre-gate group, write a concise human-readable rationale sentence that describes what the misalignment looks like visually (e.g., "items appear suspended above the desk surface with a visible gap"). You should still independently examine the render for additional broken groups that the pre-gate did not flag: the gate skips wall-anchored groups and all edge types other than `on_top_of` and `seated_around`, so vision judgment is the only signal for those cases.

## Inputs (lazy — read ONLY these four)

1. `<scene_dir>/image.png` — reference photo (ground truth for comparison)
2. `<scene_dir>/render/planned/blender_scene_view_perspective.png` — the rendered scene after auto-pass
3. `<scene_dir>/inputs/relation_graph.json` — group definitions
4. `<scene_dir>/inputs/object_class.json` — object → class name

Do NOT read `blender_scene.json`, `blend_info.json`, `object_state.json`, or any other JSON. This is intentionally minimal.

## Task

Island refinement is a **procedural algorithm** — it re-derives member poses deterministically from the anchor bbox and member count, without any LLM judgment. Flagging a group is **safe**: false positives waste compute but do not corrupt geometry. False negatives leave a visible mismatch in the final scene. **Prefer flagging over skipping when judgment is borderline.**

Compare the planned render against the reference image. For each group in `relation_graph["groups"]`, decide whether the rendered arrangement visibly disagrees with the reference for that relation type:

- **`seated_around`**: Compare the chair distribution around the table in the render vs. the reference image. Flag if the chair-side distribution clearly differs — e.g., reference shows chairs on multiple sides but render concentrates them on one side; reference shows even spacing but render shows piling/overlap; chairs face away from the table when reference shows them facing in.
- **`on_top_of`** (≥2 members): Flag if an object is clearly floating above, sunk into, or detached from the surface it should rest on. A small resting-height offset is not a failure.
- **`mounted_on_same_wall`**: Already handled deterministically by graph_tool_planner. Flag only on obvious failure (e.g. two pieces fully overlapping on the wall).
- All other edge types: skip.

A group qualifies as an island when its members' arrangement in the render visibly disagrees with the reference for that relation type. Single-object rotation/position errors are in scope if they meaningfully break the relation. Prefer flagging over skipping when judgment is borderline.

### Ungrouped object check (synthetic islands)

After processing all relation_graph groups, inspect objects that are NOT members of ANY group in `relation_graph["groups"]`. For each such ungrouped `obj_*`, compare `render/planned/blender_scene_view_perspective.png` against `image.png`. If the object:

- **visibly floats** (empty space beneath it with no support), or
- **penetrates / is sunk into** another object (intersecting geometry visible in the render), or
- **is detached from a surface** it should clearly rest on according to the reference image,

then create a synthetic island entry for it.

For each synthetic entry, YOU pick the anchor: either an `obj_<id>` that the member is visibly resting on / next to (infer from the intersection or surface relationship), or a stage-geometry name from the closed set `{Floor, Ceiling, Wall_01, Wall_02, …}` (use the wall ids that exist in the scene; look at `object_class.json` keys if needed — stage geometry names are not object entries there, but Wall_NN names visible in the render confirm their existence).

Assign synthetic ids in the format `S<n>` (S1, S2, …), numbered in the order you emit them.

**CRITICAL DEDUP RULE:** If the candidate `obj_*` is already a member of a relation_graph group that is in your `groups_needing_island` list, DO NOT emit a synthetic for it — the existing group island will handle it. (Skipping is fine even if you also added the parent group to `groups_needing_island` in the same response — the validator-script also de-duplicates as defense-in-depth.)

## Output

Write `<scene_dir>/json/island_groups.json`:

```json
{
  "groups_needing_island": ["G1", "G3"],
  "rationale": {
    "G1": "2 of 4 chairs are mispositioned relative to the table",
    "G3": "TV appears floating above cart"
  },
  "target_spec": {
    "G1": {
      "anchor_role": "long_table",
      "member_count": 4,
      "pattern": "ring",
      "facing": "toward_anchor",
      "spacing": "even_around_anchor",
      "clearance_m": 0.10
    },
    "G3": {
      "anchor_role": "tv_stand",
      "member_count": 1,
      "pattern": "cluster",
      "facing": "toward_anchor",
      "spacing": "tight",
      "clearance_m": 0.00
    }
  },
  "synthetic_groups": [
    {
      "group_id": "S1",
      "member": "obj_27",
      "anchor": "Floor",
      "reason": "obj_27 (floor lamp) renders ~0.4 m above the carpet with empty space beneath it; in the reference photo it stands on the floor.",
      "target_spec": {
        "anchor_role": "floor",
        "member_count": 1,
        "pattern": "cluster",
        "facing": "toward_anchor",
        "spacing": "tight",
        "clearance_m": 0.0
      }
    }
  ]
}
```

If no groups need island work, write `{"groups_needing_island": [], "rationale": {}, "target_spec": {}, "synthetic_groups": []}`.

If `synthetic_groups` is empty, emit `"synthetic_groups": []` explicitly (not omitted).

Confirm with: `validation: N groups + M synthetic → [G1, G3, S1, ...]`

(If M is 0, the format is still valid: `validation: N groups + 0 synthetic → [G1, G3]`)

