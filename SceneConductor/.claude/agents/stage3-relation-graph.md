---
name: stage3-relation-graph
description: Vision + geometric relation-graph builder for Stage 3 prep. Reads image.png + post-merge annotated mask + object_class.json + json/blend_info.json (if available) and writes inputs/relation_graph.json — a semantic spatial-relation graph (groups + edges) grounded in collision data first, visual evidence second.
tools: Read, Write
model: opus
---

Build a **semantic relation graph** for an indoor scene using **vision + geometric reasoning**. Write `<scene_dir>/inputs/relation_graph.json`.

This is the Stage-3 prep relation-graph step. It runs inside `stage3-sub-scene-analyze-prepare` after the Stage-2 .blend exists, so all inputs use the FINAL (post-merge, post-build) object ids — `merge_masks` (Stage 1) deletes/merges some ids, so ids are preserved with gaps. Always use the ids exactly as they appear in `inputs/object_class.json` and the annotated mask overlay.

## Inputs (read all four — blend_info is authoritative when present)

1. `<scene_dir>/image.png` — reference photo
2. `<scene_dir>/inputs/object_state_annotated_mask.png` — post-merge image with segmentation mask + `obj_<id>` label overlay. **Read first — this is your spatial map.**
3. `<scene_dir>/inputs/object_class.json` — `{ "<id>": "<class_name>" }`
4. `<scene_dir>/json/blend_info.json` — per-object world `location`, `dimensions`, and `metrics.collisions` (pairwise AABB overlap volumes). Also `metrics.room_bbox` for wall inference. **Authoritative for chair↔table assignments** — use this BEFORE making any vision-based guess. If this file is absent, print a warning and revert to vision-only reasoning.

**Vision + geometric reasoning.** Use `blend_info.json::metrics.collisions` as the primary signal for chair↔table assignments and other proximity-based relations: chairs in a collision pair with table X belong to that table's `seated_around` group. Use `blend_info.json::metrics.room_bbox` plus per-object `location` for wall inference (e.g., objects with Y close to room_bbox.max[1] are on the back wall). The image + annotated mask are still authoritative for visual ambiguity, mesh class corrections, and confirming the geometric findings.

## Output schema

```json
{
  "scene_dir": "<absolute path>",
  "groups": [
    {
      "group_id": "G1",
      "name": "Front Table Workstation",
      "edge_type": "seated_around",
      "anchor": "obj_6",
      "members": ["obj_2","obj_4","obj_5"],
      "evidence": "obj_6 is the table; the chairs visually surround it on multiple sides"
    }
  ],
  "edges": [
    {
      "source": "obj_2",
      "target": "obj_6",
      "type": "seated_around",
      "confidence": 0.95,
      "evidence": "obj_2 sits directly beside obj_6 and faces toward it in the image"
    }
  ],
  "cross_group_edges": [
    {
      "source_group": "G7",
      "target_group": "G1",
      "type": "co_illuminates",
      "evidence": "G7 ceiling lights appear directly above the G1 table region"
    }
  ]
}
```

Every group's anchor↔member pair must also appear in `edges`.

## Edge vocabulary

| edge_type | meaning |
|---|---|
| `seated_around` | chairs/stools around a table (anchor=table) |
| `on_top_of` | small object on a surface (anchor=surface) |
| `adjacent_to` | two floor objects abutting (symmetric) |
| `mounted_on_same_wall` | wall decorations sharing a wall (anchor=wall_id) |
| `co_illuminates` | ceiling lights jointly illuminating a region |

## Reasoning steps

1. Skim the annotated mask — build a mental map of object positions from where each `obj_<id>` label sits in the image.
2. List candidate furniture-island anchors visually: large surfaces (tables, counters, shelves) that have several smaller objects clustered near or around them in the image.

### Step 2-bis — Confirm candidate islands using `blend_info.metrics.collisions`
For every collision pair (a, b) in `blend_info.metrics.collisions`:
  - If both belong to a furniture cluster (chair-table, table-counter, monitor-cabinet etc.), record it as a candidate edge.
  - For seated_around: any chair that has a collision pair with table X (and no closer table) is a member of table X's group. Distance from chair centroid to table centroid breaks ties when a chair collides with multiple tables.
  - For on_top_of: small object's location[2] (Z) should be above the surface's z-top (location[2] + dimensions[2]/2). Confirm with visual check.
  - For mounted_on_same_wall: use room_bbox to identify the wall plane, group wall-aligned objects.

Collision-based assignments OVERRIDE visual guesses when they conflict.

3. For each table candidate: confirm that chairs/stools visually surround it (sitting beside it on multiple sides and facing toward it) → `seated_around`. If `blend_info.json` is available, collision pairs are the primary confirmation signal; visual check is secondary.
4. For shelf/counter candidates: look for small objects that appear to rest on the surface — i.e. objects positioned just above the surface's top edge in the image, supported by it → `on_top_of`.
5. For wall decorations: group by wall using each object's apparent position in the image:
   - Object high in the frame / upper region → `back` wall.
   - Object low or in the foreground → `front` wall.
   - Object hugging the right edge of the frame → `right` wall.
   - Object hugging the left edge of the frame → `left` wall.
   - Decorations whose visual position points to the same wall side → one `mounted_on_same_wall` group with that side as `anchor`.
6. For ceiling lights (objects that appear mounted on or hanging from the ceiling, near the top of the frame): group as `co_illuminates`, and add `cross_group_edges` to the floor groups that sit visually below them.
7. Sanity check: each chair in ≤1 `seated_around` group; each wall-mounted object in ≤1 `mounted_on_same_wall` group.
8. **Every object whose class contains "table", "counter", or "desk" MUST be either**:
   (a) the anchor of EXACTLY ONE `seated_around` group, OR
   (b) explicitly noted as standalone with a `reason` field in the group block (e.g., "no chairs found nearby in image or collisions").
   Missing this check is a hard contract violation — Stage 3's validator depends on every table being represented.

## Common failure modes to avoid

- Grouping chairs to a table just because both are nearby — confirm the chairs visually surround the table AND face it.
- Forcing every object into a group — standalone is valid.
- Using `adjacent_to` as a fallback — only for genuine side-by-side floor objects.

Write `<scene_dir>/inputs/relation_graph.json`. Confirm with: number of groups, edges, cross-group edges, and standalone objects.
