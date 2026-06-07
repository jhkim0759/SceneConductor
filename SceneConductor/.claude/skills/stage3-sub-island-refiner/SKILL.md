---
name: stage3-sub-island-refiner
description: Per-group island refiner. Dispatched once per problem group (identified by validation) via the Task tool. Each dispatch drives the FULL iteration refinement loop (default 20 iters) inside a single `stage3-island-refiner` Opus subagent for ONE relation-group island (anchor + members). The agent refines the WHOLE GROUP PATTERN per iter — multi-member translation moves for spacing, and per-member yaw targets (NEVER uniform yaw deltas) for facing direction. Triggered indirectly via stage3-scene-refinement; do not invoke directly.
---

# stage3-sub-island-refiner (island-level refiner)

## Trigger

This skill is **NOT user-facing**. It is auto-dispatched by
`stage3-scene-refinement` once per problem group after validation. Trigger token
`/island-refiner` is reserved for forced invocation; normal runs go through
`stage3-scene-refinement::_step2_island_refiner`.

## Architecture

Dispatch contract: the prompt carries ONLY call-specific parameters (group path and iter count). All behavioral directives — phase aggression policy, per-phase magnitude caps, iter_1 requirements, convergence rules — live exclusively in the `stage3-island-refiner` agent .md (system prompt), which is the single source of truth. Callers MUST NOT duplicate behavioral spec in the dispatch prompt.

```
stage3-scene-refinement (Step 2: per-island Task dispatch)
│
├── (Earlier) validation → json/island_groups.json
│       {"groups_needing_island": ["G1", "G3", ...]}
├── (Earlier) rebuild_islands → per-group relation_groups/<G>/island.blend
│       (canonical frame, anchor at origin) + metadata.json + masked.png (optional, human-debug only)
│
└── For each gid in pending:
      Task(subagent_type="stage3-island-refiner",
           prompt="Refine the island at <scene_dir>/relation_groups/<G>. "
                  "Run N iterations end-to-end.")
         │
         │  loop: N = 1..MAX_ITER
         │  ┌──────────────────────────────────────────────────────────────┐
         │  │ a. Read iter_(N-1)/render_persp.png, render_bev.png          │
         │  │ b. Read target_spec.json (first iter) + metadata.json (first iter) │
         │  │ c. Read iter_(N-1)/transforms.json (avoid repeated delta)    │
         │  │ d. DECIDE: pattern / member diagnosis → multi-member action  │
         │  │ e. Write iter_N/transforms.json (members{...}, ...)          │
         │  │ f. Bash: python iter_step.py --group-dir <DIR> --iter N      │
         │  │    (= apply_delta + render + score_info(stub) + sanity-check)│
         │  │ g. Early-stop check (N >= MIN_ITER=5)                        │
         │  └──────────────────────────────────────────────────────────────┘
         │
         └── after final iter (per group):
               iter_FINAL/island.blend ← per-group refined island
               (merge-back step picks these up)
```

Everything in the loop is **island-local**, expressed in the **canonical frame**
where the anchor sits at the origin with identity orientation. There is no
world-space delta and no cross-group reasoning inside the loop — that is the
explicit responsibility of `stage3-scene-refinement` (validation + merge-back).

## Target spec (required input)

File location per group: `<scene_dir>/relation_groups/<G>/target_spec.json`

```json
{
  "anchor_role": "long_table",
  "member_count": 5,
  "pattern": "2+2+1",
  "facing": "toward_anchor",
  "spacing": "even_along_each_edge",
  "clearance_m": 0.10,
  "free_note": "Classroom-style; chairs on long sides + 1 short-end."
}
```

### Required fields

| Field | Type | Meaning |
|---|---|---|
| `anchor_role` | string | Short label for the anchor object (e.g. `long_table`, `round_table`, `tv_stand`, `bed`, `sofa`, `shelf`). Free-form; used by the refiner for semantic reasoning. |
| `member_count` | int ≥ 1 | Expected count of refined members in this group. Must match `len(metadata.json::members) - 1` (all members minus the anchor). |
| `pattern` | enum | Spatial arrangement of members around the anchor. See enum values below. |
| `facing` | enum | Facing direction of each member. See enum values below. |
| `spacing` | enum | How members are distributed around/along the anchor. See enum values below. |
| `clearance_m` | float ≥ 0.0 | Minimum perpendicular distance in metres from the anchor surface to the nearest member surface. |

### Optional fields

| Field | Type | Meaning |
|---|---|---|
| `free_note` | string | Short prose description. REQUIRED when `pattern="free"` or `facing="mixed"`. |

### `pattern` enum values

- `"ring"` — members encircle the anchor uniformly (e.g. 6 chairs around a round table).
- `"row"` — members line up in a single straight row parallel to one anchor edge.
- `"2+2+1"` — 2 members on one long side, 2 on the other, 1 on a short end.
- `"2+2+1+1"` — 2 + 2 on long sides, 1 on each short end.
- `"L"` — members arranged in an L shape along two adjacent anchor edges.
- `"T"` — members arranged in a T shape (one edge + perpendicular centre line).
- `"cluster"` — members grouped close together without a regular geometric locus.
- `"free"` — custom arrangement; `free_note` MUST describe the geometry.

### `facing` enum values

- `"toward_anchor"` — each member's front faces the anchor (e.g. chair seat toward table).
- `"away_from_anchor"` — each member faces away from the anchor.
- `"parallel_to_anchor_long_axis"` — members face along the anchor's long axis.
- `"mixed"` — members have heterogeneous facing; `free_note` must explain.

### `spacing` enum values

- `"even_along_each_edge"` — members on each anchor edge are evenly spaced along that edge.
- `"even_around_anchor"` — members are uniformly distributed angularly around the anchor centroid.
- `"tight"` — members are placed as close to the anchor as clearance allows.
- `"loose"` — members are spread further from the anchor than the minimum clearance.

## Files in this skill

### Active files (island-level — used by the live flow)

| File | Role |
|---|---|
| `src/init_iter0.py` | iter_0 baseline. Always invoked at dispatch start; rotates any prior simple_refiner/ to a backup before creating a fresh iter_0. Copies per-group `island.blend` into `simple_refiner/iter_0/`, renders `render_persp.png` + `render_bev.png`, writes empty `transforms.json` and initial `info.json`. |
| `src/iter_step.py` | **Main loop entry point.** Runs apply_delta + render_one + score_info(stub) + compute_member_geometry in order, plus a built-in sanity check that iter_N/island.blend object positions actually changed when transforms.json had non-trivial deltas. Exit code 4 = object positions identical to previous iter despite non-trivial transforms — the loop must stop immediately on this. |
| `src/compute_member_geometry.py` | Deterministic geometry computation. Writes `<group_dir>/forward_axes.json` (cached, written only on first call — local-frame forward axis per member derived from mesh vertex distribution) and `iter_N/geometry.json` (per-iter facing alignment + clearance ground truth consumed by the agent). |
| `src/apply_delta.py` | Apply `iter_N/transforms.json::members` to the canonical-frame `island.blend` → `iter_N/island.blend`. No gates, no clamps, no anchor lock. Internal — do not call directly. |
| `src/render_one.py` | Thin wrapper that resolves `--blender-bin` and forwards to `render_island.py`. Internal — do not call directly. |
| `src/render_island.py` | Blender headless launcher. Sets up scene + cameras using `metadata.json`'s canonical-frame data, then dispatches `render_island_views.py` inside Blender. Internal. |
| `src/render_island_views.py` | In-Blender script. Produces `render_persp.png` and `render_bev.png` for the island in its canonical frame, using `metadata.json::M_inv_4x4` with a bbox-based fallback. Internal. |
| `src/score_info.py` | Writes minimal `info.json` (`iter` + `notes`). Scoring removed; INFO ONLY stub kept for backwards compatibility. |

### Archived scene-level files

The scene-level variant (`init_scene_iter0.py`, `scene_apply_delta.py`,
`render_scene_views.py`, `scene_evaluate.py`, plus
`.claude/agents/stage3-scene-refiner.md`) was archived under
`arxiv/_stage3-scene-level-refiner/` and is no longer on the active path.
See `arxiv/_stage3-scene-level-refiner/README.md` for the restore checklist.

## CLI usage (active scripts)

All scripts live under `.claude/skills/stage3-sub-island-refiner/src/`. They are
invoked by the sub-agent via Bash, never by the user directly.

```bash
# iter_0 baseline. Always invoked at dispatch start; rotates any prior simple_refiner/
# to a backup before creating a fresh iter_0.
python init_iter0.py --group-dir <SCENE_DIR>/relation_groups/<G> \
    [--samples 128] [--blender-bin PATH]

# Main per-iter entry point: apply_delta + render + score_info(stub) + sanity check.
# ONLY Bash command the agent runs inside the iter loop body after writing transforms.json.
python iter_step.py --group-dir <SCENE_DIR>/relation_groups/<G> \
    --iter N [--samples 64] [--blender-bin PATH]
```

`apply_delta.py`, `render_one.py`, `score_info.py`, `render_island.py`,
`render_island_views.py` are internal — the agent must **not** invoke them directly.
Always go through `iter_step.py` for the per-iter step.

`<SCENE_DIR>` and `<G>` are always absolute paths / explicit group ids.
`--blender-bin` falls back to `DIRECTORYS.yaml::blender_bin` then `$BLENDER`
then `blender` on PATH.

## Exit codes

| Code | Meaning | Agent action |
|---|---|---|
| 0 | All steps succeeded, sanity OK | Continue loop |
| 4 | `iter_N/island.blend` object positions identical to `iter_(N-1)` despite non-trivial deltas in transforms.json | **STOP the loop immediately** — apply failed silently |
| 5 | `target_spec.json` missing or schema-invalid | Agent aborts; user must produce/fix `target_spec.json` before re-dispatch. |
| 6 | `compute_member_geometry.py` failed | Agent stops; check inner Blender script error. |
| other non-zero | Unexpected error in apply / render / score | STOP and report |

## `transforms.json` schema (island-level, per-member only)

Written by the sub-agent at
`relation_groups/<G>/simple_refiner/iter_N/transforms.json`.

```json
{
  "iter": N,
  "members": {
    "obj_3": {"delta_xyz": [-0.30, 0.00, 0.00], "delta_yaw_deg":  30.0},
    "obj_4": {"delta_xyz": [+0.30, 0.00, 0.00], "delta_yaw_deg": -30.0},
    "obj_5": {"delta_xyz": [-0.30, 0.30, 0.00], "delta_yaw_deg":  30.0},
    "obj_6": {"delta_xyz": [+0.30, 0.30, 0.00], "delta_yaw_deg": -30.0},
    "obj_7": {"delta_xyz": [-0.30,-0.30, 0.00], "delta_yaw_deg":  30.0},
    "obj_8": {"delta_xyz": [+0.30,-0.30, 0.00], "delta_yaw_deg": -30.0}
  },
  "converged": false,
  "final": false,
  "reason_summary": "one sentence describing the multi-member action",
  "_island_meta": {
    "model": "opus",
    "image_sha256": "<sha256sum masked.png>",
    "generated_by": "stage3-island-refiner subagent",
    "mode": "island_iter",
    "iter": N,
    "timestamp_utc": "<UTC ISO 8601>"
  }
}
```

### Field semantics

| Field | Type | Required | Meaning |
|---|---|---|---|
| `iter` | int ≥ 0 | Yes | Must equal the directory's `N`. |
| `members` | object | Yes | Map `object_name → delta`. **The only allowed key for per-iter moves.** |
| `members[k].delta_xyz` | [float, float, float] | Yes | Canonical-frame additive translation in **metres**. Cap ±1.0 m per member per iter (penetration exception aside). |
| `members[k].delta_yaw_deg` | float | Yes | Additive yaw rotation about canonical Z, in **degrees**. No cap. |
| `converged` | bool | No | When `true` AND `N >= MIN_ITER=5` AND at least one prior iter K in [1, N-1] wrote a non-empty `members` dict, loop exits after this iter. Setting `converged: true` before MIN_ITER is reached, or when no real per-member delta was ever attempted, is a **contract violation** — the orchestrator's completion check will treat the group as failed. |
| `final` | bool | No | Set to `true` on the LAST iter the agent produces. Required before the agent finishes. |
| `reason_summary` | string | Recommended | One short sentence describing the multi-member intent. |
| `_island_meta` | object | Yes | Audit block. All six sub-fields required (see below). |
| `_island_meta.model` | string | Yes | Always `"opus"`. |
| `_island_meta.image_sha256` | string | Yes | `sha256sum masked.png`. Audit field only — not used for cross-dispatch resume (no-resume policy). |
| `_island_meta.generated_by` | string | Yes | Always `"stage3-island-refiner subagent"`. |
| `_island_meta.mode` | string | Yes | Always `"island_iter"`. |
| `_island_meta.iter` | int | Yes | Same as top-level `iter`. |
| `_island_meta.timestamp_utc` | string | Yes | `date -u +%Y-%m-%dT%H:%M:%SZ`. |

### Hard constraints on transforms.json

- `members` is the **only** allowed delta map. `groups`, `objects`, `deletes`
  fields are forbidden — those belong to the inactive scene-level variant.
- The anchor (`metadata.json::anchor_id`) **MUST NOT** appear in `members`.
  The anchor stays at the canonical-frame origin throughout the loop.
- `delta_xyz` is canonical-frame additive translation (same frame as
  `metadata.json::canonical_poses`). No world-space deltas.
- `delta_yaw_deg` adds to canonical yaw about the canonical Z axis.
- **`phase` field is obsolete** — the harness does not read it. Do not emit it.
- **iter_1 must be non-empty.** An empty `members` dict in `iter_1/transforms.json`
  is a contract violation. The agent MUST attempt at least one per-member translation
  OR yaw delta in the very first refinement pass. If the layout already looks correct
  after viewing iter_0, the agent must still commit at least 2 explicit "verification"
  deltas in iter_1 (e.g. tiny ±0.01 m translations that scoring can reject if
  unhelpful) — this proves the loop actually ran and allows the orchestrator's
  completion check to pass.
- **Every iter must be stepped.** The agent MUST invoke
  `iter_step.py --group-dir <DIR> --iter N` for every N in 1..N_committed.
  Skipping intermediate iters by writing only a terminal `transforms.json` is
  forbidden — `iter_step.py` produces the `island.blend`, renders, and scores for
  that iter; without it, the blend files will be byte-identical to iter_0 and the
  orchestrator will classify the group as failed (no-op refiner).
- **Convergence gate (enforced by orchestrator):** `converged: true` is only a valid
  signal when BOTH of the following hold:
  1. `N >= MIN_ITER` (where **MIN_ITER = 5**, distinct from MAX_ITER which defaults
     to 20 and is caller-specified).
  2. At least one prior iter K in [1, N-1] wrote a non-empty `members` dict (i.e.
     a real per-member delta was attempted, not just empty `members: {}`).
  If the orchestrator detects that `iter_0/island.blend` and
  `iter_<MAX_ITER>/island.blend` are byte-identical, it moves the group to
  `islands_failed` rather than `islands_completed`.

## `info.json` schema (INFO ONLY)

Written by `score_info.py` at
`relation_groups/<G>/simple_refiner/iter_N/info.json`.

```json
{
  "iter": N,
  "notes": "INFO ONLY — score-based gating has been removed from the island-refiner."
}
```

| Field | Meaning |
|---|---|
| `iter` | Iteration number inferred from the parent directory name (`iter_N`). |
| `notes` | Fixed string confirming scoring has been removed. |

Scoring has been removed entirely. The sub-agent does not read `info.json` for
decisions — visual judgement via `render_persp.png` and `render_bev.png` is the
sole signal.

## Magnitude policy

| Phase | Iter range | Scope (members per iter) | Max `delta_xyz` (m) | Max `delta_yaw_deg` (°) |
|---|---|---|---|---|
| `pattern` | 1 – 3   | **multi-member, N ≥ 3** | ±1.0 | uncapped |
| `member`  | 4 – 10  | 1 – 3 members           | ±0.3 | uncapped |
| `micro`   | 11 +    | 1 member only           | ±0.05 | uncapped |
| exception | any     | 1 member (penetration)  | ±1.0+ | uncapped |

Yaw is **always uncapped** — rotate freely whenever a member faces the wrong
direction. The `exception` row applies only to gross single-member penetration
(e.g. one chair clearly inside the table). See agent md for judgment on when to
invoke the exception.

## Loop pseudocode

**MIN_ITER = 5** (minimum iters before early-stop allowed).
**MAX_ITER** = caller-specified (default 20).

```
Step 1. Bash: python <skill>/src/init_iter0.py --group-dir <group_dir>

Step 2. consecutive_empty = 0

Step 3. For N in 1..MAX_ITER:
   a. Read <group_dir>/simple_refiner/iter_(N-1)/render_persp.png
   b. Read <group_dir>/simple_refiner/iter_(N-1)/render_bev.png
   c. Read <group_dir>/target_spec.json  (only first time; cache in context)
   d. Read <group_dir>/simple_refiner/iter_(N-1)/transforms.json
   d-bis. Read <group_dir>/simple_refiner/iter_(N-1)/geometry.json  (deterministic ground truth for facing + clearance — see compute_member_geometry.py)
   e. Read <group_dir>/metadata.json  (only first time; cache in context)
   f. DECIDE (see agent md for judgment protocol):
      i.   Pattern diagnosis FIRST
      ii.  Member diagnosis SECOND
      iii. Plan ONE coordinated multi-member action (≥3 members)
      iv.  OR decide "converged" → set converged:true, empty members
   g. Bash: date -u +%Y-%m-%dT%H:%M:%SZ             (capture timestamp)
   h. Write <group_dir>/simple_refiner/iter_N/transforms.json
   i. Bash: python <skill>/src/iter_step.py --group-dir <group_dir> --iter N --samples 64
      ── exits 4 → STOP immediately (apply failed)
      ── other non-zero → STOP and report
      ── zero → confirm "[iter_step] sanity OK" line in output
   j. EARLY STOP CHECK (only if N >= MIN_ITER=5):
      - If "converged": true in step (h):
          * ALSO verify that at least one prior iter K in [1, N-1] had
            non-empty members. If no such iter exists, do NOT stop —
            treat this iter as non-converged and continue.
          * Only STOP when both conditions are met (N >= 5 AND prior non-empty exists).
      - members at step (h) was empty → consecutive_empty += 1; else reset to 0
      - If consecutive_empty >= 3 → STOP (only valid if N >= MIN_ITER=5)
      Otherwise continue to N+1.

Step 4. FINAL = last N that ran.
        Re-write <group_dir>/simple_refiner/iter_{FINAL}/transforms.json
        with "final": true  (preserve existing members, converged, _island_meta).

Step 5. Promote the FINAL iter.
        The iter the agent marks `"final": true` is the canonical result of this
        run — promote its island.blend regardless of any score.
        Bash: cp <group_dir>/simple_refiner/iter_{FINAL}/island.blend \
                 <group_dir>/island.blend
        (No score-based comparison, no best-of selection.)

Step 6. Report ≤80 words: final iter, stop reason, iters with non-empty members,
        one sentence on subjective quality vs target_spec.
```

Note: even with empty `members`, always run `iter_step.py` so the next iter
has fresh inputs. The wrapper detects "trivial transforms" and skips the sanity
check, but still runs apply_delta + render + score. The render is cheap (~7 s).

## No-resume policy

Every dispatch starts from a fresh `simple_refiner/`. There is NO cross-dispatch
cache for transforms.json / renders / sha256.

- `init_iter0.py` rotates any pre-existing `simple_refiner/` to
  `simple_refiner.bak.<UTC_TIMESTAMP>/` (format `YYYYMMDD_HHMMSS`) before
  creating a fresh `iter_0`. All prior backup directories are preserved untouched.
- There is NO cross-dispatch cache for transforms.json / renders / sha256.
- A re-dispatch of the same group produces a full new run — identical quality to
  a cold start. This trades compute (always re-reason) for quality determinism
  (re-dispatch ≡ cold start).

The in-loop termination conditions remain unchanged:
- A completed iter has `"final": true`, **or**
- `iter == MAX_ITER` has been reached, **or**
- The early-stop conditions in Step 3k fired (converged + MIN_ITER gate, or 3
  consecutive empty-members iters, or exit code 4 from iter_step.py).

`stage3-scene-refinement::orchestrate.py` adds the group id to
`state["islands_completed"]` once its loop terminates, so re-invocation after a
crash is safe — completed groups are skipped by the orchestrator (not by the
island-refiner itself).

## Output directory layout

```
<scene_dir>/
  relation_groups/
    G1/
      island.blend                  # canonical frame, anchor at origin
      metadata.json                 # anchor_id, M_anchor_4x4, members, canonical_poses
      target_spec.json              # REQUIRED — see schema above
      masked.png                    # optional, human-debug only — no longer read by the refiner
      forward_axes.json             # NEW — cached local-frame forward axis per member (written once by compute_member_geometry.py at iter_0)
      simple_refiner/               # FIXED name — downstream tools depend on it
        iter_0/
          island.blend
          render_persp.png
          render_bev.png
          info.json
          transforms.json           # empty members (baseline)
          current_state.json        # absolute poses after iter_0
          geometry.json             # NEW — deterministic forward-axis + clearance ground truth; consumed by the agent
        iter_1/
          island.blend
          render_persp.png
          render_bev.png
          info.json
          transforms.json           # multi-member pattern delta
          current_state.json
          geometry.json             # NEW — deterministic forward-axis + clearance ground truth; consumed by the agent
        ...
        iter_FINAL/
          ...                       # transforms.json has "final": true
    G3/
      ...
  blend/
    stage3-sub-planned.blend        # input (unchanged by this skill)
    stage3-scene.blend              # final output (produced by merge_islands_back)
```

The `simple_refiner/` directory name is a fixed convention shared with
`stage3-scene-refinement`; do not rename it.

## Compatibility notes

- `relation_groups/<G>/island.blend`, `metadata.json`, and `masked.png` (optional, human-debug only)
  are produced by `stage3-scene-refinement`'s rebuild-islands step and consumed
  directly by this skill. Do not regenerate them inside the loop.
- `target_spec.json` is produced by stage3-scene-refinement's validation step (or hand-authored for tests). The refiner FAILS FAST if it is missing — there is no heuristic fallback.
- `inputs/relation_graph.json` is not consumed inside this skill — it is only
  used by `stage3-scene-refinement` to decide which groups need islands.
- `json/island_groups.json` (validation output) controls which groups receive a
  dispatch but is not read inside the loop.
- The scene-level variant has been archived under
  `arxiv/_stage3-scene-level-refiner/` and is no longer on the
  active path. See that folder's `README.md` for restoration steps.
