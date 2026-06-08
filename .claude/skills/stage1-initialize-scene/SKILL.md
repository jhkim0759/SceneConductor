---
name: stage1-initialize-scene
description: Initialize a 3D scene from a single indoor `image.png` — produces segmentation masks, deduped textured GLBs, and GALP outputs (layout_prediction.json + .glb) ready for Blender import. Accepts scene-dir (required), --gpu N (default 0), and --force (delete prior outputs + re-run everything). Trigger on "/stage1-initialize-scene".
argument-hint: <scene-dir> [--gpu N] [--force]
allowed-tools: [Bash, Read, Write, Edit, Task, Agent, Glob, Grep]
---
> For the authoritative file-layout contract see `FILE_DIRECTORY.md` at the SceneConductor root.

# stage1-initialize-scene

End-to-end initialization pipeline that turns one indoor photograph into a structured 3D scene file tree: per-object masks, textured GLB meshes (with same-object dedup), and a GALP prediction ready for Blender. This is the first step in the SceneConductor flow — it produces the inputs every downstream stage assumes already exist.

## Quick reference (arg guide)

When the user types `/stage1-initialize-scene ...`, these are the only args this skill consumes.

```
/stage1-initialize-scene <scene-dir> [--gpu N] [--force]
```

| Arg | Required? | Default | Range | Forwarded to |
|---|---|---|---|---|
| `<scene-dir>` (positional, or `--scene_dir <dir>`) | **yes** | — | absolute path containing `image.png` | `run_stage1.sh --scene_dir` for all three phases |
| `--gpu N` | no | `0` | non-negative int | `run_stage1.sh --gpu` for all three phases (GroundedSAM, SAM3D, GALP, Qwen-VL) |
| `--force` | no | off | flag (no value) | `run_stage1.sh --force` for all three phases — deletes prior outputs and bypasses the eval cache |

**What `--force` deletes (per phase, before running):**

| Phase | Deletes |
|---|---|
| `pre` | `object_class_prompt.json`, `object_class.json`, `mask_attribute.json`, `overlap_pairs.json`, `small_mask_candidates.json`, `object_state_annotated_mask.png`, `masks/` |
| `eval` | `merge_plan.json`, `remask_plan.json` (bypasses the `_evaluator_meta` cache hit — a fresh Opus API call always fires) |
| `post` | `inputs/`, `thumbnails/`, `object_state.json`, `verification_overlay.png`, `layout_prediction.json`, `layout-prediction.glb`, `pointmap_xz.ply`, `floor.obj` |

`image.png` and `logs/` are never touched. Without `--force`, the eval phase reuses a cached `merge_plan.json` when image/prompt/script/schema all match (typical re-run skips the Opus call entirely).

### Usage examples

```
/stage1-initialize-scene /data/scenes/room_01
/stage1-initialize-scene /data/scenes/room_01 --gpu 3
/stage1-initialize-scene /data/scenes/room_01 --force
/stage1-initialize-scene /data/scenes/room_01 --gpu 5 --force
```

### Argument parsing rules

- `<scene-dir>` must be an existing directory containing `image.png`; otherwise print `ERROR: <scene-dir>/image.png not found` and stop.
- `--gpu N` must be a non-negative integer; if not → print `ERROR: --gpu must be a non-negative integer` and stop.
- If `--gpu` is omitted, default to `0`. Pick a different index if GPU 0 has < 30 GiB free (`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader`).
- `--force` takes no value. It is forwarded to all three bash phases.
- Any other arg is **not supported** — reject with `ERROR: unknown arg <arg>` rather than silently ignoring.

## When to trigger

Invoke whenever the user provides a scene directory (or single image) and wants 3D objects + camera transforms extracted. Common phrasings:

- "Initialize this scene"
- "Run `stage1-initialize-scene` on `/path/to/scene/`"
- "Process `/path/to/scene/` end-to-end"
- "I have `image.png` — generate masks, SAM3D meshes, and GALP"
- "Run the full pipeline: GroundedSAM → SAM3D → GALP"

**Scope boundary:** This skill stops at `layout_prediction.json` + `layout-prediction.glb` (moved under `inputs/` by the finalize step). For downstream Blender conversion, `.blend` construction, stage-fitting, or Stage-2 corrections, hand off to the appropriate skill afterwards (`stage2-sub-pointmap-to-separable-stage` for both the trimesh→Blender JSON conversion and the .blend build, `scene-fit-stage`, etc.).

## Inputs required

Only `<scene_dir>/image.png` is required. Everything else is produced by the pipeline.

## How to run — delegation

When the user triggers `/stage1-initialize-scene`, the cheapest path is to **delegate the whole flow to the matching orchestrator agent**. Forward the parsed `<scene-dir>`, `--gpu N` (default `0`), and `--force` (default off) into the agent prompt — the agent expects all three:

```python
# gpu defaults to 0; force_arg is "--force" when the user passed --force, else "" (empty).
Agent(
    description="Stage 1 init",
    subagent_type="stage1-initialize-scene",
    prompt=f"Run Stage 1 on {scene_dir} with --gpu {gpu} {force_arg}".strip(),
    run_in_background=True,   # optional — long-running
)
```

The `stage1-initialize-scene` agent (Haiku, `tools: Read, Write, Glob, Bash`) drives the 3-step flow below. The Opus vision call is embedded in `--phase eval`; no agent is spawned for mask evaluation. The main conversation only sees the agent's final report — Bash logs stay in the agent's throwaway context.

> **NOTE:** Do not spawn `stage1-mask-evaluator` from the orchestrator — it is invoked exclusively by `--phase eval` (via `run_mask_evaluator.py`).

The remainder of this document is the **agent's contract** (the actions, scripts, and verification steps). Read it if you need to debug, customize, or run the pipeline manually without going through the orchestrator agent.

## Manual orchestration (3 actions)

All Stage-1 steps are deterministic Bash phases. The single vision judgment (mask evaluation) is embedded inside `--phase eval` as a `claude` CLI subprocess — no agent spawn is needed.

> **Removed in this revision:** the former `Action 4 — Relation graph (vision, agent)` step is gone. `inputs/relation_graph.json` is now built later by the `stage3-relation-graph` agent inside `/scene-analyze-prepare` (Stage 3 prep), which has `json/blend_info.json` available and can ground its inferences in real collision data.

### Action 1 — Pre-evaluator phase (deterministic)

```bash
mkdir -p "<scene_dir>/logs" && \
bash ./.claude/skills/stage1-initialize-scene/src/run_stage1.sh \
    --scene_dir <scene_dir> --phase pre --gpu 0 \
    > "<scene_dir>/logs/stage1_pre.log" 2>&1
```

Runs **Step 1** object-class prompt → **Step 2** GroundedSAM → **Step 3** init mask
attributes. On success `<scene_dir>` contains `object_class_prompt.json`, `masks/`,
`object_class.json`, `mask_attribute.json`. Typical runtime 5–13 min (GroundedSAM dominates).

**Verification (do NOT read the log on success).** stdout/stderr are redirected to
`logs/stage1_pre.log` so the agent does not ingest ~4–17K of GroundedSAM progress logs.
After Bash returns exit code 0, confirm by reading small artifacts only:

- `Read("<scene_dir>/object_class.json")` — should contain class entries `{"1": "...", "2": "...", ...}`
- `Glob("<scene_dir>/masks/*.png")` — mask count should be ≥ number of object_class entries
- `Glob("<scene_dir>/object_state_annotated_mask.png")` — confirm presence (no need to read PNG bytes)

Only `Read("<scene_dir>/logs/stage1_pre.log")` if Bash exited non-zero or a verification
check failed. The failing step is named in the `[run_stage1] ...` line.

### Action 2 — Phase eval (vision API call — embedded in shell script)

```bash
mkdir -p "<scene_dir>/logs" && \
bash ./.claude/skills/stage1-initialize-scene/src/run_stage1.sh \
    --scene_dir <scene_dir> --phase eval --gpu 0 \
    > "<scene_dir>/logs/stage1_eval.log" 2>&1
```

The Opus vision API call is embedded in this phase via `run_mask_evaluator.py` — **no
agent spawn is required or allowed**. The script reads the `stage1-mask-evaluator.md`
agent spec at runtime, calls the Claude CLI with `--model opus`,
`--tools Read,Write`, `--permission-mode bypassPermissions`, and a 600 s timeout. After
the call it:

1. Parses `merge_plan.json` — fails loudly if absent or invalid JSON.
2. Asserts required top-level keys (`merge_groups`, `mesh_groups`) are present.
3. Injects `_evaluator_meta` (`model`, `image_sha256`, `generated_by`, `timestamp_utc`)
   and rewrites the file.

**Idempotent:** if `merge_plan.json` already contains a valid `_evaluator_meta` with an
`image_sha256` matching the current `image.png`, the phase prints "eval cached, skipping
re-call" and exits 0 without making another API call.

**Verification.** After Bash returns exit code 0:
- `Read("<scene_dir>/merge_plan.json")` — should parse and contain `merge_groups`,
  `mesh_groups`, and `_evaluator_meta` with all four fields.

### Action 3 — Post-evaluator phase (deterministic)

```bash
mkdir -p "<scene_dir>/logs" && \
bash ./.claude/skills/stage1-initialize-scene/src/run_stage1.sh \
    --scene_dir <scene_dir> --phase post --gpu 0 \
    > "<scene_dir>/logs/stage1_post.log" 2>&1
```

Runs **Step 4-apply** (`merge_masks.py` + conditional `remask_region.py`) → **Step 5**
SAM3D (textured GLB, dedup-aware) → **Step 6** GALP → **Step 7** finalize_layout
(moves all outputs under `inputs/`). Typical runtime 12–20 min (SAM3D dominates).

**Verification (do NOT read the log on success).** stdout/stderr are redirected to
`logs/stage1_post.log` (SAM3D + GALP output is the biggest stdout offender — easily
10–30 KB per run). After Bash returns exit code 0, confirm by reading small artifacts only:

- `Glob("<scene_dir>/inputs/object/*.glb")` — count should match `inputs/object_class.json` entries (deduped instances may share byte-identical files)
- `Read("<scene_dir>/inputs/layout_prediction.json")` — small JSON, should parse and contain `camera` + per-object transforms
- `Glob("<scene_dir>/inputs/thumbnails/obj_*.png")` — count should match the post-merge object count
- `Read("<scene_dir>/inputs/object_state.json")` — should parse and contain top-level keys like `objects`, `alignment_groups`, `stacking`

Only `Read("<scene_dir>/logs/stage1_post.log")` if Bash exited non-zero or a verification
check failed. The failing step is named in the `[run_stage1] ...` line; then consult
**Failure modes** below. Do not retry blindly.

### What the driver runs internally

| Phase | Step | Script | conda env |
|---|---|---|---|
| pre  | 1 Object-class prompt   | `generate_object_classes.py` | sceneconductor (stdlib wrapper → Claude CLI) |
| pre  | 2 GroundedSAM           | `run_grounded_sam.py`        | sceneconductor wrapper → `grounded-sam` inference |
| pre  | 3 Init mask attributes  | `mask_attribute.init_attributes` | `sceneconductor` |
| pre  | 3.5 Annotated mask (pre-merge) | `make_annotated_mask.py` | `sceneconductor` |
| eval | — Opus vision API call  | `run_mask_evaluator.py`      | sceneconductor (subprocess → Claude CLI) |
| post | 4a Apply merge plan     | `merge_masks.py`             | `sceneconductor` |
| post | 4b Remask (conditional) | `remask_region.py`           | `grounded-sam` |
| post | 4c Annotated mask (post-merge) | `make_annotated_mask.py` | `sceneconductor` |
| post | 5 SAM3D textured GLB    | `run_sam3d.py`               | `sam3d-objects` |
| post | 6 GALP            | `run_galp.py`          | `sceneconductor` |
| post | 6.5 Thumbnails          | `make_thumbnails.py`         | `sceneconductor` |
| post | 6.6 Object state (Qwen-VL) | `extract_object_state.py` | `sceneconductor` |
| post | 7 Finalize layout       | `finalize_layout.py`         | sceneconductor (stdlib) |

Default GPU is **0**. If GPU 0 is busy, pass `--gpu <n>` to *both*
phases — pick one with > 30 GiB free via
`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader`.

## Final scene contract

When complete, the scene folder MUST look like this:

```
<scene_dir>/
├── image.png                      # input (required) — stays at top level
├── logs/                          # stdout/stderr from run_stage1.sh — read only on failure
│   ├── stage1_pre.log
│   └── stage1_post.log
└── inputs/                        # all Stage-1 outputs (moved by Step 7)
    ├── object_class.json              # {"1": "chair", "2": "sofa", ...}
    ├── mask_attribute.json            # bbox/area metadata + mesh_groups + history
    ├── layout_prediction.json         # camera + per-object transforms (trimesh frame)
    ├── layout-prediction.glb          # bundled scene (floor + all objects)
    ├── pointmap_xz.ply                # point cloud in XZ plane (from pointmap step)
    ├── floor.obj                      # floor mesh (from pointmap step)
    ├── masks/
    │   ├── mask.png                   # integer label map (0=bg, 1..M=objects)
    │   └── 1.png .. M.png             # binary per-object masks (1-indexed)
    ├── object/
    │   └── 1.glb .. M.glb             # per-object textured GLB (1024² baseColor, 1-indexed)
    ├── object_state_annotated_mask.png # canonical mask overlay — used by evaluator + Stage 3 planners
    ├── thumbnails/
    │   └── obj_<id>_<class>.png        # per-object crops
    ├── object_class_prompt.json       # Claude VLM output (raw prompt + class list)
    ├── object_state.json              # Qwen-VL per-object state — used by Stage 3 refinement
    ├── verification_overlay.png       # Step 4 overlay for sandwich check
    ├── merge_plan.json                # written by Mask-Evaluator
    └── remask_plan.json               # only if evaluator flagged missing objects
```

> `inputs/relation_graph.json` is **not** produced by Stage 1. It is built later by the `stage3-relation-graph` agent inside `/scene-analyze-prepare` (Stage 3 prep), where `json/blend_info.json` is available to ground the relation inference in real AABB / collision data.

1-indexed throughout. GLB indexes match mask indexes. Dedup instances (same mesh applied
at multiple positions) share byte-identical GLB files with the same sha256.

## Dedup verification (sanity check)

After the pipeline completes, verify dedup fired correctly:

```bash
jq '.mesh_groups' <scene_dir>/inputs/mask_attribute.json
sha256sum <scene_dir>/inputs/object/*.glb | sort | uniq -c -w 64
```

Groups with `instance_ids` of length > 1 should show matching hashes.

## Failure modes and recovery

| Symptom | Cause | Fix |
|---|---|---|
| GroundedSAM produces too many masks (> 40) | Prompt had too many vague classes | Re-run phase `pre`, or trust Mask-Evaluator to merge |
| SAM3D fails with `nvdiffrast ... too old GCC` | System GCC < 9 picked up | Already handled — `run_sam3d.py` forces conda gcc-12.4. If it still fires, delete `~/.cache/torch_extensions/py311_cu124/nvdiffrast_plugin` and retry phase `post` |
| SAM3D CUDA OOM | GPU 0 is full | Re-run phase `post` with `--gpu <n>` (> 30 GiB free) |
| GALP: `NP_SUPPORTED_MODULES` ImportError | (legacy env issue) | Driver already uses `sceneconductor` — should not occur. |
| GALP: `expected scalar type Half but found Float` | Old fp16 bug | Already patched — `run_galp.py` has the `.float()` cast around line 336 |
| Empty `object/` after SAM3D | SAM3D crashed mid-run | Check `logs/13*.log`; usually OOM or checkpoint-load failure. Delete `object/*.glb`, re-run phase `post` |
| `MISSING expected output: .../merge_plan.json` on phase `post` | Phase eval was skipped or failed | Run `--phase eval` before `--phase post`; check `stage1_eval.log` |

## Files produced vs reused across runs

- `inputs/masks/`, `inputs/object/`, `inputs/layout_*` are regenerated per run
- `inputs/mask_attribute.json` accumulates history entries — safe to re-run; `init_attributes` is idempotent
- `inputs/object_class_prompt.json` only regenerates when you re-run phase `pre`
- `inputs/merge_plan.json` + `inputs/remask_plan.json` are overwritten each Mask-Evaluator pass

## Don'ts

- Don't run phase `post` before phase `eval` — the driver will abort on the missing or unsigned `merge_plan.json`
- Don't apply class-based dedup in code. Dedup decisions come **only** from the Mask-Evaluator agent's visual judgment, written to `mesh_groups`
- Don't modify `.claude/skills/stage1-initialize-scene/src/` from inside this skill — the scripts are shared across workflows
- Don't delete `mask_attribute.json` between phases — it carries state (`mesh_groups`) that Step 5 needs for dedup

## Bundled agent

This skill spawns **no** agents directly. Stage 1 is now fully Bash-driven.

`stage1-mask-evaluator.md` is the spec consumed by `run_mask_evaluator.py` at runtime (loaded as the system prompt for the Claude CLI call in `--phase eval`). It is **not** spawned as an agent by the orchestrator — the script invokes the Claude CLI directly.

> Architecture note: the former `stage1_lead` (orchestrator), `stage1_executor`
> (subprocess runner), and `stage1-relation-graph` agents were retired — the skill
> orchestrates directly and `run_stage1.sh` runs the deterministic steps. The relation
> graph is now produced by `stage3-relation-graph` in `/scene-analyze-prepare`
> (Stage 3 prep). Old specs are kept under `.claude/agents/_archeive/` for reference
> when applicable.
