---
name: stage3-scene-refinement
description: Run the full Stage 3 scene-refinement pipeline end-to-end for a scene_dir. Drives orchestrate.py with --resume in a loop, dispatching sub-skills (stage3-sub-scene-analyze-prepare, stage3-sub-scene-refiner) and per-group stage3-island-refiner Tasks (in parallel) as orchestrate.py requests. Use when the user wants Stage 3 from blender_scene.blend to blend/stage3-scene.blend + render/final/*.png.
tools: Read, Write, Glob, Bash, Skill, Agent
model: sonnet
background: true
skills:
  - stage3-scene-refinement
  - stage3-sub-scene-analyze-prepare
  - stage3-sub-scene-refiner
---

Run the Stage 3 pipeline for the provided `<scene_dir>` to completion. You are the orchestrator; `orchestrate.py` is the state machine — your job is to (a) launch it, (b) react to its `awaiting_agent` dispatch messages, and (c) re-launch it with `--resume` until it reports `step=done`.

# 0. Parse user arguments

Inputs the user may supply: `scene_dir` (required, absolute), `skip_prepare` (bool, default False), `force` (bool, default False), `num_max_iter` (int, default 20).

Map to CLI flags on `python .claude/skills/stage3-scene-refinement/src/orchestrate.py <scene_dir>`:

| Arg | Flag emitted |
|---|---|
| `skip_prepare=True` | `--skip-prepare` |
| `force=True` | `--force` |
| `num_max_iter=N` (N != 20) | `--num-max-iter N` |

Echo the resolved invocation in your first status message.

If `<scene_dir>` is missing or `<scene_dir>/image.png` does not exist, stop and report the blocker.

# 1. The driver loop

Run this loop until `orchestrate.py` exits with `step=done`:

1. **Bash launch.** First iteration: `python .claude/skills/stage3-scene-refinement/src/orchestrate.py <scene_dir> [resolved flags]`. Subsequent iterations: same command with `--resume` appended (do NOT re-emit `--force` on resume — it would wipe progress).
2. **Read the tail** of stdout (the last ~40 lines is enough). Look for these signals in order:
   - `step=done` → exit the loop. Go to section 3 (final report).
   - `step_status=ok` (no `awaiting_agent`) → orchestrate.py advanced one or more deterministic steps and returned cleanly. Loop again with `--resume`.
   - `step_status=awaiting_agent` followed by an `[orchestrate:dispatch] ...` line → handle the dispatch per section 2, then loop again with `--resume`.
   - Any non-zero exit or `step_status=failed` → stop. Report the failing step, the dispatch tail, and the path to `<scene_dir>/json/stage3_state.json`. Do NOT retry blindly.
3. After handling a dispatch, ALWAYS re-run `orchestrate.py ... --resume` to let the state machine advance.

**Hard cap:** if the loop runs more than 30 iterations without reaching `step=done`, stop and report progress — there is a stuck step.

# 2. Dispatch handlers

Inspect the `[orchestrate:dispatch] ...` line emitted just before the `awaiting_agent` exit. There are three dispatch shapes:

## 2a. `/stage3-sub-scene-analyze-prepare <scene_dir>` (Step 0)

Invoke the sub-skill via the `Skill` tool:

```
Skill(skill="stage3-sub-scene-analyze-prepare", args="<scene_dir>")
```

When the skill returns, verify all three files exist before re-running orchestrate.py:
- `<scene_dir>/json/object_state.json`
- `<scene_dir>/json/blend_info.json`
- `<scene_dir>/inputs/relation_graph.json`

If any are missing, stop and report.

## 2b. `/stage3-sub-scene-refiner <scene_dir>` (Step 1)

Invoke the sub-skill via the `Skill` tool:

```
Skill(skill="stage3-sub-scene-refiner", args="<scene_dir>")
```

When it returns, verify the working blend + plans exist:
- `<scene_dir>/blend/stage3-sub-planned.blend`
- `<scene_dir>/json/operation_plan.json`

If missing, stop and report.

## 2c. `stage3-island-refiner — Task per pending group (K)` (Step 2)

The dispatch message lists K pending `group_id`s with one suggested `Task(subagent_type="stage3-island-refiner", prompt="Refine the island at <scene_dir>/relation_groups/<G>. Run N iterations end-to-end.")` block per group. The user has chosen **parallel dispatch** — emit ALL K Agent calls in a single message so they run concurrently:

```python
Agent(subagent_type="stage3-island-refiner",
      prompt="Refine the island at <scene_dir>/relation_groups/<G1>. Run <N> iterations end-to-end.",
      description="Island refine <G1>")
Agent(subagent_type="stage3-island-refiner",
      prompt="Refine the island at <scene_dir>/relation_groups/<G2>. Run <N> iterations end-to-end.",
      description="Island refine <G2>")
... (one per pending group)
```

Use `N = num_max_iter` (defaults to 20). Wait for ALL agents to complete before re-running orchestrate.py with `--resume`.

After they return, orchestrate.py on `--resume` will inspect `relation_groups/<G>/simple_refiner/iter_<N>/transforms.json::_island_meta` for each group. Completed groups short-circuit; any group still pending re-dispatches (loop handles that automatically).

# 3. Final report (after `step=done`)

Verify the final outputs exist, then report:

- `scene_dir`
- `blend/stage3-scene.blend` (final Stage-3 blend)
- `render/final/blender_scene_view_*.png` (count of PNGs; expect 5)
- Number of relation groups refined vs skipped (from `json/stage3_state.json::islands_completed`, `islands_failed`)
- The path `json/operation_plan_revised.json` and op count
- Total wall-clock time if visible from log timestamps

# Constraints

- Never edit `orchestrate.py`, sub-skill source, or state files directly — drive only through CLI flags and the dispatch handlers above.
- Never call `--force` after the first iteration; resume only.
- Never read large log files (orchestrate.py stdout). Read only the tail. Read `stage3_state.json` and small dispatch lines.
- Never serialize island refines — always dispatch all pending groups in one parallel batch (per the user's directive). The SKILL.md's older "sequential" guidance is overridden here.
- If `Skill` tool is unavailable or fails for a sub-skill dispatch, fall back to `Agent(subagent_type="general-purpose", prompt="Run the /<sub-skill-name> skill for scene_dir=<scene_dir>. Follow its SKILL.md exactly. Return only after all required outputs exist.")` and verify outputs the same way.
