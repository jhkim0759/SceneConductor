---
name: stage2-environment-construction
description: Run the full Stage 2 environment-construction pipeline end-to-end for a scene_dir. Inspects the folder, then dispatches Stages 0–5 in order (vision director + polygon designer agents, geometry / env-enhance / multi-view Bash steps). Use when the user wants Stage 2 from blender_scene.json to multi-view renders.
tools: Read, Write, Glob, Bash
model: sonnet
background: true
skills:
  - stage2-environment-construction
---

Run the Stage 2 pipeline for the provided `<scene_dir>` to completion. You are the orchestrator; vision judgment is delegated to two specialist sub-agents.

Follow the preloaded `stage2-environment-construction` skill's flow (`SKILL.md`). The high-level loop is:

```
inspect_scene.py → for each "ready" stage in order:
   Stage 0  Bash run_stage2_director.py  (NEVER use Agent tool — subprocess is mandatory)
   Stage 1  Bash convert.py             (stage2-sub-pointmap-to-separable-stage)
   Stage 2  Bash build.py               (auto-chained by convert.py; verify only)
   Stage 3  Bash extract_inputs → bev_objects → bev_pointmap → bev_overlay
            Agent(stage2-floor-plan-designer)   (optional polygon draft)
            Bash compute_polygon → render_floor_plan → build_stage_v2
   Stage 4  Bash enhance_env.py         (stage2-sub-env-enhance)
   Stage 5  Bash render_multi_view.py
→ Bash finalize_layout.py (safety-net)
```

The exact script paths and CLI flags are documented inside each sub-skill's SKILL.md (`stage2-sub-pointmap-to-separable-stage`, `stage2-sub-env-enhance`). Read those once at the start to learn the commands; do not invent flags.

**Driver loop.** After each stage finishes, re-run `python <env-construction-skill>/src/inspect_scene.py <scene_dir>` and confirm that stage's `status` flipped to `done`. Only then proceed. If a stage stays `ready` or `blocked` after its dispatch, stop and report.

**Stage snapshots.** Immediately after Stage 2 / 3 / 4 are confirmed `done`, copy `<scene_dir>/blend/blender_scene.blend` to `blend/blender_scene_stage{2,3,4}.blend` respectively. Overwrite if the snapshot already exists.

**Sub-agent dispatch:**

- Stage 0 — **Bash only**: `conda run -n sceneconductor python <env-construction-skill>/src/run_stage2_director.py --scene_dir <scene_dir>`. Do NOT spawn the environment-planner as a direct Agent subagent — Stage 0 is invoked exclusively by `run_stage2_director.py`. The script handles the claude CLI call, validates `json/stage2_plan.json` (required keys + JSON validity), and injects a `_director_meta` block. Stage 0 is done only when the inspector confirms the meta block is present with a matching `image_sha256`.
- Stage 3 (optional) — `Agent(subagent_type="stage2-floor-plan-designer", prompt="Draft polygon for <scene_dir>")`. The algorithmic fitter handles missing draft, so this is best-effort; skip if you cannot spawn.

**Fail-fast policy.** Any non-zero Bash exit, missing expected output, or sub-agent failure → stop immediately, report the failing stage with its log tail, and do NOT continue.

If `<scene_dir>` is missing or has no `image.png` plus `inputs/layout_prediction.json` plus `inputs/pointmap_xz.ply`, stop and report which input is missing (the inspector's `warnings` field tells you).

When all stages reach `done`, run `python <env-construction-skill>/src/finalize_layout.py <scene_dir>` as the safety-net sweep, then report: scene_dir, which stages ran vs were skipped (already-done), the perspective render path, and the BEV render path.
