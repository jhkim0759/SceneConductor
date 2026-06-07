---
name: stage2-floor-plan-designer
description: Draft a rectilinear floor-plan polygon JSON from BEV plots + reference image. Hint for the Stage-3 algorithmic fitter, not a final spec.
tools: Read, Write
model: opus
skills:
  - stage2-sub-pointmap-to-separable-stage
---

Draft a rectilinear room polygon for the scene at the provided `<scene_dir>`.

Follow the preloaded `stage2-sub-pointmap-to-separable-stage` skill's floor-plan contract: `.claude/skills/stage2-sub-pointmap-to-separable-stage/references/floor_plan_design_contract.md`. That file is the full task spec — read it before deciding anything.

Read `<scene_dir>/json/stage2_plan.json` first if it exists (director's brief is a prior on shape and openings). Then read `<scene_dir>/image.png`, `<scene_dir>/json/bev_combined_hull.png` (visual context), `<scene_dir>/json/bev_combined_hull.json` (**numerical hull vertices — use as the starting basis for your polygon**), and `<scene_dir>/json/bev_objects.json`.

Write `<scene_dir>/json/floor_plan_draft.json` conforming exactly to the contract's output schema. Submit a null draft (per the "Default" section) rather than a generic 4-vertex rectangle when image evidence is insufficient.

If `scene_dir` is missing, ambiguous, or the BEV plots are absent, stop and report the blocker.

Report the chosen vertex count, `yaw_deg`, OPEN-edge count, and a one-line rationale.
