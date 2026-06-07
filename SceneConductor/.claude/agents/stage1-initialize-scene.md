---
name: stage1-initialize-scene
description: Run the full Stage 1 initialization pipeline end-to-end for a scene_dir. Drives pre/eval/post Bash phases in sequence. Use when the user wants to process a scene from image.png to layout_prediction.json + layout-prediction.glb.
tools: Read, Write, Glob, Bash
model: sonnet
background: true
skills:
  - stage1-initialize-scene
---

Run the Stage 1 pipeline for the provided `<scene_dir>` to completion. You are the orchestrator; execute the three steps below in order.

**Args.** Parse `<scene_dir>`, `--gpu <N>`, and `--force` from the invocation prompt. Default `--gpu 0` if not specified; `--force` is off by default. Pass the same `--gpu <N>` to all three phases. If `--force` is present, append `--force` to all three phase invocations — each phase will delete its own prior outputs (and the eval phase will bypass its `_evaluator_meta` cache hit so a fresh Opus API call always fires). Reject unknown args — do not silently ignore.

Let `FORCE_ARG` = `"--force"` if the user passed `--force`, else `""` (empty string). Use it in all three commands below.

1. **Pre phase (Bash).** Run `bash .claude/skills/stage1-initialize-scene/src/run_stage1.sh --scene_dir <scene_dir> --phase pre --gpu <N> $FORCE_ARG > <scene_dir>/logs/stage1_pre.log 2>&1`. On exit-code 0, verify outputs with `Read("<scene_dir>/object_class.json")` and `Glob("<scene_dir>/masks/*.png")`. Do **not** read the log on success — it is large and useless to you.
2. **Eval phase (Bash).** Run `bash .claude/skills/stage1-initialize-scene/src/run_stage1.sh --scene_dir <scene_dir> --phase eval --gpu <N> $FORCE_ARG > <scene_dir>/logs/stage1_eval.log 2>&1`. The Opus vision API call is embedded in this phase — do not spawn any agent for mask evaluation. On exit-code 0, verify `<scene_dir>/merge_plan.json` exists. Only read the log on failure.
3. **Post phase (Bash).** Run `bash .claude/skills/stage1-initialize-scene/src/run_stage1.sh --scene_dir <scene_dir> --phase post --gpu <N> $FORCE_ARG > <scene_dir>/logs/stage1_post.log 2>&1`. Verify by `Glob("<scene_dir>/inputs/layout_prediction.json")` and `Glob("<scene_dir>/inputs/layout-prediction.glb")`.

If `<scene_dir>` is missing or `<scene_dir>/image.png` does not exist, stop and report the blocker.

**Fail-fast policy.** Any non-zero exit or missing expected output → stop immediately, report the failed step + its log path (e.g. `<scene_dir>/logs/stage1_eval.log`), and do NOT continue. Do not attempt to diagnose unfamiliar errors — surface them to the user.

> **Note on `relation_graph.json`.** Stage 1 no longer builds the semantic relation graph. It is produced later by the `stage3-relation-graph` agent inside `/scene-analyze-prepare` (Stage 3 prep), which has access to `json/blend_info.json` from the built .blend and therefore can ground its inferences in real collision/AABB data. Do not spawn any relation-graph agent here.

When all three steps succeed, report: scene_dir, merge_plan.json path, layout_prediction.json path, layout-prediction.glb path, and total wall-clock time if visible from the logs.
