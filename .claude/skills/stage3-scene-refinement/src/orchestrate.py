"""
stage3-scene-refinement orchestrator
=====================================
Deterministic state machine for the Stage 3 linear pipeline.

Usage:
    python orchestrate.py <scene_dir>
        [--islands-only]   # jump to Step 2 (island-refiner); assumes planned blend exists
        [--no-island]      # skip Step 2 entirely; run Steps 1 + 3 + 4 only
        [--resume]         # resume from last incomplete step via state.json
        [--num-max-iter N] # max iterations per island group (default 20)

Dispatch model
--------------
This script is the *deterministic* half of the orchestrator. It:
  - validates inputs
  - runs the purely-Python/Blender steps (Steps 1.5, 3, 4)
  - writes state.json after each step

Step 1 (scene-refiner) is agent-dispatched: orchestrate.py sets step =
"scene-refiner" and step_status = "awaiting_agent", writes state.json, then
exits. The Haiku agent invokes /stage3-sub-scene-refiner, then re-runs
orchestrate.py with --resume.

Step 1.5 (rebuild-islands) is deterministic: orchestrate.py runs
rebuild_islands.py to produce per-group island.blend files, populates
state['islands_pending'] from json/island_groups.json, then advances to
island-refiner.

Step 2 (island-refiner) is agent-dispatched per group: orchestrate.py
reads islands_pending, checks each group's simple_refiner/iter_K/transforms.json
for resume, then emits one Task per pending group (stage3-island-refiner) and
exits with step_status=awaiting_agent. On --resume, completed groups are moved
to islands_completed. When all groups are done, advances to merge-back.

Steps 3 and 4 are fully deterministic and are executed directly by
orchestrate.py via subprocess.

Pure stdlib + yaml. No anthropic SDK imports.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

# orchestrate.py is at:  <repo>/.claude/skills/stage3-scene-refinement/src/
_SRC_DIR = Path(__file__).resolve().parent          # .../src/
_SKILL_ROOT = _SRC_DIR.parent                        # .../stage3-scene-refinement/
_SKILLS_ROOT = _SKILL_ROOT.parent                    # .../skills/
_REPO_ROOT = _SKILLS_ROOT.parent.parent              # SceneConductor/


def _load_dirs() -> dict:
    """Load DIRECTORYS.yaml from repo root."""
    dirs_path = _REPO_ROOT / "DIRECTORYS.yaml"
    if not dirs_path.exists():
        _fatal(f"DIRECTORYS.yaml not found at {dirs_path}")
    try:
        import yaml  # type: ignore
    except ImportError:
        _fatal("PyYAML is required — install with: pip install pyyaml")
    return yaml.safe_load(dirs_path.read_text())


def _resolve_blender(dirs: dict) -> str:
    """Return Blender binary path: env var $BLENDER > DIRECTORYS.yaml."""
    if "BLENDER" in os.environ:
        return os.environ["BLENDER"]
    sys_platform = platform.system().lower()
    if sys_platform == "windows" and "blender_bin_windows" in dirs:
        return dirs["blender_bin_windows"]
    if sys_platform == "darwin" and "blender_bin_macos" in dirs:
        return dirs["blender_bin_macos"]
    return str(_REPO_ROOT / dirs.get("blender_bin", "blender"))


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_STEPS_ORDER = [
    "scene-analyze-prepare",  # Step -1: Pre-step to ensure required JSON files exist
    "scene-refiner",
    "planner-review",
    "apply-revised-plan",
    "render-auto",
    "validation",
    "rebuild-islands",        # Step 1.5: deterministic — produce per-group island.blend
    "island-refiner",         # Step 2: per-group Task dispatch — stage3-island-refiner
    "merge-back",
    "render",
    "done",
]

_STATE_DEFAULTS: dict = {
    "scene_dir": "",
    "step": "scene-analyze-prepare",
    "step_status": "ok",
    "started_at": "",
    "updated_at": "",
    "scene_refiner_ops": 0,
    # Island-refiner state
    "islands_pending": [],
    "islands_completed": [],
    "islands_failed": [],
    "merge_back_blend": None,
    "render_outputs": [],
}


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _state_path(scene_dir: Path) -> Path:
    return scene_dir / "json" / "stage3_state.json"


def _load_state(scene_dir: Path) -> dict:
    sp = _state_path(scene_dir)
    if sp.exists():
        try:
            return json.loads(sp.read_text())
        except Exception as exc:
            _log(f"WARNING: could not parse state.json ({exc}); starting fresh")
    state = dict(_STATE_DEFAULTS)
    state["scene_dir"] = str(scene_dir)
    state["started_at"] = _now_iso()
    return state


def _save_state(scene_dir: Path, state: dict) -> None:
    sp = _state_path(scene_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now_iso()
    sp.write_text(json.dumps(state, indent=2))


def _backup_and_reset_stage3(scene_dir: Path, skip_prepare: bool) -> None:
    """Move existing Stage-3 outputs to .stage3_backup_<UTC>/ inside scene_dir.

    When skip_prepare=True, the 3 outputs of stage3-sub-scene-analyze-prepare
    (json/object_state.json, json/blend_info.json, inputs/relation_graph.json)
    are LEFT IN PLACE so the user can re-run refinement without redoing prepare.
    When skip_prepare=False (the --force-alone case), those 3 files are also
    moved to the backup so prepare will re-run from scratch.

    Does not touch Stage-1 / Stage-2 outputs (image.png, blender_scene.blend,
    object_class.json, masks, polygon_v2.json, etc.) — only Stage-3 artifacts.

    If no Stage-3 outputs exist, prints an info message and removes the empty
    backup directory.
    """
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    backup_dir = scene_dir / f".stage3_backup_{timestamp}"

    # Stage-3-only outputs (always backed up under --force)
    stage3_outputs: list[Path] = [
        Path("json/stage3_state.json"),
        Path("json/operation_plan.json"),
        Path("json/operation_plan_revised.json"),
        Path("json/heuristic_ops.json"),
        Path("json/graph_ops.json"),
        Path("json/llm_ops.json"),
        Path("json/island_groups.json"),
        Path("json/relation_pairs.json"),
        Path("json/relation_solve_ops.json"),
        Path("blend/stage3-sub-planned.blend"),
        Path("blend/stage3-scene.blend"),
        Path("render/planned"),
        Path("render/final"),
        Path("relation_groups"),
        Path("scene-refine-loop"),
    ]

    # scene-analyze-prepare outputs — only backed up if we are NOT skipping prepare
    if not skip_prepare:
        stage3_outputs.extend([
            Path("json/object_state.json"),
            Path("json/blend_info.json"),
            Path("inputs/relation_graph.json"),
        ])

    moved_any = False
    for rel in stage3_outputs:
        src = scene_dir / rel
        if not src.exists() and not src.is_symlink():
            continue
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved_any = True
        _log(f"--force backup: {rel} → {dst.relative_to(scene_dir)}")

    if moved_any:
        _log(f"--force: backup created at {backup_dir.relative_to(scene_dir)} "
             f"(skip_prepare={skip_prepare})")
    else:
        _log("--force: no existing Stage-3 outputs found, nothing to back up")
        if backup_dir.exists():
            try:
                backup_dir.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[orchestrate] {msg}", file=sys.stderr, flush=True)


def _fatal(msg: str) -> None:
    print(f"[orchestrate] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_REQUIRED_INPUTS = [
    "image.png",
    "inputs/object_state_annotated_mask.png",
    "inputs/object_class.json",
    "inputs/relation_graph.json",
    "json/blender_scene.json",
    "json/blend_info.json",
    "json/object_state.json",
    "json/polygon_v2.json",
    "inputs/merge_plan.json",
    "blend/blender_scene.blend",
]


def _validate_inputs(scene_dir: Path) -> None:
    missing = [p for p in _REQUIRED_INPUTS if not (scene_dir / p).exists()]
    if missing:
        lines = "\n  ".join(missing)
        _fatal(
            f"Missing required inputs in {scene_dir}:\n  {lines}\n"
            "Run stage3-sub-scene-analyze-prepare first."
        )


def _load_relation_graph(scene_dir: Path) -> list[dict]:
    rg_path = scene_dir / "inputs" / "relation_graph.json"
    if not rg_path.exists():
        _fatal(f"relation_graph.json not found at {rg_path}")
    data = json.loads(rg_path.read_text())
    return data.get("groups", [])


# ---------------------------------------------------------------------------
# Step -1 — scene-analyze-prepare (pre-step: agent-dispatched or skipped)
# ---------------------------------------------------------------------------


def _blend_info_is_valid(path: Path) -> bool:
    """Return True only if json/blend_info.json looks like a well-formed extract.

    A corrupt run can leave a file that passes the size>0 check but contains
    categories whose values are plain name-strings instead of object-dicts, or
    where categories['objects'] is empty while object_count_total > 0.
    Catching this here forces a fresh re-extraction rather than silently
    feeding broken data to the scene-refiner.
    """
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False  # unreadable / not valid JSON

    cats = data.get("categories")
    if not isinstance(cats, dict):
        return False

    # categories.objects must be a list of dicts (each with a "name" key)
    objs = cats.get("objects")
    if not isinstance(objs, list):
        return False

    # If there are claimed objects but the list is empty, the file is corrupt.
    total = data.get("object_count_total", 0)
    if total > 0 and len(objs) == 0:
        _log(
            f"blend_info.json has object_count_total={total} but "
            f"categories.objects=[] — treating as corrupt, will re-extract."
        )
        return False

    # Values must be dicts, not bare name-strings
    for cat_name, items in cats.items():
        if not isinstance(items, list):
            return False
        for item in items:
            if not isinstance(item, dict):
                _log(
                    f"blend_info.json categories['{cat_name}'] contains "
                    f"{type(item).__name__!r} instead of dict — corrupt file, will re-extract."
                )
                return False

    return True


def _step_pre_scene_analyze_prepare(scene_dir: Path, state: dict, args: argparse.Namespace) -> None:
    """
    Pre-step: Check if scene-analyze-prepare outputs exist.
    If all three critical JSON files are present, non-empty, and valid, skip (pass-through).
    Otherwise, dispatch the skill to create them.
    """
    # 3 critical files that scene-analyze-prepare produces
    required_json_files = [
        scene_dir / "json" / "object_state.json",
        scene_dir / "json" / "blend_info.json",
        scene_dir / "inputs" / "relation_graph.json",
    ]

    # Check if ALL three exist and have non-zero size
    all_exist = all(f.exists() and f.stat().st_size > 0 for f in required_json_files)

    # Additionally validate blend_info.json content (size>0 is not sufficient —
    # a previous bug produced a 619-byte file with string-array categories and
    # empty objects that passed the size check but broke downstream planning).
    blend_info_ok = (
        all_exist
        and _blend_info_is_valid(scene_dir / "json" / "blend_info.json")
    )

    if all_exist and blend_info_ok:
        # All files present and valid — skip to next step
        _log("step=scene-analyze-prepare status=ok (cached) — all outputs exist")
        state["step"] = "scene-refiner"
        state["step_status"] = "ok"
        _save_state(scene_dir, state)
        return

    # Files missing, empty, or corrupt — dispatch skill and wait for completion
    _log("step=scene-analyze-prepare status=awaiting_agent — dispatching stage3-sub-scene-analyze-prepare")
    state["step"] = "scene-analyze-prepare"
    state["step_status"] = "awaiting_agent"
    _save_state(scene_dir, state)

    print(
        f"[orchestrate:dispatch] /stage3-sub-scene-analyze-prepare {scene_dir}\n"
        "Invoke the stage3-sub-scene-analyze-prepare skill to generate:\n"
        "  - json/object_state.json\n"
        "  - json/blend_info.json\n"
        "  - inputs/relation_graph.json\n"
        f"After it returns, re-run: python orchestrate.py {scene_dir} --resume"
    )
    sys.exit(0)




# ---------------------------------------------------------------------------
# Step 1 — scene-refiner (agent-dispatched)
# ---------------------------------------------------------------------------


def _step1_scene_refiner(scene_dir: Path, state: dict, args: argparse.Namespace) -> None:
    """
    The scene-refiner is agent-dispatched: orchestrate.py signals readiness
    by setting step = "scene-refiner" and step_status = "awaiting_agent",
    then exits. The Haiku agent invokes /stage3-sub-scene-refiner, verifies
    the outputs, then re-runs orchestrate.py --resume to continue.

    Expected outputs before --resume can advance past this step:
      blend/stage3-sub-planned.blend
      json/operation_plan.json
    """
    planned_blend = scene_dir / "blend" / "stage3-sub-planned.blend"
    plan_json = scene_dir / "operation_plan.json"
    heuristic_ops = scene_dir / "json" / "heuristic_ops.json"
    graph_ops = scene_dir / "json" / "graph_ops.json"

    _step1_outputs = [planned_blend, plan_json, heuristic_ops, graph_ops]
    if all(p.exists() for p in _step1_outputs):
        # All 4 outputs present — check freshness before short-circuiting.
        min_ops_mtime, min_ops_name = _min_ops_mtime(scene_dir, _step1_outputs)
        max_inp_mtime, max_inp_name = _max_input_mtime(scene_dir)

        def _iso(mtime: float) -> str:
            return datetime.datetime.fromtimestamp(
                mtime, tz=datetime.timezone.utc
            ).isoformat()

        if min_ops_mtime > max_inp_mtime:
            _log(
                f"step=1 short-circuit: all 4 ops outputs newer than all inputs "
                f"(min_ops_mtime={_iso(min_ops_mtime)}, max_input_mtime={_iso(max_inp_mtime)})"
            )
            state["step"] = "planner-review"
            state["step_status"] = "ok"
            _save_state(scene_dir, state)
            return
        else:
            _log(
                f"step=1 NOT short-circuiting: {min_ops_name} ops file "
                f"{_iso(min_ops_mtime)} is older than input {max_inp_name} "
                f"({_iso(max_inp_mtime)}) — will re-dispatch scene-refiner"
            )

    _log("step=1 status=awaiting_agent — dispatching /stage3-sub-scene-refiner")
    state["step"] = "scene-refiner"
    state["step_status"] = "awaiting_agent"
    _save_state(scene_dir, state)

    print(
        f"[orchestrate:dispatch] /stage3-sub-scene-refiner {scene_dir}\n"
        "After /stage3-sub-scene-refiner completes, re-run:\n"
        f"  python orchestrate.py {scene_dir} --resume"
    )
    sys.exit(0)


# ---------------------------------------------------------------------------
# Step planner-review — subprocess: run_stage3_planner_review.py
# ---------------------------------------------------------------------------


def _step_planner_review(scene_dir: Path, state: dict, args: argparse.Namespace) -> None:
    """
    Subprocess: run_stage3_planner_review.py (embedded Opus vision call via claude CLI).
    Reviews operation_plan.json and writes json/operation_plan_revised.json with a
    tamper-evident _planner_meta block.
    """
    _log("step=planner-review status=running run_stage3_planner_review.py")
    state["step"] = "planner-review"
    state["step_status"] = "running"
    _save_state(scene_dir, state)

    # Invoke Opus vision agent via subprocess wrapper (no Skill/Agent dispatch).
    try:
        subprocess.run(
            [sys.executable, str(_THIS_DIR / "run_stage3_planner_review.py"),
             "--scene_dir", str(scene_dir)],
            check=True,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        state["step"] = "planner-review"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            f"step=planner-review status=failed: run_stage3_planner_review.py exited {exc.returncode}"
        )
    except subprocess.TimeoutExpired:
        state["step"] = "planner-review"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal("step=planner-review status=failed: run_stage3_planner_review.py timed out (900s)")

    # Verify json/operation_plan_revised.json produced with required keys.
    revised_plan_json = scene_dir / "json" / "operation_plan_revised.json"
    if not revised_plan_json.exists():
        state["step"] = "planner-review"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            "step=planner-review status=failed: json/operation_plan_revised.json not found after planner call"
        )
    try:
        revised_data = json.loads(revised_plan_json.read_text())
    except Exception as exc:
        state["step"] = "planner-review"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            f"step=planner-review status=failed: json/operation_plan_revised.json is not valid JSON: {exc}"
        )
    if "operation_list" not in revised_data:
        state["step"] = "planner-review"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            "step=planner-review status=failed: json/operation_plan_revised.json missing key 'operation_list'"
        )

    _log(
        f"step=planner-review status=ok: "
        f"operation_list has {len(revised_data['operation_list'])} op(s)"
    )
    state["step"] = "apply-revised-plan"
    state["step_status"] = "ok"
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Step apply-revised-plan — deterministic (copy source → apply revised plan)
# ---------------------------------------------------------------------------


def _step_apply_revised_plan(scene_dir: Path, state: dict, args: argparse.Namespace) -> None:
    """Deterministic: refresh working blend from source, apply operation_plan_revised.json."""
    source_blend = scene_dir / "blend" / "blender_scene.blend"
    working_blend = scene_dir / "blend" / "stage3-sub-planned.blend"
    revised_plan_json = scene_dir / "json" / "operation_plan_revised.json"

    if not source_blend.exists():
        _fatal(f"step=apply-revised-plan: blend/blender_scene.blend not found in {scene_dir}")
    if not revised_plan_json.exists():
        _fatal(
            f"step=apply-revised-plan: json/operation_plan_revised.json not found in {scene_dir}. "
            "Step planner-review must complete first."
        )
    if not _APPLY_PLAN_PY.exists():
        _fatal(f"step=apply-revised-plan: apply_plan.py not found at {_APPLY_PLAN_PY}")

    # 1. Copy source → working blend (resolve symlinks to get real file content).
    _log("step=apply-revised-plan: copying blender_scene.blend → stage3-sub-planned.blend")
    real_source = source_blend.resolve()
    shutil.copyfile(str(real_source), str(working_blend))

    # 2. Apply json/operation_plan_revised.json via apply_plan.py.
    _log("step=apply-revised-plan: running apply_plan.py with revised plan")
    cmd_apply = [
        "conda", "run", "-n", "sceneconductor",
        "python3", str(_APPLY_PLAN_PY),
        str(revised_plan_json),
        str(working_blend),
        str(working_blend),
    ]
    _log(f"  cmd: {' '.join(cmd_apply)}")
    result = subprocess.run(cmd_apply, capture_output=False)
    if result.returncode != 0:
        state["step"] = "apply-revised-plan"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            f"step=apply-revised-plan status=failed: apply_plan.py exited {result.returncode}"
        )

    _log("step=apply-revised-plan status=ok")
    state["step"] = "render-auto"
    state["step_status"] = "ok"
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Step render-auto — render the planned blend (deterministic)
# ---------------------------------------------------------------------------

_RENDER_PLANNED_PY = (
    Path(__file__).resolve().parent.parent.parent
    / "stage3-sub-scene-refiner" / "src" / "render_planned.py"
)

_APPLY_PLAN_PY = (
    Path(__file__).resolve().parent.parent.parent
    / "stage3-sub-scene-refiner" / "src" / "apply_plan.py"
)

_RENDER_PLANNED_VIEWS = [
    "render/planned/blender_scene_view_perspective.png",
    "render/planned/blender_scene_view_bev.png",
    "render/planned/blender_scene_view_wide.png",
    "render/planned/blender_scene_view_topcorner.png",
    "render/planned/blender_scene_view_topcorner_opposite.png",
]
# render_planned.py CLI: <scene_dir> <blend_in> [output_subdir]
# output_subdir is the bare directory name under <scene_dir>/render/ — NOT a path.
_RENDER_PLANNED_SUBDIR = "planned"


def _run_render_planned(scene_dir: Path, blend_path: Path, render_subdir: str) -> None:
    """Run render_planned.py via conda and check return code.

    render_planned.py CLI: <scene_dir> <blend_in> [output_subdir]
    output_subdir is the subdirectory name under <scene_dir>/render/
    (e.g. "planned" → <scene_dir>/render/planned/).
    """
    if not _RENDER_PLANNED_PY.exists():
        _fatal(f"render_planned.py not found at {_RENDER_PLANNED_PY}")
    cmd = [
        "conda", "run", "-n", "sceneconductor",
        "python3", str(_RENDER_PLANNED_PY),
        str(scene_dir),
        str(blend_path),
        render_subdir,
    ]
    _log(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"render_planned.py exited with code {result.returncode}"
        )


def _step17_render_auto(scene_dir: Path, state: dict, args: argparse.Namespace) -> None:
    """Deterministic: render the planned blend into render/planned/."""
    planned_blend = scene_dir / "blend" / "stage3-sub-planned.blend"
    if not planned_blend.exists():
        _fatal(
            f"blend/stage3-sub-planned.blend not found in {scene_dir}.\n"
            "Step 1 (scene-refiner) must complete before render-auto."
        )

    _log("step=render-auto status=running render_planned.py")
    try:
        _run_render_planned(scene_dir, planned_blend, _RENDER_PLANNED_SUBDIR)
    except RuntimeError as exc:
        state["step"] = "render-auto"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(f"step=render-auto status=failed: {exc}")

    # Verify outputs
    missing_views = [
        v for v in _RENDER_PLANNED_VIEWS
        if not (scene_dir / v).exists()
    ]
    if missing_views:
        state["step"] = "render-auto"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            f"step=render-auto status=failed: missing render outputs:\n  "
            + "\n  ".join(missing_views)
        )

    _log("step=render-auto status=ok")
    # NOTE: relation-solve / re-apply-render steps are bypassed — they were
    # not reliably improving operation_plan.json. Jump straight to validation.
    state["step"] = "validation"
    state["step_status"] = "ok"
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Path constant for this directory (used by subprocess calls)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Step validation — subprocess: run_stage3_validation.py
# ---------------------------------------------------------------------------


def _step18_validation(scene_dir: Path, state: dict, args: argparse.Namespace) -> None:
    """
    Subprocess: run_stage3_validation.py (embedded Opus vision call via claude CLI).
    Writes json/island_groups.json with groups_needing_island and natural-language rationale.
    """
    _log("step=validation status=running run_stage3_validation.py")
    state["step"] = "validation"
    state["step_status"] = "running"
    _save_state(scene_dir, state)

    # Invoke Opus vision agent via subprocess wrapper (no Skill/Agent dispatch).
    try:
        subprocess.run(
            [sys.executable, str(_THIS_DIR / "run_stage3_validation.py"),
             "--scene_dir", str(scene_dir)],
            check=True,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        state["step"] = "validation"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            f"step=validation status=failed: run_stage3_validation.py exited {exc.returncode}"
        )
    except subprocess.TimeoutExpired:
        state["step"] = "validation"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal("step=validation status=failed: run_stage3_validation.py timed out (900s)")

    # Verify json/island_groups.json produced with required keys.
    island_groups_json = scene_dir / "json" / "island_groups.json"
    if not island_groups_json.exists():
        state["step"] = "validation"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            "step=validation status=failed: json/island_groups.json not found after planner call"
        )
    try:
        ig_data = json.loads(island_groups_json.read_text())
    except Exception as exc:
        state["step"] = "validation"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            f"step=validation status=failed: json/island_groups.json is not valid JSON: {exc}"
        )
    if "groups_needing_island" not in ig_data:
        state["step"] = "validation"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(
            "step=validation status=failed: json/island_groups.json missing key 'groups_needing_island'"
        )

    _gni = ig_data["groups_needing_island"]
    _synth = [g for g in _gni if re.match(r"^S\d+$", g)]
    _graph = [g for g in _gni if g not in _synth]
    _log(
        f"step=validation status=ok plan_path={island_groups_json.name} "
        f"groups_needing_island={_gni} (graph={len(_graph)} synthetic={len(_synth)})"
    )
    # Preserve island_groups.json output; advance to rebuild-islands.
    state["step"] = "rebuild-islands"
    state["step_status"] = "ok"
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Step 2 — island-refiner helpers
# ---------------------------------------------------------------------------

# Path to stage3-sub-island-refiner/src/ (contains init_iter0.py, apply_delta.py,
# render_one.py, score_info.py — the per-group island-refiner harness).
STAGE3_SUB_ISLAND_REFINER_SRC = (
    Path(__file__).resolve().parent.parent.parent / "stage3-sub-island-refiner" / "src"
)


def _sha256(path: Path):
    """Return the file's sha256 hex digest, or None if the path does not exist."""
    import hashlib
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str | None:
    """Return the file's MD5 hex digest, or None if the path does not exist."""
    import hashlib
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _island_actually_changed(group_dir: Path, max_iter: int = None) -> tuple[bool, str]:
    """Check whether the island-refiner produced a real (non-identity) result.

    The ``max_iter`` parameter is accepted for backwards compatibility but is
    **ignored**.  The actual final iteration is auto-detected on disk so that
    resuming with a different --num-max-iter value does not falsely mark
    previously-completed groups as failed.

    Returns (True, reason) when all conditions pass.
    Returns (False, reason) for any of:
      - no completed iter > 0 found (iter_N/island.blend + iter_N/transforms.json)
      - iter_0/island.blend is missing
      - iter_0 and iter_<actual_final> blend files are byte-identical
      - any iter_K/transforms.json for K in 1..actual_final is missing

    This is the orchestrator-side completion gate that catches the no-op bug
    where the Opus subagent short-circuits the loop at iter_1 with empty members
    and converged:true, leaving all island.blend files byte-identical to iter_0.
    """
    loop_dir = group_dir / "simple_refiner"

    # Auto-detect the highest completed iter N (N >= 1) where BOTH
    # iter_N/island.blend and iter_N/transforms.json exist on disk.
    actual_final: int | None = None
    if loop_dir.is_dir():
        for sub in loop_dir.iterdir():
            if not sub.is_dir() or not sub.name.startswith("iter_"):
                continue
            try:
                k = int(sub.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            if k < 1:
                continue
            if (sub / "island.blend").exists() and (sub / "transforms.json").exists():
                if actual_final is None or k > actual_final:
                    actual_final = k

    if actual_final is None:
        return False, f"no completed iter > 0 found under {loop_dir}"

    iter0_blend = loop_dir / "iter_0" / "island.blend"
    if not iter0_blend.exists():
        return False, f"iter_0/island.blend missing in {loop_dir}"

    iterN_blend = loop_dir / f"iter_{actual_final}" / "island.blend"
    h0 = _md5(iter0_blend)
    hN = _md5(iterN_blend)
    if h0 == hN:
        return (
            False,
            f"iter_0 and iter_{actual_final} blend identical — refiner produced identity "
            f"(MAX_ITER seen on disk = {actual_final})",
        )

    # Verify transforms.json exists for every iter 1..actual_final.
    missing_transforms = []
    for k in range(1, actual_final + 1):
        tf = loop_dir / f"iter_{k}" / "transforms.json"
        if not tf.exists():
            missing_transforms.append(k)
    if missing_transforms:
        ks = ", ".join(str(k) for k in missing_transforms[:5])
        if len(missing_transforms) > 5:
            ks += f", ... ({len(missing_transforms)} total)"
        return False, f"iter_K/transforms.json missing for K={ks}; actual final = {actual_final}"

    return True, f"verified actual_final={actual_final}, iter_0 != iter_{actual_final}"


def _find_island_final_iter(group_dir: Path, expected_sha: str | None, max_iter: int) -> int | None:
    """In-run completion detection for the dispatched island-refiner.

    Under the No-resume policy, init_iter0.py rotates any prior simple_refiner/
    to a backup at every dispatch — so any iter_K/transforms.json present here
    was produced by the agent THIS orchestration dispatched. We therefore
    ignore expected_sha (cross-dispatch resume key) and only look at the live
    `final` flag (or hitting max_iter as a fall-through).
    """
    refiner_root = group_dir / "simple_refiner"
    if not refiner_root.exists():
        return None
    candidates: list[int] = []
    for sub in refiner_root.iterdir():
        if not sub.is_dir() or not sub.name.startswith("iter_"):
            continue
        try:
            k = int(sub.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        tj = sub / "transforms.json"
        if not tj.exists():
            continue
        try:
            with tj.open() as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("final") is True or k >= max_iter:
            candidates.append(k)
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Step 1 freshness helpers
# ---------------------------------------------------------------------------


def _max_input_mtime(scene_dir: Path) -> tuple[float, str]:
    """Return (mtime_float, filename) for the newest relevant input file.

    Only files that exist are considered.  The source blend is resolved through
    symlinks so we get the real file's mtime, not the link's.
    """
    candidates: list[tuple[Path, bool]] = [
        # (relative path inside scene_dir, resolve_symlink)
        (scene_dir / "blend" / "blender_scene.blend", True),
        (scene_dir / "inputs" / "relation_graph.json", False),
        (scene_dir / "inputs" / "object_class.json", False),
        (scene_dir / "inputs" / "mask_attribute.json", False),
        (scene_dir / "inputs" / "merge_plan.json", False),
        (scene_dir / "json" / "blender_scene.json", False),
        (scene_dir / "json" / "object_state.json", False),
        (scene_dir / "json" / "polygon_v2.json", False),
    ]

    best_mtime = 0.0
    best_name = "<none>"
    for path, follow_symlink in candidates:
        if not path.exists():
            continue
        try:
            stat_path = path.resolve() if follow_symlink else path
            mtime = stat_path.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best_name = path.name
    return best_mtime, best_name


def _min_ops_mtime(scene_dir: Path, ops_files: list[Path]) -> tuple[float, str]:
    """Return (mtime_float, filename) for the oldest file among ops_files.

    All files in ops_files are assumed to exist (caller already checked).
    """
    best_mtime = float("inf")
    best_name = "<none>"
    for path in ops_files:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if mtime < best_mtime:
            best_mtime = mtime
            best_name = path.name
    return best_mtime, best_name


# ---------------------------------------------------------------------------
# Step 2 group-selection helper
# ---------------------------------------------------------------------------


def _load_island_groups(scene_dir: Path) -> list:
    """Read json/island_groups.json and return groups_needing_island list.

    Raises RuntimeError (caught by caller) if file is absent, unreadable,
    or missing the required key.  Never silently falls back to relation_graph.
    """
    ig_path = scene_dir / "json" / "island_groups.json"
    if not ig_path.exists():
        raise RuntimeError(
            f"json/island_groups.json not found in {scene_dir}. "
            "The validation step must have produced this file before rebuild-islands runs."
        )
    try:
        data = json.loads(ig_path.read_text())
    except Exception as exc:
        raise RuntimeError(
            f"json/island_groups.json is not valid JSON: {exc}"
        )
    if "groups_needing_island" not in data:
        raise RuntimeError(
            "json/island_groups.json is missing key 'groups_needing_island'. "
            "The validation agent must write this key (may be empty list)."
        )
    return data["groups_needing_island"]


# ---------------------------------------------------------------------------
# Step 1.5 bridge — generate selected_groups.json from ALL relation graph groups
# ---------------------------------------------------------------------------


def _bbox_volume(dimensions: list) -> float:
    """Volume from a [x, y, z] dimensions list; treats zero-dim as 1e-3."""
    if not dimensions or len(dimensions) < 3:
        return 0.0
    vol = 1.0
    for v in dimensions:
        vol *= max(float(v), 1e-3)
    return vol


def _pick_anchor(member_ids: list, blend_info_objs: dict) -> str:
    """Return the obj_* id with the largest bbox volume, with class-keyword tiebreak.

    Mirrors select_groups.py::pick_anchor — kept here so the orchestrator can
    resolve anchors without importing the legacy script.
    """
    PRIORITY_KW = ("table", "counter", "shelf", "cart", "cabinet")

    def score(oid: str):
        entry = blend_info_objs.get(oid)
        if not entry:
            return (0, 0.0)
        return (1, _bbox_volume(entry.get("dimensions") or []))

    sorted_by_size = sorted(member_ids, key=score, reverse=True)
    if not sorted_by_size:
        return member_ids[0] if member_ids else ""

    top_two = sorted_by_size[:2]
    for oid in top_two:
        entry = blend_info_objs.get(oid)
        if not entry:
            continue
        cls = (entry.get("class") or "").lower()
        if any(kw in cls for kw in PRIORITY_KW):
            return oid
    return sorted_by_size[0]


def _build_blend_info_lookup(blend_info: dict) -> dict:
    """Build {obj_name: {"dimensions": [...], "class": str}} from blend_info.json.

    Mirrors select_groups.py::_build_blend_info_lookup.
    """
    cats = blend_info.get("categories", {})
    geom_by_name = {g["name"]: g for g in cats.get("geometry_meshes", [])}
    world_by_parent: dict = {}
    for w in cats.get("world", []):
        parent = w.get("parent")
        if parent:
            world_by_parent[parent] = w["name"]

    out: dict = {}
    for entry in cats.get("objects", []):
        name = entry.get("name")
        if not name:
            continue
        real_dims = entry.get("dimensions") or [0.0, 0.0, 0.0]
        world_name = world_by_parent.get(name)
        if world_name:
            for _gname, gentry in geom_by_name.items():
                if gentry.get("parent") == world_name:
                    real_dims = gentry.get("dimensions") or real_dims
                    break
        out[name] = {"dimensions": real_dims, "class": entry.get("class", "")}
    return out


def _build_selected_groups_from_graph(scene_dir: Path) -> None:
    """Generate <scene_dir>/scene-refine-loop/selected_groups.json from ALL groups.

    In the new linear pipeline there is no planner-selection step, so we treat
    every group in inputs/relation_graph.json as selected.  This bridges the gap
    for rebuild_islands.py, which still reads the legacy selected_groups.json.

    Schema written (same as select_groups.py output):
      {"groups": [
          {
            "group_id":  str,
            "group_dir": str   (abs path),
            "members":   list[str],   # non-anchor obj_* ids
            "anchor_id": str,
            "reason":    str,
          }, ...
      ]}
    """
    graph_path = scene_dir / "inputs" / "relation_graph.json"
    if not graph_path.exists():
        _fatal(f"relation_graph.json not found: {graph_path}")

    blend_info_path = scene_dir / "inputs" / "blend_info.json"
    blend_info_objs: dict = {}
    if blend_info_path.exists():
        try:
            import json as _json
            blend_info_objs = _build_blend_info_lookup(
                _json.loads(blend_info_path.read_text())
            )
        except Exception as exc:
            _log(
                f"WARNING: could not parse {blend_info_path}: {exc} — "
                "abstract anchors will fall back to members[0]"
            )
    else:
        _log(
            f"WARNING: {blend_info_path} not found — "
            "abstract anchors will fall back to members[0]"
        )

    graph = json.loads(graph_path.read_text())
    raw_groups = graph.get("groups", [])

    groups = []
    for g in raw_groups:
        group_id = g.get("group_id", "")
        if not group_id:
            continue

        raw_anchor = g.get("anchor", "")
        # Only keep obj_* members (filter out stage geometry names)
        members_from_graph = [m for m in g.get("members", []) if m.startswith("obj_")]

        if not members_from_graph:
            _log(f"WARNING: group {group_id} has no obj_* members — skipping")
            continue

        # Resolve anchor_id: prefer explicit obj_* anchor, else pick by bbox
        if raw_anchor and raw_anchor.startswith("obj_"):
            anchor_id = raw_anchor
            reason = "anchor from relation_graph.json (obj_* direct)"
        else:
            anchor_id = _pick_anchor(members_from_graph, blend_info_objs)
            reason = (
                f"abstract anchor '{raw_anchor}' resolved via bbox pick_anchor "
                f"(all groups selected — no planner-selection step in new pipeline)"
            )
            if not blend_info_objs:
                _log(
                    f"WARNING: group {group_id}: abstract anchor '{raw_anchor}' "
                    f"resolved to {anchor_id} via members[0] fallback"
                )

        # members list in selected_groups.json excludes the anchor
        member_ids = [m for m in members_from_graph if m != anchor_id]
        if not member_ids:
            _log(
                f"WARNING: group {group_id}: no non-anchor members after anchor "
                f"resolution (anchor={anchor_id}) — skipping"
            )
            continue

        groups.append(
            {
                "group_id": group_id,
                "group_dir": str(scene_dir / "relation_groups" / group_id),
                "members": member_ids,
                "anchor_id": anchor_id,
                "reason": reason,
            }
        )

    graph_count = len(groups)

    # --- Append synthetic groups from json/island_groups.json::synthetic_groups ---
    island_groups_path = scene_dir / "json" / "island_groups.json"
    synthetic_count = 0
    if not island_groups_path.exists():
        _log(
            "WARNING: json/island_groups.json not found — skipping synthetic group processing"
        )
    else:
        try:
            ig_data = json.loads(island_groups_path.read_text())
        except Exception as exc:
            _log(
                f"WARNING: could not parse json/island_groups.json ({exc}) — "
                "skipping synthetic group processing"
            )
            ig_data = {}

        synthetic_groups = ig_data.get("synthetic_groups")
        if synthetic_groups and isinstance(synthetic_groups, list):
            # Build a set of (member → group_id) for members already in `groups`
            # so we can detect Change-D conflicts (defense-in-depth).
            existing_members: dict[str, str] = {}
            for eg in groups:
                for em in eg.get("members", []):
                    existing_members[em] = eg["group_id"]

            existing_gids = {eg["group_id"] for eg in groups}

            for entry in synthetic_groups:
                # Validate required keys (defense-in-depth; validation hard-fails on bad schema).
                if not isinstance(entry, dict) or not all(
                    k in entry for k in ("group_id", "member", "anchor")
                ):
                    _log(
                        f"WARNING: malformed synthetic_groups entry (missing required keys) — "
                        f"skipping: {entry}"
                    )
                    continue

                gid = entry["group_id"]
                member = entry["member"]
                anchor = entry["anchor"]

                if not gid or not member or not anchor:
                    _log(
                        f"WARNING: synthetic entry has empty group_id/member/anchor — "
                        f"skipping: {entry}"
                    )
                    continue

                # Change D: guard against member already belonging to a graph group.
                if member in existing_members:
                    _log(
                        f"WARNING: [orchestrate] synthetic {gid} member {member} "
                        f"already a member of graph group {existing_members[member]} — "
                        "dropping synthetic"
                    )
                    continue

                # Guard against duplicate group_id (should be impossible given S-prefix).
                if gid in existing_gids:
                    _log(
                        f"WARNING: synthetic group_id {gid} already present in groups — "
                        "skipping duplicate"
                    )
                    continue

                reason_text = entry.get("reason", "")
                groups.append(
                    {
                        "group_id": gid,
                        "group_dir": str(scene_dir / "relation_groups" / gid),
                        "members": [member],
                        "anchor_id": anchor,
                        "reason": f"synthetic island from validation: {reason_text}",
                    }
                )
                existing_gids.add(gid)
                existing_members[member] = gid
                synthetic_count += 1

    out_dir = scene_dir / "scene-refine-loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_groups.json"
    out_path.write_text(json.dumps({"groups": groups}, indent=2))
    total_count = len(groups)
    _log(
        f"step=1.5 selected_groups.json: "
        f"graph={graph_count} + synthetic={synthetic_count} = {total_count} group(s) → {out_path}"
    )


# ---------------------------------------------------------------------------
# Step 1.5 — rebuild islands (deterministic)
# ---------------------------------------------------------------------------


def _step15_rebuild_islands(scene_dir: Path, state: dict, blender_bin: str) -> None:
    """Populate islands_pending from island_groups.json["groups_needing_island"].

    If _migrated/rebuild_islands.py is present, invoke it to slice
    blend/stage3-sub-planned.blend into per-group island.blend files.
    If the script is missing, only populate the pending list (degraded mode).
    """
    rebuild_script = _SKILL_ROOT / "_migrated" / "rebuild_islands.py"

    planned_blend = scene_dir / "blend" / "stage3-sub-planned.blend"
    if not planned_blend.exists():
        _fatal(
            f"blend/stage3-sub-planned.blend not found in {scene_dir}.\n"
            "Step 1 (scene-refiner) must complete before rebuild_islands."
        )

    if rebuild_script.exists():
        # Bridge: generate selected_groups.json from ALL groups (new pipeline has no
        # planner-selection step — every group is rebuilt).  rebuild_islands.py still
        # reads this file, so we produce it here before calling it.
        _build_selected_groups_from_graph(scene_dir)

        _log("step=1.5 status=running rebuild_islands.py")
        cmd = [sys.executable, str(rebuild_script), str(scene_dir)]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            state["step"] = "rebuild-islands"
            state["step_status"] = "failed"
            _save_state(scene_dir, state)
            _fatal(f"step=1.5 status=failed rebuild_islands.py exited {result.returncode}")
        _log("step=1.5 status=ok rebuild_islands.py completed")
    else:
        _log(
            f"step=1.5 WARNING: rebuild_islands.py not found at {rebuild_script} — "
            "skipping per-group island.blend generation (degraded mode)"
        )

    state["step"] = "island-refiner"
    state["step_status"] = "ok"

    # Populate islands_pending from island_groups.json (written by validation step).
    # Fail-fast if the file is missing or malformed — do NOT fall back to relation_graph.
    try:
        groups_needing_island = _load_island_groups(scene_dir)
    except RuntimeError as exc:
        state["step"] = "rebuild-islands"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(f"step=1.5 status=failed: {exc}")

    if not groups_needing_island:
        _log("step=1.5: island_groups.json reports no groups need refinement — skipping island-refiner")
        state["islands_pending"] = []
        state["islands_completed"] = []
        state["islands_failed"] = []
        state["step"] = "merge-back"
        state["step_status"] = "ok"
        _save_state(scene_dir, state)
        return

    # Intersect with groups that have a valid island.blend after rebuild.
    # Synthetic ids (^S\d+$) are intentionally absent from relation_graph — bypass that check.
    all_graph_groups = {g.get("group_id", "") for g in _load_relation_graph(scene_dir)}
    pending = []
    for gid in groups_needing_island:
        is_synthetic = bool(re.match(r"^S\d+$", gid))
        if not is_synthetic and gid not in all_graph_groups:
            _log(f"WARNING: group {gid} from island_groups.json not in relation_graph — skipping")
            continue
        if rebuild_script.exists():
            island_path = scene_dir / "relation_groups" / gid / "island.blend"
            if island_path.exists():
                pending.append(gid)
            else:
                _log(f"WARNING: island.blend missing for group {gid} — will skip")
        else:
            # degraded mode: add to pending anyway (island-refiner will handle missing blend)
            pending.append(gid)

    _synth_p = [g for g in pending if re.match(r"^S\d+$", g)]
    _graph_p = [g for g in pending if g not in _synth_p]
    _log(
        f"step=1.5 islands_pending={pending} (graph={len(_graph_p)} synthetic={len(_synth_p)})"
    )
    state["islands_pending"] = pending
    state["islands_completed"] = []
    state["islands_failed"] = []
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Step 2 — island-level refiner (agent-dispatched, one Task per group)
# ---------------------------------------------------------------------------


_ISLAND_REFINER_MAX_ITER = 20  # default iteration count; overridable via --num-max-iter


def _step2_island_refiner(scene_dir: Path, state: dict, max_iter: int) -> None:
    """Island-level refiner dispatch (dispatches only the groups validation flagged).

    For each group_id in state['islands_pending']:
      - Check resume: scene_dir/relation_groups/<G>/simple_refiner/iter_K/transforms.json
      - If max iter completed → move to islands_completed
      - Else → emit Task dispatch line for stage3-island-refiner

    After all groups dispatched or completed → step = 'merge-back'
    """
    loop_root = scene_dir / "relation_groups"

    pending = list(state.get("islands_pending", []))
    completed = list(state.get("islands_completed", []))
    failed = list(state.get("islands_failed", []))
    still_pending = []

    # Resume guard: re-validate previously-completed groups.
    # If a prior run marked a group as completed but the refiner actually produced
    # no effective change (byte-identical blend files), demote it back to pending so
    # it gets re-dispatched in this run.  This prevents a cached no-op from silently
    # passing through to merge-back.
    still_completed = []
    for gid in completed:
        group_dir = loop_root / gid
        changed, reason = _island_actually_changed(group_dir)
        if changed:
            still_completed.append(gid)
        else:
            _log(
                f"group {gid} was in islands_completed but island-refiner check "
                f"failed ({reason}) — re-queuing for dispatch"
            )
            still_pending.append(gid)
            state.setdefault("islands_failed_reasons", {}).pop(gid, None)
    completed = still_completed

    for gid in pending:
        group_dir = loop_root / gid
        # final_iter always None: per-dispatch fresh-start policy, see stage3-sub-island-refiner/SKILL.md.
        image_sha = _sha256(group_dir / "masked.png")
        final_iter = _find_island_final_iter(group_dir, image_sha, max_iter)
        if final_iter is not None:
            # Gate: verify the refiner actually moved something (not a no-op run).
            changed, reason = _island_actually_changed(group_dir)
            if changed:
                if gid not in completed:
                    completed.append(gid)
                print(f"[orchestrate:island_refiner] {gid} completed at iter {final_iter}")
            else:
                _log(
                    f"group {gid} island-refiner produced no effective change — {reason}"
                )
                if gid not in failed:
                    failed.append(gid)
                # Record failure reason in state for diagnostics.
                if not isinstance(state.get("islands_failed"), list):
                    state["islands_failed"] = []
                # Store as dict entry if not already a plain list of strings.
                # We keep islands_failed as a list and record the reason separately.
                state.setdefault("islands_failed_reasons", {})[gid] = reason
        else:
            still_pending.append(gid)

    state["islands_pending"] = still_pending
    state["islands_completed"] = completed
    state["islands_failed"] = failed

    if still_pending:
        _log(
            f"step=island-refiner status=awaiting_agent — "
            f"dispatching {len(still_pending)} stage3-island-refiner Task(s)"
        )
        print(
            f"[orchestrate:dispatch] stage3-island-refiner — Task per pending group ({len(still_pending)}):"
        )
        for gid in still_pending:
            print(
                f"  Task(\n"
                f"    subagent_type=\"stage3-island-refiner\",\n"
                f"    prompt=\"Refine the island at {scene_dir}/relation_groups/{gid}. "
                f"Run {max_iter} iterations end-to-end.\"\n"
                f"  )"
            )
        print(f"After all dispatches complete, re-run: python orchestrate.py {scene_dir} --resume")
        state["step"] = "island-refiner"
        state["step_status"] = "awaiting_agent"
        _save_state(scene_dir, state)
        sys.exit(0)

    # All islands completed (or none needed) → advance to merge-back
    state["step"] = "merge-back"
    state["step_status"] = "ok"
    _save_state(scene_dir, state)
    print("[orchestrate:island_refiner] all islands completed; advancing to merge-back")


# ---------------------------------------------------------------------------
# Step 3 — merge-back (deterministic: merge per-island transforms → stage3-scene.blend)
# ---------------------------------------------------------------------------

_MERGE_ISLANDS_BACK_PY = Path(__file__).resolve().parent / "merge_islands_back.py"


def _step3_merge_back(scene_dir: Path, state: dict, blender_bin: str) -> None:
    """Merge per-island refined transforms back to stage3-scene.blend.

    Calls merge_islands_back.py which handles canonical→world conversion
    (if _migrated/merge_island_blender_inner.py exists) or falls back to
    copying stage3-sub-planned.blend.
    """
    if not _MERGE_ISLANDS_BACK_PY.exists():
        _fatal(f"merge_islands_back.py not found at {_MERGE_ISLANDS_BACK_PY}")

    state["step"] = "merge-back"
    state["step_status"] = "running"
    _save_state(scene_dir, state)

    cmd = [sys.executable, str(_MERGE_ISLANDS_BACK_PY), str(scene_dir)]
    if blender_bin:
        cmd.extend(["--blender-bin", blender_bin])
    _log(f"step=3 status=running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        state["step"] = "merge-back"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(f"step=3 status=failed merge_islands_back.py exited {result.returncode}")

    refined_blend = scene_dir / "blend" / "stage3-scene.blend"
    if not refined_blend.exists():
        state["step"] = "merge-back"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal("step=3 status=failed blend/stage3-scene.blend not produced by merge_islands_back.py")

    _log(f"step=3 status=ok merge_back_blend={refined_blend}")
    state["step"] = "render"
    state["step_status"] = "ok"
    state["merge_back_blend"] = str(refined_blend)
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Step 4 — multi-view render (deterministic)
# ---------------------------------------------------------------------------


def _step4_render(scene_dir: Path, state: dict, blender_bin: str) -> None:
    refined_blend = scene_dir / "blend" / "stage3-scene.blend"
    if not refined_blend.exists():
        _fatal(
            f"blend/stage3-scene.blend not found.\n"
            "Step 3 (merge-back) must complete before rendering."
        )

    render_script = _SKILLS_ROOT / "general-multi-view-render" / "src" / "render_multi_view.py"
    if not render_script.exists():
        _fatal(
            f"render_multi_view.py not found at {render_script}\n"
            "Ensure general-multi-view-render skill is present."
        )

    _log(f"step=4 status=running general-multi-view-render on {refined_blend.name}")
    cmd = [
        blender_bin,
        "--background", str(refined_blend),
        "--python", str(render_script),
        "--",
        "--scene-dir", str(scene_dir),
        "--samples", "256",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        state["step"] = "render"
        state["step_status"] = "failed"
        _save_state(scene_dir, state)
        _fatal(f"step=4 status=failed Blender render exited {result.returncode}")

    # Copy PNGs to render/final/
    render_dir = scene_dir / "render"
    final_dir = render_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    pngs = list(render_dir.glob("blender_scene_view_*.png"))
    if not pngs:
        _log("WARNING: no blender_scene_view_*.png found in render/ after Blender run")
    outputs = []
    for png in pngs:
        dst = final_dir / png.name
        shutil.copy2(png, dst)
        outputs.append(str(dst))
        _log(f"  copied {png.name} → render/final/")

    _log(f"step=4 status=ok render_outputs={len(outputs)}")
    state["step"] = "done"
    state["step_status"] = "ok"
    state["render_outputs"] = outputs
    _save_state(scene_dir, state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="stage3-scene-refinement deterministic state machine"
    )
    p.add_argument("scene_dir", help="Absolute path to scene directory")
    p.add_argument(
        "--islands-only",
        action="store_true",
        help="Start at Step 2 (island-refiner) — assumes stage3-sub-planned.blend exists",
    )
    p.add_argument(
        "--no-island",
        action="store_true",
        help="Skip Step 2 (island-refiner); run Steps 1 + 3 + 4 only",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last incomplete step using state.json",
    )
    p.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip Step 0 (scene-analyze-prepare); assume its 3 outputs already exist",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Backup any existing Stage-3 outputs to .stage3_backup_<UTC>/ and start fresh. "
             "Implicitly disables --resume. When combined with --skip-prepare, the 3 "
             "scene-analyze-prepare outputs are preserved.",
    )
    p.add_argument(
        "--num-max-iter",
        type=int,
        default=20,
        metavar="N",
        help="Max iter count per island group (default 20). Resume detection "
             "checks simple_refiner/iter_<N>/transforms.json _island_meta.image_sha256 — "
             "changing N between runs may force re-refinement.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.is_dir():
        _fatal(f"scene_dir does not exist or is not a directory: {scene_dir}")

    dirs = _load_dirs()
    blender_bin = _resolve_blender(dirs)
    _log(f"scene_dir={scene_dir}")
    _log(f"blender_bin={blender_bin}")

    # --force: backup any existing Stage-3 outputs and start from a clean state.
    if args.force:
        if args.resume:
            _log("--force overrides --resume: ignoring --resume and starting fresh")
            args.resume = False
        _backup_and_reset_stage3(scene_dir, args.skip_prepare)

    # Load or initialise state.
    if args.resume:
        state = _load_state(scene_dir)
        _log(f"resume: step={state['step']} step_status={state['step_status']}")
    else:
        state = dict(_STATE_DEFAULTS)
        state["scene_dir"] = str(scene_dir)
        state["started_at"] = _now_iso()
        # Save immediately so state.json exists even before first step runs.
        _save_state(scene_dir, state)

    # --islands-only: jump straight to Step 2 (island-refiner).
    islands_only = args.islands_only
    no_island = args.no_island

    # Migration shim: map legacy step names to current equivalents for --resume safety.
    _STEP_MIGRATION = {
        "identify-unwanted": "scene-refiner",
        "apply-deletes":     "scene-refiner",
        "relation-solve":    "render-auto",
        "re-apply-render":   "render-auto",
        # scene-level refiner steps → island-refiner pipeline
        "scene-refiner-loop": "rebuild-islands",
    }
    if state.get("step") in _STEP_MIGRATION:
        old = state["step"]
        state["step"] = _STEP_MIGRATION[old]
        state["step_status"] = "ok"
        _log(f"migrate: legacy step '{old}' → '{state['step']}'")
        _save_state(scene_dir, state)

    # --islands-only: jump straight to Step 2.
    if islands_only:
        _log("--islands-only: jumping to Step 2 (rebuild-islands / island-refiner)")
        state["step"] = "rebuild-islands"
        state["step_status"] = "ok"  # treat prior steps as already done
        _save_state(scene_dir, state)

    # Validate required inputs (unless resuming past the point where they're needed).
    if state.get("step") in ("scene-refiner",):
        _validate_inputs(scene_dir)

    # ---- Step -1: scene-analyze-prepare (pre-step) ----
    if state["step"] == "scene-analyze-prepare":
        if args.skip_prepare:
            _log("step=scene-analyze-prepare SKIPPED (--skip-prepare) — "
                 "advancing directly to scene-refiner")
            state["step"] = "scene-refiner"
            state["step_status"] = "ok"
            _save_state(scene_dir, state)
        else:
            _step_pre_scene_analyze_prepare(scene_dir, state, args)
            state = _load_state(scene_dir)

    # ---- Step 1: scene-refiner ----
    if state["step"] == "scene-refiner" and not islands_only:
        _step1_scene_refiner(scene_dir, state, args)
        # If _step1 didn't exit (outputs already present), reload state.
        state = _load_state(scene_dir)

    # ---- Step planner-review: Opus vision agent reviews and revises operation plan ----
    if state["step"] == "planner-review" and not islands_only:
        _step_planner_review(scene_dir, state, args)
        state = _load_state(scene_dir)

    # ---- Step apply-revised-plan: copy source + apply revised plan ----
    if state["step"] == "apply-revised-plan" and not islands_only:
        _step_apply_revised_plan(scene_dir, state, args)
        state = _load_state(scene_dir)

    # ---- Step render-auto: render the planned blend ----
    if state["step"] == "render-auto" and not islands_only:
        _step17_render_auto(scene_dir, state, args)
        state = _load_state(scene_dir)

    # ---- Step validation: agent-dispatched ----
    if state["step"] == "validation" and not islands_only:
        _step18_validation(scene_dir, state, args)
        state = _load_state(scene_dir)

    # ---- Step 1.5: rebuild-islands (deterministic) ----
    if state["step"] == "rebuild-islands" and not no_island:
        _step15_rebuild_islands(scene_dir, state, blender_bin)
        state = _load_state(scene_dir)

    if state["step"] == "rebuild-islands" and no_island:
        # Skipping island pipeline: advance directly to merge-back.
        _log("--no-island: skipping rebuild-islands and island-refiner (Step 1.5 + 2)")
        state["step"] = "merge-back"
        state["step_status"] = "ok"
        _save_state(scene_dir, state)

    # ---- Step 2: island-level refiner (agent-dispatched, one Task per group) ----
    if state["step"] == "island-refiner" and not no_island:
        _step2_island_refiner(scene_dir, state, args.num_max_iter)
        # If we reach here, all islands completed (no dispatch exit).
        state = _load_state(scene_dir)

    if state["step"] == "island-refiner" and no_island:
        # Should not happen (rebuild-islands already advanced to merge-back above),
        # but handle gracefully.
        _log("--no-island: skipping island-refiner")
        state["step"] = "merge-back"
        state["step_status"] = "ok"
        _save_state(scene_dir, state)

    # ---- Step 3: merge-back ----
    if state["step"] == "merge-back":
        _step3_merge_back(scene_dir, state, blender_bin)
        state = _load_state(scene_dir)

    # ---- Step 4: render ----
    if state["step"] == "render":
        _step4_render(scene_dir, state, blender_bin)
        state = _load_state(scene_dir)

    if state["step"] == "done":
        _log("step=done pipeline complete")
        print(
            f"[orchestrate] Pipeline complete.\n"
            f"  refined blend: {state.get('merge_back_blend')}\n"
            f"  render outputs ({len(state.get('render_outputs', []))} PNGs): "
            f"{scene_dir / 'render' / 'final'}\n"
            f"  state: {_state_path(scene_dir)}"
        )
    else:
        _log(f"Pipeline exited early at step={state['step']} status={state['step_status']}")


if __name__ == "__main__":
    main()
