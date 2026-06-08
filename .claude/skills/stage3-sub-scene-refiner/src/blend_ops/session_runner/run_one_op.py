"""run_one_op.py — Apply one scene-modification op and advance the session iterator.

CLI:
    python run_one_op.py <session_dir> <op_json_path> [--vision] [--selftest]

`session_dir` must contain a `current` symlink pointing to the latest iter_K dir.
`op_json_path` is an existing JSON file (may live inside iter_<K+1>/ already, or anywhere).

The script:
  1. Validates op JSON (action type, obj_name format).
  2. Determines current iter K from the `current` symlink, creates iter_<K+1>/.
  3. Copies current/scene.blend → iter_<K+1>/input.blend.
  4. Invokes Blender to apply the op → iter_<K+1>/scene.blend.
  5. Re-renders top + persp views → iter_<K+1>/render_top.png, render_persp.png.
  6. Re-runs physics metrics → iter_<K+1>/metrics.json.
  7. Re-runs list_objects → iter_<K+1>/list_objects.json.
  8. Advances the `current` symlink to iter_<K+1>.
  9. Optionally runs semantic evaluation (--vision).
 10. Prints a summary JSON to stdout.

Error contract: if apply/metrics/symlink fails, exits non-zero WITHOUT advancing the
`current` symlink so the agent can safely retry.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# DIRECTORYS.yaml (canonical machine-specific paths)
# ---------------------------------------------------------------------------
# This file lives at <repo>/.claude/skills/scene-refiner/src/blend_ops/session_runner/,
# so the repo root is 6 levels up.
_REPO_ROOT = Path(__file__).resolve().parents[6]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_BLENDER = os.environ.get("BLENDER", _DIRS["blender_bin"])
BLENDER = os.environ.get("SCENE_EVAL_BLENDER", _DEFAULT_BLENDER)

_SCRIPTS_DIR = Path(__file__).parent.resolve()
RUNNER = _SCRIPTS_DIR / "external_blend_runner.py"
EVALUATE_SCENE = _SCRIPTS_DIR / "evaluate_scene.py"

VALID_ACTIONS = {"update_layout", "update_rotation", "update_size", "remove_object", "flip_yaw_180"}
OBJ_NAME_RE = re.compile(r"^obj_\d+$")

# ---------------------------------------------------------------------------
# Low-level Blender subprocess helper (mirrors external_blend_tools._run)
# ---------------------------------------------------------------------------

def _blender_run(blend_in: str, blend_out: str, op: dict, timeout: int = 180) -> dict:
    """Write op to a temp file, invoke Blender, return the .result dict."""
    fd, op_path = tempfile.mkstemp(prefix="blendop_", suffix=".json")
    os.close(fd)
    result_path = op_path + ".result"
    try:
        with open(op_path, "w") as f:
            json.dump(op, f)
        proc = subprocess.run(
            [
                BLENDER, "--background", str(blend_in),
                "--python", str(RUNNER),
                "--", op_path, str(blend_out),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if not os.path.exists(result_path):
            return {
                "success": False,
                "message": (
                    f"no result file. blender exit={proc.returncode}. "
                    f"stderr tail: {proc.stderr[-800:]}"
                ),
            }
        with open(result_path) as f:
            return json.load(f)
    finally:
        for p in (op_path, result_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_current_iter(session_dir: Path) -> int:
    """Return K where session_dir/current → iter_K."""
    current_link = session_dir / "current"
    if not current_link.exists() and not current_link.is_symlink():
        raise FileNotFoundError(f"No `current` symlink found in {session_dir}")
    target = os.readlink(str(current_link))  # e.g. "iter_3" or absolute path
    # strip possible leading "./" or full path
    basename = Path(target).name
    m = re.fullmatch(r"iter_(\d+)", basename)
    if not m:
        raise ValueError(f"`current` symlink points to unexpected target: {target!r}")
    return int(m.group(1))


def _extract_metrics_summary(metrics: dict) -> dict:
    """Return a small dict with BBL, OOB, and top-collision volume."""
    if not metrics.get("success"):
        return {"BBL": None, "OOB": None, "top_collision_m3": None}
    collisions = metrics.get("collisions", [])
    top_vol = collisions[0]["volume_m3"] if collisions else 0.0
    return {
        "BBL": metrics.get("BBL_count", 0),
        "OOB": metrics.get("OOB_count", 0),
        "top_collision_m3": top_vol,
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(session_dir: Path, op_json_path: Path, vision: bool = False) -> int:
    """Execute the full pipeline; return 0 on success, 1 on error."""

    # ------------------------------------------------------------------
    # 1. Validate op JSON
    # ------------------------------------------------------------------
    if not op_json_path.exists():
        print(f"ERROR: op_json_path does not exist: {op_json_path}", file=sys.stderr)
        return 1

    try:
        op_json = json.loads(op_json_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: op JSON is not valid JSON: {exc}", file=sys.stderr)
        return 1

    action = op_json.get("action")
    if action not in VALID_ACTIONS:
        print(
            f"ERROR: action {action!r} is not one of {sorted(VALID_ACTIONS)}",
            file=sys.stderr,
        )
        return 1

    obj_name = op_json.get("obj_name", "")
    if not OBJ_NAME_RE.match(obj_name):
        print(
            f"ERROR: obj_name {obj_name!r} does not match ^obj_\\d+$",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # 2. Determine current iter, create new iter dir
    # ------------------------------------------------------------------
    try:
        k = _resolve_current_iter(session_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    new_k = k + 1
    new_iter_dir = session_dir / f"iter_{new_k}"
    new_iter_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Copy current scene.blend → new_iter/input.blend
    # ------------------------------------------------------------------
    current_dir = session_dir / f"iter_{k}"
    source_blend = current_dir / "scene.blend"
    if not source_blend.exists():
        print(f"ERROR: source blend not found: {source_blend}", file=sys.stderr)
        return 1

    input_blend = new_iter_dir / "input.blend"
    shutil.copy2(str(source_blend), str(input_blend))

    output_blend = new_iter_dir / "scene.blend"

    # Snapshot before-metrics for the summary output
    before_metrics_path = current_dir / "metrics.json"
    before_metrics_raw: dict = {}
    if before_metrics_path.exists():
        try:
            before_metrics_raw = json.loads(before_metrics_path.read_text())
        except Exception:
            pass
    before_summary = _extract_metrics_summary(before_metrics_raw)

    # ------------------------------------------------------------------
    # 4. Write a traceable copy of the op inside new_iter (traceability)
    #    and invoke Blender to apply the op.
    # ------------------------------------------------------------------
    op_trace_path = new_iter_dir / "_op_full.json"
    op_trace_path.write_text(json.dumps(op_json, indent=2))

    # Build apply op (exactly what external_blend_tools functions build)
    apply_op = dict(op_json)  # already has action + obj_name + payload field

    result_path_str = str(op_trace_path) + ".result"

    print(f"[run_one_op] Applying {action} on {obj_name} → {output_blend} ...")
    apply_result = _blender_run(
        blend_in=str(input_blend),
        blend_out=str(output_blend),
        op=apply_op,
        timeout=180,
    )

    if not apply_result.get("success"):
        msg = apply_result.get("message", "unknown error")
        print(f"ERROR: Blender apply failed: {msg}", file=sys.stderr)
        # DO NOT advance symlink
        return 1

    # ------------------------------------------------------------------
    # 5. Re-render top + persp
    # ------------------------------------------------------------------
    render_ok = True
    for view, out_name in (("top", "render_top.png"), ("persp", "render_persp.png")):
        out_png = str(new_iter_dir / out_name)
        render_op = {
            "action": "render",
            "output_png": out_png,
            "view": view,
            "resolution": [800, 600],
        }
        r = _blender_run(
            blend_in=str(output_blend),
            blend_out="/tmp/__noop.blend",
            op=render_op,
            timeout=300,
        )
        if not r.get("success"):
            err_msg = r.get("message", "unknown render error")
            err_file = new_iter_dir / "render_error.txt"
            existing = err_file.read_text() if err_file.exists() else ""
            err_file.write_text(
                existing + f"\n[view={view}] {err_msg}"
            )
            print(
                f"WARNING: render {view!r} failed (non-fatal): {err_msg}",
                file=sys.stderr,
            )
            render_ok = False

    # ------------------------------------------------------------------
    # 6. Re-run physics metrics
    # ------------------------------------------------------------------
    metrics_op = {"action": "metrics", "name_prefix": "obj_", "room_bbox": None}
    metrics_result = _blender_run(
        blend_in=str(output_blend),
        blend_out="/tmp/__noop.blend",
        op=metrics_op,
        timeout=120,
    )
    if not metrics_result.get("success"):
        msg = metrics_result.get("message", "unknown error")
        print(f"ERROR: metrics computation failed: {msg}", file=sys.stderr)
        # Metrics are critical — do NOT advance symlink
        return 1

    metrics_out = new_iter_dir / "metrics.json"
    metrics_out.write_text(json.dumps(metrics_result, indent=2))
    after_summary = _extract_metrics_summary(metrics_result)

    # ------------------------------------------------------------------
    # 7. Re-run list_objects
    # ------------------------------------------------------------------
    list_op = {"action": "list_objects", "name_prefix": "obj_"}
    list_result = _blender_run(
        blend_in=str(output_blend),
        blend_out="/tmp/__noop.blend",
        op=list_op,
        timeout=120,
    )
    if list_result.get("success"):
        (new_iter_dir / "list_objects.json").write_text(json.dumps(list_result, indent=2))
    else:
        print(
            f"WARNING: list_objects failed: {list_result.get('message')}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # 8. Advance current symlink  (only now that everything critical is done)
    # ------------------------------------------------------------------
    current_link = session_dir / "current"
    tmp_link = session_dir / f"current_{new_k}_tmp"
    try:
        # Atomic symlink swap: create a temp symlink then rename over current
        if tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(f"iter_{new_k}", str(tmp_link))
        os.replace(str(tmp_link), str(current_link))
    except Exception as exc:
        print(f"ERROR: failed to advance current symlink: {exc}", file=sys.stderr)
        # Try cleanup
        if tmp_link.is_symlink():
            try:
                tmp_link.unlink()
            except OSError:
                pass
        return 1

    # ------------------------------------------------------------------
    # 9. Run scene analysis (advisory; never blocks the pipeline)
    # ------------------------------------------------------------------
    analysis_ok = False
    try:
        subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "scene_analysis.py"),
             str(session_dir), "--iter", str(new_k)],
            check=False,
            timeout=60,
        )
        analysis_ok = (session_dir / f"iter_{new_k}" / "analysis.json").exists()
    except Exception:
        pass  # analysis is advisory; don't block the loop

    # ------------------------------------------------------------------
    # 11. Optional vision / semantic evaluation
    # ------------------------------------------------------------------
    if vision:
        classes_json = session_dir / "object_class.json"
        ref_image = session_dir / "reference_image.png"
        if not EVALUATE_SCENE.exists():
            print(
                f"WARNING: --vision requested but evaluate_scene.py not found at "
                f"{EVALUATE_SCENE} — skipping.",
                file=sys.stderr,
            )
        else:
            eval_cmd = [
                sys.executable,
                str(EVALUATE_SCENE),
                str(output_blend),
                str(classes_json),
                str(new_iter_dir),
            ]
            if ref_image.exists():
                eval_cmd += ["--reference-image", str(ref_image)]
            try:
                proc = subprocess.run(
                    eval_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    print(
                        f"WARNING: evaluate_scene.py exited {proc.returncode}: "
                        f"{proc.stderr[-400:]}",
                        file=sys.stderr,
                    )
                else:
                    print("[run_one_op] Vision evaluation complete.")
            except subprocess.TimeoutExpired:
                print("WARNING: vision evaluation timed out.", file=sys.stderr)

    # ------------------------------------------------------------------
    # 12. Print summary JSON to stdout
    # ------------------------------------------------------------------
    summary = {
        "iter": new_k,
        "action": action,
        "obj_name": obj_name,
        "before": before_summary,
        "after": after_summary,
        "render_ok": render_ok,
        "analysis_ok": analysis_ok,
        "scene_blend": str(output_blend),
    }
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _selftest():
    """Validate script syntax and that key helper paths exist (no Blender invoked)."""
    errors = []
    for p in (RUNNER, _SCRIPTS_DIR):
        if not Path(p).exists():
            errors.append(f"MISSING: {p}")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    print("selftest OK")
    print(f"  BLENDER  = {BLENDER}")
    print(f"  RUNNER   = {RUNNER}")
    print(f"  SCRIPTS  = {_SCRIPTS_DIR}")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        _selftest()

    if len(sys.argv) < 3:
        print(
            "Usage: python run_one_op.py <session_dir> <op_json_path> [--vision]",
            file=sys.stderr,
        )
        sys.exit(1)

    session_dir_arg = Path(sys.argv[1]).resolve()
    op_json_arg = Path(sys.argv[2]).resolve()
    do_vision = "--vision" in sys.argv[3:]

    if not session_dir_arg.is_dir():
        print(f"ERROR: session_dir is not a directory: {session_dir_arg}", file=sys.stderr)
        sys.exit(1)

    sys.exit(run(session_dir_arg, op_json_arg, vision=do_vision))
