#!/usr/bin/env python3
"""
verify_roundtrip.py — JSON ↔ .blend round-trip fidelity verifier.

Proves that blender_scene.json is a complete, bidirectional serialization of
blender_scene.blend by:
  1. Opening the final blender_scene.blend from a scene_dir.
  2. Re-extracting every block into a fresh blender_scene.roundtrip.json.
  3. Rebuilding a new roundtrip.blend from that JSON.
  4. Diffing invariants between original and rebuilt .blends.
  5. Emitting roundtrip_report.json in the scene_dir.

Exit codes:
  0  PASS  — all invariants within tolerance
  1  FAIL  — one or more invariants exceed tolerance
  2  INFRA_ERROR — subprocess failure or missing inputs

Usage:
  python verify_roundtrip.py <scene_dir> \\
      [--blender /path/to/blender] \\
      [--output roundtrip_report.json] \\
      [--keep-tmp]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# ── DIRECTORYS.yaml (canonical machine-specific paths) ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())

LOG = "[verify_roundtrip]"

# ---------------------------------------------------------------------------
# Tolerances (mirrors the invariant table in the spec)
# ---------------------------------------------------------------------------

# All tolerances applied in _apply_tolerances().
TOLERANCES = {
    # mesh_object_count: exact (0)
    "mesh_object_count": 0,
    # per-mesh location: 1 mm per axis
    "mesh_location": 0.001,
    # per-mesh rotation_euler: 0.5 degrees per axis
    "mesh_rotation": math.radians(0.5),
    # per-mesh scale: 0.1% relative
    "mesh_scale_rel": 0.001,
    # camera location: 1 mm
    "camera_location": 0.001,
    # camera rotation_euler: 0.1 degrees
    "camera_rotation": math.radians(0.1),
    # camera lens: 0.01 mm
    "camera_lens": 0.01,
    # camera sensor_width: 0.01 mm
    "camera_sensor_width": 0.01,
    # light count by type: exact
    "light_count": 0,
    # total light energy per type: 1% relative
    "light_energy_rel": 0.01,
    # world strength: 1% relative
    "world_strength_rel": 0.01,
    # world mode: exact
    "world_mode": 0,
    # stage material base_color L2 distance
    "stage_material_color_l2": 0.01,
    # stage material roughness: 0.01 absolute
    "stage_material_roughness": 0.01,
    # stage polygon_vertices count: exact
    "stage_polygon_count": 0,
    # stage polygon_vertices XY: 1 mm
    "stage_polygon_xy": 0.001,
    # stage floor_z / ceiling_z: 1 mm
    "stage_z": 0.001,
    # openings count: exact
    "stage_openings_count": 0,
    # per-opening xy_range: 1 mm
    "stage_openings_xy": 0.001,
    # per-opening z_range: 1 mm
    "stage_openings_z": 0.001,
    # render samples: exact
    "render_samples": 0,
    # render resolution: exact
    "render_resolution": 0,
    # render engine: exact
    "render_engine": 0,
    # point_cloud num_vertices: exact
    "point_cloud_num_vertices": 0,
}

# Factor above which a failure is escalated to "critical"
CRITICAL_MULTIPLIER = 5.0


# ---------------------------------------------------------------------------
# Resolve Blender binary
# ---------------------------------------------------------------------------

def resolve_blender(override: str | None) -> str:
    if override:
        return override
    env_bin = os.environ.get("BLENDER_BIN") or os.environ.get("BLENDER")
    if env_bin:
        return env_bin
    # Fall back to the canonical binary path from DIRECTORYS.yaml,
    # resolved relative to the repo root if it's a relative path.
    yaml_bin = _DIRS.get("blender_bin")
    if yaml_bin:
        yaml_path = Path(yaml_bin)
        if not yaml_path.is_absolute():
            yaml_path = (_REPO_ROOT / yaml_path).resolve()
        if yaml_path.exists():
            return str(yaml_path)
    found = shutil.which("blender")
    if found:
        return found
    raise FileNotFoundError(
        f"{LOG} Could not find Blender binary. "
        "Set $BLENDER_BIN or pass --blender."
    )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_blender(
    blender_bin: str,
    blend_path: str | None,
    script: str,
    script_args: list[str],
    *,
    label: str,
) -> tuple[int, str, str]:
    """Run a Blender subprocess headlessly.

    Returns (returncode, stdout, stderr).
    blend_path may be None for --background with no file.
    """
    cmd = [blender_bin, "--background"]
    if blend_path:
        cmd.append(blend_path)
    cmd += ["--python", script, "--"] + script_args

    print(f"{LOG} [{label}] Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically via a sibling tempfile + rename."""
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Invariant diff helpers
# ---------------------------------------------------------------------------

def _tol_label(tol_value: float | int, tol_type: str) -> str:
    """Human-readable tolerance string."""
    if tol_value == 0:
        return "exact"
    if "rel" in tol_type:
        return f"{tol_value * 100:.2f}% relative"
    if "rotation" in tol_type:
        return f"{math.degrees(tol_value):.3f}°"
    return f"{tol_value:.4f} m"


def _make_entry(
    name: str,
    expected,
    observed,
    tol_key: str,
    delta: float | None = None,
) -> dict:
    """Build an invariant entry dict."""
    tol_value = TOLERANCES.get(tol_key, 0)
    tol_str = _tol_label(tol_value, tol_key)

    if delta is None:
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            delta = abs(float(observed) - float(expected))
        else:
            delta = 0.0 if expected == observed else float("inf")

    if tol_value == 0:
        passing = (expected == observed)
    else:
        passing = (delta <= tol_value)

    entry = {
        "name": name,
        "expected": expected,
        "observed": observed,
        "tolerance": tol_str,
        "delta": round(delta, 8) if isinstance(delta, float) else delta,
        "pass": passing,
    }

    # Escalate to critical if delta > 5× tolerance (only for non-exact)
    if not passing and tol_value > 0 and delta > CRITICAL_MULTIPLIER * tol_value:
        entry["severity"] = "critical"
    elif not passing:
        entry["severity"] = "major"

    return entry


def apply_tolerances(raw_diff: dict) -> list[dict]:
    """
    Convert the raw diff dict from _roundtrip_diff.py into a list of
    invariant entries with pass/fail and severity.

    raw_diff keys follow the flat namespace used in _roundtrip_diff.py.
    """
    entries: list[dict] = []

    def add(name: str, expected, observed, tol_key: str, delta: float | None = None):
        entries.append(_make_entry(name, expected, observed, tol_key, delta))

    # --- mesh_object_count ---
    if "mesh_object_count" in raw_diff:
        d = raw_diff["mesh_object_count"]
        add("mesh_object_count", d["expected"], d["observed"], "mesh_object_count")

    # --- per-mesh transforms ---
    for mesh_entry in raw_diff.get("mesh_objects", []):
        name = mesh_entry["name"]
        for axis_i, axis in enumerate(["x", "y", "z"]):
            if "location" in mesh_entry:
                exp = mesh_entry["location"]["expected"][axis_i]
                obs = mesh_entry["location"]["observed"][axis_i]
                add(f"mesh.{name}.location.{axis}", exp, obs, "mesh_location")
            if "rotation_euler" in mesh_entry:
                exp = mesh_entry["rotation_euler"]["expected"][axis_i]
                obs = mesh_entry["rotation_euler"]["observed"][axis_i]
                add(f"mesh.{name}.rotation_euler.{axis}", exp, obs, "mesh_rotation")
            if "scale" in mesh_entry:
                exp = mesh_entry["scale"]["expected"][axis_i]
                obs = mesh_entry["scale"]["observed"][axis_i]
                # relative scale delta
                rel_delta = abs(obs - exp) / max(abs(exp), 1e-8)
                add(f"mesh.{name}.scale.{axis}", exp, obs, "mesh_scale_rel", rel_delta)

    # --- camera ---
    cam = raw_diff.get("camera", {})
    for axis_i, axis in enumerate(["x", "y", "z"]):
        if "location" in cam:
            add(f"camera.location.{axis}", cam["location"]["expected"][axis_i],
                cam["location"]["observed"][axis_i], "camera_location")
        if "rotation_euler" in cam:
            add(f"camera.rotation_euler.{axis}", cam["rotation_euler"]["expected"][axis_i],
                cam["rotation_euler"]["observed"][axis_i], "camera_rotation")
    if "lens" in cam:
        add("camera.lens", cam["lens"]["expected"], cam["lens"]["observed"], "camera_lens")
    if "sensor_width" in cam:
        add("camera.sensor_width", cam["sensor_width"]["expected"],
            cam["sensor_width"]["observed"], "camera_sensor_width")

    # --- lights ---
    for light_type, d in raw_diff.get("light_counts", {}).items():
        add(f"light_count.{light_type}", d["expected"], d["observed"], "light_count")
    for light_type, d in raw_diff.get("light_energy", {}).items():
        exp = d["expected"]
        obs = d["observed"]
        rel_delta = abs(obs - exp) / max(abs(exp), 1e-8) if exp != 0 else (0.0 if obs == 0 else float("inf"))
        add(f"light_energy.{light_type}", round(exp, 4), round(obs, 4),
            "light_energy_rel", rel_delta)

    # --- world ---
    world = raw_diff.get("world", {})
    if "mode" in world:
        add("world.mode", world["mode"]["expected"], world["mode"]["observed"], "world_mode")
    if "world_strength" in world:
        exp = world["world_strength"]["expected"]
        obs = world["world_strength"]["observed"]
        rel_delta = abs(obs - exp) / max(abs(exp), 1e-8) if exp != 0 else (0.0 if obs == 0 else float("inf"))
        add("world.world_strength", round(exp, 6), round(obs, 6), "world_strength_rel", rel_delta)

    # --- stage_materials ---
    for role, mat_d in raw_diff.get("stage_materials", {}).items():
        if "base_color" in mat_d:
            l2 = mat_d["base_color"].get("l2_delta", 0.0)
            exp_col = mat_d["base_color"].get("expected", [])
            obs_col = mat_d["base_color"].get("observed", [])
            add(f"stage_material.{role}.base_color", exp_col, obs_col,
                "stage_material_color_l2", l2)
        if "roughness" in mat_d:
            add(f"stage_material.{role}.roughness",
                mat_d["roughness"]["expected"], mat_d["roughness"]["observed"],
                "stage_material_roughness")

    # --- stage polygon ---
    stage = raw_diff.get("stage", {})
    if "polygon_vertices_count" in stage:
        add("stage.polygon_vertices_count",
            stage["polygon_vertices_count"]["expected"],
            stage["polygon_vertices_count"]["observed"],
            "stage_polygon_count")
    for vi, vd in enumerate(stage.get("polygon_vertices", [])):
        for axis_i, axis in enumerate(["x", "y"]):
            exp = vd["expected"][axis_i]
            obs = vd["observed"][axis_i]
            add(f"stage.polygon_vertex[{vi}].{axis}", exp, obs, "stage_polygon_xy")
    if "floor_z" in stage:
        add("stage.floor_z", stage["floor_z"]["expected"],
            stage["floor_z"]["observed"], "stage_z")
    if "ceiling_z" in stage:
        add("stage.ceiling_z", stage["ceiling_z"]["expected"],
            stage["ceiling_z"]["observed"], "stage_z")
    if "openings_count" in stage:
        add("stage.openings_count", stage["openings_count"]["expected"],
            stage["openings_count"]["observed"], "stage_openings_count")
    for oi, od in enumerate(stage.get("openings", [])):
        oid = od.get("id", str(oi))
        for xy_i, xy in enumerate(["x0", "y0", "x1", "y1"]):
            axis_i = xy_i % 2
            pt_i = xy_i // 2
            exp = od["xy_range"]["expected"][pt_i][axis_i]
            obs = od["xy_range"]["observed"][pt_i][axis_i]
            add(f"stage.opening[{oid}].xy_range.{xy}", exp, obs, "stage_openings_xy")
        for zi, zk in enumerate(["z0", "z1"]):
            add(f"stage.opening[{oid}].z_range.{zk}",
                od["z_range"]["expected"][zi], od["z_range"]["observed"][zi],
                "stage_openings_z")

    # --- render ---
    render = raw_diff.get("render", {})
    if "engine" in render:
        add("render.engine", render["engine"]["expected"],
            render["engine"]["observed"], "render_engine")
    if "samples" in render:
        add("render.samples", render["samples"]["expected"],
            render["samples"]["observed"], "render_samples")
    if "resolution" in render:
        add("render.resolution", render["resolution"]["expected"],
            render["resolution"]["observed"], "render_resolution",
            delta=0 if render["resolution"]["expected"] == render["resolution"]["observed"] else 1)

    # --- point_cloud ---
    pc = raw_diff.get("point_cloud", {})
    if "num_vertices" in pc:
        add("point_cloud.num_vertices", pc["num_vertices"]["expected"],
            pc["num_vertices"]["observed"], "point_cloud_num_vertices")

    return entries


def sort_invariants(entries: list[dict]) -> list[dict]:
    """Sort: failed first, then by name."""
    return sorted(entries, key=lambda e: (e["pass"], e["name"]))


def build_summary(entries: list[dict]) -> tuple[str, str]:
    """Return (status, summary_str)."""
    n_total = len(entries)
    n_fail = sum(1 for e in entries if not e["pass"])
    n_major = sum(1 for e in entries if not e["pass"] and e.get("severity") in ("major", "critical"))
    n_ok = n_total - n_fail

    if n_fail == 0:
        status = "PASS"
    else:
        status = "FAIL"

    summary = f"{n_ok} invariants OK, {n_fail} failed (majors: {n_major})"
    return status, summary


# ---------------------------------------------------------------------------
# Locate helper scripts alongside this file
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).parent.resolve()
_BUILD_PY = (
    Path(__file__).parent.parent.parent  # .claude/skills/
    / "stage2-sub-pointmap-to-separable-stage" / "src" / "build.py"
).resolve()
_EXTRACT_PY = _SCRIPTS_DIR / "_roundtrip_extract.py"
_DIFF_PY = _SCRIPTS_DIR / "_roundtrip_diff.py"


# ---------------------------------------------------------------------------
# Repo root (for tmp/ convention from CLAUDE.md)
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Walk up from start looking for a CLAUDE.md or .git; fall back to start."""
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / "CLAUDE.md").exists() or (parent / ".git").exists():
            return parent
    return start.resolve()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scene_dir", help="Scene folder containing blender_scene.blend + blender_scene.json")
    parser.add_argument("--blender", default=None, help="Override Blender binary path")
    parser.add_argument("--output", default="roundtrip_report.json",
                        help="Report filename in scene_dir (default: roundtrip_report.json)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep the tmp/<scene>_roundtrip/ workspace after completion")
    args = parser.parse_args(argv)

    scene_dir = Path(args.scene_dir).resolve()
    report_path = scene_dir / args.output
    timings: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Step 1: Validate inputs
    # -----------------------------------------------------------------------
    _blend_canonical = scene_dir / "blend" / "blender_scene.blend"
    _blend_legacy = scene_dir / "blender_scene.blend"
    if _blend_canonical.exists():
        original_blend = _blend_canonical
    elif _blend_legacy.exists():
        print(f"[legacy-path] reading {_blend_legacy}; canonical is {_blend_canonical}")
        original_blend = _blend_legacy
    else:
        original_blend = _blend_canonical  # let downstream fail with clearer error

    _json_canonical = scene_dir / "json" / "blender_scene.json"
    _json_legacy = scene_dir / "blender_scene.json"
    if _json_canonical.exists():
        source_json = _json_canonical
    elif _json_legacy.exists():
        print(f"[legacy-path] reading {_json_legacy}; canonical is {_json_canonical}")
        source_json = _json_legacy
    else:
        source_json = _json_canonical  # let downstream fail with clearer error

    missing = []
    if not original_blend.exists():
        missing.append(str(original_blend))
    if not source_json.exists():
        missing.append(str(source_json))

    if missing:
        err_report = {
            "scene_dir": str(scene_dir),
            "status": "INFRA_ERROR",
            "summary": f"Missing required files: {missing}",
            "invariants": [],
            "missing_blocks_in_original": [],
            "extra_blocks_in_roundtrip": [],
            "paths": {},
            "timings": {},
            "error": f"Missing: {missing}",
        }
        atomic_write_json(report_path, err_report)
        print(f"{LOG} INFRA_ERROR: missing inputs: {missing}", file=sys.stderr)
        return 2

    try:
        blender_bin = resolve_blender(args.blender)
    except FileNotFoundError as exc:
        err_report = {
            "scene_dir": str(scene_dir),
            "status": "INFRA_ERROR",
            "summary": str(exc),
            "invariants": [],
            "missing_blocks_in_original": [],
            "extra_blocks_in_roundtrip": [],
            "paths": {},
            "timings": {},
            "error": str(exc),
        }
        atomic_write_json(report_path, err_report)
        print(f"{LOG} INFRA_ERROR: {exc}", file=sys.stderr)
        return 2

    # -----------------------------------------------------------------------
    # Step 2: Make tmp dir: repo_root/tmp/<scene_name>_roundtrip/
    # -----------------------------------------------------------------------
    repo_root = find_repo_root(scene_dir)
    scene_name = scene_dir.name
    tmp_dir = repo_root / "tmp" / f"{scene_name}_roundtrip"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"{LOG} Workspace: {tmp_dir}")

    try:
        # -----------------------------------------------------------------------
        # Step 3: Copy inputs into tmp/ — never touch scene_dir originals
        # -----------------------------------------------------------------------
        original_blend_tmp = tmp_dir / "original.blend"
        roundtrip_json = tmp_dir / "blender_scene.roundtrip.json"
        roundtrip_blend = tmp_dir / "roundtrip.blend"
        diff_json = tmp_dir / "diff.json"

        print(f"{LOG} Copying original.blend ...")
        shutil.copy2(str(original_blend), str(original_blend_tmp))

        print(f"{LOG} Copying blender_scene.json -> blender_scene.roundtrip.json ...")
        shutil.copy2(str(source_json), str(roundtrip_json))

        # -----------------------------------------------------------------------
        # Step 4: Extract — re-populate roundtrip JSON from live blend state
        # -----------------------------------------------------------------------
        print(f"{LOG} Step 4: Extracting blocks from blend into roundtrip JSON ...")
        t0 = time.monotonic()
        rc, stdout, stderr = run_blender(
            blender_bin,
            str(original_blend_tmp),
            str(_EXTRACT_PY),
            [
                "--scene-json", str(roundtrip_json),
                "--scene-dir", str(scene_dir),
            ],
            label="extract",
        )
        timings["extract_s"] = round(time.monotonic() - t0, 2)
        print(f"{LOG} Extract stdout tail:\n" + "\n".join(stdout.splitlines()[-20:]))
        if rc != 0:
            stderr_tail = "\n".join(stderr.splitlines()[-40:])
            err_report = _infra_error_report(
                scene_dir, report_path,
                f"Extract step failed (rc={rc})", stderr_tail, timings
            )
            atomic_write_json(report_path, err_report)
            print(f"{LOG} INFRA_ERROR: extract failed (rc={rc})", file=sys.stderr)
            print(stderr_tail, file=sys.stderr)
            return 2

        # -----------------------------------------------------------------------
        # Step 5: Verify roundtrip JSON was written
        # -----------------------------------------------------------------------
        if not roundtrip_json.exists():
            err_report = _infra_error_report(
                scene_dir, report_path,
                "Extract step did not produce roundtrip JSON", stderr, timings
            )
            atomic_write_json(report_path, err_report)
            return 2

        # -----------------------------------------------------------------------
        # Step 6: Rebuild — build a fresh .blend from the roundtrip JSON
        # -----------------------------------------------------------------------
        if not _BUILD_PY.exists():
            err_report = _infra_error_report(
                scene_dir, report_path,
                f"build.py not found at {_BUILD_PY}", "", timings
            )
            atomic_write_json(report_path, err_report)
            return 2

        print(f"{LOG} Step 5: Rebuilding roundtrip.blend from roundtrip JSON ...")
        t0 = time.monotonic()
        rc, stdout, stderr = run_blender(
            blender_bin,
            None,
            str(_BUILD_PY),
            [
                "--input", str(roundtrip_json),
                "--output", str(roundtrip_blend),
            ],
            label="rebuild",
        )
        timings["rebuild_s"] = round(time.monotonic() - t0, 2)
        print(f"{LOG} Rebuild stdout tail:\n" + "\n".join(stdout.splitlines()[-20:]))
        if rc != 0:
            stderr_tail = "\n".join(stderr.splitlines()[-40:])
            err_report = _infra_error_report(
                scene_dir, report_path,
                f"Rebuild step failed (rc={rc})", stderr_tail, timings
            )
            atomic_write_json(report_path, err_report)
            print(f"{LOG} INFRA_ERROR: rebuild failed (rc={rc})", file=sys.stderr)
            print(stderr_tail, file=sys.stderr)
            return 2

        if not roundtrip_blend.exists():
            err_report = _infra_error_report(
                scene_dir, report_path,
                "Rebuild step did not produce roundtrip.blend", stderr, timings
            )
            atomic_write_json(report_path, err_report)
            return 2

        # -----------------------------------------------------------------------
        # Step 7: Diff — extract invariants from both blends and compare
        # -----------------------------------------------------------------------
        print(f"{LOG} Step 6: Diffing invariants ...")
        t0 = time.monotonic()
        rc, stdout, stderr = run_blender(
            blender_bin,
            None,
            str(_DIFF_PY),
            [
                "--original", str(original_blend_tmp),
                "--roundtrip", str(roundtrip_blend),
                "--out", str(diff_json),
            ],
            label="diff",
        )
        timings["diff_s"] = round(time.monotonic() - t0, 2)
        print(f"{LOG} Diff stdout tail:\n" + "\n".join(stdout.splitlines()[-20:]))
        if rc != 0:
            stderr_tail = "\n".join(stderr.splitlines()[-40:])
            err_report = _infra_error_report(
                scene_dir, report_path,
                f"Diff step failed (rc={rc})", stderr_tail, timings
            )
            atomic_write_json(report_path, err_report)
            print(f"{LOG} INFRA_ERROR: diff failed (rc={rc})", file=sys.stderr)
            print(stderr_tail, file=sys.stderr)
            return 2

        if not diff_json.exists():
            err_report = _infra_error_report(
                scene_dir, report_path,
                "Diff step did not produce diff.json", stderr, timings
            )
            atomic_write_json(report_path, err_report)
            return 2

        # -----------------------------------------------------------------------
        # Step 8: Apply tolerances and build report
        # -----------------------------------------------------------------------
        with open(str(diff_json), "r", encoding="utf-8") as fh:
            raw_diff = json.load(fh)

        entries = apply_tolerances(raw_diff)
        entries = sort_invariants(entries)
        status, summary = build_summary(entries)

        missing_blocks = raw_diff.get("missing_blocks_in_original", [])
        extra_blocks = raw_diff.get("extra_blocks_in_roundtrip", [])

        report = {
            "scene_dir": str(scene_dir),
            "status": status,
            "summary": summary,
            "invariants": entries,
            "missing_blocks_in_original": missing_blocks,
            "extra_blocks_in_roundtrip": extra_blocks,
            "paths": {
                "original_blend": str(original_blend),
                "roundtrip_blend": str(roundtrip_blend),
                "roundtrip_json": str(roundtrip_json),
            },
            "timings": timings,
        }

        atomic_write_json(report_path, report)

        # -----------------------------------------------------------------------
        # Step 9: Cleanup (unless --keep-tmp)
        # -----------------------------------------------------------------------
        if not args.keep_tmp:
            print(f"{LOG} Cleaning up tmp dir: {tmp_dir}")
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        # -----------------------------------------------------------------------
        # Step 10: Print result and exit
        # -----------------------------------------------------------------------
        n_fail = sum(1 for e in entries if not e["pass"])
        print(
            f"{LOG} {status}: {summary} | "
            f"extract={timings.get('extract_s', '?')}s "
            f"rebuild={timings.get('rebuild_s', '?')}s "
            f"diff={timings.get('diff_s', '?')}s | "
            f"report={report_path}"
        )

        if status == "FAIL":
            print(f"{LOG} Failed invariants:")
            for e in entries:
                if not e["pass"]:
                    sev = e.get("severity", "")
                    print(f"  [{sev.upper()}] {e['name']}: expected={e['expected']} "
                          f"observed={e['observed']} delta={e['delta']} tol={e['tolerance']}")

        return 0 if status == "PASS" else 1

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        err_report = _infra_error_report(
            scene_dir, report_path,
            f"Unexpected exception: {exc}", tb, timings
        )
        atomic_write_json(report_path, err_report)
        print(f"{LOG} INFRA_ERROR: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        return 2


def _infra_error_report(
    scene_dir: Path,
    report_path: Path,
    message: str,
    stderr_tail: str,
    timings: dict,
) -> dict:
    return {
        "scene_dir": str(scene_dir),
        "status": "INFRA_ERROR",
        "summary": message,
        "invariants": [],
        "missing_blocks_in_original": [],
        "extra_blocks_in_roundtrip": [],
        "paths": {
            "original_blend": str(original_blend),
            "roundtrip_blend": "",
            "roundtrip_json": "",
        },
        "timings": timings,
        "error": message,
        "stderr_tail": stderr_tail,
    }


if __name__ == "__main__":
    sys.exit(main())
