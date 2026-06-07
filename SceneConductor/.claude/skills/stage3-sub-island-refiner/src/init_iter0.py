#!/usr/bin/env python3
"""init_iter0.py — one-shot baseline setup for the island refiner.

Sets up `<group_dir>/simple_refiner/iter_0/` so the sub-agent loop can start:
    1. apply_delta --init     → iter_0/island.blend (snapshot of group_dir/island.blend)
    2. render_one             → iter_0/render_persp.png, render_bev.png
    3. score_info             → iter_0/info.json (INFO ONLY)
    4. write empty transforms → iter_0/transforms.json = {"iter":0, "members":{}}

No-resume policy: if <group_dir>/simple_refiner/ already exists, it is rotated
to <group_dir>/simple_refiner.bak.<UTC_TIMESTAMP>/ BEFORE creating a fresh iter_0.
Every dispatch is a cold start — there is no cross-dispatch resume.

This script is invoked by the stage3-island-refiner sub-agent at the start of
its iteration loop (default MAX_ITER=20), so the per-iter logic (which reads
iter_(N-1)/...) has a valid iter_0 to begin from.

Usage:
    python init_iter0.py --group-dir <DIR> [--samples 128]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SKILL_DIR        = Path(__file__).parent
APPLY_DELTA      = SKILL_DIR / "apply_delta.py"
RENDER_ONE       = SKILL_DIR / "render_one.py"
SCORE_INFO       = SKILL_DIR / "score_info.py"
COMPUTE_GEOMETRY = SKILL_DIR / "compute_member_geometry.py"
REPO_ROOT        = SKILL_DIR.parent.parent.parent.parent
DIRS_YAML   = REPO_ROOT / "DIRECTORYS.yaml"

# Re-use the same per-iter pose dumper as iter_step.py for consistency.
sys.path.insert(0, str(SKILL_DIR))
from iter_step import _dump_obj_positions  # noqa: E402


def _resolve_blender_bin(cli_arg: str | None) -> str | None:
    """Resolve blender binary from: --blender-bin > $BLENDER > DIRECTORYS.yaml."""
    if cli_arg:
        return cli_arg
    env = os.environ.get("BLENDER")
    if env:
        return env
    if DIRS_YAML.exists():
        try:
            import yaml
            d = yaml.safe_load(DIRS_YAML.read_text()) or {}
            return (d.get(f"blender_bin_{sys.platform[:5]}")  # linux/darwi/win32
                    or d.get("blender_bin_linux")
                    or d.get("blender_bin"))
        except Exception:
            pass
    return None


def _log(msg: str) -> None:
    print(f"[init_iter0] {msg}", file=sys.stderr)


_REQUIRED_SPEC_FIELDS: dict[str, type] = {
    "anchor_role": str,
    "member_count": int,
    "pattern": str,
    "facing": str,
    "spacing": str,
    "clearance_m": float,
}

_PATTERN_ENUM = {"ring", "row", "2+2+1", "2+2+1+1", "L", "T", "cluster", "free"}
_FACING_ENUM = {"toward_anchor", "away_from_anchor", "parallel_to_anchor_long_axis", "mixed"}
_SPACING_ENUM = {"even_along_each_edge", "even_around_anchor", "tight", "loose"}


def _validate_target_spec(group_dir: Path) -> None:
    """Validate target_spec.json exists and contains all required fields with correct types.

    Exits with code 5 on any failure so the refiner dispatch never starts without a
    valid spec.
    """
    spec_path = group_dir / "target_spec.json"
    if not spec_path.exists():
        print(
            f"[init_iter0] ERROR: target_spec.json missing at {spec_path}"
            " — refiner dispatch cannot proceed",
            file=sys.stderr,
        )
        sys.exit(5)

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[init_iter0] ERROR: target_spec.json schema violation: cannot parse JSON: {exc}",
              file=sys.stderr)
        sys.exit(5)

    # Check all required keys are present with correct types.
    for field, expected_type in _REQUIRED_SPEC_FIELDS.items():
        if field not in spec:
            print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                  f" missing required field '{field}'",
                  file=sys.stderr)
            sys.exit(5)
        val = spec[field]
        # member_count must be int; clearance_m accepts int or float (both are numeric).
        if field == "clearance_m":
            if not isinstance(val, (int, float)):
                print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                      f" '{field}' must be a number, got {type(val).__name__}",
                      file=sys.stderr)
                sys.exit(5)
            if val < 0.0:
                print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                      f" 'clearance_m' must be >= 0.0, got {val}",
                      file=sys.stderr)
                sys.exit(5)
        elif field == "member_count":
            if not isinstance(val, int) or isinstance(val, bool):
                print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                      f" 'member_count' must be an int, got {type(val).__name__}",
                      file=sys.stderr)
                sys.exit(5)
            if val < 1:
                print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                      f" 'member_count' must be >= 1, got {val}",
                      file=sys.stderr)
                sys.exit(5)
        else:
            if not isinstance(val, expected_type):
                print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                      f" '{field}' must be {expected_type.__name__},"
                      f" got {type(val).__name__}",
                      file=sys.stderr)
                sys.exit(5)

    # Validate enum fields.
    pattern = spec["pattern"]
    if pattern not in _PATTERN_ENUM:
        print(f"[init_iter0] ERROR: target_spec.json schema violation:"
              f" 'pattern' value '{pattern}' not in allowed enum"
              f" {sorted(_PATTERN_ENUM)}",
              file=sys.stderr)
        sys.exit(5)
    facing = spec["facing"]
    if facing not in _FACING_ENUM:
        print(f"[init_iter0] ERROR: target_spec.json schema violation:"
              f" 'facing' value '{facing}' not in allowed enum"
              f" {sorted(_FACING_ENUM)}",
              file=sys.stderr)
        sys.exit(5)
    spacing = spec["spacing"]
    if spacing not in _SPACING_ENUM:
        print(f"[init_iter0] ERROR: target_spec.json schema violation:"
              f" 'spacing' value '{spacing}' not in allowed enum"
              f" {sorted(_SPACING_ENUM)}",
              file=sys.stderr)
        sys.exit(5)

    # free_note required when pattern="free" or facing="mixed".
    if pattern == "free" or facing == "mixed":
        if not spec.get("free_note", "").strip():
            print(f"[init_iter0] ERROR: target_spec.json schema violation:"
                  f" 'free_note' is required when pattern='free' or facing='mixed'",
                  file=sys.stderr)
            sys.exit(5)

    count = spec["member_count"]
    print(f"[init_iter0] target_spec.json OK (pattern={pattern}, count={count})",
          file=sys.stderr)


def _run(cmd: list[str], step: str) -> None:
    _log(f"=== {step} ===")
    _log(f"cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        _log(f"FAILED at step '{step}' (exit {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline iter_0 setup")
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128,
                        help="Cycles sample count (default 128)")
    parser.add_argument("--blender-bin", type=str, default=None,
                        help="Override Blender executable path (forwarded to sub-scripts).")
    args = parser.parse_args()

    group_dir = args.group_dir.resolve()
    if not group_dir.is_dir():
        _log(f"ERROR: group_dir not found: {group_dir}")
        sys.exit(1)
    for needed in ("island.blend", "metadata.json"):
        if not (group_dir / needed).is_file():
            _log(f"ERROR: required input missing: {group_dir / needed}")
            sys.exit(1)

    # Validate target_spec.json (required — refiner cannot proceed without it).
    _validate_target_spec(group_dir)

    # No-resume policy: rotate any existing simple_refiner/ to a timestamped backup.
    simple_refiner_dir = group_dir / "simple_refiner"
    if simple_refiner_dir.exists():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = group_dir / f"simple_refiner.bak.{ts}"
        simple_refiner_dir.rename(backup_dir)
        print(f"[init_iter0] rotated existing simple_refiner -> simple_refiner.bak.{ts}")
    else:
        print("[init_iter0] fresh simple_refiner/")

    iter0_dir = group_dir / "simple_refiner" / "iter_0"
    iter0_dir.mkdir(parents=True, exist_ok=True)

    # 1. apply_delta --init (snapshot)
    _run(
        [sys.executable, str(APPLY_DELTA),
         "--group-dir", str(group_dir),
         "--iter", "0", "--init"],
        "apply_delta --init",
    )

    # 2. render baseline
    _run(
        [sys.executable, str(RENDER_ONE),
         "--blend", str(iter0_dir / "island.blend"),
         "--out-dir", str(iter0_dir),
         "--metadata", str(group_dir / "metadata.json"),
         "--samples", str(args.samples)],
        "render_one",
    )

    # 3. score baseline (INFO ONLY)
    _run(
        [sys.executable, str(SCORE_INFO),
         "--iter-dir", str(iter0_dir)],
        "score_info",
    )

    # 4. write empty transforms.json (iter_1's "previous delta")
    empty = {
        "iter": 0,
        "members": {},
        "_note": "Baseline; no transforms applied yet. Read by iter_1 as 'previous delta'.",
    }
    (iter0_dir / "transforms.json").write_text(
        json.dumps(empty, indent=2), encoding="utf-8"
    )

    # 5. dump baseline absolute poses → iter_0/current_state.json
    blender_for_dump = _resolve_blender_bin(getattr(args, "blender_bin", None))
    if blender_for_dump:
        raw = _dump_obj_positions(iter0_dir / "island.blend", blender_for_dump)
        if raw:
            obj_positions = {
                k: {"position": [v[0], v[1], v[2]],
                    "yaw_deg": round(math.degrees(v[3]), 4)}
                for k, v in raw.items()
            }
            state_out = {
                "iter": 0,
                "frame": "canonical",
                "_note": ("Baseline absolute object poses at iter 0. "
                          "position in metres, yaw_deg is rotation about Z."),
                "obj_positions": obj_positions,
            }
            (iter0_dir / "current_state.json").write_text(
                json.dumps(state_out, indent=2), encoding="utf-8"
            )

    # 6. compute_member_geometry: populate forward_axes.json (cached) + iter_0/geometry.json
    _log("=== compute_member_geometry (iter 0) ===")
    blender_bin_resolved = _resolve_blender_bin(args.blender_bin)
    geom_cmd = [
        sys.executable, str(COMPUTE_GEOMETRY),
        "--group-dir", str(group_dir),
        "--iter", "0",
    ]
    if blender_bin_resolved:
        geom_cmd += ["--blender-bin", blender_bin_resolved]
    result = subprocess.run(geom_cmd, text=True)
    if result.returncode != 0:
        _log(f"WARNING: compute_member_geometry exited with code {result.returncode} "
             f"— geometry.json may be missing for iter_0, continuing anyway.")
    else:
        geom_path = iter0_dir / "geometry.json"
        if geom_path.is_file():
            _log(f"geometry.json written at {geom_path}")
        else:
            _log("WARNING: compute_member_geometry succeeded but geometry.json not found.")

    _log(f"DONE — baseline ready at {iter0_dir}")


if __name__ == "__main__":
    main()
