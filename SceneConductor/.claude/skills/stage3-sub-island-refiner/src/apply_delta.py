"""apply_delta.py — Apply per-object transforms to an island.blend.

No gates, no clamps, no anchor locks, no guards of any kind.
Delta is applied exactly as specified. If the delta is 100 m, so be it.

Usage:
    # Initialization (iter 0): just copy island.blend, no transforms.
    python apply_delta.py --group-dir DIR --iter 0 --init

    # Subsequent iters: read transforms.json, apply, write new island.blend.
    python apply_delta.py --group-dir DIR --iter N [--blender-bin PATH]

Inputs:
    --init:  DIR/island.blend  →  DIR/simple_refiner/iter_0/island.blend  (copy)
    N >= 1:  DIR/simple_refiner/iter_(N-1)/island.blend  +
             DIR/simple_refiner/iter_N/transforms.json
          →  DIR/simple_refiner/iter_N/island.blend

transforms.json schema:
    {
      "iter": N,
      "members": {
        "obj_3": {"delta_xyz": [dx, dy, dz], "delta_yaw_deg": d},
        "obj_5": {"delta_xyz": [0, 0, 0],    "delta_yaw_deg": 90}
      }
    }

Missing fields default to delta_xyz=[0,0,0], delta_yaw_deg=0.
Objects absent from "members" are left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# .claude/skills/stage3-sub-island-simple/src/apply_delta.py
# parents: [0]=src  [1]=stage3-sub-island-simple  [2]=skills  [3]=.claude  [4]=repo-root
_REPO_ROOT = _THIS_FILE.parents[4]
_DIRS_YAML = _REPO_ROOT / "DIRECTORYS.yaml"


def _read_blender_bin_from_yaml(yaml_path: Path) -> str | None:
    """Return blender_bin value from DIRECTORYS.yaml without requiring PyYAML."""
    if not yaml_path.is_file():
        return None
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(yaml_path.read_text())
        return str(data.get("blender_bin", "")) or None
    except Exception:
        pass
    for line in yaml_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("blender_bin:") and not stripped.startswith("blender_bin_"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                return value
    return None


def _resolve_blender(blender_bin_arg: str | None) -> str:
    if blender_bin_arg:
        candidate = Path(blender_bin_arg)
        if not candidate.is_absolute():
            candidate = (_REPO_ROOT / candidate).resolve()
        return str(candidate)
    raw = _read_blender_bin_from_yaml(_DIRS_YAML)
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (_REPO_ROOT / candidate).resolve()
        return str(candidate)
    env_val = os.environ.get("BLENDER", "").strip()
    if env_val:
        return env_val
    return "blender"


# ---------------------------------------------------------------------------
# Blender inner script (written to a temp file and passed via -P)
# ---------------------------------------------------------------------------

_INNER_SCRIPT = '''
import bpy
import sys
import json
from mathutils import Vector
from math import radians

# Parse args after "--"
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--transforms", required=True)
ap.add_argument("--output",     required=True)
args = ap.parse_args(argv)

with open(args.transforms) as f:
    data = json.load(f)

members = data.get("members", {})

for obj_id, delta in members.items():
    obj = bpy.data.objects.get(obj_id)
    if obj is None:
        print(f"[apply_delta_inner] WARNING: object not found: {obj_id}", file=sys.stderr)
        continue

    dxyz = delta.get("delta_xyz", [0.0, 0.0, 0.0])
    dyaw = delta.get("delta_yaw_deg", 0.0)

    # Apply delta unconditionally — no clamp, no guard.
    obj.location += Vector((float(dxyz[0]), float(dxyz[1]), float(dxyz[2])))
    obj.rotation_euler.z += radians(float(dyaw))

    print(f"[apply_delta_inner] {obj_id}: "
          f"delta_xyz={dxyz}  delta_yaw_deg={dyaw}  "
          f"new_loc={list(obj.location)}  "
          f"new_rot_z_deg={obj.rotation_euler.z * (180 / 3.14159265358979):.3f}")

bpy.ops.wm.save_as_mainfile(filepath=args.output)
print(f"[apply_delta_inner] Saved to {args.output}")
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="apply_delta",
        description=(
            "Apply per-object transforms to an island.blend (no gates, no clamps). "
            "With --init: copy island.blend to iter_0/. "
            "Without --init: apply iter_N/transforms.json to iter_(N-1)/island.blend."
        ),
    )
    ap.add_argument(
        "--group-dir",
        required=True,
        metavar="DIR",
        dest="group_dir",
        help="Path to the relation group directory containing island.blend and metadata.json.",
    )
    ap.add_argument(
        "--iter",
        required=True,
        type=int,
        metavar="N",
        help="Iteration number (0 with --init = snapshot; N>=1 = apply transforms).",
    )
    ap.add_argument(
        "--init",
        action="store_true",
        help="Initialize iter_0: copy island.blend without applying any transforms.",
    )
    ap.add_argument(
        "--blender-bin",
        default=None,
        metavar="PATH",
        dest="blender_bin",
        help="Path to Blender executable. Falls back to DIRECTORYS.yaml → $BLENDER → 'blender'.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    group_dir = Path(args.group_dir).resolve()
    iter_n = args.iter
    simple_refiner_dir = group_dir / "simple_refiner"

    if args.init:
        # ------------------------------------------------------------------
        # Initialization: copy island.blend to iter_0/island.blend.
        # No transforms applied.
        # ------------------------------------------------------------------
        if iter_n != 0:
            print("ERROR: --init is only valid with --iter 0.", file=sys.stderr)
            sys.exit(1)

        src_blend = group_dir / "island.blend"
        if not src_blend.is_file():
            print(f"ERROR: source island.blend not found: {src_blend}", file=sys.stderr)
            sys.exit(1)

        out_dir = simple_refiner_dir / "iter_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        dst_blend = out_dir / "island.blend"

        shutil.copy2(src_blend, dst_blend)
        print(f"[apply_delta] --init: copied {src_blend} -> {dst_blend}")
        return

    # ----------------------------------------------------------------------
    # Normal iter (N >= 1): apply transforms to iter_(N-1)/island.blend.
    # ----------------------------------------------------------------------
    if iter_n < 1:
        print("ERROR: --iter must be >= 1 when --init is not specified.", file=sys.stderr)
        sys.exit(1)

    input_blend = simple_refiner_dir / f"iter_{iter_n - 1}" / "island.blend"
    transforms_json = simple_refiner_dir / f"iter_{iter_n}" / "transforms.json"
    out_dir = simple_refiner_dir / f"iter_{iter_n}"
    output_blend = out_dir / "island.blend"

    if not input_blend.is_file():
        print(
            f"ERROR: input island.blend not found: {input_blend}\n"
            f"       Did you run --iter {iter_n - 1} first?",
            file=sys.stderr,
        )
        sys.exit(1)

    if not transforms_json.is_file():
        print(
            f"ERROR: transforms.json not found: {transforms_json}\n"
            f"       Write this file before calling apply_delta --iter {iter_n}.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    blender_bin = _resolve_blender(args.blender_bin)

    # Write inner Blender script to a temp file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_apply_delta_inner.py", delete=False
    ) as tf:
        tf.write(_INNER_SCRIPT)
        inner_script_path = tf.name

    try:
        cmd = [
            blender_bin,
            "-b",
            str(input_blend),
            "-P",
            inner_script_path,
            "--",
            "--transforms",
            str(transforms_json),
            "--output",
            str(output_blend),
        ]

        print(f"[apply_delta] blender   : {blender_bin}")
        print(f"[apply_delta] input     : {input_blend}")
        print(f"[apply_delta] transforms: {transforms_json}")
        print(f"[apply_delta] output    : {output_blend}")
        print(f"[apply_delta] cmd       : {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in (result.stdout or "").splitlines():
            print(line)

        if result.returncode != 0:
            print(
                f"\n[apply_delta] ERROR: Blender subprocess exited with code {result.returncode}.",
                file=sys.stderr,
            )
            sys.exit(1)

        if not output_blend.is_file():
            print(
                f"[apply_delta] ERROR: output blend was not created: {output_blend}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"[apply_delta] Done. Output: {output_blend}")

    finally:
        try:
            os.unlink(inner_script_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
