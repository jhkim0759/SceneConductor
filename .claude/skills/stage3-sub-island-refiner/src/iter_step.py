#!/usr/bin/env python3
"""iter_step.py — Single-command wrapper for one island-refiner iteration step.

The stage3-island-refiner agent writes ``simple_refiner/iter_N/transforms.json``
each iter, then runs ONE Bash command (this script). This wrapper then runs the
deterministic helpers in order:

    1. ``apply_delta.py``           — apply transforms.json to iter_(N-1)/island.blend
                                     producing iter_N/island.blend
    2. ``render_one.py``            — render iter_N/{render_persp.png, render_bev.png}
    3. ``score_info.py``            — write iter_N/info.json (INFO ONLY stub, iter + notes)
    5. ``compute_member_geometry.py`` — write iter_N/geometry.json (deterministic
                                     forward-axis + clearance ground truth; consumed by
                                     the agent) and forward_axes.json (cached at group_dir,
                                     written only on first call).

PLUS a built-in **sanity check** between steps 1 and 2: if transforms.json had
any non-trivial member deltas (|xyz| > 1e-6 or |yaw| > 1e-3), this wrapper
verifies that iter_N/island.blend object positions actually differ from
iter_(N-1)/island.blend. If they do not, the wrapper exits with a non-zero
status — preventing the silent-fail scenario where transforms were written but
not applied.

Usage (agent invokes this exactly once per iter, after writing transforms.json):

    python iter_step.py --group-dir <ABS PATH> --iter N \\
        [--samples 64] [--blender-bin PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], label: str) -> str:
    """Run a subprocess. Print + return stdout. Exit on non-zero return code."""
    print(f"[iter_step] running {label}: {' '.join(cmd[:3])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        # Echo helper output so the agent can see it
        for line in result.stdout.rstrip().splitlines():
            print(f"  [{label}] {line}")
    if result.returncode != 0:
        print(f"[iter_step] FAILED at {label} (exit {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    return result.stdout


def _dump_obj_positions(blend_path: Path, blender_bin: str,
                        valid_names: set | None = None) -> dict:
    """Read all top-level obj_X EMPTY positions from a blend. Returns {name: (x,y,z,yaw_deg)}.

    Allow stage-geometry anchors (Floor / Wall_*) — used by synthetic islands.
    When valid_names is provided the filter matches that exact set; otherwise falls
    back to startswith("obj_") for backward compatibility.
    """
    if valid_names is not None:
        name_filter = f"o.name in {repr(sorted(valid_names))}"
    else:
        name_filter = 'o.name.startswith("obj_")'
    inner = f"""
import bpy, json, sys
out = {{}}
for o in bpy.data.objects:
    if o.type == "EMPTY" and {name_filter} and o.parent is None:
        out[o.name] = [round(o.location.x, 6), round(o.location.y, 6),
                       round(o.location.z, 6), round(o.rotation_euler.z, 6)]
sys.stdout.write("@@OBJPOS@@" + json.dumps(out) + "@@END@@\\n")
"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(inner)
        script_path = tf.name
    try:
        cmd = [blender_bin, "-b", str(blend_path), "-P", script_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        marker_a, marker_b = "@@OBJPOS@@", "@@END@@"
        i = result.stdout.find(marker_a)
        j = result.stdout.find(marker_b, i)
        if i < 0 or j < 0:
            return {}
        payload = result.stdout[i + len(marker_a):j]
        return json.loads(payload)
    finally:
        os.unlink(script_path)


def _transforms_have_nontrivial_deltas(transforms_path: Path) -> tuple[bool, int]:
    """Return (has_nontrivial, n_members). Trivial means |xyz| < 1e-6 AND |yaw_deg| < 1e-3."""
    if not transforms_path.is_file():
        return (False, 0)
    try:
        data = json.loads(transforms_path.read_text())
    except Exception:
        return (False, 0)
    members = data.get("members", {})
    if not isinstance(members, dict):
        return (False, 0)
    n = len(members)
    for m in members.values():
        dxyz = m.get("delta_xyz", [0, 0, 0])
        dyaw = m.get("delta_yaw_deg", 0.0)
        if (
            any(abs(float(v)) > 1e-6 for v in dxyz)
            or abs(float(dyaw)) > 1e-3
        ):
            return (True, n)
    return (False, n)


def _positions_diff(pre: dict, post: dict, tol: float = 1e-4) -> list[str]:
    """Return list of obj_ids whose (x,y,z,yaw) actually changed between pre and post."""
    changed = []
    for k, post_v in post.items():
        pre_v = pre.get(k)
        if pre_v is None:
            continue
        if any(abs(a - b) > tol for a, b in zip(pre_v, post_v)):
            changed.append(k)
    return changed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-command wrapper: apply_delta + render_one + score_info(stub) "
                    "for one island-refiner iter, with sanity check."
    )
    p.add_argument("--group-dir", type=Path, required=True,
                   help="Absolute path to relation_groups/<G>/")
    p.add_argument("--iter", dest="iter_n", type=int, required=True,
                   help="Iter number N (>=1). Reads iter_(N-1)/island.blend and "
                        "iter_N/transforms.json; writes iter_N/island.blend + renders + info.json.")
    p.add_argument("--samples", type=int, default=64,
                   help="Cycles sample count for render_one.py (default: 64).")
    p.add_argument("--blender-bin", type=str, default=None,
                   help="Override blender executable path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    group_dir = args.group_dir.resolve()
    N = args.iter_n

    if N < 1:
        print(f"[iter_step] ERROR: --iter must be >= 1 (got {N})", file=sys.stderr)
        sys.exit(2)

    loop_dir = group_dir / "simple_refiner"
    prev_dir = loop_dir / f"iter_{N - 1}"
    cur_dir = loop_dir / f"iter_{N}"
    prev_blend = prev_dir / "island.blend"
    cur_blend = cur_dir / "island.blend"
    transforms = cur_dir / "transforms.json"
    masked = group_dir / "masked.png"
    metadata = group_dir / "metadata.json"

    # Pre-conditions
    for p_, label in [
        (prev_blend, f"iter_{N-1}/island.blend"),
        (transforms, f"iter_{N}/transforms.json"),
        (masked, "masked.png"),
        (metadata, "metadata.json"),
    ]:
        if not p_.is_file():
            print(f"[iter_step] ERROR: missing input — {label} ({p_})", file=sys.stderr)
            sys.exit(2)

    cur_dir.mkdir(parents=True, exist_ok=True)

    # Build valid_names from metadata so stage-geometry anchors (Floor / Wall_*) are included.
    try:
        _meta = json.loads(metadata.read_text())
        valid_names: set | None = set(_meta.get("members", {}).keys()) or None
    except Exception:
        valid_names = None

    # Diagnose transforms
    has_nontrivial, n_members = _transforms_have_nontrivial_deltas(transforms)
    print(f"[iter_step] iter={N}  members={n_members}  nontrivial_delta={has_nontrivial}")

    # Resolve blender binary (delegate to apply_delta.py's own resolver via env)
    if args.blender_bin:
        os.environ["BLENDER"] = args.blender_bin
        blender_bin_arg = ["--blender-bin", args.blender_bin]
    else:
        blender_bin_arg = []
        os.environ.setdefault("BLENDER", os.environ.get("BLENDER", "blender"))

    # Snapshot obj positions BEFORE (only if we'll need the sanity check)
    pre_positions: dict = {}
    if has_nontrivial:
        pre_positions = _dump_obj_positions(prev_blend, os.environ["BLENDER"], valid_names)
        if not pre_positions:
            print(f"[iter_step] WARNING: could not read obj positions from {prev_blend}; "
                  f"skipping sanity check", file=sys.stderr)

    # ── Step 1: apply_delta ────────────────────────────────────────────────
    cmd_apply = [
        sys.executable, str(SCRIPT_DIR / "apply_delta.py"),
        "--group-dir", str(group_dir),
        "--iter", str(N),
        *blender_bin_arg,
    ]
    _run(cmd_apply, "apply_delta")

    if not cur_blend.is_file():
        print(f"[iter_step] ERROR: apply_delta did not produce {cur_blend}", file=sys.stderr)
        sys.exit(3)

    # ── Sanity check: did the blend actually change? ──────────────────────
    if has_nontrivial and pre_positions:
        post_positions = _dump_obj_positions(cur_blend, os.environ["BLENDER"], valid_names)
        changed = _positions_diff(pre_positions, post_positions)
        if not changed:
            print(
                f"[iter_step] ERROR: apply_delta returned 0 but iter_{N}/island.blend "
                f"object positions are IDENTICAL to iter_{N-1}. The transforms.json had "
                f"{n_members} non-trivial members. Aborting to prevent silent-fail "
                f"iteration loop.",
                file=sys.stderr,
            )
            sys.exit(4)
        print(f"[iter_step] sanity OK — {len(changed)} object(s) actually moved: "
              f"{', '.join(sorted(changed))}")
    elif not has_nontrivial:
        print(f"[iter_step] (trivial transforms — skipping sanity check)")

    # ── Step 2: render_one ─────────────────────────────────────────────────
    cmd_render = [
        sys.executable, str(SCRIPT_DIR / "render_one.py"),
        "--blend", str(cur_blend),
        "--out-dir", str(cur_dir),
        "--metadata", str(metadata),
        "--samples", str(args.samples),
        *blender_bin_arg,
    ]
    _run(cmd_render, "render_one")

    for required in (cur_dir / "render_persp.png", cur_dir / "render_bev.png"):
        if not required.is_file() or required.stat().st_size < 1024:
            print(f"[iter_step] ERROR: render_one did not produce {required}", file=sys.stderr)
            sys.exit(5)

    # ── Step 3: score_info (INFO ONLY stub) ───────────────────────────────
    cmd_score = [
        sys.executable, str(SCRIPT_DIR / "score_info.py"),
        "--iter-dir", str(cur_dir),
    ]
    _run(cmd_score, "score_info")

    info_path = cur_dir / "info.json"
    if not info_path.is_file():
        print(f"[iter_step] ERROR: score_info did not produce {info_path}", file=sys.stderr)
        sys.exit(3)

    # ── Step 4: dump current absolute poses → current_state.json ───────────
    # Always dump after apply+render — gives the sub-agent a numeric read of
    # where every object actually sits in canonical/world frame.
    blender_for_dump = args.blender_bin or os.environ.get("BLENDER")
    if not blender_for_dump:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        dirs_yaml = repo_root / "DIRECTORYS.yaml"
        if dirs_yaml.exists():
            try:
                import yaml
                d = yaml.safe_load(dirs_yaml.read_text()) or {}
                blender_for_dump = (d.get("blender_bin_linux")
                                    or d.get("blender_bin"))
            except Exception:
                pass
    if blender_for_dump:
        raw = _dump_obj_positions(cur_blend, blender_for_dump, valid_names)
        if raw:
            import math
            obj_positions = {
                k: {"position": [v[0], v[1], v[2]],
                    "yaw_deg": round(math.degrees(v[3]), 4)}
                for k, v in raw.items()
            }
            state_out = {
                "iter": N,
                "frame": "canonical",
                "_note": ("Absolute object poses after iter N apply. "
                          "position in metres, yaw_deg is rotation about Z."),
                "obj_positions": obj_positions,
            }
            (cur_dir / "current_state.json").write_text(
                json.dumps(state_out, indent=2), encoding="utf-8"
            )

    # ── Step 5: compute_member_geometry ───────────────────────────────────
    cmd_geom = [
        sys.executable, str(SCRIPT_DIR / "compute_member_geometry.py"),
        "--group-dir", str(group_dir),
        "--iter", str(N),
    ]
    if args.blender_bin:
        cmd_geom += ["--blender-bin", args.blender_bin]
    _run(cmd_geom, "compute_member_geometry")
    geom_path = cur_dir / "geometry.json"
    if not geom_path.exists():
        print(f"[iter_step] ERROR: compute_member_geometry did not produce {geom_path}",
              file=sys.stderr)
        sys.exit(6)

    # Final summary
    print(f"[iter_step] iter {N} complete")


if __name__ == "__main__":
    main()
