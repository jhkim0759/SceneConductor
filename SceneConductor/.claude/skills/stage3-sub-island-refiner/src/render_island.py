"""render_island.py — thin wrapper that renders a single island.blend via
render_island_views.py (Cycles, computed cameras, no saved camera).

Usage (plain Python, NOT inside Blender):
    python render_island.py \\
        --island <path/to/island.blend> \\
        --out-dir <path/to/iter_dir> \\
        [--samples 128] \\
        [--resolution-x 1024] \\
        [--resolution-y 768] \\
        [--blender-bin <path/to/blender>]

After a successful run the following files exist flat in <out-dir>:
    render_persp.png   (mandatory)
    render_bev.png     (mandatory)

Exit codes:
    0 — both mandatory views produced and non-empty (>1 KB)
    1 — subprocess failure or one/both mandatory views missing / empty
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# This file lives at:
#   <repo>/.claude/skills/stage3-sub-island-simple/src/render_island.py
#   parents: [0]=src [1]=stage3-sub-island-simple [2]=skills [3]=.claude [4]=repo-root
# (Vendored from the deprecated stage3-sub-island-refiner skill.)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]

_DIRS_YAML = _REPO_ROOT / "DIRECTORYS.yaml"

# The Blender -P script that does the actual rendering.
_VIEWS_SCRIPT = _THIS_FILE.parent / "render_island_views.py"


# ---------------------------------------------------------------------------
# Blender-bin resolution
# ---------------------------------------------------------------------------

def _read_blender_bin_from_yaml(yaml_path: Path) -> str | None:
    """Return the value of the `blender_bin:` key without requiring PyYAML."""
    if not yaml_path.is_file():
        return None

    # Try PyYAML first (available in most conda envs in this project).
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(yaml_path.read_text())
        return str(data.get("blender_bin", "")) or None
    except Exception:
        pass

    # Fallback: naive line scan for   blender_bin: <value>
    for line in yaml_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("blender_bin:") and not stripped.startswith(
            "blender_bin_"
        ):
            value = stripped.split(":", 1)[1].strip()
            if value:
                return value
    return None


def _resolve_blender(blender_bin_arg: str | None) -> str:
    """Resolve the blender executable path in priority order:

    1. --blender-bin CLI argument
    2. DIRECTORYS.yaml blender_bin  (relative → resolved from repo root)
    3. $BLENDER env var
    4. Literal "blender"  (must be on PATH)
    """
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="render_island",
        description=(
            "Render an island.blend via render_island_views.py (Cycles, "
            "computed cameras) and verify outputs."
        ),
    )
    ap.add_argument(
        "--island",
        required=True,
        metavar="PATH",
        help="Path to the island.blend to render.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        metavar="PATH",
        dest="out_dir",
        help="Output directory; render_persp.png and render_bev.png land here.",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=128,
        metavar="N",
        help="Cycles sample count (default: 128).",
    )
    ap.add_argument(
        "--resolution-x",
        type=int,
        default=1024,
        metavar="PX",
        dest="resolution_x",
    )
    ap.add_argument(
        "--resolution-y",
        type=int,
        default=768,
        metavar="PX",
        dest="resolution_y",
    )
    ap.add_argument(
        "--blender-bin",
        default=None,
        metavar="PATH",
        dest="blender_bin",
        help=(
            "Path to the Blender executable. "
            "Defaults to DIRECTORYS.yaml → $BLENDER → 'blender'."
        ),
    )
    ap.add_argument(
        "--metadata",
        default=None,
        metavar="PATH",
        dest="metadata",
        help="Path to metadata.json for canonical-frame camera transformation (optional).",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_MIN_FILE_BYTES = 1024  # files smaller than this are treated as empty/broken


def main() -> None:
    args = _parse_args()

    island_blend = Path(args.island).resolve()
    out_dir = Path(args.out_dir).resolve()
    blender_bin = _resolve_blender(args.blender_bin)

    # Validate inputs.
    if not island_blend.is_file():
        print(f"ERROR: island.blend not found: {island_blend}", file=sys.stderr)
        sys.exit(1)

    if not _VIEWS_SCRIPT.is_file():
        print(f"ERROR: render_island_views.py not found: {_VIEWS_SCRIPT}",
              file=sys.stderr)
        sys.exit(1)

    # Create output directory.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the Blender subprocess command.
    # --resolution-x / --resolution-y map to --persp-size W H.
    # BEV stays square via the inner default (1024 1024).
    cmd = [
        blender_bin,
        "-b",
        str(island_blend),
        "-P",
        str(_VIEWS_SCRIPT),
        "--",
        "--out-dir",
        str(out_dir),
        "--samples",
        str(args.samples),
        "--persp-size",
        str(args.resolution_x),
        str(args.resolution_y),
    ]

    # Add metadata path if provided
    if args.metadata:
        cmd.extend(["--metadata", str(args.metadata)])

    print(f"[render_island] blender   : {blender_bin}")
    print(f"[render_island] island    : {island_blend}")
    print(f"[render_island] out-dir   : {out_dir}")
    print(f"[render_island] samples   : {args.samples}")
    print(f"[render_island] resolution: {args.resolution_x}x{args.resolution_y}")
    print(f"[render_island] cmd       : {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Always echo captured output so callers have a full log.
    captured_lines = result.stdout.splitlines() if result.stdout else []
    for line in captured_lines:
        print(line)

    if result.returncode != 0:
        print(
            f"\n[render_island] ERROR: Blender subprocess exited with code "
            f"{result.returncode}.",
            file=sys.stderr,
        )
        tail = captured_lines[-20:] if len(captured_lines) > 20 else captured_lines
        print("[render_island] --- last output ---", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        print("[render_island] -------------------", file=sys.stderr)
        print("[render_island] Produced: []  success=False", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Verify mandatory outputs exist and are non-empty (> 1 KB).
    # ------------------------------------------------------------------
    MANDATORY = ["render_persp.png", "render_bev.png"]
    produced: list[str] = []
    missing: list[str] = []

    for name in MANDATORY:
        fpath = out_dir / name
        if fpath.is_file() and fpath.stat().st_size >= _MIN_FILE_BYTES:
            produced.append(name)
        else:
            reason = (
                "missing" if not fpath.exists()
                else f"too small ({fpath.stat().st_size} B)"
            )
            missing.append(f"{name} ({reason})")

    if missing:
        print(
            f"\n[render_island] ERROR: mandatory outputs not produced: {missing}",
            file=sys.stderr,
        )
        print(
            f"[render_island] Produced: {produced}  success=False",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[render_island] Produced: {produced}  success=True")


if __name__ == "__main__":
    main()
