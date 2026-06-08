"""render_one.py — Thin wrapper calling render_island.py via subprocess.

Enforces --metadata to guarantee camera consistency with masked.png.

Usage:
    python render_one.py \\
        --blend   PATH \\
        --out-dir PATH \\
        --metadata PATH \\
        [--samples N] \\
        [--blender-bin PATH]

--metadata is REQUIRED. Omitting it exits with code 1.
This prevents accidental renders with a fallback camera that is not
aligned to the masked.png viewpoint.

Outputs (produced by render_island.py):
    render_persp.png   (mandatory)
    render_bev.png     (mandatory)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path to render_island.py
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# .claude/skills/stage3-sub-island-simple/src/render_one.py
# parents: [0]=src  [1]=stage3-sub-island-simple  [2]=skills  [3]=.claude  [4]=repo-root
_REPO_ROOT = _THIS_FILE.parents[4]

# render_island.py lives in the same src/ directory (vendored from stage3-sub-island-refiner)
_RENDER_ISLAND_SCRIPT = _THIS_FILE.parent / "render_island.py"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="render_one",
        description=(
            "Render an island.blend by delegating to render_island.py. "
            "--metadata is required to guarantee camera consistency with masked.png."
        ),
    )
    ap.add_argument(
        "--blend",
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
        "--metadata",
        required=True,
        metavar="PATH",
        help=(
            "Path to metadata.json for canonical-frame camera transform (M_inv_4x4). "
            "REQUIRED — ensures perspective camera matches the masked.png viewpoint."
        ),
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=128,
        metavar="N",
        help="Cycles sample count passed to render_island.py (default: 128).",
    )
    ap.add_argument(
        "--blender-bin",
        default=None,
        metavar="PATH",
        dest="blender_bin",
        help="Path to Blender executable; forwarded to render_island.py.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    blend_path = Path(args.blend).resolve()
    out_dir = Path(args.out_dir).resolve()
    metadata_path = Path(args.metadata).resolve()

    # Validate render_island.py exists.
    if not _RENDER_ISLAND_SCRIPT.is_file():
        print(
            f"ERROR: render_island.py not found: {_RENDER_ISLAND_SCRIPT}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate inputs.
    if not blend_path.is_file():
        print(f"ERROR: blend file not found: {blend_path}", file=sys.stderr)
        sys.exit(1)

    if not metadata_path.is_file():
        print(f"ERROR: metadata.json not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    # Build forwarding command.
    cmd = [
        sys.executable,
        str(_RENDER_ISLAND_SCRIPT),
        "--island",    str(blend_path),
        "--out-dir",   str(out_dir),
        "--metadata",  str(metadata_path),
        "--samples",   str(args.samples),
    ]

    if args.blender_bin:
        cmd.extend(["--blender-bin", args.blender_bin])

    print(f"[render_one] blend    : {blend_path}")
    print(f"[render_one] out-dir  : {out_dir}")
    print(f"[render_one] metadata : {metadata_path}")
    print(f"[render_one] samples  : {args.samples}")
    print(f"[render_one] delegate : {' '.join(cmd)}")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(
            f"[render_one] ERROR: render_island.py exited with code {result.returncode}.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)

    print("[render_one] Done.")


if __name__ == "__main__":
    main()
