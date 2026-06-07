#!/usr/bin/env python3
"""Validate that scene_dir has every input the `stage3-relation-graph` agent needs (Step 3.1).

Exits non-zero with a clear list of missing files if anything is absent.
Prints the canonical "run /stage2-environment-construction first" hint on failure.
"""
import argparse
import sys
from pathlib import Path

REQUIRED_FILES = [
    "image.png",
    "inputs/blend_info.json",
    "inputs/object_class.json",
    "inputs/object_state_annotated_mask.png",
]
REQUIRED_NONEMPTY_DIRS = [
    "inputs/masks",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    args = ap.parse_args()

    scene = Path(args.scene_dir).resolve()
    if not scene.is_dir():
        print(f"scene_dir does not exist or is not a directory: {scene}", file=sys.stderr)
        return 2

    missing: list[str] = []

    for rel in REQUIRED_FILES:
        if not (scene / rel).is_file():
            missing.append(str(scene / rel))

    for rel in REQUIRED_NONEMPTY_DIRS:
        d = scene / rel
        if not d.is_dir() or not any(d.glob("*.png")):
            missing.append(f"{d} (must be a non-empty dir of *.png masks)")

    if missing:
        print("MISSING_INPUTS:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            f"\nRun  /stage2-environment-construction {scene}  first to populate these, "
            "then re-run /scene-analyze-prepare.",
            file=sys.stderr,
        )
        return 1

    print(f"OK — all required inputs present at {scene}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
