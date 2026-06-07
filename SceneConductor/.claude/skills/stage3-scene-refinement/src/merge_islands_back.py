#!/usr/bin/env python3
"""merge_islands_back.py — Merge per-island refined transforms to stage3-scene.blend.

Reads each relation_groups/<G>/simple_refiner/iter_FINAL/transforms.json,
applies M_anchor @ M_canonical for each member, writes blend/stage3-scene.blend.

Usage:
    python merge_islands_back.py <scene_dir> [--blender-bin PATH] [--manifest-out PATH]

Exit codes:
  0   success — blend/stage3-scene.blend written
  1   fatal error (missing inputs, Blender failure, etc.)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# NOTE: this file lives at <skill>/src/merge_islands_back.py, so the skill root
# is parent.parent — _migrated/ is a sibling of src/, not a child.
SRC_DIR = Path(__file__).resolve().parent
SKILL_DIR = SRC_DIR.parent
INNER_SCRIPT = SKILL_DIR / "_migrated" / "merge_island_blender_inner.py"


def build_manifest(scene_dir: Path) -> dict:
    """For each group in relation_graph.json, find the latest refined iter_K/island.blend.

    The chosen iter is the one whose transforms.json carries ``final: true`` (else
    the highest iter index that has both transforms.json and island.blend present).
    The selected iter_K/island.blend is the only file that actually carries the
    refined canonical poses — transforms.json contains per-iter deltas only, so it
    is NOT sufficient on its own.

    Returns a manifest dict consumed by ``_migrated/merge_island_blender_inner.py``:
      {"groups": [
          {
            "group_id": str,
            "group_dir": str,           # abs path to scene/relation_groups/<gid>
            "anchor_id": str,           # from metadata.json
            "metadata": str,            # abs path to group metadata.json
            "island_blend": str,        # abs path to chosen iter_K/island.blend
            "transforms": str,          # abs path to chosen iter_K/transforms.json
            "iter": int,                # chosen iter index (for logging)
            "final": bool,              # whether the iter is marked final
          }, ...
      ]}
    """
    rg_path = scene_dir / "inputs" / "relation_graph.json"
    if not rg_path.is_file():
        print(
            f"[merge_islands_back] ERROR: relation_graph.json not found at {rg_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    rg = json.loads(rg_path.read_text())
    gids: list[str] = [g.get("group_id") for g in rg.get("groups", []) if g.get("group_id")]

    # Append synthetic S* groups from json/island_groups.json (these are not in
    # relation_graph.json by design — they are created by the validation step
    # for ungrouped floating/penetrating objects).
    ig_path = scene_dir / "json" / "island_groups.json"
    if ig_path.is_file():
        try:
            ig = json.loads(ig_path.read_text())
            for entry in ig.get("synthetic_groups", []) or []:
                sgid = entry.get("group_id")
                if sgid and sgid not in gids:
                    gids.append(sgid)
        except Exception as exc:
            print(
                f"[merge_islands_back] WARNING: cannot parse {ig_path}: {exc} — "
                "synthetic islands will not be merged",
                file=sys.stderr,
            )

    manifest_groups = []
    for gid in gids:
        group_dir = scene_dir / "relation_groups" / gid
        if not group_dir.is_dir():
            continue
        metadata = group_dir / "metadata.json"
        if not metadata.is_file():
            continue
        # Read anchor_id from metadata (required by inner script).
        try:
            meta = json.loads(metadata.read_text())
        except Exception as exc:
            print(
                f"[merge_islands_back] WARNING: cannot parse {metadata}: {exc} — skipping {gid}",
                file=sys.stderr,
            )
            continue
        anchor_id = meta.get("anchor_id")
        if not anchor_id:
            print(
                f"[merge_islands_back] WARNING: anchor_id missing in {metadata} — skipping {gid}",
                file=sys.stderr,
            )
            continue

        # Find latest iter_K/island.blend (prefer transforms.json final=true,
        # else highest K with both island.blend and transforms.json).
        loop = group_dir / "simple_refiner"
        if not loop.is_dir():
            continue
        iters = sorted(
            (d for d in loop.iterdir() if d.is_dir() and d.name.startswith("iter_")
             and d.name.split("_")[1].isdigit()),
            key=lambda p: int(p.name.split("_")[1]),
            reverse=True,
        )
        chosen: Path | None = None
        chosen_final = False
        # Prefer the one marked final=true (and with island.blend present).
        for d in iters:
            tf = d / "transforms.json"
            ib = d / "island.blend"
            if not tf.is_file() or not ib.is_file():
                continue
            try:
                data = json.loads(tf.read_text())
            except Exception:
                continue
            if data.get("final") is True:
                chosen = d
                chosen_final = True
                break
        # Fallback: highest iter with both island.blend and transforms.json present.
        if chosen is None:
            for d in iters:
                if (d / "transforms.json").is_file() and (d / "island.blend").is_file():
                    chosen = d
                    break
        if chosen is None:
            print(
                f"[merge_islands_back] WARNING: no iter_K/island.blend found under {loop} — skipping {gid}",
                file=sys.stderr,
            )
            continue
        chosen_iter = int(chosen.name.split("_")[1])
        manifest_groups.append(
            {
                "group_id": gid,
                "group_dir": str(group_dir),
                "anchor_id": anchor_id,
                "metadata": str(metadata),
                "island_blend": str(chosen / "island.blend"),
                "transforms": str(chosen / "transforms.json"),
                "iter": chosen_iter,
                "final": chosen_final,
            }
        )
    # Inner script reads ``groups`` key; keep ``islands`` for backward-compat logging.
    return {"groups": manifest_groups, "islands": manifest_groups}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-island refined transforms to blend/stage3-scene.blend"
    )
    parser.add_argument("scene_dir", type=Path, help="Path to the scene working directory.")
    parser.add_argument(
        "--blender-bin",
        type=str,
        default=None,
        help="Path to Blender binary (falls back to $BLENDER env var, then 'blender').",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional: write the manifest JSON to this path for debugging.",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()

    # 1. Copy stage3-sub-planned.blend → stage3-scene.blend as starting point.
    planned = scene_dir / "blend" / "stage3-sub-planned.blend"
    refined = scene_dir / "blend" / "stage3-scene.blend"
    if not planned.is_file():
        print(
            f"[merge_islands_back] ERROR: {planned} not found — "
            "scene-refiner step must complete first.",
            file=sys.stderr,
        )
        sys.exit(1)
    refined.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(planned, refined)
    print(f"[merge_islands_back] base: {planned} → {refined}")

    # 2. Build manifest of completed island transforms.
    manifest = build_manifest(scene_dir)
    if args.manifest_out:
        args.manifest_out.write_text(json.dumps(manifest, indent=2))
        print(f"[merge_islands_back] manifest written to {args.manifest_out}")

    if not manifest["islands"]:
        print(
            "[merge_islands_back] no island transforms found; "
            "output is a copy of planned blend"
        )
        return

    print(
        f"[merge_islands_back] {len(manifest['islands'])} island(s) found in manifest"
    )

    # 3. If inner Blender script exists, invoke it to apply canonical→world transforms.
    if INNER_SCRIPT.is_file():
        import os

        blender = args.blender_bin or os.environ.get("BLENDER", "blender")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(manifest, f, indent=2)
            manifest_path = f.name

        cmd = [
            blender,
            "-b", str(refined),
            "-P", str(INNER_SCRIPT),
            "--", manifest_path,
        ]
        print(f"[merge_islands_back] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            print(
                f"[merge_islands_back] ERROR: Blender inner script exited {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
    else:
        # Degraded mode: no inner script available — output is the planned blend copy.
        print(
            f"[merge_islands_back] WARNING: {INNER_SCRIPT} not found; "
            "outputting planned blend without applying island transforms (degraded mode).",
            file=sys.stderr,
        )

    print(f"[merge_islands_back] wrote {refined}")


if __name__ == "__main__":
    main()
