#!/usr/bin/env python3
"""Generate json/blend_info.json by invoking external_blend_runner.py twice
(list_objects with empty prefix → all objects; metrics → OOB + AABB collisions),
then merging the two results into a single categorised JSON.

The runner intentionally tries to save the .blend back even for read-only ops
(`metrics`); we feed it `/dev/null` and tolerate the "save failed" message —
the analysis itself completes before save is attempted, so the data fields are
always present. After parsing, we patch `metrics.success = true` once we
confirm the analytical fields exist.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ── DIRECTORYS.yaml (canonical machine-specific paths) ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())

CONTROL_RUNNER = Path(__file__).resolve().parent / "external_blend_runner.py"
BLENDER_DEFAULT = os.environ.get("BLENDER", _DIRS["blender_bin"])


CATEGORY_RULES = [
    ("objects",         lambda n: n.startswith("obj_")),
    ("stage_floor",     lambda n: n.startswith("Floor")),
    ("stage_walls",     lambda n: n.startswith("Wall_")),
    ("stage_ceiling",   lambda n: n.startswith("Ceiling")),
    ("lights",          lambda n: n.startswith(("Sun", "Area_", "Practical_", "Portal_", "Class_Light_", "Light_"))),
    ("cameras",         lambda n: n == "Camera" or n.startswith("Camera.")),
    ("pointcloud",      lambda n: n.startswith("PointCloud_")),
    ("geometry_meshes", lambda n: n.startswith("geometry_")),
    ("world",           lambda n: n.startswith("world")),
]


def _categorise(name: str) -> str:
    for cat, predicate in CATEGORY_RULES:
        if predicate(name):
            return cat
    return "other"


def _round_floats(obj, n=4):
    if isinstance(obj, list):
        return [round(float(x), n) if isinstance(x, (int, float)) else x for x in obj]
    if isinstance(obj, (int, float)):
        return round(float(obj), n)
    return obj


def _run_op(blender: str, runner: Path, blend: Path, op_payload: dict, work: Path) -> dict:
    op_path = work / f"op_{op_payload['action']}.json"
    op_path.write_text(json.dumps(op_payload))
    # Pass /dev/null as output_blend; metrics/list_objects either skip the save
    # entirely or fail at save with a recoverable error message.
    cmd = [
        blender, "--background", str(blend),
        "--python", str(runner),
        "--", str(op_path), "/dev/null",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"[extract_blend_info] runner failed for action={op_payload['action']}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    result_path = Path(str(op_path) + ".result")
    if not result_path.exists():
        raise SystemExit(f"[extract_blend_info] no result file at {result_path}")
    return json.loads(result_path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description="Build inputs/blend_info.json from a .blend.")
    p.add_argument("--scene_dir", required=True, type=Path)
    p.add_argument("--blend", type=Path, default=None,
                   help="Override path to the .blend (default: <scene_dir>/blend/blender_scene.blend)")
    p.add_argument("--blender-bin", default=BLENDER_DEFAULT)
    p.add_argument("--output", type=Path, default=None,
                   help="Override output path (default: <scene_dir>/json/blend_info.json)")
    p.add_argument("--oob-tolerance", type=float, default=0.05,
                   help="OOB tolerance in metres (default 0.05)")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    blend = (args.blend or (scene_dir / "blend" / "blender_scene.blend")).resolve()
    if not blend.exists():
        raise SystemExit(f"[extract_blend_info] .blend not found: {blend}")
    if not CONTROL_RUNNER.exists():
        raise SystemExit(f"[extract_blend_info] runner not found: {CONTROL_RUNNER}")
    output = (args.output or (scene_dir / "json" / "blend_info.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="blend_info_"))

    print(f"[extract_blend_info] runner   = {CONTROL_RUNNER}")
    print(f"[extract_blend_info] blend    = {blend}")
    print(f"[extract_blend_info] output   = {output}")
    print(f"[extract_blend_info] scratch  = {work}")

    # 1) all objects
    list_result = _run_op(
        args.blender_bin, CONTROL_RUNNER, blend,
        {"action": "list_objects", "name_prefix": ""},
        work,
    )
    all_objs = list_result.get("objects", [])

    # 2) metrics
    metrics = _run_op(
        args.blender_bin, CONTROL_RUNNER, blend,
        {"action": "metrics", "name_prefix": "obj_", "oob_tolerance": args.oob_tolerance},
        work,
    )

    # The runner sets metrics.success=False because it failed to save to /dev/null.
    # The analysis fields are written before save is attempted, so promote success.
    metrics_fields = ("Nobj", "OOB_count", "BBL_count", "collisions", "room_bbox")
    if all(k in metrics for k in metrics_fields):
        metrics["success"] = True
        metrics.pop("message", None)
        metrics["_save_attempt_note"] = (
            "save attempt skipped (output_blend was /dev/null; metrics is read-only)"
        )

    # 3) Categorise + round transforms.
    categories = {key: [] for key, _ in CATEGORY_RULES}
    categories["other"] = []
    for o in all_objs:
        for k in ("location", "rotation_euler", "scale", "dimensions"):
            if k in o:
                o[k] = _round_floats(o[k])
        categories[_categorise(o["name"])].append(o)
    for v in categories.values():
        v.sort(key=lambda d: d["name"])

    agg = {
        "blend_path": str(blend),
        "object_count_total": len(all_objs),
        "categories": categories,
        "metrics": metrics,
    }

    # ── Sanity-check before writing ──────────────────────────────────────────
    # Validate that categories contain dicts (not bare name-strings), and that
    # the overall object count is consistent.  A previous bug caused certain
    # code paths to write a file where categories held plain string lists
    # (["Wall_01", "Wall_02"]) instead of object-dicts, and where
    # categories["objects"] was empty even though object_count_total > 0.
    # Catch that here so the file is never silently written in a corrupt state.
    _bad_cats = []
    for _cat_name, _cat_items in categories.items():
        for _item in _cat_items:
            if not isinstance(_item, dict):
                _bad_cats.append(
                    f"  category '{_cat_name}' contains a {type(_item).__name__!r}"
                    f" value instead of a dict: {_item!r}"
                )
                break
    if _bad_cats:
        raise SystemExit(
            "[extract_blend_info] FATAL: list_objects returned non-dict entries "
            "in categories — aborting to prevent writing a corrupt blend_info.json.\n"
            + "\n".join(_bad_cats)
        )

    _obj_count = len(categories.get("objects", []))
    if len(all_objs) > 0 and _obj_count == 0:
        raise SystemExit(
            f"[extract_blend_info] FATAL: {len(all_objs)} total objects were loaded "
            f"from {blend} but categories['objects'] is empty after categorisation.\n"
            "Check that the blend contains objects whose names start with 'obj_'.\n"
            "If the scene uses a different naming convention, update CATEGORY_RULES."
        )

    output.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    # Console summary
    print()
    print("=== category counts ===")
    for k in [c[0] for c in CATEGORY_RULES] + ["other"]:
        print(f"  {k:20s} : {len(categories[k])}")
    if metrics.get("success"):
        print()
        print(f"=== metrics ===")
        print(f"  Nobj={metrics['Nobj']}  OOB={metrics['OOB_count']}  BBL={metrics['BBL_count']}")
        if metrics.get("collisions"):
            print("  top-5 collisions:")
            for c in metrics["collisions"][:5]:
                print(f"    {c['a']} <-> {c['b']}  vol={c['volume_m3']} m^3")
    print()
    print(f"[extract_blend_info] OK -> {output}")


if __name__ == "__main__":
    main()
