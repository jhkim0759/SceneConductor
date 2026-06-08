"""Regenerate island.blend + metadata.json for each selected group from the
pass-1 working blend (stage3-sub-planned.blend).

This is the fix for the "merge regression" bug: when island.blend was created
upstream by /scene-relation-graph, it captured pre-pass-1 transforms (scale,
rotation, location). After pass-1 applied update_size / update_rotation /
attach ops to the working blend, the per-group island.blend files were stale.
The merge step then composed M_anchor @ M_canonical from the stale islands
back into the working blend, silently overwriting pass-1's edits.

Rebuilding islands from stage3-sub-planned.blend after pass-1 ensures
the canonical metadata reflects pass-1's corrections, so merge is idempotent.

If `relation_groups/<G>/` does not yet exist, it is created. After all islands
are built, `masked.png` is regenerated for every group via
`make_group_masked_images.py`.

Usage:
    python rebuild_islands.py <scene_dir>

Reads:
    <scene_dir>/scene-refine-loop/selected_groups.json
    <scene_dir>/blend/stage3-sub-planned.blend  (post-pass-1, never source)
    <scene_dir>/relation_groups/<G>/metadata.json  (for member list confirmation)

Writes (per selected group):
    <scene_dir>/relation_groups/<G>/island.blend             (overwritten)
    <scene_dir>/relation_groups/<G>/metadata.json            (overwritten)
    <scene_dir>/relation_groups/<G>/island.blend.before_rebuild_<ts>
    <scene_dir>/relation_groups/<G>/metadata.json.before_rebuild_<ts>

Writes (after all groups):
    <scene_dir>/relation_groups/<G>/masked.png               (all groups, via make_group_masked_images.py)

Env:
    SCENE_EVAL_BLENDER  Path to the Blender binary.
                        Falls back to env var BLENDER, then to "blender" on PATH.
                        (Canonical path lives in DIRECTORYS.yaml::blender_bin.)
"""
import datetime
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

DEFAULT_BLENDER = os.environ.get("BLENDER", _DIRS["blender_bin"])
BUILD_ISLAND = Path(__file__).parent / "build_island_canonical.py"
MAKE_MASKED = Path(__file__).parent / "make_group_masked_images.py"


def _resolve_scene_camera(blender_bin: str, blend: Path) -> str | None:
    """Return the name of a CAMERA object in *blend*, or None.

    Resolution order:
    1. Env var ``STAGE3_SCENE_CAMERA_NAME`` — use exactly that name (no Blender call).
    2. Run a tiny Blender subprocess that prints every CAMERA-type object name, then
       return the first result (alphabetically to be deterministic).
    3. If Blender fails or no camera found, fall back to the first object whose name
       starts with "Camera" in the printed output (covers default Blender naming).

    Returns None on any error, so the caller can gracefully degrade to the synthetic
    camera.
    """
    # 1. Env override — skip the Blender probe entirely.
    env_name = os.environ.get("STAGE3_SCENE_CAMERA_NAME", "").strip()
    if env_name:
        print(f"[rebuild_islands] using STAGE3_SCENE_CAMERA_NAME={env_name!r}")
        return env_name

    # 2. Inline Blender script: print one camera name per line then quit.
    _PROBE_SCRIPT = (
        "import bpy, sys\n"
        "cams = sorted(o.name for o in bpy.data.objects if o.type == 'CAMERA')\n"
        "print('CAMERAS:' + ','.join(cams))\n"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="probe_cam_", delete=False
    ) as f:
        f.write(_PROBE_SCRIPT)
        probe_path = f.name

    try:
        result = subprocess.run(
            [blender_bin, "-b", str(blend), "--python", probe_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        print(
            f"[rebuild_islands] WARNING: camera probe subprocess failed: {exc}",
            file=sys.stderr,
        )
        return None
    finally:
        try:
            Path(probe_path).unlink(missing_ok=True)
        except OSError:
            pass

    # Parse output for the "CAMERAS:..." marker line.
    camera_name: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("CAMERAS:"):
            names = [n.strip() for n in line[len("CAMERAS:"):].split(",") if n.strip()]
            if names:
                camera_name = names[0]
            break

    if camera_name:
        print(f"[rebuild_islands] probed scene camera: {camera_name!r}")
        return camera_name

    # 3. Heuristic fallback: scan stdout for any token starting with "Camera".
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Camera"):
            camera_name = stripped
            print(
                f"[rebuild_islands] WARNING: camera probe marker not found; "
                f"heuristic fallback camera: {camera_name!r}",
                file=sys.stderr,
            )
            return camera_name

    print(
        "[rebuild_islands] WARNING: no CAMERA object found in scene blend — "
        "island render will use synthetic 3/4 camera",
        file=sys.stderr,
    )
    return None


def _ts() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _backup(path: Path, ts: str) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + f".before_rebuild_{ts}")
    shutil.copy2(path, bak)
    return bak


def _rebuild_one(
    blender_bin: str,
    working_blend: Path,
    group_dir: Path,
    group_id: str,
    anchor_id: str,
    members: list[str],
    camera_name: str | None = None,
    relation_meta: dict | None = None,
) -> None:
    group_dir.mkdir(parents=True, exist_ok=True)

    island_blend = group_dir / "island.blend"
    metadata_json = group_dir / "metadata.json"

    ts = _ts()
    bak_blend = _backup(island_blend, ts)
    bak_meta = _backup(metadata_json, ts)

    member_args = [m for m in members if m != anchor_id]
    if not member_args:
        raise RuntimeError(
            f"group {group_id}: no non-anchor members to rebuild"
        )

    cmd = [
        blender_bin, "-b", str(working_blend),
        "--python", str(BUILD_ISLAND),
        "--",
        "--output-blend", str(island_blend),
        "--metadata", str(metadata_json),
        "--group-name", group_id,
        "--anchor-id", anchor_id,
        "--member-ids", *member_args,
    ]
    if camera_name:
        cmd += ["--use-scene-camera", camera_name]

    print(f"[rebuild_islands] {group_id}")
    print(f"  anchor : {anchor_id}")
    print(f"  members: {member_args}")
    _cam_str = repr(camera_name) if camera_name else "(synthetic 3/4 fallback)"
    print(f"  camera : {_cam_str}")
    _bak_blend_str = bak_blend.name if bak_blend else "(no prior file)"
    _bak_meta_str = bak_meta.name if bak_meta else "(no prior file)"
    print(f"  backup : {_bak_blend_str}, {_bak_meta_str}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[rebuild_islands] Blender exited {result.returncode} for {group_id}",
              file=sys.stderr)
        if result.stdout:
            print("--- stdout ---\n" + result.stdout, file=sys.stderr)
        if result.stderr:
            print("--- stderr ---\n" + result.stderr, file=sys.stderr)
        raise RuntimeError(f"build_island_canonical failed for {group_id}")

    if not island_blend.exists():
        raise RuntimeError(f"build_island_canonical did not produce {island_blend}")
    if not metadata_json.exists():
        raise RuntimeError(f"build_island_canonical did not produce {metadata_json}")

    # Inject relation metadata (edge_type, name, evidence) so the agent can apply
    # the matching "Ideal arrangement principles" without re-reading relation_graph.json.
    if relation_meta:
        meta_dict = json.loads(metadata_json.read_text())
        for k in ("edge_type", "name", "evidence"):
            if k in relation_meta and relation_meta[k] is not None:
                meta_dict[k] = relation_meta[k]
        metadata_json.write_text(json.dumps(meta_dict, indent=2))

    # Sanity: confirm anchor canonical pose is at origin (within numeric epsilon).
    meta = json.loads(metadata_json.read_text())
    anchor_meta = meta.get("members", {}).get(anchor_id, {})
    cl = anchor_meta.get("canonical_location") or [0, 0, 0]
    cr = anchor_meta.get("canonical_rotation_euler") or [0, 0, 0]
    eps = max(abs(v) for v in cl + cr)
    if eps > 1e-3:
        print(
            f"[rebuild_islands] WARNING: anchor canonical pose drifted "
            f"(max abs = {eps:.6f}) for {group_id}",
            file=sys.stderr,
        )

    print(f"[rebuild_islands]   ok -> {island_blend}")
    print(f"[rebuild_islands]   ok -> {metadata_json}")


def _validate_island_groups_meta(scene_dir: Path) -> None:
    """Defense-in-depth: verify json/island_groups.json was produced by run_stage3_validation.py.

    Hard-fails if the file is absent, invalid JSON, or _planner_meta.generated_by is wrong /
    image_sha256 does not match current image.png.
    """
    ig_path = scene_dir / "json" / "island_groups.json"
    if not ig_path.exists():
        print(
            f"[rebuild_islands] ERROR: json/island_groups.json not found in {scene_dir}. "
            "run_stage3_validation.py must run before rebuild_islands.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ig_data = json.loads(ig_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"[rebuild_islands] ERROR: json/island_groups.json is not valid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    meta = ig_data.get("_planner_meta", {})
    if not isinstance(meta, dict) or meta.get("generated_by") != "run_stage3_validation.py":
        print(
            "[rebuild_islands] ERROR: island_groups.json _planner_meta.generated_by is "
            f"'{meta.get('generated_by')}' — expected 'run_stage3_validation.py'. "
            "Only files produced by run_stage3_validation.py are accepted.",
            file=sys.stderr,
        )
        sys.exit(1)

    image_path = scene_dir / "image.png"
    if image_path.exists():
        import hashlib as _hashlib
        _h = _hashlib.sha256()
        with open(image_path, "rb") as _fh:
            for _chunk in iter(lambda: _fh.read(65536), b""):
                _h.update(_chunk)
        current_sha = _h.hexdigest()
        cached_sha = meta.get("image_sha256", "")
        if cached_sha and cached_sha != current_sha:
            print(
                f"[rebuild_islands] ERROR: island_groups.json _planner_meta.image_sha256 "
                f"({cached_sha[:12]}...) does not match current image.png "
                f"({current_sha[:12]}...). Re-run run_stage3_validation.py.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("[rebuild_islands] island_groups.json meta validated (generated_by + image_sha256 ok)")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <scene_dir>", file=sys.stderr)
        sys.exit(2)

    scene_dir = Path(sys.argv[1]).resolve()
    selected_groups_json = scene_dir / "scene-refine-loop" / "selected_groups.json"
    working_blend = scene_dir / "blend" / "stage3-sub-planned.blend"
    source_blend = scene_dir / "blend" / "blender_scene.blend"
    blender_bin = os.environ.get("SCENE_EVAL_BLENDER", DEFAULT_BLENDER)

    # Defense-in-depth: verify island_groups.json was produced by run_stage3_validation.py.
    _validate_island_groups_meta(scene_dir)

    if not selected_groups_json.exists():
        print(f"[rebuild_islands] ERROR: {selected_groups_json} not found", file=sys.stderr)
        sys.exit(1)
    if not working_blend.exists():
        print(f"[rebuild_islands] ERROR: working blend not found: {working_blend}",
              file=sys.stderr)
        sys.exit(1)
    if not BUILD_ISLAND.exists():
        print(f"[rebuild_islands] ERROR: build_island_canonical.py not found at "
              f"{BUILD_ISLAND}", file=sys.stderr)
        sys.exit(1)
    if working_blend.resolve() == source_blend.resolve():
        print("[rebuild_islands] ERROR: working and source blend are the same file!",
              file=sys.stderr)
        sys.exit(1)

    selected = json.loads(selected_groups_json.read_text())
    groups = selected.get("groups", [])
    if not groups:
        print("[rebuild_islands] no groups selected — nothing to rebuild")
        sys.exit(0)

    src_mtime_before = source_blend.stat().st_mtime if source_blend.exists() else None

    # Probe the working blend once for the scene camera so all islands share the same
    # viewpoint as the reference image.  Gracefully degrades to synthetic camera on failure.
    scene_camera_name = _resolve_scene_camera(blender_bin, working_blend)

    # Load relation_graph.json once so we can pass edge_type/name/evidence per group.
    relation_meta_by_gid: dict[str, dict] = {}
    rg_path = scene_dir / "inputs" / "relation_graph.json"
    if rg_path.exists():
        try:
            rg = json.loads(rg_path.read_text())
            for rg_g in rg.get("groups", []):
                relation_meta_by_gid[rg_g.get("group_id", "")] = {
                    "edge_type": rg_g.get("edge_type"),
                    "name": rg_g.get("name"),
                    "evidence": rg_g.get("evidence"),
                }
        except Exception as e:
            print(f"[rebuild_islands] WARNING: could not parse relation_graph.json: {e}",
                  file=sys.stderr)

    failures = []
    for g in groups:
        group_id = g["group_id"]
        group_dir = Path(g["group_dir"])
        anchor_id = g["anchor_id"]
        members = list(g.get("members", []))

        try:
            _rebuild_one(
                blender_bin=blender_bin,
                working_blend=working_blend,
                group_dir=group_dir,
                group_id=group_id,
                anchor_id=anchor_id,
                members=members,
                camera_name=scene_camera_name,
                relation_meta=relation_meta_by_gid.get(group_id),
            )
        except Exception as e:
            print(f"[rebuild_islands] FAILED {group_id}: {e}", file=sys.stderr)
            failures.append(group_id)

    if source_blend.exists():
        src_mtime_after = source_blend.stat().st_mtime
        if src_mtime_before != src_mtime_after:
            print(
                f"[rebuild_islands] CRITICAL: source blend mtime changed! "
                f"{src_mtime_before} -> {src_mtime_after}",
                file=sys.stderr,
            )
            sys.exit(2)

    # Propagate target_spec from island_groups.json to each group dir.
    # This is the SOLE intent input the island-refiner reads (masked.png is human-debug only).
    island_groups_path = scene_dir / "json" / "island_groups.json"
    if island_groups_path.exists():
        with island_groups_path.open() as f:
            ig = json.load(f)
        target_specs = ig.get("target_spec", {})
        for g in groups:
            gid = g["group_id"]
            group_dir = Path(g["group_dir"])
            spec = target_specs.get(gid)
            if spec is None:
                # Groups not needing island refinement have no target_spec — skip them
                continue
            out = group_dir / "target_spec.json"
            with out.open("w") as f:
                json.dump(spec, f, indent=2)
            print(
                f"[rebuild_islands] wrote {out} (pattern={spec.get('pattern')}, count={spec.get('member_count')})"
            )
    else:
        print(
            f"[rebuild_islands] ERROR: {island_groups_path} not found",
            file=sys.stderr,
        )
        sys.exit(1)

    if MAKE_MASKED.exists():
        cmd = ["python3", str(MAKE_MASKED), "--scene_dir", str(scene_dir), "--force"]
        print(f"[rebuild_islands] generating masked.png via {MAKE_MASKED.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[rebuild_islands] masked.png generation failed (exit {result.returncode})", file=sys.stderr)
            if result.stdout:
                print("--- stdout ---\n" + result.stdout, file=sys.stderr)
            if result.stderr:
                print("--- stderr ---\n" + result.stderr, file=sys.stderr)
            # Treat as fatal: masked.png is required by /island-refiner downstream.
            sys.exit(1)
    else:
        print(f"[rebuild_islands] WARNING: {MAKE_MASKED} not found — skipping masked.png", file=sys.stderr)

    if failures:
        print(f"[rebuild_islands] FAILED for {len(failures)} group(s): {failures}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[rebuild_islands] rebuilt {len(groups)} island(s) from "
          f"{working_blend.name}")


if __name__ == "__main__":
    main()
