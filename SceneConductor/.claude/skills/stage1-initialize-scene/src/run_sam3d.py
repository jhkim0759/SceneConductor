#!/usr/bin/env python3
"""
Stage 1 — SAM3D runner.

Calls run_external_sam3d.py (sam3d-objects conda env), bridging our pipeline's
1-indexed masks to SAM3D's 0-indexed input convention via a temp directory of
symlinks, then converts native GLB outputs to OBJ and enforces failure alignment.

Reads:
    <scene_dir>/image.png
    <scene_dir>/masks/1.png .. N.png
    <scene_dir>/object_class.json

Writes:
    <scene_dir>/inputs/object/1.glb .. M.glb   (M <= N after failures, textured GLB)

If any objects fail, renumbers masks/ and object_class.json so indices stay
contiguous 1..M.

Usage:
    python run_sam3d.py \\
        --scene_dir /path/to/scene \\
        [--gpu 0] \\
        [--conda_env_name sam3d-objects] \\
        [--simplify 0.95]
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


def _dir(key, default):
    p = Path(_DIRS.get(key, default))
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


# ── Hardcoded paths (overridable via CLI) ────────────────────────────────────
SAM3D_SCRIPT = Path(__file__).resolve().parent / "run_external_sam3d.py"
# SAM3D repo is vendored as a git submodule under <repo>/submodules/SAM3D.
# This file lives at <repo>/.claude/skills/stage1-initialize-scene/src/run_sam3d.py,
# so the project root is 4 levels up from this file.
SAM3D_REPO = _dir("sam3d_repo", "./submodules/SAM3D")
CONDA_ENV_NAME_DEFAULT = _DIRS["conda_envs"]["sam3d-objects"]
SAM3D_CONFIG = SAM3D_REPO / "checkpoints" / "hf" / "pipeline.yaml"


# ── Preflight checks ─────────────────────────────────────────────────────────

def preflight(conda_env_name: str, scene_dir: Path) -> list[Path]:
    """Return sorted list of input mask paths (1.png..N.png); exit on error."""
    errors = []

    if not conda_env_name:
        errors.append("Conda env name is empty")
    if not SAM3D_SCRIPT.exists():
        errors.append(f"SAM3D script not found: {SAM3D_SCRIPT}")
    if not SAM3D_REPO.exists():
        errors.append(f"SAM3D repo not found: {SAM3D_REPO}")
    if not SAM3D_CONFIG.exists():
        errors.append(f"SAM3D pipeline.yaml not found: {SAM3D_CONFIG}")

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        errors.append(f"Input image not found: {image_path}")

    masks_dir = scene_dir / "masks"
    mask_files = sorted(
        [p for p in masks_dir.iterdir() if p.stem.isdigit() and p.suffix == ".png"]
        if masks_dir.exists() else [],
        key=lambda p: int(p.stem),
    )
    if not mask_files:
        errors.append(f"No numbered mask PNGs found in {masks_dir}")

    obj_class_path = scene_dir / "object_class.json"
    if not obj_class_path.exists():
        errors.append(f"object_class.json not found: {obj_class_path}")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    return mask_files


# ── Failure alignment helper ─────────────────────────────────────────────────

def drop_failed_after_sam3d(
    scene_dir: Path,
    original_mask_files: list[Path],
    succeeded_one_indices: list[int],
) -> None:
    """
    Drop failed objects WITHOUT renumbering survivors.

    Mask IDs are now stable across the pipeline: a survivor that started life
    as mask 27 stays mask 27 even if masks 5/9/14 fail. This matches the
    'preserve original ids' contract introduced in merge_masks.py.

    original_mask_files  — the 1-indexed mask paths that SAM3D was run on
    succeeded_one_indices — original IDs that produced a valid GLB

    After this function:
      masks/<id>.png      — only survivor PNGs remain; failed PNGs are deleted
      masks/mask.png      — NOT regenerated (leave as-is; caller may do this)
      object/<id>.glb     — already written by caller using original IDs
      object_class.json   — failed entries dropped, survivor ids preserved
      mask_attribute.json — objects/mesh_groups updated to drop failed ids
    """
    masks_dir = scene_dir / "masks"
    obj_class_path = scene_dir / "object_class.json"

    # Load original class map
    with open(obj_class_path, encoding="utf-8") as f:
        orig_class: dict[str, str] = json.load(f)

    survivors = sorted(succeeded_one_indices)
    if not survivors:
        print("[WARN] All objects failed — clearing object_class.json.", file=sys.stderr)
        with open(obj_class_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        # Remove every numbered mask
        for p in masks_dir.glob("*.png"):
            if p.stem.isdigit():
                p.unlink()
        return

    # Determine which original IDs were attempted but failed.
    attempted_ids = {int(p.stem) for p in original_mask_files if p.stem.isdigit()}
    failed_ids = sorted(attempted_ids - set(survivors))

    # Delete failed mask PNGs only — survivor PNGs stay with their ORIGINAL filenames.
    for fid in failed_ids:
        p = masks_dir / f"{fid}.png"
        if p.exists():
            p.unlink()

    # Drop failed entries from object_class.json; survivor keys (= original ids) preserved.
    new_class: dict[str, str] = {
        str(sid): orig_class.get(str(sid), "unknown") for sid in survivors
    }
    with open(obj_class_path, "w", encoding="utf-8") as f:
        json.dump(new_class, f, indent=2)

    # Update mask_attribute.json: drop failed objects + filter mesh_groups members.
    attr_path = scene_dir / "mask_attribute.json"
    if attr_path.exists():
        with open(attr_path, encoding="utf-8") as f:
            attr = json.load(f)

        survivor_str = {str(s) for s in survivors}
        attr["objects"] = {
            k: v for k, v in attr.get("objects", {}).items() if k in survivor_str
        }

        # Filter mesh_groups: drop instances whose ids didn't survive,
        # promote a new canonical when the original canonical failed,
        # and remove groups that lost every member.
        new_groups = {}
        for gname, ginfo in (attr.get("mesh_groups") or {}).items():
            kept_instances = [i for i in ginfo.get("instance_ids", []) if int(i) in set(survivors)]
            if not kept_instances:
                continue
            canonical = ginfo.get("canonical_id")
            if int(canonical) not in set(survivors):
                canonical = kept_instances[0]
            new_groups[gname] = {
                "canonical_id": int(canonical),
                "instance_ids": [int(i) for i in kept_instances],
                "class": ginfo.get("class", ""),
            }
        attr["mesh_groups"] = new_groups

        with open(attr_path, "w", encoding="utf-8") as f:
            json.dump(attr, f, indent=2)

    print(f"  Dropped failed indices: {failed_ids}")
    print(f"  Survivors (original ids preserved): {survivors}")
    print(f"  Updated object_class.json: {new_class}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 SAM3D runner — wraps run_external_sam3d.py with pipeline contract"
    )
    p.add_argument("--scene_dir", required=True, type=Path,
                   help="Scene directory containing image.png + masks/")
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index (default: 0)")
    p.add_argument("--conda_env_name", type=str,
                   default=CONDA_ENV_NAME_DEFAULT,
                   help="Name of the conda env to run SAM3D in (default: sam3d-objects)")
    p.add_argument("--simplify", type=float, default=0.95,
                   help="Mesh simplification ratio passed to SAM3D (default: 0.95)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    conda_env_name = args.conda_env_name

    mask_files = preflight(conda_env_name, scene_dir)
    n_objects = len(mask_files)
    print(f"[run_sam3d] GPU={args.gpu}  scene={scene_dir}")
    print(f"[run_sam3d] Input masks: {n_objects} objects")

    object_dir = scene_dir / "inputs" / "object"
    object_dir.mkdir(parents=True, exist_ok=True)

    # ── Read mask_attribute.json's mesh_groups for dedup (if any) ────────────
    # Mask-Evaluator agent writes mesh_groups when it decides two masks share
    # the same 3D model. Without it, we process every mask (fallback).
    # Class-name-based dedup is NEVER done here — only the agent decides.
    canonical_to_instances: dict[int, list[int]] = {}
    attr_path = scene_dir / "mask_attribute.json"
    if attr_path.exists():
        with open(attr_path, "r", encoding="utf-8") as f:
            attr = json.load(f)
        for grp_name, ginfo in attr.get("mesh_groups", {}).items():
            cid = int(ginfo["canonical_id"])
            iids = [int(i) for i in ginfo.get("instance_ids", [cid])]
            canonical_to_instances[cid] = iids

    # instance_id -> canonical_id (reverse lookup)
    instance_to_canonical: dict[int, int] = {}
    for cid, iids in canonical_to_instances.items():
        for iid in iids:
            if iid != cid:
                instance_to_canonical[iid] = cid

    # Filter mask_files to canonicals only (if dedup active)
    if canonical_to_instances:
        canonical_ids = set(canonical_to_instances.keys())
        # Every mask that's NOT an instance-of-another-canonical is processed
        skipped = {iid for iid in instance_to_canonical if iid not in canonical_ids}
        mask_files_to_run = [p for p in mask_files if int(p.stem) not in skipped]
        print(f"[run_sam3d] mesh_groups active: {len(canonical_to_instances)} groups, "
              f"{len(skipped)} instance(s) will be copied from canonicals "
              f"(SAM3D runs on {len(mask_files_to_run)}/{n_objects})")
    else:
        mask_files_to_run = mask_files

    # ── Build 0-indexed symlink dir so SAM3D sees 0.png..N-1.png ─────────────
    with tempfile.TemporaryDirectory(prefix="sam3d_masks_0idx_") as tmp_masks_str:

        tmp_masks = Path(tmp_masks_str)

        for zero_idx, mask_path in enumerate(mask_files_to_run):
            link = tmp_masks / f"{zero_idx}.png"
            link.symlink_to(mask_path.resolve())

        # ── Call run_external_sam3d.py ────────────────────────────────────────
        # Discover gcc-12 from the sam3d conda env (base conda has gcc-15 which
        # CUDA 12.x rejects). We inject CC/CXX AFTER conda activation using the
        # `env` utility so they override whatever conda's activate script set.
        _sam3d_prefix = subprocess.check_output(
            ["conda", "run", "-n", conda_env_name, "python", "-c",
             "import sys; print(sys.prefix)"],
            text=True
        ).strip()
        _sam3d_bin = Path(_sam3d_prefix) / "bin"
        _gcc12 = str(_sam3d_bin / "gcc")
        _gxx12 = str(_sam3d_bin / "g++")

        cmd = [
            "conda", "run", "-n", conda_env_name,
            # `env` runs AFTER conda activation, so these override activation-set CC/CXX
            "env",
            f"CC={_gcc12}",
            f"CXX={_gxx12}",
            "NVCC_PREPEND_FLAGS=--allow-unsupported-compiler",
            "python",
            str(SAM3D_SCRIPT),
            "--repo_root", str(SAM3D_REPO),
            "--image", str(scene_dir / "image.png"),
            "--mask_dir", str(tmp_masks),
            "--output_dir", str(object_dir),
            "--seed", "42",
            "--simplify", str(args.simplify),
        ]
        # Textures MUST be baked — do NOT pass --no_texture. Output is textured GLB.

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.setdefault("LIDRA_SKIP_INIT", "1")
        env.setdefault("ATTN_BACKEND", "sdpa")
        env.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

        print(f"[run_sam3d] Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            print(f"[ERROR] run_external_sam3d.py exited with code {result.returncode}",
                  file=sys.stderr)
            sys.exit(result.returncode)

        # ── Collect textured GLBs (1-indexed, no conversion) ─────────────────
        # SAM3D outputs 0.glb..(K-1).glb for the K masks we ran (canonicals + non-dedup).
        # We keep them as GLB with baked textures — no OBJ conversion.
        # After that, fan out canonical GLBs to dedup'd instance paths.
        #
        # FIX: Use a temp staging dir to avoid rename-aliasing when zero_idx and
        # one_idx overlap (e.g. chain 0→1, 1→2, 2→3 in the same dir corrupts GLBs).
        succeeded_one_indices: list[int] = []

        with tempfile.TemporaryDirectory(prefix="sam3d_rename_stage_") as _stage_str:
            stage_dir = Path(_stage_str)
            # Pass 1: copy all 0-indexed GLBs out of object_dir into stage_dir with 1-indexed names
            for zero_idx, mask_path in enumerate(mask_files_to_run):
                one_idx = int(mask_path.stem)
                src_glb = object_dir / f"{zero_idx}.glb"
                if not src_glb.exists():
                    print(f"  [FAIL] No GLB for object {one_idx} (zero-idx={zero_idx})")
                    continue
                shutil.copy2(src_glb, stage_dir / f"{one_idx}.glb")
                succeeded_one_indices.append(one_idx)

            # Pass 2: clear 0-indexed GLBs from object_dir and move staged files in
            for p in object_dir.glob("*.glb"):
                p.unlink()
            for one_idx in succeeded_one_indices:
                shutil.move(str(stage_dir / f"{one_idx}.glb"), object_dir / f"{one_idx}.glb")
                print(f"  [OK]   object {one_idx} → inputs/object/{one_idx}.glb")

        # Fan out: for each instance, copy the canonical's GLB to instance path
        if canonical_to_instances:
            for cid, iids in canonical_to_instances.items():
                if cid not in succeeded_one_indices:
                    continue  # canonical failed, skip the whole group
                canonical_glb = object_dir / f"{cid}.glb"
                for iid in iids:
                    if iid == cid:
                        continue
                    inst_glb = object_dir / f"{iid}.glb"
                    shutil.copy2(canonical_glb, inst_glb)
                    print(f"  [COPY] object {iid} ← canonical {cid} "
                          f"→ object/{iid}.glb (dedup)")
                    succeeded_one_indices.append(iid)

    # ── Failure alignment: drop failed ids, KEEP survivor ids stable ──────────
    n_success = len(succeeded_one_indices)
    n_fail = n_objects - n_success

    if n_fail > 0:
        print(f"\n[run_sam3d] {n_fail} object(s) failed. Dropping failed ids "
              f"(survivors keep their original numbering — no renumbering).")
        # Survivor GLBs were already written under their ORIGINAL ids
        # (see line ~285+ where each GLB is named after its mask id),
        # so no GLB renaming is needed here. Just delete any stray failed GLBs
        # if SAM3D happened to write a corrupted one.
        survivor_set = set(succeeded_one_indices)
        for p in object_dir.glob("*.glb"):
            if p.stem.isdigit() and int(p.stem) not in survivor_set:
                p.unlink()

        drop_failed_after_sam3d(scene_dir, mask_files, succeeded_one_indices)
    else:
        print(f"\n[run_sam3d] All {n_success} object(s) succeeded — no cleanup needed.")

    print(f"\n[run_sam3d] Done. {n_success}/{n_objects} object(s) written to {object_dir}/")


if __name__ == "__main__":
    main()
