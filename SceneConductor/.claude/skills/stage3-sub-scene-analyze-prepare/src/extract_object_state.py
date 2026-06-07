#!/usr/bin/env python3
"""Wrap generate_object_state_json.py for the scene-operation-planner skill.

Adds a single concern beyond the underlying script: enforce that the scene_dir
follows the canonical FILE_DIRECTORY layout (image.png at top + inputs/ subtree)
and fail loudly if anything is missing, so the rest of the skill never sees a
half-set-up scene.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


CONTROL_SCRIPT = Path(__file__).resolve().parent / "generate_object_state_json.py"


REQUIRED_PATHS = [
    "image.png",
    "inputs/masks",
    "inputs/mask_attribute.json",
    "inputs/object_class.json",
]


def _check_scene(scene_dir: Path) -> None:
    missing = []
    for rel in REQUIRED_PATHS:
        if not (scene_dir / rel).exists():
            missing.append(rel)
    if missing:
        raise SystemExit(
            f"[extract_object_state] scene_dir is missing required inputs:\n  "
            + "\n  ".join(missing)
            + "\n\nDid you forget to run /stage1-initialize-scene first?"
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate json/object_state.json from image + masks (Qwen-VL)."
    )
    p.add_argument("--scene_dir", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device for Qwen-VL (default 0; pick any with > 16 GiB free)")
    p.add_argument("--model", default="Qwen/Qwen3.5-27B",
                   help="Vision-language model id or local checkpoint path. "
                        "Default is Qwen3.5-27B (dense, multimodal). "
                        "Local weights live at ./checkpoints/qwen/Qwen3.5-27B/.")
    p.add_argument("--local_files_only", action="store_true", default=True,
                   help="Forbid HF downloads (default true; the model is already cached)")
    p.add_argument("--no_local_files_only", dest="local_files_only", action="store_false",
                   help="Allow HF downloads (rare)")
    p.add_argument("--max_new_tokens", type=int, default=1024,
                   help="Generation budget. Default 1024 — the original 768 truncated complex scenes.")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()

    try:
        cached = scene_dir / "inputs" / "object_state.json"
        if cached.exists() and cached.stat().st_size > 0:
            print(
                f"[extract_object_state] reusing cached {cached} from Stage 1; "
                "skipping Qwen-VL inference"
            )
            sys.exit(0)
    except Exception:
        pass  # fall through to normal path

    _check_scene(scene_dir)
    if not CONTROL_SCRIPT.exists():
        raise SystemExit(f"[extract_object_state] missing control script: {CONTROL_SCRIPT}")

    out = scene_dir / "json" / "object_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(CONTROL_SCRIPT),
        "--scene_dir", str(scene_dir),
        "--model", args.model,
        "--gpu", str(args.gpu),
        "--max_new_tokens", str(args.max_new_tokens),
        "--output", str(out),
    ]
    if args.local_files_only:
        cmd.append("--local_files_only")

    print(f"[extract_object_state] running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[extract_object_state] underlying script failed with rc={rc}")

    if not out.exists():
        raise SystemExit(f"[extract_object_state] expected output not found: {out}")
    print(f"[extract_object_state] OK -> {out}")


if __name__ == "__main__":
    main()
