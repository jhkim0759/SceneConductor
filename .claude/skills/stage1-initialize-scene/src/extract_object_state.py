#!/usr/bin/env python3
"""Run Qwen-VL on pre-finalize scene paths and write <scene_dir>/object_state.json."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


GENERATE_SCRIPT = (
    Path(__file__).resolve().parent.parent.parent
    / "stage3-sub-scene-analyze-prepare"
    / "src"
    / "generate_object_state_json.py"
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Qwen-VL on pre-finalize paths and write object_state.json."
    )
    ap.add_argument("--scene_dir", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B",
                    help="Vision-language model id or local checkpoint path. "
                         "Default is Qwen3.5-27B. "
                         "Local weights live at ./checkpoints/qwen/Qwen3.5-27B/.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()

    if not GENERATE_SCRIPT.exists():
        raise SystemExit(f"[extract_object_state] missing: {GENERATE_SCRIPT}")

    image_path = scene_dir / "image.png"
    masks_dir = scene_dir / "masks"
    object_class_path = scene_dir / "object_class.json"
    mask_attribute_path = scene_dir / "mask_attribute.json"
    annotated_mask_path = scene_dir / "object_state_annotated_mask.png"
    out_path = scene_dir / "object_state.json"

    for required in (image_path, masks_dir):
        if not required.exists():
            raise SystemExit(f"[extract_object_state] missing required input: {required}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(GENERATE_SCRIPT),
        "--image", str(image_path),
        "--masks_dir", str(masks_dir),
        "--output", str(out_path),
        "--annotated_mask", str(annotated_mask_path),
        "--model", args.model,
        "--gpu", str(args.gpu),
        "--max_new_tokens", str(args.max_new_tokens),
        "--local_files_only",
    ]
    if object_class_path.exists():
        cmd += ["--object_class", str(object_class_path)]
    if mask_attribute_path.exists():
        cmd += ["--mask_attribute", str(mask_attribute_path)]

    print(f"[extract_object_state] running: {' '.join(cmd)}", file=sys.stderr)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"[extract_object_state] generate script failed with rc={rc}")

    if not out_path.exists():
        raise SystemExit(f"[extract_object_state] expected output not found: {out_path}")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    n_objects = len(data.get("objects", []))
    print(f'{{"out": "{out_path}", "n_objects": {n_objects}}}')


if __name__ == "__main__":
    main()
