#!/usr/bin/env python3
"""Crop per-object thumbnails from image.png using masks/<id>.png (pre-finalize paths).

Outputs `thumbnails/obj_<id>_<class>.png`. Skips existing files unless
--force is passed. The thumbnail filename embeds the class so downstream
Stage-3 consumers (planner agents, prep skill) can identify objects by
glancing at file paths.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PADDING_PX = 8


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "unknown"


def crop_one(image_arr: np.ndarray, mask_arr: np.ndarray, padding: int):
    ys, xs = np.where(mask_arr > 0)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(image_arr.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(image_arr.shape[1], int(xs.max()) + padding + 1)
    return image_arr[y0:y1, x0:x1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--padding", type=int, default=PADDING_PX)
    ap.add_argument("--force", action="store_true",
                    help="re-generate thumbnails even if they exist")
    args = ap.parse_args()

    scene = Path(args.scene_dir).resolve()
    image_path = scene / "image.png"
    masks_dir = scene / "masks"
    class_path = scene / "object_class.json"
    out_dir = scene / "thumbnails"

    for required in (image_path, masks_dir, class_path):
        if not required.exists():
            print(f"missing required input: {required}", file=sys.stderr)
            return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB"))
    classes = json.loads(class_path.read_text())

    n_made = 0
    n_skipped = 0
    n_empty = 0

    for mask_file in sorted(masks_dir.glob("*.png")):
        stem = mask_file.stem
        if not stem.isdigit():
            continue
        obj_id = stem
        cls = classes.get(obj_id, "unknown")
        out_path = out_dir / f"obj_{obj_id}_{slugify(cls)}.png"

        if out_path.exists() and not args.force:
            n_skipped += 1
            continue

        mask = np.array(Image.open(mask_file).convert("L"))
        if mask.shape != image.shape[:2]:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]), Image.NEAREST
                )
            )

        crop = crop_one(image, mask, args.padding)
        if crop is None:
            print(f"  warn: empty mask for obj_{obj_id}, skipping")
            n_empty += 1
            continue

        Image.fromarray(crop).save(out_path)
        n_made += 1

    print(f'{{"out_dir": "{out_dir}", "n_thumbnails": {n_made}}}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
