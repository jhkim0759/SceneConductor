#!/usr/bin/env python3
"""Produce object_state_annotated_mask.png from per-object mask PNGs before finalize_layout."""
import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _generate_distinct_colors(n: int, saturation: float = 0.75, value: float = 0.95):
    colors = []
    for i in range(max(n, 1)):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _build_color_map(unique_ids: list[int], background_id: int = 0):
    foreground_ids = [idx for idx in unique_ids if idx != background_id]
    foreground_colors = _generate_distinct_colors(len(foreground_ids))
    color_map = {background_id: (0, 0, 0)}
    for idx, color in zip(foreground_ids, foreground_colors):
        color_map[idx] = color
    return color_map


def _colorize_mask(mask: np.ndarray, color_map: dict) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        color_mask[mask == idx] = color
    return color_mask


def _enumerate_per_object_pngs(masks_dir: Path) -> list[tuple[int, Path]]:
    return sorted(
        (
            (int(f.stem), f)
            for f in masks_dir.glob("*.png")
            if f.stem.isdigit() and f.name != "mask.png"
        ),
        key=lambda pair: pair[0],
    )


def _reconstruct_integer_mask_and_objects_from_pngs(
    masks_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    entries = _enumerate_per_object_pngs(masks_dir)
    if not entries:
        raise FileNotFoundError(f"No per-object PNGs found in {masks_dir}")

    binaries: list[tuple[int, np.ndarray, int]] = []
    shape = None
    for label_id, png_path in entries:
        binary = np.array(Image.open(png_path).convert("L")) > 0
        if shape is None:
            shape = binary.shape
        elif binary.shape != shape:
            raise ValueError(
                f"Inconsistent PNG shapes: {png_path} {binary.shape} != {shape}"
            )
        area = int(binary.sum())
        if area == 0:
            print(f"[make-annotated-mask] WARNING: {png_path.name} is empty; skipping", file=sys.stderr)
            continue
        binaries.append((label_id, binary, area))

    if not binaries:
        raise ValueError(f"All per-object PNGs in {masks_dir} are empty")

    h, w = shape
    label_arr = np.zeros((h, w), dtype=np.int32)
    for label_id, binary, _ in sorted(binaries, key=lambda e: -e[2]):
        label_arr[binary] = label_id

    objects = []
    for label_id, binary, area in binaries:
        ys, xs = np.where(binary)
        objects.append({
            "obj_id": f"obj_{label_id}",
            "label_id": label_id,
            "center": (int(xs.mean()), int(ys.mean())),
            "pixel_count": area,
        })
    return label_arr, objects


def _draw_text_with_bg(draw, pos, text, font, text_fill=(255, 255, 255), bg_fill=(0, 0, 0)):
    x, y = pos
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=bg_fill)
    draw.text((x, y), text, font=font, fill=text_fill)


def create_annotated_mask(
    scene_image_path: Path,
    masks_dir: Path,
    object_class_path: Path | None,
    out_path: Path,
    blend_alpha: float = 0.5,
) -> dict:
    mask, objects = _reconstruct_integer_mask_and_objects_from_pngs(masks_dir)
    print(
        f"[make-annotated-mask] {len(objects)} objects from {masks_dir}",
        file=sys.stderr,
    )

    unique_ids = sorted(np.unique(mask).tolist())
    color_map = _build_color_map(unique_ids, background_id=0)
    color_mask = _colorize_mask(mask, color_map)

    scene_img = Image.open(scene_image_path).convert("RGB")
    if scene_img.size != (mask.shape[1], mask.shape[0]):
        scene_img = scene_img.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
    scene_arr = np.array(scene_img, dtype=np.float32)

    mask_arr = color_mask.astype(np.float32)
    is_foreground = (mask != 0)[..., np.newaxis]
    blended = np.where(
        is_foreground,
        scene_arr * (1.0 - blend_alpha) + mask_arr * blend_alpha,
        scene_arr,
    ).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(blended)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    class_map: dict[str, str] = {}
    if object_class_path is not None and object_class_path.exists():
        raw = json.loads(object_class_path.read_text(encoding="utf-8"))
        class_map = {str(k): str(v) for k, v in raw.items()}

    for obj in objects:
        label = obj["obj_id"]
        cx, cy = obj["center"]
        _draw_text_with_bg(draw, (cx, cy), label, font)

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return {"annotated_mask_path": str(out_path), "objects": objects, "color_map": color_map}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Produce object_state_annotated_mask.png from per-object mask PNGs."
    )
    ap.add_argument("--scene_dir", required=True, type=Path)
    ap.add_argument("--masks_dir", type=Path, default=None)
    ap.add_argument("--object_class", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()
    masks_dir = (args.masks_dir or scene_dir / "masks").resolve()
    object_class_path = (args.object_class or scene_dir / "object_class.json").resolve()
    out_path = (args.out or scene_dir / "object_state_annotated_mask.png").resolve()

    image_path = scene_dir / "image.png"
    if not image_path.exists():
        print(f"[make-annotated-mask] missing image: {image_path}", file=sys.stderr)
        return 1
    if not masks_dir.is_dir():
        print(f"[make-annotated-mask] masks_dir not found: {masks_dir}", file=sys.stderr)
        return 1

    result = create_annotated_mask(image_path, masks_dir, object_class_path, out_path)
    print(f'{{"out": "{result["annotated_mask_path"]}", "n_objects": {len(result["objects"])}}}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
