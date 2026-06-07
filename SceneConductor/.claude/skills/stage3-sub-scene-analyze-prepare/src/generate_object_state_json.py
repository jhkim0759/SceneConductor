#!/usr/bin/env python3
"""Generate object-state JSON (attachment / alignment / stacking) from an RGB image and an integer mask with Qwen VL.

For each visible object id on the annotated mask, this script asks Qwen VL to evaluate:
  1. Attachment: which structural surfaces (floor / wall / ceiling / none) the object is attached to.
  2. Alignment groups: groups of objects whose rotation+scale can be aligned, with yaw subgroups (0/90/180/270 deg).
     If a `merge_plan.json` is provided, its `merge_groups` are passed as the candidate connectivity hint.
  3. Stacking: which objects are physically supported on top of another object (not on floor/wall/ceiling).
"""

import argparse
import colorsys
import json
import os
import re
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
ENABLE_THINKING = False

ALLOWED_ATTACHMENTS = ("floor", "wall", "ceiling", "none")
ALLOWED_YAW_KEYS = ("0", "90", "180", "270")


CUSTOM_SYSTEM_PROMPT = dedent(
    """
    ## System Prompt

    You are a precise scene-understanding assistant.

    ### Inputs
    You are given:
    1. a scene image
    2. an annotated segmentation mask

    Each segmented object has a visible object ID written directly on the mask, such as `obj_1`, `obj_2`, `obj_3`.
    You may additionally receive a `Connectivity hint` block listing object IDs that are pre-grouped as connected.

    ---

    ### Rules
    1. Use only the visible object IDs written on the mask.
    2. Do not invent new object IDs.
    3. Do not merge different object IDs.
    4. First infer the most likely short category name for each object ID.
    5. Then evaluate the three tasks below using object IDs only.
    6. Do not write explanations or commentary.
    7. Every visible object ID must appear in the [Object Labels] block AND the [Attachment] block.
    8. If `Connectivity hint` groups are given, use them as the starting point for [Alignment Groups];
       otherwise infer alignment groups yourself from visual evidence.

    ---

    ### Tasks

    #### Task 1 - Attachment
    For every object, list the structural surfaces it is physically attached to or rests on.
    Allowed values (multi-select, comma-separated):
      - `floor`   : the object stands on or is fixed to the floor / ground.
      - `wall`    : the object is mounted on or pressed flush against a wall.
      - `ceiling` : the object hangs from or is fixed to the ceiling.
      - `none`    : the object is not directly attached to any structural surface
                    (e.g. a book on a table, a vase on a shelf).
    Use `none` only when no structural surface applies. Multiple values are allowed
    (e.g. a tall shelf may be `floor, wall`).

    Format (one line per object):
        obj_i = surface_a, surface_b

    #### Task 2 - Alignment Groups
    Group object IDs whose global rotation and scale frames are compatible, i.e. it would be
    safe to copy one object's rotation+scale onto the others (possibly after a 90/180/270 deg yaw step).
    For each group report:
      - `members`            : every object id in the group (a single-object group is allowed).
      - `rotation_alignable` : true if all members can share the same rotation frame (after yaw multiples of 90 deg).
      - `scale_alignable`    : true if all members can share the same scale.
      - yaw subgroups        : partition members by their yaw offset (0 / 90 / 180 / 270 deg)
                               relative to the group's reference orientation.
    Every member of the group must appear in exactly one yaw subgroup. Empty subgroups must still
    appear with an empty value.

    Format (one block per group, blank line between blocks):
        group 1:
          members = obj_i, obj_j, obj_k
          rotation_alignable = true
          scale_alignable = true
          yaw_0 = obj_i, obj_j
          yaw_90 = obj_k
          yaw_180 =
          yaw_270 =

    #### Task 3 - Stacking
    List every case where an object physically rests on top of another object
    (NOT on the floor / wall / ceiling). The supporting object is `base`,
    the supported object(s) are `top`.

    Format (one line per stacking relation):
        base = obj_x | top = obj_y, obj_z

    If no stacking relation exists, write the section header followed by the single line `(none)`.

    ---

    ### Output Format (use exactly these section headers, in this order)

    [Object Labels]
    obj_1 = category
    obj_2 = category

    [Attachment]
    obj_1 = floor
    obj_2 = wall, ceiling

    [Alignment Groups]
    group 1:
      members = obj_1, obj_2
      rotation_alignable = true
      scale_alignable = true
      yaw_0 = obj_1
      yaw_90 = obj_2
      yaw_180 =
      yaw_270 =

    [Stacking]
    base = obj_1 | top = obj_3
    """
).strip()


def _resolve_scene_paths(scene_dir: Path) -> dict[str, Path | None]:
    """Resolve standard scene_dir layout paths.

    Expected layout (matches ./test_dataset/<scene>/):
      <scene_dir>/image.png
      <scene_dir>/inputs/masks/mask.png
      <scene_dir>/inputs/merge_plan.json
      <scene_dir>/inputs/object_class.json     (singular; plural also accepted)
    Outputs:
      <scene_dir>/inputs/object_state.json
      <scene_dir>/inputs/object_state_annotated_mask.png
    """
    scene_dir = scene_dir.resolve()
    inputs_dir = scene_dir / "inputs"

    def _first_existing(candidates: list[Path]) -> Path | None:
        for path in candidates:
            if path.exists():
                return path
        return None

    image = _first_existing([scene_dir / "image.png", scene_dir / "image.jpg"])
    mask = _first_existing(
        [
            inputs_dir / "masks" / "mask.png",
            inputs_dir / "mask.png",
            inputs_dir / "masks" / "mask.npy",
            inputs_dir / "mask.npy",
        ]
    )
    merge_plan = _first_existing([inputs_dir / "merge_plan.json"])
    mask_attribute = _first_existing(
        [inputs_dir / "mask_attribute.json"]
    )
    object_class = _first_existing(
        [inputs_dir / "object_class.json"]
    )
    masks_dir = inputs_dir / "masks"
    if not masks_dir.is_dir():
        masks_dir = None
    return {
        "image": image,
        "mask": mask,
        "masks_dir": masks_dir,
        "merge_plan": merge_plan,
        "mask_attribute": mask_attribute,
        "object_class": object_class,
        "output": inputs_dir / "object_state.json",
        "annotated_mask": inputs_dir / "object_state_annotated_mask.png",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate object-state JSON (attachment / alignment / stacking) from image + mask with Qwen VL"
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=None,
        help="Scene directory. When given, image / mask / merge_plan / object_class / output / annotated_mask "
             "are auto-derived from the standard layout (image.png, inputs/masks/mask.png, inputs/merge_plan.json, "
             "inputs/object_class.json). Explicit flags below override the auto-derived paths.",
    )
    parser.add_argument("--image", type=Path, default=None, help="Input RGB image path (required if --scene_dir is omitted)")
    parser.add_argument(
        "--masks_dir",
        type=Path,
        default=None,
        help="Directory of per-object PNGs (1.png, 2.png, ...). When given (or auto-derived from "
             "scene_dir/inputs/masks/), the integer mask is rebuilt from these PNGs and obj_id "
             "is set equal to label_id (e.g. 5.png -> obj_5). This survives upstream merge steps "
             "without re-sorting and matches mesh_groups instance_ids 1:1.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Fallback integer mask file (.png or .npy). Used only when --masks_dir is unavailable. "
             "Note: under this mode obj_id is sequential (obj_1..obj_N) and may not equal label_id.",
    )
    parser.add_argument(
        "--mask_attribute",
        type=Path,
        default=None,
        help="Optional mask_attribute.json (POST-MERGE ids; preferred). "
             "Auto-derived from scene_dir/inputs/mask_attribute.json. "
             "When present, its `mesh_groups` (already remapped to current ids) drive the connectivity hint.",
    )
    parser.add_argument(
        "--merge_plan",
        type=Path,
        default=None,
        help="Optional Phase-B merge_plan.json (PRE-MERGE ids). "
             "Used only as a fallback when --mask_attribute is unavailable. "
             "Auto-derived from scene_dir/inputs/merge_plan.json when --scene_dir is given.",
    )
    parser.add_argument(
        "--object_class",
        type=Path,
        default=None,
        help="Optional object_class.json mapping {label_id|obj_id: class}. Auto-derived from scene_dir/inputs/object_class.json. "
             "When provided, real class names are passed in the prompt registry instead of just label ids",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: <scene_dir>/inputs/object_state.json or <mask_stem>_object_state.json",
    )
    parser.add_argument(
        "--annotated_mask",
        type=Path,
        default=None,
        help="Path to save the annotated mask. Default: <scene_dir>/inputs/object_state_annotated_mask.png "
             "or .cache/annotated_mask.png",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_QWEN_MODEL, help="Qwen VL model id")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/processor from local Hugging Face cache or local path only",
    )
    parser.add_argument("--background_id", type=int, default=0, help="Background label id in the mask")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Generation token budget")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p")
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow loading the Qwen model on CPU when CUDA is unavailable",
    )
    parser.add_argument(
        "--no_sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling for deterministic decoding",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Optional physical GPU id to expose via CUDA_VISIBLE_DEVICES",
    )
    parser.set_defaults(do_sample=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Annotated-mask construction (same logic as generate_connectivity_json.py)
# ---------------------------------------------------------------------------


def _generate_distinct_colors(n: int, saturation: float = 0.75, value: float = 0.95):
    colors = []
    for i in range(max(n, 1)):
        hue = i / max(n, 1)
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(red * 255), int(green * 255), int(blue * 255)))
    return colors


def _load_integer_mask(mask_path: Path) -> np.ndarray:
    if str(mask_path).endswith(".npy"):
        mask = np.load(str(mask_path)).astype(np.int32)
    else:
        mask = np.array(Image.open(mask_path))
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D integer mask, got shape={mask.shape} from {mask_path}")
    return mask


def _build_color_map(unique_ids: list[int], background_id: int = 0):
    foreground_ids = [idx for idx in unique_ids if idx != background_id]
    foreground_colors = _generate_distinct_colors(len(foreground_ids))
    color_map = {background_id: (0, 0, 0)}
    for idx, color in zip(foreground_ids, foreground_colors):
        color_map[idx] = color
    return color_map


def _colorize_mask(mask: np.ndarray, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        color_mask[mask == idx] = color
    return color_mask


def _find_object_centers_one_id_one_object(mask: np.ndarray, background_id: int = 0):
    objects = []
    counter = 1
    for label_id in sorted(np.unique(mask)):
        if int(label_id) == background_id:
            continue
        ys, xs = np.where(mask == label_id)
        if len(xs) == 0:
            continue
        center_x = int(xs.mean())
        center_y = int(ys.mean())
        objects.append(
            {
                "obj_id": f"obj_{counter}",
                "label_id": int(label_id),
                "center": (center_x, center_y),
                "pixel_count": int(len(xs)),
            }
        )
        counter += 1
    return objects


def _enumerate_per_object_pngs(masks_dir: Path) -> list[tuple[int, Path]]:
    """Return [(label_id, png_path)] for every numerically named PNG (excluding mask.png)."""
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
    """Reconstruct integer mask AND per-object metadata from individual PNGs.

    Two correctness properties guaranteed:
      (1) Every non-empty PNG yields one object record -- nothing is dropped due to overlap.
          Each object's center/pixel_count is computed from its own binary mask.
      (2) obj_id equals label_id (e.g. masks/5.png -> obj_5), so the mapping is stable
          across the upstream merge step and matches mesh_groups instance_ids 1:1.

    For the visualization mask only, overlapping pixels are painted in descending-area order
    so smaller masks land on top and stay visible (the rule that merge_masks.py inverts).
    """
    entries = _enumerate_per_object_pngs(masks_dir)
    if not entries:
        raise FileNotFoundError(f"No per-object PNGs (1.png, 2.png, ...) found in {masks_dir}")

    binaries: list[tuple[int, np.ndarray, int]] = []
    shape: tuple[int, int] | None = None
    for label_id, png_path in entries:
        binary = np.array(Image.open(png_path).convert("L")) > 0
        if shape is None:
            shape = binary.shape
        elif binary.shape != shape:
            raise ValueError(
                f"Per-object PNGs have inconsistent shapes: {png_path} {binary.shape} != {shape}"
            )
        area = int(binary.sum())
        if area == 0:
            print(
                f"[object-state] WARNING: {png_path.name} is empty; skipping",
                file=sys.stderr,
            )
            continue
        binaries.append((label_id, binary, area))

    if not binaries:
        raise ValueError(f"All per-object PNGs in {masks_dir} are empty")

    height, width = shape  # type: ignore[misc]
    label_arr = np.zeros((height, width), dtype=np.int32)
    # Paint largest first, smallest last -- keeps small masks visible in the visualization
    for label_id, binary, _ in sorted(binaries, key=lambda e: -e[2]):
        label_arr[binary] = label_id

    objects: list[dict] = []
    for label_id, binary, area in binaries:
        ys, xs = np.where(binary)
        objects.append(
            {
                "obj_id": f"obj_{label_id}",
                "label_id": label_id,
                "center": (int(xs.mean()), int(ys.mean())),
                "pixel_count": area,
            }
        )
    return label_arr, objects


def _draw_text_with_bg(
    draw: ImageDraw.ImageDraw,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_fill=(255, 255, 255),
    bg_fill=(0, 0, 0),
):
    x_pos, y_pos = pos
    bbox = draw.textbbox((x_pos, y_pos), text, font=font)
    pad = 2
    rect = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(rect, fill=bg_fill)
    draw.text((x_pos, y_pos), text, font=font, fill=text_fill)


def create_annotated_mask(
    mask_path: Path | None,
    save_path: Path,
    background_id: int = 0,
    masks_dir: Path | None = None,
    image_path: Path | None = None,
    blend_alpha: float = 0.5,
):
    """Build the annotated visualization mask and the obj_i registry.

    Source priority:
      1. masks_dir/<id>.png per-object PNGs   -> obj_id = obj_{label_id}, no objects dropped.
      2. mask_path integer label map (legacy) -> obj_id = obj_{counter}, sequential.

    When ``image_path`` is provided, the foreground regions are blended with the
    scene image at ``blend_alpha`` and the background remains the original image
    (matches make_annotated_mask.py output).
    """
    if masks_dir is not None and masks_dir.is_dir():
        mask, objects = _reconstruct_integer_mask_and_objects_from_pngs(masks_dir)
        print(
            f"[object-state] mask source: per-object PNGs in {masks_dir} "
            f"(obj_id = obj_<label_id>, no re-sorting)",
            file=sys.stderr,
        )
    else:
        if mask_path is None:
            raise ValueError("Either masks_dir or mask_path must be provided")
        mask = _load_integer_mask(mask_path)
        objects = _find_object_centers_one_id_one_object(mask, background_id=background_id)
        print(
            f"[object-state] mask source: integer mask {mask_path} (obj_id = sequential obj_1..obj_N)",
            file=sys.stderr,
        )

    unique_ids = sorted(np.unique(mask).tolist())
    color_map = _build_color_map(unique_ids, background_id=background_id)
    color_mask = _colorize_mask(mask, color_map)

    if image_path is not None:
        scene_img = Image.open(image_path).convert("RGB")
        if scene_img.size != (mask.shape[1], mask.shape[0]):
            scene_img = scene_img.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
        scene_arr = np.array(scene_img, dtype=np.float32)
        mask_arr = color_mask.astype(np.float32)
        is_foreground = (mask != background_id)[..., np.newaxis]
        blended = np.where(
            is_foreground,
            scene_arr * (1.0 - blend_alpha) + mask_arr * blend_alpha,
            scene_arr,
        ).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(blended)
    else:
        image = Image.fromarray(color_mask)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for obj in objects:
        center_x, center_y = obj["center"]
        _draw_text_with_bg(draw, (center_x, center_y), obj["obj_id"], font)

    save_path = save_path.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    return {
        "annotated_mask_path": str(save_path),
        "objects": objects,
        "color_map": color_map,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def load_object_class_map(object_class_path: Path | None) -> dict[int, str]:
    """Load object_class.json into a {label_id: class_name} dict.

    Accepts either of the two known schemas:
      - {"<label_id>": "class"}      (label_id-keyed)
      - {"obj_<n>": "class"}         (obj_id-keyed, n is the obj_id index, NOT label_id)
    The obj_id-keyed variant cannot be reliably mapped to label_ids without the mask, so we ignore it
    here (the label_id-keyed variant is preferred when both shapes are possible).
    """
    if object_class_path is None or not object_class_path.exists():
        return {}
    with open(object_class_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {}
    result: dict[int, str] = {}
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        try:
            label_id = int(key)
        except ValueError:
            continue
        result[label_id] = value
    return result


def build_object_mapping_text(objects: list[dict], class_map: dict[int, str] | None = None) -> str:
    class_map = class_map or {}
    lines = []
    for obj in objects:
        cls = class_map.get(obj["label_id"])
        if cls:
            lines.append(f'{obj["obj_id"]} = {cls} (label_id {obj["label_id"]})')
        else:
            lines.append(f'{obj["obj_id"]} = region with label_id {obj["label_id"]}')
    return "\n".join(lines)


def _label_id_to_obj_id(objects: list[dict]) -> dict[int, str]:
    return {obj["label_id"]: obj["obj_id"] for obj in objects}


def _normalize_member_token(token, label_id_to_obj_id: dict[int, str]) -> str | None:
    """Map a merge_plan member entry (label_id, str-int, or 'obj_n') to an obj_id string."""
    if isinstance(token, str):
        token_stripped = token.strip()
        if token_stripped.startswith("obj_"):
            return token_stripped
        try:
            label_id = int(token_stripped)
        except ValueError:
            return None
        return label_id_to_obj_id.get(label_id)
    if isinstance(token, (int, np.integer)):
        return label_id_to_obj_id.get(int(token))
    return None


def _build_mesh_group_hint_lines(
    mesh_groups,
    label_id_to_obj_id: dict[int, str],
    *,
    source_label: str,
) -> list[str]:
    """Translate a mesh_groups dict into instance_group hint lines (current ids only)."""
    hint_lines: list[str] = []
    if isinstance(mesh_groups, dict):
        mesh_iter = mesh_groups.items()
    elif isinstance(mesh_groups, list):
        mesh_iter = ((str(i), g) for i, g in enumerate(mesh_groups))
    else:
        return hint_lines

    for index, (name, group) in enumerate(mesh_iter, start=1):
        if not isinstance(group, dict):
            continue
        members_raw = group.get("instance_ids") or group.get("members") or group.get("ids") or []
        members_obj_ids = [
            _normalize_member_token(tok, label_id_to_obj_id) for tok in members_raw
        ]
        members_obj_ids = [m for m in members_obj_ids if m]
        if len(members_obj_ids) < 2:
            continue
        cls = group.get("class") or name
        hint_lines.append(
            f"instance_group {index} [{cls}] (separate instances of the same class, "
            f"expect alignable rotation+scale): " + ", ".join(members_obj_ids)
        )
    return hint_lines


def build_connectivity_hint_text(
    mask_attribute_path: Path | None,
    merge_plan_path: Path | None,
    objects: list[dict],
) -> str | None:
    """Translate post-merge mesh_groups into a connectivity hint.

    Source priority:
      1. mask_attribute.json["mesh_groups"]      -- POST-MERGE ids (preferred; matches current mask)
      2. merge_plan.json["history"][...].id_remap to translate merge_plan IDs to current ids
         (only if mask_attribute.json is missing)

    merge_plan.json["mesh_groups"] alone is intentionally NOT used: those ids are pre-merge
    and silently mis-map onto the post-merge mask (e.g. a chair id that became a picture-frame id
    after renumbering would corrupt the alignment hint).
    """
    label_id_to_obj_id = _label_id_to_obj_id(objects)

    # ---- 1) mask_attribute.json (preferred) -----------------------------------
    if mask_attribute_path is not None and mask_attribute_path.exists():
        with open(mask_attribute_path, "r", encoding="utf-8") as fh:
            attr = json.load(fh)
        mesh_groups = attr.get("mesh_groups") or {}
        hint_lines = _build_mesh_group_hint_lines(
            mesh_groups, label_id_to_obj_id, source_label="mask_attribute.json"
        )
        if hint_lines:
            print(
                f"[object-state] connectivity hint source: {mask_attribute_path} (post-merge ids)",
                file=sys.stderr,
            )
            return "\n".join(hint_lines)

    # ---- 2) merge_plan.json fallback (remap pre-merge -> current via id_remap) ----
    if merge_plan_path is None or not merge_plan_path.exists():
        if merge_plan_path is not None:
            print(
                f"[object-state] WARNING: merge_plan not found at {merge_plan_path}; no connectivity hint",
                file=sys.stderr,
            )
        return None

    with open(merge_plan_path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)

    print(
        f"[object-state] WARNING: mask_attribute.json unavailable; "
        f"falling back to {merge_plan_path} -- pre-merge ids will be ignored unless an id_remap is found "
        f"inside the plan or a sibling mask_attribute history.",
        file=sys.stderr,
    )

    id_remap_str = plan.get("id_remap") or {}
    # Try sibling mask_attribute.json history if the plan itself has no id_remap
    if not id_remap_str and merge_plan_path is not None:
        sibling = merge_plan_path.parent / "mask_attribute.json"
        if sibling.exists():
            with open(sibling, "r", encoding="utf-8") as fh:
                sibling_attr = json.load(fh)
            for entry in reversed(sibling_attr.get("history", []) or []):
                if entry.get("step") == "merge":
                    id_remap_str = (entry.get("plan") or {}).get("id_remap") or {}
                    if id_remap_str:
                        break

    def _remap(old_id) -> int | None:
        if not id_remap_str:
            return int(old_id) if str(old_id).isdigit() else None
        mapped = id_remap_str.get(str(old_id))
        return int(mapped) if mapped is not None else None

    mesh_groups = plan.get("mesh_groups") or {}
    if not isinstance(mesh_groups, dict):
        return None

    remapped: dict[str, dict] = {}
    for name, group in mesh_groups.items():
        if not isinstance(group, dict):
            continue
        new_canonical = _remap(group.get("canonical_id"))
        new_instances = [
            new_id for new_id in (_remap(i) for i in group.get("instance_ids", [])) if new_id is not None
        ]
        if new_canonical is None and new_instances:
            new_canonical = new_instances[0]
        if new_canonical is None or len(new_instances) < 2:
            continue
        remapped[name] = {
            "canonical_id": new_canonical,
            "instance_ids": new_instances,
            "class": group.get("class", name),
        }

    hint_lines = _build_mesh_group_hint_lines(
        remapped, label_id_to_obj_id, source_label="merge_plan.json (remapped)"
    )
    return "\n".join(hint_lines) if hint_lines else None


def build_user_prompt(object_mapping_text: str, connectivity_hint_text: str | None) -> str:
    hint_block = ""
    if connectivity_hint_text:
        hint_block = dedent(
            f"""

            Connectivity hint (use these as the starting point for [Alignment Groups]):
            {connectivity_hint_text}
            """
        )
    return dedent(
        f"""
        Analyze the scene image together with the annotated segmentation mask.

        Visible object registry:
        {object_mapping_text}{hint_block}

        Tasks:
        1. Read the visible object IDs from the annotated mask.
        2. Infer a short category for each object ID.
        3. For every object, list its structural attachment surfaces
           (any of: floor, wall, ceiling, none -- multi-select).
        4. Build alignment groups (rotation+scale alignable) and split each group
           into yaw subgroups (yaw_0, yaw_90, yaw_180, yaw_270).
           Use the Connectivity hint when available.
        5. List stacking relations where one object rests on top of another object
           (not on floor / wall / ceiling). Write `(none)` if there are no such relations.

        Output exactly in this format (use these section headers in this order):

        [Object Labels]
        obj_1 = category
        obj_2 = category

        [Attachment]
        obj_1 = floor
        obj_2 = wall, ceiling

        [Alignment Groups]
        group 1:
          members = obj_1, obj_2
          rotation_alignable = true
          scale_alignable = true
          yaw_0 = obj_1
          yaw_90 = obj_2
          yaw_180 =
          yaw_270 =

        [Stacking]
        base = obj_1 | top = obj_3
        """
    ).strip()


def build_messages(user_prompt: str, image_paths: list[Path], system_prompt: str | None = None):
    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )

    user_content = [{"type": "image", "image": str(path)} for path in image_paths]
    user_content.append({"type": "text", "text": user_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Qwen VL loading + inference
# ---------------------------------------------------------------------------


def load_qwen(model_id: str, local_files_only: bool = False, allow_cpu: bool = False):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable in the current environment. "
            f"Loading {model_id} on CPU is likely impractical. "
            "Run in a GPU-enabled environment or pass --allow_cpu to override."
        )

    dtype = torch.bfloat16 if cuda_available else torch.float32
    load_kwargs = {}
    if local_files_only:
        load_kwargs["local_files_only"] = True

    try:
        processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load Qwen processor. Use --model with a local model path, "
            "or rerun with --local_files_only after the model is cached locally."
        ) from exc

    model_kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            **model_kwargs,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            **model_kwargs,
            **load_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load Qwen model. Use --model with a local model path, "
            "or rerun with --local_files_only after the model is cached locally."
        ) from exc

    model.eval()
    model_device = next(model.parameters()).device
    return torch, processor, model, model_device


def run_qwen(
    torch_module,
    processor,
    model,
    model_device,
    user_prompt: str,
    image_paths: list[Path],
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    messages = build_messages(
        user_prompt=user_prompt,
        image_paths=image_paths,
        system_prompt=system_prompt,
    )

    apply_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        inputs = processor.apply_chat_template(
            messages,
            enable_thinking=ENABLE_THINKING,
            **apply_kwargs,
        )
    except TypeError:
        inputs = processor.apply_chat_template(messages, **apply_kwargs)

    inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["temperature"] = 1.0
        generation_kwargs["top_p"] = 1.0
        generation_kwargs["top_k"] = 50

    with torch_module.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return {
        "messages": messages,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


SECTION_HEADERS = ("[Object Labels]", "[Attachment]", "[Alignment Groups]", "[Stacking]")


def _split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {header: [] for header in SECTION_HEADERS}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped in SECTION_HEADERS:
            current = stripped
            continue
        if current is None:
            continue
        sections[current].append(line)
    return sections


def _parse_obj_id_list(value: str) -> list[str]:
    if value is None:
        return []
    items = [token.strip() for token in value.split(",")]
    return [token for token in items if token.startswith("obj_")]


def parse_object_labels(lines: list[str]) -> dict[str, str]:
    pattern = re.compile(r"^\s*(obj_\d+)\s*=\s*(.+?)\s*$")
    labels: dict[str, str] = {}
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        obj_id, category = match.groups()
        labels[obj_id] = category
    return labels


def parse_attachment(lines: list[str]) -> dict[str, list[str]]:
    pattern = re.compile(r"^\s*(obj_\d+)\s*=\s*(.+?)\s*$")
    result: dict[str, list[str]] = {}
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        obj_id, surfaces_text = match.groups()
        surfaces = [s.strip().lower() for s in surfaces_text.split(",") if s.strip()]
        surfaces = [s for s in surfaces if s in ALLOWED_ATTACHMENTS]
        if not surfaces:
            surfaces = ["none"]
        if "none" in surfaces and len(surfaces) > 1:
            surfaces = [s for s in surfaces if s != "none"]
        # de-dup preserve order
        seen: set[str] = set()
        ordered: list[str] = []
        for s in surfaces:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        result[obj_id] = ordered
    return result


def parse_alignment_groups(lines: list[str]) -> list[dict]:
    group_header_pattern = re.compile(r"^\s*group\s+(\d+)\s*:\s*$", re.IGNORECASE)
    kv_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

    groups: list[dict] = []
    current: dict | None = None

    def finalize(group: dict) -> dict:
        members = group.get("members", [])
        yaw_groups = {key: group.get(f"yaw_{key}", []) for key in ALLOWED_YAW_KEYS}
        return {
            "group_id": group["group_id"],
            "members": members,
            "rotation_alignable": group.get("rotation_alignable", False),
            "scale_alignable": group.get("scale_alignable", False),
            "yaw_groups": yaw_groups,
        }

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current is not None and current.get("members"):
                groups.append(finalize(current))
                current = None
            continue
        header_match = group_header_pattern.match(stripped)
        if header_match:
            if current is not None and current.get("members"):
                groups.append(finalize(current))
            current = {"group_id": int(header_match.group(1))}
            continue
        if current is None:
            continue
        kv_match = kv_pattern.match(line)
        if not kv_match:
            continue
        key, value = kv_match.groups()
        key_lower = key.lower()
        if key_lower == "members":
            current["members"] = _parse_obj_id_list(value)
        elif key_lower in ("rotation_alignable", "scale_alignable"):
            current[key_lower] = value.strip().lower() == "true"
        elif key_lower in (f"yaw_{deg}" for deg in ALLOWED_YAW_KEYS):
            current[key_lower] = _parse_obj_id_list(value)
        # silently ignore unknown keys

    if current is not None and current.get("members"):
        groups.append(finalize(current))

    return groups


def parse_stacking(lines: list[str]) -> list[dict]:
    pattern = re.compile(
        r"^\s*base\s*=\s*(obj_\d+)\s*\|\s*top\s*=\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    relations: list[dict] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "(none)":
            return []
        match = pattern.match(line)
        if not match:
            continue
        base_id, top_text = match.groups()
        top_ids = _parse_obj_id_list(top_text)
        if not top_ids:
            continue
        relations.append({"base": base_id, "top": top_ids})
    return relations


def parse_object_state(text: str, objects: list[dict]) -> dict:
    sections = _split_sections(text)

    labels = parse_object_labels(sections["[Object Labels]"])
    attachments = parse_attachment(sections["[Attachment]"])
    alignment_groups = parse_alignment_groups(sections["[Alignment Groups]"])
    stacking = parse_stacking(sections["[Stacking]"])

    object_records: list[dict] = []
    for obj in objects:
        obj_id = obj["obj_id"]
        object_records.append(
            {
                "obj_id": obj_id,
                "label_id": obj["label_id"],
                "category": labels.get(obj_id, "unknown"),
                "attached_to": attachments.get(obj_id, ["none"]),
            }
        )

    return {
        "objects": object_records,
        "alignment_groups": alignment_groups,
        "stacking": stacking,
    }


def save_object_state_json(parsed: dict, output_json_path: Path) -> dict:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(parsed, file, ensure_ascii=False, indent=2)
    return parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_runtime_paths(args) -> dict:
    scene_paths: dict[str, Path | None] = {}
    if args.scene_dir is not None:
        scene_dir = args.scene_dir.resolve()
        if not scene_dir.exists():
            raise FileNotFoundError(f"--scene_dir does not exist: {scene_dir}")
        scene_paths = _resolve_scene_paths(scene_dir)

    def _pick(arg_value, scene_key, *, required=False, label=""):
        if arg_value is not None:
            return arg_value.resolve()
        candidate = scene_paths.get(scene_key)
        if candidate is None and required:
            raise ValueError(
                f"Could not resolve {label}: pass --{scene_key} explicitly or supply --scene_dir with the standard layout."
            )
        return candidate.resolve() if candidate else None

    image_path = _pick(args.image, "image", required=True, label="image")
    masks_dir_path = _pick(args.masks_dir, "masks_dir")
    mask_path = _pick(args.mask, "mask")
    if masks_dir_path is None and mask_path is None:
        raise ValueError(
            "No mask source: pass --masks_dir, --mask, or --scene_dir (with inputs/masks/)."
        )
    merge_plan_path = _pick(args.merge_plan, "merge_plan")
    mask_attribute_path = _pick(args.mask_attribute, "mask_attribute")
    object_class_path = _pick(args.object_class, "object_class")

    output_path = args.output.resolve() if args.output else None
    if output_path is None:
        if scene_paths.get("output") is not None:
            output_path = scene_paths["output"].resolve()
        elif mask_path is not None:
            output_path = mask_path.with_name(f"{mask_path.stem}_object_state.json")
        else:
            output_path = masks_dir_path.parent / "object_state.json"

    annotated_mask_path = args.annotated_mask.resolve() if args.annotated_mask else None
    if annotated_mask_path is None:
        annotated_mask_path = (
            scene_paths["annotated_mask"].resolve()
            if scene_paths.get("annotated_mask") is not None
            else Path(".cache/annotated_mask.png").resolve()
        )

    return {
        "image": image_path,
        "masks_dir": masks_dir_path,
        "mask": mask_path,
        "merge_plan": merge_plan_path,
        "mask_attribute": mask_attribute_path,
        "object_class": object_class_path,
        "output": output_path,
        "annotated_mask": annotated_mask_path,
    }


def main():
    args = parse_args()
    paths = _resolve_runtime_paths(args)

    image_path = paths["image"]
    masks_dir_path = paths["masks_dir"]
    mask_path = paths["mask"]
    merge_plan_path = paths["merge_plan"]
    mask_attribute_path = paths["mask_attribute"]
    object_class_path = paths["object_class"]
    output_path = paths["output"]
    annotated_mask_path = paths["annotated_mask"]

    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if masks_dir_path is None and (mask_path is None or not mask_path.exists()):
        raise FileNotFoundError("No mask source available (need masks_dir or mask)")

    print(f"[object-state] image          = {image_path}")
    print(f"[object-state] masks_dir      = {masks_dir_path}")
    print(f"[object-state] mask (fallback)= {mask_path}")
    print(f"[object-state] mask_attribute = {mask_attribute_path}")
    print(f"[object-state] merge_plan     = {merge_plan_path}")
    print(f"[object-state] object_class   = {object_class_path}")
    print(f"[object-state] output         = {output_path}")
    print(f"[object-state] annotated      = {annotated_mask_path}")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    annotated = create_annotated_mask(
        mask_path=mask_path,
        save_path=annotated_mask_path,
        background_id=args.background_id,
        masks_dir=masks_dir_path,
        image_path=image_path,
    )
    annotated_mask_path = Path(annotated["annotated_mask_path"])
    print(f"Annotated mask saved to {annotated_mask_path}")
    if not annotated_mask_path.exists():
        raise FileNotFoundError(annotated_mask_path)

    class_map = load_object_class_map(object_class_path)
    object_mapping_text = build_object_mapping_text(annotated["objects"], class_map)
    connectivity_hint_text = build_connectivity_hint_text(
        mask_attribute_path, merge_plan_path, annotated["objects"]
    )
    user_prompt = build_user_prompt(object_mapping_text, connectivity_hint_text)

    torch_module, processor, model, model_device = load_qwen(
        args.model,
        local_files_only=args.local_files_only,
        allow_cpu=args.allow_cpu,
    )
    response = run_qwen(
        torch_module=torch_module,
        processor=processor,
        model=model,
        model_device=model_device,
        user_prompt=user_prompt,
        image_paths=[image_path, annotated_mask_path],
        system_prompt=CUSTOM_SYSTEM_PROMPT,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    parsed = parse_object_state(response["text"], annotated["objects"])
    save_object_state_json(parsed, output_path)

    summary = {
        "model": args.model,
        "scene_dir": str(args.scene_dir.resolve()) if args.scene_dir else None,
        "image": str(image_path),
        "masks_dir": str(masks_dir_path) if masks_dir_path else None,
        "mask": str(mask_path) if mask_path else None,
        "mask_attribute": str(mask_attribute_path) if mask_attribute_path else None,
        "merge_plan": str(merge_plan_path) if merge_plan_path else None,
        "object_class": str(object_class_path) if object_class_path else None,
        "annotated_mask": annotated["annotated_mask_path"],
        "output_json": str(output_path),
        "object_count": len(parsed["objects"]),
        "alignment_group_count": len(parsed["alignment_groups"]),
        "stacking_count": len(parsed["stacking"]),
        "raw_response": response["text"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[object-state-json] ERROR: {exc}", file=sys.stderr)
        raise
