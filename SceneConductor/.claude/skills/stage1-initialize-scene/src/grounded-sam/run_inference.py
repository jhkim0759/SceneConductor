#!/usr/bin/env python3
"""
Grounded SAM — Self-contained inference script.
Performs text-driven object segmentation on an image using GroundingDINO + SAM.

Usage:
    python run_inference.py \
        --image /path/to/image.png \
        --prompt "sofa. table. chair." \
        --output_dir ./output/grounded_sam
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as TS
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── sys.path setup: use the Grounded-Segment-Anything inside the skill folder ──
SKILL_DIR = Path(__file__).resolve().parent
GSA_DIR = SKILL_DIR / "Grounded-Segment-Anything"
# Checkpoints live outside .claude/ — see <repo>/checkpoints/README.md
# Resolve checkpoint dir from the canonical DIRECTORYS.yaml registry. This
# script runs in the grounded-sam conda env; if PyYAML is unavailable, fall
# back to the old hardcoded layout so inference still works.
_REPO_ROOT = SKILL_DIR.parents[4]
try:
    import yaml as _yaml

    _DIRS = _yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())

    def _dir(key, default):
        p = Path(_DIRS.get(key, default))
        return p if p.is_absolute() else (_REPO_ROOT / p).resolve()

    CKPT_DIR = _dir("checkpoints_grounded_sam", "./checkpoints/grounded-sam")
except Exception:
    CKPT_DIR = _REPO_ROOT / "checkpoints" / "grounded-sam"

for p in [str(GSA_DIR), str(GSA_DIR / "GroundingDINO"), str(GSA_DIR / "segment_anything")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor

from difflib import SequenceMatcher

# ═══════════════════════════════════════════════════════════════
# Checkpoint management
# ═══════════════════════════════════════════════════════════════
GROUNDING_DINO_CONFIG = str(GSA_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CKPT = str(CKPT_DIR / "groundingdino_swint_ogc.pth")
SAM_CKPT = str(CKPT_DIR / "sam_vit_h_4b8939.pth")

GROUNDING_DINO_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def ensure_checkpoints():
    """Automatically download checkpoints if they are missing."""
    os.makedirs(str(CKPT_DIR), exist_ok=True)

    if not os.path.exists(GROUNDING_DINO_CKPT):
        print(f"Downloading GroundingDINO checkpoint...")
        torch.hub.download_url_to_file(GROUNDING_DINO_URL, GROUNDING_DINO_CKPT)
        print(f"  Saved: {GROUNDING_DINO_CKPT}")

    if not os.path.exists(SAM_CKPT):
        # try symlinking from another location
        alt_paths = []
        copied = False
        for alt in alt_paths:
            if os.path.exists(alt):
                shutil.copy2(alt, SAM_CKPT)
                print(f"  Copied SAM from: {alt}")
                copied = True
                break
        if not copied:
            print(f"Downloading SAM checkpoint...")
            torch.hub.download_url_to_file(SAM_URL, SAM_CKPT)
            print(f"  Saved: {SAM_CKPT}")



def _phrase_name_only(phrase: str) -> str:
    # "chair(0.41)" -> "chair"
    if "(" in phrase:
        return phrase.split("(")[0].strip().lower()
    return phrase.strip().lower()


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def _union_box_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        min(float(box1[0]), float(box2[0])),
        min(float(box1[1]), float(box2[1])),
        max(float(box1[2]), float(box2[2])),
        max(float(box1[3]), float(box2[3])),
    ], dtype=box1.dtype)


def _binary_dilate(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    dilated = cv2.dilate(mask_u8, kernel, iterations=iterations)
    return dilated > 0


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = (dilated > 0) & ~(eroded > 0)
    return boundary


def _boundary_contact_ratio(mask_small: np.ndarray, mask_large: np.ndarray, dilate_iter: int = 1) -> float:
    """
    How much of the small mask's boundary is in contact with the large mask.
    This value tends to be high when the small mask is a fragment.
    """
    b_small = _mask_boundary(mask_small)
    if b_small.sum() == 0:
        return 0.0

    large_dil = _binary_dilate(mask_large, ksize=3, iterations=dilate_iter)
    contact = np.logical_and(b_small, large_dil).sum()
    return float(contact) / float(b_small.sum())


def _label_similarity(a: str, b: str) -> float:
    """
    Simple string-based similarity.
    Examples:
      chair vs chairs
      armchair vs chair
      sofa vs couch  <- purely semantic synonyms like this may score weakly
    """
    a = _phrase_name_only(a)
    b = _phrase_name_only(b)

    if a == b:
        return 1.0

    # exact token overlap bonus
    toks_a = set(a.replace(",", " ").replace(".", " ").split())
    toks_b = set(b.replace(",", " ").replace(".", " ").split())
    if len(toks_a) > 0 and len(toks_b) > 0:
        jacc = len(toks_a & toks_b) / len(toks_a | toks_b)
    else:
        jacc = 0.0

    seq = SequenceMatcher(None, a, b).ratio()

    # take whichever of the two is larger
    return max(seq, jacc)


def _is_label_compatible(label_a: str, label_b: str, same_label_only: bool, label_sim_thresh: float) -> bool:
    a = _phrase_name_only(label_a)
    b = _phrase_name_only(label_b)

    if same_label_only:
        return a == b

    sim = _label_similarity(a, b)
    return sim >= label_sim_thresh


def merge_oversegmented_masks(
    masks,
    boxes,
    pred_phrases,
    scores,
    mask_iou_thresh=0.65,
    box_iou_thresh=0.15,
    area_ratio_for_fragment=0.20,
    boundary_contact_thresh=0.35,
    same_label_only=False,
    label_sim_thresh=0.80,
):
    """
    Over-segmentation merge for Grounded-SAM post-processing.

    Args:
        masks: [N, 1, H, W] torch.Tensor(bool/0,1)
        boxes: [N, 4] xyxy torch.Tensor
        pred_phrases: list[str], e.g. ["chair(0.51)", "armchair(0.44)"]
        scores: [N] torch.Tensor
        mask_iou_thresh:
            merge when the masks themselves overlap heavily
        box_iou_thresh:
            how much the boxes overlap
        area_ratio_for_fragment:
            fragment candidate if the small-mask / large-mask area ratio is at or below this value
        boundary_contact_thresh:
            how much the small mask's boundary is attached to the large mask
        same_label_only:
            if True, merge only exact same labels
            if False, also merge similar labels with similarity >= label_sim_thresh
        label_sim_thresh:
            label similarity threshold

    Returns:
        merged_masks, merged_boxes, merged_phrases, merged_scores
    """
    if masks is None or masks.shape[0] <= 1:
        return masks, boxes, pred_phrases, scores

    masks_np = masks.detach().cpu().numpy()[:, 0].astype(bool)
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()

    N = masks_np.shape[0]
    used = np.zeros(N, dtype=bool)

    merged_masks = []
    merged_boxes = []
    merged_phrases = []
    merged_scores = []

    # anchor starting from the highest score
    order = torch.argsort(scores_cpu, descending=True).tolist()

    for i in order:
        if used[i]:
            continue

        used[i] = True
        cur_mask = masks_np[i].copy()
        cur_box = boxes_cpu[i].clone()
        cur_score = float(scores_cpu[i])
        cur_label = pred_phrases[i]
        cur_name = _phrase_name_only(cur_label)

        changed = True
        while changed:
            changed = False
            cur_area = cur_mask.sum()

            for j in order:
                if used[j]:
                    continue

                other_label = pred_phrases[j]
                other_name = _phrase_name_only(other_label)

                # whether the labels are compatible
                if not _is_label_compatible(
                    cur_name, other_name,
                    same_label_only=same_label_only,
                    label_sim_thresh=label_sim_thresh,
                ):
                    continue

                other_mask = masks_np[j]
                other_box = boxes_cpu[j]
                other_area = other_mask.sum()

                iou_m = _mask_iou(cur_mask, other_mask)
                iou_b = _box_iou_xyxy(cur_box, other_box)

                # decide which is the small side / large side
                if cur_area <= other_area:
                    small_mask, large_mask = cur_mask, other_mask
                    small_area, large_area = cur_area, other_area
                else:
                    small_mask, large_mask = other_mask, cur_mask
                    small_area, large_area = other_area, cur_area

                area_ratio = 0.0 if large_area == 0 else float(small_area) / float(large_area)
                contact_ratio = _boundary_contact_ratio(small_mask, large_mask, dilate_iter=1)

                should_merge = False

                # 1) nearly identical / strong overlap
                if iou_m >= mask_iou_thresh:
                    should_merge = True

                # 2) a small fragment attached to the boundary of a large mask
                elif (
                    area_ratio <= area_ratio_for_fragment
                    and contact_ratio >= boundary_contact_thresh
                    and iou_b >= box_iou_thresh
                ):
                    should_merge = True

                if should_merge:
                    cur_mask = np.logical_or(cur_mask, other_mask)
                    cur_box = _union_box_xyxy(cur_box, other_box)
                    cur_score = max(cur_score, float(scores_cpu[j]))
                    used[j] = True
                    changed = True

        merged_masks.append(cur_mask)
        merged_boxes.append(cur_box)
        merged_scores.append(cur_score)
        merged_phrases.append(f"{cur_name}({cur_score:.3f})")

    merged_masks = torch.from_numpy(np.stack(merged_masks, axis=0)).unsqueeze(1).to(masks.device)
    merged_boxes = torch.stack(merged_boxes, dim=0).to(boxes.device)
    merged_scores = torch.tensor(merged_scores, dtype=scores.dtype, device=scores.device)

    return merged_masks, merged_boxes, merged_phrases, merged_scores
# ═══════════════════════════════════════════════════════════════
# Core functions (from ram_utils/utils.py, Marigold removed)
# ═══════════════════════════════════════════════════════════════
def load_image(image_path):
    """Load the image and apply GroundingDINO preprocessing."""
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_grounding_dino(config_path, ckpt_path, device):
    """Load the GroundingDINO model."""
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    """Text-driven object detection with GroundingDINO."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, tags, mask_list, box_list, label_list):
    """Save mask data (visualization + JSON + NPY)."""
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")

    np.save(os.path.join(output_dir, "mask.npy"), mask_img.numpy())

    json_data = {"tags": tags, "mask": [{"value": 0, "label": "background"}]}

    for label, box in zip(label_list, box_list):
        show_box(box.numpy(), plt.gca(), label)
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]
        json_data["mask"].append({
            "value": value,
            "label": name,
            "logit": float(logit),
            "box": box.numpy().tolist(),
        })

    plt.savefig(os.path.join(output_dir, "mask.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

    with open(os.path.join(output_dir, "label.json"), "w") as f:
        json.dump(json_data, f, indent=2)


def save_individual_masks(masks, output_dir):
    """Save each object mask as an individual PNG (compatible with the SceneConductor pipeline).

    The PNG filename uses the original DINO index (`idx`) as-is — it does not
    renumber even when some masks are dropped. Thus `idx.png` always matches the
    `value = idx + 1` entry in `label.json`. (Previously the code compacted indices
    with `saved_idx`, which caused a bug where a mid-sequence drop shifted subsequent
    ids by one and left stale entries in object_class.json.)
    """
    MIN_MASK_RATIO = 0.0005  # exclude if below 0.05% of all pixels
    SAM3D_INPUT_SIZE = 518    # reference resolution for SAM3D pointmap resize
    mask_dir = output_dir
    for idx in range(masks.shape[0]):
        mask = masks[idx, 0].cpu().numpy().astype(np.uint8) * 255
        ratio = (mask > 0).sum() / mask.size
        if ratio < MIN_MASK_RATIO:
            print(f"  [SKIP] Mask {idx} too small (ratio={ratio:.6f}), skipping.")
            continue
        # check that valid pixels remain even at the resolution SAM3D resizes to internally
        resized = np.array(Image.fromarray(mask).resize(
            (SAM3D_INPUT_SIZE, SAM3D_INPUT_SIZE), Image.NEAREST
        ))
        if (resized > 0).sum() == 0:
            print(f"  [SKIP] Mask {idx} empty after resize to {SAM3D_INPUT_SIZE}x{SAM3D_INPUT_SIZE}, skipping.")
            continue
        mask_path = os.path.join(mask_dir, f"{idx}.png")
        Image.fromarray(mask).save(mask_path)


def mask_and_save(
    image_path, output_dir, tags,
    box_threshold=0.25, text_threshold=0.25, iou_threshold=0.5,
    device="cuda",
    dino_model=None, predictor=None,
):
    """Main segmentation function."""
    os.makedirs(output_dir, exist_ok=True)
    image_pil, image = load_image(image_path)
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    boxes_filt, scores, pred_phrases = get_grounding_output(
        dino_model, image, tags, box_threshold, text_threshold, device=device
    )

    image_cv = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    scores = scores[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    print(f"Detected: {pred_phrases}")

    if boxes_filt.shape[0] == 0:
        print("WARNING: No objects detected!")
        return None, pred_phrases, boxes_filt

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    merge_mask_iou=0.65
    merge_box_iou=0.15
    merge_area_ratio=0.20
    merge_boundary_contact=0.35
    label_sim_thresh = 0.7

    before_n = masks.shape[0]
    masks, boxes_filt, pred_phrases, scores = merge_oversegmented_masks(
            masks=masks,
            boxes=boxes_filt,
            pred_phrases=pred_phrases,
            scores=scores,
            mask_iou_thresh=merge_mask_iou,
            box_iou_thresh=merge_box_iou,
            area_ratio_for_fragment=merge_area_ratio,
            boundary_contact_thresh=merge_boundary_contact,
            same_label_only=False,
            label_sim_thresh=label_sim_thresh,
        )

    print(f"Merged masks: {before_n} -> {masks.shape[0]}")

    print(f"Masks shape: {masks.shape}, Image size: {size}")

    # save results
    save_mask_data(output_dir, tags, masks, boxes_filt, pred_phrases)
    save_individual_masks(masks, output_dir)

    # copy the input image
    image_pil.save(os.path.join(output_dir, "image.png"))

    return masks, pred_phrases, boxes_filt


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Grounded SAM Segmentation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", required=True, help='Text prompt (e.g., "sofa. table. chair.")')
    parser.add_argument("--output_dir", default="./output/grounded_sam", help="Output directory")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # verify/download checkpoints
    ensure_checkpoints()

    # load models
    print("Loading GroundingDINO...")
    dino_model = load_grounding_dino(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CKPT, args.device)

    print("Loading SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(args.device)
    predictor = SamPredictor(sam)

    # inference
    print(f"Processing: {args.image}")
    print(f"Prompt: {args.prompt}")

    masks, pred_phrases, boxes = mask_and_save(
        image_path=args.image,
        output_dir=args.output_dir,
        tags=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
        dino_model=dino_model,
        predictor=predictor,
    )

    if masks is not None:
        print(f"\nDone! {masks.shape[0]} objects detected.")
        print(f"Results saved to: {args.output_dir}")
        print(f"Files: image.png, 0.png~{masks.shape[0]-1}.png (individual masks), mask.npy, label.json, mask.jpg")
    else:
        print("No objects detected.")


if __name__ == "__main__":
    main()
