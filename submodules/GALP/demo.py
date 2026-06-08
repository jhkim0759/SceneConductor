#!/usr/bin/env python
"""
GALP Demo: predict 3D object layout from a single RGB image + SAM3D meshes.

Usage
-----
    python demo.py
    python demo.py --scene assets/0000000 \
                   --ckpt  checkpoints/checkpoint.pt \
                   --output output/demo_scene.glb \
                   --gpu 0
"""
import os
import sys
import argparse
from pathlib import Path

# ── path setup (must precede any src.* imports) ──────────────────────────────
FILE_ROOT = Path(__file__).resolve().parent
for p in (FILE_ROOT, FILE_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import numpy as np
import torch
import trimesh
from PIL import Image
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchvision.transforms import functional as TF

from src.datasets.data_utils import (
    mesh_to_voxel_tensor,
    preprocess_image,
    listdict_to_dictlist_safe,
)
from src.train import (
    ConditionEmbedding,
    init_ss_condition_embedder,
    init_ss_generator_v1_4,
    load_trellis_ss_wrapper,
    strip_module_prefix,
)
from src.utils.inference_utils import build_scene, run_moge

IMAGE_SIZE = (518, 518)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(ckpt_path: str, device: torch.device):
    # NOTE: <GALP>/checkpoints is a symlink to <repo>/checkpoints/galp, so the
    # README's `checkpoints/galp/<file>` layout is reached here as `checkpoints/<file>`.
    cfg_yaml        = "checkpoints/galp.yaml"
    cond_embed_ckpt = "checkpoints/condition_embedder.ckpt"
    enc_ckpt        = os.path.join("checkpoints", "ss_enc_conv3d_16l8_fp16")

    train_cfg  = OmegaConf.load("configs/mp8_nt512.yaml")
    pipe_cfg   = OmegaConf.load("checkpoints/pipeline.yaml")["ss_preprocessor"]
    preprocessor = instantiate(pipe_cfg)

    encoder = load_trellis_ss_wrapper(enc_ckpt, kind="encoder", device=device)

    generator, _, _ = init_ss_generator_v1_4(
        cfg_yaml, None,
        device=device,
        resolution=train_cfg["dataset"]["voxel_resolution"] // 4,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(strip_module_prefix(ckpt), strict=True)
    generator.eval()

    cond_embedder = init_ss_condition_embedder(cfg_yaml, cond_embed_ckpt, device=device)
    cond_embed_fn = ConditionEmbedding(
        cond_embedder, train_cfg.get("ss_condition_input_mapping", [])
    )
    return encoder, generator, cond_embed_fn, preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Asset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scene(scene_dir: str, device: torch.device, preprocessor, pointmap: torch.Tensor):
    scene_dir = Path(scene_dir)

    image_path = next(scene_dir.glob("image.*"))
    mask_paths = sorted(
        scene_dir.glob("masks/*_mask.png"),
        key=lambda p: int(p.stem.split("_")[0]),
    )
    obj_paths = sorted(
        scene_dir.glob("objects/*.glb"),
        key=lambda p: int(p.stem),
    )
    assert len(mask_paths) == len(obj_paths), (
        f"mask / object count mismatch: {len(mask_paths)} masks, {len(obj_paths)} objects"
    )

    image_pil = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    image = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).to(device) / 255.0  # [3,H,W]

    items, voxels, meshes = [], [], []
    for mask_path, obj_path in zip(mask_paths, obj_paths):
        mask_arr = np.array(Image.open(mask_path).resize(IMAGE_SIZE, Image.NEAREST))
        if mask_arr.ndim > 2:
            mask_arr = mask_arr[:, :, -1]
        mask = torch.from_numpy(mask_arr > 0).to(device)  # [H,W] bool

        rgba = torch.cat([image, mask.unsqueeze(0).float()], dim=0)  # [4,H,W]
        item = preprocess_image(rgba, preprocessor, pointmap=pointmap)
        items.append(item)

        mesh = trimesh.load(str(obj_path), force="mesh")
        voxel, _ = mesh_to_voxel_tensor(mesh, resolution=16, version="old")
        voxels.append(voxel)
        meshes.append(mesh)

    # collate
    items = listdict_to_dictlist_safe(items)
    for k in items:
        items[k] = torch.stack(items[k])
    items["voxels"]    = torch.stack(voxels)
    items["num_parts"] = torch.tensor([len(voxels)], device=device, dtype=torch.long)

    for k in items:
        if k == "num_parts":
            items[k] = items[k].to(device).long()
        elif isinstance(items[k], torch.Tensor):
            items[k] = items[k].to(device).float()

    return items, meshes


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(items, meshes, encoder, generator, cond_embed_fn):
    condition_args, _ = cond_embed_fn(items)
    cond = condition_args[0] if condition_args else None

    shape_latent = encoder(items["voxels"])["z"]
    shape_latent = shape_latent.reshape(items["voxels"].shape[0], 8, -1).transpose(1, 2)

    outputs = generator({"shape": shape_latent}, cond,
                        num_parts=items["num_parts"], cond_pm=None)

    for k in list(outputs.keys()):
        outputs[k] = outputs[k].detach().squeeze(1)
        if k == "scale":
            outputs[k] = outputs[k].mean(-1, keepdim=True)

    outputs["translation"]         = outputs.get("pred_translation",         outputs["translation"])
    outputs["6drotation_normalized"] = outputs.get("pred_6drotation_normalized", outputs["6drotation_normalized"])
    outputs["num_parts"] = items["num_parts"]
    outputs["voxels"]    = items["voxels"]
    outputs["meshes"]    = meshes
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GALP inference demo")
    parser.add_argument("--scene",  default="assets/0000000",
                        help="Scene directory (contains image.*, masks/, objects/)")
    parser.add_argument("--ckpt",   default=os.environ.get("GALP_CKPT", "checkpoints/checkpoint.pt"),
                        help="Trained GALP checkpoint (.pt). Overridable via $GALP_CKPT")
    parser.add_argument("--output", default="output/demo_scene.glb",
                        help="Output GLB path")
    parser.add_argument("--gpu",    default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Path(args.output).parent, exist_ok=True)

    # ── 1. models ────────────────────────────────────────────────────────────
    print("[1/4] Loading models ...")
    encoder, generator, cond_embed_fn, preprocessor = load_models(args.ckpt, device)

    # ── 2. pointmap ──────────────────────────────────────────────────────────
    print("[2/4] Computing pointmap (MoGe) ...")
    image_path = str(next(Path(args.scene).glob("image.*")))
    pm_np = run_moge(image_path)                                        # [H,W,3]
    pointmap = torch.from_numpy(pm_np).permute(2, 0, 1).float().to(device)  # [3,H,W]
    pointmap = TF.resize(pointmap.unsqueeze(0), list(IMAGE_SIZE),
                         interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)
    _min, _max = pointmap.flatten(1).min(1).values, pointmap.flatten(1).max(1).values
    centered  = pointmap - ((_min + _max) / 2).view(3, 1, 1)
    pointmap  = centered / centered.max()

    # ── 3. assets ────────────────────────────────────────────────────────────
    print("[3/4] Loading scene assets ...")
    items, meshes = load_scene(args.scene, device, preprocessor, pointmap)
    print(f"      {len(meshes)} objects")

    # ── 4. inference ─────────────────────────────────────────────────────────
    print("[4/4] Running inference ...")
    outputs = run_inference(items, meshes, encoder, generator, cond_embed_fn)

    scene = build_scene(outputs)
    scene.export(args.output)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
