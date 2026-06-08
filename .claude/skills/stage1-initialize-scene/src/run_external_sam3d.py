#!/usr/bin/env python3
"""Run SAM3D from the external sam-3d-objects repository."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")


def parse_args():
    parser = argparse.ArgumentParser(description="Run external sam-3d-objects inference")
    parser.add_argument("--repo_root", type=Path, required=True, help="External sam-3d-objects repo path")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--mask_dir", type=Path, required=True, help="Mask directory or single mask file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, default=None, help="Pipeline config path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stage1_only", action="store_true", help="Only run stage 1")
    parser.add_argument("--no_texture", action="store_true", help="Skip texture baking")
    parser.add_argument("--save_ply", action="store_true", help="Also save Gaussian PLY outputs")
    parser.add_argument("--texture_size", type=int, default=1024, help="Fallback GLB texture size")
    parser.add_argument("--simplify", type=float, default=0.95, help="Fallback GLB simplification ratio")
    return parser.parse_args()


def install_utils3d_compat():
    import utils3d

    try:
        import utils3d.torch as utils3d_torch

        sys.modules.setdefault("utils3d.pt", utils3d_torch)
        if getattr(utils3d, "pt", None) is None:
            utils3d.pt = utils3d_torch
    except Exception:
        pass


def load_masks(mask_path: Path):
    import numpy as np
    from PIL import Image

    if mask_path.is_file():
        mask = np.array(Image.open(mask_path).convert("L"))
        return [(mask > 127).astype(np.uint8) * 255], [mask_path.stem]

    if not mask_path.is_dir():
        raise FileNotFoundError(f"Mask path not found: {mask_path}")

    mask_files = sorted(
        [name for name in os.listdir(mask_path) if re.fullmatch(r"[0-9]+\.png", name)],
        key=lambda name: int(Path(name).stem),
    )
    if not mask_files:
        mask_files = sorted(
            [name for name in os.listdir(mask_path) if name.endswith(".png") and name != "image.png"]
        )

    masks = []
    mask_names = []
    for name in mask_files:
        mask = np.array(Image.open(mask_path / name).convert("L"))
        masks.append((mask > 127).astype(np.uint8))
        mask_names.append(Path(name).stem)
    return masks, mask_names


def main():
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"External sam-3d-objects repo not found: {repo_root}")

    sys.path.insert(0, str(repo_root))
    install_utils3d_compat()

    import torch
    from inference import Inference, load_image
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.config.expanduser().resolve() if args.config else repo_root / "checkpoints" / "hf" / "pipeline.yaml"
    print(f"Loading external SAM3D pipeline from: {config_path}")
    print(f"External SAM3D repo: {repo_root}")

    sam3d = Inference(str(config_path), compile=False)
    image = load_image(str(args.image.expanduser().resolve()))
    print(f"Image loaded: {args.image} ({image.shape})")

    mask_dir = args.mask_dir.expanduser().resolve()
    masks, mask_names = load_masks(mask_dir)
    print(f"Masks loaded: {len(masks)} objects")

    failed_names = []

    for idx, (mask, name) in enumerate(zip(masks, mask_names)):
        print(f"\n[{idx + 1}/{len(masks)}] Processing object '{name}'...")
        rgba = sam3d.merge_mask_to_rgba(image, mask)
        output = sam3d._pipeline.run(
            rgba,
            None,
            seed=args.seed,
            stage1_only=args.stage1_only,
            with_mesh_postprocess=True,
            with_texture_baking=not args.no_texture,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=None,
        )

        glb = output.get("glb")
        if glb is None:
            app_rep = output.get("gaussian")
            if isinstance(app_rep, (list, tuple)):
                app_rep = app_rep[0]
            mesh = output.get("mesh")
            if isinstance(mesh, (list, tuple)):
                mesh = mesh[0]
            if app_rep is not None and mesh is not None:
                glb = postprocessing_utils.to_glb(
                    app_rep,
                    mesh,
                    simplify=args.simplify,
                    texture_size=args.texture_size,
                )

        if glb is not None:
            glb_path = output_dir / f"{name}.glb"
            glb.export(str(glb_path))
            print(f"  GLB saved: {glb_path}")
        else:
            print(f"  GLB failed — will remove mask '{name}'")
            failed_names.append(name)

        gs = output.get("gs")
        if gs is not None and args.save_ply:
            ply_path = output_dir / f"{name}.ply"
            gs.save_ply(str(ply_path))
            print(f"  PLY saved: {ply_path}")

    # Remove masks that have no corresponding GLB
    if failed_names:
        import json as _json
        import numpy as np
        print(f"\nRemoving {len(failed_names)} unused mask(s): {failed_names}")

        # 1. Delete individual mask PNGs
        for name in failed_names:
            mask_png = mask_dir / f"{name}.png"
            if mask_png.exists():
                mask_png.unlink()
                print(f"  Deleted {mask_png.name}")

        # 2. Update mask.npy — zero out pixels for failed masks
        mask_npy_path = mask_dir / "mask.npy"
        if mask_npy_path.exists():
            mask_arr = np.load(str(mask_npy_path))
            for name in failed_names:
                if name.isdigit():
                    pixel_val = int(name) + 1   # 0.png → value 1 in mask.npy
                    mask_arr[mask_arr == pixel_val] = 0
            np.save(str(mask_npy_path), mask_arr.astype(np.float32))
            print(f"  Updated mask.npy")

        # 3. Update label.json — remove failed entries
        label_json_path = mask_dir / "label.json"
        if label_json_path.exists():
            with open(label_json_path, encoding="utf-8") as f:
                payload = _json.load(f)
            failed_values = {int(n) + 1 for n in failed_names if n.isdigit()}
            payload["mask"] = [
                e for e in payload.get("mask", [])
                if e.get("value", 0) not in failed_values
            ]
            with open(label_json_path, "w", encoding="utf-8") as f:
                _json.dump(payload, f, indent=2)
            print(f"  Updated label.json (removed {len(failed_values)} entries)")

    print(f"\nDone! Results saved to: {output_dir}")
    if failed_names:
        print(f"Skipped (no GLB): {failed_names}")


if __name__ == "__main__":
    main()
