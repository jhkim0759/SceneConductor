"""
run_galp.py — Standalone GALP inference runner.

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n sceneconductor python \\
        ./.claude/skills/stage1-initialize-scene/src/run_galp.py \\
        --scene_dir /path/to/scene_dir

Input contract (1-indexed, GLB meshes with baked textures):
    <scene_dir>/image.png
    <scene_dir>/masks/1.png  2.png  ... N.png   (binary per-object masks)
    <scene_dir>/object/1.glb 2.glb  ... N.glb   (per-object textured meshes)
    <scene_dir>/object_class.json               ({"1": "chair", ...})

Outputs:
    <scene_dir>/layout_prediction.json
    <scene_dir>/layout-prediction.glb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
os.environ['CUDA_VISIBLE_DEVICES']="0"
# ---------------------------------------------------------------------------
# Environment setup — must happen before any CUDA/torch import
# ---------------------------------------------------------------------------

def _apply_env(gpu: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("LIDRA_SKIP_INIT", "1")


# ---------------------------------------------------------------------------
# Path wiring — add GALP repo and its src/ to sys.path
# ---------------------------------------------------------------------------

RUNTIME_DIR = Path(__file__).resolve().parent / "galp_runtime"
# GALP repo is vendored as a git submodule under <repo>/submodules/GALP.
# This file lives at <repo>/.claude/skills/stage1-initialize-scene/src/run_galp.py,
# so the project root is 4 levels up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIRS = yaml.safe_load((_REPO_ROOT / "DIRECTORYS.yaml").read_text())


def _dir(key, default):
    p = Path(_DIRS.get(key, default))
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


GALP_REPO = _dir("galp_repo", "./submodules/GALP")

def _wire_paths():
    for p in (GALP_REPO, GALP_REPO / "src", RUNTIME_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Lazy imports (all done after env + path setup)
# ---------------------------------------------------------------------------

def _import_heavy():
    global torch, F, np, Image, trimesh, OmegaConf, instantiate
    global tf, rotation_6d_to_matrix
    global mesh_to_voxel_tensor, preprocess_image, listdict_to_dictlist_safe
    global init_ss_generator, load_trellis_ss_wrapper, init_ss_condition_embedder
    global ConditionEmbedding, strip_module_prefix
    global build_scene
    global MoGeModel, look_at_view_transform, Transform3d
    global intrinsic_to_blender_focal_mm

    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import trimesh
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from torchvision.transforms import functional as tf
    from pytorch3d.transforms import rotation_6d_to_matrix, Transform3d
    from pytorch3d.renderer import look_at_view_transform

    from src.datasets.data_utils import (
        mesh_to_voxel_tensor,
        preprocess_image,
        listdict_to_dictlist_safe,
    )
    from src.train import (
        init_ss_generator_v1_4 as init_ss_generator,
        load_trellis_ss_wrapper,
        ConditionEmbedding,
        strip_module_prefix,
        init_ss_condition_embedder,
    )
    from inference_utils import build_scene
    from moge.model.v1 import MoGeModel

    def intrinsic_to_blender_focal_mm(K, sensor_width_mm=36.0):
        # K is normalized (MoGe convention): fx_norm, cx_norm=0.5
        # focal_mm = fx_norm * sensor_width_mm (equivalent to fx_px/W * 36)
        K = np.asarray(K, dtype=float)
        fx = K[0, 0]
        image_width = K[0, 2] * 2  # = 1.0 for normalized intrinsics
        return fx * sensor_width_mm / float(image_width)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(ckpt_path: Path, device: torch.device):
    """Load all models needed for GALP inference."""
    log = logging.getLogger(__name__)

    # All checkpoints live flat under GALP_REPO/checkpoints/ (= checkpoints/galp/).
    galp_ckpt_dir = _dir("checkpoints_galp", "./checkpoints/galp")
    configs_dir = GALP_REPO / "configs"

    ss_generator_config = str(galp_ckpt_dir / "galp.yaml")
    trellis_encoder_base = str(galp_ckpt_dir / "ss_enc_conv3d_16l8_fp16")

    # --- global config (for voxel_resolution) ---
    global_cfg = OmegaConf.load(str(configs_dir / "mp8_nt512.yaml"))
    voxel_res = global_cfg["dataset"]["voxel_resolution"] // 4  # == 4

    # --- ss_preprocessor (from pipeline.yaml) ---
    pipeline_cfg = OmegaConf.load(str(galp_ckpt_dir / "pipeline.yaml"))
    ss_preprocessor = instantiate(pipeline_cfg["ss_preprocessor"])
    log.info("ss_preprocessor loaded")

    # --- Trellis encoder ---
    ss_encoder = load_trellis_ss_wrapper(trellis_encoder_base, kind="encoder", device=device)
    ss_encoder.eval()
    log.info("Trellis encoder loaded")

    # --- MoGe depth model ---
    log.info("Loading MoGe...")
    moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    moge.eval()
    log.info("MoGe loaded")

    # --- ss_generator + trained checkpoint ---
    ss_generator, _, _ = init_ss_generator(
        ss_generator_config,
        None,
        device=device,
        resolution=voxel_res,
    )
    log.info("Loading trained checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ss_generator.load_state_dict(strip_module_prefix(ckpt), strict=True)
    ss_generator.eval()
    log.info("ss_generator loaded and weights applied")

    # --- condition embedder (DINOv2 x2 + PointPatchEmbed, frozen) ---
    # The generator backbone has condition_embedder: null because conditioning
    # is embedded EXTERNALLY here and passed in as `cond`. The embedder is
    # frozen during GALP training, so its weights are NOT in checkpoint.pt
    # (which holds generator weights only). They live in the standalone
    # condition_embedder.ckpt (722 keys under _base_models.condition_embedder.*,
    # extracted from the shared base model). galp.yaml's condition_embedder
    # config matches the architecture that produced these weights.
    ss_condition_embedder = init_ss_condition_embedder(
        ss_generator_config,
        str(galp_ckpt_dir / "condition_embedder.ckpt"),
        device=device,
    )
    ss_condition_embedding = ConditionEmbedding(ss_condition_embedder, [])
    log.info("Condition embedder loaded (frozen, from condition_embedder.ckpt)")

    return {
        "ss_encoder": ss_encoder,
        "ss_generator": ss_generator,
        "ss_condition_embedding": ss_condition_embedding,
        "ss_preprocessor": ss_preprocessor,
        "moge": moge,
    }


# ---------------------------------------------------------------------------
# MoGe inference → pointmap
# ---------------------------------------------------------------------------

def _moge_convention_transform(device):
    """R3 camera space → PyTorch3D camera space rotation (same as MeshLayout)."""
    R, _ = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return Transform3d(device=device).rotate(R)


def run_moge(moge, image_path: str, device):
    """Run MoGe; returns (pointmap H×W×3 in PyTorch3D convention, meta dict)."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = (
        torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    ).to(device)

    with torch.no_grad():
        moge_output = moge.infer(image_tensor)

    # Apply R3 → PyTorch3D convention transform (matching MeshLayout/inference_utils)
    pointmaps = moge_output["points"]  # (H, W, 3)
    conv_transform = _moge_convention_transform(device)
    points = conv_transform.transform_points(pointmaps).cpu().numpy()  # (H, W, 3)

    meta = {
        "intrinsics": moge_output["intrinsics"].cpu(),  # (3,3) normalized
        "depth": moge_output["depth"].cpu(),
        "mask": moge_output["mask"].cpu(),
    }
    return points, meta


# ---------------------------------------------------------------------------
# Per-scene inference
# ---------------------------------------------------------------------------

def run_scene(scene_dir: Path, models: dict, device: torch.device) -> dict:
    """
    Run GALP inference on a single scene directory.

    Returns a dict with keys matching layout_prediction.json.
    """
    log = logging.getLogger(__name__)

    IMAGE_SIZE = (518, 518)

    # ------------------------------------------------------------------
    # 1. Discover input files
    # ------------------------------------------------------------------
    image_path = scene_dir / "image.png"

    # Path resolution: ALWAYS prefer <scene>/X over <scene>/inputs/X when
    # <scene>/X exists and has payload. <scene>/inputs/X is only used as a
    # legacy fallback (post-finalize standalone GALP rerun). This guards
    # against stale inputs/masks/ on partial reruns where the user revised
    # the merge plan and reran post-phase only.
    top_masks = scene_dir / "masks"
    inputs_masks = scene_dir / "inputs" / "masks"
    top_objects = scene_dir / "object"
    inputs_objects = scene_dir / "inputs" / "object"

    def _numeric_pngs(d: Path) -> set[str]:
        if not d.is_dir():
            return set()
        return {p.stem for p in d.iterdir() if p.suffix == ".png" and p.stem.isdigit()}

    def _numeric_glbs(d: Path) -> set[str]:
        if not d.is_dir():
            return set()
        return {p.stem for p in d.iterdir() if p.suffix == ".glb" and p.stem.isdigit()}

    top_mask_set = _numeric_pngs(top_masks)
    inputs_mask_set = _numeric_pngs(inputs_masks)
    if top_mask_set:
        masks_dir = top_masks
    elif inputs_masks.is_dir():
        masks_dir = inputs_masks
    else:
        masks_dir = top_masks
    if top_mask_set and inputs_mask_set and top_mask_set != inputs_mask_set:
        log.warning(
            "[run_galp] WARN stale inputs/masks detected: top has %s but inputs has %s. "
            "Using top (current). To clear: rm -rf %s",
            sorted(int(s) for s in top_mask_set),
            sorted(int(s) for s in inputs_mask_set),
            inputs_masks,
        )

    top_obj_set = _numeric_glbs(top_objects)
    inputs_obj_set = _numeric_glbs(inputs_objects)
    if top_obj_set:
        objects_dir = top_objects
    elif inputs_objects.is_dir():
        objects_dir = inputs_objects
    else:
        objects_dir = top_objects
    if top_obj_set and inputs_obj_set and top_obj_set != inputs_obj_set:
        log.warning(
            "[run_galp] WARN stale inputs/object detected: top has %s but inputs has %s. "
            "Using top (current). To clear: rm -rf %s",
            sorted(int(s) for s in top_obj_set),
            sorted(int(s) for s in inputs_obj_set),
            inputs_objects,
        )

    if not image_path.exists():
        raise FileNotFoundError(f"image.png not found in {scene_dir}")

    # Collect 1-indexed mask and object files
    mask_indices = sorted(
        int(p.stem) for p in masks_dir.glob("*.png") if p.stem.isdigit()
    ) if masks_dir.exists() else []

    obj_paths = []
    mask_paths = []
    for idx in mask_indices:
        mpath = masks_dir / f"{idx}.png"
        opath = objects_dir / f"{idx}.glb"
        if opath.exists():
            mask_paths.append(mpath)
            obj_paths.append(opath)
        else:
            log.warning("No mesh found for mask index %d — skipping", idx)

    if not obj_paths:
        raise RuntimeError(
            f"No matching mask+mesh pairs found in {scene_dir}. "
            "Expected masks/1.png ... N.png and object/1.glb ... N.glb"
        )

    log.info("Found %d objects in %s", len(obj_paths), scene_dir)

    # ------------------------------------------------------------------
    # 2. Load image + run MoGe
    # ------------------------------------------------------------------
    pointmap_hw3, moge_meta = run_moge(models["moge"], str(image_path), device)

    image_pil = Image.open(image_path).resize(IMAGE_SIZE).convert("RGB")
    image_tensor = (
        torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float().to(device) / 255.0
    )  # (3, H, W)

    # pointmap: (H, W, 3) → (3, H, W), resize to 518×518, normalise
    pointmap = torch.tensor(pointmap_hw3).permute(2, 0, 1).float().to(device)  # (3, H, W)
    pointmap = tf.resize(
        pointmap.unsqueeze(0), IMAGE_SIZE, interpolation=tf.InterpolationMode.BILINEAR
    ).squeeze(0)  # (3, 518, 518)
    pointmap_org = pointmap.clone()

    _min = pointmap.flatten(1).min(1).values
    _max = pointmap.flatten(1).max(1).values
    pm_center = (_min + _max) / 2  # (3,)
    centered = pointmap - pm_center.unsqueeze(1).unsqueeze(1)
    pm_scale = centered.max()
    pointmap_norm = centered / pm_scale  # normalised

    # ------------------------------------------------------------------
    # 3. Per-object preprocessing
    # ------------------------------------------------------------------
    ss_preprocessor = models["ss_preprocessor"]

    def read_mask(mask_path: Path) -> np.ndarray:
        m = np.array(Image.open(mask_path).resize(IMAGE_SIZE, Image.NEAREST))
        if m.ndim > 2:
            m = m[:, :, -1]
        return m > 0

    items = []
    voxels = []
    for mpath, opath in zip(mask_paths, obj_paths):
        mask_np = read_mask(mpath)
        mask = torch.from_numpy(mask_np).to(device)

        rgba = torch.cat([image_tensor, mask.unsqueeze(0)], dim=0)  # (4, H, W)
        item = preprocess_image(rgba, ss_preprocessor, pointmap=pointmap_norm)
        items.append(item)

        voxel = mesh_to_voxel_tensor(str(opath), resolution=16, version="old")[0]
        voxels.append(voxel)

    # ------------------------------------------------------------------
    # 4. Collate batch
    # ------------------------------------------------------------------
    items = listdict_to_dictlist_safe(items)
    for key in items:
        items[key] = torch.stack(items[key])
    items["voxels"] = torch.stack(voxels)
    items["num_parts"] = torch.tensor([len(obj_paths)]).to(device)

    for key in items:
        if key == "num_parts":
            items[key] = items[key].long()
        elif isinstance(items[key], torch.Tensor):
            items[key] = items[key].to(device).float()

    # ------------------------------------------------------------------
    # 5. Forward pass
    # ------------------------------------------------------------------
    ss_encoder = models["ss_encoder"]
    ss_generator = models["ss_generator"]
    ss_condition_embedding = models["ss_condition_embedding"]

    condition_args, condition_kwargs = ss_condition_embedding(items)
    cond = condition_args[0] if len(condition_args) > 0 else None

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            shape_latent = ss_encoder(items["voxels"])["z"]
            shape_latent = shape_latent.reshape(items["voxels"].shape[0], 8, -1).transpose(1, 2)

            latents = {"shape": shape_latent}
            outputs = ss_generator(latents, cond, num_parts=items["num_parts"], cond_pm=None)

    for key in outputs:
        outputs[key] = outputs[key].detach().squeeze(1)
        # Cast to float32 — autocast produced Half (fp16) but postprocess mixes
        # with pm_scale/pm_center (Float32), so torch.bmm below needs matching dtypes.
        if torch.is_tensor(outputs[key]) and outputs[key].dtype != torch.float32:
            outputs[key] = outputs[key].float()
        if key == "scale":
            outputs[key] = outputs[key].mean(-1).unsqueeze(1)

    outputs["num_parts"] = items["num_parts"]
    outputs["meshes"] = [str(p) for p in obj_paths]

    # ------------------------------------------------------------------
    # 6. Postprocess: denormalise (notebook convention)
    # ------------------------------------------------------------------
    # V1_4 wrapper stores the XZ-plane predictions before xz2f rotation in
    # "pred_6drotation_normalized" and "translation" (which already has xz2f
    # applied as scene_t = xz2f @ T_xz).
    # Following Inference Custom.ipynb: use pred_6drotation_normalized for
    # object rotations and apply xz2f^T to the denormalised translation.
    xz2f_rot_all = rotation_6d_to_matrix(outputs["xz2f_rot"])             # (N, 3, 3) per-object
    xz2f_rot = rotation_6d_to_matrix(outputs["xz2f_rot"].mean(0))         # (3, 3) camera only

    outputs["scale"] = outputs["scale"] * pm_scale
    outputs["translation"] = outputs["translation"] * pm_scale + pm_center
    outputs["translation"] = torch.bmm(
        xz2f_rot_all.transpose(2, 1), outputs["translation"].unsqueeze(-1)
    ).squeeze(-1)

    outputs["rotation"] = rotation_6d_to_matrix(outputs["pred_6drotation_normalized"])

    # ------------------------------------------------------------------
    # 7. Camera / Blender parameters
    # ------------------------------------------------------------------
    camera_rotation = _rotation_to_blender_rotation(xz2f_rot.cpu())

    # MoGe has no extrinsics; camera sits at origin in its own space
    intrinsics = moge_meta["intrinsics"]  # (3,3) normalized
    c2w = np.eye(4)
    c2w[:3, :3] = xz2f_rot.cpu().numpy()

    blender_focal_mm = intrinsic_to_blender_focal_mm(
        intrinsics.numpy() if hasattr(intrinsics, "numpy") else np.array(intrinsics)
    )

    return {
        "outputs": outputs,
        "xz2f_rot": xz2f_rot,
        "pm_center": pm_center,
        "pm_scale": pm_scale,
        "intrinsics": intrinsics,
        "c2w": c2w,
        "camera_rotation": camera_rotation,
        "blender_focal_mm": blender_focal_mm,
        "pointmap_org": pointmap_org,
        "obj_paths": obj_paths,
        "items": items,
        "image_tensor_518": tf.resize(
            image_tensor.unsqueeze(0), IMAGE_SIZE,
            interpolation=tf.InterpolationMode.BILINEAR
        ).squeeze(0),
    }


# ---------------------------------------------------------------------------
# Blender camera rotation helper (from notebook)
# ---------------------------------------------------------------------------

def _rotation_to_blender_rotation(xz2f_rot, ext=None):
    from pytorch3d.transforms import matrix_to_euler_angles

    if ext is None:
        ext = torch.eye(3)
    c2w = ext @ torch.linalg.inv(xz2f_rot)
    T = torch.tensor([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
    rot = matrix_to_euler_angles(torch.tensor(c2w @ T), "XYZ")
    x, z, y = (np.array(rot / torch.pi * 180).round(3) * -1).tolist()
    return [x, z, y]


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def save_outputs(result: dict, scene_dir: Path):
    """Write layout_prediction.json and layout-prediction.glb."""
    log = logging.getLogger(__name__)

    outputs = result["outputs"]
    xz2f_rot = result["xz2f_rot"]
    pm_center = result["pm_center"]
    pm_scale = result["pm_scale"]
    intrinsics = result["intrinsics"]
    c2w = result["c2w"]
    camera_rotation = result["camera_rotation"]
    blender_focal_mm = result["blender_focal_mm"]
    obj_paths = result["obj_paths"]

    # Object IDs are derived from each mesh file's numeric stem (which equals
    # the original mask id), so survivors keep their original ids even when
    # earlier indices are missing because of merge/delete/SAM3D failures.
    # E.g. obj_paths = [3.glb, 4.glb, 7.glb] -> object_ids = [obj_3, obj_4, obj_7].
    object_ids = [f"obj_{int(Path(p).stem)}" for p in obj_paths]

    # --- floor geometry (derived from scene bounds) ---
    scene = build_scene(outputs)

    # pointmap_xz.ply is still useful downstream; keep exporting it but do NOT
    # use its bounds to estimate the floor — scene.bounds is the source of truth.
    _, point_map_mesh = _make_floor(result)
    pointmap_path = str(scene_dir / "pointmap_xz.ply")
    if point_map_mesh is not None:
        point_map_mesh.export(pointmap_path)
        log.info("Wrote %s (%d vertices)", pointmap_path, len(point_map_mesh.vertices))
    else:
        log.warning("Skipped pointmap_xz.ply (point_map_mesh is None)")

    # Floor = scene's minimum y. Any object whose bottom is within 0.1 m of
    # the floor (including objects sitting below it with negative gap) is
    # snapped down/up to touch the floor exactly. Objects more than 0.1 m
    # above the floor are intentionally left floating (e.g. wall art, lights).
    min_y = scene.bounds[0, 1]
    translation_arr = outputs["translation"]  # (N, 3) tensor — kept in sync below
    for i, (_name, geom) in enumerate(scene.geometry.items()):
        obj_min_y = geom.bounds[0, 1]
        gap = obj_min_y - min_y
        if gap < 0.05:
            dy = float(min_y - obj_min_y)
            geom.apply_translation(np.array([0.0, dy, 0.0]))
            translation_arr[i, 1] += dy

    sx, _, sz = scene.bounds[1] - scene.bounds[0]
    center = (scene.bounds[1] + scene.bounds[0]) / 2
    floor = trimesh.creation.box(extents=(sx, 1e-2, sz))
    floor.apply_translation([center[0], min_y - 1e-2, center[2]])
    floor_path = str(scene_dir / "floor.obj")
    floor.export(floor_path)

    floor_scale = float(max(sz, sx) / 2)
    floor_translation = [float(center[0]), float(min_y - 1e-2), float(center[2])]
    floor_rotation = torch.eye(3).tolist()

    # --- layout_prediction.json ---
    translations = outputs["translation"].cpu().tolist()
    rotations = outputs["rotation"].cpu().tolist()
    scales = outputs["scale"].cpu().tolist()

    intr_np = intrinsics.cpu().numpy() if hasattr(intrinsics, "cpu") else np.array(intrinsics)

    layout = {
        "translation": translations + [floor_translation],
        "rotation": rotations + [floor_rotation],
        "scale": scales + [floor_scale],
        "xz2floor": xz2f_rot.cpu().tolist(),
        "num_parts": outputs["num_parts"].cpu().tolist(),
        "meshes": [str(p) for p in obj_paths] + [floor_path],
        "object_id": object_ids + ["floor"],
        "shifted_center": pm_center.cpu().tolist(),
        "shifted_scale": float(pm_scale.cpu()),
        "c2w_extrinsic": c2w.tolist(),
        "intrinsics": intr_np.tolist(),
        "blender_focal_length": float(blender_focal_mm),
        "blender_camera_rotation": camera_rotation,
    }

    json_path = scene_dir / "layout_prediction.json"
    with open(json_path, "w") as f:
        json.dump(layout, f, indent=4)
    log.info("Wrote %s", json_path)

    # --- layout-prediction.glb ---
    # scene.geometry has already been snapped to the floor above; just bundle.
    new_scene = trimesh.Scene()
    new_scene.add_geometry(floor, node_name="floor", geom_name="floor")

    for obj_id, (name, geom) in zip(object_ids, scene.geometry.items()):
        new_scene.add_geometry(geom, node_name=obj_id, geom_name=obj_id)

    glb_path = scene_dir / "layout-prediction.glb"
    new_scene.export(str(glb_path))
    log.info("Wrote %s", glb_path)


def pointmap_to_vertices(pointmap: torch.Tensor, rgb: np.ndarray, stride: int):
    pts = pointmap.detach().cpu().permute(1, 2, 0).numpy()[::stride, ::stride]
    colors = rgb[::stride, ::stride]
    valid = np.isfinite(pts).all(axis=-1)
    vertices = pts[valid]
    vertex_colors = colors[valid]
    ys, xs = np.nonzero(valid)
    uv = np.stack(
        [
            (xs.astype(np.float32) + 0.5) / valid.shape[1],
            1.0 - (ys.astype(np.float32) + 0.5) / valid.shape[0],
        ],
        axis=1,
    )
    index_map = -np.ones(valid.shape, dtype=np.int32)
    index_map[valid] = np.arange(len(vertices), dtype=np.int32)
    return vertices, vertex_colors, uv, index_map, valid

def build_surface_faces(index_map: np.ndarray, valid: np.ndarray, flip_normals: bool = True) -> np.ndarray:
    faces: list[list[int]] = []
    for y in range(valid.shape[0] - 1):
        for x in range(valid.shape[1] - 1):
            a = int(index_map[y, x])
            b = int(index_map[y, x + 1])
            c = int(index_map[y + 1, x])
            d = int(index_map[y + 1, x + 1])
            if a >= 0 and b >= 0 and c >= 0:
                faces.append([a, c, b] if flip_normals else [a, b, c])
            if b >= 0 and d >= 0 and c >= 0:
                faces.append([b, c, d] if flip_normals else [b, d, c])
    return np.asarray(faces, dtype=np.int64)

def export_pointmap_glb(pointmap: torch.Tensor, rgb: np.ndarray, output_path: Path=None, stride: int = 1) -> dict:
    # pointmap 3, H, W 
    # rgb H, W ,3 
    flip_normals = True
    vertices, vertex_colors, _, index_map, valid = pointmap_to_vertices(pointmap, rgb, stride=stride)
    faces = build_surface_faces(index_map, valid)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_colors=vertex_colors,
        process=False,
    )
    if output_path is not None:
        mesh.export(output_path)
    return mesh

def _make_floor(result: dict):
    """Build (a) PointCloud for floor-y estimation and (b) surface mesh of the
    camera-frame pointmap rotated into the xz-floor frame, suitable for export
    as `pointmap_xz.ply`. Always returns a 2-tuple `(floor_pc_or_None,
    pointmap_mesh_or_None)` so callers can tuple-unpack safely."""
    log = logging.getLogger(__name__)
    try:
        pointmap_org = result["pointmap_org"]      # (3, 518, 518), camera frame
        xz2f_rot = result["xz2f_rot"]              # (3, 3) camera→xz-floor
        # Rotate every point: (3,N).T = (N,3); @ R = (N,3) in xz-floor frame
        rot_pm = pointmap_org.reshape(3, -1).T.cpu() @ xz2f_rot.cpu()  # (N,3)
        # Build (3, 518, 518) for the surface-mesh exporter
        rot_pm_3hw = rot_pm.T.reshape(3, 518, 518)
        # RGB: use the original 518×518 image, in [0,255] uint8 for trimesh
        rgb_3hw = result["image_tensor_518"]  # (3, 518, 518), [0,1]
        rgb_hw3 = (rgb_3hw.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        # Surface mesh of the rotated pointmap (vertices + faces + colors)
        pointmap_mesh = export_pointmap_glb(rot_pm_3hw, rgb_hw3)
        # PointCloud for floor-y estimation
        verts = rot_pm.numpy()
        valid = np.isfinite(verts).all(axis=1)
        if valid.sum() == 0:
            log.warning("_make_floor: zero finite points after rotation")
            return None, pointmap_mesh
        floor_pc = trimesh.PointCloud(verts[valid])
        return floor_pc, pointmap_mesh
    except Exception as e:
        log.warning("_make_floor failed (%s); falling back to scene bounds and skipping pointmap_xz.ply", e)
        return None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GALP standalone inference runner")
    parser.add_argument("--scene_dir", required=True, help="Path to scene directory")
    parser.add_argument(
        "--ckpt",
        default=str(_dir("checkpoints_galp", "./checkpoints/galp") / "checkpoint.pt"),
        help="Path to trained checkpoint (.pt). Default: checkpoints/galp/checkpoint.pt",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value (default: 0)")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Apply env before any CUDA ops
    _apply_env(args.gpu)
    _wire_paths()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    log = logging.getLogger(__name__)

    _import_heavy()

    scene_dir = Path(args.scene_dir).resolve()
    ckpt_path = Path(args.ckpt).resolve()

    if not scene_dir.is_dir():
        log.error("scene_dir does not exist: %s", scene_dir)
        sys.exit(1)

    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        log.error("Default: %s", GALP_REPO / "checkpoints" / "checkpoint.pt")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading models...")
    models = load_models(ckpt_path, device)

    log.info("Running inference on: %s", scene_dir)
    result = run_scene(scene_dir, models, device)

    log.info("Saving outputs...")
    save_outputs(result, scene_dir)

    log.info("Done. Outputs written to %s", scene_dir)


if __name__ == "__main__":
    main()
