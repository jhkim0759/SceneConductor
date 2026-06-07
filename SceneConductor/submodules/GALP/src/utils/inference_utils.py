import trimesh 
import numpy as np
from src.datasets.data_utils import voxels_to_box_mesh  # noqa: E402
from src.utils.train_utils import get_configs  # noqa: E402
import torch.nn.functional as F
import torch
from typing import Union

import os 
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import sys
from pathlib import Path
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger

from pytorch3d.ops import knn_points

# -------------------------
# Keep your utilities as-is
# -------------------------
def sample_scene_objects(scene, n_points=10240):
    """
    scene: trimesh.Scene
    n_points: number of points to sample per object
    return: dict {object_name: (N,3) numpy array}
    """
    sampled_points = []

    for node_name in scene.graph.nodes_geometry:
        # get the geometry name
        geom_name = scene.graph[node_name][1]
        mesh = scene.geometry[geom_name]

        # surface sampling (local coordinates)
        points_local = mesh.sample(n_points)
    
        sampled_points.append(points_local)
    sampled_points = np.concatenate(sampled_points, axis=0)

    return sampled_points
def chamfer_distance(pc1, pc2):
    # pc1, pc2: torch tensor (N,3), (M,3)
    x = pc1.unsqueeze(0)  # (1,N,3)
    y = pc2.unsqueeze(0)  # (1,M,3)

    # x -> y
    knn_xy = knn_points(x, y, K=1, return_nn=False)
    # knn_xy.dists: (1, N, 1)  (note: usually squared L2)
    dxy = knn_xy.dists.squeeze(0).squeeze(-1)

    # y -> x
    knn_yx = knn_points(y, x, K=1, return_nn=False)
    dyx = knn_yx.dists.squeeze(0).squeeze(-1)

    # PyTorch3D knn_points dists are squared L2; take sqrt to get L2
    dxy = torch.sqrt(torch.clamp(dxy, min=0.0)).cpu()
    dyx = torch.sqrt(torch.clamp(dyx, min=0.0)).cpu()

    return float(dxy.mean() + dyx.mean())

def normalize_point_cloud(pc: torch.Tensor) -> torch.Tensor:
    """
    pc: (N, 3) torch.Tensor
    returns: normalized point cloud (N, 3)
    """
    # center: (min + max) / 2
    centroid = (pc.min(dim=0).values + pc.max(dim=0).values) * 0.5
    pc = pc - centroid

    # max distance
    max_= pc.max() 
    return pc / max_

def normalized_chamfer_distance(pc1, pc2):
    return chamfer_distance(normalize_point_cloud(pc1), normalize_point_cloud(pc2))

# -------------------------
# Model/dataset init helpers
# -------------------------
def _ensure_paths():
    repo_root = Path.cwd()
    src_dir = repo_root / "src"
    for p in (repo_root, src_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return repo_root
def strip_module_prefix(sd: dict) -> dict:
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }


_moge_model = None
def _get_moge_model(device='cuda'):
    global _moge_model
    if _moge_model is None:
        from moge.model.v1 import MoGeModel
        _moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        _moge_model.eval()
    return _moge_model

device='cuda'

import os
import torch
from PIL import Image
import numpy as np
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d
from collections import namedtuple

DecomposedTransform = namedtuple(
    "DecomposedTransform", ["scale", "rotation", "translation"]
)

def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def run_moge(image_path, return_output=False):
    image = Image.open(image_path).convert('RGB')
    image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        moge_output = _get_moge_model(device).infer(image_tensor)

    pointmaps = moge_output['points']

    camera_convention_transform = (
        Transform3d()
        .rotate(camera_to_pytorch3d_camera(device=device).rotation)
        .to(device)
    )
    points_tensor = camera_convention_transform.transform_points(pointmaps).cpu().numpy()

    if return_output:
        return points_tensor, moge_output
    return points_tensor



def create_palette():
    # Define a palette with 32 colors for labels 0-31
    palette = [
        0, 0, 0,        # Label 0 (black)
        255, 0, 0,      # Label 1 (red)
        0, 255, 0,      # Label 2 (green)
        0, 0, 255,      # Label 3 (blue)
        255, 255, 0,    # Label 4 (yellow)
        255, 0, 255,    # Label 5 (magenta)
        0, 255, 255,    # Label 6 (cyan)
        128, 0, 0,      # Label 7
        0, 128, 0,      # Label 8
        0, 0, 128,      # Label 9
        128, 128, 0,    # Label 10
        128, 0, 128,    # Label 11
        0, 128, 128,    # Label 12
        64, 0, 0,       # Label 13
        0, 64, 0,       # Label 14
        0, 0, 64,       # Label 15
        64, 64, 0,      # Label 16
        64, 0, 64,      # Label 17
        0, 64, 64,      # Label 18
        192, 192, 192,  # Label 19
        128, 128, 128,  # Label 20
        255, 165, 0,    # Label 21
        75, 0, 130,     # Label 22
        238, 130, 238,  # Label 23
        210, 105, 30,   # Label 24
        123, 104, 238,  # Label 25
        0, 191, 255,    # Label 26
        154, 205, 50,   # Label 27
        255, 20, 147,   # Label 28
        46, 139, 87,    # Label 29
        255, 215, 0,    # Label 30
        0, 206, 209,    # Label 31
        139, 69, 19,    # Label 32
        233, 150, 122,  # Label 33
        255, 105, 180,  # Label 34
        127, 255, 212,  # Label 35
        100, 149, 237,  # Label 36
        220, 20, 60,    # Label 37
        0, 250, 154,    # Label 38
        70, 130, 180,   # Label 39
        255, 228, 181,  # Label 40
        244, 164, 96,   # Label 41
        176, 224, 230,  # Label 42
        189, 183, 107,  # Label 43
        255, 182, 193,  # Label 44
        0, 100, 0,      # Label 45
        85, 107, 47,    # Label 46
        199, 21, 133,   # Label 47
        72, 61, 139,    # Label 48
        255, 99, 71,    # Label 49
        32, 178, 170,   # Label 50
        160, 82, 45,    # Label 51
    ]

    # Extend the palette to have 768 values (256 * 3)
    palette.extend([0] * (768 - len(palette)))
    return palette


def create_segmentation_mask(binary_masks):
    B, H, W = binary_masks.shape
    final_mask = np.zeros((H, W), dtype=np.int32)

    for b in range(B):
        labeled = (binary_masks[b]>0)
        final_mask[labeled > 0] = b+1
    
    final_mask = Image.fromarray(final_mask.astype(np.uint8), mode="P")
    palette = []
    for i in range(256):
        palette.extend(((i * 37) % 256, (i * 67) % 256, (i * 97) % 256))
    final_mask.putpalette(palette)

    return final_mask

# def rot6d_to_rotmat(x6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     Convert a 6D rotation representation to a 3x3 rotation matrix.
#     Mirrors the helper in position_Test.ipynb.
#     """
#     x6d = torch.as_tensor(x6d, dtype=torch.float32)
#     if x6d.ndim == 1:
#         x6d = x6d.unsqueeze(0)

#     a1 = x6d[..., 0:3]
#     a2 = x6d[..., 3:6]

#     b1 = F.normalize(a1, dim=-1, eps=eps)
#     proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
#     b2 = F.normalize(a2 - proj, dim=-1, eps=eps)
#     b3 = torch.cross(b1, b2, dim=-1)
#     return torch.stack([b1, b2, b3], dim=-1)

# def normalize_mesh(mesh,):
#     vertices = mesh.vertices
#     center = (vertices.min(0)+vertices.max(0))/2
#     vertices = vertices - center
#     mesh.vertices = vertices/vertices.max()

#     return mesh

def normalize_scene(scene):
    # compute the bounding box of the whole scene
    bounds = scene.bounds  # shape (2, 3)
    min_bound, max_bound = bounds

    center = (min_bound + max_bound) / 2.0
    scale = (max_bound - min_bound).max() /2.0 

    # apply the same transform to every geometry
    for name, geom in scene.geometry.items():
        vertices = geom.vertices.copy()
        vertices = vertices - center
        vertices = vertices / scale
        geom.vertices = vertices

    return scene
def normalize_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        return normalize_scene(mesh)
    
    vertices = mesh.vertices
    center = (vertices.min(0)+vertices.max(0))/2
    vertices = vertices - center
    mesh.vertices = vertices/vertices.max()

    return mesh

def rotation_matrix_to_6d(R):
    """
    R: (3,3) rotation matrix
    return: (6,) 6D rotation representation
    """
    # use the first two columns (standard definition)
    return R[:, :2].reshape(-1)

from pytorch3d.transforms import rotation_6d_to_matrix


def build_scene(sample: dict, version='mesh', normalize_mode='object') -> trimesh.Scene:
    """
    Build scene from sample data.
    
    Args:
        sample: Dictionary containing meshes and transform parameters
        version: 'mesh' or 'voxel'
        normalize_mode: 'object' (normalize each object independently) or 
                       'scene' (normalize entire scene - SceneGen style)
    """
    num_parts = int(sample["num_parts"].item())
    translations = sample["translation"].cpu().numpy()
    if 'rotation' in sample:
        rotations = sample["rotation"].cpu().numpy()
    else:
        rotations = rotation_6d_to_matrix(sample["6drotation_normalized"]).cpu().numpy()
    scales = sample["scale"].view(-1).cpu().numpy()

    scene = trimesh.Scene()
    for idx in range(num_parts):
        if version == "mesh":
            if isinstance(sample["meshes"][idx], trimesh.Trimesh):
                part_mesh = sample["meshes"][idx].copy()
            elif isinstance(sample["meshes"][idx], trimesh.Scene):
                part_mesh = trimesh.util.concatenate(
                    tuple(sample["meshes"][idx].geometry.values())
                )   
            elif isinstance(sample["meshes"][idx], str):
                part_mesh = trimesh.load(sample["meshes"][idx])
        elif version == "voxel":
            voxel_tensor = sample["voxels"][idx].cpu()
            part_mesh = voxel_tensor_to_mesh(voxel_tensor)

        # Normalize each object if mode is 'object'
        if normalize_mode == 'object':
            part_mesh = normalize_mesh(part_mesh)
        
        if not isinstance(part_mesh, trimesh.Trimesh):
            part_mesh = part_mesh.to_geometry()
        transform = np.eye(4)
        transform[:3, :3] = rotations[idx] * float(scales[idx])
        transform[:3, 3] = translations[idx]

        part_mesh.apply_transform(transform)
        scene.add_geometry(part_mesh)
    
    # Normalize entire scene if mode is 'scene' (SceneGen methodology)
    if normalize_mode == 'scene':
        bounds = scene.bounds
        scene_min, scene_max = bounds
        scene_center = (scene_min + scene_max) / 2.0
        extents = scene_max - scene_min
        max_extent = extents.max()

        # SceneGen uses 2% margin from each side
        margin = 0.02
        target_half_size = 1 - margin
        scale_factor = target_half_size * 2 / max_extent

        normalize_transform = trimesh.transformations.compose_matrix(
            translate=-scene_center,
            scale=[scale_factor, scale_factor, scale_factor]
        )
        scene.apply_transform(normalize_transform)
    
    return scene


def build_scene_scenegen(sample: dict, version='mesh') -> trimesh.Scene:
    """
    Build scene following SceneGen methodology:
    1. Apply transformations to meshes without individual normalization
    2. Normalize the entire scene to fit within [-0.98, 0.98]
    
    This matches SceneGen's evaluation approach where GT scenes are 
    normalized at the scene level, not object level.
    """
    return build_scene(sample, version=version, normalize_mode='scene')

def voxel_tensor_to_mesh(voxel_tensor: torch.Tensor) -> trimesh.Trimesh:
    """
    Turn a [1, R, R, R] occupancy grid into a centered mesh for visualization.
    """
    if voxel_tensor.ndim == 3:
        voxel_tensor = voxel_tensor.unsqueeze(0)
    res = int(voxel_tensor.shape[-1])
    mesh = voxels_to_box_mesh(voxel_tensor, batch_id=0, pitch=2.0 / res)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh





# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# # insert the project root at the top of sys.path
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# SRC_MAP = {
#     "0": "src_org",
#     "1": "src_new",
#     "2": "src_mot",
# }

# mode = sys.argv[1]
# pkg_name = SRC_MAP[mode]
# pkg_path = os.path.join(PROJECT_ROOT, pkg_name)
# real_pkg = importlib.import_module(pkg_name)
# sys.modules["src"] = real_pkg
# print("src alias ->", real_pkg.__name__, "at", getattr(real_pkg, "__file__", None))



# def init_ss_generator(ss_generator_config_path, ss_generator_ckpt_path, device="cuda"):
#     from src.train import SparseStructureFlowTdfyWrapper  # if this is the real class path
#     cfg = OmegaConf.load(ss_generator_config_path)
#     flow_cfg = cfg["module"]["generator"]["backbone"]["reverse_fn"]["backbone"]
#     model: SparseStructureFlowTdfyWrapper = instantiate(flow_cfg)

#     ckpt = strip_module_prefix(torch.load(ss_generator_ckpt_path, map_location=device, weights_only=False))
    
#     logger.info(f"LOAD CKPT: {ss_generator_ckpt_path} ")

#     missing, unexpected = model.load_state_dict(ckpt, strict=True)
#     if missing:
#         logger.warning("Missing keys while loading flow: %s", missing)
#     if unexpected:
#         logger.warning("Unexpected keys while loading flow: %s", unexpected)
#     else:
#         logger.info("Loaded flow weights from %s", ss_generator_ckpt_path)

#     return model.to(device)
