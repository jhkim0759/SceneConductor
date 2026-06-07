import torch
from torchvision.transforms import functional as tf
import numpy as np
import trimesh

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    o3d = None
    _HAS_OPEN3D = False
from src.utils.typing_utils import *

from src.sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)

import math
import torch
import torch.nn.functional as F

from pytorch3d.transforms import matrix_to_rotation_6d, euler_angles_to_matrix, rotation_6d_to_matrix

ROTATION_6D_MEAN = torch.tensor(
    [
        -0.06366084883674913,
        0.008438224692279752,
        0.00017084786438302483,
        0.0007126610473540038,
        -0.0030916726538816417,
        0.5166093753457688,
    ]
)
ROTATION_6D_STD = torch.tensor(
    [
        0.6656971967514863,
        0.6787012271867754,
        0.30345010594844524,
        0.4394504420678794,
        0.39817973931717104,
        0.6176286868761914,
    ]
)

def voxels_to_box_mesh(vox_idx: torch.Tensor, batch_id: int = 0, pitch: float = 1.0, normalize: bool = True) -> trimesh.Trimesh:

    if vox_idx.ndim>2:
        vox_idx = torch.argwhere(vox_idx>0)[:, [0, 3, 2, 1]].int()
    
    assert vox_idx.ndim == 2 and vox_idx.shape[1] == 4

    v = vox_idx.detach().cpu().numpy()
    v = v[v[:, 0] == batch_id]
    if len(v) == 0:
        raise ValueError(f"No voxels for batch_id={batch_id}")

    zyx = v[:, 1:4].astype(np.int64)

    zmax, ymax, xmax = zyx.max(axis=0)
    occ = np.zeros((zmax + 1, ymax + 1, xmax + 1), dtype=bool)
    occ[zyx[:, 0], zyx[:, 1], zyx[:, 2]] = True

    # (z,y,x) -> (x,y,z)
    occ_xyz = np.transpose(occ, (2, 1, 0))

    # voxel size = pitch
    transform = np.eye(4)
    transform[:3, :3] *= pitch

    vg = trimesh.voxel.VoxelGrid(
        encoding=occ_xyz,
        transform=transform
    )

    mesh = vg.as_boxes()
    
    if normalize:
        mesh = mesh_normalize(mesh, scale=1.0)
    return mesh


def mesh_normalize_bbox(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
    """
    Normalize mesh to bbox-centered canonical space.
    Longest bbox half-extent becomes `scale`.

    Result:
        bbox center ~= (0,0,0)
        vertices roughly inside [-scale, scale]
    """
    mesh = mesh.copy()

    vertices = mesh.vertices.astype(np.float32)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    center = (bbox_min + bbox_max) * 0.5
    vertices = vertices - center

    half_extent = np.abs(vertices).max()
    if half_extent > 1e-8:
        vertices = vertices / half_extent * scale

    mesh.vertices = vertices
    return mesh


def mesh_to_voxel_tensor_old(mesh, resolution=8, sample_points=True):
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, force="mesh", skip_materials=True)
    elif isinstance(mesh, trimesh.Scene):
        # trimesh 4.x: to_geometry() was removed
        mesh = mesh.dump(concatenate=True)
    
    
    if sample_points:
        points = torch.from_numpy(mesh.sample(4096).astype(np.float32))
        points = points - (points.min(0).values+points.max(0).values)/2.0
        points = points / (points.abs().max() + 1e-8)

    vertices = mesh.vertices
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    scale = float(np.max(max_bound - min_bound))
    scale = scale if scale > 1e-6 else 1.0
    mesh.apply_translation(-min_bound)
    mesh.apply_scale(1.0 / scale)

    voxelized = mesh.voxelized(pitch=1.0 / resolution)
    filled = voxelized.matrix.astype(np.int64)
    coords = np.argwhere(filled > 0)
    coords = np.clip(coords, 0, resolution - 1)

    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    
    if sample_points:
        return ss, points
    return ss

def mesh_to_voxel_tensor(
    mesh,
    resolution: int = 16,
    sample_points: bool = True,
    num_points: int = 4096,
    version: str = "old"
):
    """
    Output:
        ss: (1, R, R, R) binary voxel tensor

    Coordinate convention:
        mesh canonical space is [-1, 1]
        voxel grid is also interpreted as centered [-1, 1]
    """

    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, force="mesh", skip_materials=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    mesh = mesh_normalize_bbox(mesh, scale=1.0)

    if version=="old":
        return mesh_to_voxel_tensor_old(mesh, resolution=resolution, sample_points=sample_points)

    # sample points in same canonical space as voxel
    if sample_points:
        points_np = mesh.sample(num_points).astype(np.float32)
        points = torch.from_numpy(points_np)

    # voxelization
    pitch = 2.0 / resolution

    # trimesh voxelized uses mesh coordinates directly.
    voxelized = mesh.voxelized(pitch=pitch)

    filled = voxelized.matrix.astype(np.int64)  # local voxel grid
    coords = np.argwhere(filled > 0)

    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)

    if coords.shape[0] > 0:
        # Convert local voxel indices back to world/canonical coordinates
        # voxelized.transform maps voxel indices -> world coordinates
        coords_h = np.concatenate(
            [coords.astype(np.float32), np.ones((coords.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        centers_world = (voxelized.transform @ coords_h.T).T[:, :3]

        # world [-1,1] -> tensor index [0, R)
        idx = np.floor((centers_world + 1.0) / 2.0 * resolution).astype(np.int64)
        idx = np.clip(idx, 0, resolution - 1)

        ss[:, idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    if sample_points:
        return ss, points

    return ss


def _load_mesh_robust(mesh):
    """Robustly load a trimesh.Trimesh. Merge if a Scene; load if a str path."""
    if isinstance(mesh, str):
        loaded = trimesh.load(mesh, skip_materials=True)
    else:
        loaded = mesh
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            return None
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        return None
    return loaded


def mesh_to_voxel_tensor_with_pose(mesh, rotation_matrix, scale, translation, resolution=16, n_sample=8192):
    """
    V4: sample points from the mesh, apply the init pose, then voxelize in [-1, 1] space.
    Point-based voxelization handles Scenes / complex meshes robustly.

    Args:
        mesh: trimesh.Trimesh, trimesh.Scene, or str path
        rotation_matrix: (3,3) numpy array
        scale: float or (3,) numpy array
        translation: (3,) numpy array
        resolution: int, voxel grid resolution
        n_sample: number of surface sampling points

    Returns:
        voxel: (1, R, R, R) torch.LongTensor — sparse layout voxel
        points: (4096, 3) torch.FloatTensor — canonical mesh points (for CD eval)
    """
    mesh = _load_mesh_robust(mesh)
    if mesh is None:
        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        canon_pts = torch.zeros(4096, 3)
        return ss, canon_pts

    # Sample canonical points (for CD evaluation)
    canon_pts = torch.from_numpy(mesh.sample(4096).astype(np.float32))
    canon_center = (canon_pts.min(0).values + canon_pts.max(0).values) / 2.0
    canon_pts = canon_pts - canon_center
    canon_pts = canon_pts / (canon_pts.abs().max() + 1e-8)

    # Sample surface points for voxelization (more points = denser voxel)
    pts = mesh.sample(n_sample).astype(np.float64)

    # 1. Center
    center = (pts.min(0) + pts.max(0)) / 2.0
    pts = pts - center
    # 2. Normalize to unit
    max_ext = np.abs(pts).max()
    if max_ext > 1e-6:
        pts = pts / max_ext
    # 3. Apply scale
    pts = pts * np.asarray(scale, dtype=np.float64)
    # 4. Apply rotation
    pts = pts @ np.asarray(rotation_matrix, dtype=np.float64).T
    # 5. Apply translation
    pts = pts + np.asarray(translation, dtype=np.float64)
    # 6. Map [-1,1] → voxel grid indices
    pts_01 = (pts + 1.0) / 2.0
    voxel_idx = (pts_01 * resolution).astype(np.int64)
    voxel_idx = np.clip(voxel_idx, 0, resolution - 1)

    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[0, voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = 1

    return ss, canon_pts


def _apply_transform(input: torch.Tensor, transform):
    if input is not None:
        input = transform(input)
    return input

def _preprocess_image_and_mask(
        rgb_image, mask_image, img_mask_joint_transform
    ):
    for trans in img_mask_joint_transform:
        rgb_image, mask_image = trans(rgb_image, mask_image)
    return rgb_image, mask_image

def preprocess_image(
    rgba_image: Union[torch.Tensor, np.ndarray], preprocessor, pointmap=None, device="cpu"
) -> torch.Tensor:
    # All input should be CHW size 
    
    rgb_image = rgba_image[:3]
    rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()
    processed_rgb_image, processed_mask = _preprocess_image_and_mask(
        rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
    )

    # transform tensor to model input
    processed_rgb_image = _apply_transform(
        processed_rgb_image, preprocessor.img_transform
    )
    processed_mask = _apply_transform(
        processed_mask, preprocessor.mask_transform
    )

    # full image, with only processing from the image
    rgb_image = _apply_transform(rgb_image, preprocessor.img_transform)
    rgb_image_mask = _apply_transform(
        rgb_image_mask, preprocessor.mask_transform
    )
    preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
        rgb_image, rgb_image_mask, pointmap
    )
    
    # Put in a for loop?
    _item = preprocessor_return_dict
    item = {
        "mask": _item["mask"].to(device),
        "image": _item["image"].to(device),
        "rgb_image": _item["rgb_image"].to(device),
        "rgb_image_mask": _item["rgb_image_mask"].to(device),
    }

    if pointmap is not None and preprocessor.pointmap_transform != (None,):
        item["pointmap"] = _item["pointmap"].to(device)
        item["rgb_pointmap"] = _item["rgb_pointmap"].to(device)
        item["pointmap_scale"] = _item["pointmap_scale"].to(device)
        item["pointmap_shift"] = _item["pointmap_shift"].to(device)
        item["rgb_pointmap_scale"] = _item["rgb_pointmap_scale"].to(device)
        item["rgb_pointmap_shift"] = _item["rgb_pointmap_shift"].to(device)
    return item

def listdict_to_dictlist_safe(list_of_dicts):
    all_keys = set().union(*(d.keys() for d in list_of_dicts))
    
    result = {k: [] for k in all_keys}
    
    for d in list_of_dicts:
        for k in all_keys:
            result[k].append(d.get(k))  # None if missing
    
    return result


def mesh_normalize(mesh: trimesh.Trimesh, scale=1) -> trimesh.Trimesh:
    vertices = mesh.vertices
    centroid = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices -= centroid

    max_ = vertices.max()
    if max_ > 0:
        vertices /= max_
        vertices *= scale

    mesh.vertices = vertices
    return mesh

def rand_angles():
    # degrees -> radians; yaw-only discrete rotations 0/90/180/270 deg
    angles_ay = torch.tensor(
        [0.0, 90.0, 180.0, 270.0],
    ) * math.pi / 180.0

    idx_y = torch.randint(0, 4, (1,))

    ax = torch.tensor(0.0)
    ay = angles_ay[idx_y[0]]
    az = torch.tensor(0.0)

    return ax, ay, az


def rotate_mesh_with_R(mesh: trimesh.Trimesh,
                       R: np.ndarray) -> trimesh.Trimesh:
    R = np.asarray(R, dtype=np.float32)
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3,3), got {R.shape}")

    m = mesh.copy()
    c = m.centroid
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R

    m.apply_translation(-c)
    m.apply_transform(T)
    m.apply_translation(+c)
    return m

def augment_pointmap(
    pointmap: torch.Tensor,
    gamma: float = 1.0,
    z_stretch: float = 1.0,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """
    Apply augmentation to a normalized pointmap (3, H, W).

    Args:
        pointmap     : (3, H, W) normalized camera-space 3D coordinates
        gamma        : nonlinear depth warp - z' = sign(z) * |z|^gamma (gamma!=1 -> S-curve distortion)
        z_stretch    : z-axis scale factor (1.0 = identity)
        noise_std    : Gaussian noise standard deviation
        dropout_prob : per-pixel random dropout probability

    Returns:
        augmented pointmap (3, H, W)
    """
    pm = pointmap.clone()

    # 1. Nonlinear depth distortion: z' = sign(z) * |z|^γ
    if abs(gamma - 1.0) > 1e-6:
        z = pm[2]
        pm[2] = torch.sign(z) * (z.abs() + 1e-8).pow(gamma)

    # 2. Z-axis stretch
    if abs(z_stretch - 1.0) > 1e-6:
        pm[2] = pm[2] * z_stretch

    # 3. Gaussian noise
    if noise_std > 0.0:
        pm = pm + torch.randn_like(pm) * noise_std

    # 4. Point dropout (zero-out random pixels)
    if dropout_prob > 0.0:
        keep = (torch.rand(pm.shape[1], pm.shape[2]) > dropout_prob).float()
        pm = pm * keep.unsqueeze(0)

    return pm


import random
def augment_mesh_and_rotation(
    mesh,
    R_gt,
    scale,
    p=0.7,
):
    center = (mesh.vertices.min(0)+mesh.vertices.max(0))/2
    mesh.vertices = mesh.vertices-center
    mesh.vertices = mesh.vertices/mesh.vertices.max()
    
    if random.random() < p:
        return mesh, R_gt, scale, torch.eye(3)
    # print("Applying mesh augmentation with random rotation")
    if R_gt.shape==torch.Size([6]):
        R_gt = rotation_6d_to_matrix(R_gt)


    # sample augmentation rotation
    ax, ay, az = rand_angles()
    R_aug = euler_angles_to_matrix(torch.tensor([ax, ay, az]), "XYZ")

    # rotate vox
    mesh_aug = rotate_mesh_with_R(
        mesh, R_aug
    )

    aug_max = abs(mesh_aug.vertices).max()
    
    scale_aug = scale*aug_max
    mesh_aug.vertices = mesh_aug.vertices/aug_max
    
    R_gt_aug = matrix_to_rotation_6d(R_gt @ R_aug.transpose(-1, -2))
    
    return mesh_aug, R_gt_aug, scale_aug, R_aug 
