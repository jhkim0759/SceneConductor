import math

import torch.nn.functional as F
import torch
from torch import nn
from typing import Dict, Optional

from pytorch3d.transforms import (
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)





def _apply_pose_transform(
    points: torch.Tensor,
    rot_6d_normalized: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Following the pose_decoder / compose_transform convention in inference_utils.py,
    transform mesh points into world space.

    Steps (see inference_utils.py):
      1. 6D rotation un-normalize : rot_6d = rot_6d_normalized * STD + MEAN
      2. Gram-Schmidt (column stacking) → rotation matrix R
      4. compose_transform order   : (points * s) @ R + t  (scale -> rotate -> translate)

    Args:
        points          : (B, N, 3)  — canonical mesh points
        rot_6d_normalized: (B, 6)   — normalized 6D rotation (model output / GT label)
        scale_log       : (B, 1) or (B, 3) — log-scale (model output / GT label)
        translation     : (B, 3)    — translation

    Returns:
        (B, N, 3) world-space points
    """
    B = points.shape[0]

    rot_6d = rot_6d_normalized

    # 2. Gram-Schmidt — columns = [b1, b2, b3]  (inference convention)
    R  = rotation_6d_to_matrix(rot_6d)  # (B, 3, 3)
    s = scale.view(B, 1, -1)          # (B, 1, 1) or (B, 1, 3)

    # 4. scale → rotate → translate  (matches compose_transform order)
    t = translation.view(B, 1, 3)
    return (points * s) @ R.transpose(1, 2) + t

def compute_chamfer_loss(pred_points: torch.Tensor, gt_points: torch.Tensor, weights: Dict[str, float]) -> torch.Tensor:
    import pytorch3d.loss

    chamfer_loss = pytorch3d.loss.chamfer_distance(pred_points, gt_points)[0]
    return chamfer_loss * weights.get("chamfer_loss", 1.0)



def compute_normal_loss(pred_normals: torch.Tensor, gt_normals: torch.Tensor, weights: Dict[str, float]) -> torch.Tensor:
    normal_loss = F.mse_loss(pred_normals, gt_normals)
    return normal_loss * weights.get("normal_loss", 1.0)


def compute_scene_pc_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Scene-level point cloud loss.

    Transform each object's point cloud by the predicted r, t, s, group per scene,
    and compare against the GT transform. Normalize by the scene bounding box size
    to compute a scale-invariant loss.

    Args:
        outputs  : transformer output dict ('translation', '6drotation_normalized', 'scale')
        batch    : GT batch dict ('mesh_points', 'translation', '6drotation_normalized', 'scale')
        num_parts: (B,) number of objects per scene
        weight   : loss weight

    Returns:
        scalar scene-level L1 loss
    """
    if "mesh_points" not in batch:
        return torch.tensor(0.0, device=device)

    points = batch["mesh_points"]  # (N_total, P, 3)

    # GT transform
    rot_gt   = batch["6drotation_normalized"]  # (N_total, 6) or (N_total, 1, 6)
    scale_gt = batch["scale"]                  # (N_total, 1)
    trans_gt = batch["translation"]            # (N_total, 3)

    # Pred transform - squeeze token dim if present (N_total, 1, D) -> (N_total, D)
    rot_pred   = outputs["6drotation_normalized"].squeeze(1)
    scale_pred = outputs["scale"].squeeze(1)
    trans_pred = outputs["translation"].squeeze(1)

    if rot_gt.dim() == 3:
        rot_gt   = rot_gt.squeeze(1)
        scale_gt = scale_gt.squeeze(1)
        trans_gt = trans_gt.squeeze(1)

    # transform each object into world space -> (N_total, P, 3)
    scene_gt   = _apply_pose_transform(points, rot_gt,   scale_gt,   trans_gt)
    scene_pred = _apply_pose_transform(points, rot_pred, scale_pred, trans_pred)

    total_loss = torch.tensor(0.0, device=device or points.device)
    num_scenes = 0
    idx = 0

    for n_tensor in num_parts:
        n = int(n_tensor.item())
        if n == 0:
            continue

        gt_flat   = scene_gt[idx : idx + n].reshape(-1, 3)    # (n*P, 3)
        pred_flat = scene_pred[idx : idx + n].reshape(-1, 3)  # (n*P, 3)

        # Scene bounding box scale (based on GT)
        scene_min   = gt_flat.min(dim=0)[0]
        scene_max   = gt_flat.max(dim=0)[0]
        scene_scale = (scene_max - scene_min).max().clamp(min=1e-8)

        gt_norm   = (gt_flat   - scene_min) / scene_scale
        pred_norm = (pred_flat - scene_min) / scene_scale

        total_loss = total_loss + (gt_norm - pred_norm).abs().mean()
        num_scenes += 1
        idx += n

    if num_scenes > 0:
        total_loss = total_loss / num_scenes

    return total_loss * weight


def compute_iou_loss_aabb(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    weight: float = 0.1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    [OLD] Pairwise 3D AABB IoU loss to penalise object penetration.

    Transform each object's point cloud into world space by the predicted r, t, s,
    compute the Axis-Aligned Bounding Box (AABB), and minimize the IoU between
    object pairs in the scene. If they do not overlap, loss = 0 (no gradient).

    Args:
        outputs  : transformer output dict
        batch    : GT batch dict (requires 'mesh_points')
        num_parts: (B,) number of objects per scene
        weight   : loss weight

    Returns:
        scalar pairwise-IoU loss (mean over all pairs)
    """
    if "mesh_points" not in batch:
        return torch.tensor(0.0, device=device)

    points = batch["mesh_points"]  # (N_total, P, 3)

    rot_pred   = outputs["6drotation_normalized"].squeeze(1) # (N_total, 6)
    scale_pred = outputs["scale"].squeeze(1).mean(1, keepdim=True) # (N_total, 1)
    trans_pred = outputs["translation"].squeeze(1) # (N_total, 3)

    # World-space point clouds — gradient flows through pred r/t/s
    pred_world = _apply_pose_transform(points, rot_pred, scale_pred, trans_pred)  # (N_total, P, 3)

    # AABB per object
    pred_min = pred_world.min(dim=1)[0]  # (N_total, 3)
    pred_max = pred_world.max(dim=1)[0]  # (N_total, 3)

    total_iou = torch.tensor(0.0, device=device or points.device)
    num_pairs = 0
    idx = 0

    for n_tensor in num_parts:
        n = int(n_tensor.item())
        if n < 2:
            idx += n
            continue

        b_min = pred_min[idx : idx + n]  # (n, 3)
        b_max = pred_max[idx : idx + n]  # (n, 3)

        # Vectorised pairwise intersection
        inter_min = torch.max(b_min.unsqueeze(1), b_min.unsqueeze(0))  # (n, n, 3)
        inter_max = torch.min(b_max.unsqueeze(1), b_max.unsqueeze(0))  # (n, n, 3)
        inter_dims = (inter_max - inter_min).clamp(min=0.0)             # (n, n, 3)
        inter_vol  = inter_dims.prod(dim=-1)                            # (n, n)

        vol = (b_max - b_min).clamp(min=0.0).prod(dim=-1)  # (n,)
        union_vol = vol.unsqueeze(1) + vol.unsqueeze(0) - inter_vol      # (n, n)
        iou = inter_vol / (union_vol + 1e-8)                             # (n, n)

        # Upper triangle only (i < j), no diagonal
        mask = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=iou.device), diagonal=1
        )
        pair_iou = iou[mask]

        total_iou = total_iou + pair_iou.sum()
        num_pairs += pair_iou.numel()
        idx += n

    if num_pairs > 0:
        total_iou = total_iou / num_pairs

    return total_iou * weight


# ── Voxel IoU loss helpers ────────────────────────────────────────────────

def _canonical_voxel_coords(mesh_points: torch.Tensor, R: int) -> torch.Tensor:
    """
    mesh_points (P, 3) in [-1, 1] → unique occupied voxel centers in [-1, 1].

    Coord semantics: based on voxel CENTER.
    Mapping: idx = floor((pt + 1) * 0.5 * R), clamp [0, R-1]
    Center:  (idx + 0.5) * 2 / R - 1

    Returns: (K, 3)  unique occupied voxel centers, K ≤ R³
    """
    idx = ((mesh_points + 1.0) * 0.5 * R).long().clamp(0, R - 1)          # (P, 3)
    flat = idx[:, 0] * (R * R) + idx[:, 1] * R + idx[:, 2]                # (P,)
    unique_flat = flat.unique()                                            # (K,)
    ix = unique_flat // (R * R)
    iy = (unique_flat % (R * R)) // R
    iz = unique_flat % R
    return (torch.stack([ix, iy, iz], dim=1).float() + 0.5) * (2.0 / R) - 1.0  # (K, 3)


def _hard_voxelize_scene(
    pts: torch.Tensor, R: int,
    grid_origin: torch.Tensor, grid_extent: torch.Tensor,
) -> torch.Tensor:
    """pts (K,3) scene-space → (R,R,R) hard binary occupancy."""
    norm = (pts - grid_origin) / grid_extent                               # ~[0,1]
    idx = (norm * R).long().clamp(0, R - 1)
    occ = torch.zeros(R, R, R, dtype=torch.float32, device=pts.device)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return occ


def _soft_voxelize_scene(
    pts: torch.Tensor, R: int,
    grid_origin: torch.Tensor, grid_extent: torch.Tensor,
) -> torch.Tensor:
    """
    pts (K,3) scene-space → (R,R,R) soft occupancy via trilinear splatting.
    Differentiable w.r.t. pts.

    Mapping: cont_idx = (pts - origin) / extent * R  (same as hard: * R)
    """
    cont_idx = (pts - grid_origin) / grid_extent * R                       # [0, R)
    cont_idx = cont_idx.clamp(0.0, R - 1e-4)                              # just below R

    p0 = cont_idx.detach().floor().long().clamp(0, R - 2)                  # (K, 3)
    p1 = p0 + 1
    frac = cont_idx - p0.float()                                          # (K, 3) diff!

    occ = torch.zeros(R, R, R, dtype=pts.dtype, device=pts.device)
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                ix = p1[:, 0] if dx else p0[:, 0]
                iy = p1[:, 1] if dy else p0[:, 1]
                iz = p1[:, 2] if dz else p0[:, 2]
                wx = frac[:, 0] if dx else (1.0 - frac[:, 0])
                wy = frac[:, 1] if dy else (1.0 - frac[:, 1])
                wz = frac[:, 2] if dz else (1.0 - frac[:, 2])
                occ.index_put_((ix, iy, iz), wx * wy * wz, accumulate=True)
    return occ.clamp(0.0, 1.0)


def _soft_iou(occ_pred: torch.Tensor, occ_gt: torch.Tensor, eps: float = 1e-8):
    """
    Soft IoU using min/max (differentiable).
    identical occupancy -> IoU = 1 (exact by min/max properties).
    """
    inter = torch.min(occ_pred, occ_gt).sum()
    union = torch.max(occ_pred, occ_gt).sum()
    if union < 1e-6:
        return torch.tensor(1.0, device=occ_pred.device, dtype=occ_pred.dtype)
    return inter / (union + eps)


def compute_voxel_iou_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    resolution: int = 16,
    weight: float = 1.0,
    device: Optional[torch.device] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Per-object voxel occupancy IoU loss: GT placement vs Pred placement.

    Quantize mesh_points ([-1,1] normalized) into an R^3 voxel grid,
    place them into scene space with GT/Pred pose, then compare soft IoU.

    NOTE: batch["voxels"] is [0,~1] normalized (min->origin, max_extent->1) while
          mesh_points is [-1,1] normalized (center->origin, max_abs->1), so the
          coordinate frames differ -> use mesh_points-based _canonical_voxel_coords.

    loss = 1 - IoU  (mean)

    Differentiable: gradients flow through Pred via soft trilinear splatting.
    """
    if "mesh_points" not in batch:
        return torch.tensor(0.0, device=device)

    points = batch["mesh_points"]                                          # (N, P, 3)
    dev = device or points.device
    R = resolution

    # GT poses
    rot_gt   = batch["6drotation_normalized"]
    scale_gt = batch["scale"]
    trans_gt = batch["translation"]

    # Pred poses
    rot_pred   = outputs["6drotation_normalized"].squeeze(1)
    scale_pred = outputs["scale"].squeeze(1)
    trans_pred = outputs["translation"].squeeze(1)

    if rot_gt.dim() == 3:
        rot_gt   = rot_gt.squeeze(1)
        scale_gt = scale_gt.squeeze(1)
        trans_gt = trans_gt.squeeze(1)

    total_loss = torch.tensor(0.0, device=dev)
    num_valid  = 0
    debug_union_zero = 0
    idx = 0

    for n_tensor in num_parts:
        n = int(n_tensor.item())
        for i in range(n):
            g = idx + i

            # 1. mesh_points [-1,1] → canonical voxel centers [-1,1]
            with torch.no_grad():
                centers = _canonical_voxel_coords(points[g], R)            # (K, 3)
            K = centers.shape[0]
            if K == 0:
                continue
            centers_b = centers.unsqueeze(0)                               # (1, K, 3)

            # 2. GT scene positions (detached)
            with torch.no_grad():
                gt_scene = _apply_pose_transform(
                    centers_b,
                    rot_gt[g:g+1], scale_gt[g:g+1], trans_gt[g:g+1],
                ).squeeze(0)                                               # (K, 3)

            # 3. Pred scene positions (differentiable)
            pred_scene = _apply_pose_transform(
                centers_b,
                rot_pred[g:g+1], scale_pred[g:g+1], trans_pred[g:g+1],
            ).squeeze(0)                                                   # (K, 3)

            # 4. Shared grid from GT bbox (detached)
            with torch.no_grad():
                gt_min  = gt_scene.min(dim=0)[0]
                gt_max  = gt_scene.max(dim=0)[0]
                margin  = (gt_max - gt_min).max() * 0.1
                g_origin = gt_min - margin
                g_extent = (gt_max - gt_min).max() + 2 * margin
                g_extent = g_extent.clamp(min=1e-8)

            # 5. Soft voxelize: GT detached, Pred differentiable
            with torch.no_grad():
                occ_gt = _soft_voxelize_scene(gt_scene, R, g_origin, g_extent)
            occ_pred = _soft_voxelize_scene(pred_scene, R, g_origin, g_extent)

            # 6. Soft IoU → loss = 1 - IoU
            iou = _soft_iou(occ_pred, occ_gt)
            obj_loss = 1.0 - iou

            if debug:
                with torch.no_grad():
                    gt_occ_cnt  = int(occ_gt.sum().item())
                    pred_occ_cnt = int((occ_pred > 0.5).sum().item())
                    union_cnt   = int(((occ_gt > 0) | (occ_pred > 0.5)).sum().item())
                    if union_cnt == 0:
                        debug_union_zero += 1
                    print(f"  [VoxIoU] obj={g}: K={K}, gt_occ={gt_occ_cnt}, "
                          f"pred_occ={pred_occ_cnt}, IoU={iou.item():.4f}, "
                          f"loss={obj_loss.item():.4f}")

            total_loss = total_loss + obj_loss
            num_valid += 1
        idx += n

    if debug and debug_union_zero > 0:
        print(f"  [VoxIoU] WARNING: union=0 occurred {debug_union_zero} times")

    if num_valid > 0:
        total_loss = total_loss / num_valid

    return total_loss * weight

def compute_iou_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    weight: float = 0.1,
    device: Optional[torch.device] = None,
    iou_mode: str = "voxel",
    iou_voxel_resolution: int = 16,
    iou_debug: bool = False,
) -> torch.Tensor:
    """
    IoU loss router. Calls AABB or Voxel IoU loss depending on iou_mode.

    Args:
        iou_mode : "voxel" (default) | "aabb" (legacy)
    """
    if iou_mode == "aabb":
        return compute_iou_loss_aabb(
            outputs, batch, num_parts, weight=weight, device=device,
        )
    else:
        return compute_voxel_iou_loss(
            outputs, batch, num_parts,
            resolution=iou_voxel_resolution, weight=weight,
            device=device, debug=iou_debug,
        )


def compute_translation_spacing_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Translation pairwise spacing ratio loss.

    For each scene, compute pairwise distances between trainable object translations (centers),
    and evaluate whether the GT distance-ratio structure is preserved in the prediction.

    Normalization: divide both GT and pred by the mean GT pairwise distance in the scene.
    (evaluates relative ratio structure, not absolute scale difference)

    * skip scenes with <= 1 object (pairwise not computable)
    * exclude trainable=False objects

    Args:
        outputs  : transformer output dict ('translation': (N_total, 1, 3))
        batch    : GT batch dict ('translation': (N_total, 3), 'trainable': (N_total,) bool)
        num_parts: (B,) number of objects per scene
        weight   : loss weight

    Returns:
        scalar spacing-ratio L1 loss
    """
    t_pred = outputs["translation"].squeeze(1)  # (N_total, 3)
    t_gt   = batch["translation"]               # (N_total, 3)
    _dev   = device or t_gt.device

    trainable = batch.get(
        "trainable",
        torch.ones(t_gt.shape[0], dtype=torch.bool, device=_dev),
    )
    trainable = trainable.bool().to(_dev)

    total_loss = torch.tensor(0.0, device=_dev)
    n_valid    = 0
    offset     = 0

    for n_tensor in num_parts:
        n    = int(n_tensor.item())
        mask = trainable[offset : offset + n]

        tp = t_pred[offset : offset + n][mask]  # (K, 3)
        tg = t_gt  [offset : offset + n][mask]  # (K, 3)
        offset += n

        K = tp.shape[0]
        if K < 2:
            continue

        # pairwise distance matrix (K, K)
        D_gt   = torch.cdist(tg, tg, p=2)
        D_pred = torch.cdist(tp, tp, p=2)

        # use only upper triangle (i < j) - remove duplicates
        tri = torch.triu(torch.ones(K, K, dtype=torch.bool, device=_dev), diagonal=1)
        d_gt   = D_gt  [tri]   # (K*(K-1)/2,)
        d_pred = D_pred[tri]

        # normalize by mean GT distance -> scale-invariant ratio comparison
        mean_gt   = d_gt.mean().clamp(min=1e-6)
        d_gt_norm   = d_gt   / mean_gt
        d_pred_norm = d_pred / mean_gt

        total_loss = total_loss + (d_gt_norm - d_pred_norm).abs().mean()
        n_valid   += 1

    if n_valid > 0:
        total_loss = total_loss / n_valid

    return total_loss * weight


def compute_proximity_contact_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    weight: float = 1.0,
    threshold: float = 0.05,
    n_sample: int = 512,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Proximity contact loss: encourage nearby object pairs to touch.

    For each scene, transform object-pair mesh_points into world space, then
    compute the per-pair min distance.
    - min_dist < threshold -> loss = min_dist (push toward contact)
    - min_dist >= threshold -> loss = 0 (already far, leave as is)

    Args:
        threshold: apply contact loss within this distance (scene-normalized space)
        n_sample: number of points sampled per object (speed optimization)
    """
    if "mesh_points" not in batch:
        return torch.tensor(0.0, device=device)

    mp = batch["mesh_points"].float()  # (N_total, P, 3)
    pred_rot = outputs["6drotation_normalized"].squeeze(1)
    pred_scale = outputs["scale"].squeeze(1)
    pred_trans = outputs["translation"].squeeze(1)

    if pred_scale.dim() == 2 and pred_scale.shape[-1] > 1:
        pred_scale = pred_scale.mean(-1, keepdim=True)

    # World space points
    world_pts = _apply_pose_transform(mp, pred_rot, pred_scale, pred_trans)  # (N, P, 3)

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0
    idx = 0

    for n_p in num_parts:
        n_p = int(n_p.item())
        if n_p < 2:
            idx += n_p
            continue

        pts_scene = world_pts[idx: idx + n_p]  # (n_p, P, 3)

        # Subsample for speed
        if pts_scene.shape[1] > n_sample:
            perm = torch.randperm(pts_scene.shape[1], device=pts_scene.device)[:n_sample]
            pts_scene = pts_scene[:, perm]

        # Pairwise min distances
        for i in range(n_p):
            for j in range(i + 1, n_p):
                # min distance between object i and j
                d = torch.cdist(pts_scene[i], pts_scene[j])  # (S, S)
                min_dist = d.min()

                if min_dist < threshold:
                    total_loss = total_loss + min_dist
                    n_valid += 1

        idx += n_p

    if n_valid > 0:
        total_loss = total_loss / n_valid

    return total_loss * weight


def compute_symmetry_aware_rot_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Symmetry-aware rotation loss.

    For each object, generate 4 yaw-only canonical candidates of the GT rotation
    (0, 90, 180, 270 deg) and use the minimum L1 loss against the prediction.
    This absorbs canonical-rotation ambiguity of rotationally symmetric objects
    (round tables, square boxes, etc.).

    Args:
        outputs  : transformer output dict ('6drotation_normalized': (N, 1, 6))
        batch    : GT batch dict ('6drotation_normalized': (N, 6) or (N, 1, 6))
        weight   : loss weight

    Returns:
        scalar symmetry-aware rotation L1 loss
    """
    rot_pred = outputs["6drotation_normalized"].squeeze(1)   # (N, 6)
    rot_gt   = batch["6drotation_normalized"]                # (N, 6) or (N, 1, 6)
    if rot_gt.dim() == 3:
        rot_gt = rot_gt.squeeze(1)

    dev = device or rot_gt.device
    N   = rot_gt.shape[0]

    # 4 yaw rotations about the Y axis (canonical candidates)
    yaw_angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    losses = []

    rot_gt_mat = rotation_6d_to_matrix(rot_gt)               # (N, 3, 3)

    for yaw in yaw_angles:
        R_yaw = euler_angles_to_matrix(
            torch.tensor([0.0, yaw, 0.0], device=dev), "XYZ"
        )                                                     # (3, 3)
        # new canonical = R_yaw @ original canonical
        # to keep the same world-space pose: R_gt_new = R_gt @ R_yaw^T
        R_cand_mat  = torch.bmm(
            rot_gt_mat,
            R_yaw.unsqueeze(0).expand(N, -1, -1).transpose(-1, -2),
        )                                                     # (N, 3, 3)
        rot_cand_6d = matrix_to_rotation_6d(R_cand_mat)      # (N, 6)
        loss_cand   = (rot_pred - rot_cand_6d).abs().mean(dim=-1)  # (N,)
        losses.append(loss_cand)

    # pick the minimum among the 4 candidates per object
    min_loss = torch.stack(losses, dim=-1).min(dim=-1).values  # (N,)
    return min_loss.mean() * weight

def trimmed_chamfer(A, B, keep=0.9):
    """A, B: (N, 3) — Trimmed bidirectional chamfer distance."""
    d = torch.cdist(A, B, p=2)
    a2b = d.min(1).values
    b2a = d.min(0).values
    th1 = a2b.kthvalue(int(max(1, keep * a2b.numel()))).values
    th2 = b2a.kthvalue(int(max(1, keep * b2a.numel()))).values
    loss = (a2b[a2b <= th1] ** 2).mean() + (b2a[b2a <= th2] ** 2).mean()
    return loss


def trimmed_chamfer_one_sided(A, B, keep=0.9):
    """A, B: (N, 3) — Trimmed one-sided chamfer distance: A → B only.

    When only the visible surface exists (like pm_surface), the B -> A direction
    unfairly penalizes the mesh backside, so only the A -> B direction is computed.
    """
    d = torch.cdist(A, B, p=2)
    a2b = d.min(1).values
    th = a2b.kthvalue(int(max(1, keep * a2b.numel()))).values
    return (a2b[a2b <= th] ** 2).mean()


def compute_pointmap_surface_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weight: float = 1.0,
    device: Optional[torch.device] = None,
    trim_ratio: float = 0.3,
    one_sided_weight: float = 5.0,
) -> torch.Tensor:
    """
    Trimmed Chamfer Distance: bidirectional + one-sided (mesh_cam → pm_surf).

    Compare the canonical mesh points (transformed by the predicted pose) with the pointmap visible-surface points.

    loss = trimmed_chamfer_bi(mesh, pm)           # bidirectional (original)
         + one_sided_weight * trimmed_chamfer_one_sided(mesh, pm)  # extra mesh->pm

    With this setup the mesh -> pm direction acts (1 + one_sided_weight)x stronger,
    while the pm -> mesh direction stays at 1x.
    Since pm_surf is the partial surface visible from the camera, penalizing the
    mesh -> pm direction more strongly is geometrically correct.

    Args:
        outputs           : transformer output dict
        batch             : GT batch dict ('pm_surface_pts': (N, 256, 3), 'mesh_points': (N, 100, 3))
        weight            : overall loss weight
        trim_ratio        : fraction of top distances to trim (default 30%)
        one_sided_weight  : extra weight for the one-sided (mesh->pm) term (default 1.0)

    Returns:
        scalar trimmed chamfer surface loss
    """
    if "pm_surface_pts" not in batch:
        return torch.tensor(0.0)

    dev = device or outputs["6drotation_normalized"].device

    rot_pred  = outputs["6drotation_normalized"].squeeze(1)   # (N, 6)
    t_pred    = outputs["translation"].squeeze(1)             # (N, 3)
    s_pred    = outputs["scale"].squeeze(1)                   # (N, 1)
    mesh_pts  = batch["mesh_points"]                          # (N, 100, 3)
    pm_surf   = batch["pm_surface_pts"].to(dev)               # (N, 256, 3)

    mesh_cam = _apply_pose_transform(
        mesh_pts, rot_pred, s_pred, t_pred, normalized=True
    )  # (N, 100, 3)

    losses = []
    for i in range(rot_pred.shape[0]):
        pm_pts = pm_surf[i]                           # (256, 3)
        valid  = pm_pts.norm(dim=1) > 1e-6            # exclude padding zero rows
        if valid.sum() < 100:
            continue
        pm_pts = pm_pts[valid]                        # (K, 3)
        mp = mesh_cam[i]       
        

        # xyz percentile comparison: xy uses 05/95, z GT min uses 10
        mp_min = torch.quantile(mp[:, :3], 0.05, dim=0)        # (3,) [x, y, z] lower 5%
        mp_max = torch.quantile(mp[:, :2], 0.95, dim=0)        # (3,) [x, y, z] upper 5%
        pm_min_xy = torch.quantile(pm_pts[:, :2], 0.05, dim=0)  # (2,) [x, y] lower 5%
        pm_min_z = torch.quantile(pm_pts[:, 2], 0.10)           # z lower 10%
        pm_min = torch.cat([pm_min_xy, pm_min_z.unsqueeze(0)])  # (3,) [x, y, z]
        pm_max = torch.quantile(pm_pts[:, :2], 0.95, dim=0)    # (3,) [x, y, z] upper 5%

        bbox_loss = (pm_max - mp_max).abs().sum() + (pm_min - mp_min).abs().sum()

        keep = 1.0 - trim_ratio
        loss_one = trimmed_chamfer_one_sided(pm_pts, mp, keep=keep) * 10 
        losses.append(loss_one + bbox_loss)
        
    if not losses:
        return torch.tensor(0.0, device=dev)
    return torch.stack(losses).mean() * weight


def compute_projection_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    proj_resolution: int = 128,
    seg_weight: float = 1.0,
    depth_weight: float = 1.0,
    seg_loss_type: str = "dice",
    depth_loss_type: str = "l1",
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    nvdiffrast-based differentiable projection loss.

    Render the canonical mesh with the predicted pose and compare against the GT seg mask.
    nvdiffrast antialiasing provides edge-aware gradients.

    Args:
        outputs    : model outputs (translation, 6drotation_normalized, scale)
        batch      : data batch (meshes, rgb_image_mask, proj_unnorm_scale,
                     proj_unnorm_center)
        num_parts  : (B,) per-scene object count
        proj_resolution : render resolution (square)
        seg_weight / depth_weight : per-loss weight (skip if 0)
        device     : computation device

    Returns:
        dict: {"proj_seg_loss": ...} (after applying weights)
    """
    from src.projection_nvdiff import (
        NvdiffrastProjection, _load_canonical_mesh,
        apply_pose_to_mesh, soft_dice_loss, union_silhouette,
        DEFAULT_FOV,
    )

    dev = device or outputs["translation"].device
    result = {}

    if seg_weight <= 0 and depth_weight <= 0:
        return result

    # -- check required data --
    if "meshes" not in batch:
        return result

    mesh_paths = batch["meshes"]
    rot_pred = outputs["6drotation_normalized"].squeeze(1).to(dev)  # (N, 6)
    t_pred   = outputs["translation"].squeeze(1).to(dev)            # (N, 3)
    s_pred   = outputs["scale"].squeeze(1).to(dev)                  # (N, 1)

    N_total = rot_pred.shape[0]

    us_all = batch.get("proj_unnorm_scale")    # (B,) or None
    uc_all = batch.get("proj_unnorm_center")   # (B, 3) or None
    gt_masks = batch.get("rgb_image_mask")     # (N, 1, H, W)
    fov_all = batch.get("fov")*1.03                 # (B,) or None
    fx_all  = batch.get("fx")                  # (B,) or None
    fy_all  = batch.get("fy")                  # (B,) or None

    proj = NvdiffrastProjection(
        image_size=(proj_resolution, proj_resolution),
        fov=DEFAULT_FOV,
        device=dev,
    )

    seg_losses = []

    # -- per-scene processing (batch render objects) --
    parts = num_parts.cpu().tolist()
    B = len(parts)
    obj_offset = 0

    for b_idx in range(B):
        n_obj = int(parts[b_idx])
        if n_obj == 0:
            obj_offset += n_obj
            continue

        # 1. load meshes of all objects in the scene + apply pose
        verts_list = []
        faces_list = []
        valid_indices = []

        for i in range(n_obj):
            g_idx = obj_offset + i
            if g_idx >= N_total:
                break

            mesh_path = mesh_paths[g_idx] if isinstance(mesh_paths, list) else mesh_paths
            try:
                verts_np, faces_np = _load_canonical_mesh(mesh_path)
            except Exception:
                continue

            verts_can = torch.tensor(verts_np, dtype=torch.float32, device=dev)
            faces_t = torch.tensor(faces_np, dtype=torch.int64, device=dev)

            verts_cam = apply_pose_to_mesh(
                verts_can, rot_pred[g_idx], s_pred[g_idx], t_pred[g_idx],
            )

            if us_all is not None and uc_all is not None:
                us_i = us_all[b_idx].to(dev)
                uc_i = uc_all[b_idx].to(dev)
                verts_cam = verts_cam * us_i + uc_i

            verts_list.append(verts_cam)
            faces_list.append(faces_t)
            valid_indices.append(g_idx)

        if not verts_list:
            obj_offset += n_obj
            continue

        # 2. render the whole scene once with nvdiffrast
        scene_fov = float(fov_all[b_idx]) if fov_all is not None else None
        scene_fx, scene_fy = None, None
        if fx_all is not None and fy_all is not None:
            scale_render = proj_resolution / 518.0
            scene_fx = float(fx_all[b_idx]) * scale_render
            scene_fy = float(fy_all[b_idx]) * scale_render
        try:
            render_out = proj(verts_list, faces_list, fov=scene_fov,
                              return_per_obj=True, fx=scene_fx, fy=scene_fy)
        except Exception:
            obj_offset += n_obj
            continue

        # 3. compute seg loss
        if seg_weight > 0 and gt_masks is not None:
            # 3a. Per-object Dice loss (antialias silhouette)
            per_obj_dice = []
            sil_per_obj = render_out["silhouette_per_obj"]  # (N_obj, H, W)

            for local_idx, g_idx in enumerate(valid_indices):
                P_soft = sil_per_obj[local_idx]  # (H, W) — antialiased

                gt_m = gt_masks[g_idx, 0].to(dev).float()
                gt_m_small = F.interpolate(
                    gt_m.unsqueeze(0).unsqueeze(0),
                    size=(proj_resolution, proj_resolution),
                    mode='bilinear', align_corners=False,
                ).squeeze().clamp(0, 1)

                dice = soft_dice_loss(P_soft, gt_m_small)

                # Threshold: skip objects with low GT agreement
                with torch.no_grad():
                    mean_diff = (P_soft - gt_m_small).abs().mean()
                    if dice.item() >= 0.3 or mean_diff.item() >= 0.005:
                        continue

                per_obj_dice.append(dice)

            # 3b. Union mask Dice loss (antialias)
            union_soft = union_silhouette(
                render_out["rast_out"],
                render_out["combined_verts"],
                render_out["combined_faces"],
            )

            gt_union_list = []
            for g_idx in valid_indices:
                gt_m = gt_masks[g_idx, 0].to(dev).float()
                gt_m_small = F.interpolate(
                    gt_m.unsqueeze(0).unsqueeze(0),
                    size=(proj_resolution, proj_resolution),
                    mode='bilinear', align_corners=False,
                ).squeeze().clamp(0, 1)
                gt_union_list.append(gt_m_small)
            gt_union = torch.stack(gt_union_list).max(dim=0).values

            union_dice = soft_dice_loss(union_soft, gt_union)

            if per_obj_dice:
                per_obj_mean = torch.stack(per_obj_dice).mean()
                scene_seg_loss = 0.5 * per_obj_mean + 0.5 * union_dice
            else:
                scene_seg_loss = union_dice

            seg_losses.append(scene_seg_loss)

        obj_offset += n_obj

    # -- assemble results --
    if seg_weight > 0:
        if seg_losses:
            result["proj_seg_loss"] = torch.stack(seg_losses).mean() * seg_weight
        else:
            result["proj_seg_loss"] = torch.tensor(0.0, device=dev, requires_grad=True)

    return result


def compute_rendering_mask_loss(
    render_result: dict,
    gt_masks: torch.Tensor,
    valid_indices: list,
    proj_resolution: int,
    dice_weight: float = 0.75,
    bce_weight: float = 0.25,
    device: torch.device = None,
    dice_threshold: float = 0.3,
    diff_threshold: float = 0.005,
) -> torch.Tensor:
    """
    Per-object Dice + BCE between rendered silhouette and GT mask.

    Objects with low agreement between GT and the render are excluded from the loss:
      - dice loss >= dice_threshold → skip
      - mean |diff| >= diff_threshold → skip

    Args:
        render_result    : NvdiffrastProjection forward result dict
                           ("silhouette_per_obj": (N_obj, H, W))
        gt_masks         : (N_total, 1, H, W) or (N_total, H, W) — GT binary masks
        valid_indices    : list[int] - global object indices actually rendered
        proj_resolution  : render resolution (H=W)
        dice_weight      : Dice loss weight (default 0.75)
        bce_weight       : BCE loss weight (default 0.25)
        device           : computation device
        dice_threshold   : skip objects with dice loss above this (default 0.3)
        diff_threshold   : skip objects with mean |diff| above this (default 0.005)

    Returns:
        scalar mask loss (Dice + BCE sum, only objects passing threshold)
    """
    sil_per_obj = render_result.get("silhouette_per_obj")
    if sil_per_obj is None or gt_masks is None:
        return torch.tensor(0.0, device=device)

    losses = []
    for local_idx, g_idx in enumerate(valid_indices):
        P_soft = sil_per_obj[local_idx]  # (H, W)

        gt_m = gt_masks[g_idx, 0].to(device).float() if gt_masks.dim() == 4 else gt_masks[g_idx].to(device).float()
        gt_m_small = F.interpolate(
            gt_m.unsqueeze(0).unsqueeze(0),
            size=(proj_resolution, proj_resolution),
            mode='bilinear', align_corners=False,
        ).squeeze().clamp(0.0, 1.0)

        # Dice loss
        smooth = 1.0
        intersection = (P_soft * gt_m_small).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (P_soft.sum() + gt_m_small.sum() + smooth)

        # Threshold filtering: skip objects with low GT agreement
        with torch.no_grad():
            mean_diff = (P_soft - gt_m_small).abs().mean()
            if dice.item() >= dice_threshold or mean_diff.item() >= diff_threshold:
                continue

        # BCE loss
        bce = F.binary_cross_entropy(P_soft.clamp(1e-6, 1.0 - 1e-6), gt_m_small)

        obj_loss = dice_weight * dice + bce_weight * bce
        losses.append(obj_loss)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def compute_rendering_bbox2d_loss(
    render_result: dict,
    gt_masks: torch.Tensor,
    valid_indices: list,
    proj_resolution: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Soft 2D BBox L1 loss between rendered silhouette bbox and GT mask bbox.

    Uses differentiable bbox extraction based on weighted mean +/- std.
    Compares each object's rendered-silhouette bbox against the GT mask bbox.

    Args:
        render_result    : NvdiffrastProjection forward result dict
        gt_masks         : (N_total, 1, H, W) GT binary masks
        valid_indices    : list[int]
        proj_resolution  : render resolution
        device           : computation device

    Returns:
        scalar 2D bbox L1 loss
    """
    sil_per_obj = render_result.get("silhouette_per_obj")
    if sil_per_obj is None or gt_masks is None:
        return torch.tensor(0.0, device=device)

    H = W = proj_resolution
    # coordinate grids [0, 1]
    ys = torch.linspace(0.0, 1.0, H, device=device)  # (H,)
    xs = torch.linspace(0.0, 1.0, W, device=device)  # (W,)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    def _soft_bbox(sil: torch.Tensor):
        """Differentiable soft bbox: returns (x_min, x_max, y_min, y_max) in [0,1]."""
        w_sum = sil.sum().clamp(min=1e-8)
        cx = (sil * grid_x).sum() / w_sum
        cy = (sil * grid_y).sum() / w_sum
        std_x = ((sil * (grid_x - cx) ** 2).sum() / w_sum).clamp(min=0.0).sqrt()
        std_y = ((sil * (grid_y - cy) ** 2).sum() / w_sum).clamp(min=0.0).sqrt()
        return cx - std_x, cx + std_x, cy - std_y, cy + std_y

    losses = []
    for local_idx, g_idx in enumerate(valid_indices):
        P_soft = sil_per_obj[local_idx]  # (H, W)

        gt_m = gt_masks[g_idx, 0].to(device).float() if gt_masks.dim() == 4 else gt_masks[g_idx].to(device).float()
        gt_m_small = F.interpolate(
            gt_m.unsqueeze(0).unsqueeze(0),
            size=(proj_resolution, proj_resolution),
            mode='bilinear', align_corners=False,
        ).squeeze().clamp(0.0, 1.0)

        if gt_m_small.sum() < 1.0:
            continue

        pred_bbox = _soft_bbox(P_soft)
        with torch.no_grad():
            gt_bbox = _soft_bbox(gt_m_small)

        bbox_pred = torch.stack(list(pred_bbox))  # (4,)
        bbox_gt = torch.stack(list(gt_bbox))      # (4,)
        losses.append((bbox_pred - bbox_gt).abs().mean())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def compute_rendering_depth_loss(
    render_result: dict,
    gt_depth: torch.Tensor,
    gt_masks_combined: torch.Tensor,
    proj_resolution: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    L1 depth loss on GT mask region only.

    Compare GT depth (scene_pointmap Z channel) and rendered depth only inside
    the GT mask.

    Args:
        render_result      : NvdiffrastProjection forward result dict ("depth": (H,W))
        gt_depth           : (H, W) — camera-space GT depth (unnormalized)
        gt_masks_combined  : (H, W) — union of all object masks (binary)
        proj_resolution    : render resolution
        device             : computation device

    Returns:
        scalar depth L1 loss
    """
    pred_depth = render_result.get("depth")  # (H, W), -1 = background
    if pred_depth is None or gt_depth is None:
        return torch.tensor(0.0, device=device)

    # resize GT depth to the render resolution
    # gt_depth: (H, W) expected, defensively fix dimensions
    _gt_depth = gt_depth.float()
    if _gt_depth.ndim == 1:
        # if flattened - skip (not a normal case)
        return torch.tensor(0.0, device=device)
    while _gt_depth.ndim < 4:
        _gt_depth = _gt_depth.unsqueeze(0)
    gt_d_small = F.interpolate(
        _gt_depth,
        size=(proj_resolution, proj_resolution),
        mode='bilinear', align_corners=False,
    ).squeeze(0).squeeze(0)  # (proj_res, proj_res)

    _gt_mask = gt_masks_combined.float()
    while _gt_mask.ndim < 4:
        _gt_mask = _gt_mask.unsqueeze(0)
    gt_mask_small = F.interpolate(
        _gt_mask,
        size=(proj_resolution, proj_resolution),
        mode='bilinear', align_corners=False,
    ).squeeze(0).squeeze(0) > 0.5

    # exclude background (-1) from rendered depth
    valid = gt_mask_small & (pred_depth > 0)
    if valid.sum() < 16:
        return torch.tensor(0.0, device=device)

    # check valid GT depth values (exclude 0 or nan)
    gt_valid = gt_d_small[valid]
    pred_valid = pred_depth[valid]
    valid_gt_mask = gt_valid.isfinite() & (gt_valid.abs() > 1e-6)
    if valid_gt_mask.sum() < 16:
        return torch.tensor(0.0, device=device)

    return (pred_valid[valid_gt_mask] - gt_valid[valid_gt_mask]).abs().mean()


def compute_all_rendering_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    render_resolution: tuple = (128, 128),
    mask_weight: float = 1.0,
    bbox2d_weight: float = 2.0,
    depth_weight: float = 0.5,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """
    Router function for render-based losses.

    Render the canonical mesh with the predicted pose and compare against GT mask/depth to
    compute mask loss, 2D bbox loss, and depth loss.

    Follows the same mesh-loading / pose-application / rendering pattern as compute_projection_losses.

    Args:
        outputs         : transformer output dict (translation, 6drotation_normalized, scale)
        batch           : data batch (meshes, rgb_image_mask, scene_pointmap,
                          proj_unnorm_scale, proj_unnorm_center, fov)
        num_parts       : (B,) per-scene object count
        render_resolution : (H, W) - render resolution (default 128x128)
        mask_weight     : rendering mask loss weight
        bbox2d_weight   : 2D bbox loss weight
        depth_weight    : depth loss weight
        device          : computation device

    Returns:
        dict:
            "mask_loss"    : per-object Dice+BCE mask loss
            "bbox2d_loss"  : per-object 2D bbox L1 loss
            "depth_loss"   : scene-level depth L1 loss
            "render_total" : weighted total
    """
    from src.projection_nvdiff import (
        NvdiffrastProjection, _load_canonical_mesh,
        apply_pose_to_mesh, DEFAULT_FOV,
    )

    dev = device or outputs["translation"].device
    result: Dict[str, torch.Tensor] = {}

    if "meshes" not in batch:
        zero = torch.tensor(0.0, device=dev)
        return {"mask_loss": zero, "bbox2d_loss": zero, "depth_loss": zero, "render_total": zero}

    proj_res_h, proj_res_w = render_resolution
    assert proj_res_h == proj_res_w, "only square resolution is currently supported"
    proj_resolution = proj_res_h

    mesh_paths = batch["meshes"]
    rot_pred = outputs["6drotation_normalized"].squeeze(1).to(dev)  # (N, 6)
    t_pred   = outputs["translation"].squeeze(1).to(dev)            # (N, 3)
    s_pred   = outputs["scale"].squeeze(1).to(dev)                  # (N, 1)

    N_total = rot_pred.shape[0]

    us_all    = batch.get("proj_unnorm_scale")   # (B,) or None
    uc_all    = batch.get("proj_unnorm_center")  # (B, 3) or None
    gt_masks  = batch.get("rgb_image_mask")      # (N, 1, H, W) or None
    fov_all   = batch.get("fov")                 # (B,) or None
    fx_all    = batch.get("fx")                  # (B,) or None - focal length at 518px
    fy_all    = batch.get("fy")                  # (B,) or None
    scene_pm  = batch.get("scene_pointmap")      # (B, 3, H, W) or None

    proj = NvdiffrastProjection(
        image_size=(proj_resolution, proj_resolution),
        fov=DEFAULT_FOV,
        device=dev,
    )

    mask_losses   = []
    bbox2d_losses = []
    depth_losses  = []

    parts = num_parts.cpu().tolist()
    B = len(parts)
    obj_offset = 0

    for b_idx in range(B):
        n_obj = int(parts[b_idx])
        if n_obj == 0:
            obj_offset += n_obj
            continue

        # 1. load meshes of all objects in the scene + apply pose
        verts_list   = []
        faces_list   = []
        valid_indices = []

        for i in range(n_obj):
            g_idx = obj_offset + i
            if g_idx >= N_total:
                break

            mesh_path = mesh_paths[g_idx] if isinstance(mesh_paths, list) else mesh_paths
            try:
                verts_np, faces_np = _load_canonical_mesh(mesh_path)
            except Exception:
                continue

            verts_can = torch.tensor(verts_np, dtype=torch.float32, device=dev)
            faces_t   = torch.tensor(faces_np, dtype=torch.int64, device=dev)

            verts_cam = apply_pose_to_mesh(
                verts_can, rot_pred[g_idx], s_pred[g_idx], t_pred[g_idx],
            )

            # Unnormalize: model space → actual camera space
            if us_all is not None and uc_all is not None:
                us_i = us_all[b_idx].to(dev)
                uc_i = uc_all[b_idx].to(dev)
                verts_cam = verts_cam * us_i + uc_i

            verts_list.append(verts_cam)
            faces_list.append(faces_t)
            valid_indices.append(g_idx)

        if not verts_list:
            obj_offset += n_obj
            continue

        # 2. render the whole scene once with nvdiffrast
        scene_fov = float(fov_all[b_idx]) if fov_all is not None else None
        # if fx/fy exist, rescale to the render resolution (from 518px to proj_resolution)
        scene_fx, scene_fy = None, None
        if fx_all is not None and fy_all is not None:
            scale_render = proj_resolution / 518.0  # 518 = model input resolution
            scene_fx = float(fx_all[b_idx]) * scale_render
            scene_fy = float(fy_all[b_idx]) * scale_render
        try:
            render_out = proj(verts_list, faces_list, fov=scene_fov,
                              return_per_obj=True, fx=scene_fx, fy=scene_fy)
        except Exception:
            obj_offset += n_obj
            continue

        # Y-flip: reconcile nvdiffrast OpenGL clip-space Y axis with image coordinates
        if "silhouette_per_obj" in render_out:
            render_out["silhouette_per_obj"] = [
                s.flip(0) for s in render_out["silhouette_per_obj"]
            ]
        if "depth" in render_out:
            render_out["depth"] = render_out["depth"].flip(0)

        # 3. Mask loss
        if mask_weight > 0 and gt_masks is not None:
            ml = compute_rendering_mask_loss(
                render_out, gt_masks, valid_indices, proj_resolution, device=dev,
            )
            mask_losses.append(ml)

        # 4. 2D BBox loss
        if bbox2d_weight > 0 and gt_masks is not None:
            bl = compute_rendering_bbox2d_loss(
                render_out, gt_masks, valid_indices, proj_resolution, device=dev,
            )
            bbox2d_losses.append(bl)

        # 5. Depth loss
        if depth_weight > 0 and scene_pm is not None:
            # scene_pointmap: (B, 3, H, W) → Z channel (index 2)
            gt_depth_full = scene_pm[b_idx, 2].to(dev)  # (H, W)

            # unnormalize depth: pointmap is already camera space, so only unnorm is applied
            if us_all is not None and uc_all is not None:
                us_i = us_all[b_idx].float().to(dev)
                uc_i = uc_all[b_idx].float().to(dev)  # (1, 3) or (3,)
                uc_z = uc_i.view(-1)[2]
                gt_depth_full = gt_depth_full * us_i.squeeze() + uc_z

            # union of GT masks (valid region only)
            if gt_masks is not None:
                gt_union = torch.zeros(gt_masks.shape[-2], gt_masks.shape[-1], device=dev)
                for g_idx in valid_indices:
                    gt_union = torch.max(gt_union, gt_masks[g_idx, 0].to(dev).float())
            else:
                gt_union = (gt_depth_full > 0).float()

            dl = compute_rendering_depth_loss(
                render_out, gt_depth_full, gt_union, proj_resolution, device=dev,
            )
            depth_losses.append(dl)

        obj_offset += n_obj

    # assemble results
    zero = torch.tensor(0.0, device=dev)

    mask_loss_val   = torch.stack(mask_losses).mean()   if mask_losses   else zero
    bbox2d_loss_val = torch.stack(bbox2d_losses).mean() if bbox2d_losses else zero
    depth_loss_val  = torch.stack(depth_losses).mean()  if depth_losses  else zero

    # NaN guard: replace per-loss NaN with 0
    if mask_loss_val.isnan():
        mask_loss_val = zero
    if bbox2d_loss_val.isnan():
        bbox2d_loss_val = zero
    if depth_loss_val.isnan():
        depth_loss_val = zero

    render_total = (
        mask_weight   * mask_loss_val +
        bbox2d_weight * bbox2d_loss_val +
        depth_weight  * depth_loss_val
    )

    return {
        "mask_loss":    mask_loss_val,
        "bbox2d_loss":  bbox2d_loss_val,
        "depth_loss":   depth_loss_val,
        "render_total": render_total,
    }


def compute_penetration_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    resolution: int = 32,
    weight: float = 1.0,
    device: Optional[torch.device] = None,
    normalize: str = "union",
    debug: bool = False,
) -> torch.Tensor:
    """
    Predictions-only self-collision loss via soft voxel overlap on a shared grid.

    Uses world-frame pose keys: outputs["translation"], outputs["6drotation_normalized"], outputs["scale"].
    Per scene: splat each object onto a shared voxel grid (auto-sized to the scene bbox
    with 10% margin), compute overlap_mass = sum_occ(x) - any_occ(x) summed over voxels,
    normalize by union mass, and average across scenes.

    normalize: "union" (default) | "mean_obj_mass" | "none"
    """
    points = batch["mesh_points"]
    dev = device or points.device
    R = resolution

    rot_pred   = outputs["6drotation_normalized"].squeeze(1)
    scale_pred = outputs["scale"].squeeze(1)
    trans_pred = outputs["translation"].squeeze(1)

    total = torch.tensor(0.0, device=dev)
    nb = 0
    idx = 0
    for n_tensor in num_parts:
        n = int(n_tensor.item())
        if n < 2:
            idx += n
            continue
        scene_pts_list = []
        for i in range(n):
            g = idx + i
            with torch.no_grad():
                centers = _canonical_voxel_coords(points[g], R).unsqueeze(0)
            scene_pts_list.append(_apply_pose_transform(
                centers, rot_pred[g:g+1], scale_pred[g:g+1], trans_pred[g:g+1]
            ).squeeze(0))
        all_pts = torch.cat(scene_pts_list, dim=0)
        with torch.no_grad():
            lo = all_pts.min(0)[0]
            hi = all_pts.max(0)[0]
            margin = (hi - lo).max() * 0.1
            origin = lo - margin
            extent = ((hi - lo).max() + 2 * margin).clamp(min=1e-8)
        occ_list = [_soft_voxelize_scene(p, R, origin, extent) for p in scene_pts_list]
        stacked  = torch.stack(occ_list, dim=0)
        sum_occ  = stacked.sum(dim=0)
        any_occ  = stacked.amax(dim=0)
        overlap_mass = (sum_occ - any_occ).clamp(min=0.0).sum()
        if normalize == "union":
            ratio = overlap_mass / (any_occ.sum() + 1e-8)
        elif normalize == "mean_obj_mass":
            ratio = overlap_mass / (stacked.view(n, -1).sum(dim=1).mean() + 1e-8)
        else:
            ratio = overlap_mass
        if debug:
            print(f"  [Pen] n={n} overlap={overlap_mass.item():.2f} union={any_occ.sum().item():.2f} ratio={ratio.item():.4f}")
        total = total + ratio
        nb += 1
        idx += n
    if nb > 0:
        total = total / nb
    return total * weight


def compute_floor_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    num_parts: torch.Tensor,
    threshold: float = 0.05,
    weight: float = 1.0,
    device: Optional[torch.device] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Floor-alignment loss in the xz-plane frame (gravity = -y).

    Uses xz-plane-local pred keys: outputs["pred_translation"], outputs["pred_6drotation_normalized"],
    outputs["scale"] — these are BEFORE xz2f rotation.
    Per scene: compute min_y of each posed object. floor_y = min(min_y).detach() anchors the lowest
    object. Only objects with (min_y - floor_y) < threshold contribute; loss = mean over masked
    objects of (min_y - floor_y)**2. Objects above threshold (e.g. lamp on a table) are excluded.
    """
    points = batch["mesh_points"]
    dev = device or points.device

    rot_xz   = outputs["pred_6drotation_normalized"].squeeze(1)
    trans_xz = outputs["pred_translation"].squeeze(1)
    scale    = outputs["scale"].squeeze(1)

    total = torch.tensor(0.0, device=dev)
    nb = 0
    idx = 0
    for n_tensor in num_parts:
        n = int(n_tensor.item())
        if n < 1:
            idx += n
            continue
        min_ys = []
        for i in range(n):
            g = idx + i
            pts = _apply_pose_transform(
                points[g].unsqueeze(0), rot_xz[g:g+1], scale[g:g+1], trans_xz[g:g+1]
            ).squeeze(0)
            min_ys.append(pts[:, 1].min())
        min_y   = torch.stack(min_ys)
        floor_y = min_y.min().detach()
        dist    = min_y - floor_y
        mask    = (dist < threshold).float()
        scene_loss = (dist ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
        if debug:
            print(f"  [Floor] n={n} floor_y={floor_y.item():.3f} min_y={[round(y.item(),3) for y in min_y]} "
                  f"mask={[int(m) for m in mask]} loss={scene_loss.item():.5f}")
        total = total + scene_loss
        nb += 1
        idx += n
    if nb > 0:
        total = total / nb
    return total * weight


def compute_total_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, float], transformer=None, shape_latent=None) -> torch.Tensor:
    loss = 0.0
    detail_losses = {}
    
    # Chamfer Loss
    if "mesh_points" in batch and "predicted_mesh_points" in outputs:
        chamfer_loss = compute_chamfer_loss(outputs["predicted_mesh_points"], batch["mesh_points"], weights)
        loss += chamfer_loss
        detail_losses["chamfer_loss"] = chamfer_loss.item()
    
    # Normal Loss
    if "mesh_normals" in batch and "predicted_mesh_normals" in outputs:
        normal_loss = compute_normal_loss(outputs["predicted_mesh_normals"], batch["mesh_normals"], weights)
        loss += normal_loss
        detail_losses["normal_loss"] = normal_loss.item()
    
    return loss, detail_losses
    
