"""
Merged Dataset for MeshLayout.
Combines 3D-FUTURE and ScanNet datasets into a single unified dataset.
Each data_config has a 'dataset' field to differentiate processing logic.
"""
from src.utils.typing_utils import *

import json
import os
import random
import glob

import torch
from torchvision.transforms import functional as tf
import numpy as np
from PIL import Image
import trimesh
from tqdm import tqdm
from src.datasets.data_utils import (
    preprocess_image, listdict_to_dictlist_safe, mesh_to_voxel_tensor, augment_mesh_and_rotation
)
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion


from hydra.utils import instantiate
from omegaconf import OmegaConf

TEST_SET_ID = [
    'scene0011_00', 'scene0015_00', 'scene0019_00', 'scene0025_00',
    'scene0030_00', 'scene0046_02', 'scene0050_00', 'scene0063_00',
    'scene0064_00', 'scene0077_00', 'scene0081_00', 'scene0084_01',
    'scene0086_00', 'scene0088_00', 'scene0095_00', 'scene0100_02',
    'scene0131_00', 'scene0139_00', 'scene0144_00', 'scene0146_00',
    'scene0149_00', 'scene0153_00', 'scene0164_01', 'scene0169_00',
    'scene0187_00', 'scene0193_00', 'scene0196_00', 'scene0203_00',
    'scene0207_00', 'scene0208_00', 'scene0217_00', 'scene0222_00',
    'scene0231_00', 'scene0246_00', 'scene0249_00', 'scene0251_00',
    'scene0256_00', 'scene0257_00', 'scene0277_00', 'scene0278_00',
    'scene0300_00', 'scene0304_00', 'scene0307_00', 'scene0314_00',
    'scene0316_00', 'scene0328_00', 'scene0329_00', 'scene0334_00',
    'scene0338_00', 'scene0342_00', 'scene0343_00', 'scene0351_00',
    'scene0353_00', 'scene0354_00', 'scene0355_00', 'scene0356_00',
    'scene0357_00', 'scene0377_00', 'scene0378_00', 'scene0382_00',
    'scene0389_00', 'scene0406_00', 'scene0412_00', 'scene0414_00',
    'scene0423_00', 'scene0426_00', 'scene0427_00', 'scene0430_00',
    'scene0432_00', 'scene0435_00', 'scene0458_00', 'scene0461_00',
    'scene0462_00', 'scene0474_00', 'scene0490_00', 'scene0494_00',
    'scene0496_00', 'scene0518_00', 'scene0527_00', 'scene0535_00',
    'scene0550_00', 'scene0552_00', 'scene0553_00', 'scene0558_00',
    'scene0565_00', 'scene0568_00', 'scene0574_00', 'scene0575_00',
    'scene0578_00', 'scene0580_00', 'scene0583_00', 'scene0591_00',
    'scene0595_00', 'scene0598_00', 'scene0599_00', 'scene0606_00',
    'scene0607_00', 'scene0608_00', 'scene0616_00', 'scene0618_00',
    'scene0621_00', 'scene0633_00', 'scene0643_00', 'scene0644_00',
    'scene0645_00', 'scene0647_00', 'scene0651_00', 'scene0652_00',
    'scene0653_00', 'scene0655_00', 'scene0658_00', 'scene0660_00',
    'scene0663_00', 'scene0664_00', 'scene0665_00', 'scene0670_00',
    'scene0671_00', 'scene0678_00', 'scene0684_00', 'scene0686_00',
    'scene0689_00', 'scene0690_00', 'scene0693_00', 'scene0695_00',
    'scene0696_00', 'scene0697_00', 'scene0699_00', 'scene0700_00',
    'scene0702_00', 'scene0704_00',
]


def _resolve_existing_path(primary: str, *fallbacks: str) -> str:
    """Return the first existing path from primary + fallbacks, else primary."""
    for path in (primary, *fallbacks):
        if path and os.path.exists(path):
            return path
    return primary

class MergedDataset(torch.utils.data.Dataset):
    """
    Unified dataset that loads both 3D-FUTURE and ScanNet data.
    Each sample's data_config includes a 'dataset' field ('future3d' or 'scannet')
    to determine which processing pipeline to use.
    """
    
    def __init__(
        self,
        configs: DictConfig,
        training: bool = True,
        stage: int = 1,
        train_vae: bool = False,
        subset: bool = False,
        positioned: bool = True,
        use_latent: bool = False,
        normalize_scene: bool = True,
        use_future3d: bool = True,
        use_scannet: bool = True,
        use_coco: bool = False,
        use_pointmap: Optional[bool] = None,
        use_high_pointmap: bool = False,   # whether to use pointmaps_high (DA3-LARGE)
        # Augmentation settings
        use_pointmap_aug: bool = False,
        mesh_aug_prob: float = 0.0,
        aug_progress=None,           # multiprocessing.Value(ctypes.c_float) — shared variable for curriculum
        curriculum_warmup_ratio: float = 0.1,
        model_version: str = "v3_input",  # "v3_input" | "v3_attn_fix" | "v4"
        # V3 init augmentation
        v3_init_aug: bool = False,
        # V4 init augmentation
        v4_aug_translation: bool = False,
        v4_aug_scale: bool = False,
        v4_aug_translation_std: float = 0.05,
        v4_aug_scale_range: tuple = (0.8, 1.2),
    ):
        super().__init__()
        self.configs = configs
        self.training = training
        self.stage = stage
        self.train_vae = train_vae
        self.use_latent = use_latent
        self.normalize_scene = normalize_scene
        self.positioned = positioned
        self.use_high_pointmap = use_high_pointmap

        self.min_num_parts = configs["dataset"]["min_num_parts"]
        self.max_num_parts = configs["dataset"]["max_num_parts"]
        self.val_min_num_parts = configs["val"]["min_num_parts"]
        self.val_max_num_parts = configs["val"]["max_num_parts"]
        self.voxel_resolution = configs["dataset"]["voxel_resolution"]
        # use_pointmap can be overridden via constructor arg (e.g. --no_pointmap CLI flag).
        # When None, falls back to the YAML config value (default: True).
        if use_pointmap is not None:
            self.use_pointmap = use_pointmap
        else:
            self.use_pointmap = configs["dataset"].get("use_pointmap", True)
        self.shuffle_parts = configs["dataset"]["shuffle_parts"]

        self.model_version = model_version
        self.v3_init_aug = v3_init_aug
        self.v4_aug_translation = v4_aug_translation
        self.v4_aug_scale = v4_aug_scale
        self.v4_aug_translation_std = v4_aug_translation_std
        self.v4_aug_scale_range = v4_aug_scale_range

        # Augmentation settings
        self.use_pointmap_aug = use_pointmap_aug
        self.mesh_aug_prob = mesh_aug_prob
        self.aug_progress = aug_progress          # multiprocessing.Value or None
        self.curriculum_warmup_ratio = curriculum_warmup_ratio

        self.image_size = (518, 518)

        # Load preprocessor
        config = OmegaConf.load(os.path.join("./checkpoints/hf/pipeline.yaml"))["ss_preprocessor"]
        ss_preprocessor = instantiate(config)
        self.preprocessor = ss_preprocessor

        # Initialize data_configs list
        self.data_configs = []

        # ==================== Load 3D-FUTURE ====================
        if use_future3d:
            self._load_future3d_configs(configs, subset)

        # ==================== Load ScanNet ====================
        if use_scannet:
            self._load_scannet_configs(configs, subset)

        # ==================== Load COCO ====================
        if use_coco:
            self._load_coco_configs(configs, subset)

        # Shuffle all configs together
        if self.shuffle_parts:
            random.shuffle(self.data_configs)

        print(f"Loaded {len(self.data_configs)} frames total:")
        future_count = sum(1 for c in self.data_configs if c.get('dataset') == 'future3d')
        scannet_count = sum(1 for c in self.data_configs if c.get('dataset') == 'scannet')
        coco_count = sum(1 for c in self.data_configs if c.get('dataset') == 'coco')
        print(f"  - 3D-FUTURE: {future_count}")
        print(f"  - ScanNet: {scannet_count}")
        print(f"  - COCO: {coco_count}")

    def _load_future3d_configs(self, configs: DictConfig, subset: bool):
        """Load 3D-FUTURE dataset configurations."""
        print("Loading 3D-FUTURE dataset...")
        
        set_name = "train" if self.training else "test"
        features_root = _resolve_existing_path(
            configs["dataset"].get("data_root", "data/3D-FUTURE"),
        )
        
        # Load data configs
        data_configs = np.load(f"./{set_name}_3dfuture.npy", allow_pickle=True).tolist()
        
        # Load aligned transforms if available
        aligned_transform_pc = {}
        aligned_path = f"./outputs/aligned_transforms_{set_name}.npz"
        if os.path.exists(aligned_path):
            for info in np.load(aligned_path, allow_pickle=True)['infos']:
                aligned_transform_pc[info['uid']] = info
            self.aligned = True
        else:
            self.aligned = False
            print("  ⚠️  Aligned transforms not found, using raw coordinates")
        
        if subset:
            data_configs = data_configs[:300]
        
        # Process each config
        for data in data_configs:
            if len(data.get('object_id', [])) < 1:
                continue
            
            num_parts = data['num_parts']
            
            # Filter by num_parts
            if self.training:
                if num_parts < 3:
                    continue
            
            uid = os.path.basename(data['image_path']).split(".")[0]
            
            # Build paths
            data['image_path'] = os.path.join(features_root, "3D-FUTURE-scene", set_name, "image", f"{uid}.jpg")
            data['mesh_paths'] = [
                os.path.join(features_root, "3D-FUTURE-model", muid, "normalized_model.obj") 
                for muid in data['object_id']
            ]
            data['point_paths'] = [
                os.path.join(features_root, "3D-FUTURE-model", muid, "points.npy") 
                for muid in data['object_id']
            ]
            _pm_dir_test = os.path.join(features_root, "moge_pointmap_test")
            _pm_dir      = os.path.join(features_root, "moge_pointmap")
            data['point_path'] = os.path.join(
                _pm_dir if (set_name == "train" or not os.path.isdir(_pm_dir_test)) else _pm_dir_test,
                f"{uid}.npy"
            )
            data['mask_paths'] = sorted(glob.glob(
                os.path.join(
                    features_root, 
                    "masked_images" if set_name == "train" else "masked_images_test", 
                    uid, 
                    "*_mask.png"
                )
            ))
            data['dataset'] = 'future3d'
            data['uid'] = uid
            
            if self.training and self.aligned and uid in aligned_transform_pc:
                data['align_transform'] = aligned_transform_pc[uid]
            
            self.data_configs.append(data)
        
        print(f"  Loaded {len([c for c in self.data_configs if c.get('dataset') == 'future3d'])} 3D-FUTURE samples")

    def _load_scannet_configs(self, configs: DictConfig, subset: bool):
        """Load ScanNet dataset configurations."""
        print("Loading ScanNet dataset...")
        
        features_root = _resolve_existing_path(
            configs["dataset"].get("scannet_data_root", "data/scannet"),
        )

        # Load dataset.npy file
        dataset_path = _resolve_existing_path(
            os.path.join(features_root, "scannet_dataset.npy"),
        )
        
        
        # Camera convention transform for ScanNet
        self.convention_transform = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.convention_rot = self.convention_transform[:3, :3]
        
        all_frames = []
        for frame_info in np.load(dataset_path, allow_pickle=True).tolist():
            num_parts = frame_info.get("num_parts", 0)
            if self.training:
                if frame_info['scene_id'] in TEST_SET_ID:
                    continue
                if num_parts < self.min_num_parts or num_parts > self.max_num_parts:
                    continue
            else:
                if not frame_info['scene_id'] in TEST_SET_ID:
                    continue
                
            all_frames.append(frame_info)
            
        print(f"  Loaded {len(all_frames)} ScanNet frames from dataset.npy, processing...")

        # Process each frame
        for frame_info in all_frames:
            num_parts = frame_info.get("num_parts", 0)
            # Filter by num_parts
            
            f2y_rot = torch.tensor(frame_info["f2y_rot"]).float()
            xz2f_rot = torch.linalg.inv(f2y_rot)

            data_entry = {
                "scene_id": frame_info["scene_id"],
                "frame_id": frame_info["frame_id"],
                "image_path": frame_info["image_path"],
                "mask_path": frame_info.get("mask_path"),
                "pointmap_path": frame_info["pointmap_path"].replace(
                    "/pointmaps/", "/pointmaps_high/"
                ) if self.use_high_pointmap else frame_info["pointmap_path"],
                "object_infos": frame_info.get("object_infos", []),
                "num_parts": num_parts,
                "uid": f"{frame_info['scene_id']}_{frame_info['frame_id']}",
                "dataset": "scannet",
                "features_root": features_root,
                "f2y_rot": f2y_rot,
                "xz2f_rot": xz2f_rot
            }
            self.data_configs.append(data_entry)
        
        if subset:
            # Keep only subset of ScanNet samples
            scannet_configs = [c for c in self.data_configs if c.get('dataset') == 'scannet']
            other_configs = [c for c in self.data_configs if c.get('dataset') != 'scannet']
            self.data_configs = other_configs + scannet_configs[:50]
        
        pm_tag = "pointmaps_high (DA3-LARGE)" if self.use_high_pointmap else "pointmaps (DA3-SMALL)"
        n_scannet = len([c for c in self.data_configs if c.get('dataset') == 'scannet'])
        print(f"  Loaded {n_scannet} ScanNet samples  [pointmap: {pm_tag}]")

    def _load_coco_configs(self, configs: DictConfig, subset: bool):
        """Load filtered COCO indoor subset configurations."""
        print("Loading COCO dataset...")

        subset_root = configs["dataset"].get(
            "coco_subset_root",
            "data/coco_indoor",
        )
        subset_root = _resolve_existing_path(
            subset_root,
        )
        split_name = "train2017" if self.training else "val2017"
        split_root = os.path.join(subset_root, split_name)

        if not os.path.isdir(split_root):
            print(f"  WARNING: COCO split root not found: {split_root}")
            return

        images_dir = os.path.join(split_root, "images")
        pointmaps_dir = os.path.join(split_root, "pointmaps")
        masks_dir = os.path.join(split_root, "masks")
        objects_dir = os.path.join(split_root, "objects")

        if not all(os.path.isdir(p) for p in [images_dir, pointmaps_dir, masks_dir, objects_dir]):
            print(f"  WARNING: COCO split is incomplete under {split_root}")
            return

        if self.training:
            min_parts, max_parts = self.min_num_parts, self.max_num_parts
        else:
            min_parts, max_parts = self.val_min_num_parts, self.val_max_num_parts

        image_stems = {
            os.path.splitext(name)[0]
            for name in os.listdir(images_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        pointmap_stems = {
            os.path.splitext(name)[0]
            for name in os.listdir(pointmaps_dir)
            if name.endswith(".npy")
        }
        mask_stems = {
            name for name in os.listdir(masks_dir)
            if os.path.isdir(os.path.join(masks_dir, name))
        }
        object_stems = {
            name for name in os.listdir(objects_dir)
            if os.path.isdir(os.path.join(objects_dir, name))
        }

        valid_stems = sorted(image_stems & pointmap_stems & mask_stems & object_stems)

        coco_configs = []
        for stem in valid_stems:
            mask_dir = os.path.join(masks_dir, stem)
            object_dir = os.path.join(objects_dir, stem)

            mask_ids = {
                os.path.splitext(name)[0]
                for name in os.listdir(mask_dir)
                if name.endswith(".png")
            }
            mesh_ids = {
                os.path.splitext(name)[0]
                for name in os.listdir(object_dir)
                if name.endswith(".obj")
            }
            pose_ids = {
                os.path.splitext(name)[0]
                for name in os.listdir(object_dir)
                if name.endswith(".npz")
            }

            ann_ids = sorted(mask_ids & mesh_ids & pose_ids)
            num_parts = len(ann_ids)
            if num_parts < min_parts or num_parts > max_parts:
                continue

            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(images_dir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                continue

            coco_configs.append({
                "image_id": int(stem),
                "image_path": image_path,
                "pointmap_path": os.path.join(pointmaps_dir, f"{stem}.npy"),
                "mask_paths": [os.path.join(mask_dir, f"{ann_id}.png") for ann_id in ann_ids],
                "mesh_paths": [os.path.join(object_dir, f"{ann_id}.obj") for ann_id in ann_ids],
                "pose_paths": [os.path.join(object_dir, f"{ann_id}.npz") for ann_id in ann_ids],
                "num_parts": num_parts,
                "uid": f"coco_{split_name}_{stem}",
                "dataset": "coco",
            })

        if subset:
            coco_configs = coco_configs[:50]

        self.data_configs.extend(coco_configs)
        print(f"  Loaded {len(coco_configs)} COCO samples from {split_root}")

    
    def __len__(self) -> int:
        return len(self.data_configs)

    def _get_aug_factor(self) -> float:
        if self.aug_progress is None:
            return 1.0
        progress = float(self.aug_progress.value)   # 0.0 → 1.0
        warmup = self.curriculum_warmup_ratio
        if progress < warmup:
            return 0.0
        return min(1.0, (progress - warmup) / max(1e-6, 1.0 - warmup))

    def _get_data_by_config_future3d(self, data_config):
        """Process 3D-FUTURE data config."""
        voxels = []
        points = []
        items = []
        
        # Get transformations
        translations = data_config['translation']
        rotation = data_config['rotation']
        scales = data_config['scale']

        xz2f_rot = torch.eye(3)
        xz2f_rot_6d = matrix_to_rotation_6d(xz2f_rot)

        rot_matrix = torch.tensor(rotation)
        t = torch.tensor(translations)
        s = torch.tensor(scales).unsqueeze(1)
        r = matrix_to_rotation_6d(rot_matrix)

        # Apply aligned transform
        if self.aligned:
            aligned_transform = torch.diag(torch.tensor([-1., 1., -1., 1.]))
            transform = torch.zeros((rot_matrix.shape[0], 4, 4))
            transform[:, :3, :3] = rot_matrix
            transform[:, :3, 3] = t
            transform[:, 3, 3] = 1.

            transform_aligned = aligned_transform.unsqueeze(0).repeat(rot_matrix.shape[0], 1, 1)
            transform = torch.matmul(transform_aligned, transform)
            r = matrix_to_rotation_6d(transform[:, :3, :3])
            t = transform[:, :3, 3]

            if self.training and 'align_transform' in data_config:
                align_s = torch.tensor(data_config['align_transform']["s"]).unsqueeze(0)
                align_t = torch.tensor(data_config['align_transform']["t"]).unsqueeze(0)
                t = t * align_s + align_t
                s = s * align_s

        num_parts = torch.tensor([data_config['num_parts']])
        
        # Load image
        image = Image.open(data_config['image_path']).convert("RGB").resize(self.image_size, Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0
        pm_data = np.load(data_config['point_path'], allow_pickle=True).item()
        pointmap_raw = pm_data['pt3d_points']  # (H,W,3) MeshLayout convention
        pointmap = torch.from_numpy(pointmap_raw).permute(2, 0, 1).float()
        pointmap = tf.resize(pointmap.unsqueeze(0), self.image_size, interpolation=tf.InterpolationMode.BILINEAR).squeeze(0)

        # Normalize scene
        if self.normalize_scene:
            _min, _max = pointmap.flatten(1).min(1).values, pointmap.flatten(1).max(1).values
            center = (_min + _max) / 2
            centered = pointmap - center.unsqueeze(1).unsqueeze(1)
            scale = centered.max()
            if scale > 0:
                pointmap = centered / scale
                s = s / scale
                t = (t - center) / scale
            else:
                scale = torch.tensor(1.0)


        # Process each object
        meshes = []
        pm_surface_pts_list = []
        for idx in range(data_config['num_parts']):
            mesh_path = data_config['mesh_paths'][idx]
            mask_path = data_config['mask_paths'][idx]

            if not os.path.exists(mesh_path):
                continue

            # Load mesh and generate voxels
            mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True)

            # Mesh rotation augmentation (applied before voxelization)
            if self.mesh_aug_prob > 0 and self.training:
                skip_prob  = 1.0 - self.mesh_aug_prob 

                mesh, r_aug, s_aug, _ = augment_mesh_and_rotation(
                    mesh, r[idx], s[idx], p=skip_prob
                )
                r[idx] = r_aug
                s[idx] = s_aug

            voxel_tensor, point = mesh_to_voxel_tensor(mesh, resolution=self.voxel_resolution, sample_points=True)

            # Load mask
            if not os.path.exists(mask_path):
                continue

            meshes.append(mesh_path)
            mask = Image.open(mask_path).convert("L").resize(self.image_size, Image.NEAREST)
            mask = torch.from_numpy(np.array(mask))

            mask_bool = mask > 127
            n_px = mask_bool.sum().item()
            N_points = 4096
            if n_px >= 256:
                pm_pts_all = pointmap[:, mask_bool]        # (3, K)
                k = min(N_points, pm_pts_all.shape[1])
                _pm_idx = torch.randperm(pm_pts_all.shape[1])[:k]
                pm_samp = pm_pts_all[:, _pm_idx].T                # (k, 3)
                if k < N_points:
                    pm_samp = torch.cat([pm_samp, torch.zeros(N_points - k, 3)], dim=0)
            else:
                pm_samp = torch.zeros(N_points, 3)
            pm_surface_pts_list.append(pm_samp)

            # Extract pm_surface_pts: visible surface points in the mask region from the pre-aug pointmap
            rgba = torch.cat([image, mask.unsqueeze(0) / 255.0], 0)
            
            # Preprocess
            if self.use_latent:
                item = {"image": rgba, "mask": mask, "pointmap": pointmap}
            elif self.use_pointmap:
                item = preprocess_image(rgba, self.preprocessor, pointmap=pointmap)
            else:
                item = preprocess_image(rgba, self.preprocessor, pointmap=None)

            # item["pointmap"] = item['rgb_pointmap'].clone()
            # item["pointmap"][:,~mask.bool()] = torch.nan

            items.append(item)
            points.append(point)
            voxels.append(voxel_tensor)

        # Stack all items
        items = listdict_to_dictlist_safe(items)
        for key in items:
            items[key] = torch.stack(items[key])
        
        items['translation'] = t
        items['6drotation_normalized'] = r
        items["pred_translation"] = t
        items["pred_6drotation_normalized"] = r
        items['scale'] = s.clamp(min=1e-6)
        items['xz2f_rot'] = torch.stack([xz2f_rot_6d]*len(translations))
        items['num_parts'] = num_parts
        items['voxels'] = torch.stack(voxels)
        items['uid'] = data_config['uid']
        items['mesh_points'] = torch.stack(points)
        items['meshes'] = meshes
        items['dataset'] = 'future3d'
        items["trainable"] = torch.ones(data_config['num_parts'], dtype=torch.float32)
        items["trainable_precise"] = torch.ones(data_config['num_parts'], dtype=torch.float32)
        items['scene_pointmap'] = pointmap
        items['pm_surface_pts'] = torch.stack(pm_surface_pts_list)
        items['is_coco'] = torch.tensor([False] * data_config['num_parts'])  # always False for future3d
        items['is_scannet'] = torch.tensor([False] * data_config['num_parts'])

        return items

    def _load_pointmap_scannet(self, pointmap_path: str):
        """Load ScanNet pointmap with convention transform.
        Returns: (cam_points_tensor (3,H,W), fov (float), fx_target (float|None), fy_target (float|None))
        """
        if not os.path.exists(pointmap_path):
            pointmap_path = pointmap_path.replace("/pointmaps_high/", "/pointmaps/")

        pointmap_data = np.load(pointmap_path, allow_pickle=True).item()
        cam_points = pointmap_data.get("cam_points")
        if cam_points.ndim == 3:  # H,W,3
            cam_points = np.dot(cam_points, self.convention_rot.T)
            cam_points_t = torch.from_numpy(cam_points).permute(2, 0, 1).float()
        else:
            cam_points = np.dot(cam_points, self.convention_rot.T)
            cam_points_t = torch.from_numpy(cam_points).float()

        cam_points_t = tf.resize(
            cam_points_t.unsqueeze(0), self.image_size, interpolation=tf.InterpolationMode.NEAREST
        ).squeeze(0)
        return cam_points_t

    def _load_instance_mask_scannet(self, mask_path: Optional[str]) -> Optional[np.ndarray]:
        """Load ScanNet instance mask."""
        mask_img = Image.open(mask_path)
        mask_indices = np.array(mask_img, dtype=np.int32)
        mask_indices_resized = Image.fromarray(mask_indices, mode="I")
        mask_indices_resized = mask_indices_resized.resize(self.image_size, Image.NEAREST)
        return np.array(mask_indices_resized)

    def _get_data_by_config_scannet(self, data_config):
        """Process ScanNet data config."""
        voxels = []
        points = []
        items = []
        translations = []
        rot6d = []
        scales = []
        trainable_flags = []
        trainable_flags2 = []
        mesh_paths = []

        convention_rot_t = torch.tensor(self.convention_rot, dtype=torch.float32)
        
        num_parts = data_config["num_parts"]
        features_root = data_config["features_root"]
        f2y_rot = torch.tensor(data_config["f2y_rot"])
        f2y_rot = convention_rot_t @ f2y_rot @ convention_rot_t 
        xz2f_rot = f2y_rot.T
        xz2f_rot_6d = matrix_to_rotation_6d(xz2f_rot)

        if num_parts > self.max_num_parts:
            rand_idx = random.sample(range(num_parts), self.max_num_parts)
            object_infos = [data_config["object_infos"][i] for i in rand_idx]
        else:
            object_infos = data_config["object_infos"]

        # Load image
        image = Image.open(data_config["image_path"]).convert("RGB").resize(self.image_size, Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0

        # Load pointmap with convention transform
        cam_pointmap = self._load_pointmap_scannet(data_config["pointmap_path"])
            

        # Normalize pointmap
        if self.normalize_scene:
            _min = cam_pointmap.flatten(1).min(1).values
            _max = cam_pointmap.flatten(1).max(1).values
            global_center = (_min + _max) / 2
            centered = cam_pointmap - global_center.unsqueeze(1).unsqueeze(1)
            global_scale = centered.max()
            if global_scale > 0:
                cam_pointmap = centered / global_scale
            else:
                global_scale = torch.tensor(1.0)
        else:
            global_center = torch.zeros(3)
            global_scale = torch.tensor(1.0)


        # Load instance mask
        mask_indices = self._load_instance_mask_scannet(data_config.get("mask_path"))
        pm_surface_pts_list = []
        # Process each object
        for obj_info in object_infos:
            obj_id = obj_info["object_id"]

            # Check mesh exists
            mesh_simple_path = os.path.join(
                features_root,
                "mesh_simple",
                data_config["scene_id"],
                f"{obj_id}_simple.obj",
            )
            
            # Extract pose and apply convention transform
            rotation_quat = torch.tensor(obj_info["rotation"], dtype=torch.float32)
            translation = torch.tensor(obj_info["translation"], dtype=torch.float32)
            scale_val = torch.tensor(obj_info["scale"], dtype=torch.float32).unsqueeze(0)

            rotation_mat = quaternion_to_matrix(rotation_quat)
            
            rotation_mat = convention_rot_t @ rotation_mat
            translation  = convention_rot_t  @ translation

            # Normalize with global scale/center
            if self.normalize_scene:
                scale_val = scale_val / global_scale
                translation = (translation - global_center) / global_scale

            rotation_6d = matrix_to_rotation_6d(rotation_mat)
            is_trainable = bool(obj_info["trainable"])
            is_trainable_precise = bool(obj_info["trainable_precise"])

            # Load mesh and generate voxels
            mesh = trimesh.load(mesh_simple_path, force='mesh', skip_materials=True)
            voxel_tensor, mesh_points = mesh_to_voxel_tensor(
                mesh, resolution=self.voxel_resolution, sample_points=True
            )

            # Create object mask
            object_mask = (mask_indices == obj_id).astype(np.uint8) * 255
            mask_tensor = torch.from_numpy(object_mask)

            mask_bool = mask_tensor > 127
            n_px = mask_bool.sum().item()
            N_points = 4096
            if n_px >= 256:
                pm_pts_all = cam_pointmap[:, mask_bool]    # (3, K)
                k = min(N_points, pm_pts_all.shape[1])
                _pm_idx = torch.randperm(pm_pts_all.shape[1])[:k]
                pm_samp = pm_pts_all[:, _pm_idx].T                # (k, 3)
                if k < N_points:
                    pm_samp = torch.cat([pm_samp, torch.zeros(N_points - k, 3)], dim=0)
            else:
                pm_samp = torch.zeros(N_points, 3)
            pm_surface_pts_list.append(pm_samp)

            rgba = torch.cat([image, mask_tensor.unsqueeze(0) / 255.0], 0)

            # Preprocess
            if self.use_latent:
                item = {"image": rgba, "mask": mask_tensor, "pointmap": cam_pointmap}
            elif self.use_pointmap:
                item = preprocess_image(rgba, self.preprocessor, pointmap=cam_pointmap)
            else:
                item = preprocess_image(rgba, self.preprocessor, pointmap=None)

            # item["pointmap"] = item['rgb_pointmap'].clone()
            # item["pointmap"][:,~mask_tensor.bool()] = torch.nan

            items.append(item)
            translations.append(translation)
            rot6d.append(rotation_6d)
            scales.append(scale_val)
            trainable_flags.append(is_trainable)
            trainable_flags2.append(is_trainable_precise)
            voxels.append(voxel_tensor)
            points.append(mesh_points)
            mesh_paths.append(mesh_simple_path)

        # Stack all items
        items = listdict_to_dictlist_safe(items)
        for key in items:
            items[key] = torch.stack(items[key])

        items["translation"] = torch.stack(translations)
        items["6drotation_normalized"] = torch.stack(rot6d)
        items["pred_translation"] = (f2y_rot @ torch.stack(translations).T).T # N, 3
        items["pred_6drotation_normalized"] = matrix_to_rotation_6d(f2y_rot @ rotation_6d_to_matrix(torch.stack(rot6d))) # N, 6
        items["scale"] = torch.stack(scales).clamp(min=1e-6)
        items['xz2f_rot'] = torch.stack([xz2f_rot_6d]*len(translations))
        items["num_parts"] = torch.tensor([len(translations)], dtype=torch.long)
        items["voxels"] = torch.stack(voxels)
        items["mesh_points"] = torch.stack(points)
        items["trainable"] = torch.tensor(trainable_flags, dtype=torch.float32)
        items["trainable_precise"] = torch.tensor(trainable_flags2, dtype=torch.float32)
        items["uid"] = data_config["uid"]
        items["meshes"] = mesh_paths
        items['dataset'] = 'scannet'
        items['scene_pointmap'] = cam_pointmap
        items['pm_surface_pts'] = torch.stack(pm_surface_pts_list)
        items['is_coco'] = torch.tensor([False] * data_config['num_parts']) 
        items['is_scannet'] = torch.tensor([True] * data_config['num_parts'])

        return items

    def _get_data_by_config_coco(self, data_config):
        """Process filtered COCO indoor subset sample."""
        voxels = []
        points = []
        items = []
        mesh_paths = []

        num_parts = len(data_config["mesh_paths"])
        image = Image.open(data_config["image_path"]).convert("RGB").resize(self.image_size, Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1) / 255.0


        pm_data = np.load(data_config["pointmap_path"], allow_pickle=True).item()
        pointmap_raw = pm_data["pt3d_points"]
        pointmap_raw = np.asarray(pointmap_raw, dtype=np.float32)

        pointmap = torch.from_numpy(pointmap_raw).permute(2, 0, 1).float()
        pointmap = tf.resize(
            pointmap.unsqueeze(0), self.image_size, interpolation=tf.InterpolationMode.BILINEAR
        ).squeeze(0)

        if self.normalize_scene:
            _min = pointmap.flatten(1).min(1).values
            _max = pointmap.flatten(1).max(1).values
            center = (_min + _max) / 2
            centered = pointmap - center.unsqueeze(1).unsqueeze(1)
            scene_scale = centered.max()
            if scene_scale > 0:
                pointmap = centered / scene_scale
            else:
                center = torch.zeros(3)
                scene_scale = torch.tensor(1.0)
        else:
            center = torch.zeros(3)
            scene_scale = torch.tensor(1.0)

        pm_surface_pts_list = []
        for mesh_path, mask_path in zip(
            data_config["mesh_paths"], data_config["mask_paths"]
        ):
            mesh = trimesh.load(mesh_path, force="mesh", skip_materials=True)
            voxel_tensor, mesh_points = mesh_to_voxel_tensor(
                mesh, resolution=self.voxel_resolution, sample_points=True
            )

            mask = Image.open(mask_path).convert("L").resize(self.image_size, Image.NEAREST)
            mask = torch.from_numpy(np.array(mask))
        
            mask_bool = mask > 127
            n_px = mask_bool.sum().item()
            N_points = 4096
            pm_pts_all = pointmap[:, mask_bool]
            k = min(N_points, pm_pts_all.shape[1])
            _pm_idx = torch.randperm(pm_pts_all.shape[1])[:k]
            pm_samp = pm_pts_all[:, _pm_idx].T
            if k < N_points:
                pm_samp = torch.cat([pm_samp, torch.zeros(N_points - k, 3)], dim=0)
            
            pm_surface_pts_list.append(pm_samp)

            rgba = torch.cat([image, mask.unsqueeze(0) / 255.0], 0)

            item = preprocess_image(rgba, self.preprocessor, pointmap=pointmap)
            # item["pointmap"] = item["rgb_pointmap"].clone()
            # item["pointmap"][:, ~mask.bool()] = torch.nan

            items.append(item)
            voxels.append(voxel_tensor)
            points.append(mesh_points)
            mesh_paths.append(mesh_path)

        items = listdict_to_dictlist_safe(items)
        for key in items:
            items[key] = torch.stack(items[key])

        items["translation"] = torch.zeros((num_parts,3))
        items["6drotation_normalized"] = torch.zeros((num_parts,6))
        # COCO uses the same canonical frame for GT/pred supervision as 3D-FUTURE.
        items["pred_translation"] = torch.zeros((num_parts,3))
        items["pred_6drotation_normalized"] = torch.zeros((num_parts,6))
        items["scale"] = torch.zeros((num_parts,1))
        items["xz2f_rot"] = torch.zeros((num_parts,6))
        items["num_parts"] = torch.tensor([num_parts], dtype=torch.long)
        items["voxels"] = torch.stack(voxels)
        items["mesh_points"] = torch.stack(points)
        items["trainable"] = torch.zeros(num_parts, dtype=torch.float32)
        items["trainable_precise"] = torch.zeros(num_parts, dtype=torch.float32)
        items["uid"] = data_config["uid"]
        items["meshes"] = mesh_paths
        items["dataset"] = "coco"
        items["scene_pointmap"] = pointmap
        items['pm_surface_pts'] = torch.stack(pm_surface_pts_list)
        items['is_coco'] = torch.tensor([True] * data_config['num_parts']) 
        items['is_scannet'] = torch.tensor([False] * data_config['num_parts'])

        return items

    def _get_data_by_config(self, data_config):
        """Route to appropriate processing function based on dataset type."""
        dataset_type = data_config.get('dataset', 'future3d')
        
        if dataset_type == 'future3d':
            return self._get_data_by_config_future3d(data_config)
        elif dataset_type == 'scannet':
            return self._get_data_by_config_scannet(data_config)
        elif dataset_type == 'coco':
            return self._get_data_by_config_coco(data_config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        max_retries = 10
        
        for _ in range(max_retries):
            data = self._get_data_by_config(data_config)
            if data is not None:
                return data
            # Try next sample
            idx = (idx + 1) % len(self.data_configs)
            data_config = self.data_configs[idx]
        
        raise RuntimeError(f"Failed to load any valid sample after {max_retries} retries")


class BatchedMergedDataset(MergedDataset):
    """
    Batched version of MergedDataset.
    Groups samples from both datasets into batches of fixed total num_parts.
    """

    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
        stage: int = 1,
        use_latent: bool = False,
        use_future3d: bool = True,
        use_scannet: bool = True,
        use_coco: bool = False,
        use_pointmap: Optional[bool] = None,
        use_high_pointmap: bool = False,
        # Augmentation settings
        use_pointmap_aug: bool = False,
        mesh_aug_prob: float = 0.0,
        aug_progress=None,
        curriculum_warmup_ratio: float = 0.1,
        model_version: str = "v3_input",
        v3_init_aug: bool = False,
        v4_aug_translation: bool = False,
        v4_aug_scale: bool = False,
        v4_aug_translation_std: float = 0.05,
        v4_aug_scale_range: tuple = (0.8, 1.2),
    ):
        assert training
        assert batch_size > 1

        super().__init__(
            configs,
            training,
            stage,
            use_latent=use_latent,
            use_future3d=use_future3d,
            use_scannet=use_scannet,
            use_coco=use_coco,
            use_pointmap=use_pointmap,
            use_high_pointmap=use_high_pointmap,
            use_pointmap_aug=use_pointmap_aug,
            mesh_aug_prob=mesh_aug_prob,
            aug_progress=aug_progress,
            curriculum_warmup_ratio=curriculum_warmup_ratio,
            model_version=model_version,
            v3_init_aug=v3_init_aug,
            v4_aug_translation=v4_aug_translation,
            v4_aug_scale=v4_aug_scale,
            v4_aug_translation_std=v4_aug_translation_std,
            v4_aug_scale_range=v4_aug_scale_range,
        )
        
        self.batch_size = batch_size
        self.is_main_process = is_main_process

        # Filter configs by batch_size
        if batch_size < self.max_num_parts:
            self.data_configs = [c for c in self.data_configs if c["num_parts"] <= batch_size]

        if shuffle:
            random.shuffle(self.data_configs)

        # Separate single objects and multi-part scenes
        self.object_configs = [c for c in self.data_configs if c["num_parts"] == 1]
        self.parts_configs = [c for c in self.data_configs if c["num_parts"] > 1]

        # Keep only a ratio of single objects
        self.object_ratio = configs["dataset"].get("object_ratio", 0.2)
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        print(f"Single object configs: {len(self.object_configs)}")

        # Combine and shuffle
        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        # Create batched configs
        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)

        print(f"Created {len(self.data_configs)} batches (batch_size={batch_size})")

    def _get_batched_configs(self, data_configs, batch_size):
        """Group configs into batches by total num_parts."""
        batched_data_configs = []
        
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop()
                num_parts = config["num_parts"]
                
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config)
            
            data_configs = data_configs + unchosen_configs
            
            if temp_num_parts == batch_size:
                # Successfully formed a batch
                if len(temp_batch) < batch_size:
                    # Pad batch if needed
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
        
        progress_bar.close()
        return batched_data_configs

    def __getitem__(self, idx: int):
        """Get batched item."""
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            # Return empty dict for padding
            return {}
        data = self._get_data_by_config(data_config)
        return data

    def collate_fn(self, batch):
        """Collate batch items."""
        batch = [data for data in batch if len(data) > 0]
        
        if len(batch) == 0:
            return {}
        
        outputs = {}
        for key in batch[0]:
            if key == "uid":
                outputs[key] = [data[key] for data in batch]
                continue
            if key == "meshes":
                outputs[key] = []
                for data in batch:
                    outputs[key].extend(data[key])
                continue
            if key == "dataset":
                outputs[key] = [data[key] for data in batch]
                continue

            outputs[key] = torch.cat([data[key] for data in batch], dim=0)

            if key == "6drotation_normalized":
                # 6drot --> rotmat --> quat --> rotmat --> 6drot (keep only the normalized rotation)
                rot6d = outputs[key]
                rotmat = rotation_6d_to_matrix(rot6d)
                quat = matrix_to_quaternion(rotmat)
                rotmat = quaternion_to_matrix(quat)
                rot6d_norm = matrix_to_rotation_6d(rotmat)
                outputs[key] = rot6d_norm
                
            
        assert outputs["image"].shape[0] == outputs["num_parts"].sum() == self.batch_size
        return outputs
