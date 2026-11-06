import os
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils import (
    get_mirror_params,
    apply_mirror,
    get_transpose_params,
    apply_transpose,
    get_jitter_params,
    apply_jitter,
    add_walls,
    apply_mirror_feats,
    apply_transpose_feats,
    apply_jitter_feats,
)
CATEGORY_TO_INDEX = {
    # seen / training categories (0–17)
    '02747177': 0,
    '02871439': 1,
    '02880940': 2,
    '02933112': 3,
    '03001627': 4,
    '03085013': 5,
    '03207941': 6,
    '03211117': 7,
    '03325088': 8,
    '03337140': 9,
    '03467517': 10,
    '03761084': 11,
    '03928116': 12,
    '03991062': 13,
    '04256520': 14,
    '04330267': 15,
    '04379243': 16,
    '04554684': 17,

    # novel / test-only categories (18–25)
    '02773838': 18,
    '02801938': 19,
    '02808440': 20,
    '02818832': 21,
    '02828884': 22,
    '03636649': 23,
    '03642806': 24,
    '04004475': 25,
}

# Human-readable ShapeNet category names
shapenet_category_pair = {
    'table' : '04379243',
    'jar' : '03593526',
    'skateboard' : '04225987',
    'car' : '02958343',
    'bottle' : '02876657',
    'tower' : '04460130',
    'chair' : '03001627',
    'bookshelf' : '02871439',
    'camera' : '02942699',
    'airplane' : '02691156',
    'laptop' : '03642806',
    'basket' : '02801938',
    'sofa' : '04256520',
    'knife' : '03624134',
    'can' : '02946921',
    'rifle' : '04090263',
    'train' : '04468005',
    'pillow' : '03938244',
    'lamp' : '03636649',
    'trash bin' : '02747177',
    'mailbox' : '03710193',
    'watercraft' : '04530566',
    'motorbike' : '03790512',
    'dishwasher' : '03207941',
    'bench' : '02828884',
    'pistol' : '03948459',
    'rocket' : '04099429',
    'loudspeaker' : '03691459',
    'file cabinet' : '03337140',
    'bag' : '02773838',
    'cabinet' : '02933112',
    'bed' : '02818832',
    'birdhouse' : '02843684',
    'display' : '03211117',
    'piano' : '03928116',
    'earphone' : '03261776',
    'telephone' : '04401088',
    'stove' : '04330267',
    'microphone' : '03759954',
    'bus' : '02924116',
    'mug' : '03797390',
    'remote' : '04074963',
    'bathtub' : '02808440',
    'bowl' : '02880940',
    'keyboard' : '03085013',
    'guitar' : '03467517',
    'washer' : '04554684',
    'bicycle' : '02834778',
    'faucet' : '03325088',
    'printer' : '04004475',
    'cap' : '02954340',
    'clock' : '03046257',
    'helmet' : '03513137',
    'flowerpot' : '03991062',
    'microwaves' : '03761084'
}


class DinoTSDFDataset(Dataset):
    """
    Dataset for TSDF + DINOv3 features.

    Supports optional `list_file`:
      - Each line:  <category>/<model_id>
      Example: 02871439/777785bf709e10ccffba1ef7a9b50ff2
    """

    def __init__(
        self,
        cfg,
        train,
        list_file,
    ):
        super().__init__()

        self.root_dir = cfg.data_root
        self._trunc = cfg.truncation
        self._res =  cfg.resolution
        self._size =  cfg.tsdf_size
        self.input_ids =  cfg.input_ids
        self.grid_size =  cfg.grid_size
        self.dino_dim =  cfg.dino_dim
        self.normalize_dino =  cfg.normalize_dino
        self.use_tudf =  cfg.use_tudf
        self.use_gt_input =  cfg.use_gt_input
        self.use_gt_teacher =  cfg.use_gt_teacher
        self.augmentation = cfg.augmentation
        self.dataset_tag = cfg.dataset_tag

        # augmentation flags
        self.train = train
        self._walls_aug =  cfg.walls_aug
        self._jitter_range =  cfg.jitter_range
        self.filter_input_tsdf = getattr(cfg, "filter_input_tsdf", False)
        self.filter_missing_to = getattr(cfg, "filter_missing_to", "zero")        # "zero" or "keep"
        self.filter_hallucinated_to = getattr(cfg, "filter_hallucinated_to", "trunc")  # "trunc" or "keep"
        self.mask_eps = getattr(cfg, "mask_eps", 1e-10)
        self.return_fpfn_masks = getattr(cfg, "return_fpfn_masks", False)
        self.dino_model_name = getattr(cfg, "dino_model_name", "dinov3_vits16")
        # ScanNet-like augmentation flag
                # ScanNet-like augmentation flag
        self.scannet_like_augmentation = getattr(cfg, "scannet_like_augmentation", False)

        # --------------------------------------------------
        # ScanNet-like augmentation modes:
        #   "value"     -> old noise / dropout corruption
        #   "bbox_cube" -> NEW: insert +/- trunc cubes inside GT bbox
        # --------------------------------------------------
        self.scannet_like_aug_mode = getattr(cfg, "scannet_like_aug_mode", "bbox_cube")

        # ----- old value corruption params -----
        self.tsdf_noise_std = float(getattr(cfg, "tsdf_noise_std", 0.0))
        self.tsdf_dropout_p = float(getattr(cfg, "tsdf_dropout_p", 0.0))
        self.tsdf_dropout_to = getattr(cfg, "tsdf_dropout_to", "trunc")
        self.tsdf_value_scale = float(getattr(cfg, "tsdf_value_scale", 0.0))

        # ----- new bbox cube corruption params -----
        self.cube_aug_prob = float(getattr(cfg, "cube_aug_prob", 1.0))
        self.cube_aug_num_cubes_min = int(getattr(cfg, "cube_aug_num_cubes_min", 1))
        self.cube_aug_num_cubes_max = int(getattr(cfg, "cube_aug_num_cubes_max", 1))

        # cube sizes are in voxel units
        self.cube_aug_size_min = int(getattr(cfg, "cube_aug_size_min", 4))
        self.cube_aug_size_max = int(getattr(cfg, "cube_aug_size_max", 10))

        # "random" => randomly choose +trunc or -trunc for each cube
        # "pos"    => always +trunc
        # "neg"    => always -trunc
        self.cube_aug_fill_mode = getattr(cfg, "cube_aug_fill_mode", "random")

        # optional padding from bbox boundary so cube is not forced to touch walls
        self.cube_aug_bbox_margin = int(getattr(cfg, "cube_aug_bbox_margin", 0))

        if self.scannet_like_augmentation:
            print(
                f"[DinoTSDFDataset] ScanNet-like augmentation ENABLED "
                f"(mode={self.scannet_like_aug_mode})"
            )


        # -------------------------
        # Case A: load only shapes from txt file
        # -------------------------
        print(f"[DinoTSDFDataset] Loading shape list from {list_file}")
        self.shape_dirs = self._load_shape_dirs_from_list(list_file)
        

        # Build (shape_dir, input_id) samples
        self.samples: List[Tuple[str, int]] = []
        #print(self.shape_dirs)
        if not self.use_gt_input:
            for shape_dir in self.shape_dirs:
                for k in self.input_ids:
                    input_path = os.path.join(shape_dir, f"input_{k}.npz")
                    if self.use_gt_teacher:
                        teacher_path = os.path.join(shape_dir, f"{self.dino_model_name}_gt_{self.grid_size}_{self.dataset_tag}.npz")
                    else:
                        teacher_path = os.path.join(shape_dir, f"{self.dino_model_name}_{k}_{self.grid_size}_{self.dataset_tag}.npz")
                    if os.path.exists(input_path) and os.path.exists(teacher_path):
                        self.samples.append((shape_dir, k))
        else:
            for shape_dir in self.shape_dirs:
                input_path = os.path.join(shape_dir, f"gt.npz")
                teacher_path = os.path.join(shape_dir, f"{self.dino_model_name}_gt_{self.grid_size}_{self.dataset_tag}.npz")

                if os.path.exists(input_path) and os.path.exists(teacher_path):
                    self.samples.append((shape_dir, 0))

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found. Check list_file or directory paths.")

        print(f"[DinoTSDFDataset] Final sample count: {len(self.samples)}")

    # -------------------------------------------------------
    # Load shapes from split file (train/val/test txt)
    # -------------------------------------------------------
    def _filter_input_by_gt(self, input_tsdf: torch.Tensor, gt_tsdf: torch.Tensor):
        """
        input_tsdf, gt_tsdf: [1,X,Y,Z] torch
        returns filtered_input, missing_mask, hallucinated_mask
        """
        eps = self.mask_eps

        # same definition you used in loss
        missing_mask = (gt_tsdf <= eps) & (input_tsdf > eps)
        hallucinated_mask = (gt_tsdf > eps) & (input_tsdf <= eps)

        filtered = input_tsdf.clone()

        if self.filter_hallucinated_to == "trunc":
            filtered[hallucinated_mask] = self._trunc

        if self.filter_missing_to == "zero":
            filtered[missing_mask] = 0.0

        filtered = torch.clamp(filtered, -self._trunc, self._trunc)
        return filtered, missing_mask, hallucinated_mask

    def _load_shape_dirs_from_list(self, list_file: str) -> List[str]:
        shape_dirs = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Example line: 02871439/777785bf709e10ccffba1ef7a9b50ff2
                full_dir = os.path.join(self.root_dir, line)

                if os.path.isdir(full_dir):
                    shape_dirs.append(full_dir)
                else:
                    print(f"[WARN] Shape directory missing: {full_dir}")

        return shape_dirs

    def _sample_cube_fill_value(self):
        if self.cube_aug_fill_mode == "pos":
            return self._trunc
        elif self.cube_aug_fill_mode == "neg":
            return -self._trunc
        else:
            return self._trunc if random.random() < 0.5 else -self._trunc

    def _apply_scannet_cube_augmentation(self, input_tsdf: torch.Tensor, gt_tsdf: torch.Tensor):
        """
        NEW augmentation:
        Insert one or more axis-aligned cubes inside the GT bbox and fill them with
        either +trunc or -trunc.

        input_tsdf, gt_tsdf: [1, X, Y, Z]
        returns: augmented input_tsdf
        """
        if random.random() > self.cube_aug_prob:
            return input_tsdf

        out = input_tsdf.clone()

        # bbox from GT after spatial augmentations
        gt_np = gt_tsdf[0].cpu().numpy()   
        bbox = self.get_bbox(gt_np)        # [cx, cy, cz, sx, sy, sz]
        cx, cy, cz, sx, sy, sz = bbox.tolist()

        # bbox corners in XYZ convention
        x1 = int(round(cx - sx / 2.0))
        x2 = int(round(cx + sx / 2.0))
        y1 = int(round(cy - sy / 2.0))
        y2 = int(round(cy + sy / 2.0))
        z1 = int(round(cz - sz / 2.0))
        z2 = int(round(cz + sz / 2.0))

        # clip to valid voxel range
        x1 = max(0, min(self._size - 1, x1))
        y1 = max(0, min(self._size - 1, y1))
        z1 = max(0, min(self._size - 1, z1))

        x2 = max(x1 + 1, min(self._size, x2))
        y2 = max(y1 + 1, min(self._size, y2))
        z2 = max(z1 + 1, min(self._size, z2))

        # optional margin
        m = self.cube_aug_bbox_margin
        x1m, x2m = x1 + m, x2 - m
        y1m, y2m = y1 + m, y2 - m
        z1m, z2m = z1 + m, z2 - m

        # if margin collapses bbox, fall back
        if x2m <= x1m:
            x1m, x2m = x1, x2
        if y2m <= y1m:
            y1m, y2m = y1, y2
        if z2m <= z1m:
            z1m, z2m = z1, z2

        bbox_sx = max(1, x2m - x1m)
        bbox_sy = max(1, y2m - y1m)
        bbox_sz = max(1, z2m - z1m)

        num_cubes = random.randint(self.cube_aug_num_cubes_min, self.cube_aug_num_cubes_max)

        for _ in range(num_cubes):
            # cube size limited by bbox size
            max_cube_x = min(self.cube_aug_size_max, bbox_sx)
            max_cube_y = min(self.cube_aug_size_max, bbox_sy)
            max_cube_z = min(self.cube_aug_size_max, bbox_sz)

            min_cube_x = min(self.cube_aug_size_min, max_cube_x)
            min_cube_y = min(self.cube_aug_size_min, max_cube_y)
            min_cube_z = min(self.cube_aug_size_min, max_cube_z)

            if max_cube_x < 1 or max_cube_y < 1 or max_cube_z < 1:
                continue

            # use cubic size, but still clamp per axis
            cube_size = random.randint(
                max(1, min(min_cube_x, min_cube_y, min_cube_z)),
                max(1, min(max_cube_x, max_cube_y, max_cube_z))
            )

            # choose cube start position inside bbox
            xs_min = x1m
            ys_min = y1m
            zs_min = z1m

            xs_max = max(xs_min, x2m - cube_size)
            ys_max = max(ys_min, y2m - cube_size)
            zs_max = max(zs_min, z2m - cube_size)

            xs = random.randint(xs_min, xs_max)
            ys = random.randint(ys_min, ys_max)
            zs = random.randint(zs_min, zs_max)

            xe = min(xs + cube_size, self._size)
            ye = min(ys + cube_size, self._size)
            ze = min(zs + cube_size, self._size)

            fill_value = self._sample_cube_fill_value()

            out[0, xs:xe, ys:ye, zs:ze] = fill_value

        out = torch.clamp(out, -self._trunc, self._trunc)
        return out

    def _apply_scannet_value_augmentation(self, tsdf: torch.Tensor):
        """
        Old ScanNet-like corruption to TSDF ONLY.
        tsdf: [1, x, y, z]
        """
        # Gaussian noise
        if self.tsdf_noise_std > 0:
            tsdf = tsdf + torch.randn_like(tsdf) * self.tsdf_noise_std

        # Small multiplicative bias
        if self.tsdf_value_scale > 0:
            scale = 1.0 + (torch.rand(1, device=tsdf.device) * 2 - 1) * self.tsdf_value_scale
            tsdf = tsdf * scale

        # Random voxel dropout
        if self.tsdf_dropout_p > 0:
            drop_mask = (torch.rand_like(tsdf) < self.tsdf_dropout_p)

            if self.tsdf_dropout_to == "trunc":
                tsdf = tsdf.masked_fill(drop_mask, self._trunc)
            else:
                tsdf = tsdf.masked_fill(drop_mask, 0.0)

        tsdf = torch.clamp(tsdf, -self._trunc, self._trunc)
        return tsdf

    def _apply_scannet_like_augmentation(self, input_tsdf: torch.Tensor, gt_tsdf: torch.Tensor):
        """
        Dispatcher:
          - mode == "value"     -> old value corruption
          - mode == "bbox_cube" -> new cube-in-bbox corruption
        """
        if not self.scannet_like_augmentation:
            return input_tsdf

        if self.scannet_like_aug_mode == "bbox_cube":
            return self._apply_scannet_cube_augmentation(input_tsdf, gt_tsdf)
        else:
            return self._apply_scannet_value_augmentation(input_tsdf)
    # -------------------------------------------------------
    # Scan all shapes in the directory (old behavior)
    # -------------------------------------------------------
    def _scan_all_shape_dirs(self, root_dir: str) -> List[str]:
        shape_dirs = []

        for dirpath, _, filenames in os.walk(root_dir):
            if "gt.npz" in filenames:
                shape_dirs.append(dirpath)

        return shape_dirs

    # -------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ---------- bbox util (same as in your other dataset) ----------
    def get_bbox(self, gt_tsdf: np.ndarray) -> np.ndarray:
        """
        gt_tsdf: [X,Y,Z] np.float32
        Returns [cx, cy, cz, sx, sy, sz] in voxel space.
        """
        mask = gt_tsdf < self._trunc
        if not np.any(mask):
            # fallback trivial box
            return np.array([self._size/2]*3 + [1,1,1], dtype=np.float32)

        coords_zyx = np.argwhere(mask)
        minc = coords_zyx.min(axis=0)
        maxc = coords_zyx.max(axis=0)

        center_xyz = (minc + maxc) / 2.0
        size_xyz = (maxc - minc + 1)

        cx, cy, cz = center_xyz[0], center_xyz[1], center_xyz[2]
        sx, sy, sz = size_xyz[0], size_xyz[1], size_xyz[2]

        return np.array([cx, cy, cz, sx, sy, sz], dtype=np.float32)

    def _load_tsdf(self, path: str) -> np.ndarray:
        data = np.load(path)
        tsdf = data["tsdf"].astype(np.float32) * self._res
        return np.clip(tsdf, -self._trunc, self._trunc)
    
    def _load_tudf(self, path: str) -> np.ndarray:
        data = np.load(path)
        tudf = data["tsdf"].astype(np.float32) #* self._res
        return tudf

    # -------------------------------------------------------
    def __getitem__(self, idx: int):
        shape_dir, input_id = self.samples[idx]

        gt_path = os.path.join(shape_dir, "gt.npz")
        if self.use_gt_input:
            input_path = gt_path
        else:
            input_path = os.path.join(shape_dir, f"input_{input_id}.npz")
        if self.use_gt_teacher: 
            teacher_path = os.path.join(shape_dir, f"{self.dino_model_name}_gt_{self.grid_size}_{self.dataset_tag}.npz")
        else:
            teacher_path = os.path.join(shape_dir, f"{self.dino_model_name}_{input_id}_{self.grid_size}_{self.dataset_tag}.npz")

        if self.use_tudf:
            input_tsdf_np = self._load_tudf(input_path)
            gt_tsdf_np = self._load_tudf(gt_path)
        else:
            input_tsdf_np = self._load_tsdf(input_path)
            gt_tsdf_np = self._load_tsdf(gt_path)      # [X,Y,Z]
        t = np.load(teacher_path)

        teacher_np = t["data"].astype(np.float32)  # [C,G,G,G]

        teacher_cls_np = t["cls"].astype(np.float32) if "cls" in t.files else None     # [C]
        teacher_reg_np = t["reg"].astype(np.float32) if "reg" in t.files else None     # [R,C]

        if self.normalize_dino:
            eps = 1e-6
            # voxel features
            norm = np.linalg.norm(teacher_np, axis=0, keepdims=True) + eps
            teacher_np = teacher_np / norm

            # cls/reg tokens (optional but recommended for cosine losses)
            if teacher_cls_np is not None:
                teacher_cls_np = teacher_cls_np / (np.linalg.norm(teacher_cls_np) + eps)
            if teacher_reg_np is not None:
                teacher_reg_np = teacher_reg_np / (np.linalg.norm(teacher_reg_np, axis=1, keepdims=True) + eps)

        # bbox for possible add_walls
        bbox_np = self.get_bbox(gt_tsdf_np)           # (6,)
        bbox_t = torch.from_numpy(bbox_np).float()

        # ---- parse category and model id from shape_dir ----
        rel_path = os.path.relpath(shape_dir, self.root_dir)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            category_id = parts[0]
            model_id = parts[1]
        else:
            category_id = parts[0]
            model_id = os.path.basename(shape_dir)

        category_index = CATEGORY_TO_INDEX.get(category_id, -1)

        # to tensors
        input_tsdf = torch.from_numpy(input_tsdf_np).float().unsqueeze(0)  # [1,X,Y,Z]
        gt_tsdf = torch.from_numpy(gt_tsdf_np).float().unsqueeze(0)        # [1,X,Y,Z]
        teacher = torch.from_numpy(teacher_np).float()  # [C,G,G,G]

        teacher_cls = torch.from_numpy(teacher_cls_np).float() if teacher_cls_np is not None else None
        teacher_reg = torch.from_numpy(teacher_reg_np).float() if teacher_reg_np is not None else None

        # -----------------------------
        # DATA AUGMENTATION (train only)
        # -----------------------------
        if self.train and self.augmentation:
            # shared random params
            flip_type = get_mirror_params()           # 0..3
            trans_idx = get_transpose_params()        # 0 or 1
            shifts = get_jitter_params(self._jitter_range)

            # optionally add walls BEFORE spatial ops
            if self._walls_aug:
                input_tsdf = add_walls(input_tsdf, bbox_t)

            # ---- volumes ----
            input_tsdf = apply_mirror(input_tsdf, flip_type)
            gt_tsdf    = apply_mirror(gt_tsdf,    flip_type)

            input_tsdf = apply_transpose(input_tsdf, trans_idx)
            gt_tsdf    = apply_transpose(gt_tsdf,    trans_idx)

            input_tsdf = apply_jitter(input_tsdf, shifts, self._trunc)
            gt_tsdf    = apply_jitter(gt_tsdf,    shifts, self._trunc)

            # ---- DINO features: same spatial ops on [C,G,G,G] ----
            teacher = apply_mirror_feats(teacher, flip_type)
            teacher = apply_transpose_feats(teacher, trans_idx)
            teacher = apply_jitter_feats(teacher, shifts)
            # ---- ScanNet-like value augmentation (TSDF ONLY) ----
            #input_tsdf = self._apply_scannet_like_augmentation(input_tsdf, gt_tsdf)

        
        missing_mask = None
        hallucinated_mask = None
        if teacher_cls is None:
            teacher_cls = torch.zeros(self.dino_dim, dtype=torch.float32)
        if teacher_reg is None:
            teacher_reg = torch.zeros(0, self.dino_dim, dtype=torch.float32)  # [0,C]
        if self.filter_input_tsdf:
            input_tsdf, missing_mask, hallucinated_mask = self._filter_input_by_gt(input_tsdf, gt_tsdf)
        return {
            "input_tsdf": input_tsdf,                                       # [1,32,32,32]
            "gt_tsdf": gt_tsdf,                                             # [1,32,32,32]
            "teacher": teacher,                                             # [C,G,G,G]
            "shape_dir": shape_dir,
            "view_id": input_id,
            "category_id": category_id,                                     # synset string
            "category_index": torch.tensor(category_index, dtype=torch.long),
            "class_id": torch.tensor(category_index, dtype=torch.long),
            "model_id": model_id,
            "teacher_cls": teacher_cls,  # [C] or None
            "teacher_reg": teacher_reg,  # [R,C] or None
        }
