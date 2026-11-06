import os
import glob
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from data.augmentations import *
import trimesh

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

class ShapeNetDataset(Dataset):
    """
    Single-scan ShapeNet TSDF dataset for complementary training.

    Returns per sample:
        input      : [1, X, Y, Z]
        gt         : [1, X, Y, Z]
        in_dino    : [F, X, Y, Z]   (or empty tensor if disabled)
        gt_dino    : [F, X, Y, Z]   (or empty tensor if disabled)
        class_id   : int
        name       : str

    """

    def __init__(self, args, filename, train=False):
        self._file = filename
        self._data_path = args.data_path
        self.train = train

        self._trunc = float(args.truncation)
        self._res = int(args.patch_res)
        self._size = int(getattr(args, "tsdf_size", 32))

        self._jitter_range = int(getattr(args, "jitter_range", 2))
        self.return_bbox = bool(getattr(args, "return_bbox", False))

        # DINO
        self.return_dino_gt = bool(getattr(args, "return_dino_gt", False))
        self.return_dino_input = bool(getattr(args, "return_dino_input", False))
        self.dino_grid = str(getattr(args, "dino_grid", "32"))
        self.dino_key = str(getattr(args, "dino_key", "data"))
        self.dino_tag = str(getattr(args, "dino_tag", "inside"))
        self.dino_model_name = str(getattr(args, "dino_model_name", "dinov3_vits16"))

        self.samples = []
        self.cat_idx_dict = {cat: [] for cat in CATEGORY_TO_INDEX.keys()}
        self._read_data()

    # --------------------------------------------------
    # IO
    # --------------------------------------------------
    def _load_tsdf(self, path: str) -> np.ndarray:
        """
        Loads TSDF and returns [X,Y,Z]
        """
        npy_path = path.replace(".npz", ".npy")

        if os.path.exists(npy_path):
            tsdf = np.load(npy_path, mmap_mode="r")
        else:
            tsdf = np.load(path)["tsdf"]

        tsdf = tsdf.astype(np.float32) * self._res
        tsdf = np.clip(tsdf, -self._trunc, self._trunc)


        return tsdf  # [X,Y,Z]

    def _load_dino(self, path: str) -> np.ndarray:
        """
        Loads DINO voxel features and returns [F,X,Y,Z]

        Assumes stored format is [F,Z,Y,X]
        """
        t = np.load(path)
        if self.dino_key not in t.files:
            raise KeyError(f"Key '{self.dino_key}' not found in {path}. Available: {t.files}")

        feats = t[self.dino_key].astype(np.float32)
        if feats.ndim != 4:
            raise ValueError(f"Expected [F,Z,Y,X], got {feats.shape} from {path}")


        return feats

    # --------------------------------------------------
    # bbox
    # --------------------------------------------------
    def get_bbox(self, gt_tsdf_xyz: np.ndarray) -> np.ndarray:
        """
        gt_tsdf_xyz: [X,Y,Z]
        returns [cx, cy, cz, sx, sy, sz]
        """
        mask = gt_tsdf_xyz < self._trunc
        if not np.any(mask):
            return np.array([self._size / 2] * 3 + [1, 1, 1], dtype=np.float32)

        coords_xyz = np.argwhere(mask)  # [:,0]=x [:,1]=y [:,2]=z
        minc = coords_xyz.min(axis=0)
        maxc = coords_xyz.max(axis=0)

        center_xyz = (minc + maxc) / 2.0
        size_xyz = (maxc - minc + 1)

        cx, cy, cz = center_xyz[0], center_xyz[1], center_xyz[2]
        sx, sy, sz = size_xyz[0], size_xyz[1], size_xyz[2]

        return np.array([cx, cy, cz, sx, sy, sz], dtype=np.float32)



    # --------------------------------------------------
    # indexing
    # --------------------------------------------------
    def _read_data(self):
        with open(self._file, "r") as f:
            rels = [ln.strip() for ln in f if ln.strip()]

        for rel in rels:
            shape_dir = os.path.join(self._data_path, rel)
            gt_path = os.path.join(shape_dir, "gt.npz")
            if not os.path.exists(gt_path):
                continue

            category = os.path.normpath(shape_dir).split(os.sep)[-2]
            if category not in self.cat_idx_dict:
                continue

            input_files = sorted(glob.glob(os.path.join(shape_dir, "input*.npz")))
            if not input_files:
                continue

            for p in input_files:
                self.samples.append((shape_dir, p))
                self.cat_idx_dict[category].append(len(self.samples) - 1)

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    # main
    # --------------------------------------------------
    def __getitem__(self, idx):
        shape_dir, input_path = self.samples[idx]

        gt_path = os.path.join(shape_dir, "gt.npz")
        gt_np = self._load_tsdf(gt_path)                   # [X,Y,Z]
        input_np = self._load_tsdf(input_path)[None, ...] # [1,X,Y,Z]
        name = input_path
        base = os.path.basename(input_path)
        digits = "".join([ch for ch in base if ch.isdigit()])
        if digits == "":
            raise ValueError(f"Could not parse input id from filename: {input_path}")
        scan_id = int(digits)

        if self.return_dino_input:
            dp = os.path.join(
                shape_dir,
                f"{self.dino_model_name}_{scan_id}_{self.dino_grid}_{self.dino_tag}.npz"
            )
            if not os.path.exists(dp):
                raise FileNotFoundError(dp)
            in_dino_np = self._load_dino(dp)   # [F,X,Y,Z]
        else:
            in_dino_np = None

        if self.return_dino_gt:
            dgt = os.path.join(
                shape_dir,
                f"{self.dino_model_name}_gt_{self.dino_grid}_{self.dino_tag}.npz"
            )
            if not os.path.exists(dgt):
                raise FileNotFoundError(dgt)
            gt_dino_np = self._load_dino(dgt)  # [F,X,Y,Z]
        else:
            gt_dino_np = None

        if self.return_bbox:
            bbox_np = self.get_bbox(gt_tsdf.squeeze(0).cpu().numpy())
            bbox_t = torch.from_numpy(bbox_np).float()
        else:
            bbox_t = torch.zeros(6, dtype=torch.float32)

        input_tsdf = torch.from_numpy(input_np).float()          # [1,X,Y,Z]
        gt_tsdf = torch.from_numpy(gt_np).float().unsqueeze(0)   # [1,X,Y,Z]

        if self.return_dino_input:
            in_dino = torch.from_numpy(in_dino_np).float()       # [F,X,Y,Z]
        else:
            in_dino = torch.empty(0, dtype=torch.float32)

        if self.return_dino_gt:
            gt_dino = torch.from_numpy(gt_dino_np).float()       # [F,X,Y,Z]
        else:
            gt_dino = torch.empty(0, dtype=torch.float32)

        if self.train:
            flip_type = get_mirror_params()
            trans_idx = get_transpose_params()
            shifts = get_jitter_params(self._jitter_range)

            input_tsdf = apply_mirror(input_tsdf, flip_type)
            gt_tsdf = apply_mirror(gt_tsdf, flip_type)

            input_tsdf = apply_transpose(input_tsdf, trans_idx)
            gt_tsdf = apply_transpose(gt_tsdf, trans_idx)

            input_tsdf = apply_jitter(input_tsdf, shifts, self._trunc)
            gt_tsdf = apply_jitter(gt_tsdf, shifts, self._trunc)

            if self.return_dino_input:
                in_dino = apply_mirror_feats(in_dino, flip_type)
                in_dino = apply_transpose_feats(in_dino, trans_idx)
                in_dino = apply_jitter_feats(in_dino, shifts)

            if self.return_dino_gt:
                gt_dino = apply_mirror_feats(gt_dino, flip_type)
                gt_dino = apply_transpose_feats(gt_dino, trans_idx)
                gt_dino = apply_jitter_feats(gt_dino, shifts)

        category = os.path.normpath(shape_dir).split(os.sep)[-2]
        class_id = CATEGORY_TO_INDEX[category]

        return {
            "input1": input_tsdf,       # [1,X,Y,Z]
            "gt": gt_tsdf,             # [1,X,Y,Z]
            "in_dino": in_dino,        # [F,X,Y,Z] #debug purpuses
            "gt_dino": gt_dino,        # [F,X,Y,Z] #debug purpuses
            "class_id": class_id, #debug purpuses
            "name": name, #debug purpuses
            "bbox": bbox_t, #debug purpuses
        }

class ScannetDataset(Dataset):
    """
    This class reads all the training dataset for Scannet
    """
    def __init__(self, args, file_name, use_bbox=True):
        """
        Initializes ScannetDataset object. Sets up related pathes
        and other parameters.

        :param file: file contains scene names
        :type  file: str
        :param data_path: path for the data
        :type data_path: str
        :param truncation: value for truncation
        :typr truncation: float
        :param res: input resolution
        :type res: int
        :param use_bbox: whether using bbox to crop inputs or not
        :type use_bbox: bool
        """
        # path parameters
        self._file = file_name
        self._data_path = args.scannet_data_path
        self._truncation = args.truncation
        self._use_bbox = use_bbox
        self._res = args.patch_res
        self._bbox_mesh = trimesh.load('data/bbox.ply', process=False) # scan2CAD bbox mesh
        # read data
        self._input_sdf = []
        self._gt_sdf = []
        self._mask_names = []
        self._bbox = []
        self._read_data()

    def get_bbox_anno(self, mask_file):
        bbox_mesh = copy.deepcopy(self._bbox_mesh)
        with np.load(mask_file) as data:
            voxel_size = data["voxel_size"]
            voxel_origin=data['voxel_origin']
            Mbbox2model=data["Mbbox2model"]
            Tscales=data["T_scales"]
        model_bbox_model = bbox_mesh.apply_transform(Mbbox2model)
        model_bbox_model.apply_transform(Tscales)
        bbox_ori = model_bbox_model.bounds[0] - voxel_origin
        bbox_ori /= voxel_size
        bbox_ori = np.array(bbox_ori, dtype=int)
        bbox_end = model_bbox_model.bounds[1] - voxel_origin
        bbox_end /= voxel_size
        bbox_end = np.array(bbox_end, dtype=int)
        bbox_end += 1
        bbox_ori += 1
        bbox_ori[bbox_ori < 0] = 0
        bbox_end[bbox_end > self._res - 1] = self._res - 1

        return [[bbox_ori[0], bbox_end[0]], [bbox_ori[1], bbox_end[1]], [bbox_ori[2], bbox_end[2]]], voxel_size

    def get_bbox(self,gt_tsdf):
        """
        Computes the bounding box from a TSDF in a vectorized way.
        Assumes voxels with value <= 0 are considered inside the shape.
        
        Returns:
            np.ndarray: shape (6,), containing [cx, cy, cz, sx, sy, sz]
        """
        # Mask where shape is inside or near-surface
        mask = gt_tsdf <= 0  # shape: (32, 32, 32)

        # Early exit if empty
        if not np.any(mask):
            return np.array([16, 16, 16, 1, 1, 1], dtype=np.float32)

        # Get the bounding indices
        coords = np.argwhere(mask)  # shape: [N, 3]
        min_corner = coords.min(axis=0)  # [x_min, y_min, z_min]
        max_corner = coords.max(axis=0)  # [x_max, y_max, z_max]

        # Compute center and size
        center = (min_corner + max_corner) / 2.0
        size = max_corner - min_corner + 1
        return np.concatenate([center, size]).astype(np.float32)
    
    def _read_data(self):
        """
        This function reads data from data path
        """
        with open(self._file, 'r') as data:
            print (self._file)
            mask_files = data.readlines()
            voxel_size = 0.0
            for mask_file in mask_files:
                mask_file = os.path.join(self._data_path, mask_file.split('\n')[0])
                sdf_file = mask_file[:-4] + "_sdf.npz"
                gt_sdf_file = mask_file[:-8] + "scaled_sdf_gt.npy"
                if (os.path.exists(mask_file) is False or 
                    os.path.exists(sdf_file) is False or
                    os.path.exists(gt_sdf_file) is False):
                    continue
                # get bbox
                bbox, voxel_size = self.get_bbox_anno(mask_file)
                self._bbox.append(bbox)
                # get input data
                with np.load(sdf_file, 'rb') as sdf:
                    input_sdf = sdf["instance_sdf"]
                    input_sdf /= voxel_size # convert to voxel unit
                    input_sdf[np.where(input_sdf>self._truncation)] = self._truncation
                    input_sdf[np.where(input_sdf<-1*self._truncation)] = -1*self._truncation
                    if self._use_bbox:
                        input_sdf_new = np.ones(input_sdf.shape) * self._truncation
                        input_sdf_new[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = input_sdf[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
                        input_sdf = copy.deepcopy(input_sdf_new)
                    self._input_sdf.append(input_sdf)
                # get gt data
                with open(gt_sdf_file, 'rb') as data:
                    gt_sdf = np.load(data)
                    gt_sdf /= voxel_size
                    gt_sdf[np.where(gt_sdf>self._truncation)] = self._truncation
                    gt_sdf[np.where(gt_sdf< -1 * self._truncation)] = -1 * self._truncation
                    self._gt_sdf.append(gt_sdf)
                self._mask_names.append(mask_file)
        print (len(self._mask_names))

    def __len__(self):
        """
        This function returns the number of traing samples
        """
        return len(self._mask_names)

    def __getitem__(self, idx):
        """
        This function returns idx-th sdf and labels in the dataset
        """
        input_sdf = copy.deepcopy(self._input_sdf[idx])
        gt_sdf = copy.deepcopy(self._gt_sdf[idx])
        input_sdf = torch.from_numpy(input_sdf).unsqueeze(0).float()
        gt_sdf = torch.from_numpy(gt_sdf).unsqueeze(0).float()
        bbox = self.get_bbox(gt_sdf.squeeze(0).numpy())
        bbox = torch.from_numpy(np.array(bbox)).float()

        input_dict = {"input1": input_sdf, "gt":gt_sdf, "bbox":bbox, "class_id":-1, "name":self._mask_names[idx]}
        return input_dict