#!/usr/bin/env python3
"""
Precompute *incomplete* (single-view) DINO voxel features from an already-fused GT DINO.

What it does:
- Loads fused GT DINO:   tsdf_root/cat/shape/dino_features_gt_{G}.npz  (key "data")
- For each target view vid (e.g. 5,6,7,8):
  - Reads depth_{vid:03d}.exr
  - Computes per-view TRILINEAR COVERAGE mask in voxel grid (wsum_view > 0)
  - Applies mask to fused GT DINO => "partial/incomplete DINO" for that view
  - Saves to: tsdf_root/cat/shape/dino_features_{input_id}_{G}.npz  with key "data" [C,G,G,G]
    where input_id = vid - 1 (so v005 -> input_4, v006 -> input_5, ...)

Dataset convention:
- images/depth start at 001 (depthrenderer)
- input_id in filenames is 0-based relative to view_id-1(PatchComplete)

Example:
  python precompute_incompletedino_from_gtdino.py \
    --render_root / ../../../depth_renderer/datasets/ShapeNetRenderings \
    --tsdf_root   /../../../PatchComplete/data/shapenet \
    --camera_root /../../../depth_renderer/datasets/camera_settings \
    --list_file   /../../data/data_splits/shapenet_test.txt \
    --grid_sizes 32 \
    --views 5 6 7 8 \
    --gpus 0 1 2 4 5 \
    --num_workers 5

Outputs (for G=32):
  dino_features_4_32.npz  (from v005)
  dino_features_5_32.npz  (from v006)
  dino_features_6_32.npz  (from v007)
  dino_features_7_32.npz  (from v008)

Notes:
- No DINO model is used here. This is depth-only masks + masking fused GT.
- GT TSDF inside mask (recommended) to keep consistency with GT preprocessing.
  Default: enabled.
"""

import os
import argparse
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

import multiprocessing as mp

MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
# -----------------------------
# Canonical cube convention
# -----------------------------
CAN_MIN = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
CAN_MAX = np.array([ 0.5,  0.5,  0.5], dtype=np.float32)
CAN_RANGE = CAN_MAX - CAN_MIN  # [1,1,1]


# -----------------------------
# Camera loading
# -----------------------------
def load_cam_K(camera_root: str) -> np.ndarray:
    kpath = os.path.join(camera_root, "cam_K", "cam_K.txt")
    if not os.path.exists(kpath):
        raise FileNotFoundError(f"cam_K not found: {kpath}")
    K = np.loadtxt(kpath).astype(np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K, got {K.shape} at {kpath}")
    return K

def load_cam_RT(camera_root: str, view_id: int) -> np.ndarray:
    rtpath = os.path.join(camera_root, "cam_RT", f"cam_RT_{view_id:03d}.txt")
    if not os.path.exists(rtpath):
        raise FileNotFoundError(f"cam_RT not found: {rtpath}")

    RT = np.loadtxt(rtpath).astype(np.float32)
    if RT.shape == (4, 4):
        RT = RT[:3, :4]
    if RT.shape != (3, 4):
        raise ValueError(f"Expected 3x4 (or 4x4) RT, got {RT.shape} at {rtpath}")
    return RT

def get_surface_mask(tsdf):
    surface = np.zeros_like(tsdf, dtype=bool)

    # x direction
    surface[:-1, :, :] |= (tsdf[:-1, :, :] * tsdf[1:, :, :]) < 0
    surface[1:, :, :]  |= (tsdf[:-1, :, :] * tsdf[1:, :, :]) < 0

    # y direction
    surface[:, :-1, :] |= (tsdf[:, :-1, :] * tsdf[:, 1:, :]) < 0
    surface[:, 1:, :]  |= (tsdf[:, :-1, :] * tsdf[:, 1:, :]) < 0

    # z direction
    surface[:, :, :-1] |= (tsdf[:, :, :-1] * tsdf[:, :, 1:]) < 0
    surface[:, :, 1:]  |= (tsdf[:, :, :-1] * tsdf[:, :, 1:]) < 0

    return surface
# -----------------------------
# EXR reading (ray distance)
# -----------------------------
def read_exr_dist(depth_path: str) -> np.ndarray:
    d = imageio.imread(depth_path)
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


# -----------------------------
# dist (ray) -> z-depth (camera)
# -----------------------------
def dist_to_depth_single(dist_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = dist_map.shape
    depth = np.full((H, W), np.inf, dtype=np.float32)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dist = dist_map.astype(np.float32)
    valid = np.isfinite(dist) & (dist < np.inf)
    if not np.any(valid):
        return depth

    u_valid = u[valid].astype(np.float32)
    v_valid = v[valid].astype(np.float32)
    dist_valid = dist[valid]

    x_temp = (u_valid - K[0, 2]) / K[0, 0]
    y_temp = (v_valid - K[1, 2]) / K[1, 1]
    depth_valid = dist_valid / np.sqrt(x_temp**2 + y_temp**2 + 1.0)

    depth[valid] = depth_valid.astype(np.float32)
    return depth


# -----------------------------
# Patch centers in canonical (mean over pixels in patch block)
# -----------------------------
def patch_centers_in_canonical(
    dist_np: np.ndarray,
    K: np.ndarray,
    RT: np.ndarray,
    Hp: int,
    Wp: int,
):
    dist = dist_np
    if dist.ndim == 3:
        dist = dist[..., 0]
    H, W = dist.shape

    depth = dist_to_depth_single(dist, K)
    valid = np.isfinite(depth) & (depth < np.inf)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.copy()
    z[~valid] = 0.0

    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    point_cam = np.stack([x, y, z], axis=-1).astype(np.float32)

    # X_can = (X_cam - t) @ R^T
    R = RT[:, :3].astype(np.float32)
    t = RT[:, 3].astype(np.float32)
    point_can = np.einsum("hwc,dc->hwd", point_cam - t[None, None, :], R.T).astype(np.float32)
    point_can[~valid] = np.nan

    block_h = H // Hp
    block_w = W // Wp
    if block_h * Hp != H or block_w * Wp != W:
        raise ValueError(f"Image ({H},{W}) not divisible by patch grid ({Hp},{Wp}).")

    patch_points = np.zeros((Hp, Wp, 3), dtype=np.float32)
    patch_valid = np.zeros((Hp, Wp), dtype=bool)

    for i in range(Hp):
        y0, y1 = i * block_h, (i + 1) * block_h
        for j in range(Wp):
            x0, x1 = j * block_w, (j + 1) * block_w
            block = point_can[y0:y1, x0:x1, :]
            ok = ~np.isnan(block[..., 0])
            if not np.any(ok):
                patch_valid[i, j] = False
                patch_points[i, j] = 0.0
            else:
                patch_valid[i, j] = True
                patch_points[i, j] = np.nanmean(block, axis=(0, 1))

    return patch_points, patch_valid


# -----------------------------
# Per-view TRILINEAR COVERAGE (weight_sum only)
# mask = weight_sum > 0
# -----------------------------
def trilinear_coverage_wsum_from_depth(
    dist_np: np.ndarray,
    K: np.ndarray,
    RT: np.ndarray,
    G: int,
    Hp: int,
    Wp: int,
) -> np.ndarray:
    patch_points, patch_valid = patch_centers_in_canonical(dist_np, K, RT, Hp, Wp)

    pts_norm = (patch_points - CAN_MIN[None, None, :]) / CAN_RANGE[None, None, :]
    pts_norm = np.clip(pts_norm, 0.0, 1.0)
    xyz = pts_norm * (G - 1)

    x = xyz[..., 0]; y = xyz[..., 1]; z = xyz[..., 2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)

    x1 = np.clip(x0 + 1, 0, G - 1)
    y1 = np.clip(y0 + 1, 0, G - 1)
    z1 = np.clip(z0 + 1, 0, G - 1)

    dx = (x - x0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    dz = (z - z0).astype(np.float32)

    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w100 = (dx)     * (1 - dy) * (1 - dz)
    w010 = (1 - dx) * (dy)     * (1 - dz)
    w110 = (dx)     * (dy)     * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * (dz)
    w101 = (dx)     * (1 - dy) * (dz)
    w011 = (1 - dx) * (dy)     * (dz)
    w111 = (dx)     * (dy)     * (dz)

    weight_sum = np.zeros((G, G, G), dtype=np.float32)

    valid_flat = patch_valid.reshape(-1)
    x0f = x0.reshape(-1); y0f = y0.reshape(-1); z0f = z0.reshape(-1)
    x1f = x1.reshape(-1); y1f = y1.reshape(-1); z1f = z1.reshape(-1)

    w000f = w000.reshape(-1); w100f = w100.reshape(-1)
    w010f = w010.reshape(-1); w110f = w110.reshape(-1)
    w001f = w001.reshape(-1); w101f = w101.reshape(-1)
    w011f = w011.reshape(-1); w111f = w111.reshape(-1)

    for n in range(valid_flat.shape[0]):
        if not valid_flat[n]:
            continue

        gx0, gy0, gz0 = int(x0f[n]), int(y0f[n]), int(z0f[n])
        gx1, gy1, gz1 = int(x1f[n]), int(y1f[n]), int(z1f[n])

        w = float(w000f[n]); 
        if w > 0: weight_sum[gx0, gy0, gz0] += w
        w = float(w100f[n]); 
        if w > 0: weight_sum[gx1, gy0, gz0] += w
        w = float(w010f[n]); 
        if w > 0: weight_sum[gx0, gy1, gz0] += w
        w = float(w110f[n]); 
        if w > 0: weight_sum[gx1, gy1, gz0] += w
        w = float(w001f[n]); 
        if w > 0: weight_sum[gx0, gy0, gz1] += w
        w = float(w101f[n]); 
        if w > 0: weight_sum[gx1, gy0, gz1] += w
        w = float(w011f[n]); 
        if w > 0: weight_sum[gx0, gy1, gz1] += w
        w = float(w111f[n]); 
        if w > 0: weight_sum[gx1, gy1, gz1] += w

    return weight_sum


# -----------------------------
# GT TSDF -> inside mask in G^3 (optional AND)
# -----------------------------
def load_gt_tsdf_np(shape_dir: str, trunc: float = 3.0, res: int = 32) -> np.ndarray:
    gt = np.load(os.path.join(shape_dir, "gt.npz"))["tsdf"].astype(np.float32) * float(res)
    gt = np.clip(gt, -float(trunc), float(trunc))
    return gt  # [32,32,32]

def tsdf_inside_mask_G(
    args,
    gt_tsdf_32: np.ndarray,
    G: int,
    filter_value: Union[float, Tuple[float, float]] = 0.0
) -> np.ndarray:
    """
    inside mask:
        - if float: tsdf <= filter_value
        - if (a, b): a < tsdf < b  (exclusive)

    returns [G,G,G] bool
    """
    if args.dataset_tag == "surfacemask":
        m32 = get_surface_mask(gt_tsdf_32)
    # --- handle scalar vs range ---
    else:
        if isinstance(filter_value, (tuple, list)):
            if len(filter_value)>1:
                assert len(filter_value) == 2, "filter_value must be (a, b)"
                a, b = filter_value
                m32 = (gt_tsdf_32 > a) & (gt_tsdf_32 < b)
            else:
                m32 = (gt_tsdf_32 <= filter_value[0])

    # --- resize if needed ---
    if G == gt_tsdf_32.shape[0]:
        return m32

    m = torch.from_numpy(m32[None, None].astype(np.float32))   # [1,1,32,32,32]
    mG = F.interpolate(m, size=(G, G, G), mode="nearest")       # [1,1,G,G,G]

    return (mG[0, 0].numpy() > 0.5)


# -----------------------------
# IO utils
# -----------------------------
def load_shape_list(list_file: str) -> List[str]:
    out: List[str] = []
    with open(list_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    n = max(1, n)
    out = [[] for _ in range(n)]
    for i, x in enumerate(xs):
        out[i % n].append(x)
    return out

def load_dino_gt(args, shape_dir: str, G: int, model: str) -> np.ndarray:
    p = os.path.join(shape_dir, f"{model}_gt_{G}_{args.dataset_tag}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return np.load(p)["data"].astype(np.float32)  # [C,G,G,G]

def out_partial_path(args, shape_dir: str, input_id: int, G: int, model: str) -> str:
    return os.path.join(shape_dir, f"{model}_{input_id}_{G}_{args.dataset_tag}.npz")


# -----------------------------
# Worker
# -----------------------------
def worker_main(worker_id: int, gpu_id: int, rels: List[str], args_dict: dict):
    torch.set_num_threads(1)
    args = argparse.Namespace(**args_dict)

    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    print(f"[Worker {worker_id}] gpu_id={gpu_id} shapes={len(rels)} views={args.views} grids={args.grid_sizes}")

    K = load_cam_K(args.camera_root)

    # preload RTs for all needed vids
    RTs: Dict[int, np.ndarray] = {}
    for vid in args.views:
        RTs[int(vid)] = load_cam_RT(args.camera_root, int(vid))

    pbar = tqdm(rels, desc=f"Worker{worker_id}", position=worker_id, leave=True)
    for rel in pbar:
        cat, sid = rel.split("/")
        render_dir = os.path.join(args.render_root, cat, sid)
        shape_dir  = os.path.join(args.tsdf_root,  cat, sid)

        if not os.path.isdir(render_dir) or not os.path.isdir(shape_dir):
            continue

        # load TSDF once if we and masks
        tsdf_mask_G = {}
        if args.and_tsdf_inside:
            try:
                gt_tsdf_32 = load_gt_tsdf_np(shape_dir, trunc=args.tsdf_trunc, res=args.tsdf_res)
            except Exception as e:
                print(f"[Worker {worker_id}] WARN: missing/bad gt.npz for {rel}: {e}")
                continue
            for G in args.grid_sizes:
                tsdf_mask_G[int(G)] = tsdf_inside_mask_G(gt_tsdf_32, G=int(G), filter_value=args.tsdf_filter_value).astype(bool)

        # load fused GT DINO per G
        dino_gt = {}
        for G in args.grid_sizes:
            G = int(G)
            try:
                dino_gt[G] = load_dino_gt(args, shape_dir, G=G, model=args.model)  # [C,G,G,G]
            except Exception as e:
                print(f"[Worker {worker_id}] WARN: missing {args.model}_gt_{G} for {rel}: {e}")
                dino_gt[G] = None

        # determine patch grid (Hp/Wp) for depth-to-coverage
        # For ViT-L/16 at 512x512 => 32x32 patch grid.
        Hp = args.patch_grid
        Wp = args.patch_grid

        for vid in args.views:
            vid = int(vid)
            input_id = vid - 1  # v005 -> 4, v006 -> 5, etc.

            dpath = os.path.join(render_dir, f"depth_{vid:03d}.exr")
            if not os.path.exists(dpath):
                continue

            # skip if exists
            if not args.overwrite:
                all_exist = True
                for G in args.grid_sizes:
                    G = int(G)
                    if not os.path.exists(out_partial_path(args, shape_dir, input_id=input_id, G=G, model=args.model)):
                        all_exist = False
                        break
                if all_exist:
                    continue

            try:
                dist = read_exr_dist(dpath)
            except Exception as e:
                print(f"[Worker {worker_id}] WARN: depth read fail {rel} v{vid:03d}: {e}")
                continue

            RT = RTs[vid]

            for G in args.grid_sizes:
                G = int(G)
                if dino_gt[G] is None:
                    continue

                wsum = trilinear_coverage_wsum_from_depth(dist, K, RT, G=G, Hp=Hp, Wp=Wp)
                mask = (wsum > 0.0)

                if args.and_tsdf_inside:
                    mask = mask & tsdf_mask_G[G]

                # apply mask to fused GT
                partial = dino_gt[G] * mask[None, ...].astype(np.float32)

                outp = out_partial_path(shape_dir, input_id=input_id, G=G, model=args.model)
                np.savez_compressed(outp, data=partial.astype(np.float32))

    print(f"[Worker {worker_id}] done.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Precompute incomplete (single-view) DINO features by masking fused GT DINO with per-view trilinear coverage from depth.")

    parser.add_argument("--render_root", type=str,
                        default="../../../depth_renderer/datasets/ShapeNetRenderings")
    parser.add_argument("--tsdf_root", type=str,
                        default="../../../PatchComplete/data/shapenet")
    parser.add_argument("--camera_root", type=str,
                        default="../../../depth_renderer/datasets/camera_settings")
    parser.add_argument("--list_file", type=str, required=True)

    parser.add_argument("--grid_sizes", type=int, nargs="+", default=[32],
                        help="Voxel grid sizes to process (must exist as dino_features_gt_{G}.npz).")

    parser.add_argument("--views", type=int, nargs="+", default=[5, 6, 7, 8],
                        help="View ids to generate partials from (depth_{vid:03d}.exr). Example: 5 6 7 8.")

    parser.add_argument("--patch_grid", type=int, default=32,
                        help="Patch grid size Hp=Wp. For ViT-L/16 at 512x512, this is 32.")

    parser.add_argument("--and_tsdf_inside", action="store_true", default=True,
                        help="AND per-view coverage with GT TSDF inside mask. Default: enabled.")
    parser.add_argument("--no_and_tsdf_inside", action="store_false", dest="and_tsdf_inside",
                        help="Disable AND with TSDF inside mask.")

    parser.add_argument("--tsdf_res", type=int, default=32,
                        help="Multiply stored gt tsdf by this.")
    parser.add_argument("--tsdf_trunc", type=float, default=3.0,
                        help="Clip scaled gt tsdf to [-trunc, trunc].")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dino_features_{input_id}_{G}.npz")
    parser.add_argument("--dry_run_one", type=str, default=None,
                        help="Only process this one relative id, e.g. 03325088/61a58b...")
    parser.add_argument("--max_shapes", type=int, default=None,
                        help="Optional: limit number of shapes for debugging.")

    # multiprocessing / multi-gpu harness (even though compute is mostly CPU)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPU ids (used only for process placement). Use -1 for CPU worker.")
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--model", type=str, default="dinov3_vitl16",
                        choices=list(MODEL_TO_NUM_LAYERS.keys()))

    parser.add_argument("--tsdf_filter_value", type=float, nargs="+", default=[0.0], help="TSDF filter. Example: --tsdf_filter 0.0 OR --tsdf_filter -0.1 0.1" )
    parser.add_argument("--dataset_tag", type=str, default="")

    args = parser.parse_args()

    rels = load_shape_list(args.list_file)
    if args.dry_run_one is not None:
        rels = [args.dry_run_one]
        print(f"[Only one shape] {rels[0]}")
    if args.max_shapes is not None:
        rels = rels[:args.max_shapes]
        print(f"[Limit] max_shapes={args.max_shapes}")

    num_workers = args.num_workers if args.num_workers is not None else len(args.gpus)
    num_workers = max(1, num_workers)

    gpu_cycle = [args.gpus[i % len(args.gpus)] for i in range(num_workers)]
    chunks = chunk_list(rels, num_workers)

    print(f"[Shapes] {len(rels)}")
    print(f"[Workers] {num_workers}  gpu_cycle={gpu_cycle}")
    print(f"[Grids] {args.grid_sizes}")
    print(f"[Views] {args.views}  (saves input_id = view_id-1)")
    print(f"[PatchGrid] Hp=Wp={args.patch_grid}")
    print(f"[AND TSDF inside] {args.and_tsdf_inside}  (tsdf_res={args.tsdf_res}, tsdf_trunc={args.tsdf_trunc})")
    print(f"[Overwrite] {args.overwrite}")

    args_dict = vars(args)

    mp.set_start_method("spawn", force=True)
    procs = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_main,
            args=(wid, gpu_cycle[wid], chunks[wid], args_dict),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise SystemExit(f"Some workers failed, exitcodes={bad}")

    print("Done.")


if __name__ == "__main__":
    main()