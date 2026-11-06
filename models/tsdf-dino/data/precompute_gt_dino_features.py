#!/usr/bin/env python3
"""
Precompute GT fused DINOv3 voxel features for ShapeNet renderings (PatchComplete-style),
MULTIPROCESS / MULTI-GPU, with:

TRILINEAR weighted splatting voxelization
FILTERING BY GT TSDF "inside" mask (gt_tsdf <= 0) resampled to G^3
Fusion across 20 views with weight-sum weighted mean
Saves:
    tsdf_root/cat/shape/dino_features_gt_{G}.npz
      - data:  [C,G,G,G] float32
      - count: [G,G,G]   float32   (weight_sum after TSDF filtering)

Example:
  python precompute_gt_dino_features.py \
    --render_root /../../depth_renderer/datasets/ShapeNetRenderings \
    --tsdf_root   /../../PatchComplete/data/shapenet \
    --camera_root /../../depth_renderer/datasets/camera_settings \
    --list_file   /../../data/data_splits/shapenet_test.txt \
    --model dinov3_vits16 \
    --grid_sizes 32 \
    --gpus 0 1 2 4 5 \
    --num_workers 5

Notes:
- TSDF mask is computed from tsdf_root/cat/shape/gt.npz key "tsdf".
- Mask mode is "inside": tsdf <= 0.0 (after scaling+clipping).
- Faster DINO: we only get last layer via n=1.
"""

import os
import argparse
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

import multiprocessing as mp


# -----------------------------
# Canonical cube convention
# -----------------------------
CAN_MIN = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
CAN_MAX = np.array([ 0.5,  0.5,  0.5], dtype=np.float32)
CAN_RANGE = CAN_MAX - CAN_MIN  # [1,1,1]

import numpy as np

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
# DINO preprocessing
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def preprocess_for_dino_no_resize(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    x = TF.to_tensor(pil_img)
    x = TF.normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return x.unsqueeze(0).to(device)


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
) -> Tuple[np.ndarray, np.ndarray]:
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
                patch_points[i, j, 0] = np.nanmean(block[..., 0])
                patch_points[i, j, 1] = np.nanmean(block[..., 1])
                patch_points[i, j, 2] = np.nanmean(block[..., 2])

    return patch_points, patch_valid


# -----------------------------
# Trilinear weighted splatting voxelization
# -----------------------------
def voxelize_view_trilinear(
    patch_feats_t: torch.Tensor,   # [C,Hp,Wp]
    dist_np: np.ndarray,           # [H,W]
    K: np.ndarray,                 # [3,3]
    RT: np.ndarray,                # [3,4]
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    patch_feats = patch_feats_t.detach().cpu().numpy().astype(np.float32)
    C, Hp, Wp = patch_feats.shape

    patch_points, patch_valid = patch_centers_in_canonical(dist_np, K, RT, Hp, Wp)

    pts_norm = (patch_points - CAN_MIN[None, None, :]) / CAN_RANGE[None, None, :]
    pts_norm = np.clip(pts_norm, 0.0, 1.0)

    G = int(grid_size)
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

    voxel_sum = np.zeros((C, G, G, G), dtype=np.float32)
    weight_sum = np.zeros((G, G, G), dtype=np.float32)

    feats_flat = patch_feats.reshape(C, Hp * Wp)
    valid_flat = patch_valid.reshape(Hp * Wp)

    x0f = x0.reshape(Hp * Wp); y0f = y0.reshape(Hp * Wp); z0f = z0.reshape(Hp * Wp)
    x1f = x1.reshape(Hp * Wp); y1f = y1.reshape(Hp * Wp); z1f = z1.reshape(Hp * Wp)

    w000f = w000.reshape(Hp * Wp); w100f = w100.reshape(Hp * Wp)
    w010f = w010.reshape(Hp * Wp); w110f = w110.reshape(Hp * Wp)
    w001f = w001.reshape(Hp * Wp); w101f = w101.reshape(Hp * Wp)
    w011f = w011.reshape(Hp * Wp); w111f = w111.reshape(Hp * Wp)

    def splat(n: int, gx: int, gy: int, gz: int, w: float):
        voxel_sum[:, gx, gy, gz] += feats_flat[:, n] * w
        weight_sum[gx, gy, gz] += w

    for n in range(Hp * Wp):
        if not valid_flat[n]:
            continue

        gx0, gy0, gz0 = int(x0f[n]), int(y0f[n]), int(z0f[n])
        gx1, gy1, gz1 = int(x1f[n]), int(y1f[n]), int(z1f[n])

        w = float(w000f[n]); 
        if w > 0.0: splat(n, gx0, gy0, gz0, w)
        w = float(w100f[n]); 
        if w > 0.0: splat(n, gx1, gy0, gz0, w)
        w = float(w010f[n]); 
        if w > 0.0: splat(n, gx0, gy1, gz0, w)
        w = float(w110f[n]); 
        if w > 0.0: splat(n, gx1, gy1, gz0, w)
        w = float(w001f[n]); 
        if w > 0.0: splat(n, gx0, gy0, gz1, w)
        w = float(w101f[n]); 
        if w > 0.0: splat(n, gx1, gy0, gz1, w)
        w = float(w011f[n]); 
        if w > 0.0: splat(n, gx0, gy1, gz1, w)
        w = float(w111f[n]); 
        if w > 0.0: splat(n, gx1, gy1, gz1, w)

    voxel_avg = voxel_sum
    nz = weight_sum > 0
    if np.any(nz):
        voxel_avg[:, nz] /= weight_sum[nz][None, :]

    return voxel_avg.astype(np.float32), weight_sum.astype(np.float32)


# -----------------------------
# TSDF -> inside mask in G^3
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
# List utils
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


# -----------------------------
# DINO model loading
# -----------------------------
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

def load_dinov3_torchhub(model_name: str, device: torch.device, weights_path: str, repo_or_dir: str):
    model = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=model_name,
        source="local",
        weights=weights_path,
    )
    model.eval().to(device)
    return model


# -----------------------------
# Worker
# -----------------------------
def worker_main(worker_id: int, gpu_id: int, rels: List[str], args_dict: dict):
    torch.set_num_threads(1)
    args = argparse.Namespace(**args_dict)

    if gpu_id < 0 or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

    print(f"[Worker {worker_id}] device={device} shapes={len(rels)}")
    
    print(args.model, device, args.weights_path, args.repo_or_dir)

    model = load_dinov3_torchhub(
        args.model,
        device=device,
        weights_path=args.weights_path,
        repo_or_dir=args.repo_or_dir,
    )
    n_layers = MODEL_TO_NUM_LAYERS[args.model]
    use_last_only = not args.slow_all_layers

    # camera
    K = load_cam_K(args.camera_root)
    RTs: Dict[int, np.ndarray] = {}
    for vid in range(1, 21):
        rt_path = os.path.join(args.camera_root, "cam_RT", f"cam_RT_{vid:03d}.txt")
        if os.path.exists(rt_path):
            RTs[vid] = load_cam_RT(args.camera_root, vid)

    pbar = tqdm(rels, desc=f"Worker{worker_id}", position=worker_id, leave=True)
    for rel in pbar:
        cat, sid = rel.split("/")
        render_dir = os.path.join(args.render_root, cat, sid)
        tsdf_dir   = os.path.join(args.tsdf_root,  cat, sid)

        if not os.path.isdir(render_dir) or not os.path.isdir(tsdf_dir):
            continue

        # skip if exists
        if not args.overwrite:
            all_exist = True
            for G in args.grid_sizes:
                outp = os.path.join(tsdf_dir, f"{args.model}_gt_{G}_{args.dataset_tag}.npz")
                if not os.path.exists(outp):
                    all_exist = False
                    break
            if all_exist:
                continue

        # --- load GT TSDF once per shape, build masks per G ---
        try:
            gt_tsdf_32 = load_gt_tsdf_np(tsdf_dir, trunc=args.tsdf_trunc, res=args.tsdf_res)
        except Exception as e:
            print(f"[Worker {worker_id}] WARN: missing/bad gt.npz for {rel}: {e}")
            continue

        mask_G = {}
        for G in args.grid_sizes:
            mask_G[G] = tsdf_inside_mask_G(args, gt_tsdf_32, G=int(G), filter_value = args.tsdf_filter_value)  # bool [G,G,G]

        fused_sum = {G: None for G in args.grid_sizes}    # [C,G,G,G]
        fused_wsum = {G: None for G in args.grid_sizes}   # [G,G,G]
        used_views = 0

        with torch.inference_mode():
            autocast_ok = (device.type == "cuda")
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if autocast_ok else torch.no_grad()
            with ctx:
                for vid in range(1, 21):
                    if vid not in RTs:
                        continue

                    cpath = os.path.join(render_dir, f"color_{vid:03d}.png")
                    dpath = os.path.join(render_dir, f"depth_{vid:03d}.exr")
                    if not (os.path.exists(cpath) and os.path.exists(dpath)):
                        continue

                    img = Image.open(cpath).convert("RGB")
                    if img.size != (512, 512):
                        img = img.resize((512, 512), Image.BICUBIC)

                    try:
                        dist = read_exr_dist(dpath)
                    except Exception:
                        continue

                    x = preprocess_for_dino_no_resize(img, device=device)

                    # last layer patch grid
                    if use_last_only:
                        outs = model.get_intermediate_layers(x, n=1, reshape=True, norm=True)
                        patch_feats = outs[0].squeeze(0).float()  # [C,Hp,Wp]
                    else:
                        outs = model.get_intermediate_layers(x, n=range(n_layers), reshape=True, norm=True)
                        patch_feats = outs[-1].squeeze(0).float()  # [C,Hp,Wp]

                    RT = RTs[vid]

                    for G in args.grid_sizes:
                        G = int(G)
                        vf, wsum = voxelize_view_trilinear(patch_feats, dist, K, RT, grid_size=G)

                        # --- APPLY GT TSDF INSIDE MASK HERE ---
                        m = mask_G[G].astype(np.float32)       # [G,G,G]
                        vf = vf * m[None, ...]
                        wsum = wsum * m

                        if fused_sum[G] is None:
                            fused_sum[G] = np.zeros_like(vf, dtype=np.float32)
                            fused_wsum[G] = np.zeros_like(wsum, dtype=np.float32)

                        fused_sum[G] += vf * wsum[None, ...]
                        fused_wsum[G] += wsum

                    used_views += 1

        if used_views == 0:
            continue

        if args.dry_run:
            continue

        # finalize and save
        for G in args.grid_sizes:
            G = int(G)
            w = fused_wsum[G]
            feats_avg = fused_sum[G] / (w[None, ...] + 1e-8)

            out_path = os.path.join(tsdf_dir, f"{args.model}_gt_{G}_{args.dataset_tag}.npz")
            np.savez_compressed(out_path, data=feats_avg.astype(np.float32), count=w.astype(np.float32))

    print(f"[Worker {worker_id}] done.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser("Precompute GT fused DINOv3 features (trilinear + TSDF-inside filter) with multiprocessing.")

    parser.add_argument("--render_root", type=str,
                        default="/../../../depth_renderer//datasets/ShapeNetRenderings")
    parser.add_argument("--tsdf_root", type=str,
                        default="/../../../PatchComplete/data/shapenet")
    parser.add_argument("--camera_root", type=str,
                        default="/../../../depth_renderer/datasets/camera_settings")
    parser.add_argument("--list_file", type=str, required=True)

    parser.add_argument("--model", type=str, default="dinov3_vits16",
                        choices=list(MODEL_TO_NUM_LAYERS.keys()))
    parser.add_argument("--grid_sizes", type=int, nargs="+", default=[32])

    parser.add_argument("--tsdf_res", type=int, default=32, help="Multiply stored gt tsdf PatchComplete resolution.")
    parser.add_argument("--tsdf_trunc", type=float, default=3.0, help="Clip scaled gt tsdf to [-trunc, trunc].")
    parser.add_argument("--tsdf_filter_value", type=float, nargs="+", default=[0.0], help="TSDF filter. Example: --tsdf_filter 0.0 OR --tsdf_filter -0.1 0.1" )
    parser.add_argument("--dataset_tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_one", type=str, default=None)
    parser.add_argument("--max_shapes", type=int, default=None)

    parser.add_argument("--repo_or_dir", type=str, default="/fm/Projects/dino-tsdf/dinov3")
    parser.add_argument("--weights_path", type=str,
                        default="/../../../dino-tsdf/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")

    parser.add_argument("--gpus", type=int, nargs="+", default=[0],
                        help="GPU ids. Example: --gpus 0 1 2 4 5. Use -1 for CPU worker.")
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--slow_all_layers", action="store_true", default=False,
                        help="Use n=range(n_layers) and take last (slower). Default uses n=1 (faster).")

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
    print(f"[TSDF] inside-mask using gt.npz with tsdf_res={args.tsdf_res}, tsdf_trunc={args.tsdf_trunc}")
    if args.dry_run:
        print("[Mode] DRY RUN (no files written)")

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