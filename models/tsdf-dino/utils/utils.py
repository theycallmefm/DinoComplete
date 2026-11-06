import numpy as np
import torch
import random

def apply_mirror_pts(pts_xyznp, flip_type, res):
    # pts: (N,3) = (x,y,z) in voxel space
    x, y, z = pts_xyznp[:,0], pts_xyznp[:,1], pts_xyznp[:,2]
    if flip_type == 1:           # flip X
        x = (res - 1) - x
    elif flip_type == 2:         # flip Z
        z = (res - 1) - z
    elif flip_type == 3:         # flip X and Z
        x = (res - 1) - x
        z = (res - 1) - z
    out = np.stack([x, y, z], axis=-1)
    return out

def apply_transpose_pts(pts_xyznp, idx):
    # your transposes: [[0,1,2], [2,1,0]]  -> either identity or swap X<->Z
    if idx == 1:
        x, y, z = pts_xyznp[:,0], pts_xyznp[:,1], pts_xyznp[:,2]
        out = np.stack([z, y, x], axis=-1)
        return out
    return pts_xyznp

def apply_jitter_pts(pts_xyznp, shifts, res):
    i, j, k = shifts  # (dx, dy, dz)
    out = pts_xyznp.copy()
    out[:, 0] += i
    out[:, 1] += j
    out[:, 2] += k
    # clamp to valid voxel cube just like you clamp volumes by padding with truncation
    np.clip(out, 0, res - 1, out=out)
    return out


def get_mirror_params():
    return random.randint(0, 3)

def apply_mirror(x, flip_type):
    if flip_type == 0:
        return x
    elif flip_type == 1:
        return torch.flip(x, [1])  # Flip X
    elif flip_type == 2:
        return torch.flip(x, [3])  # Flip Z
    else:
        return torch.flip(x, [1, 3])  # Flip X and Z

def get_transpose_params():
    trans_list = [[0, 1, 2], [2, 1, 0]]
    return random.randint(0, len(trans_list) - 1)

def apply_transpose(x, idx):
    trans_list = [[0, 1, 2], [2, 1, 0]]
    axis_order = [0] + [i + 1 for i in trans_list[idx]]
    return x.permute(axis_order)

def get_jitter_params(jitter_range):
    return [random.randint(-jitter_range, jitter_range) for _ in range(3)]

def apply_jitter(x, shifts, truncation):
    i, j, k = shifts
    dst = torch.full_like(x, truncation)

    def get_indices(shift, size):
        if shift >= 0:
            return shift, size, 0, size - shift
        else:
            return 0, size + shift, -shift, size

    xidx = get_indices(i, x.shape[1])
    yidx = get_indices(j, x.shape[2])
    zidx = get_indices(k, x.shape[3])

    dst[:, xidx[0]:xidx[1], yidx[0]:yidx[1], zidx[0]:zidx[1]] = x[:, xidx[2]:xidx[3], yidx[2]:yidx[3], zidx[2]:zidx[3]]
    return dst


def jitter(src, tgt, jitter, truncation, tgtTruncation):
    """
    Applies random spatial jittering (shifts) to `src` and `tgt`.

    Args:
        src (torch.Tensor): Source SDF of shape (1, 32, 32, 32).
        tgt (torch.Tensor): Target SDF of shape (1, 32, 32, 32).
        jitter (int): Maximum shift range in each dimension.
        truncation (float): Truncation limit for src.
        tgtTruncation (float): Truncation limit for tgt.

    Returns:
        torch.Tensor, torch.Tensor: Jittered src and tgt.
    """
    dst = torch.full_like(src, truncation)
    tgtdst = torch.full_like(tgt, tgtTruncation)

    # Generate random shifts
    i = random.randint(-jitter, jitter)
    j = random.randint(-jitter, jitter)
    k = random.randint(-jitter, jitter)

    # Compute slicing indices
    def get_indices(shift, size):
        if shift >= 0:
            return shift, size, 0, size - shift
        else:
            return 0, size + shift, -shift, size

    xidx = get_indices(i, src.shape[1])
    yidx = get_indices(j, src.shape[2])
    zidx = get_indices(k, src.shape[3])

    # Apply jitter
    dst[:, xidx[0]:xidx[1], yidx[0]:yidx[1], zidx[0]:zidx[1]] = src[:, xidx[2]:xidx[3], yidx[2]:yidx[3], zidx[2]:zidx[3]]
    tgtdst[:, xidx[0]:xidx[1], yidx[0]:yidx[1], zidx[0]:zidx[1]] = tgt[:, xidx[2]:xidx[3], yidx[2]:yidx[3], zidx[2]:zidx[3]]

    return dst, tgtdst

def transpose_axis(inputs, labels, trans_list):
    """
    Randomly transposes the input and label along predefined axes.
    
    Args:
        inputs (torch.Tensor): Input tensor of shape (1, 32, 32, 32).
        labels (torch.Tensor): Label tensor of shape (1, 32, 32, 32).
        trans_list (list): List of axis permutations.

    Returns:
        torch.Tensor, torch.Tensor: Transposed input and label.
    """
    pick = random.randint(0, len(trans_list) - 1)
    axis_order = [0] + [x + 1 for x in trans_list[pick]]  # Adjust for 3D tensors

    return inputs.permute(axis_order), labels.permute(axis_order)

def mirror(inputs, labels):
    """
    Randomly mirrors (flips) the input and label along different axes.

    Args:
        inputs (torch.Tensor): Input tensor of shape (1, 32, 32, 32).
        labels (torch.Tensor): Label tensor of shape (1, 32, 32, 32).

    Returns:
        torch.Tensor, torch.Tensor: Flipped input and label.
    """
    pick = random.randint(0, 3)
    if pick == 0:
        return inputs, labels
    elif pick == 1:
        return torch.flip(inputs, [1]), torch.flip(labels, [1])  # Flip X-axis
    elif pick == 2:
        return torch.flip(inputs, [3]), torch.flip(labels, [3])  # Flip Z-axis
    else:
        return torch.flip(inputs, [1, 3]), torch.flip(labels, [1, 3])  # Flip both X and Z

def add_walls(inputs, bbox):
    """
    Adds artificial walls at the bounding box edges using [cx, cy, cz, sx, sy, sz] format.

    Args:
        inputs (torch.Tensor): Input tensor of shape (1, D, H, W).
        bbox (torch.Tensor): Bounding box tensor of shape (6,) = [cx, cy, cz, sx, sy, sz].

    Returns:
        torch.Tensor: Modified input tensor with walls.
    """
    assert bbox.shape[-1] == 6, f"Expected bbox shape [6], got {bbox.shape}"

    cx, cy, cz, sx, sy, sz = bbox.tolist()

    # Recompute min and max corners
    x1 = int(round(cx - sx / 2))
    x2 = int(round(cx + sx / 2)) - 1
    y1 = int(round(cy - sy / 2))
    y2 = int(round(cy + sy / 2)) - 1
    z1 = int(round(cz - sz / 2))
    z2 = int(round(cz + sz / 2)) - 1

    pick = random.randint(0, 5)

    if pick == 0:
        inputs[:, x1, :, :] = -0.5  # X min
    elif pick == 1:
        inputs[:, x2, :, :] = -0.5  # X max
    elif pick == 2:
        inputs[:, :, y1, :] = -0.5  # Y min
    elif pick == 3:
        inputs[:, :, y2, :] = -0.5  # Y max
    elif pick == 4:
        inputs[:, :, :, z1] = -0.5  # Z min
    elif pick == 5:
        inputs[:, :, :, z2] = -0.5  # Z max

    return inputs

def apply_mirror_feats(feats: torch.Tensor, flip_type: int) -> torch.Tensor:
    """
    feats: [C, X, Y, Z]
    flip_type: same as apply_mirror (0..3)
    """
    if flip_type == 0:
        return feats
    elif flip_type == 1:
        # flip X -> axis 1
        return torch.flip(feats, [1])
    elif flip_type == 2:
        # flip Z -> axis 3
        return torch.flip(feats, [3])
    else:
        # flip X and Z
        return torch.flip(feats, [1, 3])


def apply_transpose_feats(feats: torch.Tensor, idx: int) -> torch.Tensor:
    """
    feats: [C, X, Y, Z]
    idx: same as get_transpose_params (0: identity, 1: swap X<->Z)
    """
    trans_list = [[0, 1, 2], [2, 1, 0]]
    axis_order = [0] + [i + 1 for i in trans_list[idx]]  # channel stays first
    return feats.permute(axis_order)


def apply_jitter_feats(feats: torch.Tensor, shifts) -> torch.Tensor:
    """
    feats: [C, X, Y, Z]
    shifts: (dx, dy, dz) integer voxel shifts
    Fill with zeros where we "pad".
    """
    i, j, k = shifts
    dst = torch.zeros_like(feats)

    def get_indices(shift, size):
        if shift >= 0:
            return shift, size, 0, size - shift
        else:
            return 0, size + shift, -shift, size

    xidx = get_indices(i, feats.shape[1])
    yidx = get_indices(j, feats.shape[2])
    zidx = get_indices(k, feats.shape[3])

    dst[:, xidx[0]:xidx[1], yidx[0]:yidx[1], zidx[0]:zidx[1]] = \
        feats[:, xidx[2]:xidx[3], yidx[2]:yidx[3], zidx[2]:zidx[3]]
    return dst
