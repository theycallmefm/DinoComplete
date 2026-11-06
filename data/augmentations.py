import torch
import random

def get_transpose_params():
    trans_list = [[0, 1, 2], [2, 1, 0]]
    return random.randint(0, len(trans_list) - 1)

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