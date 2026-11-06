import os
import torch
from models.create_hilbert_curve_template import convert_to_index


def hilbert_order_3d(D: int, H: int, W: int, device="cpu") -> torch.Tensor:
    """
    Create Hilbert order (permutation indices) for a 3D grid of size (D,H,W).
    Assumes a cubic, power-of-two grid: D == H == W == 2^n.

    Returns:
        order: LongTensor of length D*H*W, such that:
            flat_hilbert = x_flat[:, order, :]
    """
    assert D == H == W, f"Hilbert 3D requires cubic grid, got {(D,H,W)}"
    N = D
    # Check power of two
    assert (N & (N - 1)) == 0, f"N={N} must be a power of 2 for Hilbert."

    # num_bits = log2(N)
    num_bits = int(torch.log2(torch.tensor(N)).item())

    # Build coordinates (z,y,x) over the whole grid
    z, y, x = torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
    )
    coords = torch.stack([z, y, x], dim=-1).view(-1, 3).to(device)

    # Use the original convert_to_index from the Voxel-Mamba script
    index = convert_to_index(coords, num_dims=3, num_bits=num_bits)

    # Sort coordinates by Hilbert index → permutation vector
    _, order = torch.sort(index)
    return order  # [D*H*W] long tensor
