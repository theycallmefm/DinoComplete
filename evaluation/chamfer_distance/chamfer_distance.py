import torch
from torch.utils.cpp_extension import load

# Build the extension from the CUDA and binding sources.
# Adjust paths if this file lives elsewhere.
cd = load(
    name="cd",
    sources=[
        "chamfer_distance/binding.cpp",
        "chamfer_distance/chamfer_distance.cu",
    ],
    extra_cflags=[
        "-O3",
        "-std=c++17",
        # Keep/remove to match your PyTorch build. Your env uses 0.
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
    ],
    verbose=True,
)


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1: torch.Tensor, xyz2: torch.Tensor):
        """
        xyz1: [B, N, 3] float32 CUDA
        xyz2: [B, M, 3] float32 CUDA
        returns:
          dist1: [B, N] float32 CUDA
          dist2: [B, M] float32 CUDA
        """
        # Safety checks
        if not (xyz1.is_cuda and xyz2.is_cuda):
            raise RuntimeError("ChamferDistance only supports CUDA tensors.")
        if xyz1.dtype != torch.float32 or xyz2.dtype != torch.float32:
            raise RuntimeError("Inputs must be float32.")
        if xyz1.dim() != 3 or xyz2.dim() != 3 or xyz1.size(-1) != 3 or xyz2.size(-1) != 3:
            raise RuntimeError("xyz1, xyz2 must be shaped [B, N, 3] and [B, M, 3].")

        # Ensure contiguous memory
        xyz1c = xyz1.contiguous()
        xyz2c = xyz2.contiguous()

        # The C++ forward returns four tensors
        dist1, idx1, dist2, idx2 = cd.forward(xyz1c, xyz2c)

        # Save for backward
        ctx.save_for_backward(xyz1c, xyz2c, idx1, idx2)

        # By design, the module returns only distances
        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1: torch.Tensor, grad_dist2: torch.Tensor):
        """
        grad_dist1: [B, N] float32 CUDA
        grad_dist2: [B, M] float32 CUDA
        returns grads for inputs of forward: grad_xyz1, grad_xyz2
        """
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        if grad_dist1 is None or grad_dist2 is None:
            # Shouldn’t happen in normal autograd usage, but be robust
            return torch.zeros_like(xyz1), torch.zeros_like(xyz2)

        g1 = grad_dist1.contiguous()
        g2 = grad_dist2.contiguous()

        # Call CUDA backward implemented in binding.cpp/.cu
        grad_xyz1, grad_xyz2 = cd.backward(xyz1, xyz2, idx1, idx2, g1, g2)
        return grad_xyz1, grad_xyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor):
        # Returns (dist1, dist2); indices are internal for backward
        return ChamferDistanceFunction.apply(xyz1, xyz2)
