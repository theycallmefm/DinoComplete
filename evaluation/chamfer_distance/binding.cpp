#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// C-linkage launchers implemented in the .cu
extern "C" void ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i);

extern "C" void ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyz1,
    const int m,
    const float* xyz2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyz1,
    float* grad_xyz2);

// Forward: returns [dist1, idx1, dist2, idx2]
std::vector<at::Tensor> chamfer_forward(at::Tensor xyz1, at::Tensor xyz2) {
  TORCH_CHECK(xyz1.is_cuda() && xyz2.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(xyz1.scalar_type() == at::kFloat && xyz2.scalar_type() == at::kFloat,
              "Inputs must be float32");
  TORCH_CHECK(xyz1.dim() == 3 && xyz2.dim() == 3 && xyz1.size(2) == 3 && xyz2.size(2) == 3,
              "Expected xyz1, xyz2 of shape [B, N, 3] and [B, M, 3]");

  auto x1 = xyz1.contiguous();
  auto x2 = xyz2.contiguous();

  const int64_t B  = x1.size(0);
  const int64_t N1 = x1.size(1);
  const int64_t N2 = x2.size(1);

  auto opts_f = x1.options().dtype(at::kFloat);
  auto opts_i = x1.options().dtype(at::kInt);

  auto dist1 = at::empty({B, N1}, opts_f);
  auto idx1  = at::empty({B, N1}, opts_i);
  auto dist2 = at::empty({B, N2}, opts_f);
  auto idx2  = at::empty({B, N2}, opts_i);

  ChamferDistanceKernelLauncher(
      static_cast<int>(B), static_cast<int>(N1),
      x1.data_ptr<float>(),
      static_cast<int>(N2),
      x2.data_ptr<float>(),
      dist1.data_ptr<float>(),
      idx1.data_ptr<int>(),
      dist2.data_ptr<float>(),
      idx2.data_ptr<int>());

  return {dist1, idx1, dist2, idx2};
}

// Backward: returns [grad_xyz1, grad_xyz2]
std::vector<at::Tensor> chamfer_backward(
    at::Tensor xyz1, at::Tensor xyz2,
    at::Tensor idx1, at::Tensor idx2,
    at::Tensor grad_dist1, at::Tensor grad_dist2)
{
  TORCH_CHECK(xyz1.is_cuda() && xyz2.is_cuda(), "Inputs must be CUDA tensors");
  TORCH_CHECK(idx1.is_cuda() && idx2.is_cuda(), "Indices must be CUDA tensors");
  TORCH_CHECK(grad_dist1.is_cuda() && grad_dist2.is_cuda(), "Grad inputs must be CUDA tensors");
  TORCH_CHECK(xyz1.scalar_type() == at::kFloat && xyz2.scalar_type() == at::kFloat,
              "xyz must be float32");
  TORCH_CHECK(grad_dist1.scalar_type() == at::kFloat && grad_dist2.scalar_type() == at::kFloat,
              "grad_dist must be float32");
  TORCH_CHECK(idx1.scalar_type() == at::kInt && idx2.scalar_type() == at::kInt,
              "idx must be int32");

  auto x1   = xyz1.contiguous();
  auto x2   = xyz2.contiguous();
  auto g1   = grad_dist1.contiguous();
  auto g2   = grad_dist2.contiguous();
  auto i1   = idx1.contiguous();
  auto i2   = idx2.contiguous();

  const int64_t B  = x1.size(0);
  const int64_t N1 = x1.size(1);
  const int64_t N2 = x2.size(1);

  auto grad_x1 = at::empty_like(x1);
  auto grad_x2 = at::empty_like(x2);

  ChamferDistanceGradKernelLauncher(
      static_cast<int>(B), static_cast<int>(N1),
      x1.data_ptr<float>(),
      static_cast<int>(N2),
      x2.data_ptr<float>(),
      g1.data_ptr<float>(),
      i1.data_ptr<int>(),
      g2.data_ptr<float>(),
      i2.data_ptr<int>(),
      grad_x1.data_ptr<float>(),
      grad_x2.data_ptr<float>());

  return {grad_x1, grad_x2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward",  &chamfer_forward,  "Chamfer forward (CUDA)");
  m.def("backward", &chamfer_backward, "Chamfer backward (CUDA)");
}
