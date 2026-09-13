// SPDX-License-Identifier: Apache-2.0
// GLM53-only launch choices over SGLang's existing CUTLASS W4A8 mainloop.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "glm53_w4a8.h"

namespace {
void run(torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b,
         const torch::Tensor& as, const torch::Tensor& bs,
         const torch::Tensor& offsets, const torch::Tensor& problems,
         const torch::Tensor& astr, const torch::Tensor& bstr,
         const torch::Tensor& dstr, const torch::Tensor& sstr,
         int64_t group_size, int64_t variant) {
  TORCH_CHECK(a.is_cuda() && a.dim() == 2 && b.dim() == 3 && out.dim() == 2,
              "GLM53 W4A8 requires CUDA matrices and grouped packed weights");
  const c10::cuda::CUDAGuard guard(a.device());
  const auto* prop = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(prop->major == 9 && prop->minor == 0, "GLM53 W4A8 requires SM90");
  const auto k = a.size(1), n = out.size(1), rows = a.size(0);
  TORCH_CHECK(((n == 4096 && k == 6144) || (n == 6144 && k == 2048)) &&
              rows > 0 && rows <= 512 && rows % 8 == 0 &&
              out.size(0) == rows && b.size(0) == 32 && b.size(1) == n && b.size(2) * 2 == k &&
              group_size == 128, "Unsupported GLM53 decode GEMM geometry");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kInt8 &&
              out.scalar_type() == torch::kBFloat16 && bs.scalar_type() == torch::kBFloat16 &&
              as.scalar_type() == torch::kFloat32 && as.numel() == 1,
              "Expected static FP8 activations, signed INT4 and BF16 group scales");
  TORCH_CHECK(offsets.numel() == 32 && problems.sizes() == torch::IntArrayRef({32, 3}),
              "Expected 32 local expert groups");
  TORCH_CHECK(bs.dim() == 3 && bs.size(0) == 32 && bs.size(1) == k / 512 &&
              bs.size(2) == n * 4, "Expected original interleaved group-128 scales");
  for (const auto& t : {out, a, b, as, bs, offsets, problems, astr, bstr, dstr, sstr}) {
    TORCH_CHECK(t.is_cuda() && t.device() == a.device() && t.is_contiguous(),
                "All inputs must be contiguous tensors on the same CUDA device");
  }
  for (const auto& t : {astr, bstr, dstr, sstr}) {
    TORCH_CHECK(t.scalar_type() == torch::kInt64 && t.sizes() == torch::IntArrayRef({32, 3}),
                "Expected original [32,3] per-expert int64 strides");
  }
  // The original caller consumes actual GPU problem sizes, including empty
  // groups. Tile M is the output-channel dimension of the transposed GEMM;
  // tile N is its token dimension. K=512 preserves scale packing exactly.
#define CALL(ID) \
  glm53_cutlass::launch_variant_##ID( \
      out, a, b, as, bs, offsets, problems, astr, bstr, dstr, sstr, group_size)
  switch (variant) {
    case 1: CALL(1); break;
    case 2: CALL(2); break;
    case 3: CALL(3); break;
    case 4: CALL(4); break;
    case 5: CALL(5); break;
    case 6: CALL(6); break;
    case 7: CALL(7); break;
    default: TORCH_CHECK(false, "Unknown GLM53 CUTLASS variant: ", variant);
  }
#undef CALL
}
}  // namespace

TORCH_LIBRARY(glm53_cutlass, m) {
  m.def("w4a8_mm(Tensor(a!) out, Tensor a, Tensor b, Tensor a_scale, Tensor b_scale, "
        "Tensor offsets, Tensor problems, Tensor a_stride, Tensor b_stride, "
        "Tensor d_stride, Tensor s_stride, int group_size, int variant) -> ()");
}
TORCH_LIBRARY_IMPL(glm53_cutlass, CUDA, m) {
  m.impl("w4a8_mm", &run);
}
