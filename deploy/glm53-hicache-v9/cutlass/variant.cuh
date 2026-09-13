// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <type_traits>
#include "glm53_w4a8.h"
#include "moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh"

namespace {
template <int M, int N, int ClusterM, bool Pingpong>
using Gemm = cutlass_3x_w4a8_group_gemm<
    cute::Shape<cute::Int<M>, cute::Int<N>, cute::_512>,
    cute::Shape<cute::Int<ClusterM>, cute::_1, cute::_1>,
    std::conditional_t<Pingpong,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>,
    std::conditional_t<Pingpong,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>>;
}  // namespace

// Expand exactly once in each .cu. Keeping separate translation units releases
// compiler memory between variants; MAX_JOBS=1 prevents concurrent compilers.
#define GLM53_DEFINE_VARIANT(ID, M, N, C, PP) \
  void glm53_cutlass::launch_variant_##ID( \
      torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b, \
      const torch::Tensor& as, const torch::Tensor& bs, \
      const torch::Tensor& offsets, const torch::Tensor& problems, \
      const torch::Tensor& astr, const torch::Tensor& bstr, \
      const torch::Tensor& dstr, const torch::Tensor& sstr, int64_t group_size) { \
    cutlass_w4a8_group_gemm_caller<Gemm<M, N, C, PP>>( \
        out, a, b, as, bs, offsets, problems, astr, bstr, dstr, sstr, group_size); \
  }
