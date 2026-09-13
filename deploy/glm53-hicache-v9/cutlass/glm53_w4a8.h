// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <torch/types.h>

// No CUTLASS headers here: the dispatcher must not instantiate CUDA kernels.
namespace glm53_cutlass {
using Launcher = void(
    torch::Tensor out, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& as, const torch::Tensor& bs,
    const torch::Tensor& offsets, const torch::Tensor& problems,
    const torch::Tensor& astr, const torch::Tensor& bstr,
    const torch::Tensor& dstr, const torch::Tensor& sstr, int64_t group_size);

Launcher launch_variant_1;
Launcher launch_variant_2;
Launcher launch_variant_3;
Launcher launch_variant_4;
Launcher launch_variant_5;
Launcher launch_variant_6;
Launcher launch_variant_7;
}  // namespace glm53_cutlass
