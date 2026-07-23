/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_DEQUANTIZE_MXFP8_CUTEDSL_CUH_

#include <transformer_engine/transformer_engine.h>

#include <cstddef>

#include "../../common.h"
#include "dequantize_mxfp8_cutedsl.cuh"

namespace transformer_engine {
namespace cutedsl_backend {

// Grouped MXFP8 dequantize on the CuTeDSL backend.
//
// Same reuse argument as mxfp8_group_quantize_cutedsl: for the `is_single_tensor`
// shape representations (SAME_BOTH_DIMS, VARYING_FIRST_DIM) every tensor shares
// the last dim and has a 128-aligned first dim, so the group's FP8 data and its
// E8M0 scales stack contiguously into exactly the layout a single 2D
// [first_logical_dim, last_logical_dim] tensor would have (the CUDA grouped
// kernel likewise uses scales_base_offset == 0 and a uniform scale_stride for
// these reps). So we flatten and reuse the single-tensor CuTeDSL dequantize.
//
// VARYING_LAST_DIM / VARYING_BOTH_DIMS need per-group TMA descriptors and fall
// back to the CUDA grouped kernel.
inline bool mxfp8_group_dequantize_cutedsl(const GroupedTensor *input, GroupedTensor *output,
                                           cudaStream_t stream) {
  const bool same_both = input->all_same_shape();
  const bool same_first = input->all_same_first_dim();
  const bool same_last = input->all_same_last_dim();
  const bool is_single_tensor = same_both || (!same_first && same_last);
  if (!is_single_tensor) {
    return false;
  }

  // The single-tensor kernel dequantizes exactly one direction; mirror the CUDA
  // grouped kernel's requirement that exactly one of them is present.
  const bool rowwise = input->has_data();
  const bool colwise = input->has_columnwise_data();
  if (rowwise == colwise) {
    return false;  // neither, or both -> let the CUDA kernel raise/handle it
  }
  if (input->with_gemm_swizzled_scales) {
    return false;  // grouped path requires compact scales
  }

  const size_t M = input->logical_shape.data[0];
  const size_t N = input->logical_shape.data[1];

  // Flattened 2D views over the contiguous grouped buffers.
  Tensor in2d;
  in2d.scaling_mode = NVTE_MXFP8_1D_SCALING;
  Tensor out2d;
  out2d.data = SimpleTensor(output->data.dptr, {M, N}, output->dtype());

  if (rowwise) {
    const size_t sY = DIVUP_TO_MULTIPLE(M, scale_tensor_alignment_Y_rowwise);
    const size_t sX =
        DIVUP_TO_MULTIPLE(DIVUP(N, static_cast<size_t>(32)), scale_tensor_alignment_X_rowwise);
    in2d.data = SimpleTensor(input->data.dptr, {M, N}, input->dtype());
    in2d.scale_inv = SimpleTensor(input->scale_inv.dptr, {sY, sX}, DType::kFloat8E8M0);
  } else {
    const size_t sY =
        DIVUP_TO_MULTIPLE(DIVUP(M, static_cast<size_t>(32)), scale_tensor_alignment_Y_colwise);
    const size_t sX = DIVUP_TO_MULTIPLE(N, scale_tensor_alignment_X_colwise);
    in2d.columnwise_data = SimpleTensor(input->columnwise_data.dptr, {M, N}, input->dtype());
    in2d.columnwise_scale_inv =
        SimpleTensor(input->columnwise_scale_inv.dptr, {sY, sX}, DType::kFloat8E8M0);
  }

  return mxfp8_dequantize_cutedsl(in2d, &out2d, stream);
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
