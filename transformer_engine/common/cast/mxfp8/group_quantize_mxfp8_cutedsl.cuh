/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_QUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_QUANTIZE_MXFP8_CUTEDSL_CUH_

#include <transformer_engine/transformer_engine.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

using namespace tvm_ffi_bridge;

// Mirrors dispatch::common::MAX_SUPPORTED_TENSOR_DESCRIPTORS (grouped_tma.cuh).
constexpr size_t kMaxGroupTensorDescriptors = 64;
// One TMA descriptor is 128B; the kernel keeps 3 per tensor (input, rowwise out,
// colwise out).
constexpr size_t kBytesPerTensormap = 128;
constexpr size_t kNumTensormaps = 3;

/*! \brief Process-wide scratch for the per-tensor TMA descriptors.
 *
 *  The CUDA kernel keeps these in a `static __device__ TensorMapStorage
 *  g_tensor_maps` global; this is the same thing for the CuTeDSL kernel, with the
 *  same lifetime and the same (single-writer-per-launch) concurrency behaviour.
 *  Intentionally leaked -- it lives for the process, like the CUDA global.
 */
inline void *group_tensormap_workspace() {
  static void *ws = []() {
    void *ptr = nullptr;
    constexpr size_t bytes = kMaxGroupTensorDescriptors * kNumTensormaps * kBytesPerTensormap;
    NVTE_CHECK_CUDA(cudaMalloc(&ptr, bytes));
    return ptr;
  }();
  return ws;
}

struct MXFP8GroupQuantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_group_quantization_function";

  DType dtype;            // input format
  DType fp8_dtype;        // fp8 output format
  bool rowwise;           // quantize rowwise
  bool colwise;           // quantize columnwise
  const char *shape_rep;  // "same_both_dims" | "varying_first_dim" | "varying_last_dim"

  std::string to_key() const {
    std::string key;
    key.reserve(64);
    key.append("cutedsl_mxfp8_group_")
        .append(te_dtype_to_str(dtype))
        .append("_")
        .append(te_dtype_to_str(fp8_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
        .append("_")
        .append(shape_rep);
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(te_dtype_to_str(dtype)),
                      tvm::ffi::String(te_dtype_to_str(fp8_dtype)), rowwise, colwise,
                      tvm::ffi::String(shape_rep));
    return result.try_cast<bool>().value_or(false);
  }
};

/*! \brief Grouped MXFP8 quantize on the CuTeDSL backend.
 *
 *  Signature mirrors mxfp8::group_quantize's inputs. Returns false to fall back to
 *  the CUDA grouped kernel (unsupported config / shape rep / fusion).
 *
 *  The CuTeDSL kernel is cast-only with compact scales and covers the
 *  SAME_BOTH_DIMS, VARYING_FIRST_DIM and VARYING_LAST_DIM shape representations.
 */
template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
bool mxfp8_group_quantize_cutedsl(const GroupedTensor *input,
                                  const GroupedTensor * /*activations*/, const Tensor *noop_tensor,
                                  GroupedTensor *output, GroupedTensor * /*dbias*/,
                                  Tensor * /*workspace*/, cudaStream_t stream) {
  // The CuTeDSL grouped kernel is cast-only.
  if constexpr (IS_ACT) {
    return false;
  } else {
    const bool rowwise = output->has_data();
    const bool colwise = output->has_columnwise_data();
    if (!(rowwise || colwise)) {
      return false;
    }
    // Unsupported fusions / layouts: defer to CUDA so results can't drift.
    if (output->amax.dptr != nullptr) return false;
    if (output->with_gemm_swizzled_scales) return false;
    if (noop_tensor != nullptr && noop_tensor->data.dptr != nullptr) return false;

    // Shape representation, mirroring mxfp8::group_quantize's classification.
    const char *shape_rep = nullptr;
    bool is_single_tensor = true;
    if (output->all_same_shape()) {
      shape_rep = "same_both_dims";
    } else if (output->all_same_first_dim()) {
      shape_rep = "varying_last_dim";
      is_single_tensor = false;
    } else if (output->all_same_last_dim()) {
      shape_rep = "varying_first_dim";
    } else {
      return false;  // VARYING_BOTH_DIMS: logical shape [1, total] is not tileable.
    }

    const size_t num_tensors = input->num_tensors;
    const size_t M = input->logical_shape.data[0];
    const size_t N = input->logical_shape.data[1];

    // Contract of the compiled kernel (and of the CUDA kernel's own checks).
    if (M % 128 != 0 || N % 32 != 0) return false;
    if (!is_single_tensor && num_tensors > kMaxGroupTensorDescriptors) return false;
    if (input->tensor_offsets.dptr == nullptr) return false;
    if (rowwise) {
      NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
    }
    if (colwise) {
      NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
                 "Columnwise scaling tensor must be allocated");
    }

    const MXFP8GroupQuantConfig config{/*dtype=*/input->dtype(),
                                       /*fp8_dtype=*/output->dtype(),
                                       /*rowwise=*/rowwise,
                                       /*colwise=*/colwise,
                                       /*shape_rep=*/shape_rep};

    std::optional<tvm::ffi::Function> fn =
        tvm_ffi_bridge::TVMFFICentral::getInstance().lazyload_function(config);
    if (!fn.has_value()) {
      return false;
    }
    checkCuDriverContext(stream);

    // The kernel builds a TMA atom for every operand, so a direction that is not
    // produced still needs a valid stand-in buffer; it is never written (the
    // direction is a compile-time constant inside the kernel).
    const SimpleTensor &data_row = rowwise ? output->data : output->columnwise_data;
    const SimpleTensor &data_col = colwise ? output->columnwise_data : output->data;
    const SimpleTensor &scale_row = rowwise ? output->scale_inv : output->columnwise_scale_inv;
    const SimpleTensor &scale_col = colwise ? output->columnwise_scale_inv : output->scale_inv;

    // Data operands are viewed with the group's logical 2D shape; the scale buffers
    // are passed flat (the kernel derives per-tensor bases/strides itself).
    SimpleTensor in2d(input->data.dptr, {M, N}, input->dtype());
    SimpleTensor out_row2d(data_row.dptr, {M, N}, output->dtype());
    SimpleTensor out_col2d(data_col.dptr, {M, N}, output->dtype());
    SimpleTensor s_row1d(scale_row.dptr, {scale_row.numel()}, DType::kFloat8E8M0);
    SimpleTensor s_col1d(scale_col.dptr, {scale_col.numel()}, DType::kFloat8E8M0);
    SimpleTensor tmaps(group_tensormap_workspace(),
                       {num_tensors, kNumTensormaps, kBytesPerTensormap / 8}, DType::kInt64);
    // first_dims / last_dims are only read for the rep that varies that dim; pass
    // the offsets array as a same-dtype stand-in when absent.
    const SimpleTensor &first_dims =
        input->first_dims.has_data() ? input->first_dims : input->tensor_offsets;
    const SimpleTensor &last_dims =
        input->last_dims.has_data() ? input->last_dims : input->tensor_offsets;

    tvm_ffi_bridge::DLTensorWrapper mX(in2d), mO_row(out_row2d), mO_col(out_col2d),
        mS_row(s_row1d), mS_col(s_col1d), mOffsets(input->tensor_offsets), mFirstDims(first_dims),
        mLastDims(last_dims);
    // The descriptor scratch is rank 3; DLTensorWrapper would otherwise flatten it to 2D.
    tvm_ffi_bridge::DLTensorWrapper mTensormaps(tmaps, /*flatten_2D=*/false);

    // stream is a tvm-ffi opaque "handle"; pass the CUDA stream as void*.
    (*fn)(&mX, &mO_row, &mO_col, &mS_row, &mS_col, &mOffsets, &mFirstDims, &mLastDims, &mTensormaps,
          static_cast<void *>(stream));
    return true;
  }
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_GROUP_QUANTIZE_MXFP8_CUTEDSL_CUH_
