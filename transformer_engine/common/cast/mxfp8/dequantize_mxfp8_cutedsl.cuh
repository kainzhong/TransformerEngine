/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_DEQUANTIZE_MXFP8_CUTEDSL_CUH_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <optional>
#include <string>

#include "../../common.h"
#include "../../tvm_ffi_bridge.h"

namespace transformer_engine {
namespace cutedsl_backend {

// te_dtype_to_str, DLTensorWrapper, TVMFFICentral all live in
// transformer_engine::tvm_ffi_bridge (tvm_ffi_bridge.h).
using namespace tvm_ffi_bridge;

struct MXFP8DequantConfig {
  static constexpr const char *kEntrypointName = "get_mxfp8_dequantization_function";

  DType out_dtype;  // The higher-precision output format
  DType fp8_dtype;  // The fp8 input format
  bool rowwise;     // If dequantize using the rowwise data + scales
  bool colwise;     // If dequantize using the columnwise data + scales
  bool swizzled;    // If the input scales use cuBLAS's swizzled layout

  std::string to_key() const {
    std::string key;
    key.reserve(48);
    key.append("cutedsl_mxfp8_dequantize_")
        .append(te_dtype_to_str(out_dtype))
        .append("_")
        .append(te_dtype_to_str(fp8_dtype))
        .append("_")
        .append(rowwise ? "1" : "0")
        .append("_")
        .append(colwise ? "1" : "0")
        .append("_")
        .append(swizzled ? "1" : "0");
    return key;
  }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    auto entrypoint = tvm::ffi::Function::GetGlobal(kEntrypointName);
    if (!entrypoint.has_value()) {
      return false;
    }
    tvm::ffi::Any result =
        (*entrypoint)(tvm::ffi::String(fn_name), tvm::ffi::String(te_dtype_to_str(out_dtype)),
                      tvm::ffi::String(te_dtype_to_str(fp8_dtype)), rowwise, colwise, swizzled);
    return result.try_cast<bool>().value_or(false);
  }
};

// Mirrors mxfp8::dequantize (dequantize_mxfp8.cuh). Returns false to fall back to
// the CUDA kernel (unsupported config, misaligned shape, or missing Python entry).
inline bool mxfp8_dequantize_cutedsl(const Tensor &input, Tensor *output, cudaStream_t stream) {
  constexpr size_t kCuTeDSLMXFP8ShapeAlignment = 32;

  const bool rowwise = input.has_data();
  const bool colwise = input.has_columnwise_data();
  NVTE_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions.");
  if (rowwise) {
    NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  }
  if (colwise) {
    NVTE_CHECK(is_fp8_dtype(input.columnwise_data.dtype), "Input must have FP8 type.");
  }
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->shape() == input.shape(), "Input and output shapes need to match.");

  const auto [flat_m, flat_n] = input.flat_2d_dims();
  if (flat_m % kCuTeDSLMXFP8ShapeAlignment != 0 || flat_n % kCuTeDSLMXFP8ShapeAlignment != 0) {
    return false;
  }

  const MXFP8DequantConfig config{/*out_dtype=*/output->dtype(),
                                  /*fp8_dtype=*/input.dtype(),
                                  /*rowwise=*/rowwise,
                                  /*colwise=*/colwise,
                                  /*swizzled=*/input.with_gemm_swizzled_scales};

  std::optional<tvm::ffi::Function> mxfp8_dequant_func_opt =
      tvm_ffi_bridge::TVMFFICentral::getInstance().lazyload_function(config);
  if (!mxfp8_dequant_func_opt.has_value()) {
    return false;
  }

  checkCuDriverContext(stream);

  // The kernel dequantizes exactly one direction and prefers rowwise when both
  // are present (matches the CUDA kernel's use_rowwise_scaling = has_data()).
  const SimpleTensor &input_data = rowwise ? input.data : input.columnwise_data;
  const SimpleTensor &scale_inv = rowwise ? input.scale_inv : input.columnwise_scale_inv;

  // Data tensors auto-flatten to 2D (DLTensorWrapper's default), matching the
  // kernel's flat (rows, cols) view; the 2D scale tensor passes through.
  tvm_ffi_bridge::DLTensorWrapper mX(input_data);
  tvm_ffi_bridge::DLTensorWrapper mO(output->data);
  tvm_ffi_bridge::DLTensorWrapper mS(scale_inv);
  // stream is a tvm-ffi opaque "handle"; pass the CUDA stream as void*.
  (*mxfp8_dequant_func_opt)(&mX, &mO, &mS, static_cast<void *>(stream));
  return true;
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_DEQUANTIZE_MXFP8_CUTEDSL_CUH_
