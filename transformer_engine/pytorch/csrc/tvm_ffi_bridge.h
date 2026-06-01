/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TVM_FFI_BRIDGE_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TVM_FFI_BRIDGE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <c10/core/ScalarType.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include "transformer_engine/transformer_engine.h"
#include "util/logging.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// dtype conversion helpers — overload resolution picks by argument type.
// ---------------------------------------------------------------------------

inline DLDataType convert_to_dltype(c10::ScalarType type) {
  switch (type) {
    case c10::ScalarType::Float:    return DLDataType{kDLFloat,  32, 1};
    case c10::ScalarType::Half:     return DLDataType{kDLFloat,  16, 1};
    case c10::ScalarType::BFloat16: return DLDataType{kDLBfloat, 16, 1};
    case c10::ScalarType::Byte:     return DLDataType{kDLUInt,    8, 1};
    case c10::ScalarType::Char:     return DLDataType{kDLInt,     8, 1};
    case c10::ScalarType::Int:      return DLDataType{kDLInt,    32, 1};
    case c10::ScalarType::Long:     return DLDataType{kDLInt,    64, 1};
    default: NVTE_ERROR("unsupported torch dtype for DLPack");
  }
}

inline DLDataType convert_to_dltype(NVTEDType type) {
  switch (type) {
    case kNVTEFloat32:    return DLDataType{kDLFloat,  32, 1};
    case kNVTEFloat16:    return DLDataType{kDLFloat,  16, 1};
    case kNVTEBFloat16:   return DLDataType{kDLBfloat, 16, 1};
    case kNVTEByte:       return DLDataType{kDLUInt,    8, 1};
    case kNVTEInt32:      return DLDataType{kDLInt,    32, 1};
    case kNVTEInt64:      return DLDataType{kDLInt,    64, 1};
    // FP8 / E8M0 → raw 1-byte uint; the kernel interprets the bits.
    case kNVTEFloat8E4M3: return DLDataType{kDLUInt,    8, 1};
    case kNVTEFloat8E5M2: return DLDataType{kDLUInt,    8, 1};
    case kNVTEFloat8E8M0: return DLDataType{kDLUInt,    8, 1};
    default: NVTE_ERROR("unsupported NVTEDType: ", static_cast<int>(type));
  }
}

// ---------------------------------------------------------------------------
// DLTensorWrapper — DLTensor with managed shape/strides storage.
//
// Subclassing DLTensor (a POD C struct) lets the wrapper IS-A DLTensor: you
// can take its address and pass it directly to `tvm::ffi::TensorView`. The
// shape/strides arrays the base struct points at are either borrowed from a
// PyTorch tensor (zero copy) or owned by the wrapper itself (when built
// from an NVTE tensor that doesn't store them in int64_t form).
// ---------------------------------------------------------------------------
class DLTensorWrapper : public DLTensor {
 public:
  // Zero-copy borrow via torch's own non-owning DLPack export: fills our
  // base DLTensor in place (data/shape/strides/dtype/device/byte_offset)
  // using torch's canonical field extraction — no heap alloc, no deleter,
  // no refcount. shape/strides point into the at::Tensor's internal arrays,
  // so the caller must keep `tensor` alive through any use of this wrapper.
  explicit DLTensorWrapper(const at::Tensor &tensor) {
    NVTE_CHECK(tensor.defined(), "DLTensorWrapper: undefined at::Tensor");
    at::toDLPackNonOwning(tensor, static_cast<DLTensor *>(this));
  }

  // NVTEBasicTensor stores shape as size_t and has no strides. We allocate
  // owned int64 buffers for both: copy the shape, synthesize row-major
  // contiguous strides (TE tensors are always contiguous).
  DLTensorWrapper(const NVTEBasicTensor &tensor, int32_t device_index) {
    const int n = static_cast<int>(tensor.shape.ndim);
    shape_buf_   = std::make_unique<int64_t[]>(n);
    strides_buf_ = std::make_unique<int64_t[]>(n);
    int64_t stride = 1;
    for (int i = n - 1; i >= 0; --i) {
      shape_buf_[i]   = static_cast<int64_t>(tensor.shape.data[i]);
      strides_buf_[i] = stride;
      stride *= shape_buf_[i];
    }
    this->data        = tensor.data_ptr;
    this->device      = DLDevice{kDLCUDA, device_index};
    this->ndim        = n;
    this->dtype       = convert_to_dltype(tensor.dtype);
    this->shape       = shape_buf_.get();
    this->strides     = strides_buf_.get();
    this->byte_offset = 0;
  }

  ~DLTensorWrapper() = default;
  DLTensorWrapper(const DLTensorWrapper &) = delete;
  DLTensorWrapper &operator=(const DLTensorWrapper &) = delete;
  DLTensorWrapper(DLTensorWrapper &&) = default;
  DLTensorWrapper &operator=(DLTensorWrapper &&) = default;

 private:
  std::unique_ptr<int64_t[]> shape_buf_;
  std::unique_ptr<int64_t[]> strides_buf_;
};

// ---------------------------------------------------------------------------
// applyTVMFunction — generic dispatcher.
//
// `fn_name` is the global-registry key for a `tvm::ffi::Function` (registered
// from Python via `tvm_ffi.register_global_func`). `args` is the positional
// arg list the kernel expects; entries that are `None` (std::nullopt) become
// TVM FFI `None`, useful for slots the AOT kernel was compiled with as None.
//
// C++ holds wrappers + TensorViews on the stack for the duration of the
// call so the AnyView pointers into them stay valid through CallPacked.
// ---------------------------------------------------------------------------
inline void applyTVMFunction(const std::string &fn_name,
                             const std::vector<std::optional<at::Tensor>> &args) {
  const size_t n = args.size();
  std::vector<std::optional<DLTensorWrapper>> wrappers(n);
  std::vector<std::optional<tvm::ffi::TensorView>> tvs(n);
  std::vector<tvm::ffi::AnyView> tvm_args(n);  // default ctor → kTVMFFINone
  for (size_t i = 0; i < n; ++i) {
    if (args[i].has_value() && args[i]->defined()) {
      wrappers[i].emplace(*args[i]);
      tvs[i].emplace(static_cast<DLTensor *>(&*wrappers[i]));
      tvm_args[i] = *tvs[i];
    }
  }
  auto fn = tvm::ffi::Function::GetGlobalRequired(fn_name);
  tvm::ffi::Any result;
  fn.CallPacked(tvm_args.data(), static_cast<int32_t>(n), &result);
}

inline void applyTVMFunction(const std::string &fn_name,
                             const std::vector<std::optional<DLTensorWrapper>> &args) {
  // Overload for callers that have already built `DLTensorWrapper`s. We
  // borrow them by const ref — no copy (DLTensorWrapper has copy deleted)
  // and no re-wrapping. Caller must keep `args` alive through this call.
  const size_t n = args.size();
  std::vector<std::optional<tvm::ffi::TensorView>> tvs(n);
  std::vector<tvm::ffi::AnyView> tvm_args(n);  // default ctor → kTVMFFINone
  for (size_t i = 0; i < n; ++i) {
    if (args[i].has_value()) {
      // DLTensorWrapper IS-A DLTensor; take the base pointer.
      tvs[i].emplace(static_cast<const DLTensor *>(&args[i].value()));
      tvm_args[i] = *tvs[i];
    }
  }
  auto fn = tvm::ffi::Function::GetGlobalRequired(fn_name);
  tvm::ffi::Any result;
  fn.CallPacked(tvm_args.data(), static_cast<int32_t>(n), &result);
}

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TVM_FFI_BRIDGE_H_
