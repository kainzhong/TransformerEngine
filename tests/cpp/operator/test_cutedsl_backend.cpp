/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

extern "C" int nvte_get_tvm_ffi_available();

// Test that the CuTeDSL Python and TVM-FFI backend is available to a standalone C++ process.
TEST(CuTeDSLBackend, TVMFFIAvailable) {
  const char *enabled = std::getenv("NVTE_ENABLE_CUTEDSL_BACKEND");
  if (enabled == nullptr || std::strcmp(enabled, "0") == 0) {
    GTEST_SKIP() << "CuTeDSL backend is disabled";
  }

  EXPECT_NE(nvte_get_tvm_ffi_available(), 0);
}
