/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <Python.h>

#include "tvm_ffi_bridge.h"

namespace transformer_engine {
namespace tvm_ffi_bridge {

namespace {

bool call_python_function(PyObject *module, const char *name) {
  PyObject *function = PyObject_GetAttrString(module, name);
  if (function == nullptr) {
    return false;
  }
  PyObject *result = PyObject_CallObject(function, nullptr);
  Py_DECREF(function);
  if (result == nullptr) {
    return false;
  }
  Py_DECREF(result);
  return true;
}

bool import_and_register_cutedsl_backends(bool embedding_python) {
  if (embedding_python) {
    PyObject *modules = PyImport_GetModuleDict();
    if (modules == nullptr ||
        PyDict_SetItemString(modules, "transformer_engine.pytorch", Py_None) != 0 ||
        PyDict_SetItemString(modules, "transformer_engine.jax", Py_None) != 0) {
      return false;
    }
  }

  PyObject *common = PyImport_ImportModule("transformer_engine.common");
  if (common == nullptr) {
    return false;
  }
  // Try to initialize again because it's possible that the CuTeDSL backend could be disabled with 
  // NVTE_ENABLE_CUTEDSL_BACKEND=0 when launching the program
  const bool initialized = call_python_function(common, "_load_tvm_ffi_library") &&
                           call_python_function(common, "_register_cutedsl_backends");
  Py_DECREF(common);
  return initialized;
}

}  // namespace

// Initialize the Python interpreter and import the CuTeDSL backend module. This is only compiled
// with NVTE_WITH_CUTEDSL=ON in CMake and will be ignored otherwise
bool initialize_python_cutedsl_backend() {
  const bool embedding_python = !Py_IsInitialized();

  // TE is loaded as a C++ library and it's not loaded from python. We need to launch an embedded
  // python so we can compile CuTeDSL backends
  if (embedding_python) {
    // Py_Initialize leaves the calling thread attached and holding the GIL.
    Py_Initialize();
    if (!Py_IsInitialized()) {
      NVTE_WARN(
          "Failed to initialize CuTeDSL backend: unable to initialize Python interpreter. Using "
          "CUDA backend as fallback.");
      return false;
    }
    const bool initialized = import_and_register_cutedsl_backends(embedding_python);
    if (!initialized) {
      PyErr_Print();
      NVTE_WARN(
          "Failed to initialize CuTeDSL backend: unable to import transformer_engine.common from "
          "python. Using CUDA backend as fallback.");
    }
    // Detach the initializing C++ thread and release the GIL so TVM-FFI callbacks can acquire it
    // from any thread. The interpreter has process lifetime, so no later restore is needed.
    (void)PyEval_SaveThread();
    return initialized;
  }

  // Python is already running in this process but we don't know if it has imported TE or not,
  // so we import here just to be sure
  const PyGILState_STATE gil_state = PyGILState_Ensure();
  const bool initialized = import_and_register_cutedsl_backends(embedding_python);
  if (!initialized) {
    PyErr_Print();
    NVTE_WARN(
        "Failed to initialize CuTeDSL backend: unable to import transformer_engine.common from "
        "python. Using CUDA backend as fallback.");
  }
  PyGILState_Release(gil_state);
  return initialized;
}

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine
