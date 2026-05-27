# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""

import importlib.util
import os
from pathlib import Path

import setuptools

from .utils import (
    all_files_in_dir,
    cuda_version,
    get_cuda_include_dirs,
    debug_build_enabled,
    setup_mpi_flags,
)
from typing import List


def install_requirements() -> List[str]:
    """Install dependencies for TE/PyTorch extensions."""
    return ["torch>=2.1", "einops", "onnxscript", "onnx", "packaging", "pydantic", "nvdlfw-inspect"]


def test_requirements() -> List[str]:
    """Test dependencies for TE/PyTorch extensions."""
    return [
        "numpy",
        "torchvision",
        "transformers",
        "torchao==0.13",
        "onnxruntime",
        "onnxruntime_extensions",
    ]


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    sources = all_files_in_dir(Path(csrc_source_files), name_extension="cpp")

    # Header files
    include_dirs = get_cuda_include_dirs()
    include_dirs.extend(
        [
            common_header_files,
            common_header_files / "common",
            common_header_files / "common" / "include",
            csrc_header_files,
        ]
    )

    # nvidia_cutlass_dsl ships CuteDSLRuntime.h inside its pip wheel's include/ dir.
    cutlass_dsl_spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if cutlass_dsl_spec is None or not cutlass_dsl_spec.submodule_search_locations:
        raise RuntimeError(
            "nvidia_cutlass_dsl package not found; install it (e.g. `pip install nvidia-cutlass-dsl`)"
            " — required for CuteDSLRuntime.h"
        )
    cutlass_dsl_root = Path(cutlass_dsl_spec.submodule_search_locations[0])
    cutlass_dsl_include = cutlass_dsl_root / "include"
    cutlass_dsl_lib_dir = cutlass_dsl_root / "lib"
    if not cutlass_dsl_include.is_dir():
        raise RuntimeError(
            f"nvidia_cutlass_dsl include directory not found at {cutlass_dsl_include}"
        )
    if not (cutlass_dsl_lib_dir / "libcute_dsl_runtime.so").exists():
        raise RuntimeError(
            f"libcute_dsl_runtime.so not found at {cutlass_dsl_lib_dir}"
        )
    include_dirs.append(cutlass_dsl_include)

    # apache-tvm-ffi headers + libtvm_ffi.so are required by `quantize_with_func`
    # to dispatch AOT-compiled CuTe DSL kernels.
    tvm_ffi_spec = importlib.util.find_spec("tvm_ffi")
    if tvm_ffi_spec is None or not tvm_ffi_spec.submodule_search_locations:
        raise RuntimeError(
            "apache-tvm-ffi package not found; install it (e.g. `pip install apache-tvm-ffi`)"
            " — required for AOT CuTe DSL kernel dispatch"
        )
    tvm_ffi_root = Path(tvm_ffi_spec.submodule_search_locations[0])
    tvm_ffi_include = tvm_ffi_root / "include"
    tvm_ffi_lib_dir = tvm_ffi_root / "lib"
    if not tvm_ffi_include.is_dir() or not (tvm_ffi_lib_dir / "libtvm_ffi.so").exists():
        raise RuntimeError(
            f"apache-tvm-ffi assets missing at {tvm_ffi_root} (need include/ and lib/libtvm_ffi.so)"
        )
    include_dirs.append(tvm_ffi_include)

    # Compiler flags
    cxx_flags = ["-O3", "-fvisibility=hidden"]
    if debug_build_enabled():
        cxx_flags.append("-g")
        cxx_flags.append("-UNDEBUG")
    else:
        cxx_flags.append("-g0")

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA version")
    else:
        if version < (12, 0):
            raise RuntimeError("Transformer Engine requires CUDA 12.0 or newer")

    setup_mpi_flags(include_dirs, cxx_flags)

    # Link TE against libtvm_ffi.so AND libcute_dsl_runtime.so so they're
    # already in the process when an AOT kernel .so is dlopen'd at runtime
    # (the kernel .so's DT_NEEDED references both by SONAME). User-built
    # kernel .so files are loaded dynamically and do NOT trigger a TE rebuild.
    #
    # libcute_dsl_runtime isn't directly referenced by TE's own C++ code,
    # so we wrap that single -l with --no-as-needed to keep its DT_NEEDED
    # entry even with the default --as-needed link mode.
    library_dirs = [tvm_ffi_lib_dir, cutlass_dsl_lib_dir]
    libraries = ["tvm_ffi"]
    extra_link_args = [
        f"-Wl,-rpath,{tvm_ffi_lib_dir}",
        f"-Wl,-rpath,{cutlass_dsl_lib_dir}",
        "-Wl,--no-as-needed", "-lcute_dsl_runtime", "-Wl,--as-needed",
    ]
    if bool(int(os.getenv("NVTE_ENABLE_NVSHMEM", 0))):
        assert (
            os.getenv("NVSHMEM_HOME") is not None
        ), "NVSHMEM_HOME must be set when compiling with NVTE_ENABLE_NVSHMEM=1"
        nvshmem_home = Path(os.getenv("NVSHMEM_HOME"))
        include_dirs.append(nvshmem_home / "include")
        library_dirs.append(nvshmem_home / "lib")
        libraries.append("nvshmem_host")
        cxx_flags.append("-DNVTE_ENABLE_NVSHMEM")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={"cxx": cxx_flags},
        extra_link_args=extra_link_args,
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
