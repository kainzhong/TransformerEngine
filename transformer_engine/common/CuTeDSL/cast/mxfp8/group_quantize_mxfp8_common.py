# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped MXFP8 quantization in CuTeDSL: shared definitions and kernel dispatch.

Holds the grouped config, the constants the grouped kernels share, and
MXFP8GroupQuantizeEntry, which picks the kernel for a config: the single-tensor entry
(quantize_mxfp8_common.py) on a stacked or flat view of the group, the GMEM-to-RMEM grouped
kernel (group_quantize_mxfp8_g2r.py), or the generic grouped kernel
(group_quantize_mxfp8_general.py). Also compiles and registers the TVM-FFI function the C++
bridge (group_quantize_mxfp8_cutedsl.cuh) calls.
"""

# pylint: disable=missing-class-docstring

import logging
import os
from typing import Optional

import cutlass
from cutlass import cute
from cutlass import Float32, Int32, Int64
from cutlass.utils import HardwareInfo
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module
import tvm_ffi

from transformer_engine.common.CuTeDSL.utils import (
    str_to_cutlass_dtype,
    device_compute_capability,
)
from .quantize_mxfp8_common import (
    SYM_N_DIVISIBILITY,
    FP8E4M3_MAX_NORM_RCP,
    FP8E5M2_MAX_NORM_RCP,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_DACTIVATIONS,
)

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"
logger = logging.getLogger("transformer_engine.cutedsl.mxfp8")

THREADS_PER_WARP = 32
# CUDA stores four descriptor arrays: input, act_input, output_rowwise, output_colwise.
NUM_TENSORMAPS = 4
INPUT_SLOT = 0
ACT_INPUT_SLOT = 1
ROWWISE_OUTPUT_SLOT = 2
COLWISE_OUTPUT_SLOT = 3

# Shape representations, mirroring ShapeRepresentation in common/utils.cuh.
SAME_BOTH_DIMS = "sbd"
VARYING_FIRST_DIM = "vfd"
VARYING_LAST_DIM = "vld"
VARYING_BOTH_DIMS = "vbd"
SUPPORTED_SHAPE_REPS = (SAME_BOTH_DIMS, VARYING_FIRST_DIM, VARYING_LAST_DIM, VARYING_BOTH_DIMS)

# Upper bound on the group size (MAX_SUPPORTED_TENSOR_DESCRIPTORS in grouped_tma.cuh); sizes
# the fixed binary search over the offsets.
MAX_SUPPORTED_TENSORS = 64
# How many bytes is a TensorMap
BYTES_PER_TENSORMAP = 128
INT64_PER_TENSORMAP = BYTES_PER_TENSORMAP // 8
# The byte offset of fields in the TensorMap struct
ROWS_OFFSET = NUM_TENSORMAPS * MAX_SUPPORTED_TENSORS * INT64_PER_TENSORMAP
COLS_OFFSET = ROWS_OFFSET + MAX_SUPPORTED_TENSORS
OFFSETS_OFFSET = COLS_OFFSET + MAX_SUPPORTED_TENSORS
# Size of the TensorMap struct in int64
TENSOR_MAP_STORAGE_INT64_COUNT = OFFSETS_OFFSET + MAX_SUPPORTED_TENSORS


class MXFP8GroupQuantizeConfig:
    """Compile-time config for the grouped MXFP8 quantize kernel."""

    def __init__(
        self,
        dtype: str,
        fp8_dtype: str,
        rowwise: bool,
        colwise: bool,
        shape_rep: str,
        with_gemm_swizzled_scales: bool = False,
        with_dbias: bool = False,
        with_dact: bool = False,
        with_act: bool = False,
        activation: str = "none",
    ):
        if dtype not in ("Float32", "Float16", "BFloat16"):
            raise ValueError(f"unknown input dtype {dtype!r}; expected Float32|Float16|BFloat16")
        self.DTYPE = str_to_cutlass_dtype(dtype)
        self.DTYPE_STR = dtype
        if fp8_dtype not in ("Float8E4M3", "Float8E5M2"):
            raise ValueError(
                f"unknown FP8 dtype {fp8_dtype!r}; expected 'Float8E4M3' or 'Float8E5M2'"
            )
        self.FP8_DTYPE = str_to_cutlass_dtype(fp8_dtype)
        self.FP8_DTYPE_STR = fp8_dtype
        if not (rowwise or colwise):
            raise ValueError("at least one of rowwise or colwise must be true")
        self.ROWWISE = rowwise
        self.COLWISE = colwise
        if shape_rep not in SUPPORTED_SHAPE_REPS:
            raise ValueError(
                f"unsupported shape representation {shape_rep!r}; expected one of"
                f" {SUPPORTED_SHAPE_REPS}"
            )
        self.SHAPE_REP = shape_rep
        # Only can view the grouped tensor as one tensor when the last dim is same across all tensors
        self.IS_SINGLE_TENSOR = shape_rep in (SAME_BOTH_DIMS, VARYING_FIRST_DIM)
        self.MAX_NORM_RCP = (
            FP8E4M3_MAX_NORM_RCP if fp8_dtype == "Float8E4M3" else FP8E5M2_MAX_NORM_RCP
        )

        self.WITH_GEMM_SWIZZLED_SCALES = with_gemm_swizzled_scales
        if with_dbias and not self.IS_SINGLE_TENSOR:
            # mxfp8::group_quantize raises for this.
            raise ValueError("dbias is only supported for tensors with a common last dimension")
        self.WITH_DBIAS = with_dbias

        if with_dact and with_act:
            raise ValueError("with_dact and with_act cannot both be set")
        if with_dact:
            if activation not in SUPPORTED_DACTIVATIONS:
                raise ValueError(
                    f"unknown activation {activation!r} for with_dact=True; expected one of"
                    f" {sorted(SUPPORTED_DACTIVATIONS)}"
                )
            self.ACTIVATION = activation
        elif with_act:
            if activation not in SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"unknown activation {activation!r} for with_act=True; expected one of"
                    f" {sorted(SUPPORTED_ACTIVATIONS)}"
                )
            self.ACTIVATION = activation
        else:
            if activation != "none":
                raise ValueError("activation must be none when with_dact and with_act are False")
            self.ACTIVATION = None
        self.WITH_DACT = with_dact
        self.WITH_ACT = with_act

    def __str__(self):
        return (
            f"MXFP8GroupQuantizeConfig(dtype={self.DTYPE_STR}, fp8_dtype={self.FP8_DTYPE_STR}, "
            f"rowwise={self.ROWWISE}, colwise={self.COLWISE}, shape_rep={self.SHAPE_REP}, "
            f"swizzled={self.WITH_GEMM_SWIZZLED_SCALES}, with_dbias={self.WITH_DBIAS}, "
            f"with_dact={self.WITH_DACT}, with_act={self.WITH_ACT}, "
            f"activation={self.ACTIVATION})"
        )

    __repr__ = __str__


class MXFP8GroupQuantizeEntry:
    """Pick the kernel for a grouped config at compile time.

    Plain casts (no dbias/act/dact) whose scales need no per-member addressing are run as a
    single-tensor cast by the single-tensor entry, which dispatches to the GMEM-to-RMEM
    specialized kernels:

    * "stacked": SAME_BOTH_DIMS / VARYING_FIRST_DIM members start on 128-row boundaries and
      share the column count, so unless colwise scales are GEMM-swizzled (whose indices
      restart per member) the group is byte-for-byte the stacked (sum(rows), cols) tensor.
    * "flat_rowwise": with packed (non-swizzled) rowwise scales only, scale i covers elements
      32i..32i+31 of the flat payload whatever the member shapes, as long as every member's
      cols are a multiple of 128 (which both grouped kernels require for VARYING_LAST_DIM /
      VARYING_BOTH_DIMS). Any group is then a cast of a (total / 128, 128) tensor.

    * "grouped_g2r": the remaining BF16 plain casts (colwise / both directions of
      VARYING_LAST_DIM / VARYING_BOTH_DIMS, and swizzled colwise of every representation) run
      MXFP8GroupQuantizeG2RKernel, which addresses each member on its own.

    Everything else, and any call these paths cannot take at runtime (a cast-noop flag,
    2^31+ elements, cols not a multiple of 128, unaligned buffers), runs
    MXFP8GroupQuantizeKernel.
    """

    def __init__(self, cfg: MXFP8GroupQuantizeConfig, sm_count: int) -> None:
        # pylint: disable=import-outside-toplevel
        from .group_quantize_mxfp8_general import MXFP8GroupQuantizeKernel

        self.cfg = cfg
        self.grouped_kernel = MXFP8GroupQuantizeKernel(cfg, sm_count)
        plain_cast = not (cfg.WITH_DBIAS or cfg.WITH_ACT or cfg.WITH_DACT)
        self.SINGLE_TENSOR_MODE = None
        if plain_cast and cfg.IS_SINGLE_TENSOR:
            if not (cfg.COLWISE and cfg.WITH_GEMM_SWIZZLED_SCALES):
                self.SINGLE_TENSOR_MODE = "stacked"
        elif plain_cast and cfg.ROWWISE and not cfg.COLWISE and not cfg.WITH_GEMM_SWIZZLED_SCALES:
            self.SINGLE_TENSOR_MODE = "flat_rowwise"
        self.single_tensor_entry = None
        self.g2r_kernel = None
        if self.SINGLE_TENSOR_MODE is None:
            # pylint: disable=import-outside-toplevel
            from .group_quantize_mxfp8_g2r import MXFP8GroupQuantizeG2RKernel

            if MXFP8GroupQuantizeG2RKernel.supports(cfg):
                self.g2r_kernel = MXFP8GroupQuantizeG2RKernel(cfg, sm_count)
        if self.SINGLE_TENSOR_MODE is not None:
            # pylint: disable=import-outside-toplevel
            from .quantize_mxfp8_common import MXFP8QuantizeConfig, MXFP8QuantizeEntry

            self.single_tensor_entry = MXFP8QuantizeEntry(
                MXFP8QuantizeConfig(
                    dtype=cfg.DTYPE_STR,
                    fp8_dtype=cfg.FP8_DTYPE_STR,
                    rowwise=cfg.ROWWISE,
                    colwise=cfg.COLWISE,
                    with_gemm_swizzled_scales=cfg.WITH_GEMM_SWIZZLED_SCALES,
                    with_amax=False,
                    activation="none",
                )
            )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mOffsets: Optional[cute.Tensor],
        mFirstDims: Optional[cute.Tensor],
        mLastDims: Optional[cute.Tensor],
        mTensormaps: Optional[cute.Tensor],
        num_tensors: Int32,
        mNoop: cute.Pointer,
        mActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ) -> None:
        grouped_args = (
            mX,
            mO_row,
            mO_col,
            mS_row,
            mS_col,
            mOffsets,
            mFirstDims,
            mLastDims,
            mTensormaps,
            num_tensors,
            mNoop,
            mActInput,
            mWorkspace,
            stream,
        )
        # The single-tensor kernels index with Int32, so groups of 2^31 or more elements
        # stay on the grouped kernel.
        num_elems = Int64(mX.shape[0]) * Int64(mX.shape[1])
        if cutlass.const_expr(self.SINGLE_TENSOR_MODE == "stacked"):
            # The grouped kernels zero the scale padding and the single-tensor kernels leave it
            # untouched, so stack only when there is none: rows are already 128-aligned, and
            # N % 128 == 0 leaves no padded rowwise (N / 32 % 4) or colwise (N % 128) columns.
            if mX.shape[1] % 128 == 0 and num_elems < Int64(2**31):
                # Restate the divisibility the Int32 casts drop -- M by the compiled sym, N by
                # the check above -- so the vectorized GMEM copies can prove their alignment.
                M = cute.assume(Int32(mX.shape[0]), divby=128)
                N = cute.assume(Int32(mX.shape[1]), divby=128)
                self._run_single_tensor(mX, mO_row, mO_col, mS_row, mS_col, mNoop, stream, M, N)
            else:
                self.grouped_kernel(*grouped_args)
        elif cutlass.const_expr(self.SINGLE_TENSOR_MODE == "flat_rowwise"):
            if num_elems < Int64(2**31):
                M = Int32(num_elems // 128)
                self._run_single_tensor(mX, mO_row, mO_col, mS_row, mS_col, mNoop, stream, M, 128)
            else:
                self.grouped_kernel(*grouped_args)
        elif cutlass.const_expr(self.g2r_kernel is not None):
            eligible = (
                num_elems < Int64(2**31)
                and mNoop.toint() == 0
                and mX.iterator.toint() % 32 == 0
            )
            if cutlass.const_expr(self.cfg.COLWISE):
                eligible = eligible and mS_col.iterator.toint() % 16 == 0
            if cutlass.const_expr(self.cfg.IS_SINGLE_TENSOR):
                # VARYING_LAST_DIM / VARYING_BOTH_DIMS members have 128-aligned cols already.
                eligible = eligible and mX.shape[1] % 128 == 0
            if eligible:
                self.g2r_kernel(
                    mX,
                    mO_row,
                    mO_col,
                    mS_row,
                    mS_col,
                    mOffsets,
                    mFirstDims,
                    mLastDims,
                    num_tensors,
                    stream,
                )
            else:
                self.grouped_kernel(*grouped_args)
        else:
            self.grouped_kernel(*grouped_args)

    @cute.jit
    def _run_single_tensor(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_row: cute.Tensor,
        mS_col: cute.Tensor,
        mNoop: cute.Pointer,
        stream: CUstream,
        M: Int32,
        N: Int32,
    ) -> None:
        """Quantize the group's flat payload as one (M, N) tensor with the single-tensor entry.

        The single-tensor kernels are compiled for Int32 extents. The scales arrive as flat
        buffers; view them at the 2D shapes TE allocates for a single (M, N) tensor, which
        carry no padding here (M % 128 == N % 128 == 0).
        """
        data_layout = cute.make_layout((M, N), stride=(N, 1))
        x = cute.make_tensor(mX.iterator, data_layout)
        o_row = s_row = o_col = s_col = None
        if cutlass.const_expr(self.cfg.ROWWISE):
            o_row = cute.make_tensor(mO_row.iterator, data_layout)
            s_row = cute.make_tensor(mS_row.iterator, cute.make_layout((M, N // 32), stride=(N // 32, 1)))
        if cutlass.const_expr(self.cfg.COLWISE):
            o_col = cute.make_tensor(mO_col.iterator, data_layout)
            s_col = cute.make_tensor(mS_col.iterator, cute.make_layout((M // 32, N), stride=(N, 1)))
        self.single_tensor_entry(x, o_row, s_row, o_col, s_col, None, mNoop, None, None, stream)


def compile_cutedsl_function_from_cfg(cfg: MXFP8GroupQuantizeConfig):
    """Return the compiled CuTeDSL function object for the given grouped config."""
    # CUDA requires the group's first logical dim to be a multiple of 128 (and each
    # tensor's rows likewise). VARYING_BOTH_DIMS is the exception: its logical shape is
    # [1, total]. The last dim only needs the 16-byte TMA row alignment; a partial 32-element
    # scale block at the end of a row is zero-filled by TMA, as in the CUDA kernel.
    if cfg.SHAPE_REP == VARYING_BOTH_DIMS:
        sym_M = cute.sym_int64()
    else:
        sym_M = cute.sym_int64(divisibility=128)
    sym_N = cute.sym_int64(divisibility=SYM_N_DIVISIBILITY)
    logical_shape = (sym_M, sym_N)

    out_dtype = cfg.FP8_DTYPE
    scale_dtype = cutlass.Float8E8M0FNU

    out_col_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.COLWISE
        else None
    )

    out_row_fake = (
        cute.runtime.make_fake_compact_tensor(
            out_dtype,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.ROWWISE
        else None
    )

    # The kernel only takes the base address of the scale buffers (per-tensor strides
    # are derived from cols), so their fake shape is a flat 1D byte run.
    # One compact view covers CUDA's descriptor and metadata arrays in their native order.
    tensormaps_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int64,
            (TENSOR_MAP_STORAGE_INT64_COUNT,),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=128,
        )
        if not cfg.IS_SINGLE_TENSOR
        else None
    )
    # The cast-noop flag is an always-present f32 pointer instead of an optional tensor, so
    # that one compiled kernel serves both an absent and a present flag (noop_flag_is_set).
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)
    act_input_fake = (
        cute.runtime.make_fake_compact_tensor(
            cfg.DTYPE,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if cfg.WITH_DACT
        else None
    )
    workspace_fake = (
        cute.runtime.make_fake_compact_tensor(
            Float32,
            (cute.sym_int64(), cute.sym_int64()),
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if cfg.WITH_DBIAS
        else None
    )

    first_dims_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int64,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=8,
        )
        if cfg.SHAPE_REP in (VARYING_FIRST_DIM, VARYING_BOTH_DIMS)
        else None
    )

    last_dims_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int64,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=8,
        )
        if cfg.SHAPE_REP in (VARYING_LAST_DIM, VARYING_BOTH_DIMS)
        else None
    )

    offsets_fake = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Int64,
            (cute.sym_int32(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=8,
        )
        if cfg.SHAPE_REP != SAME_BOTH_DIMS
        else None
    )

    sm_count = HardwareInfo().get_device_multiprocessor_count()
    kernel_obj = MXFP8GroupQuantizeEntry(cfg, sm_count)
    return cute.compile(
        kernel_obj,
        cute.runtime.make_fake_compact_tensor(  # mX
            cfg.DTYPE,
            logical_shape,
            stride_order=(1, 0),
            memspace=cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        out_row_fake,  # mO_row
        out_col_fake,  # mO_col
        cute.runtime.make_fake_compact_tensor(  # mS_row
            scale_dtype,
            (cute.sym_int64(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        cute.runtime.make_fake_compact_tensor(  # mS_col
            scale_dtype,
            (cute.sym_int64(),),
            stride_order=(0,),
            memspace=cute.AddressSpace.gmem,
            assumed_align=4,
        ),
        offsets_fake,  # mOffsets
        first_dims_fake,  # mFirstDims
        last_dims_fake,  # mLastDims
        tensormaps_fake,  # mTensormaps
        Int32(1),  # num_tensors, supplied at runtime
        noop_fake,  # mNoop
        act_input_fake,  # mActInput
        workspace_fake,  # mWorkspace
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_mxfp8_group_quantization_function(
    fn_name: str,
    dtype: str,
    fp8_dtype: str,
    rowwise: bool,
    colwise: bool,
    shape_rep: str,
    with_gemm_swizzled_scales: bool,
    with_dbias: bool,
    with_dact: bool,
    with_act: bool,
    activation: str,
) -> bool:
    """Compile the grouped MXFP8 quantize kernel for this config and register it in the TVM-FFI
    global registry under EXACTLY `fn_name` (the key the C++ dispatcher built; Python treats it as
    an opaque name). Returns True if a kernel is successfully registered under `fn_name` (the C++
    side then fetches it with GetGlobal(fn_name)); False if the config is unsupported, so the caller
    caches the negative result and falls back to the CUDA C++ grouped kernel.
    """
    try:
        # Already registered (e.g. by a prior call) -> supported.
        if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
            return True

        major, minor = device_compute_capability()
        if major < 10:
            logger.warning(
                "CuTeDSL MXFP8 backend requires compute capability >= 10.0 (Blackwell), "
                "but detected %d.%d; falling back to the CUDA C++ kernel.",
                major,
                minor,
            )
            return False

        try:
            cfg = MXFP8GroupQuantizeConfig(
                dtype=dtype,
                fp8_dtype=fp8_dtype,
                rowwise=rowwise,
                colwise=colwise,
                shape_rep=shape_rep,
                with_gemm_swizzled_scales=with_gemm_swizzled_scales,
                with_dbias=with_dbias,
                with_dact=with_dact,
                with_act=with_act,
                activation=activation,
            )
        except ValueError as e:
            logger.warning(
                "CuTeDSL grouped MXFP8 backend does not support this config, "
                "falling back to the CUDA C++ kernel: %s",
                e,
            )
            return False

        logger.debug("Compiling CuTeDSL grouped MXFP8 quantization kernel for %s", cfg)
        compiled = compile_cutedsl_function_from_cfg(cfg)
        # Register the native TVM-FFI function rather than its Python argument-parsing wrapper;
        # see get_mxfp8_quantization_function for why.
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "CuTeDSL grouped MXFP8 kernel compilation & registration failed, falling back to the"
            " CUDA C++ kernel: %s",
            e,
        )
        # Unconditionally fallback to CUDA path because we can't tell if this exception is
        # transient or permanent.
        return False
