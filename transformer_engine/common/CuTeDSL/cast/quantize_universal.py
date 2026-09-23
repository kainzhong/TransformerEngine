# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Prototype builder for CTA-tiled CuTeDSL quantization kernels.

The builder owns the TMA pipeline and CTA-level tensor partitions.  A user callback owns
the thread mapping inside one tile, register fragments, quantization, and any collective
operations.  This first prototype intentionally supports only the following data path::

    GMEM --TMA--> SMEM --user--> RMEM --user--> SMEM --TMA--> GMEM

Scale outputs may either be staged in per-pipeline-slot SMEM buffers and flushed with
vectorized SIMT stores, or exposed as GMEM tile views for the callback to write directly.

Limitations of the prototype:

* ``DST`` must be ``"SMEM"`` and ``USE_TMA`` must be true.
* Scale tensors must use a compact, non-swizzled two-dimensional layout.
* Amax handling is deliberately left to a future version.
* Derived quantizers must provide ``TAG`` and ``_TILE_QUANTIZER_FUNC`` attributes.
"""

import math
import os
from typing import Callable, Optional, Type

import cutlass
from cutlass import cute, pipeline
from cutlass import Int32, Int64, Uint8
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

CUTEDSL_DEBUG_LOGGING = os.environ.get("CUTEDSL_DEBUG_LOGGING", "0") == "1"
TMA_ALIGN_BITS = 128

# GMEM (tensor scope) -> SMEM (tile scope) -> RMEM (MX block scope) -> RMEM (thread scope)
# there is one more block scope because multiple threads are allowed to cooperate to quantize a single MX block.

# Multiple threads cooperate to quantize a single MX block: QuantizeAtom
# A CTA quantizes a single tile: TiledQuantize

# If not quantize in colwise:
# You can select how many threads to cooperate to quantize a MX block. You probably need warp shuffle for something
# but we can give you some APIs so you can reduce using some ops across a warp. There are at most 5 possibilities for cooperating threads:
# 1, 2, 4, 8, 16, so we can just write 5 versions of it. And you give us the RMEM tensor SSA to reduce and this function reduce it for you

# If quantize in colwise:
# If using SMEM, then colwise quantizaiton can use SMEM
# If not, then you are restricted of using multiple threads for a single MX block, 
# because you need a warp to form MX rows for colwise quantization via warp shuffle

class UniversalQuantizer:
    """Base class for TMA-staged quantizers built around a tile callback.

    Parameters use uppercase names to emphasize that they are trace-time constants and
    potential autotuning dimensions.

    Every derived quantizer must set ``TAG`` to a non-empty logging tag and
    ``_TILE_QUANTIZER_FUNC`` to its curried tile callback before invoking this
    constructor. The base class owns the complete public ``__call__`` entrypoint.

    ``TILER`` is the logical input tile ``(rows, cols)``. Quantized output tiles have the
    same logical shape, although sub-byte outputs are exposed to the callback as packed
    byte tensors. Scale tile shapes are derived from ``MX_BLOCK_SIZE``.

    ``EXTRA_SMEM_FACTORY``, when present, is called once while tracing as::

        EXTRA_SMEM_FACTORY(smem_allocator, TILER, THREADS_PER_CTA, PIPELINE_DEPTH)

    It may allocate arbitrary additional SMEM from the supplied allocator and return a
    tensor, tuple of tensors, or other callback-specific bundle.

    Runtime tensor rows must have 16-byte-aligned pitches for every tensor transferred by
    TMA. ``required_n_divisibility`` exposes the resulting logical element constraint for
    callers constructing symbolic tensor signatures.
    """

    def __init__(
        self,
        *,
        DST: str = "SMEM",
        THREADS_PER_CTA: int,
        TILER: tuple[int, int],
        ROWWISE: bool,
        COLWISE: bool,
        COLWISE_TRANSPOSED: bool = False,
        INPUT_DTYPE: Type[cutlass.Numeric],
        OUTPUT_DTYPE: Type[cutlass.Numeric],
        SCALE_DTYPE: Type[cutlass.Numeric],
        MX_BLOCK_SIZE: int,
        STASH_SCALE_TO_SMEM: bool = True,
        EXTRA_SMEM_FACTORY: Optional[Callable] = None,
        NUM_TILES_X: int = 1,
        NUM_TILES_Y: int = 1,
        PIPELINE_DEPTH: int = 2,
        USE_TMA: bool = True,
    ):
        if not isinstance(getattr(self, "TAG", None), str) or not self.TAG:
            raise TypeError("Derived UniversalQuantizer classes must define a non-empty TAG")
        if not callable(getattr(self, "_TILE_QUANTIZER_FUNC", None)):
            raise TypeError("Derived UniversalQuantizer classes must define _TILE_QUANTIZER_FUNC")
        if DST.upper() != "SMEM":
            raise NotImplementedError(
                "UniversalQuantizer currently supports only DST='SMEM'; "
                "the future RMEM path will pass GMEM tiles to the callback"
            )
        if not USE_TMA:
            raise NotImplementedError("UniversalQuantizer currently supports only TMA copies")
        if not ROWWISE and not COLWISE:
            raise ValueError("At least one of ROWWISE or COLWISE must be enabled")
        if COLWISE_TRANSPOSED:
            raise NotImplementedError(
                "UniversalQuantizer does not currently support transposed columnwise output"
            )
        if THREADS_PER_CTA <= 0 or THREADS_PER_CTA % 32 != 0:
            raise ValueError("THREADS_PER_CTA must be a positive multiple of 32")
        if len(TILER) != 2 or TILER[0] <= 0 or TILER[1] <= 0:
            raise ValueError("TILER must be a pair of positive integers")
        if MX_BLOCK_SIZE <= 0:
            raise ValueError("MX_BLOCK_SIZE must be positive")
        if ROWWISE and TILER[1] % MX_BLOCK_SIZE != 0:
            raise ValueError("The tile column count must be divisible by MX_BLOCK_SIZE")
        if COLWISE and TILER[0] % MX_BLOCK_SIZE != 0:
            raise ValueError("The tile row count must be divisible by MX_BLOCK_SIZE")
        if NUM_TILES_X <= 0 or NUM_TILES_Y <= 0:
            raise ValueError("NUM_TILES_X and NUM_TILES_Y must be positive")
        if PIPELINE_DEPTH <= 0:
            raise ValueError("PIPELINE_DEPTH must be positive")
        if INPUT_DTYPE.width < 8 or INPUT_DTYPE.width % 8 != 0:
            raise ValueError("The TMA input dtype must contain a whole number of bytes")
        if OUTPUT_DTYPE.width not in (4, 8, 16, 32):
            raise ValueError("The prototype supports 4-, 8-, 16-, or 32-bit output dtypes")
        if SCALE_DTYPE.width != 8:
            raise ValueError("The prototype currently supports one-byte scale dtypes")

        TILE_ROWS, TILE_COLS = TILER
        INPUT_ALIGNMENT_ELEMS = TMA_ALIGN_BITS // INPUT_DTYPE.width
        OUTPUT_ALIGNMENT_ELEMS = TMA_ALIGN_BITS // OUTPUT_DTYPE.width
        if TILE_COLS % INPUT_ALIGNMENT_ELEMS != 0:
            raise ValueError("The input tile row must contain a multiple of 16 bytes")
        if TILE_COLS % OUTPUT_ALIGNMENT_ELEMS != 0:
            raise ValueError("The rowwise output tile row must contain a multiple of 16 bytes")
        self.DST = DST.upper()
        self.THREADS_PER_CTA = THREADS_PER_CTA
        self.TILER = TILER
        self.ROWWISE = ROWWISE
        self.COLWISE = COLWISE
        self.INPUT_DTYPE = INPUT_DTYPE
        self.OUTPUT_DTYPE = OUTPUT_DTYPE
        self.SCALE_DTYPE = SCALE_DTYPE
        self.MX_BLOCK_SIZE = MX_BLOCK_SIZE
        self.STASH_SCALE_TO_SMEM = STASH_SCALE_TO_SMEM
        self.EXTRA_SMEM_FACTORY = EXTRA_SMEM_FACTORY
        self.NUM_TILES_X = NUM_TILES_X
        self.NUM_TILES_Y = NUM_TILES_Y
        self.PIPELINE_DEPTH = PIPELINE_DEPTH
        self.USE_TMA = USE_TMA

        self.required_n_divisibility = math.lcm(
            INPUT_ALIGNMENT_ELEMS, OUTPUT_ALIGNMENT_ELEMS
        )
        self._OUTPUT_STORAGE_DTYPE = Uint8 if OUTPUT_DTYPE.width == 4 else OUTPUT_DTYPE
        OUTPUT_PACK = 8 // OUTPUT_DTYPE.width if OUTPUT_DTYPE.width < 8 else 1
        self._ROW_OUTPUT_TILER = (TILE_ROWS, TILE_COLS // OUTPUT_PACK)
        self._COL_OUTPUT_TILER = (TILE_ROWS // OUTPUT_PACK, TILE_COLS)
        self._ROW_SCALE_TILER = (TILE_ROWS, TILE_COLS // MX_BLOCK_SIZE)
        self._COL_SCALE_TILER = (TILE_ROWS // MX_BLOCK_SIZE, TILE_COLS)

        # TE pads the compact scale tensor's inner dimension to four bytes.  A
        # non-transposed columnwise scale row follows the input/output row and is
        # therefore at least 16-byte aligned.  Restrict the vector width to those
        # guarantees so every store is a complete vector, including padding.
        self._ROW_SCALE_VECTOR = math.gcd(self._ROW_SCALE_TILER[1], 4)
        self._COL_SCALE_VECTOR = math.gcd(self._COL_SCALE_TILER[1], 16)

    # The unused arguments preserve the common TE quantizer entrypoint ABI. Their
    # interpretation remains the responsibility of future tile callbacks.
    # pylint: disable=unused-argument
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO_row: Optional[cute.Tensor],
        mS_row: Optional[cute.Tensor],
        mO_col: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        mAmax: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        mDActInput: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        stream: CUstream,
    ):
        """Create TMA atoms and launch the universal kernel.

        Scale tensors must already carry their final compact layout. Swizzled scale
        layouts are intentionally unsupported by this first prototype.
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(f"[CuTeDSL] {self.TAG}\n")

        if cutlass.const_expr(mX.element_type != self.INPUT_DTYPE):
            raise TypeError("mX dtype does not match INPUT_DTYPE")
        if cutlass.const_expr(self.ROWWISE):
            if cutlass.const_expr(mO_row.element_type != self.OUTPUT_DTYPE):
                raise TypeError("mO_row dtype does not match OUTPUT_DTYPE")
            if cutlass.const_expr(mS_row.element_type != self.SCALE_DTYPE):
                raise TypeError("mS_row dtype does not match SCALE_DTYPE")
            self._validate_scale_layout(mS_row, "mS_row", self._ROW_SCALE_VECTOR)
        if cutlass.const_expr(self.COLWISE):
            if cutlass.const_expr(mO_col.element_type != self.OUTPUT_DTYPE):
                raise TypeError("mO_col dtype does not match OUTPUT_DTYPE")
            if cutlass.const_expr(mS_col.element_type != self.SCALE_DTYPE):
                raise TypeError("mS_col dtype does not match SCALE_DTYPE")
            self._validate_scale_layout(mS_col, "mS_col", self._COL_SCALE_VECTOR)

        rows, cols = mX.shape
        cute.testing.assert_(  # pylint: disable=deprecated-method
            cols % self.required_n_divisibility == 0,
            "Input/output row pitches must be divisible by 16 bytes",
        )
        # Create TMA atom for inputs which will load a TILER at a time
        in_smem_layout = cute.make_ordered_layout(self.TILER, order=(1, 0))
        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_in, tma_view_in = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load,
            mX,
            in_smem_layout,
            self.TILER,
            num_multicast=1,
        )

        # Create TMA atoms for outputs which will load TILER's corresponding output regions 
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_row = None
        tma_view_row = None
        tma_atom_col = None
        tma_view_col = None

        if cutlass.const_expr(self.ROWWISE):
            mO_row_storage = self._output_storage_view(mO_row)
            row_smem_layout = cute.make_ordered_layout(self._ROW_OUTPUT_TILER, order=(1, 0))
            tma_atom_row, tma_view_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store,
                mO_row_storage,
                row_smem_layout,
                self._ROW_OUTPUT_TILER,
                num_multicast=1,
            )

        if cutlass.const_expr(self.COLWISE):
            mO_col_storage = self._output_storage_view(mO_col)
            col_smem_layout = cute.make_ordered_layout(self._COL_OUTPUT_TILER, order=(1, 0))
            tma_atom_col, tma_view_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store,
                mO_col_storage,
                col_smem_layout,
                self._COL_OUTPUT_TILER,
                num_multicast=1,
            )

        grid = [
            cute.ceil_div(Int32(cols), self.TILER[1] * self.NUM_TILES_X),
            cute.ceil_div(rows, self.TILER[0] * self.NUM_TILES_Y),
            1,
        ]
        self.kernel(
            mX,
            mS_row,
            mS_col,
            tma_atom_in,
            tma_view_in,
            tma_atom_row,
            tma_view_row,
            tma_atom_col,
            tma_view_col,
        ).launch(
            grid=grid,
            block=[self.THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    # pylint: enable=unused-argument

    @cute.jit
    def _output_storage_view(self, tensor: cute.Tensor):
        if cutlass.const_expr(self.OUTPUT_DTYPE.width == 4):
            return cute.recast_tensor(tensor, Uint8)
        return tensor

    @cute.jit
    def _validate_scale_layout(
        self,
        tensor: cute.Tensor,
        name: cutlass.Constexpr[str],
        vector_elements: cutlass.Constexpr[int],
    ):
        """Reject non-compact or insufficiently padded scale layouts in v1."""
        if cutlass.const_expr(not cute.is_congruent(tensor.shape, (1, 1))):
            raise ValueError(f"{name} must have a flat two-dimensional shape")
        if cutlass.const_expr(not cute.is_major(1, tensor.stride)):
            raise ValueError(f"{name} must be compact and row-major")
        cute.testing.assert_(  # pylint: disable=deprecated-method
            tensor.stride[0] == cute.size(tensor, mode=[1]),
            f"{name} must be compact and row-major",
        )
        cute.testing.assert_(  # pylint: disable=deprecated-method
            cute.size(tensor, mode=[1]) % vector_elements == 0,
            f"{name} inner dimension must be padded to the scale-store vector width",
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mS_row: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        tma_atom_in: cute.CopyAtom,
        tma_view_in: cute.Tensor,
        tma_atom_row: Optional[cute.CopyAtom],
        tma_view_row: Optional[cute.Tensor],
        tma_atom_col: Optional[cute.CopyAtom],
        tma_view_col: Optional[cute.Tensor],
    ):
        """Device entrypoint."""
        self._kernel_main(
            mX,
            mS_row,
            mS_col,
            tma_atom_in,
            tma_view_in,
            tma_atom_row,
            tma_view_row,
            tma_atom_col,
            tma_view_col,
        )

    @cute.jit
    def _kernel_main(
        self,
        mX: cute.Tensor,
        mS_row: Optional[cute.Tensor],
        mS_col: Optional[cute.Tensor],
        tma_atom_in: cute.CopyAtom,
        tma_view_in: cute.Tensor,
        tma_atom_row: Optional[cute.CopyAtom],
        tma_view_row: Optional[cute.Tensor],
        tma_atom_col: Optional[cute.CopyAtom],
        tma_view_col: Optional[cute.Tensor],
    ):
        tile_rows, tile_cols = self.TILER
        out_dtype = self._OUTPUT_STORAGE_DTYPE

        row_output_elems = (
            self._ROW_OUTPUT_TILER[0] * self._ROW_OUTPUT_TILER[1] * self.PIPELINE_DEPTH
            if self.ROWWISE
            else 1
        )
        col_output_elems = (
            self._COL_OUTPUT_TILER[0] * self._COL_OUTPUT_TILER[1] * self.PIPELINE_DEPTH
            if self.COLWISE
            else 1
        )
        row_scale_elems = (
            self._ROW_SCALE_TILER[0] * self._ROW_SCALE_TILER[1] * self.PIPELINE_DEPTH
            if self.ROWWISE and self.STASH_SCALE_TO_SMEM
            else 1
        )
        col_scale_elems = (
            self._COL_SCALE_TILER[0] * self._COL_SCALE_TILER[1] * self.PIPELINE_DEPTH
            if self.COLWISE and self.STASH_SCALE_TO_SMEM
            else 1
        )

        @cute.struct
        class SharedStorage:
            """Builder-owned barriers and pipeline ring buffers."""

            # TODO: Specialize SharedStorage for ROWWISE/COLWISE so disabled
            # directions do not reserve output or scale buffers.

            mbar_storage: cute.struct.MemRange[Int64, 2 * self.PIPELINE_DEPTH]
            sX: cute.struct.Align[
                cute.struct.MemRange[self.INPUT_DTYPE, tile_rows * tile_cols * self.PIPELINE_DEPTH],
                128,
            ]
            sO_row: cute.struct.Align[
                cute.struct.MemRange[out_dtype, row_output_elems],
                128,
            ]
            sO_col: cute.struct.Align[
                cute.struct.MemRange[out_dtype, col_output_elems],
                128,
            ]
            sS_row: cute.struct.Align[
                cute.struct.MemRange[self.SCALE_DTYPE, row_scale_elems],
                16,
            ]
            sS_col: cute.struct.Align[
                cute.struct.MemRange[self.SCALE_DTYPE, col_scale_elems],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sX = storage.sX.get_tensor(
            cute.make_layout(
                (self.TILER, self.PIPELINE_DEPTH),
                stride=((tile_cols, 1), tile_rows * tile_cols),
            )
        )
        sO_row = None
        sO_col = None
        sS_row = None
        sS_col = None
        if cutlass.const_expr(self.ROWWISE):
            sO_row = storage.sO_row.get_tensor(self._make_stage_layout(self._ROW_OUTPUT_TILER))
            if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                sS_row = storage.sS_row.get_tensor(self._make_stage_layout(self._ROW_SCALE_TILER))
        if cutlass.const_expr(self.COLWISE):
            sO_col = storage.sO_col.get_tensor(self._make_stage_layout(self._COL_OUTPUT_TILER))
            if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                sS_col = storage.sS_col.get_tensor(self._make_stage_layout(self._COL_SCALE_TILER))

        scratch = None
        if cutlass.const_expr(self.EXTRA_SMEM_FACTORY is not None):
            scratch = self.EXTRA_SMEM_FACTORY(
                smem,
                self.TILER,
                self.THREADS_PER_CTA,
                self.PIPELINE_DEPTH,
            )

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rows, cols = mX.shape

        tile_x_base = bidx * self.NUM_TILES_X
        tile_y_base = bidy * self.NUM_TILES_Y
        num_tiles_x = cutlass.min(
            self.NUM_TILES_X,
            cute.ceil_div(
                Int32(cols) - bidx * tile_cols * self.NUM_TILES_X,
                tile_cols,
            ),
        )
        num_tiles_y = cutlass.min(
            self.NUM_TILES_Y,
            cute.ceil_div(
                Int32(rows) - bidy * tile_rows * self.NUM_TILES_Y,
                tile_rows,
            ),
        )
        num_tiles = num_tiles_x * num_tiles_y

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_in)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.THREADS_PER_CTA // 32
        )
        tx_count = tile_rows * tile_cols * self.INPUT_DTYPE.width // 8
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_storage.data_ptr(),
            num_stages=self.PIPELINE_DEPTH,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=None,
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.PIPELINE_DEPTH
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.PIPELINE_DEPTH
        )

        gX_tiled = cute.zipped_divide(tma_view_in, self.TILER)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_in,
            0,
            cute.make_layout(1),
            sX,
            gX_tiled,
        )

        if cutlass.const_expr(self.ROWWISE):
            gO_row_tiled = cute.zipped_divide(tma_view_row, self._ROW_OUTPUT_TILER)
            tOsO_row, tOgO_row = cute.nvgpu.cpasync.tma_partition(
                tma_atom_row,
                0,
                cute.make_layout(1),
                sO_row,
                gO_row_tiled,
            )
        if cutlass.const_expr(self.COLWISE):
            gO_col_tiled = cute.zipped_divide(tma_view_col, self._COL_OUTPUT_TILER)
            tOsO_col, tOgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_col,
                0,
                cute.make_layout(1),
                sO_col,
                gO_col_tiled,
            )

        cute.arch.sync_threads()

        if warp_idx == 0:
            for stage in cutlass.range_constexpr(self.PIPELINE_DEPTH):
                if stage < num_tiles:
                    local_y = stage // num_tiles_x
                    local_x = stage % num_tiles_x
                    mainloop_pipeline.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom_in,
                        tXgX[(None, (tile_y_base + local_y, tile_x_base + local_x))],
                        tXsX[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    )
                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()

        for tile_idx in cutlass.range(num_tiles, unroll=1):
            mainloop_pipeline.consumer_wait(consumer_state)
            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(self.PIPELINE_DEPTH - 1, read=True)
            cute.arch.sync_threads()

            stage_idx = consumer_state.index
            local_y = tile_idx // num_tiles_x
            local_x = tile_idx % num_tiles_x
            tile_y = tile_y_base + local_y
            tile_x = tile_x_base + local_x

            gS_row_tile = None
            gS_col_tile = None
            callback_sS_row = None
            callback_sS_col = None
            if cutlass.const_expr(self.ROWWISE):
                gS_row_tile = cute.local_tile(
                    mS_row,
                    self._ROW_SCALE_TILER,
                    (tile_y, tile_x),
                )
                callback_sS_row = gS_row_tile
                if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                    callback_sS_row = sS_row[(None, stage_idx)]
                    self._zero_scale_tile(callback_sS_row, tidx)
            if cutlass.const_expr(self.COLWISE):
                gS_col_tile = cute.local_tile(
                    mS_col,
                    self._COL_SCALE_TILER,
                    (tile_y, tile_x),
                )
                callback_sS_col = gS_col_tile
                if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                    callback_sS_col = sS_col[(None, stage_idx)]
                    self._zero_scale_tile(callback_sS_col, tidx)

            if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                cute.arch.sync_threads()

            self._TILE_QUANTIZER_FUNC(
                sX[(None, stage_idx)],
                sO_row[(None, stage_idx)] if self.ROWWISE else None,
                callback_sS_row,
                sO_col[(None, stage_idx)] if self.COLWISE else None,
                callback_sS_col,
                scratch,
                tidx,
                tile_y * tile_rows,
                tile_x * tile_cols,
                rows,
                cols,
            )

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            if cutlass.const_expr(self.STASH_SCALE_TO_SMEM):
                if cutlass.const_expr(self.ROWWISE):
                    self._flush_scale_tile(
                        callback_sS_row,
                        gS_row_tile,
                        tidx,
                        tile_y * self._ROW_SCALE_TILER[0],
                        tile_x * self._ROW_SCALE_TILER[1],
                        cute.size(mS_row, mode=[0]),
                        cute.size(mS_row, mode=[1]),
                        self._ROW_SCALE_TILER,
                        self._ROW_SCALE_VECTOR,
                    )
                if cutlass.const_expr(self.COLWISE):
                    self._flush_scale_tile(
                        callback_sS_col,
                        gS_col_tile,
                        tidx,
                        tile_y * self._COL_SCALE_TILER[0],
                        tile_x * self._COL_SCALE_TILER[1],
                        cute.size(mS_col, mode=[0]),
                        cute.size(mS_col, mode=[1]),
                        self._COL_SCALE_TILER,
                        self._COL_SCALE_VECTOR,
                    )

            mainloop_pipeline.consumer_release(consumer_state)

            if warp_idx == 0:
                if cutlass.const_expr(self.ROWWISE):
                    cute.copy(
                        tma_atom_row,
                        tOsO_row[(None, stage_idx)],
                        tOgO_row[(None, (tile_y, tile_x))],
                    )
                if cutlass.const_expr(self.COLWISE):
                    cute.copy(
                        tma_atom_col,
                        tOsO_col[(None, stage_idx)],
                        tOgO_col[(None, (tile_y, tile_x))],
                    )
                cute.arch.cp_async_bulk_commit_group()

            consumer_state.advance()

            next_tile_idx = tile_idx + self.PIPELINE_DEPTH
            if next_tile_idx < num_tiles:
                if warp_idx == 0:
                    next_local_y = next_tile_idx // num_tiles_x
                    next_local_x = next_tile_idx % num_tiles_x
                    mainloop_pipeline.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom_in,
                        tXgX[
                            (
                                None,
                                (
                                    tile_y_base + next_local_y,
                                    tile_x_base + next_local_x,
                                ),
                            )
                        ],
                        tXsX[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                    )
                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()

        cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def _make_stage_layout(self, tile_shape: tuple[int, int]):
        rows, cols = tile_shape
        return cute.make_layout(
            (tile_shape, self.PIPELINE_DEPTH),
            stride=((cols, 1), rows * cols),
        )

    @cute.jit
    def _zero_scale_tile(self, sS: cute.Tensor, tidx: Int32):
        flat = cute.flatten(sS)
        elements = cute.size(flat)
        zero = Uint8(0).bitcast(self.SCALE_DTYPE)
        for wave in cutlass.range_constexpr(cute.ceil_div(elements, self.THREADS_PER_CTA)):
            index = wave * self.THREADS_PER_CTA + tidx
            if index < elements:
                flat[index] = zero

    @cute.jit
    def _flush_scale_tile(
        self,
        sS: cute.Tensor,
        gS_tile: cute.Tensor,
        tidx: Int32,
        global_row0: Int32,
        global_col0: Int32,
        physical_rows: Int32,
        physical_cols: Int32,
        tile_shape: tuple[int, int],
        vector_elements: int,
    ):
        """Flush one compact, padded scale tile with complete SIMT vector stores.

        TODO: Select ``cp.async.bulk`` or a TMA store when the scale tile's physical
        layout, alignment, and transfer size make a bulk copy legal and profitable.
        """
        rows, cols = tile_shape
        active_cols = cols // vector_elements
        _, tv_layout = cute.make_layout_tv(
            thr_layout=cute.make_layout(
                (rows, active_cols),
                stride=(active_cols, 1),
            ),
            val_layout=cute.make_layout(
                (1, vector_elements),
                stride=(vector_elements, 1),
            ),
        )
        total_vectors = rows * active_cols
        sS_tv = cute.composition(sS, tv_layout)
        gS_tv = cute.composition(gS_tile, tv_layout)

        for wave in cutlass.range_constexpr(cute.ceil_div(total_vectors, self.THREADS_PER_CTA)):
            vector_idx = wave * self.THREADS_PER_CTA + tidx
            if vector_idx < total_vectors:
                row = vector_idx // active_cols
                col = (vector_idx % active_cols) * vector_elements
                if (
                    global_row0 + row < physical_rows
                    and global_col0 + col < physical_cols
                ):
                    cute.autovec_copy(
                        sS_tv[vector_idx, None],
                        gS_tv[vector_idx, None],
                    )
