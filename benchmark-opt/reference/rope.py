# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Rotary Position Embedding (RoPE) kernels for NeuronCore.

RoPE encodes positional information by rotating embedding dimension pairs using
precomputed sine/cosine frequencies, enabling position-aware attention without
absolute position embeddings.
"""

# BEGIN-HARNESS
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import torch_xla.core.xla_model as xm

import nki
import nki.isa as nisa
import nki.language as nl


# ---------------------------------------------------------------------------
# Inlined helpers (from nki-library, not importable in all environments)
# ---------------------------------------------------------------------------

def kernel_assert(condition: bool, error_text: str):
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
        " - Please check the validation message and adjust kernel inputs accordingly"
    )


def _get_program_sharding_info() -> Tuple[int, int, int]:
    grid_ndim = nl.program_ndim()
    n_prgs, prg_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    return grid_ndim, n_prgs, prg_id


def get_verified_program_sharding_info(
    kernel_name: str = "",
    allowed_ndims: Optional[Tuple[int, ...]] = None,
    max_sharding: Optional[int] = None,
) -> Tuple[int, int, int]:
    return _get_program_sharding_info()


def _is_hbm_buffer(tensor: nl.ndarray) -> bool:
    return tensor.buffer in (nl.hbm, nl.shared_hbm, nl.private_hbm)


class TensorView(nl.NKIObject):
    """Minimal TensorView for NKI tensor view operations (slice, broadcast, expand_dim)."""

    base_tensor: nl.ndarray
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int
    dtype: object
    scalar_offset: nl.ndarray = None
    indirect_dim: Optional[int] = None

    @staticmethod
    def get_trivial_strides(shape: Tuple[int, ...], base_stride: int = 1) -> Tuple[int, ...]:
        strides = [base_stride]
        for i in range(1, len(shape)):
            strides.append(strides[i - 1] * shape[len(shape) - i])
        ret = []
        for i in range(len(shape)):
            ret.append(strides[len(shape) - i - 1])
        return tuple(ret)

    def __init__(self, base_tensor: nl.ndarray):
        kernel_assert(base_tensor is not None, "Base tensor cannot be None")
        if isinstance(base_tensor, TensorView):
            self.base_tensor = base_tensor.base_tensor
            self.shape = base_tensor.shape
            self.strides = base_tensor.strides
            self.offset = base_tensor.offset
            self.dtype = base_tensor.dtype
            self.scalar_offset = base_tensor.scalar_offset
            self.vector_offset = base_tensor.vector_offset
            self.indirect_dim = base_tensor.indirect_dim
        else:
            self.base_tensor = base_tensor
            self.shape = tuple(base_tensor.shape)
            self.strides = TensorView.get_trivial_strides(self.shape)
            self.offset = 0
            self.dtype = base_tensor.dtype
            self.scalar_offset = None
            self.vector_offset = None
            self.indirect_dim = None

    def is_sbuf(self) -> bool:
        return self.base_tensor.buffer == nl.sbuf

    def is_hbm(self) -> bool:
        return _is_hbm_buffer(self.base_tensor)

    def _copy(self, shape=None, strides=None, offset=None, scalar_offset=None,
              vector_offset=None, indirect_dim=None, dtype=None, base_tensor=None) -> "TensorView":
        view = TensorView(base_tensor if base_tensor is not None else self.base_tensor)
        view.shape = tuple(shape) if shape is not None else self.shape
        view.strides = tuple(strides) if strides is not None else self.strides
        view.offset = offset if offset is not None else self.offset
        view.scalar_offset = scalar_offset if scalar_offset is not None else self.scalar_offset
        view.vector_offset = vector_offset if vector_offset is not None else self.vector_offset
        view.indirect_dim = indirect_dim if indirect_dim is not None else self.indirect_dim
        view.dtype = dtype if dtype is not None else self.dtype
        return view

    def get_view(self) -> nl.ndarray:
        ap_pattern = []
        for i in range(len(self.shape)):
            ap_pattern.append((self.strides[i], self.shape[i]))
        if self.indirect_dim is not None:
            if self.vector_offset is not None:
                return self.base_tensor.ap(
                    pattern=ap_pattern, offset=self.offset,
                    vector_offset=self.vector_offset, indirect_dim=self.indirect_dim, dtype=self.dtype,
                )
            return self.base_tensor.ap(
                pattern=ap_pattern, offset=self.offset,
                scalar_offset=self.scalar_offset, indirect_dim=self.indirect_dim, dtype=self.dtype,
            )
        return self.base_tensor.ap(pattern=ap_pattern, offset=self.offset, dtype=self.dtype)

    def slice(self, dim: int, start: int, end: int, step: int = 1) -> "TensorView":
        kernel_assert(dim < len(self.shape), f"Dimension {dim} out of range")
        kernel_assert(start >= 0, "Start index must be non-negative")
        kernel_assert(end > start, "End index must be greater than start")
        end = min(end, self.shape[dim])
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim] = (end - start + step - 1) // step
        new_strides[dim] = self.strides[dim] * step
        new_offset = self.offset + self.strides[dim] * start
        return self._copy(shape=new_shape, strides=new_strides, offset=new_offset)

    def expand_dim(self, dim: int) -> "TensorView":
        kernel_assert(dim <= len(self.shape), f"Dimension {dim} out of range")
        if self.is_sbuf():
            kernel_assert(dim > 0, "partition dim cannot be expanded")
        new_stride = 1 if dim == len(self.shape) else self.strides[dim] * self.shape[dim]
        new_shape = tuple(list(self.shape[:dim]) + [1] + list(self.shape[dim:]))
        new_strides = tuple(list(self.strides[:dim]) + [new_stride] + list(self.strides[dim:]))
        return self._copy(shape=new_shape, strides=new_strides)

    def broadcast(self, dim: int, size: int) -> "TensorView":
        kernel_assert(dim < len(self.shape), f"Dimension {dim} out of range")
        kernel_assert(self.shape[dim] == 1, f"Can only broadcast size-1 dimensions")
        if self.is_sbuf():
            kernel_assert(dim != 0, "Cannot broadcast on partition dimension for SBUF tensors")
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim] = size
        new_strides[dim] = 0
        return self._copy(shape=new_shape, strides=new_strides)


@nki.jit(platform_target="trn1")
def RoPE(
    x_in: nl.ndarray,
    cos: nl.ndarray,
    sin: nl.ndarray,
    lnc_shard: bool = False,
    contiguous_layout: bool = True,
    relayout_in_sbuf: bool = False,
) -> nl.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) to input embeddings.
    Standalone kernel with HBM I/O and optional LNC sharding.

    RoPE Formula:
        out[even] = x[even]*cos - x[odd]*sin
        out[odd] = x[odd]*cos + x[even]*sin

    Args:
        x_in: [d_head, B, n_heads, S] @ HBM
            - d_head: head dimension (must be even, ≤128)
            - B: batch size
            - n_heads: number of attention heads
            - S: sequence length (divisible by n_prgs if lnc_shard=True)
        cos: [d_head//2, B, S] @ HBM - cosine frequencies
        sin: [d_head//2, B, S] @ HBM - sine frequencies
        lnc_shard: parallelize across LNC cores by tiling sequence dimension
        contiguous_layout: memory layout in d_head dimension
            - True: [first_half, second_half] (default, more efficient)
            - False: [even, odd, even, odd, ...] (interleaved)
        relayout_in_sbuf: use SBUF matmul for layout conversion (only for small tensors)

    Returns:
        [d_head, B, n_heads, S] @ HBM - RoPE applied output
    """

    _validate_rope_inputs(x_in, cos, sin, 'RoPE')

    d_head, B, n_heads, S = x_in.shape
    half_d = d_head // 2

    # Determine parallelization across LNC cores
    n_prgs, prg_id = 1, 0
    if lnc_shard:
        _, n_prgs, prg_id = get_verified_program_sharding_info("RoPE", (0, 1))

    # Tile along sequence dimension
    kernel_assert(S % n_prgs == 0, f'RoPE: sequence length {S} not divisible by {n_prgs} programs')
    tile_size = S // n_prgs
    tile_start = tile_size * prg_id

    # Determine layout conversion strategy: DMA (strided access) vs SBUF (matmul)
    # SBUF relayout limited by gemm_moving_fmax, fallback to DMA for large tensors
    is_relayout_in_sbuf_supported = B * n_heads * S <= nl.tile_size.gemm_moving_fmax
    is_dma_relayout = not contiguous_layout and (not relayout_in_sbuf or not is_relayout_in_sbuf_supported)
    is_relayout_in_sbuf = not contiguous_layout and relayout_in_sbuf and is_relayout_in_sbuf_supported

    # Load input to SBUF with optional layout conversion via strided DMA
    x_in_sb = nl.ndarray((d_head, B, n_heads, tile_size), dtype=x_in.dtype, buffer=nl.sbuf)
    if is_dma_relayout:
        # Gather even/odd indices with stride=2 in d_head dimension
        nisa.dma_copy(
            dst=x_in_sb[:half_d, :, :, :],
            src=TensorView(x_in)
            .slice(dim=0, start=0, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
        )
        nisa.dma_copy(
            dst=x_in_sb[half_d:, :, :, :],
            src=TensorView(x_in)
            .slice(dim=0, start=1, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
        )
    else:
        nisa.dma_copy(dst=x_in_sb, src=x_in[:, :, :, tile_start : tile_start + tile_size])

    # Load cos/sin frequency tiles
    cos_sb = nl.ndarray((half_d, B, tile_size), dtype=cos.dtype, buffer=nl.sbuf)
    sin_sb = nl.ndarray((half_d, B, tile_size), dtype=sin.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_sb, src=cos[:, :, tile_start : tile_start + tile_size])
    nisa.dma_copy(dst=sin_sb, src=sin[:, :, tile_start : tile_start + tile_size])

    # Compute RoPE rotation in SBUF
    x_out_sb = nl.ndarray(x_in_sb.shape, dtype=x_in_sb.dtype, buffer=nl.sbuf)
    RoPE_sbuf(x_in_sb, cos_sb, sin_sb, x_out_sb, convert_from_interleaved=is_relayout_in_sbuf)

    # Store output to HBM with optional layout conversion via strided DMA
    x_out = nl.ndarray(x_in.shape, dtype=x_in.dtype, buffer=nl.shared_hbm)
    if is_dma_relayout:
        # Scatter even/odd indices with stride=2 in d_head dimension
        nisa.dma_copy(
            dst=TensorView(x_out)
            .slice(dim=0, start=0, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
            src=x_out_sb[:half_d, :, :, :],
        )
        nisa.dma_copy(
            dst=TensorView(x_out)
            .slice(dim=0, start=1, end=d_head, step=2)
            .slice(dim=3, start=tile_start, end=tile_start + tile_size)
            .get_view(),
            src=x_out_sb[half_d:, :, :, :],
        )
    else:
        nisa.dma_copy(dst=x_out[:, :, :, tile_start : tile_start + tile_size], src=x_out_sb)

    return x_out


def RoPE_sbuf(
    x_in_sb: nl.ndarray,
    cos_sb: nl.ndarray,
    sin_sb: nl.ndarray,
    x_out_sb: nl.ndarray,
    convert_from_interleaved: bool = False,
) -> nl.ndarray:
    """
    Apply RoPE on tensors in SBUF (for megakernel fusion).
    Helper function that operates entirely in SBUF without HBM I/O.

    RoPE Formula:
        out[even] = x[even]*cos - x[odd]*sin
        out[odd] = x[odd]*cos + x[even]*sin

    Args:
        x_in_sb: [d_head, B, n_heads, S] @ SBUF - input embeddings
        cos_sb: [d_head//2, B, S] @ SBUF - cosine frequencies
        sin_sb: [d_head//2, B, S] @ SBUF - sine frequencies
        x_out_sb: [d_head, B, n_heads, S] @ SBUF - output buffer
        convert_from_interleaved: convert from interleaved to contiguous layout
            (only for small tensors: B*n_heads*S ≤ gemm_moving_fmax)

    Returns:
        x_out_sb with RoPE applied (modified in-place)

    Notes:
        - Assumes contiguous layout unless convert_from_interleaved=True
        - For large tensors with interleaved layout, use RoPE() with strided DMA
    """

    d_head, B, n_heads, S = x_out_sb.shape
    half_d = d_head // 2

    _validate_rope_inputs(x_in_sb, cos_sb, sin_sb, 'RoPE_sbuf')
    kernel_assert(x_in_sb.dtype == x_out_sb.dtype, 'RoPE_sbuf: dtype mismatch between x_in_sb and x_out_sb')

    # Convert interleaved to contiguous layout if needed
    if convert_from_interleaved:
        convert_to_interleaved_mat = _compute_convert_to_interleaved_mat(x_in_sb)
        x_in_sb = _convert_from_interleaved(x_in_sb, convert_to_interleaved_mat)

    # Copy odd half to separate buffer (required for tensor_tensor base partition alignment)
    sb_odd = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb_odd, src=x_in_sb[half_d:, :, :, :])

    # Allocate buffers for intermediate products
    even_cos = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    odd_cos = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    even_sin = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    odd_sin = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)

    # Compute RoPE: out_even = even*cos - odd*sin, out_odd = odd*cos + even*sin
    # Use access patterns to broadcast cos/sin across n_heads dimension
    nisa.tensor_tensor(
        even_cos,
        x_in_sb[:half_d, :, :, :],
        TensorView(cos_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )
    nisa.tensor_tensor(
        odd_cos,
        sb_odd[:half_d, :, :, :],
        TensorView(cos_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )
    nisa.tensor_tensor(
        even_sin,
        x_in_sb[:half_d, :, :, :],
        TensorView(sin_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )
    nisa.tensor_tensor(
        odd_sin,
        sb_odd[:half_d, :, :, :],
        TensorView(sin_sb).expand_dim(2).broadcast(dim=2, size=n_heads).get_view(),
        nl.multiply,
    )

    nisa.tensor_tensor(x_out_sb[:half_d, :, :, :], even_cos, odd_sin, nl.subtract)
    nisa.tensor_tensor(x_out_sb[half_d:, :, :, :], odd_cos, even_sin, nl.add)

    # Convert back to interleaved layout if needed
    if convert_from_interleaved:
        x_out_sb = _convert_to_interleaved(x_out_sb, convert_to_interleaved_mat)

    return x_out_sb


def _compute_convert_to_interleaved_mat(x_sb: nl.ndarray) -> nl.ndarray:
    """
    Generate permutation matrix for layout conversion.

    Creates matrix P where:
        P @ X: [e0,e1,...,o0,o1,...] -> [e0,o0,e1,o1,...] (contiguous to interleaved)
        P^T @ X: [e0,o0,e1,o1,...] -> [e0,e1,...,o0,o1,...] (interleaved to contiguous)

    Implementation:
        Uses strided access on identity matrix to build permutation.
        For d_head=4: P = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]
        - Even rows (0,2) get 1s in first half columns (0,1)
        - Odd rows (1,3) get 1s in second half columns (2,3)

    Returns:
        [d_head, d_head] @ SBUF - permutation matrix
    """
    d_head, B, n_heads, S = x_sb.shape
    half_d = d_head // 2
    kernel_assert(d_head % 2 == 0, f'_compute_convert_to_interleaved_mat: d_head must be even, got {d_head}')
    kernel_assert(
        B * n_heads * S <= nl.tile_size.gemm_moving_fmax,
        f'_compute_convert_to_interleaved_mat: B*n_heads*S={B * n_heads * S} '
        f'exceeds gemm_moving_fmax={nl.tile_size.gemm_moving_fmax}',
    )

    identity_hbm = nl.shared_constant(np.eye(d_head, dtype=np.float32))
    identity_sb = nl.ndarray((d_head, d_head), dtype=x_sb.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=identity_sb, src=identity_hbm)

    # Extract permutation via strided access pattern:
    # Pattern [[d_head, d_head], [1, 2], [2, half_d]] reads identity with stride=2 in innermost dim
    # For each row i: reads [i[0], i[2], i[4], ...] then [i[1], i[3], i[5], ...]
    # Destination reshape (d_head, 2, half_d) writes: row i -> [[even_cols], [odd_cols]]
    # Result: even rows get 1 in first half, odd rows get 1 in second half
    # This creates P where P@X transforms [e0,e1,...,o0,o1,...] -> [e0,o0,e1,o1,...]
    convert_to_interleaved_mat = nl.ndarray((d_head, d_head), dtype=x_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=convert_to_interleaved_mat.reshape((d_head, 2, half_d)),
        src=identity_sb.ap(pattern=[[d_head, d_head], [1, 2], [2, half_d]]),
        engine=nisa.scalar_engine,
    )

    return convert_to_interleaved_mat


def _convert_from_interleaved(x_sb: nl.ndarray, convert_to_interleaved_mat: nl.ndarray) -> nl.ndarray:
    """
    Convert interleaved to contiguous layout: [e0,o0,e1,o1,...] -> [e0,e1,...,o0,o1,...].
    Uses P^T @ x_sb via nc_matmul. Returns a new buffer (does not modify input).
    """
    d_head, B, n_heads, S = x_sb.shape
    kernel_assert(x_sb.buffer == nl.sbuf, '_convert_from_interleaved: input must be in SBUF')
    kernel_assert(
        B * n_heads * S <= nl.tile_size.gemm_moving_fmax,
        f'_convert_from_interleaved: B*n_heads*S={B * n_heads * S} '
        f'exceeds gemm_moving_fmax={nl.tile_size.gemm_moving_fmax}',
    )

    x_converted_sb = nl.ndarray(x_sb.shape, dtype=x_sb.dtype, buffer=nl.sbuf)
    x_psum = nl.ndarray((d_head, B * n_heads * S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=x_psum, stationary=convert_to_interleaved_mat, moving=x_sb.reshape((d_head, B * n_heads * S)))
    nisa.activation(dst=x_converted_sb, op=nl.copy, data=x_psum.reshape((d_head, B, n_heads, S)))
    return x_converted_sb


def _convert_to_interleaved(x_sb: nl.ndarray, convert_to_interleaved_mat: nl.ndarray) -> nl.ndarray:
    """
    Convert contiguous to interleaved layout: [e0,e1,...,o0,o1,...] -> [e0,o0,e1,o1,...].
    Uses P @ x_sb via nc_matmul (with pre-transpose to compensate for implicit transpose).
    """
    d_head, B, n_heads, S = x_sb.shape
    kernel_assert(x_sb.buffer == nl.sbuf, '_convert_to_interleaved: input must be in SBUF')
    kernel_assert(
        B * n_heads * S <= nl.tile_size.gemm_moving_fmax,
        f'_convert_to_interleaved: B*n_heads*S={B * n_heads * S} '
        f'exceeds gemm_moving_fmax={nl.tile_size.gemm_moving_fmax}',
    )

    # Pre-transpose to compensate for nc_matmul's implicit transpose
    convert_from_interleaved_sb = nl.ndarray((d_head, d_head), dtype=convert_to_interleaved_mat.dtype, buffer=nl.sbuf)
    convert_from_interleaved_psum = nl.ndarray((d_head, d_head), dtype=convert_to_interleaved_mat.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=convert_from_interleaved_psum, data=convert_to_interleaved_mat)
    nisa.tensor_copy(dst=convert_from_interleaved_sb, src=convert_from_interleaved_psum, engine=nisa.scalar_engine)

    x_psum = nl.ndarray((d_head, B * n_heads * S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=x_psum, stationary=convert_from_interleaved_sb, moving=x_sb.reshape((d_head, B * n_heads * S)))
    nisa.tensor_copy(dst=x_sb, src=x_psum.reshape((d_head, B, n_heads, S)), engine=nisa.scalar_engine)
    return x_sb


def _validate_rope_inputs(x_in: nl.ndarray, cos: nl.ndarray, sin: nl.ndarray, func_name: str) -> None:
    """Validate RoPE input shapes and dtypes."""
    d_head, B, n_heads, S = x_in.shape
    half_d = d_head // 2

    kernel_assert(d_head in (64, 128), f'{func_name}: d_head must be 64 or 128, got {d_head}')
    kernel_assert(0 < B <= 64, f'{func_name}: B must be in (0,64], got {B}')
    kernel_assert(0 < S <= 512, f'{func_name}: S must be in (0,512], got {S}')
    kernel_assert(0 < n_heads <= 16, f'{func_name}: n_heads must be in (0,16], got {n_heads}')
    kernel_assert(
        tuple(cos.shape) == (half_d, B, S),
        f'{func_name}: cos.shape expected ({half_d},{B},{S}), got {cos.shape}',
    )
    kernel_assert(
        tuple(sin.shape) == (half_d, B, S),
        f'{func_name}: sin.shape expected ({half_d},{B},{S}), got {sin.shape}',
    )
    kernel_assert(cos.dtype == sin.dtype, f'{func_name}: cos/sin dtype mismatch')



def rope(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to Q using the NKI kernel.

    Matches the Triton wrapper signature:
        Q   : (batch, seq_len, n_heads, head_dim)
        cos : (seq_len, head_dim // 2)
        sin : (seq_len, head_dim // 2)

    Returns a tensor with the same shape as Q on CPU.

    The RoPE kernel requires S <= 512 and n_heads <= 16, so large inputs are
    tiled along both dimensions and processed in chunks.
    """
    device = xm.xla_device()
    B, S, n_heads, d_head = Q.shape
    half_d = d_head // 2

    S_TILE = 512
    # SBUF capacity limits n_heads per tile: empirically n_heads*B*S ≤ ~16K elements.
    # With B=4, S=512 → n_heads ≤ 8.  Use 8 as a safe default for typical B values.
    H_TILE = 8

    out = torch.empty_like(Q)

    for h0 in range(0, n_heads, H_TILE):
        h1 = min(h0 + H_TILE, n_heads)
        h_size = h1 - h0

        for s0 in range(0, S, S_TILE):
            s1 = min(s0 + S_TILE, S)
            s_size = s1 - s0

            Q_slice = Q[:, s0:s1, h0:h1, :]   # (B, s_size, h_size, d_head)
            cos_sl  = cos[s0:s1, :]             # (s_size, half_d)
            sin_sl  = sin[s0:s1, :]             # (s_size, half_d)

            # Pad to full tile dims if the slice is smaller (tail tile)
            if s_size < S_TILE or h_size < H_TILE:
                Q_k = torch.zeros((B, S_TILE, H_TILE, d_head), dtype=Q.dtype)
                Q_k[:, :s_size, :h_size, :] = Q_slice
                cos_k_t = torch.zeros((S_TILE, half_d), dtype=cos.dtype)
                cos_k_t[:s_size, :] = cos_sl
                sin_k_t = torch.zeros((S_TILE, half_d), dtype=sin.dtype)
                sin_k_t[:s_size, :] = sin_sl
            else:
                Q_k, cos_k_t, sin_k_t = Q_slice, cos_sl, sin_sl

            # (B, S_TILE, H_TILE, d_head) -> (d_head, B, H_TILE, S_TILE)
            x_in = Q_k.permute(3, 0, 2, 1).contiguous().to(device)
            # (S_TILE, half_d) -> (half_d, B, S_TILE)
            cos_k = cos_k_t.T.unsqueeze(1).expand(half_d, B, S_TILE).contiguous().to(device)
            sin_k = sin_k_t.T.unsqueeze(1).expand(half_d, B, S_TILE).contiguous().to(device)

            result = RoPE(x_in, cos_k, sin_k)
            # (d_head, B, H_TILE, S_TILE) -> (B, S_TILE, H_TILE, d_head)
            result_cpu = result.cpu().permute(1, 3, 2, 0).contiguous()
            out[:, s0:s1, h0:h1, :] = result_cpu[:, :s_size, :h_size, :]

    return out
# END-HARNESS
