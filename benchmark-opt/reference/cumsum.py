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

"""Cumulative sum kernel for NKI. Computes cumulative sum along the last dimension."""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_xla.core import xla_model as xm

import math
from typing import List, Tuple, Union

from nki.language import NKIObject


class TiledRangeIterator(NKIObject):
    """
    Represents a single tile in a tiled range.

    - size: The size of this tile
    - index: The index of this tile in the range
    - start_offset: Absolute starting offset (i.e., subtile offset is calculated properly)
    - end_offset: Absolute ending offset (i.e., subtile offset is calculated properly)
    """

    def __init__(self, tile_size: int, tile_index: int, start_offset: int, end_offset: int):
        """
        Initialize a TiledRangeIterator.

        Args:
            tile_size: The size of this tile
            tile_index: The index of this tile (0-based)
            start_offset: The starting offset in the original dimension
            end_offset: The ending offset in the original dimension
        """
        self.size = tile_size
        self.index = tile_index
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __repr__(self) -> str:
        return f"TiledRangeIterator(size={self.size}, index={self.index}, start_offset={self.start_offset}, end_offset={self.end_offset})"


def TiledRange(size: Union[int, TiledRangeIterator], tile_size: int) -> Tuple[TiledRangeIterator, ...]:
    """
    Divides a dimension into tiles and returns a tuple of TiledRangeIterators.

    Args:
        size: Either an integer representing the total size to tile,
              or a TiledRangeIterator for nested tiling
        tile_size: The size of each tile

    Returns:
        A tuple of TiledRangeIterator objects

    Example:
        >>> tiles = TiledRange(300, 128)
        >>> for tile in tiles:
        ...     print(f"size={tile.size}, index={tile.index}, start_offset={tile.start_offset}")
        size=128, index=0, start_offset=0, end_offset=128
        size=128, index=1, start_offset=128, end_offset=256
        size=44, index=2, start_offset=256, end_offset=300

    Supports nested tiling:
        >>> outer_tiles = TiledRange(300, 128)
        >>> for outer_tile in outer_tiles:
        ...     inner_tiles = TiledRange(outer_tile, 64)
        ...     for inner_tile in inner_tiles:
        ...         print(f"outer_idx={outer_tile.index}, inner_idx={inner_tile.index}, size={inner_tile.size}")
    """
    if isinstance(size, TiledRangeIterator):
        # Subtiled case: use the size of the TiledRangeIterator
        total_size = size.size
        base_offset = size.start_offset
    else:
        total_size = size
        base_offset = 0

    # Calculate the number of tiles needed (ceiling division)
    num_tiles = math.ceil(total_size / tile_size)

    # Create the tuple of TiledRangeIterators
    iterators = []
    for i in range(num_tiles):
        relative_offset = i * tile_size
        # Last tile may be smaller
        current_tile_size = min(tile_size, total_size - relative_offset)
        # Add base_offset to make the start_offset absolute
        start_offset = base_offset + relative_offset
        end_offset = base_offset + relative_offset + current_tile_size
        iterators.append(TiledRangeIterator(current_tile_size, i, start_offset, end_offset))

    return tuple(iterators)

P_MAX = 128  # Partition dimension max (nl.tile_size.pmax returns non-int, compiler requires Python int for shapes)
F_TILE_SIZE = 2048

def div_ceil(n, d):
    return (n + d - 1) // d

def reduce(op='mul', input: List = None, initial_value=None):
    """
    Perform reduction operation on a list of values.

    Applies a reduction operation (multiply, add, min, or max) across all elements
    in the input list, starting from an initial value.

    Args:
        op (str): Reduction operation - 'mul', 'add', 'min', or 'max'.
        input (List): List of values to reduce.
        initial_value: Starting value for the reduction.

    Returns:
        result: Reduced value after applying operation to all input elements.

    Notes:
        - Supports 'mul', 'add', 'min', and 'max' operations
        - Requires both input and initial_value to be non-None
        - Used for computing products, sums, minimums, or maximums of values

    Pseudocode:
        validate initial_value is not None
        validate input is not None
        validate op in ['mul', 'add', 'max', 'min']
        result = initial_value
        for value in input:
            if op == 'mul':
                result = result * value
            elif op == 'add':
                result = result + value
            elif op == 'min':
                result = min(result, value)
            elif op == 'max':
                result = max(result, value)
        return result
    """
    assert(initial_value is not None, f"initial_value need to set")
    assert(input is not None, f"input need to be set")
    assert(op in ['mul', 'add', 'max', 'min'], f"only op {op} is supported")
    for value in input:
        if op == 'mul':
            initial_value = initial_value * value
        elif op == 'add':
            initial_value = initial_value + value
        elif op == 'min':
            initial_value = min(initial_value, value)
        elif op == 'max':
            initial_value = max(initial_value, value)
    return initial_value

@nki.jit(platform_target="trn1")
def cumsum_kernel(x: nl.ndarray, axis: int = -1) -> nl.ndarray:
    """
    Compute cumulative sum along the last dimension.

    This kernel computes the cumulative sum of elements along the last dimension
    of the input tensor. Optimized for batch sizes up to 2048 and hidden dimension
    sizes up to 8192. Supports 3D inputs with sequence length up to 10.

    Dimensions:
        B: Batch size, up to 2048
        S: Sequence length for 3D inputs, up to 10
        H: Hidden dimension size, up to 8192
        outer_dim: Product of all dimensions except the last (B for 2D, B * S for 3D)

    Args:
        x (nl.ndarray): Input HBM tensor of shape [B, H] for 2D or [B, S, H] for 3D.
            dtype can be any NKI-supported type.
        axis (int): Axis along which to compute cumsum. Must be -1 or the last
            dimension index. Default: -1.

    Returns:
        nl.ndarray: Output HBM tensor with same shape and dtype as input, containing
            cumulative sums along the last dimension.

    Notes:
        - Only supports cumsum along the last dimension
        - Uses float32 accumulation internally for numerical stability
        - For very long hidden dimensions (>5K), expect ~1e-2 absolute error due to fp32 accumulation

    Pseudocode:
        # Reshape to 2D: (outer_dim, hidden)
        x_2d = x.reshape(outer_dim, hidden)
        y_2d = zeros_like(x_2d)

        for p_tile_idx in range(num_partition_tiles):
            init = 0  # carry from previous free tile
            for f_tile_idx in range(num_free_tiles):
                tile = load(x_2d[p_tile, f_tile])
                # cumsum: result[i] = result[i-1] + tile[i]
                result = tensor_tensor_scan(ones, tile, init, multiply, add)
                store(y_2d[p_tile, f_tile], result)
                init = result[:, -1]  # carry forward last column

        return y_2d.reshape(x.shape)
    """
    rank = len(x.shape)

    # Normalize axis to positive index
    if axis < 0:
        axis = rank + axis

    # Only support last dimension
    assert axis == rank - 1, f"Only support cumsum over last dim, got {axis=}, {rank=}"

    # Reshape to 2D: (outer_dim, last_dim)
    x_shape = x.shape
    outer_dim = reduce(op='mul', input=list(x_shape[:-1]), initial_value=1) if rank > 1 else 1
    last_dim = x_shape[-1]
    shape_2d = (outer_dim, last_dim)

    x_2d = x.reshape(shape_2d)

    # Allocate output on HBM
    y = nl.ndarray(x_shape, dtype=x.dtype, buffer=nl.shared_hbm)
    y_2d = y.reshape(shape_2d)

    num_f_tiles = div_ceil(last_dim, F_TILE_SIZE)

    # Process partition tiles
    for p_tile in TiledRange(outer_dim, P_MAX):
        # Initialize carry for this partition tile
        # Note: float32 used for numerical stability; tensor_tensor_scan auto-casts to float32 internally
        init_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=init_sb, value=0.0)

        # Allocate ones tensor for scan (data0 * prev + data1)
        ones_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_sb, value=1.0)

        # Sequential loop over free dimension tiles (must be sequential for cumsum)
        for f_tile_idx in nl.sequential_range(num_f_tiles):
            f_start = f_tile_idx * F_TILE_SIZE
            f_end = min(f_start + F_TILE_SIZE, last_dim)
            f_size = f_end - f_start

            # Load input tile to SBUF
            data_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=data_sb[0 : p_tile.size, 0:f_size],
                src=x_2d[p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
            )

            # Compute cumsum using tensor_tensor_scan
            # result[i] = ones[i] * result[i-1] + data[i] = result[i-1] + data[i]
            result_sb = nl.ndarray((P_MAX, F_TILE_SIZE), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=result_sb[0 : p_tile.size, 0:f_size],
                data0=ones_sb[0 : p_tile.size, 0:f_size],
                data1=data_sb[0 : p_tile.size, 0:f_size],
                initial=init_sb[0 : p_tile.size, 0:1],
                op0=nl.multiply,
                op1=nl.add,
            )

            # Store result to HBM
            nisa.dma_copy(
                dst=y_2d[p_tile.start_offset : p_tile.start_offset + p_tile.size, f_start:f_end],
                src=result_sb[0 : p_tile.size, 0:f_size],
            )

            # Update init for next tile: carry forward last column
            if f_tile_idx + 1 < num_f_tiles:
                last_col_idx = f_size - 1
                nisa.tensor_copy(
                    dst=init_sb[0 : p_tile.size, 0:1], src=result_sb[0 : p_tile.size, last_col_idx : last_col_idx + 1]
                )

    return y


# Wrapper Function:
def cumsum(g, chunk_size, reverse=False, scale=None):
    """
    Interface for the chunk-local cumsum kernel.
    Expected input shape: (B, H, T)

    Computes cumsum independently within each chunk of size chunk_size along the T dim.
    Supports reverse cumsum and optional scaling.
    """
    B, H, T = g.shape
    T_pad = ((T + chunk_size - 1) // chunk_size) * chunk_size
    num_chunks = T_pad // chunk_size

    # Pad last dim to multiple of chunk_size if needed
    if T_pad > T:
        g_padded = torch.zeros((B, H, T_pad), dtype=g.dtype)
        g_padded[:, :, :T] = g
    else:
        g_padded = g

    # Reshape to (B*H*num_chunks, chunk_size) so each row is one chunk —
    # the kernel's cumsum over the last dim gives chunk-local cumsum.
    g_reshaped = g_padded.reshape(B * H * num_chunks, chunk_size)

    device = xm.xla_device()
    g_reshaped = g_reshaped.to(device)

    if reverse:
        g_flipped = torch.flip(g_reshaped, dims=[-1])
        result = cumsum_kernel(g_flipped, axis=-1)
        result = torch.flip(result, dims=[-1])
    else:
        result = cumsum_kernel(g_reshaped, axis=-1)

    if scale is not None:
        result = result * scale

    result = result.to("cpu")

    # Reshape back to (B, H, T_pad) and trim to original T
    result_padded = result.reshape(B, H, T_pad)
    return result_padded[:, :, :T]
