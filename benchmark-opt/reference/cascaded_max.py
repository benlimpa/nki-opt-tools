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
Cascaded max reduction kernel from nki-library/src/nkilib_src/nkilib/core/max/.

Finds the global maximum value and its index across the last dimension of a
tensor using multi-stage parallel reduction optimised for the NeuronCore
architecture.  Utility helpers (kernel_assert, reduce, predicated_folded_load,
div_ceil, get_verified_program_sharding_info) are inlined from the same library.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl


# ---------------------------------------------------------------------------
# Inlined utilities (from nkilib/core/utils/ and nkilib/core/max/)
# ---------------------------------------------------------------------------

def _kernel_assert(condition: bool, error_text: str):
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: "
        f"{error_text} - Please check the validation message and adjust "
        f"kernel inputs accordingly"
    )


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _get_program_sharding_info() -> Tuple[int, int, int]:
    grid_ndim = nl.program_ndim()
    n_prgs, prg_id = (
        (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    )
    return grid_ndim, n_prgs, prg_id


def _get_verified_program_sharding_info(
    kernel_name: str = "",
    allowed_ndims: Optional[Tuple[int, ...]] = None,
    max_sharding: Optional[int] = None,
) -> Tuple[int, int, int]:
    grid_ndim, n_prgs, prg_id = _get_program_sharding_info()
    return grid_ndim, n_prgs, prg_id


def _reduce(op: str = 'mul', input_list: Optional[List] = None, initial_value=None):
    supported_ops = ['mul', 'add', 'max', 'min']
    _kernel_assert(initial_value is not None, "initial_value must be set")
    _kernel_assert(input_list is not None, "input_list must be set")
    _kernel_assert(op in supported_ops, f"only ops in {supported_ops} are supported, got {op}")
    for element in input_list:
        if op == 'mul':
            initial_value = initial_value * element
        elif op == 'add':
            initial_value = initial_value + element
        elif op == 'min':
            initial_value = min(initial_value, element)
        elif op == 'max':
            initial_value = max(initial_value, element)
    return initial_value


def _predicated_folded_load(
    data_hbm: nl.ndarray,
    fold_factor: int,
    program_id: int = 0,
    n_programs: int = 1,
    fill_value: float = -9948.0,
    data_sb: Optional[nl.ndarray] = None,
    batch_start: Optional[int] = None,
    batch_end: Optional[int] = None,
) -> Optional[nl.ndarray]:
    _kernel_assert(len(data_hbm.shape) == 2, "Expected input tensor to have shape [B, N]")
    batch_size, n = data_hbm.shape

    batch_start = batch_start if batch_start is not None else 0
    batch_end = batch_end if batch_end is not None else batch_size
    batch_size_in_range = batch_end - batch_start
    batch_size_sharded = (batch_size_in_range + n_programs - 1) // n_programs
    batch_line_offset = batch_start + program_id * batch_size_sharded

    P_MAX = nl.tile_size.pmax
    _kernel_assert(
        batch_size_sharded * fold_factor <= P_MAX,
        f"fold_factor x max local batch size ({batch_size_sharded}) exceeds tile limit {P_MAX}",
    )

    n_folded = math.ceil(n / fold_factor)

    return_out = False
    if data_sb is None:
        return_out = True
        data_sb = nl.ndarray(
            (batch_size_sharded * fold_factor, n_folded), dtype=data_hbm.dtype, buffer=nl.sbuf
        )
    nisa.memset(data_sb, fill_value)

    if n == fold_factor * n_folded:
        src_hbm_reshape = data_hbm.reshape((batch_size * fold_factor, n_folded))
        base_offset = batch_line_offset * fold_factor
        batch_bound = min(batch_size_sharded * fold_factor, batch_size * fold_factor - base_offset)
        ix, iy = nl.ds(0, batch_bound), nl.ds(0, n_folded)
        ix_src = nl.ds(base_offset, batch_bound)
        nisa.dma_copy(src=src_hbm_reshape[ix_src, iy], dst=data_sb[ix, iy])
        if return_out:
            return data_sb
        else:
            return None

    src_hbm_flat = data_hbm.reshape((batch_size * n,))
    batch_size_sharded_bounded = min(batch_size_sharded, batch_size - batch_line_offset)
    for batch_line_idx in nl.affine_range(batch_size_sharded_bounded):
        row_idx = batch_line_idx + batch_line_offset
        base_idx = row_idx * n
        remainder = n % n_folded
        ix_0_dst = nl.ds(batch_line_idx * fold_factor, fold_factor)
        iy_0_dst = nl.ds(0, remainder)
        ix_1_dst = nl.ds(batch_line_idx * fold_factor, fold_factor - 1)
        iy_1_dst = nl.ds(remainder, n_folded - remainder)
        src_ap_0 = [[n_folded, fold_factor], [1, remainder]]
        src_ap_1 = [[n_folded, fold_factor - 1], [1, (n_folded - remainder)]]
        nisa.dma_copy(dst=data_sb[ix_0_dst, iy_0_dst], src=src_hbm_flat.ap(pattern=src_ap_0, offset=base_idx))
        nisa.dma_copy(
            dst=data_sb[ix_1_dst, iy_1_dst],
            src=src_hbm_flat.ap(pattern=src_ap_1, offset=base_idx + remainder),
        )
    if return_out:
        return data_sb
    return None


# ---------------------------------------------------------------------------
# Kernel implementation (from nkilib/core/max/cascaded_max.py)
# ---------------------------------------------------------------------------

@dataclass
class CascadedMaxConfig(nl.NKIObject):
    """Configuration for the cascaded max reduction algorithm."""

    def __init__(self, inp_shape: Tuple, inp_dtype=None) -> None:
        _kernel_assert(inp_dtype is not None, "inp_dtype must be set")
        self.inp_dtype = inp_dtype
        self.index_dtype = nl.uint32

        self.BxS = _reduce('mul', inp_shape[:-1], 1)
        self.vocab_size = inp_shape[-1]
        self.out_shape = list(inp_shape[:-1]) + [1]
        self.inp_shape = [self.BxS, self.vocab_size]

        shard_info = _get_verified_program_sharding_info("max", (0, 1), 2)
        self.n_prgs = shard_info[1]
        self.prg_id = shard_info[2]
        if self.BxS == 1:
            self.prg_id = 0
            self.n_prgs = 1
        self.per_lnc_BxS = _div_ceil(self.BxS, self.n_prgs)
        self.per_lnc_BxS_unpadded = min(self.per_lnc_BxS, self.BxS - self.per_lnc_BxS * self.prg_id)
        _kernel_assert(
            self.BxS_dim_valid(),
            f"cascaded_max expects BxS per worker {self.per_lnc_BxS} <= max_pdim {nl.tile_size.pmax}",
        )
        _kernel_assert(self.inp_shape_valid(), f"cascaded_max expects input to be 2D, got {len(inp_shape)}")

        const_info = self._calculate_cascading_constants()
        self.n_stages = const_info[0]
        self.stage_free_size = const_info[1]
        self.padded_vocab_size = const_info[2]

    def _calculate_cascading_constants(self) -> Tuple[int, int, int]:
        P_MAX = nl.tile_size.pmax
        max_n_stages = math.floor(P_MAX // self.per_lnc_BxS)
        n_stages = min(max_n_stages, self.vocab_size // P_MAX)
        chunk_size = math.ceil(self.vocab_size / n_stages)
        padded_vocab_size = chunk_size * n_stages
        return n_stages, chunk_size, padded_vocab_size

    def inp_shape_valid(self) -> bool:
        return len(self.inp_shape) == 2

    def BxS_dim_valid(self) -> bool:
        return self.per_lnc_BxS <= nl.tile_size.pmax


def _grouped_reduce_max(
    input_tensor: nl.ndarray,
    index: nl.ndarray,
    fold_factor: int,
) -> Tuple[nl.ndarray, nl.ndarray]:
    b, n = input_tensor.shape
    elts_per_fold = n // fold_factor
    reshaped_shape = (b, fold_factor, elts_per_fold)
    repeat_interleave_ap = [[fold_factor, b], [1, fold_factor], [0, elts_per_fold]]
    mask = nl.ndarray((reshaped_shape), dtype=nl.uint8, buffer=nl.sbuf)
    reduced_max = nl.ndarray((b, fold_factor), dtype=input_tensor.dtype, buffer=nl.sbuf)
    masked_index = nl.ndarray((b, n), dtype=index.dtype, buffer=nl.sbuf)
    final_index = nl.ndarray((b, fold_factor), dtype=index.dtype, buffer=nl.sbuf)

    nisa.tensor_reduce(dst=reduced_max, op=nl.maximum, data=input_tensor.reshape(reshaped_shape), axis=1)
    nisa.tensor_tensor(
        dst=mask,
        data1=input_tensor.reshape(reshaped_shape),
        op=nl.equal,
        data2=reduced_max.ap(repeat_interleave_ap),
    )
    nisa.tensor_tensor(dst=masked_index, data1=mask.reshape((b, n)), data2=index, op=nl.multiply)
    nisa.tensor_reduce(dst=final_index, op=nl.maximum, data=masked_index.reshape(reshaped_shape), axis=1)
    return reduced_max, final_index


def _repeat(n_stages: int, stage_size: int, repeat_count: int) -> nl.ndarray:
    offset_r = nl.ndarray((1, repeat_count * n_stages), dtype=nl.uint32, buffer=nl.sbuf)
    pattern = [[0, 1], [0, repeat_count], [stage_size, n_stages]]
    nisa.iota(pattern=pattern, dst=offset_r)
    return offset_r


def _cascaded_max_core(
    inp: nl.ndarray,
    config: CascadedMaxConfig,
) -> Tuple[nl.ndarray, nl.ndarray]:
    n_stages = config.n_stages
    BxS_size = config.per_lnc_BxS
    n_programs, program_id = config.n_prgs, config.prg_id
    stage_free_size = config.stage_free_size

    total_partition_dim = n_stages * BxS_size

    i_p = nl.ds(0, total_partition_dim)
    i_f = nl.ds(0, stage_free_size)

    values = _predicated_folded_load(
        data_hbm=inp, fold_factor=n_stages, n_programs=n_programs, program_id=program_id
    )
    _kernel_assert(
        values.shape == (total_partition_dim, stage_free_size),
        f"shape mismatch expected {(total_partition_dim, stage_free_size)} but got {values.shape}",
    )

    value = nl.ndarray((total_partition_dim, 1), dtype=inp.dtype)
    ind_buf = nl.ndarray((total_partition_dim, 8), dtype=config.index_dtype)
    broadcast_ap = [[1, total_partition_dim], [0, 8]]

    nisa.tensor_reduce(op=nl.maximum, data=values[i_p, i_f], dst=value[...], axis=1)
    nisa.nc_find_index8(data=values[:, :], vals=value.ap(broadcast_ap), dst=ind_buf[...])

    ind_offset = _repeat(n_stages, stage_free_size, BxS_size)

    i_identity_nx1_p = nl.ds(0, total_partition_dim)
    i_identity_nx1_x = nl.ds(0, total_partition_dim)
    identity_load = nl.ndarray((total_partition_dim, total_partition_dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.memset(dst=identity_load, value=1.0)
    pattern = [[1, total_partition_dim]]
    nisa.affine_select(
        dst=identity_load[...],
        pattern=pattern,
        channel_multiplier=-1,
        cmp_op=nl.equal,
        on_true_tile=identity_load[i_identity_nx1_p, i_identity_nx1_x],
        on_false_value=0.0,
    )
    ind_t = nl.ndarray((1, total_partition_dim), dtype=value.dtype, buffer=nl.psum)
    ind_buf_float = nl.ndarray((total_partition_dim, 1), dtype=value.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(src=ind_buf[:, 0:1], dst=ind_buf_float)
    nisa.nc_matmul(dst=ind_t[...], stationary=ind_buf_float, moving=identity_load, is_transpose=True)

    ind_shifted = nl.ndarray((1, total_partition_dim), dtype=value.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ind_shifted, data1=ind_t, op=nl.add, data2=ind_offset)

    value_psum = nl.ndarray((1, total_partition_dim), dtype=value.dtype, buffer=nl.psum)
    nisa.nc_matmul(dst=value_psum, stationary=value[:, 0:1], moving=identity_load, is_transpose=True)
    final_max, global_index = _grouped_reduce_max(value_psum, ind_shifted, fold_factor=BxS_size)
    return final_max, global_index


@nki.jit(platform_target="trn1")
def cascaded_max_kernel(input_tensor: nl.ndarray) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Find global maximum value and its index across input tensor using cascaded reduction.

    Copied from nki-library/src/nkilib_src/nkilib/core/max/cascaded_max.py.

    Args:
        input_tensor (nl.ndarray): [B, S, V] or [BxS, V], Input tensor in HBM.

    Returns:
        Tuple[nl.ndarray, nl.ndarray]:
            - max_values:  shape with last dim replaced by 1, e.g. [B, S, 1]
            - max_indices: same shape, uint32 indices into the last dimension
    """
    config = CascadedMaxConfig(input_tensor.shape, inp_dtype=input_tensor.dtype)
    val, indx = _cascaded_max_core(input_tensor.reshape(config.inp_shape), config)

    BxS_size = config.per_lnc_BxS
    BxS_global = config.BxS
    n_programs, program_id = config.n_prgs, config.prg_id

    output_shape_t = (1, BxS_global)
    max_values = nl.ndarray(output_shape_t, dtype=val.dtype, buffer=nl.shared_hbm, name="max_values")
    max_indices = nl.ndarray(output_shape_t, dtype=config.index_dtype, buffer=nl.shared_hbm, name="max_indices")
    unpadded_bxs_size = min(BxS_size, BxS_global - BxS_size * program_id)
    src_ap = [[BxS_size, 1], [1, unpadded_bxs_size]]
    dst_ap = [[BxS_global, 1], [1, unpadded_bxs_size]]
    offset = program_id * BxS_size
    nisa.dma_copy(dst=max_values.ap(dst_ap, offset=offset), src=val.ap(src_ap))
    nisa.dma_copy(dst=max_indices.ap(dst_ap, offset=offset), src=indx.ap(src_ap))

    return max_values.reshape(config.out_shape), max_indices.reshape(config.out_shape)


# ---------------------------------------------------------------------------
# Wrapper function (Triton-compatible interface)
# ---------------------------------------------------------------------------

def cascaded_max(x):
    """Find the maximum value and its index over the last dimension of x.

    This is a drop-in NKI replacement for ``torch.max(x, dim=-1, keepdim=True)``.

    Args:
        x (torch.Tensor): Input tensor of any shape [..., V] on CPU.
            V (vocabulary / reduction dimension) must satisfy:
            - V >= 128 (at least one tile)
            - BxS (product of all dims except last) <= 128

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - max_values:  shape [..., 1], float32, maximum values
            - max_indices: shape [..., 1], int64, indices of the maxima
    """
    import torch
    import torch_xla
    from torch_xla.core import xla_model as xm

    original_shape = x.shape
    device = torch_xla.device()
    x_dev = x.to(device)

    max_vals_dev, max_idx_dev = cascaded_max_kernel(x_dev)

    max_vals = max_vals_dev.cpu()
    max_idx = max_idx_dev.cpu().to(torch.int64)

    return max_vals, max_idx
