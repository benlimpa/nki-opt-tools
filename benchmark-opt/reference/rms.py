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

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch_xla
from torch_xla.core import xla_model as xm

@nki.jit(platform_target="trn1")
def nki_rms_norm_kernel(x, weight, eps, N_original):
    M, N = x.shape
    y = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    TILE_M = 128
    TILE_N = 512
    NUM_N = N // TILE_N

    # ones_row (1, TILE_M) used as stationary in nc_matmul to broadcast a
    # (1, TILE_N) weight tile to (TILE_M, TILE_N): stationary^T @ moving.
    ones_row = nl.ndarray((1, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    eps_vec = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_vec, value=eps)

    # Ping-pong accumulators so dma_compute src and dst are always distinct.
    sum_acc = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_tmp = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)

    for m in nl.affine_range(M // TILE_M):
        # ── Pass 1: accumulate sum(x²) over all N-tiles ──────────────────────
        nisa.memset(dst=sum_acc, value=0.0)

        for n in nl.static_range(NUM_N):
            x_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])

            x_sq = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=x_sq, data=x_tile, op=nl.square)

            partial = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=partial, data=x_sq, op=nl.add, axis=1, keepdims=True)

            nisa.dma_compute(dst=sum_tmp, srcs=[sum_acc, partial], scales=[1.0, 1.0], reduce_op=nl.add)
            nisa.dma_copy(dst=sum_acc, src=sum_tmp)

        # inv_var = rsqrt(sum(x²) / N_original + eps)
        inv_var = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv_var, data=sum_acc, op=nl.rsqrt, scale=1.0/N_original, bias=eps_vec)

        # ── Pass 2: normalize and scale each N-tile ───────────────────────────
        for n in nl.static_range(NUM_N):
            x_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])

            w_tile = nl.ndarray((1, TILE_N), dtype=weight.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_tile, src=weight[0:1, n*TILE_N:(n+1)*TILE_N])

            w_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=w_psum, stationary=ones_row, moving=w_tile)
            w_exp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=w_exp, data=w_psum, op=nl.copy)

            # y = x * inv_var * w  (inv_var (TILE_M,1) broadcasts across free dim)
            y_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=y_tile, data=x_tile,
                                      op0=nl.multiply, operand0=inv_var,
                                      op1=nl.multiply, operand1=w_exp)

            nisa.dma_copy(dst=y[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N], src=y_tile)

    return y

# Wrapper Function:
def rms_layernorm(x, weight, eps=1e-5):
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    M, N = x_2d.shape

    TILE_M = 128
    TILE_N = 512

    M_pad = ((M + TILE_M - 1) // TILE_M) * TILE_M
    N_pad = ((N + TILE_N - 1) // TILE_N) * TILE_N

    x_padded = torch.zeros((M_pad, N_pad), dtype=x.dtype)
    x_padded[:M, :N] = x_2d

    w_padded = torch.zeros((1, N_pad), dtype=weight.dtype)
    w_padded[0, :N] = weight.reshape(-1)

    device = torch_xla.device()
    x_dev = x_padded.to(device)
    w_dev = w_padded.to(device)

    y_dev = nki_rms_norm_kernel(x_dev, w_dev, eps, N)

    y_cpu = y_dev.to("cpu")
    y_cropped = y_cpu[:M, :N]

    return y_cropped.reshape(original_shape)
