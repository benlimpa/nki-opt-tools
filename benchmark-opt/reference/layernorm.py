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


@nki.jit(platform_target="trn1")
def nki_layer_norm_kernel(x, weight, bias, mask, eps, N_original):
    M, N = x.shape
    y = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    TILE_M = 128
    TILE_N = 512
    NUM_N = N // TILE_N

    # ones_row (1, TILE_M) used as stationary in nc_matmul to broadcast a
    # (1, TILE_N) tile to (TILE_M, TILE_N): stationary^T @ moving.
    ones_row = nl.ndarray((1, TILE_M), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)

    eps_vec = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_vec, value=eps)

    # Running accumulators (copy-back pattern avoids in-place dma_compute).
    sum_acc   = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    sum_tmp   = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    sumsq_acc = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    sumsq_tmp = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    mean_vec  = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
    rstd      = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)

    for m in nl.affine_range(M // TILE_M):

        # ── Pass 1: accumulate sum(x) across N-tiles ─────────────────────────
        nisa.memset(dst=sum_acc, value=0.0)
        for n in nl.static_range(NUM_N):
            x_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])
            partial = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=partial, op=nl.add, data=x_tile, axis=1, keepdims=True)
            nisa.dma_compute(dst=sum_tmp, srcs=[sum_acc, partial], scales=[1.0, 1.0], reduce_op=nl.add)
            nisa.dma_copy(dst=sum_acc, src=sum_tmp)

        # mean = sum(x) / N_original  (padded zeros don't contribute)
        nisa.activation(dst=mean_vec, op=nl.copy, data=sum_acc, scale=1.0/N_original)

        # ── Pass 2: accumulate sum((x-mean)^2 * mask) ────────────────────────
        nisa.memset(dst=sumsq_acc, value=0.0)
        for n in nl.static_range(NUM_N):
            x_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])

            m_tile = nl.ndarray((1, TILE_N), dtype=mask.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=m_tile, src=mask[0:1, n*TILE_N:(n+1)*TILE_N])
            m_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=m_psum, stationary=ones_row, moving=m_tile)
            m_exp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=m_exp, op=nl.copy, data=m_psum)

            x_centered = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_centered, data=x_tile, op0=nl.subtract, operand0=mean_vec)
            nisa.tensor_tensor(dst=x_centered, data1=x_centered, data2=m_exp, op=nl.multiply)

            x_sq = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=x_sq, op=nl.square, data=x_centered)
            partial = nl.ndarray((TILE_M, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=partial, op=nl.add, data=x_sq, axis=1, keepdims=True)
            nisa.dma_compute(dst=sumsq_tmp, srcs=[sumsq_acc, partial], scales=[1.0, 1.0], reduce_op=nl.add)
            nisa.dma_copy(dst=sumsq_acc, src=sumsq_tmp)

        # rstd = rsqrt(sum(x_centered²) / N_original + eps)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=sumsq_acc, scale=1.0/N_original, bias=eps_vec)

        # ── Pass 3: normalize, scale, add bias, store ─────────────────────────
        for n in nl.static_range(NUM_N):
            x_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N])

            m_tile = nl.ndarray((1, TILE_N), dtype=mask.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=m_tile, src=mask[0:1, n*TILE_N:(n+1)*TILE_N])
            m_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=m_psum, stationary=ones_row, moving=m_tile)
            m_exp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=m_exp, op=nl.copy, data=m_psum)

            w_tile = nl.ndarray((1, TILE_N), dtype=weight.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_tile, src=weight[0:1, n*TILE_N:(n+1)*TILE_N])
            w_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=w_psum, stationary=ones_row, moving=w_tile)
            w_exp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=w_exp, op=nl.copy, data=w_psum)

            b_tile = nl.ndarray((1, TILE_N), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b_tile, src=bias[0:1, n*TILE_N:(n+1)*TILE_N])
            b_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=b_psum, stationary=ones_row, moving=b_tile)
            b_exp = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=b_exp, op=nl.copy, data=b_psum)

            # x_centered = (x - mean) * mask
            x_centered = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=x_centered, data=x_tile, op0=nl.subtract, operand0=mean_vec)
            nisa.tensor_tensor(dst=x_centered, data1=x_centered, data2=m_exp, op=nl.multiply)

            # y = x_centered * rstd * weight + bias
            y_tile_f32 = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=y_tile_f32, data=x_centered,
                                      op0=nl.multiply, operand0=rstd,
                                      op1=nl.multiply, operand1=w_exp)
            nisa.tensor_tensor(dst=y_tile_f32, data1=y_tile_f32, data2=b_exp, op=nl.add)

            y_tile = nl.ndarray((TILE_M, TILE_N), dtype=x.dtype, buffer=nl.sbuf)
            nisa.activation(dst=y_tile, op=nl.copy, data=y_tile_f32)

            nisa.dma_copy(dst=y[m*TILE_M:(m+1)*TILE_M, n*TILE_N:(n+1)*TILE_N], src=y_tile)

    return y

def layer_norm(x, weight, bias, eps):
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

    b_padded = torch.zeros((1, N_pad), dtype=bias.dtype)
    b_padded[0, :N] = bias.reshape(-1)

    mask = torch.zeros((1, N_pad), dtype=torch.float32)
    mask[0, :N] = 1.0

    device = torch_xla.device()
    x_dev = x_padded.to(device)
    w_dev = w_padded.to(device)
    b_dev = b_padded.to(device)
    mask_dev = mask.to(device)

    y_dev = nki_layer_norm_kernel(x_dev, w_dev, b_dev, mask_dev, eps, N)

    y_cpu = y_dev.to("cpu")
    y_cropped = y_cpu[:M, :N]

    return y_cropped.reshape(original_shape)
