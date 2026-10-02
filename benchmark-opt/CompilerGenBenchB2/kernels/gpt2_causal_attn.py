"""Compiler-generated NKI kernel for `gpt2_causal_attn`.

Source:
  HLO from `pipeline/models.py::gpt2_causal_attn` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `gpt2_causal_attn-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  GPT-2 scaled-dot causal attention (body of FlaxGPT2Attention.__call__ inlining dot_product_attention_weights). Source: models/gpt2/modeling_flax_gpt2.py @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[1, 128, 4, 64], [1, 128, 4, 64], [1, 128, 4, 64], [128, 128]]
  kernel inputs  : [[128, 2, 128], [128, 2, 128], [128, 2, 128], [128, 128]]
  kernel output  : [[128, 2, 128]]
  logical output : [1, 128, 4, 64]
"""

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki.language import par_dim

import torch
import torch_xla.core.xla_model as xm

@nki.jit
def sg_kernel(v1, v2, v3, v4):
  v5 = nl.ndarray((128, 2, 128), dtype=np.float32, buffer=nl.shared_hbm)


  v6 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="add.1", buffer=nl.sbuf)
  v7 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="divide.1", buffer=nl.sbuf)
  v8 = nl.ndarray((2, nl.par_dim(128), 128), dtype=np.float32, name="input2_local_186", buffer=nl.sbuf)
  v9 = nl.zeros((2, nl.par_dim(128), 256), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v10 = nl.ndarray((2, nl.par_dim(128), 128), dtype=np.float32, name="input3_local_178", buffer=nl.sbuf)
  v11 = nl.ndarray((2, nl.par_dim(128), 2), dtype=np.float32, name="", buffer=nl.sbuf)
  v12 = nl.ndarray((2, nl.par_dim(64), 256), dtype=np.float32, name="", buffer=nl.sbuf)
  v13 = nl.ndarray((2, nl.par_dim(64), 256), dtype=np.float32, name="", buffer=nl.sbuf)
  v14 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="", buffer=nl.sbuf)
  v15 = nl.zeros((2, nl.par_dim(64), 256), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v16 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="identity_local_262", buffer=nl.sbuf)
  v17 = nl.zeros((2, nl.par_dim(64), 256), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v18 = nl.zeros((2, nl.par_dim(128), 256), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v19 = nl.ndarray((2, nl.par_dim(128), 128), dtype=np.float32, name="", buffer=nl.sbuf)
  v20 = nl.ndarray((2, nl.par_dim(128), 128), dtype=np.float32, name="", buffer=nl.sbuf)
  v21 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="divide.12.278", buffer=nl.sbuf)
  v22 = nl.ndarray((2, nl.par_dim(128), 2), dtype=np.float32, name="reduce.22.280", buffer=nl.sbuf)
  v23 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="subtract.28.282", buffer=nl.sbuf)
  v24 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="exponential.29.284", buffer=nl.sbuf)
  v25 = nl.ndarray((2, nl.par_dim(128), 2), dtype=np.float32, name="reduce.34.286", buffer=nl.sbuf)
  v26 = nl.ndarray((2, nl.par_dim(128), 128), dtype=np.float32, name="", buffer=nl.sbuf)
  v27 = nl.zeros((2, nl.par_dim(128), 128), dtype=np.float32, name="dot.40.290", buffer=nl.psum, lazy_initialization=True)
  v28 = nl.zeros((2, nl.par_dim(128), 2), dtype=np.float32, name="reduce.1", buffer=nl.psum, lazy_initialization=True)
  v29 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name="memset.294", buffer=nl.sbuf)
  v30 = nl.shared_constant(np.identity(128, dtype=np.float32))

  def BB_entry_1():
    v29[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v16[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v30[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)

    for i0 in nl.affine_range(2):
      v19[i0, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v1[nl.arange(128)[:, None], i0, nl.arange(128)[None, :]], dtype=np.float32, mask=None)

      for i1 in nl.affine_range(2):
        v15[i0, nl.arange(64)[:, None], 128*i1+nl.arange(128)[None, :]] = nisa.nc_matmul(v19[i0, nl.arange(128)[:, None], 64*i1+nl.arange(64)[None, :]], v16[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
        """ end loop i1 """
      v12[i0, nl.arange(64)[:, None], nl.arange(256)[None, :]] = nl.copy(v15[i0, nl.arange(64)[:, None], nl.arange(256)[None, :]], dtype=np.float32, mask=None)
      """ end loop i0 """

    for i2 in nl.affine_range(2):
      v20[i2, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v2[nl.arange(128)[:, None], i2, nl.arange(128)[None, :]], dtype=np.float32, mask=None)

      for i3 in nl.affine_range(2):
        v17[i2, nl.arange(64)[:, None], 128*i3+nl.arange(128)[None, :]] = nisa.nc_matmul(v20[i2, nl.arange(128)[:, None], 64*i3+nl.arange(64)[None, :]], v16[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
        """ end loop i3 """
      v13[i2, nl.arange(64)[:, None], nl.arange(256)[None, :]] = nl.copy(v17[i2, nl.arange(64)[:, None], nl.arange(256)[None, :]], dtype=np.float32, mask=None)
      """ end loop i2 """

    for i4 in nl.affine_range(2):
      v8[i4, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v3[nl.arange(128)[:, None], i4, nl.arange(128)[None, :]], dtype=np.float32, mask=None)
      """ end loop i4 """

    for i5 in nl.affine_range(2):
      v10[i5, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v4[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)

      for i6 in nl.affine_range(2):
        v9[i5, nl.arange(128)[:, None], 128*i6+nl.arange(128)[None, :]] = nisa.nc_matmul(v12[i5, nl.arange(64)[:, None], 128*i6+nl.arange(128)[None, :]], v13[i5, nl.arange(64)[:, None], 128*i6+nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
        """ end loop i6 """
      v21[i5, nl.arange(128)[:, None], nl.arange(256)[None, :]] = nisa.tensor_scalar(data=v9[i5, nl.arange(128)[:, None], nl.arange(256)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.125), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
      v6[i5, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]] = nl.add(v21[i5, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]], v10[i5, nl.arange(128)[:, None, None], nl.arange(128)[None, None, :]], mask=None, dtype=np.float32)
      v22[i5, nl.arange(128)[:, None], nl.arange(2)[None, :]] = nisa.tensor_reduce(nl.maximum, data=v6[i5, nl.arange(128)[:, None, None], nl.arange(128)[None, None, :]+128*nl.arange(2)[None, :, None]], mask=None, axis=[2], dtype=np.float32, negate=False)
      v28[i5, nl.arange(128)[:, None], nl.arange(2)[None, :]] = nisa.tensor_scalar(data=v22[i5, nl.arange(128)[:, None], nl.arange(2)[None, :]], op0=nl.maximum, operand0=-np.inf, reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
      v23[i5, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]] = nl.subtract(v6[i5, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]], v28[i5, nl.arange(128)[:, None, None], nl.arange(2)[None, :, None]], mask=None, dtype=np.float32)
      v24[i5, nl.arange(128)[:, None], nl.arange(256)[None, :]] = nisa.activation(op=nl.exp, data=v23[i5, nl.arange(128)[:, None], nl.arange(256)[None, :]], bias=v29[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
      v25[i5, nl.arange(128)[:, None], nl.arange(2)[None, :]] = nisa.tensor_reduce(nl.add, data=v24[i5, nl.arange(128)[:, None, None], nl.arange(128)[None, None, :]+128*nl.arange(2)[None, :, None]], mask=None, axis=[2], dtype=np.float32, negate=False)
      v11[i5, nl.arange(128)[:, None], nl.arange(2)[None, :]] = nisa.reciprocal(data=v25[i5, nl.arange(128)[:, None], nl.arange(2)[None, :]], mask=None, dtype=np.float32)
      v7[i5, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]] = nl.multiply(v24[i5, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]], v11[i5, nl.arange(128)[:, None, None], nl.arange(2)[None, :, None]], mask=None, dtype=np.float32)

      for i7 in nl.affine_range(2):
        v18[i5, nl.arange(128)[:, None], 128*i7+nl.arange(128)[None, :]] = nisa.nc_matmul(v7[i5, nl.arange(128)[:, None], 128*i7+nl.arange(128)[None, :]], v16[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
        """ end loop i7 """
      v14[i5, nl.arange(128)[:, None], nl.arange(256)[None, :]] = nl.copy(v18[i5, nl.arange(128)[:, None], nl.arange(256)[None, :]], dtype=np.float32, mask=None)

      for i8 in nl.affine_range(2):
        v27[i5, nl.arange(128)[:, None], 64*i8+nl.arange(64)[None, :]] = nisa.nc_matmul(v14[i5, nl.arange(128)[:, None], 128*i8+nl.arange(128)[None, :]], v8[i5, nl.arange(128)[:, None], 64*i8+nl.arange(64)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
        """ end loop i8 """
      v26[i5, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(v27[i5, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
      """ end loop i5 """

    for i9 in nl.affine_range(2):
      nl.store(v5[nl.arange(128)[:, None], i9, nl.arange(128)[None, :]], value=v26[i9, nl.arange(128)[:, None], nl.arange(128)[None, :]], mask=None)
      """ end loop i9 """

  BB_entry_1()
  return v5

def gpt2_causal_attn(x0, x1, x2, x3):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((128, 2, 128)).to(device), x1.reshape((128, 2, 128)).to(device), x2.reshape((128, 2, 128)).to(device), x3.to(device))
    return out.cpu().reshape((1, 128, 4, 64))
