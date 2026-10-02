"""Compiler-generated NKI kernel for `clip_image_text_logits`.

Source:
  HLO from `pipeline/models.py::clip_image_text_logits` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `clip_image_text_logits-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  CLIP image↔text all-pairs scaled-dot logits matrix. Source: models/clip/modeling_flax_clip.py:1245-1247 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[256, 768], [256, 768], []]
  kernel inputs  : [[2, 128, 768], [256, 768], [1]]
  kernel output  : [[2, 128, 256]]
  logical output : [256, 256]
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
def sg_kernel(v1, v2, v3):
  v4 = nl.ndarray((2, 128, 256), dtype=np.float32, buffer=nl.shared_hbm)


  v5 = nl.ndarray((2, nl.par_dim(128), 768), dtype=np.float32, name="input0_pftranspose_130", buffer=nl.sbuf)
  v6 = nl.zeros((2, nl.par_dim(128), 256), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v7 = nl.ndarray((2, nl.par_dim(128), 1536), dtype=np.float32, name="input1_pftranspose_135", buffer=nl.sbuf)
  v8 = nl.ndarray((2, nl.par_dim(1), 1), dtype=np.float32, name="exponential.4.158", buffer=nl.sbuf)
  v9 = nl.ndarray((2, 2, nl.par_dim(128), 768), dtype=np.float32, name="", buffer=nl.sbuf)
  v10 = nl.zeros((2, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v11 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="identity_local_188", buffer=nl.sbuf)
  v12 = nl.ndarray((2, nl.par_dim(128), 768), dtype=np.float32, name="", buffer=nl.sbuf)
  v13 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v14 = nl.ndarray((2, nl.par_dim(128), 1), dtype=np.float32, name="exponential.4.199", buffer=nl.sbuf)
  v15 = nl.ndarray((2, nl.par_dim(128), 1), dtype=np.float32, name="exponential.4.201", buffer=nl.sbuf)
  v16 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="multiply.8.203", buffer=nl.sbuf)
  v17 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name="memset.206", buffer=nl.sbuf)
  v18 = nl.shared_constant(np.identity(128, dtype=np.float32))

  def BB_entry_1():
    v17[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v11[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v18[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)

    for i0 in nl.affine_range(2):
      for i1 in nl.affine_range(2):
        v9[i0, i1, nl.arange(128)[:, None], nl.arange(768)[None, :]] = nl.load(v2[128*i1+nl.arange(128)[:, None], nl.arange(768)[None, :]], dtype=np.float32, mask=None)

        for i2 in nl.affine_range(2):
          for i3 in nl.affine_range(4):
            v10[i0, i1, i2, nl.arange(128)[:, None], 128*i3+nl.arange(128)[None, :]] = nisa.nc_matmul(v9[i0, i1, nl.arange(128)[:, None], 128*i3+512*i2+nl.arange(128)[None, :]], v11[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=(-1*i3+-4*i2+5 >= 0), is_transpose=True)
            """ end loop i3 """
          v7[i0, nl.arange(128)[:, None, None], 128*i1+1024*i2+256*nl.arange(4)[None, :, None]+nl.arange(128)[None, None, :]] = nl.copy(v10[i0, i1, i2, nl.arange(128)[:, None, None], 128*nl.arange(4)[None, :, None]+nl.arange(128)[None, None, :]], dtype=np.float32, mask=(-1*i2 >= 0))
          v7[i0, nl.arange(128)[:, None, None], 128*i1+1024*i2+256*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]] = nl.copy(v10[i0, i1, i2, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]], dtype=np.float32, mask=(i2+-1 >= 0))
          """ end loop i2 """
        """ end loop i1 """
      v12[i0, nl.arange(128)[:, None], nl.arange(768)[None, :]] = nl.load(v1[i0, nl.arange(128)[:, None], nl.arange(768)[None, :]], dtype=np.float32, mask=None)

      for i4 in nl.affine_range(2):
        for i5 in nl.affine_range(4):
          v13[i0, i4, nl.arange(128)[:, None], 128*i5+nl.arange(128)[None, :]] = nisa.nc_matmul(v12[i0, nl.arange(128)[:, None], 128*i5+512*i4+nl.arange(128)[None, :]], v11[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=(-1*i5+-4*i4+5 >= 0), is_transpose=True)
          """ end loop i5 """
        v5[i0, nl.arange(128)[:, None, None], 512*i4+128*nl.arange(4)[None, :, None]+nl.arange(128)[None, None, :]] = nl.copy(v13[i0, i4, nl.arange(128)[:, None, None], 128*nl.arange(4)[None, :, None]+nl.arange(128)[None, None, :]], dtype=np.float32, mask=(-1*i4 >= 0))
        v5[i0, nl.arange(128)[:, None, None], 512*i4+128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]] = nl.copy(v13[i0, i4, nl.arange(128)[:, None, None], 128*nl.arange(2)[None, :, None]+nl.arange(128)[None, None, :]], dtype=np.float32, mask=(i4+-1 >= 0))
        """ end loop i4 """

      for i6 in nl.affine_range(6):
        v6[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]] += nisa.nc_matmul(v5[i0, nl.arange(128)[:, None], 128*i6+nl.arange(128)[None, :]], v7[i0, nl.arange(128)[:, None], 256*i6+nl.arange(256)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
        """ end loop i6 """
      v8[i0, 0, 0] = nl.load(v3[0], dtype=np.float32, mask=None)
      v14[i0, nl.arange(32)[:, None], 0] = v8[i0, 0, 0].broadcast_to([32, 1])
      v14[i0, nl.arange(32)[:, None]+32, 0] = v8[i0, 0, 0].broadcast_to([32, 1])
      v14[i0, nl.arange(32)[:, None]+64, 0] = v8[i0, 0, 0].broadcast_to([32, 1])
      v14[i0, nl.arange(32)[:, None]+96, 0] = v8[i0, 0, 0].broadcast_to([32, 1])
      v15[i0, nl.arange(128)[:, None], 0] = nisa.activation(op=nl.exp, data=v14[i0, nl.arange(128)[:, None], 0], bias=v17[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
      v16[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]] = nisa.tensor_scalar(data=v6[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]], op0=nl.multiply, operand0=v15[i0, nl.arange(128)[:, None], 0], reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
      nl.store(v4[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]], value=v16[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]], mask=None)
      """ end loop i0 """

  BB_entry_1()
  return v4

def clip_image_text_logits(x0, x1, x2):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((2, 128, 768)).to(device), x1.to(device), x2.reshape((1)).to(device))
    return out.cpu().reshape((256, 256))
