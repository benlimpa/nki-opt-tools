"""Compiler-generated NKI kernel for `bert_pooler`.

Source:
  HLO from `pipeline/models.py::bert_pooler` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `bert_pooler-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  BERT pooler — tanh(dense(hidden_states[:, 0])). Source: models/bert/modeling_flax_bert.py:669-684 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[4, 512, 768], [768, 768], [768]]
  kernel inputs  : [[4, 512, 768], [6, 128, 2, 384], [2, 3, 128]]
  kernel output  : [[4, 2, 384]]
  logical output : [4, 768]
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
  v4 = nl.ndarray((4, 2, 384), dtype=np.float32, buffer=nl.shared_hbm)


  v5 = nl.ndarray((2, nl.par_dim(128), 3), dtype=np.float32, name="input2_pftranspose_117", buffer=nl.sbuf)
  v6 = nl.ndarray((2, 6, nl.par_dim(128), 384), dtype=np.float32, name="input1_local_137", buffer=nl.sbuf)
  v7 = nl.zeros((2, 3, nl.par_dim(128), 4), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v8 = nl.ndarray((2, nl.par_dim(128), 24), dtype=np.float32, name="input0_pftranspose_113", buffer=nl.sbuf)
  v9 = nl.ndarray((2, 3, nl.par_dim(128), 4), dtype=np.float32, name="tanh.12.162", buffer=nl.sbuf)
  v10 = nl.zeros((2, nl.par_dim(128), 24), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v11 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="identity_local_172", buffer=nl.sbuf)
  v12 = nl.zeros((2, nl.par_dim(128), 3), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v13 = nl.ndarray((2, nl.par_dim(4), 384), dtype=np.float32, name="122.183", buffer=nl.sbuf)
  v14 = nl.zeros((2, nl.par_dim(4), 384), dtype=np.float32, name="122.186", buffer=nl.psum, lazy_initialization=True)
  v15 = nl.ndarray((2, nl.par_dim(4), 768), dtype=np.float32, name="", buffer=nl.sbuf)
  v16 = nl.ndarray((2, nl.par_dim(3), 128), dtype=np.float32, name="", buffer=nl.sbuf)
  v17 = nl.shared_constant(np.identity(128, dtype=np.float32))

  def BB_entry_1():
    v11[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v17[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)

    for i0 in nl.affine_range(2):
      for i1 in nl.affine_range(6):
        v6[i0, i1, nl.arange(128)[:, None], nl.arange(384)[None, :]] = nl.load(v2[i1, nl.arange(128)[:, None], i0, nl.arange(384)[None, :]], dtype=np.float32, mask=None)
        """ end loop i1 """
      """ end loop i0 """

    for i2 in nl.affine_range(2):
      v15[i2, nl.arange(4)[:, None], nl.arange(768)[None, :]] = nl.load(v1[nl.arange(4)[:, None], 0, nl.arange(768)[None, :]], dtype=np.float32, mask=None)

      for i3 in nl.affine_range(6):
        v10[i2, nl.arange(128)[:, None], 4*i3+nl.arange(4)[None, :]] = nisa.nc_matmul(v15[i2, nl.arange(4)[:, None], 128*i3+nl.arange(128)[None, :]], v11[nl.arange(4)[:, None], nl.arange(4)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
        """ end loop i3 """
      v8[i2, nl.arange(128)[:, None], nl.arange(24)[None, :]] = nl.copy(v10[i2, nl.arange(128)[:, None], nl.arange(24)[None, :]], dtype=np.float32, mask=None)
      v16[i2, nl.arange(3)[:, None], nl.arange(128)[None, :]] = nl.load(v3[i2, nl.arange(3)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
      v12[i2, nl.arange(128)[:, None], nl.arange(3)[None, :]] = nisa.nc_matmul(v16[i2, nl.arange(3)[:, None], nl.arange(128)[None, :]], v11[nl.arange(3)[:, None], nl.arange(3)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
      v5[i2, nl.arange(128)[:, None], nl.arange(3)[None, :]] = nl.copy(v12[i2, nl.arange(128)[:, None], nl.arange(3)[None, :]], dtype=np.float32, mask=None)

      for i4 in nl.affine_range(6):
        for i5 in nl.affine_range(3):
          v7[i2, i5, nl.arange(128)[:, None], nl.arange(4)[None, :]] += nisa.nc_matmul(v6[i2, i4, nl.arange(128)[:, None], 128*i5+nl.arange(128)[None, :]], v8[i2, nl.arange(128)[:, None], 4*i4+nl.arange(4)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
          """ end loop i5 """
        """ end loop i4 """

      for i6 in nl.affine_range(3):
        v9[i2, i6, nl.arange(128)[:, None], nl.arange(4)[None, :]] = nisa.activation(op=nl.tanh, data=v7[i2, i6, nl.arange(128)[:, None], nl.arange(4)[None, :]], bias=v5[i2, nl.arange(128)[:, None], i6], scale=1.0, mask=None, dtype=np.float32)
        """ end loop i6 """
      """ end loop i2 """

    for i7 in nl.affine_range(2):
      for i8 in nl.affine_range(3):
        v14[i7, nl.arange(4)[:, None], 128*i8+nl.arange(128)[None, :]] = nisa.nc_matmul(v9[i7, i8, nl.arange(128)[:, None], nl.arange(4)[None, :]], v11[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
        """ end loop i8 """
      v13[i7, nl.arange(4)[:, None], nl.arange(384)[None, :]] = nl.copy(v14[i7, nl.arange(4)[:, None], nl.arange(384)[None, :]], dtype=np.float32, mask=None)
      nl.store(v4[nl.arange(4)[:, None], i7, nl.arange(384)[None, :]], value=v13[i7, nl.arange(4)[:, None], nl.arange(384)[None, :]], mask=None)
      """ end loop i7 """

  BB_entry_1()
  return v4

def bert_pooler(x0, x1, x2):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.to(device), x1.reshape((6, 128, 2, 384)).to(device), x2.reshape((2, 3, 128)).to(device))
    return out.cpu().reshape((4, 768))
