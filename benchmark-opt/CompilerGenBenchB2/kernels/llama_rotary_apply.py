"""Compiler-generated NKI kernel for `llama_rotary_apply`.

Source:
  HLO from `pipeline/models.py::llama_rotary_apply` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `llama_rotary_apply-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  Llama rotary position embedding (split-half rotation). Source: models/llama/modeling_flax_llama.py:140-150 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[4, 1024, 16, 64], [1, 1024, 1, 64], [1, 1024, 1, 64]]
  kernel inputs  : [[2, 2, 128, 8, 1024], [128, 512], [128, 512]]
  kernel output  : [[2, 2, 128, 8, 1024]]
  logical output : [4, 1024, 16, 64]
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
  v4 = nl.ndarray((2, 2, 128, 8, 1024), dtype=np.float32, buffer=nl.shared_hbm)


  v5 = nl.ndarray((8, 2, 2, nl.par_dim(128), 1024), dtype=np.float32, name="concatenate.1", buffer=nl.sbuf)
  v6 = nl.ndarray((2, 2, 8, nl.par_dim(128), 1024), dtype=np.float32, name="input0_local_241", buffer=nl.sbuf)
  v7 = nl.ndarray((2, 2, 8, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v8 = nl.ndarray((2, nl.par_dim(128), 512), dtype=np.float32, name="input2_local_252", buffer=nl.sbuf)
  v9 = nl.ndarray((2, nl.par_dim(128), 512), dtype=np.float32, name="input1_local_260", buffer=nl.sbuf)
  v10 = nl.ndarray((2, 2, 8, nl.par_dim(128), 512), dtype=np.float32, name="negate.9.292", buffer=nl.sbuf)
  v11 = nl.ndarray((2, 2, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name="multiply.7.299", buffer=nl.sbuf)
  v12 = nl.zeros((2, 2, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name="multiply.15.301", buffer=nl.psum, lazy_initialization=True)
  v13 = nl.full((2, 2, 8, nl.par_dim(128), 960), fill_value=np.dtype(np.float32).type(0.0), dtype=np.float32, name="input0_nostride_304", buffer=nl.sbuf)
  v14 = nl.full((2, 2, 8, nl.par_dim(128), 960), fill_value=np.dtype(np.float32).type(0.0), dtype=np.float32, name="input0_nostride_307", buffer=nl.sbuf)

  def BB_entry_1():
    for i0 in nl.affine_range(2):
      for i1 in nl.affine_range(2):
        for i2 in nl.affine_range(8):
          for i3 in nl.affine_range(2):
            v13[i0, i1, i2, nl.arange(128)[:, None], 480*i3+nl.arange(480)[None, :]] = nl.load(v1[i0, i1, nl.arange(128)[:, None], i2, 512*i3+nl.arange(480)[None, :]+32], dtype=np.float32, mask=None)
            """ end loop i3 """
          v10[i0, i1, i2, nl.arange(128)[:, None, None, None], 256*nl.arange(2)[None, :, None, None]+32*nl.arange(8)[None, None, :, None]+nl.arange(32)[None, None, None, :]] = nl.copy(v13[i0, i1, i2, nl.arange(128)[:, None, None, None], 480*nl.arange(2)[None, :, None, None]+64*nl.arange(8)[None, None, :, None]+nl.arange(32)[None, None, None, :]], dtype=np.float32, mask=None)
          v5[i2, i0, i1, nl.arange(128)[:, None, None], 64*nl.arange(16)[None, :, None]+nl.arange(32)[None, None, :]] = nisa.tensor_scalar(data=v10[i0, i1, i2, nl.arange(128)[:, None, None], 32*nl.arange(16)[None, :, None]+nl.arange(32)[None, None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(-1), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
          """ end loop i2 """
        """ end loop i1 """
      """ end loop i0 """

    for i4 in nl.affine_range(2):
      for i5 in nl.affine_range(2):
        for i6 in nl.affine_range(8):
          for i7 in nl.affine_range(2):
            v14[i4, i5, i6, nl.arange(128)[:, None], 480*i7+nl.arange(480)[None, :]] = nl.load(v1[i4, i5, nl.arange(128)[:, None], i6, 512*i7+nl.arange(480)[None, :]], dtype=np.float32, mask=None)
            """ end loop i7 """
          v5[i6, i4, i5, nl.arange(128)[:, None, None, None], 512*nl.arange(2)[None, :, None, None]+64*nl.arange(8)[None, None, :, None]+nl.arange(32)[None, None, None, :]+32] = nl.copy(v14[i4, i5, i6, nl.arange(128)[:, None, None, None], 480*nl.arange(2)[None, :, None, None]+64*nl.arange(8)[None, None, :, None]+nl.arange(32)[None, None, None, :]], dtype=np.float32, mask=None)
          """ end loop i6 """
        """ end loop i5 """
      """ end loop i4 """

    for i8 in nl.affine_range(2):
      v9[i8, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.load(v2[nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
      v8[i8, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.load(v3[nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)

      for i9 in nl.affine_range(2):
        for i10 in nl.affine_range(2):
          for i11 in nl.affine_range(4):
            v6[i8, i9, i11+4*i10, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i8, i9, nl.arange(128)[:, None], i11+4*i10, nl.arange(1024)[None, :]], dtype=np.float32, mask=None)

            for i12 in nl.affine_range(2):
              v11[i8, i9, i11+4*i10, i12, nl.arange(128)[:, None, None, None], 64*nl.arange(8)[None, :, None, None]+32*nl.arange(2)[None, None, :, None]+nl.arange(32)[None, None, None, :]] = nl.multiply(v6[i8, i9, i11+4*i10, nl.arange(128)[:, None, None, None], 512*i12+64*nl.arange(8)[None, :, None, None]+32*nl.arange(2)[None, None, :, None]+nl.arange(32)[None, None, None, :]], v8[i8, nl.arange(128)[:, None, None, None], 64*i11+256*i10+32*nl.arange(2)[None, None, :, None]+nl.arange(32)[None, None, None, :]], mask=None, dtype=np.float32)
              v12[i8, i9, i11+4*i10, i12, nl.arange(128)[:, None, None, None], 64*nl.arange(8)[None, :, None, None]+32*nl.arange(2)[None, None, :, None]+nl.arange(32)[None, None, None, :]] = nl.multiply(v5[i11+4*i10, i8, i9, nl.arange(128)[:, None, None, None], 512*i12+64*nl.arange(8)[None, :, None, None]+32*nl.arange(2)[None, None, :, None]+nl.arange(32)[None, None, None, :]], v9[i8, nl.arange(128)[:, None, None, None], 64*i11+256*i10+32*nl.arange(2)[None, None, :, None]+nl.arange(32)[None, None, None, :]], mask=None, dtype=np.float32)
              v7[i8, i9, i11+4*i10, nl.arange(128)[:, None], 512*i12+nl.arange(512)[None, :]] = nl.add(v11[i8, i9, i11+4*i10, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], v12[i8, i9, i11+4*i10, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
              """ end loop i12 """
            nl.store(v4[i8, i9, nl.arange(128)[:, None], i11+4*i10, nl.arange(1024)[None, :]], value=v7[i8, i9, i11+4*i10, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
            """ end loop i11 """
          """ end loop i10 """
        """ end loop i9 """
      """ end loop i8 """

  BB_entry_1()
  return v4

def llama_rotary_apply(x0, x1, x2):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((2, 2, 128, 8, 1024)).to(device), x1.reshape((128, 512)).to(device), x2.reshape((128, 512)).to(device))
    return out.cpu().reshape((4, 1024, 16, 64))
