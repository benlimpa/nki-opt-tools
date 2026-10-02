"""Compiler-generated NKI kernel for `roformer_rotary_apply`.

Source:
  HLO from `pipeline/models.py::roformer_rotary_apply` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `roformer_rotary_apply-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  RoFormer interleaved-pair rotary embedding (distinct from Llama's halved variant). Source: models/roformer/modeling_flax_roformer.py:264-280 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[4, 512, 8, 64], [1, 1, 512, 64]]
  kernel inputs  : [[4, 2, 128, 1024], [1, 1, 2, 128, 2, 2, 32]]
  kernel output  : [[4, 2, 128, 1024]]
  logical output : [4, 512, 8, 64]
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
def sg_kernel(v1, v2):
  v3 = nl.ndarray((4, 2, 128, 1024), dtype=np.float32, buffer=nl.shared_hbm)


  v4 = nl.ndarray(shape=(4, 2, 128, 2, 512), dtype=np.float32, name="input0_d2dtranspose_307", buffer=nl.hbm)
  v5 = nl.ndarray((2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="concatenate.3", buffer=nl.sbuf)
  v6 = nl.ndarray((2, 4, nl.par_dim(128), 512), dtype=np.float32, name="input0_d2dtranspose_307_local_313", buffer=nl.sbuf)
  v7 = nl.ndarray((2, 2, nl.par_dim(128), 32), dtype=np.float32, name="input1_local_324", buffer=nl.sbuf)
  v8 = nl.ndarray((2, 2, nl.par_dim(128), 32), dtype=np.float32, name="input1_local_332", buffer=nl.sbuf)
  v9 = nl.ndarray((2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="input0_local_335", buffer=nl.sbuf)
  v10 = nl.ndarray((2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v11 = nl.ndarray((2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="dot.37.368", buffer=nl.sbuf)
  v12 = nl.ndarray((2, nl.par_dim(128), 4096), dtype=np.float32, name="", buffer=nl.sbuf)
  v13 = nl.ndarray((2, nl.par_dim(128), 4096), dtype=np.float32, name="tiled_dve_transpose_10tiled_dve_transpose_10_sg0000_372_transpose_temp", buffer=nl.sbuf)
  v14 = nl.zeros((2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="dot.44.370", buffer=nl.psum, lazy_initialization=True)
  v15 = nl.ndarray((2, 4, nl.par_dim(128), 512), dtype=np.float32, name="gather.27.387", buffer=nl.sbuf)

  def BB_entry_1():
    for i0 in nl.affine_range(2):
      for i1 in nl.affine_range(4):
        v12[i0, nl.arange(128)[:, None, None], 1024*i1+2*nl.arange(512)[None, :, None]+nl.arange(2)[None, None, :]] = nl.load(v1.reshape((1024, 512, 2))[128*i1+512*i0+nl.arange(128)[:, None, None], nl.arange(512)[None, :, None], nl.arange(2)[None, None, :]], dtype=np.float32, mask=None)
        """ end loop i1 """
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]], dtype=np.float32, mask=None, engine=nki.isa.engine.scalar)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+64] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+128], dtype=np.float32, mask=None, engine=nki.isa.engine.vector)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+128] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+256], dtype=np.float32, mask=None, engine=nki.isa.engine.scalar)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+192] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+384], dtype=np.float32, mask=None, engine=nki.isa.engine.vector)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+256] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+512], dtype=np.float32, mask=None, engine=nki.isa.engine.scalar)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+320] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+640], dtype=np.float32, mask=None, engine=nki.isa.engine.vector)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+384] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+768], dtype=np.float32, mask=None, engine=nki.isa.engine.scalar)
      v13[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+512*nl.arange(2)[None, None, :, None]+nl.arange(64)[None, None, None, :]+448] = nisa.tensor_copy(v12[i0, nl.arange(128)[:, None, None, None], 1024*nl.arange(4)[None, :, None, None]+nl.arange(2)[None, None, :, None]+2*nl.arange(64)[None, None, None, :]+896], dtype=np.float32, mask=None, engine=nki.isa.engine.vector)

      for i2 in nl.affine_range(4):
        nl.store(v4.reshape((1024, 2, 512))[128*i2+512*i0+nl.arange(128)[:, None, None], nl.arange(2)[None, :, None], nl.arange(512)[None, None, :]], value=v13[i0, nl.arange(128)[:, None, None], 1024*i2+512*nl.arange(2)[None, :, None]+nl.arange(512)[None, None, :]], mask=None)
        """ end loop i2 """
      """ end loop i0 """
    # NkiCodegen.codegenCoreBarrierIntrinsic is not implemented

    for i3 in nl.affine_range(2):
      for i4 in nl.affine_range(4):
        v6[i3, i4, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.load(v4[i4, i3, nl.arange(128)[:, None], 1, nl.arange(512)[None, :]], dtype=np.float32, mask=None)
        v5[i3, i4, nl.arange(128)[:, None], 2*nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v6[i3, i4, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(-1), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
        """ end loop i4 """
      """ end loop i3 """

    for i5 in nl.affine_range(2):
      for i6 in nl.affine_range(4):
        v15[i5, i6, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.load(v4[i6, i5, nl.arange(128)[:, None], 0, nl.arange(512)[None, :]], dtype=np.float32, mask=None)
        v5[i5, i6, nl.arange(128)[:, None], 2*nl.arange(512)[None, :]+1] = nl.copy(v15[i5, i6, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
        """ end loop i6 """
      """ end loop i5 """

    for i7 in nl.affine_range(2):
      for i8 in nl.affine_range(2):
        v8[i7, i8, nl.arange(128)[:, None], nl.arange(32)[None, :]] = nl.load(v2[0, 0, i7, nl.arange(128)[:, None], i8, 1, nl.arange(32)[None, :]], dtype=np.float32, mask=None)
        """ end loop i8 """

      for i9 in nl.affine_range(2):
        v7[i7, i9, nl.arange(128)[:, None], nl.arange(32)[None, :]] = nl.load(v2[0, 0, i7, nl.arange(128)[:, None], i9, 0, nl.arange(32)[None, :]], dtype=np.float32, mask=None)
        """ end loop i9 """

      for i10 in nl.affine_range(4):
        v9[i7, i10, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i10, i7, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)

        for i11 in nl.affine_range(2):
          v11[i7, i10, i11, nl.arange(128)[:, None, None, None], 64*nl.arange(8)[None, :, None, None]+2*nl.arange(32)[None, None, :, None]+nl.arange(2)[None, None, None, :]] = nl.multiply(v8[i7, i11, nl.arange(128)[:, None, None, None], nl.arange(32)[None, None, :, None]], v9[i7, i10, nl.arange(128)[:, None, None, None], 512*i11+64*nl.arange(8)[None, :, None, None]+2*nl.arange(32)[None, None, :, None]+nl.arange(2)[None, None, None, :]], mask=None, dtype=np.float32)
          v14[i7, i10, i11, nl.arange(128)[:, None, None, None], 64*nl.arange(8)[None, :, None, None]+2*nl.arange(32)[None, None, :, None]+nl.arange(2)[None, None, None, :]] = nl.multiply(v7[i7, i11, nl.arange(128)[:, None, None, None], nl.arange(32)[None, None, :, None]], v5[i7, i10, nl.arange(128)[:, None, None, None], 512*i11+64*nl.arange(8)[None, :, None, None]+2*nl.arange(32)[None, None, :, None]+nl.arange(2)[None, None, None, :]], mask=None, dtype=np.float32)
          v10[i7, i10, nl.arange(128)[:, None], 512*i11+nl.arange(512)[None, :]] = nl.add(v11[i7, i10, i11, nl.arange(128)[:, None], nl.arange(512)[None, :]], v14[i7, i10, i11, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
          """ end loop i11 """
        nl.store(v3[i10, i7, nl.arange(128)[:, None], nl.arange(1024)[None, :]], value=v10[i7, i10, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
        """ end loop i10 """
      """ end loop i7 """

  BB_entry_1()
  return v3

def roformer_rotary_apply(x0, x1):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((4, 2, 128, 1024)).to(device), x1.reshape((1, 1, 2, 128, 2, 2, 32)).to(device))
    return out.cpu().reshape((4, 512, 8, 64))
