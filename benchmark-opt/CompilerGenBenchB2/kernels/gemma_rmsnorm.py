"""Compiler-generated NKI kernel for `gemma_rmsnorm`.

Source:
  HLO from `pipeline/models.py::gemma_rmsnorm` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `gemma_rmsnorm-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  Gemma RMSNorm with (1 + weight) trick. Source: models/gemma/modeling_flax_gemma.py:148-164 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[4, 1024, 4096], [4096]]
  kernel inputs  : [[2, 2, 2, 512, 4096], [32, 128]]
  kernel output  : [[2, 2, 2, 512, 4096]]
  logical output : [4, 1024, 4096]
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
  v3 = nl.ndarray((2, 2, 2, 512, 4096), dtype=np.float32, buffer=nl.shared_hbm)


  v4 = nl.ndarray((2, 2, 2, 8, 4, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.sbuf)
  v5 = nl.ndarray((2, 2, 2, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name="output0_pftranspose_193", buffer=nl.sbuf)
  v6 = nl.ndarray((2, nl.par_dim(128), 32), dtype=np.float32, name="add.1", buffer=nl.sbuf)
  v7 = nl.ndarray((nl.par_dim(1), 1), dtype=np.float32, name="add.18.253", buffer=nl.sbuf)
  v8 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="memset.269", buffer=nl.sbuf)
  v9 = nl.zeros((2, 2, 2, nl.par_dim(1), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v10 = nl.zeros((2, nl.par_dim(128), 32), dtype=np.float32, name="input1_pftranspose_188", buffer=nl.psum, lazy_initialization=True)
  v11 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="identity_local_279", buffer=nl.sbuf)
  v12 = nl.ndarray((2, 2, 2, 4, 4, nl.par_dim(128), 1024), dtype=np.float32, name="193.289", buffer=nl.sbuf)
  v13 = nl.zeros((2, 2, 2, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name="193.292", buffer=nl.psum, lazy_initialization=True)
  v14 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="memset.298", buffer=nl.sbuf)
  v15 = nl.ndarray((nl.par_dim(32), 128), dtype=np.float32, name="", buffer=nl.sbuf)
  v16 = nl.ndarray((2, 2, 2, 4, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v17 = nl.ndarray((2, 2, 2, nl.par_dim(1), 512), dtype=np.float32, name="sqrt.19.303", buffer=nl.sbuf)
  v18 = nl.ndarray((2, 2, 2, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name="divide.23.305", buffer=nl.sbuf)
  v19 = nl.ndarray((2, 2, 2, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name="divide.23.307", buffer=nl.sbuf)
  v20 = nl.ndarray((2, 2, 2, 4, 2, nl.par_dim(128), 2048), dtype=np.float32, name="", buffer=nl.sbuf)
  v21 = nl.zeros((2, 2, 2, 4, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v22 = nl.ndarray((2, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.sbuf)
  v23 = nl.zeros((2, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name="divide.23.314", buffer=nl.psum, lazy_initialization=True)
  v24 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name="memset.317", buffer=nl.sbuf)
  v25 = nl.shared_constant(np.identity(128, dtype=np.float32))

  def BB_entry_1():
    v24[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v14[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.memset(shape=(128, 128), value=np.dtype(np.float32).type(1), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v11[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v25[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    v8[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.memset(shape=(128, 128), value=np.dtype(np.float32).type(1), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v7[0, 0] = nisa.memset(shape=(1, 1), value=np.dtype(np.float32).type(9.999999974752427e-07), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v15[nl.arange(32)[:, None], nl.arange(128)[None, :]] = nl.load(v2[nl.arange(32)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)

    for i0 in nl.affine_range(2):
      v10[i0, nl.arange(128)[:, None], nl.arange(32)[None, :]] = nisa.nc_matmul(v15[nl.arange(32)[:, None], nl.arange(128)[None, :]], v11[nl.arange(32)[:, None], nl.arange(32)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
      v6[i0, nl.arange(128)[:, None], nl.arange(32)[None, :]] = nisa.tensor_scalar(data=v10[i0, nl.arange(128)[:, None], nl.arange(32)[None, :]], op0=nl.add, operand0=np.dtype(np.float32).type(1.0), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)

      for i1 in nl.affine_range(2):
        for i2 in nl.affine_range(2):
          for i3 in nl.affine_range(4):
            for i4 in nl.affine_range(4):
              v16[i0, i1, i2, i3, i4, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i0, i1, i2, 128*i4+nl.arange(128)[:, None], 1024*i3+nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
              """ end loop i4 """

            for i5 in nl.affine_range(4):
              for i6 in nl.affine_range(2):
                for i7 in nl.affine_range(4):
                  v21[i0, i1, i2, i3, i5, i6, nl.arange(128)[:, None], 128*i7+nl.arange(128)[None, :]] = nisa.nc_matmul(v16[i0, i1, i2, i3, i5, nl.arange(128)[:, None], 512*i6+128*i7+nl.arange(128)[None, :]], v11[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                  """ end loop i7 """
                v20[i0, i1, i2, i3, i6, nl.arange(128)[:, None, None], 128*i5+512*nl.arange(4)[None, :, None]+nl.arange(128)[None, None, :]] = nl.copy(v21[i0, i1, i2, i3, i5, i6, nl.arange(128)[:, None, None], 128*nl.arange(4)[None, :, None]+nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                """ end loop i6 """
              """ end loop i5 """

            for i8 in nl.affine_range(2):
              for i9 in nl.affine_range(4):
                v4[i0, i1, i2, 2*i3+i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.square, data=v20[i0, i1, i2, i3, i8, nl.arange(128)[:, None], 512*i9+nl.arange(512)[None, :]], bias=v24[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
                """ end loop i9 """
              """ end loop i8 """
            """ end loop i3 """

          for i10 in nl.affine_range(8):
            for i11 in nl.affine_range(4):
              v9[i0, i1, i2, 0, nl.arange(512)[None, :]] += nisa.nc_matmul(v8[nl.arange(128)[:, None], 0], v4[i0, i1, i2, i10, i11, nl.arange(128)[:, None], nl.arange(512)[None, :]], is_stationary_onezero=True, is_moving_onezero=False, mask=None)
              """ end loop i11 """
            """ end loop i10 """
          v17[i0, i1, i2, 0, nl.arange(512)[None, :]] = nisa.activation(op=nl.sqrt, data=v9[i0, i1, i2, 0, nl.arange(512)[None, :]], bias=v7[0, 0], scale=0.000244140625, mask=None, dtype=np.float32)
          v23[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.nc_matmul(v14[0, nl.arange(128)[None, :]], v17[i0, i1, i2, 0, nl.arange(512)[None, :]], is_stationary_onezero=True, is_moving_onezero=False, mask=None)
          v22[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v23[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)

          for i12 in nl.affine_range(4):
            for i13 in nl.affine_range(2):
              for i14 in nl.affine_range(4):
                v18[i0, i1, i2, i12, i13, i14, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.reciprocal(data=v22[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                v19[i0, i1, i2, i12, i13, i14, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v20[i0, i1, i2, i12, i13, nl.arange(128)[:, None], 512*i14+nl.arange(512)[None, :]], v18[i0, i1, i2, i12, i13, i14, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                v5[i0, i1, i2, i12, i13, i14, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v19[i0, i1, i2, i12, i13, i14, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=v6[i0, nl.arange(128)[:, None], 8*i12+4*i13+i14], reverse0=True, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
                """ end loop i14 """

              for i15 in nl.affine_range(4):
                for i16 in nl.affine_range(4):
                  v13[i0, i1, i2, i12, i13, i15, nl.arange(128)[:, None], 128*i16+nl.arange(128)[None, :]] = nisa.nc_matmul(v5[i0, i1, i2, i12, i13, i16, nl.arange(128)[:, None], 128*i15+nl.arange(128)[None, :]], v11[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                  """ end loop i16 """
                v12[i0, i1, i2, i12, i15, nl.arange(128)[:, None], 512*i13+nl.arange(512)[None, :]] = nl.copy(v13[i0, i1, i2, i12, i13, i15, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                """ end loop i15 """
              """ end loop i13 """

            for i17 in nl.affine_range(4):
              nl.store(v3[i0, i1, i2, 128*i17+nl.arange(128)[:, None], 1024*i12+nl.arange(1024)[None, :]], value=v12[i0, i1, i2, i12, i17, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
              """ end loop i17 """
            """ end loop i12 """
          """ end loop i2 """
        """ end loop i1 """
      """ end loop i0 """

  BB_entry_1()
  return v3

def gemma_rmsnorm(x0, x1):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((2, 2, 2, 512, 4096)).to(device), x1.reshape((32, 128)).to(device))
    return out.cpu().reshape((4, 1024, 4096))
