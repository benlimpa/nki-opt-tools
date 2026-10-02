"""Compiler-generated NKI kernel for `bart_qkv_projection`.

Source:
  HLO from `pipeline/models.py::bart_qkv_projection` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `bart_qkv_projection-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  BART fused Q/K/V projection (3 parallel dense, concat → (B,S,3*embed)). Source: models/bart/modeling_flax_bart.py FlaxBartAttention.__call__ lines 321-330 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[4, 512, 1024], [1024, 1024], [1024, 1024], [1024, 1024]]
  kernel inputs  : [[2, 2, 4, 128, 1024], [8, 128, 1024], [8, 128, 1024], [8, 128, 1024]]
  kernel output  : [[2, 2, 4, 128, 3, 1024]]
  logical output : [4, 512, 3072]
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
  v5 = nl.ndarray((2, 2, 4, 128, 3, 1024), dtype=np.float32, buffer=nl.shared_hbm)


  v6 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="input0_pftranspose_305", buffer=nl.sbuf)
  v7 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v8 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v9 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v10 = nl.zeros((2, 2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v11 = nl.zeros((2, 2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v12 = nl.zeros((2, 2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v13 = nl.ndarray((2, 8, nl.par_dim(128), 1024), dtype=np.float32, name="input1_local_316", buffer=nl.sbuf)
  v14 = nl.ndarray((2, 8, nl.par_dim(128), 1024), dtype=np.float32, name="input2_local_329", buffer=nl.sbuf)
  v15 = nl.ndarray((2, 8, nl.par_dim(128), 1024), dtype=np.float32, name="input3_local_342", buffer=nl.sbuf)
  v16 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="input0_pftranspose_305_reload_386", buffer=nl.sbuf)
  v17 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="input0_pftranspose_305_reload_390", buffer=nl.sbuf)
  v18 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v19 = nl.zeros((2, 2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v20 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="identity_local_400", buffer=nl.sbuf)
  v21 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v22 = nl.zeros((2, 2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v23 = nl.ndarray((2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v24 = nl.zeros((2, 2, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v25 = nl.shared_constant(np.identity(128, dtype=np.float32))

  def BB_entry_1():
    v20[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v25[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)

    for i0 in nl.affine_range(2):
      for i1 in nl.affine_range(8):
        v13[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
        """ end loop i1 """

      for i2 in nl.affine_range(2):
        for i3 in nl.affine_range(4):
          v18[i0, i2, i3, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i0, i2, i3, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
          """ end loop i3 """

        for i4 in nl.affine_range(4):
          for i5 in nl.affine_range(2):
            for i6 in nl.affine_range(4):
              v19[i0, i2, i4, i5, nl.arange(128)[:, None], 128*i6+nl.arange(128)[None, :]] = nisa.nc_matmul(v18[i0, i2, i4, nl.arange(128)[:, None], 128*i6+512*i5+nl.arange(128)[None, :]], v20[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
              """ end loop i6 """
            v6[i0, i2, i4, nl.arange(128)[:, None], 512*i5+nl.arange(512)[None, :]] = nl.copy(v19[i0, i2, i4, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            """ end loop i5 """
          """ end loop i4 """

        for i7 in nl.affine_range(8):
          for i8 in nl.affine_range(4):
            for i9 in nl.affine_range(2):
              v10[i0, i2, i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v6[i0, i2, i8, nl.arange(128)[:, None], 128*i7+nl.arange(128)[None, :]], v13[i0, i7, nl.arange(128)[:, None], 512*i9+nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
              """ end loop i9 """
            """ end loop i8 """
          """ end loop i7 """

        for i10 in nl.affine_range(4):
          for i11 in nl.affine_range(2):
            v7[i0, i2, i10, nl.arange(128)[:, None], 512*i11+nl.arange(512)[None, :]] = nl.copy(v10[i0, i2, i10, i11, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            """ end loop i11 """
          """ end loop i10 """

        for i12 in nl.affine_range(4):
          nl.store(v5[i0, i2, i12, nl.arange(128)[:, None], 0, nl.arange(1024)[None, :]], value=v7[i0, i2, i12, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
          """ end loop i12 """
        """ end loop i2 """
      """ end loop i0 """

    for i13 in nl.affine_range(2):
      for i14 in nl.affine_range(8):
        v14[i13, i14, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v3[i14, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
        """ end loop i14 """

      for i15 in nl.affine_range(2):
        for i16 in nl.affine_range(4):
          v21[i13, i15, i16, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i13, i15, i16, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
          """ end loop i16 """

        for i17 in nl.affine_range(4):
          for i18 in nl.affine_range(2):
            for i19 in nl.affine_range(4):
              v22[i13, i15, i17, i18, nl.arange(128)[:, None], 128*i19+nl.arange(128)[None, :]] = nisa.nc_matmul(v21[i13, i15, i17, nl.arange(128)[:, None], 128*i19+512*i18+nl.arange(128)[None, :]], v20[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
              """ end loop i19 """
            v16[i13, i15, i17, nl.arange(128)[:, None], 512*i18+nl.arange(512)[None, :]] = nl.copy(v22[i13, i15, i17, i18, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            """ end loop i18 """
          """ end loop i17 """

        for i20 in nl.affine_range(8):
          for i21 in nl.affine_range(4):
            for i22 in nl.affine_range(2):
              v11[i13, i15, i21, i22, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v16[i13, i15, i21, nl.arange(128)[:, None], 128*i20+nl.arange(128)[None, :]], v14[i13, i20, nl.arange(128)[:, None], 512*i22+nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
              """ end loop i22 """
            """ end loop i21 """
          """ end loop i20 """

        for i23 in nl.affine_range(4):
          for i24 in nl.affine_range(2):
            v8[i13, i15, i23, nl.arange(128)[:, None], 512*i24+nl.arange(512)[None, :]] = nl.copy(v11[i13, i15, i23, i24, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            """ end loop i24 """
          """ end loop i23 """

        for i25 in nl.affine_range(4):
          nl.store(v5[i13, i15, i25, nl.arange(128)[:, None], 1, nl.arange(1024)[None, :]], value=v8[i13, i15, i25, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
          """ end loop i25 """
        """ end loop i15 """
      """ end loop i13 """

    for i26 in nl.affine_range(2):
      for i27 in nl.affine_range(8):
        v15[i26, i27, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v4[i27, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
        """ end loop i27 """

      for i28 in nl.affine_range(2):
        for i29 in nl.affine_range(4):
          v23[i26, i28, i29, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i26, i28, i29, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
          """ end loop i29 """

        for i30 in nl.affine_range(4):
          for i31 in nl.affine_range(2):
            for i32 in nl.affine_range(4):
              v24[i26, i28, i30, i31, nl.arange(128)[:, None], 128*i32+nl.arange(128)[None, :]] = nisa.nc_matmul(v23[i26, i28, i30, nl.arange(128)[:, None], 128*i32+512*i31+nl.arange(128)[None, :]], v20[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
              """ end loop i32 """
            v17[i26, i28, i30, nl.arange(128)[:, None], 512*i31+nl.arange(512)[None, :]] = nl.copy(v24[i26, i28, i30, i31, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            """ end loop i31 """
          """ end loop i30 """

        for i33 in nl.affine_range(8):
          for i34 in nl.affine_range(4):
            for i35 in nl.affine_range(2):
              v12[i26, i28, i34, i35, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v17[i26, i28, i34, nl.arange(128)[:, None], 128*i33+nl.arange(128)[None, :]], v15[i26, i33, nl.arange(128)[:, None], 512*i35+nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
              """ end loop i35 """
            """ end loop i34 """
          """ end loop i33 """

        for i36 in nl.affine_range(4):
          for i37 in nl.affine_range(2):
            v9[i26, i28, i36, nl.arange(128)[:, None], 512*i37+nl.arange(512)[None, :]] = nl.copy(v12[i26, i28, i36, i37, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            """ end loop i37 """
          """ end loop i36 """

        for i38 in nl.affine_range(4):
          nl.store(v5[i26, i28, i38, nl.arange(128)[:, None], 2, nl.arange(1024)[None, :]], value=v9[i26, i28, i38, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
          """ end loop i38 """
        """ end loop i28 """
      """ end loop i26 """

  BB_entry_1()
  return v5

def bart_qkv_projection(x0, x1, x2, x3):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((2, 2, 4, 128, 1024)).to(device), x1.reshape((8, 128, 1024)).to(device), x2.reshape((8, 128, 1024)).to(device), x3.reshape((8, 128, 1024)).to(device))
    return out.cpu().reshape((4, 512, 3072))
