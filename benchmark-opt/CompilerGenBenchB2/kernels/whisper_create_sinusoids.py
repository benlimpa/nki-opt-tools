"""Compiler-generated NKI kernel for `whisper_create_sinusoids`.

Source:
  HLO from `pipeline/models.py::whisper_create_sinusoids` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `whisper_create_sinusoids-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  Whisper sinusoidal positional embedding table generator. Source: models/whisper/modeling_flax_whisper.py:62-72 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[1024], [256]]
  kernel inputs  : [[2, 4, 128], [256]]
  kernel output  : [[2, 4, 128, 512]]
  logical output : [1024, 512]
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
  v3 = nl.ndarray((2, 4, 128, 512), dtype=np.float32, buffer=nl.shared_hbm)


  v4 = nl.ndarray((2, 2, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v5 = nl.ndarray((2, nl.par_dim(128), 4), dtype=np.float32, name="input0_pftranspose_89", buffer=nl.sbuf)
  v6 = nl.zeros((2, nl.par_dim(128), 4), dtype=np.float32, name="", buffer=nl.psum, lazy_initialization=True)
  v7 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="identity_local_304", buffer=nl.sbuf)
  v8 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name="memset.311", buffer=nl.sbuf)
  v9 = nl.ndarray((nl.par_dim(1), 256), dtype=np.float32, name="multiply.11.314", buffer=nl.sbuf)
  v10 = nl.ndarray((2, nl.par_dim(4), 128), dtype=np.float32, name="", buffer=nl.sbuf)
  v11 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="multiply.11.317", buffer=nl.sbuf)
  v12 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.319", buffer=nl.sbuf)
  v13 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.int32, name="sine.12.321", buffer=nl.sbuf)
  v14 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.uint8, name="sine.12.325", buffer=nl.sbuf)
  v15 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.uint8, name="sine.12.327", buffer=nl.sbuf)
  v16 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.int32, name="sine.12.329", buffer=nl.sbuf)
  v17 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.331", buffer=nl.sbuf)
  v18 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.335", buffer=nl.sbuf)
  v19 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.339", buffer=nl.sbuf)
  v20 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.341", buffer=nl.sbuf)
  v21 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.343", buffer=nl.sbuf)
  v22 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.int32, name="cosine.13.345", buffer=nl.sbuf)
  v23 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.uint8, name="cosine.13.349", buffer=nl.sbuf)
  v24 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.uint8, name="cosine.13.351", buffer=nl.sbuf)
  v25 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.int32, name="cosine.13.353", buffer=nl.sbuf)
  v26 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.355", buffer=nl.sbuf)
  v27 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.359", buffer=nl.sbuf)
  v28 = nl.ndarray((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.363", buffer=nl.sbuf)
  v29 = nl.ndarray((2, nl.par_dim(128), 256), dtype=np.float32, name="", buffer=nl.sbuf)
  v30 = nl.zeros((2, nl.par_dim(128), 256), dtype=np.float32, name="multiply.11.367", buffer=nl.psum, lazy_initialization=True)
  v31 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.323", buffer=nl.psum, lazy_initialization=True)
  v32 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.333", buffer=nl.psum, lazy_initialization=True)
  v33 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="sine.12.337", buffer=nl.psum, lazy_initialization=True)
  v34 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.347", buffer=nl.psum, lazy_initialization=True)
  v35 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.357", buffer=nl.psum, lazy_initialization=True)
  v36 = nl.zeros((2, 2, nl.par_dim(128), 512), dtype=np.float32, name="cosine.13.361", buffer=nl.psum, lazy_initialization=True)
  v37 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name="memset.376", buffer=nl.sbuf)
  v38 = nl.shared_constant(np.identity(128, dtype=np.float32))

  def BB_entry_1():
    v37[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v8[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.memset(shape=(128, 128), value=np.dtype(np.float32).type(1), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v7[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v38[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    v9[0, nl.arange(256)[None, :]] = nl.load(v2[nl.arange(256)[None, :]], dtype=np.float32, mask=None)

    for i0 in nl.affine_range(2):
      v30[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]] = nisa.nc_matmul(v8[0, nl.arange(128)[None, :]], v9[0, nl.arange(256)[None, :]], is_stationary_onezero=True, is_moving_onezero=False, mask=None)
      v29[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]] = nl.copy(v30[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]], dtype=np.float32, mask=None)
      v10[i0, nl.arange(4)[:, None], nl.arange(128)[None, :]] = nl.load(v1[i0, nl.arange(4)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
      v6[i0, nl.arange(128)[:, None], nl.arange(4)[None, :]] = nisa.nc_matmul(v10[i0, nl.arange(4)[:, None], nl.arange(128)[None, :]], v7[nl.arange(4)[:, None], nl.arange(4)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
      v5[i0, nl.arange(128)[:, None], nl.arange(4)[None, :]] = nl.copy(v6[i0, nl.arange(128)[:, None], nl.arange(4)[None, :]], dtype=np.float32, mask=None)

      for i1 in nl.affine_range(2):
        for i2 in nl.affine_range(2):
          v11[i0, i1, nl.arange(128)[:, None], 256*i2+nl.arange(256)[None, :]] = nisa.tensor_scalar(data=v29[i0, nl.arange(128)[:, None], nl.arange(256)[None, :]], op0=nl.multiply, operand0=v5[i0, nl.arange(128)[:, None], i2+2*i1], reverse0=True, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
          """ end loop i2 """
        v12[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v11[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]],  op0=nl.add, operand0=np.dtype(np.float32).type(3.141592653589793), reverse0=False, op1=nl.multiply, operand1=np.dtype(np.float32).type(0.15915494309189535), reverse1=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
        v13[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v12[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.int32, mask=None)
        v31[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v13[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
        v14[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.greater(v31[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v12[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.uint8)
        v15[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v14[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.logical_xor, operand0=np.dtype(np.uint8).type(1), reverse0=False, dtype=np.uint8, mask=None, engine=nki.isa.engine.unknown)
        v16[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v13[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]],  op0=nl.multiply, operand0=np.dtype(np.int32).type(1), reverse0=False, op1=nl.add, operand1=np.dtype(np.int32).type(-1), reverse1=False, dtype=np.int32, mask=None, engine=nki.isa.engine.unknown)
        v17[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v14[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v16[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v32[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v15[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v13[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v18[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v17[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v32[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v33[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v18[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(-6.283185307179586), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
        v19[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v11[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v33[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v4[i1, i0, nl.arange(128)[:, None, None], 512*nl.arange(2)[None, :, None]+nl.arange(256)[None, None, :]] = nisa.activation(op=nl.sin, data=v19[i0, i1, nl.arange(128)[:, None, None], 256*nl.arange(2)[None, :, None]+nl.arange(256)[None, None, :]], bias=v37[nl.arange(128)[:, None, None], 0], scale=1.0, mask=None, dtype=np.float32)
        v20[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v11[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.add, operand0=np.dtype(np.float32).type(1.5707963267948966), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
        v21[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v20[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]],  op0=nl.add, operand0=np.dtype(np.float32).type(3.141592653589793), reverse0=False, op1=nl.multiply, operand1=np.dtype(np.float32).type(0.15915494309189535), reverse1=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
        v22[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v21[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.int32, mask=None)
        v34[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v22[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
        v23[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.greater(v34[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v21[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.uint8)
        v24[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v23[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.logical_xor, operand0=np.dtype(np.uint8).type(1), reverse0=False, dtype=np.uint8, mask=None, engine=nki.isa.engine.unknown)
        v25[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v22[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]],  op0=nl.multiply, operand0=np.dtype(np.int32).type(1), reverse0=False, op1=nl.add, operand1=np.dtype(np.int32).type(-1), reverse1=False, dtype=np.int32, mask=None, engine=nki.isa.engine.unknown)
        v26[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v23[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v25[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v35[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v24[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v22[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v27[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v26[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v35[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v36[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v27[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(-6.283185307179586), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
        v28[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v20[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], v36[i0, i1, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
        v4[i1, i0, nl.arange(128)[:, None, None], 512*nl.arange(2)[None, :, None]+nl.arange(256)[None, None, :]+256] = nisa.activation(op=nl.sin, data=v28[i0, i1, nl.arange(128)[:, None, None], 256*nl.arange(2)[None, :, None]+nl.arange(256)[None, None, :]], bias=v37[nl.arange(128)[:, None, None], 0], scale=1.0, mask=None, dtype=np.float32)
        nl.store(v3[i0, 2*i1+nl.arange(2)[None, :, None, None], nl.arange(128)[:, None, None, None], 256*nl.arange(2)[None, None, :, None]+nl.arange(256)[None, None, None, :]], value=v4[i1, i0, nl.arange(128)[:, None, None, None], 512*nl.arange(2)[None, :, None, None]+256*nl.arange(2)[None, None, :, None]+nl.arange(256)[None, None, None, :]], mask=None)
        """ end loop i1 """
      """ end loop i0 """

  BB_entry_1()
  return v3

def whisper_create_sinusoids(x0, x1):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((2, 4, 128)).to(device), x1.to(device))
    return out.cpu().reshape((1024, 512))
