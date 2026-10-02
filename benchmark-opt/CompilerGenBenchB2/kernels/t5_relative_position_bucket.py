"""Compiler-generated NKI kernel for `t5_relative_position_bucket`.

Source:
  HLO from `pipeline/models.py::t5_relative_position_bucket` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `t5_relative_position_bucket-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  T5 log-bucketed relative-position bucketing. Source: models/t5/modeling_flax_t5.py:232-265 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[128, 128]]
  kernel inputs  : [[128, 2, 64]]
  kernel output  : [[128, 2, 64]]
  logical output : [128, 128]
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
def sg_kernel(v1):
  v2 = nl.ndarray((128, 2, 64), dtype=np.int32, buffer=nl.shared_hbm)


  v3 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="", buffer=nl.sbuf)
  v4 = nl.ndarray((nl.par_dim(128), 1), dtype=np.int32, name="multiply.22.130", buffer=nl.sbuf)
  v5 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="abs.16.289", buffer=nl.sbuf)
  v6 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="abs.16.291", buffer=nl.sbuf)
  v7 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="abs.16.293", buffer=nl.sbuf)
  v8 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.uint8, name="compare.13.295", buffer=nl.sbuf)
  v9 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.15.297", buffer=nl.sbuf)
  v10 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.15.299", buffer=nl.sbuf)
  v11 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.uint8, name="compare.17.301", buffer=nl.sbuf)
  v12 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.float32, name="abs.16.303", buffer=nl.sbuf)
  v13 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.float32, name="log.20.305", buffer=nl.sbuf)
  v14 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.float32, name="multiply.22.307", buffer=nl.sbuf)
  v15 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.float32, name="multiply.22.309", buffer=nl.sbuf)
  v16 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.float32, name="multiply.22.311", buffer=nl.sbuf)
  v17 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.313", buffer=nl.sbuf)
  v18 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.uint8, name="multiply.22.317", buffer=nl.sbuf)
  v19 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.uint8, name="multiply.22.319", buffer=nl.sbuf)
  v20 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.321", buffer=nl.sbuf)
  v21 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.323", buffer=nl.sbuf)
  v22 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.325", buffer=nl.sbuf)
  v23 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.327", buffer=nl.sbuf)
  v24 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.329", buffer=nl.sbuf)
  v25 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.331", buffer=nl.sbuf)
  v26 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="multiply.22.333", buffer=nl.sbuf)
  v27 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="minimum.0.335", buffer=nl.sbuf)
  v28 = nl.ndarray((2, nl.par_dim(128), 64), dtype=np.int32, name="select.0.337", buffer=nl.sbuf)
  v29 = nl.zeros((2, nl.par_dim(128), 64), dtype=np.float32, name="multiply.22.315", buffer=nl.psum, lazy_initialization=True)
  v30 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name="memset.341", buffer=nl.sbuf)

  def BB_entry_1():
    v30[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
    v4[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.int32).type(8), dtype=np.int32, mask=None, engine=nki.isa.engine.unknown)

    for i0 in nl.affine_range(2):
      v5[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.load(v1[nl.arange(128)[:, None], i0, nl.arange(64)[None, :]], dtype=np.int32, mask=None)
      v6[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.activation(op=nl.sign, data=v5[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], bias=v30[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.int32)
      v7[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.multiply(v5[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v6[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      v8[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.tensor_scalar(data=v5[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], op0=nl.greater, operand0=np.dtype(np.int32).type(0), reverse0=False, dtype=np.uint8, mask=None, engine=nki.isa.engine.unknown)
      v9[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.memset(shape=(128, 64), value=np.dtype(np.int32).type(16), dtype=np.int32, mask=None, engine=nki.isa.engine.unknown)
      v10[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.where(v8[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], x=v9[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], y=np.dtype(np.int32).type(0), mask=None, dtype=np.int32)
      v11[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.tensor_scalar(data=v7[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], op0=nl.less, operand0=np.dtype(np.int32).type(8), reverse0=False, dtype=np.uint8, mask=None, engine=nki.isa.engine.unknown)
      v12[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.copy(v7[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], dtype=np.float32, mask=None)
      v13[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.activation(op=nl.log, data=v12[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], bias=v30[nl.arange(128)[:, None], 0], scale=0.125, mask=None, dtype=np.float32)
      v14[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.tensor_scalar(data=v13[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(2.885390043258667), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
      v15[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.activation(op=nl.sign, data=v14[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], bias=v30[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
      v16[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.multiply(v14[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v15[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.float32)
      v17[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.copy(v16[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], dtype=np.int32, mask=None)
      v29[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.copy(v17[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], dtype=np.float32, mask=None)
      v18[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.greater(v29[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v16[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.uint8)
      v19[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.tensor_scalar(data=v18[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], op0=nl.logical_xor, operand0=np.dtype(np.uint8).type(1), reverse0=False, dtype=np.uint8, mask=None, engine=nki.isa.engine.unknown)
      v20[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.tensor_scalar(data=v17[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]],  op0=nl.multiply, operand0=np.dtype(np.int32).type(1), reverse0=False, op1=nl.add, operand1=np.dtype(np.int32).type(-1), reverse1=False, dtype=np.int32, mask=None, engine=nki.isa.engine.unknown)
      v21[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.multiply(v18[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v20[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      v22[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.multiply(v19[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v17[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      v23[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.add(v21[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v22[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      v24[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.copy(v15[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], dtype=np.int32, mask=None)
      v25[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.multiply(v23[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v24[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      v26[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.add(v25[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v4[nl.arange(128)[:, None], 0], mask=None, dtype=np.int32)
      v27[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nisa.tensor_scalar(data=v26[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], op0=nl.minimum, operand0=np.dtype(np.int32).type(15), reverse0=False, dtype=np.int32, mask=None, engine=nki.isa.engine.unknown)
      v28[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.where(v11[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], x=v7[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], y=v27[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      v3[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.add(v10[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], v28[i0, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None, dtype=np.int32)
      """ end loop i0 """

    for i1 in nl.affine_range(2):
      nl.store(v2[nl.arange(128)[:, None], i1, nl.arange(64)[None, :]], value=v3[i1, nl.arange(128)[:, None], nl.arange(64)[None, :]], mask=None)
      """ end loop i1 """

  BB_entry_1()
  return v2

def t5_relative_position_bucket(x0):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((128, 2, 64)).to(device))
    return out.cpu().reshape((128, 128))
