"""Compiler-generated NKI kernel for `mistral_attn_bias_from_mask`.

Source:
  HLO from `pipeline/models.py::mistral_attn_bias_from_mask` (JAX trace) compiled by neuronx-cc
  with `--internal-tensorizer-debug-pass=' --dump-after=All --dump-nki --dump-files ...'`.
  From `mistral_attn_bias_from_mask-sg0000/nki_penguin-159-InferSharedMemLoc.py` (last non-empty Penguin pass). Mechanically converted from the
  compiler's destination-passing `@trace` form to the `@nki.jit` return-value form
  used by NKIBenchB2 — see `pipeline/post_process.py`.

Description:
  Mistral bool→float attention-bias conversion (0 / -inf). Source: models/mistral/modeling_flax_mistral.py:325-329 @ v4.46.3

Compiler-chosen layout:
  logical inputs : [[4, 1, 512, 512]]
  kernel inputs  : [[2, 2, 1, 128, 2048]]
  kernel output  : [[2, 2, 1, 128, 2048]]
  logical output : [4, 1, 512, 512]
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
  v2 = nl.ndarray((2, 2, 1, 128, 2048), dtype=np.float32, buffer=nl.shared_hbm)


  v3 = nl.ndarray((2, 2, 2, nl.par_dim(128), 1024), dtype=np.float32, name="input0_local_223", buffer=nl.sbuf)
  v4 = nl.ndarray((2, 2, 2, nl.par_dim(128), 1024), dtype=np.float32, name="", buffer=nl.sbuf)
  v5 = nl.ndarray((2, 2, 2, 2, nl.par_dim(128), 512), dtype=np.uint8, name="compare.6.251", buffer=nl.sbuf)
  v6 = nl.ndarray((2, 2, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name="select.7.253", buffer=nl.sbuf)

  def BB_entry_1():
    for i0 in nl.affine_range(2):
      for i1 in nl.affine_range(2):
        for i2 in nl.affine_range(2):
          v3[i0, i1, i2, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[i0, i1, 0, nl.arange(128)[:, None], 1024*i2+nl.arange(1024)[None, :]], dtype=np.float32, mask=None)

          for i3 in nl.affine_range(2):
            v5[i0, i1, i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v3[i0, i1, i2, nl.arange(128)[:, None], 512*i3+nl.arange(512)[None, :]], op0=nl.greater, operand0=np.dtype(np.float32).type(0.0), reverse0=False, dtype=np.uint8, mask=None, engine=nki.isa.engine.unknown)
            v6[i0, i1, i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.memset(shape=(128, 512), value=np.dtype(np.float32).type(0.0), dtype=np.float32, mask=None, engine=nki.isa.engine.unknown)
            v4[i0, i1, i2, nl.arange(128)[:, None], 512*i3+nl.arange(512)[None, :]] = nl.where(v5[i0, i1, i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]], x=v6[i0, i1, i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]], y=nl.fp32.min, mask=None, dtype=np.float32)
            """ end loop i3 """
          nl.store(v2[i0, i1, 0, nl.arange(128)[:, None], 1024*i2+nl.arange(1024)[None, :]], value=v4[i0, i1, i2, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
          """ end loop i2 """
        """ end loop i1 """
      """ end loop i0 """

  BB_entry_1()
  return v2

def mistral_attn_bias_from_mask(x0):
    """Logical entry point. See module docstring for the layout mapping."""
    device = xm.xla_device()
    out = sg_kernel(x0.reshape((2, 2, 1, 128, 2048)).to(device))
    return out.cpu().reshape((4, 1, 512, 512))
