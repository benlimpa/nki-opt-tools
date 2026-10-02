#!/usr/bin/env python3
"""Verify >=2 NeuronCores can be ALLOCATED inside this container.

Launches a trivial SPMD NKI kernel as `kernel[2](...)`. If NRT can't allocate
the 2 LNCs (stuck driver state, wrong LNC config, no second core), the launch
fails immediately and we exit non-zero. This is stronger than `neuron-ls`,
which only reports static topology and won't catch the `Available:0` failure
mode (see Case 1 in benchmark-opt/reports/mg_v2_giveup_traces.md).

Run inside the agent's container as the first command of its `bash -c` so the
result reflects exactly what the agent will see.
"""
import sys
import time

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import numpy as np
except Exception as e:
    print(f"[neuron-probe] could not import NKI: {e}", file=sys.stderr)
    sys.exit(2)


@nki.jit
def _probe(x_in):
    out = nl.ndarray(x_in.shape, dtype=x_in.dtype, buffer=nl.shared_hbm)
    tile = nl.load(x_in)
    nl.store(out, tile)
    return out


def main() -> int:
    t0 = time.time()
    try:
        x = np.zeros((128, 1), dtype=np.float32)
        result = _probe[2](x)
        _ = np.asarray(result)
        print(f"[neuron-probe] OK — kernel[2] allocated and ran in {time.time()-t0:.1f}s")
        return 0
    except Exception as e:
        msg = str(e)
        if "Logical Neuron Core" in msg or "nrt_allocate" in msg or "Available:0" in msg:
            print("[neuron-probe] NRT could not allocate 2 LNCs (likely stuck driver state):", file=sys.stderr)
        else:
            print(f"[neuron-probe] kernel[2] launch failed ({type(e).__name__}):", file=sys.stderr)
        print(f"[neuron-probe]   {msg[:800]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
