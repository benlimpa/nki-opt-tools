# opt38 SPMD launch grid — implementation reference

This document specifies the **correct codegen pattern** for opt38 (SPMD launch
grid). A buggy pattern recurred in earlier runs and produced **half-output
defects** whenever the outer-loop iteration count did not divide evenly across
the launch grid. The pattern below is the verified replacement.

The bug: contiguous `ceil`-divided ranges leave one program with zero tiles
when `outer_iters < G`, or with disjoint ranges that the launch-grid output
buffer aggregation does not merge. The fix: a **strided tile assignment**
that is correct for *any* outer-loop iteration count, including odd counts
and counts smaller than `G`.

---

## The pattern (use this)

```python
@nki.jit
def kernel_spmd(x, ...):
    # ... allocate output as shared_hbm so all programs write into the
    # same buffer; each program writes a disjoint subset of rows.
    out = nl.ndarray(shape=(M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    NUM_LNCS  = nl.num_programs(0)   # launch-grid size G
    LNC_ID    = nl.program_id(0)     # this program's index in [0, G)

    n_p_tiles = (M + TILE_P - 1) // TILE_P   # ceil — works for any M

    # Strided assignment: program k handles tiles {k, k+G, k+2G, ...}
    # Correct for any n_p_tiles, including n_p_tiles < G (some programs idle).
    for i_m in nl.affine_range(n_p_tiles):
        if i_m % NUM_LNCS != LNC_ID:
            continue

        m_start = i_m * TILE_P
        # In-kernel bounds guard — the last tile may be a partial tile.
        # No need for separate "last-tile" code paths.
        m_end   = nl.minimum(m_start + TILE_P, M)
        # ... load x[m_start:m_end, :], compute, store into out[m_start:m_end, :]

    return out
```

### Wrapper

```python
# Wrapper Function:
def kernel(x, ...):
    G = 2                       # target Trainium2 LNC count
    return kernel_spmd[G](x, ...)
```

---

## Why this pattern is correct on every shape

| shape | n_p_tiles | G | program 0 takes | program 1 takes | full coverage? |
|---|---|---|---|---|---|
| M=128 (1 tile)  | 1   | 2 | {0}                | {}                  | yes (program 1 idles, no missing rows) |
| M=256 (2 tiles) | 2   | 2 | {0}                | {1}                 | yes |
| M=5472 (43 tiles)  | 43  | 2 | {0, 2, ..., 42}    | {1, 3, ..., 41}     | yes (interleaved) |
| M=16416 (129 tiles)| 129 | 2 | {0, 2, ..., 128}   | {1, 3, ..., 127}    | yes (interleaved) |

Both the **oracle correctness shape** (small M) and the **bench shape** (large
M) are served by the same kernel. There is no need for a wrapper-level
small-shape fallback or for an even-tile parity guard.

---

## Common buggy patterns — DO NOT use

### Anti-pattern 1: contiguous `ceil`-division

```python
# WRONG — empties one program when n_p_tiles < G
tiles_per_program = (n_p_tiles + NUM_LNCS - 1) // NUM_LNCS
start = LNC_ID * tiles_per_program
end   = min(start + tiles_per_program, n_p_tiles)
for i_m in nl.affine_range(end - start):
    ...
```

Fails on `M=128, G=2`: `tiles_per_program=1`, program 0 takes tile 0,
program 1 starts at index 1 ≥ end=1 → takes zero tiles → output rows
TILE_P..M are never written → **half-output defect**.

### Anti-pattern 2: even-tile parity guard

```python
# WRONG — silently runs single-core whenever n_p_tiles is odd
grid = 2 if n_p_tiles % 2 == 0 and n_p_tiles >= 2 else 1
```

This guard is a **defensive reaction** to the contiguous-split bug; once you
use the strided pattern above, the guard is unnecessary. The cost of the
guard on the `bench3` scoreboard is that 2 of 3 shapes silently fall back
to `G=1` whenever the tile count is odd (e.g. `n_p_tiles` ∈ {43, 86, 129}).


---

## Verification checklist (run BEFORE shipping)

Run the oracle on the **smallest correctness shape**, not just `bench()`:

```bash
./nki-profile --kernel-path optimized_kernel.py --oracle-path oracle.py
```

Without `--bench` the profiler uses the small `run_test()` shapes. A passing
profile here means the strided pattern handles short tile counts correctly.

Then run `--bench3` to confirm the speedup holds across all three benchmark
shapes — not just the canonical `bench()` shape:

```bash
./nki-profile --kernel-path optimized_kernel.py --oracle-path oracle.py --bench3
```

A correct opt38 lands a non-trivial speedup at every shape. If the small
shape regresses and the mid shape speeds up, you are still in the
contiguous-split / parity-guard failure mode.

---

## When SPMD alone is not enough — bandwidth-bound kernels

If the predictor's `notes` for opt38 say "DMA-share guard active: NN% of
t_total is on the shared HBM bus", the kernel is bandwidth-bound. The two
physical NCs share the HBM bus, so sharding alone gives a speedup well below
G. To get a real win you must **reduce HBM traffic first** (larger tiles,
blocked operand reuse, packed stores) and then add SPMD on top.

Check `optimization-quick-reference.md` for the traffic-reducing opts that
pair with opt38 on bandwidth-bound kernels (opt23 larger SBUF tile, opt32
multi-level tiling, opt33 coalesce stores, opt17 SBUF-instead-of-HBM).
