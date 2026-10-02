# Optimization catalog

Measured speedups are not distributed.

| ID | Optimization | Apply when | Skip when |
|---|---|---|---|
| 01 | SBUF Accumulator | Loop accumulates into HBM buffer (`private_hbm`); `spill_save_bytes > 0` | Accumulator too large for SBUF |
| 03 | Full Partition Utilization | `TILE_P < 128`; tensor partition dim ≥ 128 | Already TILE_P=128 |
| 05 | Inline Function Fusion | Separate functions share a tile loop; HBM intermediate between them; `spill_save_bytes > 0` | Functions compute on different dimensions/tiles |
| 06 | Skip Masked Tiles | Some tiles are all-padding/invalid; loop iterates all tiles including padding | All tiles contain valid data |
| 07 | Loop Fusion | ≥2 loops over same dim; HBM intermediate between them; `spill_save_bytes > 0` | Cross-tile data dependencies between loops |
| 10 | Delay Softmax Division | Softmax stores exp tiles to HBM between reduction and normalization passes | Reduction and normalization can't be separated |
| 11 | nc_matmul Max Moving Tile | `TILE_N < 512` with `nc_matmul`; N is 512-aligned | Already using TILE_N=512 |
| 12 | PSUM Accumulation Group | K-tile loop uses `sequential_range` + manual VectorE add; high VectorE% in matmul | Already using `affine_range` for K-tiles |
| 13 | Lower Input Precision | nc_matmul inputs are float32; bf16 input precision acceptable; DMA-bound | Output precision requires float32 inputs |
| 14 | Double Buffering | Sequential load→compute loop; DMA load is significant; SBUF fits 2× tile | Loop has <2 iterations; SBUF too tight |
| 16 | Full Compute-DMA Pipeline | Load + compute + store all significant; `active_time% < 85%`; SBUF fits 2×input + 2×output | Output store is negligible |
| 17 | SBUF Intermediate (not HBM) | Two-loop pass: loop1 writes intermediate to HBM; loop2 reads it; `spill_save_bytes > 0` | Intermediate too large for SBUF |
| 18 | Stronger K-Tiling | `TILE_K < 128`; K > 128; rhs same across M-tiles; many nc_matmul calls per M-tile | Already TILE_K=128 |
| 20 | Fuse Dependent Scalar Ops | Consecutive `tensor_scalar` chain fused via two-op form; kernel is compute-bound | Kernel is DMA-bound (gain < 5%) |
| 21 | Fuse Producer-Consumer Loops | Loop A writes HBM; Loop B reads it; same iteration bounds; no other consumers of HBM output | Other code needs the HBM intermediate |
| 24 | Lower Precision SBUF Buffers | SBUF buffers are float32 for bf16 input data; compute-bound; large SBUF footprint | Compute requires float32 intermediate precision |
| 29 | PSUM Fused Bias Eviction | Scalar scale/bias applied right after PSUM eviction as separate instruction; compute-bound | Kernel is DMA-bound (gain negligible) |
| 30 | Fuse Reduction + Transform | Two-pass: pass1 reduces to HBM (per-row scalar); pass2 reads reduction + input | Reduction output needed by other kernels |
| 31 | Load Larger Free-Dim Blocks | `TILE_N << N`; full tile fits in SBUF; compiler NOT already caching (check `hbm_read_bytes > 1×`) | Compiler already loads data only once (verify!) |
| 32 | Multi-Level Tiling | Stationary (A) matrix reloaded per N-tile; multiple N-tiles fit in PSUM | A already hoisted by compiler LICM |
| 33 | Coalesce Stores | Many small SBUF→HBM stores to adjacent addresses; high DMA packet count for stores | Stores already contiguous and large |
| 34 | Parallelize 1D Scan | Serial cumsum / prefix scan over large 1D array; underutilizing all 128 partitions | Operation has non-associative dependencies |
| 35 | Specialized Instructions | Explicit Python loop over free dim for elementwise op; `nisa.activation` or `tensor_scalar` can replace it | Operation isn't expressible as a fused instruction |
| 37 | DMA Transpose | `nc_transpose` used to transpose a tile loaded from HBM | Tile is generated in SBUF (no HBM load to intercept) |
| 38 | SPMD Launch Grid | Kernel is *not* launched with an SPMD grid (`nl.num_programs(0) == 1`); outer loop iterations are independent (no cross-iteration dependencies / running scan / cross-tile accumulator); outer dim divisible by candidate grid sizes; running on multi-NC hardware (sweep `G ∈ {1, 2}` and pick best) | Kernel already launched with an SPMD grid (`kernel[G](...)` with G > 1); cross-iteration dependencies; outer dim too small to shard; single-NC target |
