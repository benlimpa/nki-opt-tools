# Knob Extraction Guide (v2)

Each NKI optimization performance model in v2 has a typed `Knobs` dataclass
of structural facts the agent must extract from kernel source. Profile
metrics (latency, engine times, byte counts) are NOT knobs — they come from
`profile.json` (the output of `nki-profile`).

Workflow:
1. Read your kernel source and the profiler output.
2. For each opt below, decide whether the pattern applies. Set `applicable: true`
   only if the pattern is present and not already optimized.
3. Fill in the knob values. Use `null` for any field you cannot determine.
4. Save as `knobs.json` and run:
   ```bash
   ./python-in-venv nki-opt-models-v2/predict_all_v2.py \
       --profile profile.json --knobs knobs.json --output predictions.json
   ```
5. Read `predictions.json` — pick the highest-speedup applicable opt that has
   no `extrapolation_warning`, then implement it.

---

## opt1 — SBUF Accumulator

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_hbm_accumulator` — True iff the kernel allocates a running accumulator in HBM (buffer=nl.private_hbm) and reloads + stores it on every iteration of an outer range(M // TILE_M) loop. The profiler signature is spill_save_bytes > 0 with a per-iteration store+reload pattern. Set to false if the accumulator is already in SBUF or no such pattern exists.
- `M` — Total partition-axis dimension being reduced (rows). The outer loop is range(M // TILE_M). Null if pattern does not apply.
- `TILE_M` — Partition tile size driving the reduction loop. Must be ≤ 128. Null if pattern does not apply.
- `N` — Free-axis size of the accumulator tile. Governs the bytes spilled per round-trip (TILE_M * N * dtype_bytes). Null if pattern does not apply.

## opt3 — Full Partition Utilization

**Required profile fields:** `t_total_us`

- `has_underutilized_partition_tiling` — True iff the kernel loops sequential_range(M // TILE_P) with TILE_P < 128 over tiles of shape [TILE_P, *] in SBUF (i.e. the partition axis is being under-tiled). Set to false (model returns None) if the kernel already uses TILE_P=128 or doesn't tile the partition axis this way.
- `M` — Total partition-axis dimension (the dimension being looped over in tiles of TILE_P). Read from the kernel's primary ndarray shape (e.g. M in nl.ndarray((TILE_P, N), ...) inside range(M // TILE_P)). Must be divisible by 128. Null if not applicable.
- `TILE_P_current` — Current partition tile size — the divisor in range(M // TILE_P). Any value < 128 is a symptom of under-utilization. Null if TILE_P is already 128 or the pattern does not apply.

## opt5 — Loop Fusion — 2-loop single-intermediate variant

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_hbm_intermediate` — True iff the kernel has exactly one private_hbm buffer used as an intermediate between two sequential affine_range/sequential_range loops over the same tile range. Profiler signature: spill_save_bytes > 0 with spill_reload_bytes > 0. If there are 2+ intermediates, use opt07 instead.
- `n_tiles` — Tile-loop trip count (= M // TILE_P). Used for applicability + extrapolation — the speedup is governed by the structural ratio rather than n_tiles. Null if pattern does not apply.

## opt6 — Skip Masked Compute

**Required profile fields:** `t_total_us`

- `has_padding_iterations` — True iff the kernel's tile loop iterates over more tiles than contain real data — i.e. the upper bound includes padding tiles whose output is unused. Set to false (model returns None) if every tile holds valid data.
- `total_tiles` — Current loop bound — the range argument of the tile loop (typically M // TILE_P). Null if the pattern does not apply.
- `valid_tiles` — Number of tiles containing real (non-padding) data. Must be a Python int known at trace time and strictly less than total_tiles. Null if all tiles are valid.

## opt7 — Loop Fusion — Eliminate HBM Intermediates

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_hbm_intermediate` — True iff the kernel has ≥1 private_hbm buffer used as an intermediate between two sequential affine_range/sequential_range loops over the same tile range. Profiler signature: spill_save_bytes > 0 with spill_reload_bytes > 0. Set to false if all loops are already fused.
- `n_intermediates` — Number of HBM intermediates that will be eliminated by fusion. For K sequential loops sharing the same tile range, this is K-1 (2 for the canonical 3-loop pipeline). Null if pattern does not apply.
- `n_tiles` — Tile-loop trip count (= M // TILE_P). Used only for applicability + extrapolation — the speedup is governed by n_intermediates rather than n_tiles. Null if pattern does not apply.

## opt10 — Delay Softmax Division

**Required profile fields:** `t_total_us`

- `has_premature_exp_store` — True iff the kernel stores exp(x - max) to nl.private_hbm during the softmax reduction pass and reloads it for the normalization pass (profiler shows spill_save_bytes > 0 during the reduction). Set to false if division is already delayed in pass 3.
- `n_tiles` — Number of column tiles in the softmax loop (= N_total // TILE_N). Each tile contributes one write + one read of an exp intermediate that gets eliminated by the optimization. Null if the pattern does not apply.

## opt11 — nc_matmul on Large Contiguous Blocks

**Required profile fields:** `t_total_us`

- `has_small_nc_matmul_tiles` — True iff the kernel calls nisa.nc_matmul with a moving-tile width (the inner N-loop step / second dim of the moving operand) strictly less than min(N, 512). Set to false if already at the hardware maximum.
- `M` — Matmul output rows (= stationary operand free dim). Null if not applicable.
- `K` — Matmul contraction dim (= partition dim of both operands). Null if not applicable.
- `N` — Matmul output cols (= moving operand free dim). Null if not applicable.
- `TILE_N_current` — Current moving-tile width (the value of the inner N-loop step). Must be < min(N, 512) to be optimizable. Null if not applicable.

## opt12 — nc_matmul PSUM Accumulation Group

**Required profile fields:** `t_total_us`

- `uses_sequential_range_for_k` — True iff the K-tile reduction loop uses nl.sequential_range, which forces a PSUM→SBUF eviction + VectorE add per K-tile. Set to false if the K-loop already uses nl.affine_range (compiler forms a PSUM accumulation group automatically).
- `n_m_tiles` — Number of M-tiles in the outer loop (= M // TILE_M). Null if the pattern does not apply.
- `n_k_tiles` — Number of K-tiles per M-tile in the inner reduction loop (= K // TILE_K). Must be ≥ 2 for the optimization to have any effect. Null if the pattern does not apply.

## opt13 — Lower Input Precision (bfloat16 matmul inputs)

**Required profile fields:** `t_total_us`, `t_dma_us`, `t_tensor_us`

- `has_fp32_inputs` — True iff the matmul currently loads input tiles (lhsT, rhs) as float32 in HBM. Profiler signature: hbm_read_bytes matches (M·K + K·N)·4. Set false if inputs are already bfloat16/fp16. Output precision is unchanged (PSUM accumulates in f32 regardless).
- `M` — Matmul output rows (lhsT.T @ rhs → [M, N]).
- `K` — Matmul reduction dimension.
- `N` — Matmul output columns.
- `dtype_ratio` — Ratio target_bytes/current_bytes for the input dtype switch. 0.5 for bfloat16 from float32 (default). 0.25 for int8 from float32. Set null to use the default 0.5.

## opt14 — Double Buffering

**Required profile fields:** `t_total_us`, `t_dma_us`, `t_tensor_us`

- `has_serial_dma_compute` — True iff a sequential_range (or otherwise serial-dep) loop over M-tiles uses a single SBUF buffer for the loop-carried operand (e.g. lhsT). Profiler signature: per-tile DMA on the loop-carried operand is comparable to per-tile TensorE time, and they execute serially. Set false if the kernel already double-buffers, has no per-tile DMA, or n_tiles < 2.
- `loop_carried_operand_is_per_tile_loaded` — v3 — REQUIRED. True iff the matmul kernel has a single SBUF buffer for an operand that is RE-LOADED FROM HBM every tile (the canonical double-buffering target). False if the loop-carried operand is computed in SBUF (no DMA to overlap), or if DMA cost is already dominated by writes rather than the per-tile read. Without this, the model silently over-predicts on layernorm-style kernels where the per-tile DMA is store traffic.
- `n_tiles` — Number of sequential_range iterations the optimization will double-buffer (typically M // TILE_M). Must be ≥ 2.

## opt16 — Pipeline Compute and DMA (3-Stage)

**Required profile fields:** `t_total_us`, `t_dma_us`, `t_tensor_us`

- `n_tiles` — Number of pipeline tiles (M // TILE_M). Pattern requires n_tiles >= 3 for the 3-stage pipeline to have steady-state savings. Set to null if the kernel is not tile-loop structured.
- `has_serial_dma_compute` — True iff the kernel currently serializes DMA load → compute → DMA store per tile (no software pipeline). Set to false if already pipelined with ping-pong buffers.

## opt17 — SBUF Instead of HBM (Eliminate private_hbm Intermediate)

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_private_hbm_intermediate` — True iff the kernel allocates at least one nl.ndarray with buffer=nl.private_hbm for a value produced and consumed within the same kernel invocation (not an HBM input/output). Profiler signature: spill_save_bytes > 0 with spill_reload_bytes > 0. Set false if already using nl.sbuf or nl.psum, or if the intermediate is too large to fit in SBUF.
- `n_intermediates` — Number of private_hbm intermediates that will be moved to SBUF. Typically 1 per private_hbm allocation. Default 1.
- `n_tiles` — Tile-loop trip count. Used for applicability + extrapolation; the speedup is governed by the structural ratio rather than n_tiles. Null if pattern does not apply.

## opt18 — Stronger K-Tiling

**Required profile fields:** `t_total_us`, `t_tensor_us`, `t_dma_us`

- `has_fine_k_tiling` — True iff the kernel runs an inner loop over K-tiles (e.g. `for k in affine_range(K // TILE_K)`) with TILE_K < 128 inside an nc_matmul accumulation pattern. Set to false otherwise.
- `K` — Full contraction-axis size of the matmul (e.g. inner dim of A.T @ B). Must be ≥ 128 and divisible by 128 for the optimization to apply. Null if not applicable.
- `TILE_K_current` — Kernel's current K-tile size. The optimization replaces it with TILE_K=128 (or a smaller divisor of K if K < 128). Null if not applicable.

## opt20 — Algebraic Scalar Fusion

**Required profile fields:** `t_total_us`

- `has_consecutive_scalar_ops` — True iff the kernel chains 2+ consecutive nisa.tensor_scalar (or scalar nl.add/nl.multiply) ops on the same tile in the same iter, with no intervening non-scalar ops that break the chain. Profiler signature: scalar_engine_active_time is non-trivial AND vector_engine_instruction_count is dominated by scalar dispatches (>= 2x the count needed for the kernel's arithmetic intensity).
- `n_consecutive_ops` — Length of the longest collapsible scalar-op chain (must be >= 2). For AdamW's 4-step update (m_t, v_t, m_hat, p update), this is 4. The optimization replaces n_consecutive_ops dispatches with 1.
- `n_tiles` — Outer tile-loop iteration count. Each tile pays the per-chain dispatch overhead once.

## opt21 — Fuse Single Loop (Small Intermediate, Packet Overhead)

**Required profile fields:** `t_total_us`

- `has_small_hbm_intermediate` — True iff the kernel uses a *small* private_hbm intermediate between two affine_range loops (profiler shows spill_save_bytes > 0 but the spill volume is tiny — e.g. row means [M,1] = ~bytes, not full tiles [M,N] = ~MB). The bottleneck is DMA packet setup overhead, not bandwidth. Set false if already fused, or if the intermediate is large (use opt07 instead — bandwidth-anchored model).
- `n_tiles` — Tile-loop trip count for each loop (M // TILE_M). Each tile incurs the per-boundary packet overhead.
- `n_loop_boundaries` — Number of inter-loop boundaries that will be fused out (= n_loops − 1). Typically 1 for a 2-loop kernel.

## opt24 — Lower-Precision Internal Buffers

**Required profile fields:** `t_total_us`, `t_dma_us`, `t_vector_us`

- `has_fp32_internal_buffers` — True iff the kernel allocates internal SBUF tiles as nl.float32 while the HBM input/output tensors are bfloat16 (or float16). Each dma_copy then implicitly casts and doubles DMA packets. Set to false if the SBUF dtype already matches the HBM dtype.
- `P` — Partition-axis size of the SBUF tile (typically 128 = NeuronCore partition count). Null if not applicable.
- `N` — Free-axis total size of the input tensor. Null if not applicable.
- `n_fp32_intermediates` — Number of fp32 SBUF allocations per tile iteration that will be downcast to bf16. Higher counts amplify the savings. Null if not applicable.

## opt30 — Fuse Reduction-Transformation Loops (Small Intermediate)

**Required profile fields:** `t_total_us`

- `has_reduction_transformation_split` — True iff the kernel splits a reduction (variance/RMS/mean/softmax-denom) and the corresponding transformation across two affine_range loops, with a small private_hbm intermediate for the reduction result. Profiler signature: spill_save_bytes > 0 (small, ~KB) AND static_dma_packet_count >> 9. Set false if already fused, or if the intermediate is large (use opt07 instead).
- `n_tiles` — Tile-loop trip count for each loop (M // TILE_M). Each tile incurs the per-boundary packet overhead.
- `n_loop_boundaries` — Number of inter-loop boundaries that will be fused out (= n_loops − 1). Typically 1 for a reduction+transformation kernel.

## opt31 — Load Larger Blocks for SBUF Reuse

**Required profile fields:** `t_total_us`, `t_dma_us`, `t_vector_us`

- `has_small_tile_pattern` — True iff the kernel tiles the free dimension into small chunks and inner-loops over them (e.g. `for n in affine_range(N // TILE_N): load x[..., n*TILE_N:(n+1)*TILE_N]`). Set to false if the kernel already loads full free-dimension blocks.
- `tile_body_contains_full_row_reduce` — v3 — REQUIRED for applicability. True iff the inner tile loop body contains a `tensor_reduce` (or equivalent partial-reduce) over the tiled axis whose result is later combined across tiles. The opt's speedup comes from collapsing partial reductions into one full-row reduce (see nki-optimizations/31-load-larger-blocks/README.md). Set false for kernels whose tile loop is a pure element-wise or store loop — those won't benefit from this optimization, and the model would silently over-predict on producer-consumer pipelines.
- `is_producer_consumer_pipeline` — v3 — REQUIRED. True iff this tile loop sits in a producer-consumer pipeline (e.g. matmul output feeding a downstream norm/reduce). When True, the model returns None because the HBM-caching assumption (reload same row repeatedly) does not hold — the producer's stores evict the cache between consumer reads. Set False for standalone row-wise kernels that re-read the same input across tiles.
- `N` — Full free-axis size (e.g. the kernel's `N` constant or the full second dim of the input). Null if not applicable.
- `TILE_N_current` — The kernel's current free-axis tile size. The optimization replaces this with TILE_N=N (one block per M-tile). Null if not applicable.

## opt32 — Multi-level Tiling (Stationary Operand Reuse)

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_n_outer_k_inner_with_a_reload` — True iff the matmul kernel iterates N-tiles in the OUTER loop and K-tiles in the INNER loop, with A (lhsT) loaded inside the K loop (so A reloads once per N-tile). Profiler signature: hbm_read_bytes > (M·K + K·N)·4. Set false if the compiler already hoists A loads or the loop order is K-outer.
- `M` — Matmul output rows (lhsT.T @ rhs → [M, N]).
- `K` — Matmul reduction dimension.
- `N` — Matmul output columns.
- `n_n_tiles` — Number of N-tiles in the outer loop (= N // TILE_N). Each N-tile currently triggers a full A reload.

## opt33 — Coalesce SBUF Tiles for HBM Store

**Required profile fields:** `t_total_us`

- `has_fragmented_stores` — True iff the kernel issues multiple nisa.dma_copy(...→HBM) calls to *adjacent* HBM addresses inside the same iteration. Profiler signature: high dma_packet_count relative to total HBM write bytes. Set false if stores are already wide / coalesced via a single dma_copy of a wide SBUF tile.
- `n_iters` — Outer loop iteration count of the kernel section that performs the fragmented stores.
- `n_fragments_per_iter` — Number of separate SBUF→HBM dma_copy calls per iteration that will be merged into one. Must be ≥ 2.

## opt34 — Parallel Scan with MatMul Carry-over

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_sequential_scan` — True iff the kernel runs a row-by-row 1D scan via sequential_range(P) with a per-iteration carry-over (e.g., nisa.tensor_tensor_scan with `initial=carry_sb`). Set to false if the scan is already parallelized or the kernel is not a 1D scan.
- `N_total` — Total number of elements being scanned (e.g., the length of the 1D input array). Must be divisible by 128 (hardware partition count) for the optimization to apply. Null if not applicable.

## opt35 — Specialized Instructions

**Required profile fields:** `t_total_us`, `t_dma_us`

- `has_manual_broadcast` — True iff the kernel uses an inner affine_range(N) loop of nisa.tensor_tensor to broadcast a [P, 1] tile across [P, N]. v3 — ALSO TRUE for any pattern that a specialized NKI ISA instruction collapses into one dispatch, including: (a) sqrt+add(eps)+reciprocal chains replaceable by rsqrt, (b) erf/tanh-based GELU sequences replaceable by nisa.activation with the gelu opcode, (c) softmax-denom chains (exp -> tensor_reduce -> reciprocal -> tensor_tensor) replaceable by nisa.softmax. The speedup mechanism is the same: N inner-loop instructions collapse into 1 fused ISA call. Set false only if no specialized instruction matches the kernel's scalar chain.
- `broadcast_dim_N` — Size of the free axis (N) being manually broadcast in the inner loop (e.g., the range argument of affine_range(N)). Set to null if the pattern does not apply.
- `fused_ops` — Number of element-wise ops (scale, bias) sharing the manual broadcast loop, fused into one nisa.activation call after the optimization. Typically 1 (scale-only) or 2 (scale + bias). Null if not applicable.
- `num_outer_iters` — Number of times the enclosing outer loop runs (e.g., the range argument of the surrounding affine_range / sequential_range). Null if not applicable.

## opt37 — DMA Transpose

**Required profile fields:** `t_total_us`, `t_tensor_us`

- `has_nc_transpose` — True iff the kernel calls nisa.nc_transpose on an SBUF tile that was just loaded from HBM (the load+transpose can be fused into a single nisa.dma_transpose). Set to false if no such pattern.
- `num_iters` — How many times the transpose runs (e.g. range argument of the enclosing affine_range/sequential_range). 1 if no outer loop. Null if the pattern does not apply.
- `D0` — Partition-axis size of the transposed tile (≤128, typically 128). Null if not applicable.
- `D3` — Free-axis size of the transposed tile (the columns being swapped with rows). Null if not applicable.

## opt38 — SPMD Launch Grid

**Required profile fields:** `t_total_us`

- `has_no_spmd_grid` — True iff the wrapper invokes the kernel as kernel(args) (no launch-grid syntax) — i.e. nl.num_programs(0) == 1 inside the kernel. Set False (model returns None) if the kernel is already launched as kernel[G](args) with G > 1.
- `has_independent_outer_loop` — True iff the outer sequential_range iterations are independent — no running scan, accumulator, cross-iteration data dependency, or any other carry-over between loop iterations. Set False if any iteration reads state written by a previous iteration.
- `outer_loop_count` — Count of outer-loop iterations (e.g. M // TILE_P in the typical tile-row pattern). Must be divisible by target_grid_size for the shard math to work. Null if not applicable.
- `target_grid_size` — Candidate SPMD grid size to predict for (the G in kernel[G](...)). Sweep candidates by setting this to 1, 2 (and optionally 4) and re-running the predictor; pick the highest-speedup non-extrapolating prediction. Must be ≥ 1 and ≤ 8 (NKI hard limit).
