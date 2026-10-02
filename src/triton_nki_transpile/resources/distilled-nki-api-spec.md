# NKI API Reference

## Module: `nki`
- `nki.jit(func=None, mode='auto', **kwargs)`
  - Decorator to compile a top-level NKI function for NeuronDevices.

## Module: `nki.isa` (Instruction Set Architecture)

### Computation & Logic
- `nki.isa.activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None)`
  - Apply an activation function (and optional reduction) on input tile using Scalar Engine.
- `nki.isa.activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None)`
  - Apply activation and reduction along the free dimension (equivalent to `activation` with `reset_reduce`).
- `nki.isa.affine_select(dst, pattern, offset, channel_multiplier, on_true_tile, on_false_value, cmp_op=<function equal>, name=None)`
  - Select elements between a tile and a scalar based on an affine pattern predicate using GpSimd Engine.
- `nki.isa.dma_compute(dst, srcs, scales, reduce_op, name=None)`
  - Perform scaled element-wise operations and reduction using DMA engines.
- `nki.isa.dropout(dst, data, prob, name=None)`
  - Randomly replace elements with zeros based on probability, using Vector Engine.
- `nki.isa.iota(dst, pattern, offset, channel_multiplier=0, name=None)`
  - Generate a constant literal pattern in SBUF using GpSimd Engine.
- `nki.isa.max8(dst, src, name=None)`
  - Find the 8 largest values in each partition of the source tile.
- `nki.isa.nc_find_index8(dst, data, vals, name=None)`
  - Find indices of 8 given values in each partition of the data tensor.
- `nki.isa.nc_match_replace8(dst, data, vals, imm, dst_idx=None, name=None)`
  - Replace first occurrence of values in `vals` with `imm` in `data`.
- `nki.isa.nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, tile_position=(), tile_size=(), perf_mode=matmul_perf_mode.none, name=None)`
  - Matrix multiplication (`dst = stationary.T @ moving`) using Tensor Engine.
- `nki.isa.nc_matmul_mx(dst, stationary, moving, stationary_scale, moving_scale, tile_position=None, tile_size=None, name=None)`
  - Matrix multiplication of MXFP8/MXFP4 quantized matrices (NeuronCore-v4+).
- `nki.isa.range_select(dst, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.idle, reduce_res=None, reduce_op=<function maximum>, range_start=0.0, on_false_value=0.0, name=None)`
  - Select elements based on comparison with bounds using Vector Engine (NeuronCore-v3+).
- `nki.isa.reciprocal(dst, data, name=None)`
  - Compute element-wise reciprocal (1.0/x) using Vector Engine.
- `nki.isa.scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None)`
  - Apply `(data <op0> operand0) <op1> operand1` with Vector Engine.
- `nki.isa.select_reduce(dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=<function maximum>, reverse_pred=False, name=None)`
  - Selectively copy elements based on predicate with optional reduction.
- `nki.isa.sequence_bounds(dst, segment_ids, name=None)`
  - Compute sequence bounds (start/end indices) for segment IDs using GpSimd Engine.
- `nki.isa.tensor_partition_reduce(dst, op, data, name=None)`
  - Apply reduction across partitions using GpSimd Engine.
- `nki.isa.tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None)`
  - Reduce free axes of input tile using Vector Engine.
- `nki.isa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=engine.unknown, name=None)`
  - Apply up to two math operators by broadcasting scalar/vector operands.
- `nki.isa.tensor_scalar_cumulative(dst, src, op0, op1, imm0, imm1=None, reduce_cmd=reduce_cmd.reset_reduce)`
  - Tensor-scalar arithmetic with cumulative reduction.
- `nki.isa.tensor_scalar_reduce(dst, data, op0, operand0, reduce_op, reduce_res, reverse0=False, name=None)`
  - Perform `tensor_scalar` op then reduce along free dimension.
- `nki.isa.tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None)`
  - Element-wise operation of two tiles using Vector or GpSimd Engine.
- `nki.isa.tensor_tensor_scan(dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None)`
  - Scan operation of two input tiles using Vector Engine.

### Data Movement & Memory
- `nki.isa.dma_copy(dst, src, dst_rmw_op=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, name=None)`
  - Copy data from src to dst using DMA, with optional read-modify-write.
- `nki.isa.dma_transpose(dst, src, axes=None, dge_mode=dge_mode.unknown, name=None)`
  - Transpose input src using DMA Engine.
- `nki.isa.local_gather(dst, src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, name=None)`
  - Gather SBUF data using index on GpSimd Engine.
- `nki.isa.memset(dst, value, engine=engine.unknown, name=None)`
  - Initialize dst with a constant value.
- `nki.isa.nc_n_gather(dst, data, indices, name=None)`
  - Gather elements from data by indices using GpSimd Engine.
- `nki.isa.nc_stream_shuffle(dst, src, shuffle_mask, name=None)`
  - Cross-partition data movement within a quadrant of 32 partitions.
- `nki.isa.nc_transpose(dst, data, engine=engine.unknown, name=None)`
  - 2D transpose between partition and free axis.
- `nki.isa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, name=None)`
  - Point-to-point NeuronCore communication (NeuronCore-v3+).
- `nki.isa.tensor_copy(dst, src, engine=engine.unknown, name=None)`
  - Copy src tile to dst within on-chip SRAMs.
- `nki.isa.tensor_copy_dynamic_dst(dst, src, engine=engine.unknown, name=None)`
  - Copy tile to dynamic offset destination.
- `nki.isa.tensor_copy_dynamic_src(dst, src, engine=engine.unknown, name=None)`
  - Copy tile from dynamic offset source.
- `nki.isa.tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None)`
  - Conditionally copy elements based on a predicate.

### Random Number Generation
- `nki.isa.rand2(dst, min, max, name=None)`
  - Generate uniform random numbers (NeuronCore-v4+).
- `nki.isa.rand_get_state(dst, engine=engine.unknown, name=None)`
  - Store current PRNG states to SBUF.
- `nki.isa.rand_set_state(src_seeds, engine=engine.unknown, name=None)`
  - Seed the PRNG inside the engine.
- `nki.isa.rng(dst, engine=engine.unknown, name=None)`
  - Generate 32 random bits per element.
- `nki.isa.set_rng_seed(src_seeds, name=None)`
  - Seed the PRNG in the Vector Engine.

### Utilities & Control Flow
- `nki.isa.bn_aggr(dst, data, name=None)`
  - Aggregate `bn_stats` outputs to generate mean and variance.
- `nki.isa.bn_stats(dst, data, name=None)`
  - Compute mean and variance stats for batch normalization.
- `nki.isa.core_barrier(data, cores, engine=engine.unknown, name=None)`
  - Synchronize multiple NeuronCores (NeuronCore-v3+).
- `nki.isa.get_nc_version()`
  - Return the current target's `nc_version`.
- `nki.isa.quantize_mx(dst, src, dst_scale, name=None)`
  - Quantize FP16/BF16 to MXFP8 tensors (NeuronCore-v4+).
- `nki.isa.register_alloc(x=None)`
  - Allocate a virtual register, optionally initialized with `x`.
- `nki.isa.register_load(dst, src)`
  - Load scalar value from memory into virtual register.
- `nki.isa.register_move(dst, imm)`
  - Move compile-time constant into virtual register.
- `nki.isa.register_store(dst, src)`
  - Store virtual register value into memory.

### Enums & Classes
- `nki.isa.dge_mode`: `unknown`, `swdge`, `hwdge`, `none`
- `nki.isa.engine`: `tensor`, `vector`, `scalar`, `gpsimd`, `dma`, `sync`, `unknown`
- `nki.isa.nc_version`: `gen2` (Trn1/Inf2), `gen3` (Trn2), `gen4` (Trn3)
- `nki.isa.reduce_cmd`: `idle`, `reset`, `reduce`, `reset_reduce`, `load_reduce`

## Module: `nki.language` (Language Primitives)

### Tensor Creation
- `nki.language.ndarray(shape, dtype, buffer=None, name='')`
  - Create a tensor of given shape and dtype on buffer.
- `nki.language.zeros(shape, dtype, buffer=None, name='')`
  - Create a tensor filled with zeros.

### Iterators & Flow
- `nki.language.affine_range(start, stop=None, step=1)`
  - Parallel loop iterator (use when no loop-carried dependency).
- `nki.language.sequential_range(start, stop, step)`
  - Sequential loop iterator (use when loop-carried dependency exists).
- `nki.language.static_range(start, stop=None, step=1)`
  - Fully unrolled loop iterator.

### Utilities
- `nki.language.device_print(print_prefix, tensor)`
  - Print message and tensor value during execution (debug only).
- `nki.language.ds(start, size)`
  - Build a dynamic slice for tensor indexing.
- `nki.language.num_programs(axes=None)`
  - Get number of SPMD programs along axes.
- `nki.language.program_id(axis)`
  - Get index of current SPMD program along axis.
- `nki.language.program_ndim()`
  - Get number of dimensions in SPMD launch grid.

### Constants & Types
- Buffers: `sbuf`, `psum`, `hbm` (`private_hbm`), `shared_hbm`
- Types: `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32`, `float16`, `float32`, `bfloat16`, `tfloat32`, `bool_`, `float8_e4m3`, `float8_e5m2`
- Tile Sizes: `nki.language.tile_size` (constants like `pmax`, `psum_fmax`, etc.)
