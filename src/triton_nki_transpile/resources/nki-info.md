# **Information about the NKI API (Beta 2)**

1. **Layout:** By default, NKI infers the first dimension (leftmost) as the partition dimension of Tensor. In Beta 2, the partition dimension **must** be the leftmost dimension. The dimensions on the right of partition dimensions are the free dimension F where elements are read and written sequentially.
2. **Constraints:** NKI requires the free dimensions size of PSUM to not exceed the architecture limitation of 512. Each partition of SBUF buffer cannot exceed 192KB.
3. **Partition Limits:** NKI requires the number of partitions of a tile to not exceed the architecture limitation of 128.
4. `nc_matmul`: `nisa.nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False)`: `nisa.nc_matmul` computes transpose(stationary) @ moving matrix multiplication using Tensor Engine. The `nc_matmul` instruction must read inputs from SBUF and write outputs to PSUM. Therefore, the stationary and moving must be SBUF tiles, and the result tile (`dst`) is a PSUM tile. 128x128 stationary + 128x512 moving can achieve optimal throughput. Note: In Beta 2, `dst` is a mandatory first argument.
   Parameters:
   * dst - the destination operand on PSUM.
   * stationary - the stationary operand on SBUF; layout: (partition axis <= 128, free axis <= 128)
   * moving - the moving operand on SBUF; layout: (partition axis <= 128, free axis <= 512)
   * is_stationary_onezero - hints to the compiler whether the stationary operand is a tile with ones/zeros only.
   * is_moving_onezero - hints to the compiler if the moving operand is a tile with ones/zeros only.
   * is_transpose - hints to the compiler that this is a transpose operation with moving as an identity matrix.
5. **Transposition:** `nisa.nc_transpose(dst, src)` is equivalent to and has the same performance as `nisa.nc_matmul(dst, src, identity_matrix, is_moving_onezero=True, is_transpose=True)`.
6. **Activation Functions:** `nki.language.sigmoid`, `nki.language.rsqrt`, and `nki.language.silu` can be used as activation functions for `nisa.activation(dst, src, op)`.
7. **Data Movement:** `nl.load` and `nl.store` are deprecated. Use `nisa.dma_copy(dst, src)` to move data between HBM and SBUF.

## **Output dependencies**

NKI requires iterations between `affine_range` can be executed in parallel require synchronization on the output. As a result, each iteration of the loop has to write to a different memory location.
Wrong code:
```
a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
for i in nl.affine_range(4):
    a[0] = 0 # Unexpected output dependencies, different iterations of i loop write to ‘a[0]‘
```

To fix the problem, you could either index the destination with the missing indices:
Correct code:
```
a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
for i in nl.affine_range(4):
    a[i] = 0 # Ok
```

Or if you want to write to the same memory location, you could use `sequential_range` which allows writing to the same memory location:
Alternative code:
```
a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
for i in nl.sequential_range(4):
    a[0] = 0 # Also ok, we dont expect the sequential_range to execute in parallel
```

## **Tensor indexing**

NKI requires either use basic indexing or advanced indexing but not both. Beta 2 uses standard Python indexing/slicing; `nl.arange` and `nl.mgrid` are deprecated for indexing.
Basic indexing:
Given an N-dimensional array, `x`, `x[index]` invokes basic indexing whenever index is a tuple containing any combination of the following types of objects:

* integers
* slice objects
* Ellipsis objects
* None

Examples of basic indexing:
```
x[..., 0]
x[:, k * TILE_K: (k + 1) * TILE_K]
x[k * TILE_K: (k + 1) * TILE_K, n * TILE_N: (n + 1) * TILE_N]
```

Advanced indexing:
Given an N-dimensional array, `x`, `x[index]` invokes advanced indexing whenever index is:

* an integer-type or boolean-type `nl.ndarray`
* a tuple with at least one sequence-type object as an element (deprecated in Beta 2 indexing context; use standard slices for boundary conditions instead of masks).

Example of updated boundary handling (replacing masks):
```
p_start = i * 128
p_end = min(M, p_start + 128)
p = slice(p_start, p_end)
nisa.dma_copy(dst=result[p, f], src=t[p, f])
```

## **Tensor usage scope**

In NKI, control blocks in if/else/for statements will introduce their own scope for tensors. A tensor defined in if/else/for control blocks are not allowed to be used outside of the scope.
Wrong code:
```
for i in range(4):
    if i < 2:
        tmp = nl.ndarray(shape=a.shape, dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tmp, src=a)
    else:
        tmp = nl.ndarray(shape=b.shape, dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tmp, src=b)
    nisa.dma_copy(dst=c, src=tmp) # Error: Local variable ’tmp’ is referenced outside of its parent scope ...
```

Correct code:
```
for i in range(4):
    tmp = nl.ndarray(shape=a.shape, dtype=a.dtype, buffer=nl.sbuf)
    if i < 2:
        nisa.dma_copy(dst=tmp, src=a)
    else:
        nisa.dma_copy(dst=tmp, src=b)
    nisa.dma_copy(dst=c, src=tmp)
```

Wrong code (Shadowing):
```
data = nl.ndarray((128, 128), dtype=np.float32, buffer=nl.sbuf)
nisa.memset(data, 0)
for i in nl.sequential_range(4):
    i_tile = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(i_tile, 0) # simplified example
    data = data + i_tile # Warning: shadowing local tensor with a new object
    nisa.dma_copy(dst=ptr, src=data)
```

Correct code:
```
data = nl.ndarray((128, 128), dtype=np.float32, buffer=nl.sbuf)
nisa.memset(data, 0)
for i in nl.sequential_range(4):
    i_tile = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(i_tile, 0)
    data[...] = data + i_tile # Update the existing object
    nisa.dma_copy(dst=ptr, src=data)
```

## **Access variables**

1. Don’t use slice with variable size (must be compile-time constant).
2. List indices must be integers or slices.
3. Shape elements must be integers.
4. `InstTile` (intermediate results) cannot be directly assigned to a tensor; use `nisa` instructions to write to an allocated `nl.ndarray`.

## SPMD (Single Program Multiple Data) execution with LNC (Logical Neuron Cores)

A kernel may be launched with multiple logical neuron cores (LNCs) in SPMD execution mode. Each LNC executes the same code but operates on different data. The `nl.program_id(dim)` function can be used to get the current LNC's ID, which can be used for indexing and partitioning data across LNCs.

However, in the current architecture, the limit is 2 LNCs, therefore, we can only have one dimension for `nl.program_id()`, which is `nl.program_id(0)`. The value of `nl.program_id(0)` can be either 0 or 1, indicating which LNC is executing the code.

Launching a kernel in SPMD mode is done like so:
```
@nki.jit
def kernel(input):
    lnc_id = nl.program_id(0)
    ...
# Launch the kernel with 2 LNCs
kernel[2](input)
```
