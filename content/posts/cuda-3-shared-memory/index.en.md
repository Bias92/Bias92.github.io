---
title: "03 CUDA Shared Memory: Tiling, Bank Conflicts, and Reduction"
date: 2026-07-14
draft: false
tags: ["CUDA", "GPU Programming", "Shared Memory", "Warp Divergence", "Parallel Programming", "Reduction"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "Global memory coalescing, shared memory tiling, bank conflicts with padding and swizzle, occupancy, warp divergence and predication, and four stages of reduction, explained by how the code works."
---

A function that runs on the GPU is called a kernel, and the unit of work that executes a kernel is a thread. Threads are grouped into blocks, and one block runs from start to finish on a single SM (Streaming Multiprocessor), the execution unit inside the GPU. The threads of a block read and write two kinds of memory. Global memory is the large memory attached to the GPU that every thread can reach, and shared memory is a small memory inside the SM that only the threads of the same block use together. Shared memory is far faster than global memory, but the kernel code itself decides what to load into it and when to discard it, which makes it different from a cache that hardware fills on its own.

Shared memory lets a block reuse data that was read from global memory once. In exchange, because the threads of a block use the same memory together, three new problems appear: a barrier that orders writes and reads, a bank conflict that arises when several threads touch the same storage unit at the same time, and an occupancy problem where a larger block reduces the number of threads an SM can hold. Matrix multiplication, transpose, and reduction expose these three in turn. The starting point is how the data that will be loaded into shared memory is read from global memory.

## Global Memory and Coalescing

The GPU does not run threads one at a time. It bundles 32 of them and issues the same instruction to all of them at once; this bundle of 32 is a warp, and each thread's slot inside the warp is a lane. When a warp executes a global memory load, the 32 lanes each present their own address, and the memory system fetches those addresses in 32-byte units called sectors. Merging the accesses of many lanes into a small number of sector transfers is called coalescing. So the cost of a global memory access is set not by the number of lanes but by the number of distinct sectors actually touched.

A `float` is 4 bytes, so 32 lanes reading 32 consecutive floats cover a 128-byte range, and if the starting address is aligned to a 32-byte boundary that takes 4 sectors. If the starting address is shifted by one `float`, the same 128 bytes straddle one more sector boundary and take 5, and if the gap between lanes grows to 32 bytes or more, each lane lands in a different sector and the count grows to 32. In all three cases the useful data is the same 128 bytes, but the amount actually transferred is 128, 160, and 1024 bytes.

![Coalescing](images/coalescing.svg?v=2)

The pointer returned by `cudaMalloc` is sufficiently aligned, but the start of a sub-range made by adding an offset to it may not be. And the number 4 comes from the condition "32 lanes, one `float` each", so when the lane count or the data width changes, the minimum sector count changes with it.

## Shared Memory and Tiling

In the matrix product $C = A \times B$, one element of $C$ needs one row of $A$ and one column of $B$. The simplest kernel gives each thread one element of $C$ and reads every value it needs from global memory each time.

```cpp
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

Here `threadIdx` is the thread's index inside its block, `blockIdx` is the block's index inside the grid, and `blockDim` is the number of threads along one side of the block. Each iteration of the inner loop reads 8 bytes and does one multiply and one add. But the same row of $A$ is needed again for all $N$ elements in that row of $C$, so this kernel reads the same values from global memory many times.

Tiling manages that reuse directly in shared memory. A tile is a $T \times T$ square piece of a matrix. The threads of a block together copy one tile of $A$ and one tile of $B$ into shared memory, accumulate every partial product that tile pair allows, and then move on to the next tile pair. In this way each element of a tile is read from global memory once and reused $T$ times from shared memory.

![Tiling](images/tiling.svg?v=2)

```cpp
#define T 32

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N / T; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * T + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * T + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < T; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}
```

`__shared__` declares that a variable lives in shared memory. `__syncthreads()` is a barrier: threads that arrive at this line wait until every thread of the block has arrived. The first barrier keeps a thread from reading a tile that other threads have not finished filling, and the second keeps the next iteration from overwriting a tile that is still being read. The barrier comes with a rule. If some threads of a block return early while the rest reach `__syncthreads()`, the behavior is undefined. The code above has every thread follow the same path because $N$ is a multiple of $T$; code that accepts arbitrary sizes keeps out-of-range threads in the loop and the barrier instead of returning them, fills their loads with 0, and guards only the final store to `C` with a range check.

The reuse shows up as arithmetic intensity, the number of operations performed per byte read from global memory. One block copies $N/T$ tile pairs, reading $8NT$ bytes, and its $T^2$ threads each perform $2N$ operations, so

$$
I_{\text{tiled}} = \frac{2NT^2}{8NT} = \frac{T}{4}\ \text{FLOP/B}
$$

With $T = 32$ that is 8 FLOP/B, 32 times the 0.25 FLOP/B of reading every value without a tile. The tile copies read consecutive addresses along a row, so they also satisfy the coalescing condition from the previous section.

## Bank Conflict

Shared memory is divided into 32 banks. A bank is an independent storage unit that makes up shared memory, and consecutive 4-byte words are assigned to banks 0, 1, ..., 31, 0, 1, ... in rotation. When the 32 lanes of a warp touch different banks, they are served at once; when they touch different addresses in the same bank, they are served one after another. This serialization of $n$ accesses to the same bank is an $n$-way bank conflict. Several lanes reading the same address is a broadcast that hands one value to all of them, so it is not a conflict.

Conflicts typically appear when a two-dimensional tile is read column-wise, and a transpose through shared memory is the example. Transpose swaps the rows and columns of a matrix; by placing a tile read row-wise from global memory into shared memory and taking it out column-wise, both the read and the write can be coalesced.

```cpp
__shared__ float tile[32][32];

tile[threadIdx.y][threadIdx.x] = in[...];   // row-wise write: banks spread
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y];  // column-wise read: 32-way conflict
```

In the column-wise read, the lanes read `tile[0][c], tile[1][c], ...`, which are 32 words apart, so all of them land in bank `c`. There are two ways to fix this.

The first is padding. Making the row length 33 shifts the bank number by one per row as the access walks down a column.

```cpp
__shared__ float tile[32][33];
```

In general, reading with stride $S$ gives a conflict degree of $\gcd(S, 32)$. Stride 32 is 32-way, 33 is conflict-free, and 2 and 4 are 2-way and 4-way. Padding is therefore the act of making the common divisor of the stride and the bank count equal to 1, at a cost of 4 wasted bytes per row.

The second is swizzle. XOR-ing the column index with the row number at store time gives the same effect without using more memory.

```cpp
__shared__ float tile[32][32];

tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[...];   // write: banks spread
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];  // read also spread
```

Once the columns inside a row are rearranged by XOR, both row-wise and column-wise accesses land on each of the 32 banks exactly once. The cost is one index computation and the requirement that the tile width be a power of two.

![Bank conflict](images/bank-conflict.svg?v=3)

It is also possible to flip the mapping between lanes and rows and columns so that the shared memory access becomes consecutive, but then the global memory access becomes strided again. That moves the conflict on one side into a coalescing failure on the other, so it is not a fix. The tiled matrix multiply in the previous section does not have this problem. `Bs[k][threadIdx.x]` runs along a row, so the banks spread, and `As[threadIdx.y][k]` is a broadcast because the lanes of a warp read the same address.

## Occupancy and Block Size

Occupancy is the number of warps actually resident on an SM divided by the maximum number of warps the SM can hold at once. It matters because of how the GPU hides memory latency. While one warp waits for a global memory response, the SM runs another warp that is resident on the same SM, so the more warps are resident, the more of the waiting is covered. The number of blocks an SM can hold is set by whichever of four limits runs out first: threads, block slots, registers, and shared memory. A register is the fastest storage inside the SM, where a thread keeps the values it is computing with.

In the tiled matrix multiply, $T = 32$ makes one block 1024 threads. Compute capability is the number that identifies the generation of CUDA hardware features a GPU supports, and a GPU with compute capability 8.9 has a resident-thread limit of 1536 per SM, so only one 1024-thread block fits on an SM. That fills 1024 of the 1536 thread slots, and occupancy becomes 66.7%. With $T = 16$ the block is 256 threads, six of them fit, and all 1536 slots are filled.

![Occupancy](images/occupancy-residency.svg)

Enlarging the tile increases reuse but also enlarges the block and reduces the parallelism an SM can hold. This is why the way to stack more reuse is not a bigger block but register tiling, where one thread keeps several elements of $C$ in registers and accumulates them. And occupancy is only an upper bound on the number of warps the SM can choose from, not a fraction of execution time, so a kernel with low occupancy can still be fast if each warp has plenty to do.

## Warp Divergence

A warp issues the same instruction to its 32 lanes at once. When the condition of an `if` differs from lane to lane, some lanes of the warp must take the true path and the rest the false path. This situation is warp divergence. With lane number $\ell \in \{0,\ldots,31\}$ and branch condition $p_\ell$, the set $A$ of true lanes and the set $B$ of false lanes are

$$
A = \{\ell \mid p_\ell = 1\}, \qquad
B = \{\ell \mid p_\ell = 0\}
$$

and divergence occurs when both sets are non-empty.

```cpp
int lane = threadIdx.x & 31;

if (lane < 16)
    A();
else
    B();
```

In this code, lanes 0–15 of every warp choose `A` and lanes 16–31 choose `B`. A warp cannot run both paths at once, so it first runs the `A` path with only lanes 0–15 enabled, and then the `B` path with only lanes 16–31 enabled. The 32-bit value that records which lanes write the result of the current instruction is the active mask.

Suppose the `A` and `B` paths compile to $n_A$ and $n_B$ warp instructions. If only one path is chosen, that region issues $n_A$ instructions; if both are chosen, it issues $n_A + n_B$. The lane count does not multiply the issue count, so whether the split is 16:16 or 31:1, the number of instructions issued is the same if the two paths are the same length. With $\eta$ as the fraction of issued lane slots that are actually enabled,

$$
\eta =
\frac{|A|n_A + |B|n_B}
     {32(n_A+n_B)}
$$

and when the two paths are the same length, $n_A = n_B$, this gives $\eta = 1/2$.

To avoid divergence, align the condition with warp boundaries.

```cpp
int warp = threadIdx.x >> 5;

if ((warp & 1) == 0)
    A();
else
    B();
```

Even-numbered warps choose `A` with all 32 lanes and odd-numbered warps choose `B` with all 32, so the condition is uniform inside each warp. Different warps running different code is not divergence.

A diverging `if` in the source does not always produce a real branch either. When the body is short, the compiler removes the branch and turns it into a predicated instruction. Predication issues the instruction once to all lanes and lets only the lanes whose condition is true write the result.

```cpp
float y = x;
if (lane < 16)
    y = 2.0f * x;
```

In SASS, the GPU machine code, this conceptually becomes the form below. The exact instruction names and registers vary with architecture and compiler version.

```text
ISETP.LT ... P0, lane, 16
@P0 FMUL  y, x, 2.0
```

In this case the warp does not split into two paths. `FMUL` is issued once and only the lanes with predicate `P0` true write the result. So there is no branch divergence, but the disabled lanes do no useful work on that instruction. A divergent branch and predication can both lower the fraction of enabled lanes, but they are not the same phenomenon.

## Reduction

Reduction turns an array of $N$ values into a single value. Sums, maxima, and means belong here, and they keep appearing inside ML kernels as the maximum and denominator sum of softmax and the mean and variance of layernorm.

Folding as a tree, half of the threads combine two values at each step and the number of steps is $\log_2 N$. The total number of additions is still $N - 1$, so parallelization reduces depth, not the amount of work. In return, every step needs a guarantee that the previous writes have finished, and how far that synchronization cost is reduced is the difference between the four versions below.

![Reduction tree](images/reduction-tree.svg?v=2)

The common structure is multi-pass. Each block forms its share of the partial sum in shared memory, and the same kernel is run again on the array of partial sums until one value remains. For example, with $2^{24}$ inputs and a block of 256, the kernel runs three times: 65,536 → 256 → 1.

Version 0 is the literal tree. `tid` is the thread's index inside the block, and `buf` is the input the block has loaded into shared memory.

```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        buf[tid] += buf[tid + s];
    __syncthreads();
}
```

Two things make it slow. The first is scattered active lanes. The lanes that satisfy the condition thin out to the even-numbered ones at `s = 1` and multiples of four at `s = 2`, and although the number of working lanes drops to 16, 8, 4, ..., the warp's instructions are still issued. The second is the `%` operation. The divisor `2 * s` changes every iteration, so the compiler cannot turn it into a bit operation and a cluster of division instructions remains.

Version 1 is sequential addressing. It gathers the working threads at the front of the block.

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        buf[tid] += buf[tid + s];
    __syncthreads();
}
```

Both versions reduce 256 elements in eight steps, and the number of threads doing an addition at step $j \in \{0,\ldots,7\}$ is the same, $a_j = 256 / 2^{j+1}$. The difference is how many warps those $a_j$ threads are spread across. In version 0 the active threads are scattered across the whole block, so the number of warps with at least one true condition is $\min(8, a_j)$, which sums to 47 over the eight steps. Version 1 packs the active threads at the front, giving $\lceil a_j / 32 \rceil$, which sums to 12. So at `s = 128` the first four warps work as whole warps, at `s = 64` two, and at `s = 32` one, and in these three steps the condition is uniform per warp. From `s = 16` on, the condition splits inside the first warp, but the active lanes are contiguous from the front. `buf[tid]` and `buf[tid + s]` are also consecutive addresses, so there is no bank conflict, and the `%` is gone.

![Reduction lanes](images/reduction-lanes.svg)

Version 2 is warp shuffle. From `s = 16` on, only the first warp works, so from that boundary the fold happens in registers without shared memory or `__syncthreads()`.

```cpp
if (tid < 32) {
    float x = buf[tid] + buf[tid + 32];
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, off);
    if (tid == 0) out[blockIdx.x] = x;
}
```

`__shfl_down_sync` passes a register value directly between lanes inside a warp; its first argument is a mask of the participating lanes and its third is how many lanes below to take the value from. This removes the shared memory round trips and block barriers of the last six steps. The mask `0xffffffffu` is valid because the whole first warp satisfies `tid < 32`; in code where only some lanes participate, every participating lane must execute the same function with the same mask.

Version 3 is one atomic per block. An atomic is an operation that guarantees updates to the same address from many threads are applied one at a time even when they arrive together. Instead of multi-pass, lane 0 of each block executes `atomicAdd(out, x)` once to add the block's partial sum straight into the result. An atomic per element would touch the same address $N$ times, but after the block reduction it is touched only as many times as there are blocks. This version needs a `cudaMemset` to zero the result variable before it runs, and `float` atomics add in an order that changes from run to run, so they do not guarantee bit-identical results.

A reduction with one operation per element is purely bound by memory bandwidth, so the ceiling for a well-written reduction is the speed of a memcpy of the same size. `DeviceReduce::Sum` from CUB, the library shipped with CUDA, reaches that range for arbitrary types and sizes, and version 3 above shows the same structure fixed to a single `float` array.

## Source Code

The full sources of the three kernels are in [gemm_bench.cu](/code/cuda-03/gemm_bench.cu), [transpose_bench.cu](/code/cuda-03/transpose_bench.cu), and [reduce_bench.cu](/code/cuda-03/reduce_bench.cu).

```bash
nvcc -O3 -arch=sm_89 -o gemm_bench gemm_bench.cu
nvcc -O3 -arch=sm_89 -o transpose_bench transpose_bench.cu
nvcc -O3 -arch=sm_89 -std=c++17 -o reduce_bench reduce_bench.cu
```

## References

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): the reference document for coalescing, shared memory, bank conflicts, occupancy, and branch predication
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): the exact meaning of SIMT divergence, synchronization, atomics, and warp intrinsics
- [Mark Harris, An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/): the standard example showing coalescing, shared tiles, and padding in one place
- [Andreas Holt, Shared-Memory Tiled Matrix Multiplication](https://andreasholt.com/posts/shared-tiled-matmul/): tiled GEMM with figures and boundary handling
- [Lei Mao, CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/): details of bank address mapping
- [Lei Mao, CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/): details of swizzle address mapping
- [Fabian Schütze, Visualizing Bank Conflicts](https://fabianschuetze.github.io/bankconflictscuda.html): supplementary notes on bank behavior in modern architectures
- [Mark Harris, Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf): the classic seven-step reduction; it is old, so do not copy its warp-synchronous code as is
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/): shuffle and hierarchical atomics; read the code with the modern `__shfl_down_sync()`
- [Lei Mao, CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/): a write-up centered on a batched reduction implementation
- [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html): the higher-level reference on threadblock/warp/thread tiling, register reuse, and double buffering
- [Simon Boehm, How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM): a worklog from register tiling up to warptiling
- [CUB](https://nvidia.github.io/cccl/cub/): the production implementation layered as `WarpReduce → BlockReduce → DeviceReduce`
