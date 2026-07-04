---
title: "01 CUDA C Basics"
date: 2026-05-22
draft: false
tags: ["CUDA", "GPU Programming", "Parallel Programming", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "The CUDA C execution model from the ground up: host/device code separation, the 3-stage memory transfer, the thread/block/grid ↔ SM/warp/core mapping, and kernel launch syntax."
---

> Source: [01 CUDA C Basics](https://youtu.be/OsK8YFHTtNs)

# What Is CUDA?

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It's a C/C++ extension that opens up the GPU as a general-purpose compute device (**GPGPU**) rather than a graphics-only one, and since its 2007 release it has become the de facto standard for deep learning infrastructure.

But when someone says "I use CUDA," what exactly are they using? Running a model in PyTorch is CUDA. Writing a `__global__` kernel by hand is also CUDA. The confusion comes from the fact that CUDA is **a stack, not a single layer.**

![CUDA Stack](./images/image-7.png#small)

What people usually call "CUDA" spans **layers 3–5 (CUDA C/C++ · PTX · SASS).**

| Layer | Role |
| --- | --- |
| **CUDA C/C++** | The programming model you write directly. `__global__`, `threadIdx`, the Grid/Block/Thread abstraction |
| **CUDA Runtime API** | `cudaMalloc`, `cudaMemcpy`, kernel launches, etc. |
| **nvcc** | NVIDIA's compiler that lowers the above to `PTX` |
| **PTX** | A virtual ISA. Handles forward compatibility across generations (older-arch code still JITs onto newer GPUs) |
| **SASS** | The actual machine code, compiled from `PTX` per GPU generation. Architecture-specific |

---

## GPGPU: Using the GPU for General-Purpose Compute

**GPGPU** (General-Purpose computing on GPU) means exactly what it says: using the GPU for compute beyond graphics. Before deep learning took off, the GPU was mostly a device for drawing polygons — but today, any large-scale parallel numerical workload gets offloaded to it.

![VEGAS Pro GPU acceleration option](./images/image.png)
![Per-application GPU setting in NVIDIA Control Panel](./images/image-1.png)

This is why you see options like `CUDA - GPUs` in a video editor (VEGAS Pro) or in the NVIDIA Control Panel. They let a specific program pick which GPU runs its CUDA work — not for games, but for **GPGPU workloads** like video editing and machine learning.

The workloads where GPGPU shines all share one trait: **they repeat the same operation over huge amounts of data, independently.**

| Workload | Underlying operation |
| --- | --- |
| Video encoding / filters | Parallel numeric ops over a pixel matrix |
| DL training / inference | Tensor (multi-dim matrix) MatMul |
| Crypto mining | Massively parallel hash execution |
| Scientific simulation | Parallel updates of grid / particle systems |
| 3D rendering (Blender Cycles, etc.) | Per-ray parallel computation |

> As an aside: one of the earliest cases of using a GPU to train a neural network was a 2004 paper by Korean researchers (Oh & Jung, *"GPU implementation of neural networks,"* Pattern Recognition). There was no CUDA back then — they ran the network on shaders.

## Heterogeneous Computing: Dividing Work Between CPU and GPU

**Heterogeneous computing** means different architectures (CPU and GPU) cooperating within a single system. The key idea in CUDA is *not* "run everything on the GPU." Control flow and light logic stay on the CPU (Host); only the **heavy compute — matrix and tensor ops — is offloaded to the GPU (Device).**

![Heterogeneous Computing](./images/image-8.png)

### The 3-Stage Data Flow

The CPU (Host) and GPU (Device) have **separate, independent memory spaces.** A variable created on the CPU is not automatically visible in GPU memory, so you have to move the data yourself. Every CUDA program therefore goes through these three stages to bridge that **memory gap.**

![3-stage data flow](./images/image-10.png)

**1. Host → Device** — `cudaMemcpy`

```cpp
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
```

Copy the source data from CPU memory to GPU memory over a high-speed bus (PCIe, NVLink, etc.).

**2. Execute Kernel** — `<<<...>>>`

```cpp
kernel<<<gridDim, blockDim>>>(d_data);
```

Run the parallel kernel on the GPU to do the actual work.

**3. Device → Host** — `cudaMemcpy`

```cpp
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
```

Bring the finished result back from GPU memory to CPU memory.

![data flow diagram](./images/image-14.png)

> **Key Takeaway** — CPU and GPU memory are physically separate and don't share automatically. These three stages are the backbone of every CUDA program — and the **Host↔Device transfer is also CUDA's single biggest optimization bottleneck.** (How to reduce it is the subject of the next post.)

---

# CUDA C Syntax and Kernels

What kind of "heavy computation" is worth paying the communication bottleneck we just described? The example that most clearly shows the payoff in intro CUDA is **vector addition.** In `c[i] = a[i] + b[i]`, each index is independent, so one thread per element is all you need. A textbook embarrassingly parallel problem.

To run this on the GPU, you first have to declare **where each function runs and where it's called from.** CUDA C adds **function qualifiers** to C/C++ for exactly this.

| Qualifier | Runs on | Called from | Notes |
| --- | --- | --- | --- |
| `__global__` | Device (GPU) | Host (CPU) | The **kernel** that runs on the GPU. Must return `void` due to the asynchronous execution model |
| `__device__` | Device (GPU) | Device (GPU) | A helper callable only from within the GPU |
| `__host__` | Host (CPU) | Host (CPU) | An ordinary C/C++ function (the default, can be omitted). Any unqualified function is `__host__` |

## How CPU and GPU Code Coexist in One File: `nvcc`

A single `.cu` file can freely mix CPU code (`main`) and GPU code (`__global__`). NVIDIA's compiler **`nvcc`** scans the source, hands the ordinary code to a standard C compiler (GCC, MSVC, etc.), and pulls out only the `__global__` kernels to compile separately into GPU machine code.

## How Many Threads and Blocks Can You Have?

- **Threads per block** max out at **1024.** You're free to split across dimensions (`dim3`), but if the product exceeds 1024 the launch fails with `cudaErrorInvalidConfiguration`. `dim3(32, 32, 1)` (=1024) passes; `dim3(32, 32, 2)` (=2048) dies. There's also a separate cap of **64 on the z-axis** that's easy to forget.
- **Grids** are far more generous: 2³¹−1 on the x-axis and 65535 each on y/z. You won't hit these with any reasonable dataset.
- **Shared memory** is capped at **48 KB per block** for static allocation. That limit is identical across architectures because it's a backward-compatibility default. To use more, you opt in to dynamic allocation via `cudaFuncSetAttribute`, and the ceiling there varies by GPU (A100 ~163 KB, H100 / B200 ~227 KB). Beginners often miss this.

## Warps: Keep Thread Counts a Multiple of 32

The GPU executes threads in groups called **warps.** A warp is **32 threads** — a number fixed across every generation of NVIDIA GPU that developers cannot change.

So picking an awkward thread-per-block count costs you. Take **100**: the GPU reserves scheduling slots for 4 warps (128 threads) but only 100 actually work. The remaining **28 slots sit idle, so utilization drops to 100/128 ≈ 78%** — you're throwing away ~22% before you even start.

That's why block sizes are usually chosen from **128 / 256 / 512**, with **256** as the go-to default. But bigger isn't automatically better: as the block grows, per-thread register and shared-memory demand can hit the SM's limits and lower **occupancy** (how many blocks/warps an SM can hold at once). Too small, and scheduling overhead grows instead. It ultimately comes down to **balancing register / shared-memory usage against occupancy.**

## Blocks Are Strangers to Each Other

Threads within the same block can share shared memory and synchronize with `__syncthreads()`, because they physically live on the same SM.

But **different blocks can't do any of that.** They can't communicate and have no guaranteed execution order — Block 7 might finish before Block 0. If you need synchronization across blocks, you split the work into two separate kernel launches. (Hopper's Thread Block Clusters let you work around this partially, but we'll skip that here.)

The reason is that the **block↔SM mapping is a hardware constraint.** Threads in the same block can share shared memory because it physically sits in that SM's SRAM; blocks can't talk to each other because they're on different SMs with separate SRAM. In other words, the **Grid/Block/Thread software hierarchy is a direct exposure of the SM/warp/core hardware structure.**

Here's that hierarchy laid out by dimension:

![1D Grid/Block/Thread](./images/image-11.png)
*1D layout: `kernel<<<4, 8>>>` — 4 blocks × 8 threads = 32 threads total*

![2D Grid/Block/Thread](./images/image-12.png)
*2D layout: `kernel<<<dim3(2,2), dim3(4,4)>>>`*

![3D Grid/Block/Thread](./images/image-13.png)
*3D layout*

## Triple Chevron `<<< >>>`: Kernel Launch Syntax

Calling a `__global__` function like a normal function is a compile error. You must use the **triple chevron** syntax.

```cpp
mykernel<<<gridDim, blockDim>>>(args);
//        ^^^^^^^  ^^^^^^^^
//        number of blocks, threads per block
```

- **`gridDim`** — how many blocks are in the grid
- **`blockDim`** — how many threads are in each block
- Total threads = `gridDim × blockDim`

The simplest example:

```cpp
mykernel<<<1, 1>>>();   // 1 block, 1 thread → effectively sequential
```

To process N elements like in vector addition, you need N threads. The original video simplifies this to `<<<N, 1>>>`, but because of the warp efficiency we saw earlier, **it's more efficient to use 128–512 threads per block in practice.**

```cpp
int N = 10000;
int blockSize = 256;
int gridSize = (N + blockSize - 1) / blockSize;  // ceiling division
add<<<gridSize, blockSize>>>(a, b, c, N);
```

The numbers inside `<<< >>>` map directly onto the GPU's physical structure (SM · warp · block). Unlike other languages that hide the hardware, **CUDA exposes it and puts performance in the developer's hands.** That's both why you learn CUDA and why it's hard.
