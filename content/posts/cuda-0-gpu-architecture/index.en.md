---
title: "00 GPU Architecture Primer: The Tesla Foundation"
date: 2026-05-15
draft: false
tags: ["CUDA", "GPU Architecture", "Tesla", "SIMT", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "Before CUDA C: the Tesla unified architecture (G80/GT200) from the 2008 IEEE Micro paper. The graphics-to-compute data path (Input Assembler, work distribution, SPA, TPC, SM/SP, SFU/LSU, ROP, DRAM), how the warp and SIMT were born in hardware, clock domains, and how every structure becomes a CUDA term."
---

> Source: Lindholm, Nickolls, Oberman, Montrym, *"NVIDIA Tesla: A Unified Graphics and Computing Architecture,"* IEEE Micro 28(2), 2008.

## Why Read This Before CUDA C

The [CUDA C post](../cuda-c-basics/) talks about blocks, warps, SMs, occupancy, and memory coalescing as if they were language features. They are not. They are the software-visible names of a hardware architecture NVIDIA shipped in November 2006 as the G80 (GeForce 8800 GTX) and refined in 2008 as the GT200 (GTX 280). NVIDIA calls that architecture Tesla, and the reference is the IEEE Micro paper above.

If you learn CUDA C without this layer, `warp = 32` and "coalesce your accesses" are rules to memorize. If you see the hardware first, they are consequences. This post walks the Tesla data path once, top to bottom, so that the CUDA abstractions in the CUDA C post land on something physical.

One caveat up front: Tesla (G80/GT200) is the historical anchor here, not a snapshot of a modern GPU. The SM has been rebuilt many times since 2006. FP32 lanes per SM went from 8 to 128, one warp scheduler became four, shared memory grew from 16 KB to over 200 KB, and units that did not exist then (a general-purpose L1 data cache, tensor cores, async copy) were added. What survives is the naming lineage and the mental model: SPA to TPC to SM to SP is the genealogy behind CUDA's blocks, warps, and SM scheduling. Read the numbers below as a 2008 snapshot, not today's spec sheet, and take the concepts as the part that lasts.

## The Unified Shift

A pre-Tesla GPU had a fixed pipeline of specialized processors: vertex shaders, then rasterization, then pixel (fragment) shaders, each a different unit with its own instruction set and its own silicon. The ratio between vertex and pixel work is fixed at design time, so a vertex-heavy or pixel-heavy frame leaves half the chip idle.

Tesla threw that out. It replaced the separate shader stages with one array of identical programmable processors that every shader type time-shares. Vertices, geometry, and pixels all run on the same cores; the hardware just re-points the array at whatever stage has work. NVIDIA calls that array the SPA (Streaming Processor Array).

That single decision is what created GPGPU. Once you have a general, programmable processor array with its own memory system, aiming it at non-graphics compute is a matter of exposing it, which is exactly what CUDA does. CUDA C is not bolted onto a graphics chip; it is a second front-end to the same unified array.

![NVIDIA Tesla (G80) unified architecture](./images/tesla1.svg?v=2)
*G80 shown: 8 TPC × 2 SM × 8 SP = 128 SP, 6 DRAM partitions. The SPA is the array of 8 TPCs; the compute work distribution path (magenta) is the one CUDA drives.*

## Walking the Data Path

Follow the work from memory, through the chip, and back to memory. In graphics mode the path is:

1. **Input Assembler.** Reads vertex indices and attributes from DRAM and assembles them into primitives (points, lines, triangles). This is the front door.
2. **Work distribution.** Tesla has separate distributors for vertex, pixel, and compute work. Each one hands batches of work to the SPA and load-balances them across the processors. In compute mode, the *compute work distribution* unit is the piece that hands thread blocks to SMs, one block at a time as SMs free up. That load-balancer is the physical origin of CUDA's transparent scalability.
3. **SPA.** The processor array executes the shader (or the kernel). For graphics it runs vertex shading, then later pixel shading; for CUDA it runs the kernel. It is the compute engine and everything else feeds it or drains it.
4. **Setup / raster (graphics only).** Between vertex and pixel work, fixed-function units clip, set up triangles, and rasterize them into fragments, which the pixel distributor then feeds back into the SPA.
5. **ROP (Raster Operations Processor).** After pixel shading, the ROP does the fixed-function back end: depth and stencil test, color blend, antialiasing, and the final write to the framebuffer. Each ROP is tied to a DRAM partition.
6. **DRAM.** The memory partitions, where everything starts and ends.

For CUDA the graphics-specific stages (setup, raster, ROP) sit idle, and the path collapses to: host launches a grid, the compute work distributor spreads blocks over SMs, the SPA runs the kernel, and loads/stores move data to and from DRAM. Same silicon, fewer stages lit up.

## The Compute Hierarchy: SPA to TPC to SM to SP

The SPA is not a flat pool of cores. It is a three-level hierarchy, and each level maps to a CUDA concept.

- **TPC (Texture / Processor Cluster).** The SPA is divided into TPCs. A TPC bundles a texture unit with a small number of SMs that share it. G80 has 8 TPCs of 2 SMs each; GT200 has 10 TPCs of 3 SMs each.
- **SM (Streaming Multiprocessor).** The unit that actually runs threads. Each SM holds 8 SPs, 2 SFUs, a multithreaded instruction fetch/issue unit, a register file, and 16 KB of shared memory. The SM is where a CUDA thread block lands and stays.
- **SP (Streaming Processor).** A scalar ALU that executes one thread's floating-point and integer work, primarily a MAD (multiply-add). Eight per SM. This is the unit later marketing renamed the "CUDA core."

The totals fall straight out of the hierarchy:

$$
\text{G80: } 8\ \text{TPC} \times 2\ \text{SM} \times 8\ \text{SP} = 128\ \text{SP}
\qquad
\text{GT200: } 10 \times 3 \times 8 = 240\ \text{SP}
$$

Two more execution units per SM matter for CUDA:

- **SFU (Special Function Unit).** Two per SM. It computes transcendentals (reciprocal, reciprocal-sqrt, sin, cos, log, exp) and, in graphics, interpolates pixel attributes. When a CUDA kernel calls `__sinf` or `rsqrtf`, this is the unit.
- **LSU (Load/Store Unit).** The path that issues loads and stores to global and local memory through the memory pipeline. Its behavior under a warp is the whole subject of coalescing in the CUDA C post.

## SIMT and the Birth of the Warp

Here is where the CUDA abstraction is literally defined by the hardware. The SM's instruction unit does not track threads one at a time. It creates, manages, schedules, and executes them in groups of 32 called a warp. The Tesla paper coined the term SIMT (Single-Instruction, Multiple-Thread) for this: the SM issues one instruction to a warp, and all 32 threads execute it, each on its own data and its own registers.

Why 32, and why does one warp instruction take more than one clock? Count the lanes. An SM has 8 SPs, but a warp is 32 threads, so the SM streams the warp over the 8 SPs across four fast clocks:

$$
\frac{32\ \text{threads/warp}}{8\ \text{SP}} = 4\ \text{shader clocks per warp instruction}
$$

So the physical SIMD width is 8, but the architectural width the programmer sees is 32. NVIDIA fixed the warp at 32 here and has never changed it, which is why CUDA code a generation later still assumes 32. When a warp branches (divergence), the threads that do not take the branch are masked for those clocks, which is the cost the CUDA post describes.

Threads are not free-floating either. Each SM has a fixed register file and 16 KB of shared memory, and every resident warp carves its registers and shared memory out of those pools. A G80 SM holds up to 24 warps (768 threads) at once; how many actually fit depends on how greedy each thread is. That trade, resident warps versus per-thread resources, is exactly occupancy, and it exists because these pools are finite hardware on the SM.

![SPA to TPC to SM to SP hierarchy](./images/tesla2.svg?v=1)

## Clock Domains

A Tesla GPU does not run at one frequency. It has separate clock domains, and mixing them up wrecks any performance estimate.

- The core (graphics) clock drives the front end, setup, raster, and ROP.
- The shader clock drives the SPs, and it runs much faster than the core clock. On the 8800 GTX the core is 575 MHz while the shaders run at 1.35 GHz, roughly 2.35x.
- The memory clock is separate again, driving the GDDR3 interface.

The shaders run hot on purpose: throughput comes from the SP array, so NVIDIA clocks it as high as the process allows and lets the rest of the chip stay slower and cooler. This is why a FLOP estimate uses the shader clock, not the core clock. For the 8800 GTX, counting one MAD (2 FLOP) per SP per shader clock:

$$
128\ \text{SP} \times 1.35\ \text{GHz} \times 2\ \text{FLOP} \approx 346\ \text{GFLOP/s}
$$

NVIDIA quoted a higher figure (518 GFLOP/s) by also counting a MUL that the SFU can co-issue in the same clock, which real kernels rarely sustain. That gap between the marketing peak and the achievable peak is a habit worth keeping: always ask which clock and which instruction mix a number assumes.

## The Memory Subsystem: DRAM Partitions and the Road to Coalescing

Tesla's DRAM is not one monolithic memory. It is split into independent partitions, each with its own memory controller and its own ROP. G80 has 6 partitions of 64 bits each (a 384-bit aggregate bus); GT200 has 8 (512-bit). Addresses are interleaved across partitions so that sequential memory marches across all controllers in parallel, and total bandwidth is the sum of the partitions.

For the 8800 GTX, with a 384-bit bus and GDDR3 at 900 MHz (double data rate, so 1.8 Gb/s per pin):

$$
\frac{384\ \text{bit}}{8} \times 1.8 \times 10^{9}\ \text{s}^{-1} = 48\ \text{B} \times 1.8\ \text{GT/s} \approx 86.4\ \text{GB/s}
$$

This partitioned, interleaved, wide memory is the reason coalescing exists as a rule in CUDA. When a warp's 32 lanes issue loads, the LSU turns them into memory transactions that the partitions service. If the 32 addresses are contiguous, they fall into a few wide transactions spread across the controllers and the bus runs full. If they are scattered, each address drags its own transaction and most of every fetched line is thrown away. The CUDA-level advice "make warp accesses contiguous" is just "feed the partitioned memory system the wide, aligned transactions it was built for."

## What Carried Over, What Changed

Everything above is a 2008 chip. Open an A100 (2020) or H100 (2022) SM and the numbers are unrecognizable, but the skeleton is the same. That split, stable concepts and shifting magnitudes, is the whole reason Tesla is worth reading.

| | G80 (2006) | A100 (2020) | H100 (2022) |
| --- | --- | --- | --- |
| FP32 lanes per SM | 8 | 64 | 128 |
| Warp schedulers per SM | 1 | 4 | 4 |
| Warp size | 32 | 32 | 32 |
| Shared memory per SM | 16 KB | up to 164 KB | up to 228 KB |
| 32-bit registers per SM | 8,192 | 65,536 | 65,536 |
| Added since | baseline | L1 data cache, tensor cores, ITS, async copy | + TMA, thread block clusters |

What did not move: the warp is still 32 threads, a block still runs on one SM, the SM still schedules warps and hides latency by switching between them, shared memory still lives on the SM, and global memory is still partitioned and wants coalesced access. Those are the invariants CUDA is built on. What did move: the SM got wider (more lanes, more schedulers), deeper (a general-purpose L1 data cache, a larger register file), and specialized (tensor cores, TMA). So use Tesla for the model and the vocabulary, and use the current architecture whitepaper for any number you plan to optimize against.

## Tesla Hardware to CUDA Software

Every abstraction in the CUDA C post is a name for something on this chip:

| Tesla hardware | CUDA C concept |
| --- | --- |
| Compute work distributor | how a grid's blocks spread over SMs (transparent scalability) |
| SM | where one thread block runs, start to finish |
| Warp (32 threads, 4 clocks over 8 SP) | the SIMT execution and scheduling unit |
| SP (scalar MAD ALU) | the "CUDA core" |
| SFU | `__sinf`, `rsqrtf`, and other intrinsics |
| Register file + 16 KB shared memory per SM | the resources occupancy trades against |
| Shared memory | `__shared__` |
| LSU + interleaved DRAM partitions | why coalescing and bandwidth matter |

Read the CUDA C post after this and the abstractions stop being arbitrary. `warp = 32` is the SM streaming 32 threads over 8 SPs. Occupancy is the SM's finite register file and shared memory. Coalescing is the DRAM partitions wanting wide transactions. CUDA did not invent these; it exposed them.

## References

- Lindholm, Nickolls, Oberman, Montrym, [*"NVIDIA Tesla: A Unified Graphics and Computing Architecture"*](https://ieeexplore.ieee.org/document/4523358): the primary source for the SPA/TPC/SM/SP hierarchy, SIMT, and the graphics data path.
- [NVIDIA GeForce 8800 GPU Architecture Technical Brief](https://www.nvidia.com/): G80 clocks, partitions, and register/shared-memory sizes.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): how the software model maps onto this hardware, on current architectures.
- [NVIDIA Hopper (H100) Architecture](https://www.nvidia.com/en-us/data-center/h100/): the modern SM (128 FP32 lanes, 4th-gen tensor cores, TMA) that the Tesla lineage grew into.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, occupancy, pinned memory, and host-device transfer applied to today's hardware.
