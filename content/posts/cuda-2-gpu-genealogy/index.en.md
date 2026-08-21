---
title: "01 NVIDIA GPU Architecture Genealogy: Tesla to Rubin"
date: 2026-05-22
draft: false
tags: ["CUDA", "GPU Architecture", "Tensor Core", "NVIDIA", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "One line from Tesla (2006) to Rubin (2026): how the NVIDIA SM stayed the same (SIMT, warp = 32, block per SM) while accreting specialized accelerators, how the Tensor Core evolved across five generations, why the consumer and datacenter lines split, and why 'Rubin' is a platform, not just a GPU."
---

> The primary sources for this post are NVIDIA's architecture whitepapers and official product pages. Narrative and microarchitecture interpretation draw on Fabien Sanglard, Chips and Cheese, and SemiAnalysis. Full sources are listed in the references at the end.

## Overview: How to Read the Genealogy

From Tesla in 2006 to Rubin in 2026, NVIDIA has shipped over a dozen GPU architectures in twenty years. Listed by name, that looks like a lot to memorize. In practice, the whole lineage fits one simple frame, and the goal of this post is to build it: take any NVIDIA GPU, place it on a timeline, and state in one sentence what it changed and why.

This post uses the 2006 Tesla chip — covered in the [primer post (CUDA 0)](../cuda-0-gpu-architecture/) — as its anchor, and follows that anchor forward.

![NVIDIA GPU architecture family tree, Tesla to Rubin](./images/timeline.svg?v=1)
*The family tree: a shared trunk through Pascal, then a fork at Volta into a datacenter line (top) and a graphics line (bottom).*

The frame is this: **there is a foundation that almost never changes, and every other change is a response to one of two pressures.**

The unchanging foundation is the execution model and the memory hierarchy. The SIMT (Single Instruction, Multiple Threads) model, in which one instruction is executed simultaneously by a bundle of 32 threads (a warp); the thread block that runs to completion on a single SM (Streaming Multiprocessor — the basic building block of a GPU, bundling execution units, schedulers, and shared memory); and the memory hierarchy of registers → shared memory → global DRAM. Learn this once in the [CUDA C post](../cuda-c-basics/) and it applies unchanged from G80 to Rubin. It is why CUDA code written twenty years ago still compiles for today's GPUs.

The two pressures driving change are these. First, **the workload moved.** As the GPU's main customer shifted from graphics to AI, the SM kept its general-purpose cores and kept bolting specialized units around them: Tensor Cores first, then RT Cores, then the Transformer Engine. Second, **the scale pressure.** When one die stopped being enough, the unit of design grew from a chip to two dies to an entire rack. In short, the pattern of the genealogy is: the general-purpose SM stays, accelerators accrete around it, and the package keeps growing.

This post follows the SM for a simple reason: the SM is the unit onto which CUDA programs are scheduled. The rest of the GPU (L2 cache, memory controllers, ROPs, copy engines, host interface, fabric) matters for performance and system design, but the changes a programmer actually feels — warp execution, registers, shared memory, Tensor Cores, RT Cores, TMA/TMEM — show up at the SM.

![Anatomy of a GPU die, where the SM sits](./images/gpu-anatomy.svg?v=1)
*A GPU die is an array of SMs wrapped by L2, memory controllers, DRAM, graphics fixed-function, and host/fabric interfaces. Every per-generation diagram below zooms into one SM.*

## The Era of Unification: Tesla and Fermi

**Tesla (2006, G80)** is the starting point of the lineage. GPUs before it had fixed pipelines with separate hardware for vertex and pixel processing. Tesla replaced both with a single unified array of programmable cores. That one decision turned the GPU from a graphics-only device into a general-purpose compute engine, and made the CUDA programming model possible. The configuration looks modest by today's standards: 8 scalar processors (SPs) per SM, one warp scheduler, 90nm process.

![Tesla SM component diagram](./images/sm-tesla.svg?v=1)
*Tesla SM (G80): 8 scalar SPs, 1 scheduler, 16 KB shared memory. The origin of everything.*

**Fermi (2010, GF100)** is the generation that deliberately turned a graphics chip into a compute chip. Going from "a GPU can compute" to "a GPU is a serious programming target" required specific things, and Fermi added them: a real L1 data cache and an L2, ECC memory, fused multiply-add (FMA), fully IEEE-compliant double precision (FP64), and C++ support. The SM itself grew to 32 CUDA cores with 2 warp schedulers, and the texture units moved inside the SM. If Tesla proved GPU compute was possible, Fermi made it something you could build numerical libraries on.

![Fermi SM component diagram](./images/sm-fermi.svg?v=1)
*Fermi SM (GF100): 32 cores, 2 schedulers, the first L1 data cache on a GPU.*

## The Era of Efficiency: Kepler, Maxwell, Pascal

**Kepler (2012, GK110)** was a bet on throughput. It widened the SM dramatically — renamed SMX, with 192 CUDA cores — while moving much of instruction scheduling from hardware to the compiler to save power. The bet was that many cores plus a simple scheduler, run at lower clocks, would win on performance per watt. It half worked. Kepler was efficient in aggregate but hard to keep fed, and per-core utilization suffered. It remains the generation people point to when they say "a wider SM is not automatically a faster SM."

![Kepler SMX component diagram](./images/sm-kepler.svg?v=3)
*Kepler SMX (GK110): 192 cores, 4 schedulers, compiler-driven scheduling.*

**Maxwell (2014, GM200)** corrected Kepler's overreach. It narrowed the SM back to 128 cores and split it into 4 processing blocks of 32 cores each, each block with its own scheduler and register file. Since 32 is exactly the warp size, this partitioning mapped the hardware cleanly back onto warps, 1:1. With no new process node — just a cleaner design — Maxwell delivered one of NVIDIA's biggest efficiency jumps ever. It is the standard example that tidy architecture can beat brute width. The "SM = 4 warp-sized partitions" structure that settled here carries forward through every generation since.

**Pascal (2016)** is where the fork becomes visible. The consumer part (GP102, GTX 1080 Ti) was essentially Maxwell moved to 16nm with GDDR5X attached — a process-and-bandwidth generation. The datacenter part (GP100, P100) was a different machine: only 64 FP32 lanes per SM, but serious FP64 hardware, plus the first appearance of NVLink (high-speed GPU-to-GPU interconnect) and HBM2 (high-bandwidth stacked memory). Pascal is the point where consumer and datacenter stopped being "the same chip, different bin" (binning: grading the same die by yield and performance and selling the grades as different products).

![Maxwell and Pascal SM component diagram](./images/sm-maxwell-pascal.svg?v=2)
*Maxwell and Pascal: the SM split into 4 warp-sized partitions, a structure that persists to this day.*

## The Pivot to AI: Volta and Turing

**Volta (2017, GV100)** is the hinge of the entire genealogy. This is where the first Tensor Core appears: a dedicated unit that performs a small matrix multiply-accumulate (MMA, hereafter) as a single instruction. The reason it was needed: when you do matrix multiplication with ordinary FP instructions, most of the energy goes not into arithmetic but into instruction fetch/decode/schedule overhead. Batch the work into matrix-sized instructions and that overhead disappears.

Volta's second legacy is independent thread scheduling. From this generation on, each thread in a warp has its own program counter. The lockstep assumption — that every thread in a warp executes the same instruction on the same beat — broke here — which is exactly why the CUDA C post has to qualify warp lockstep and introduce `__syncwarp()`. Volta shipped as datacenter-only, with no consumer part. Everything in modern AI hardware starts here.

![Volta SM component diagram](./images/sm-volta.svg?v=1)
*Volta SM (GV100): the first Tensor Core joins the CUDA cores.*

**Turing (2018, TU102)** brought Volta's ideas to the graphics line. It put a 2nd-generation Tensor Core and a brand-new RT Core (dedicated ray-tracing hardware) into consumer GPUs, and split the datapath so the SM could issue FP32 and INT32 instructions concurrently — a real win, since address arithmetic and other integer work is constantly interleaved with FP math in practice. This is the moment the graphics line stopped being purely graphics and started carrying AI and ray-tracing accelerators, and it is what makes DLSS (rendering at low resolution and upscaling frames with a neural network to buy performance) possible.

![Turing and Ada SM component diagram](./images/sm-turing-ada.svg?v=1)
*Turing and Ada: RT Cores and graphics-facing Tensor Cores enter the SM.*

## The Datacenter Arms Race: Ampere and Hopper

**Ampere (2020, GA100)** is about scale and formats. The 3rd-generation Tensor Core added TF32 (FP32's exponent range with a shortened mantissa — a drop-in for training code) and BF16, and claimed 2× throughput with structured sparsity (zeroing half the weights in a fixed pattern, then skipping those zeros at compute time). Just as important is the `cp.async` instruction: previously, moving data from global to shared memory had to route through registers; `cp.async` performs the copy without touching them, relieving the register pressure that chronically limits Tensor Core kernels. Ampere also introduced MIG (Multi-Instance GPU), which partitions one A100 into fully isolated GPU instances. Note that within the same Ampere name, the datacenter A100 has 64 FP32 lanes per SM while consumer RTX 30 parts have 128 — the SMs genuinely differ. The [two lines](#two-lines-consumer-and-datacenter) section below returns to this.

![Ampere SM component diagram](./images/sm-ampere.svg?v=1)
*Ampere SM (GA100): 3rd-gen Tensor Cores, and cp.async feeding shared memory directly.*

**Hopper (2022, GH100)** is the Transformer Engine generation. The Transformer Engine is a hardware-plus-software mechanism that automatically picks the right precision (FP8 vs FP16) per layer — taking low precision's speed while guarding against accuracy collapse. The 4th-generation Tensor Core added FP8 (E4M3, E5M2), and Hopper wrapped it in machinery aimed squarely at LLMs: asynchronous matrix instructions issued at warpgroup granularity (`wgmma`, where a warpgroup is 4 warps), TMA (Tensor Memory Accelerator — a bulk asynchronous copy engine that a single thread kicks off and hardware completes), and thread block clusters with distributed shared memory, letting SMs exchange data directly. In SemiAnalysis's framing, the motivating problem is that "Tensor Core throughput doubles every generation, but global memory latency does not improve." So Hopper spent its budget not on raw FLOPs but on hiding latency and feeding the units. The flagship is H100, with HBM3 and 900 GB/s NVLink 4.

![Hopper SM component diagram](./images/sm-hopper.svg?v=2)
*Hopper SM (GH100): FP8 Tensor Cores, TMA, wgmma, thread block clusters.*

## The Era of Scale: Ada, Blackwell, Rubin

**Ada (2022, AD102)** is the graphics-line counterpart to Hopper, launched the same year. It carries 4th-generation Tensor Cores and 3rd-generation RT Cores, adds Shader Execution Reordering (SER — hardware that regroups divergent ray-tracing threads to recover efficiency), and the DLSS 3 frame-generation stack. The flagship is the RTX 4090 on TSMC 4nm.

**Blackwell (2024)** can be summarized as "two chips, one name." The datacenter part (B200) is where the GPU stopped being a single die: two dies, each grown to the reticle limit (the largest area a lithography machine can expose at once), are fused with a 10 TB/s link and presented to software as one GPU — 208 billion transistors combined, with HBM3e. The consumer part (GB202 family, RTX 5090) takes the opposite approach: a single die near 750mm² with GDDR7. The full GB202 die is laid out for 192 SMs; the RTX 5090 ships with 170 of them enabled (21,760 CUDA cores). Chips and Cheese reads this design as "scale over specialization": a 64-bank L2 (~8.7 TB/s) that chooses bandwidth over latency, winning through sheer core density rather than per-core cleverness.

Both parts share the 5th-generation Tensor Core. Its additions: FP4 (the NVFP4 and microscaling MXFP formats), a dedicated Tensor Memory (TMEM) that holds matrix operands outside the register file, and CTA-pair MMA, where two SMs cooperate on one matrix operation (CTA, cooperative thread array, is the hardware-side name for a thread block). At the system level, GB200 pairs two datacenter Blackwell GPUs with a Grace CPU, and GB200 NVL72 links 72 such GPUs into an NVLink domain that behaves like a single rack-scale GPU.

![Blackwell SM component diagram](./images/sm-blackwell.svg?v=2)
*Blackwell SM (B200): FP4 Tensor Cores, dedicated TMEM, CTA-pair MMA.*

**Rubin (2026)** is the current generation, and the point where the shift of the design unit from chip to rack completes. NVIDIA's public material presents the Rubin GPU at 336 billion transistors, 288GB of HBM4 at 22 TB/s, and NVLink 6 at 3.6 TB/s. But NVIDIA marks these public specs as preliminary, and does not describe the product as simply "the Rubin GPU." It describes a platform: Vera Rubin. Why that distinction matters is covered [below](#rubin-a-rack-scale-platform).

## The Evolution of the Tensor Core

Among all the threads in this genealogy, the one that determined where transistors and R&D actually went is the Tensor Core. It is a matrix multiply-accumulate unit, and its five generations from Volta to Blackwell move along two axes: **precision** and **asynchrony**.

Precision dropped every generation: FP16 (Volta) → INT8/INT4 (Turing) → TF32 and BF16 (Ampere) → FP8 (Hopper) → FP4 (Blackwell). The move is possible because AI workloads tolerate low precision, and every halving of precision doubles the arithmetic per transistor and per byte moved. The Transformer Engine introduced in the Hopper section is exactly the safety mechanism for this move.

The easily missed half of the story is that Tensor Cores grew **by tile size, not by count**. A matrix multiply performs roughly $N^3$ operations while moving roughly $N^2$ data, so arithmetic intensity — operations per byte moved — rises with the tile's edge length:

$$I \sim \frac{N^3}{N^2} = N$$

Bigger tiles amortize data movement better. So instead of stamping out more small units, NVIDIA made each instruction compute a bigger matrix every generation (4×4×4, then 8×8×4, then 16×8×16 and beyond). This is why the Tensor Core count per SM actually fell from 8 in Volta to 4 from Ampere onward — each unit got much larger, and throughput still doubled per generation.

The execution model evolved for the same reason. Tensor throughput keeps doubling while memory latency does not improve, so the ability to overlap compute with data movement becomes the whole game: synchronous warp-level MMA (Volta) → asynchronous warpgroup MMA (Hopper's `wgmma`) → fully asynchronous single-thread MMA with operands resident in dedicated Tensor Memory (Blackwell). The thesis that runs the entire length of the lineage: **the bottleneck is not the math — it is feeding the math.**

## Two Lines: Consumer and Datacenter

From Volta onward, the family runs as two branches that share DNA but optimize for different things.

The **datacenter line** (GV100 → GA100 → GH100 → B200 → Rubin) maximizes AI throughput and interconnect: fewer FP32 lanes per SM but more INT32/FP64/Tensor hardware, HBM instead of GDDR, NVLink growing into full rack-scale fabric (the network that ties chips and nodes into one domain), and datacenter-only features like MIG and thread block clusters.

The **graphics line** (TU102 → GA102 → AD102 → GB202) keeps enough Tensor Cores for DLSS, adds RT Cores and rendering features, and uses GDDR memory.

One caveat worth internalizing: the names Ampere and Blackwell exist on both branches. "An Ampere GPU" can mean an A100 or an RTX 3090, and their SMs differ substantially (64 vs 128 FP32 lanes). The generation name alone is not enough — you have to say which line.

## The Family Tree

| Generation | Year | SM / codename | Defining change | Process | Flagship |
| --- | --- | --- | --- | --- | --- |
| Tesla | 2006 | SM, 8 SP (G80) | unified shaders, SIMT, CUDA | 90 nm | 8800 GTX |
| Fermi | 2010 | SM, 32 (GF100) | L1 data cache, FMA, FP64, C++ | 40 nm | GTX 480 |
| Kepler | 2012 | SMX, 192 (GK110) | compiler scheduling, wide SM | 28 nm | K20 |
| Maxwell | 2014 | SMM, 128 (GM200) | efficiency, 4x32 partitions | 28 nm | GTX 980 Ti |
| Pascal | 2016 | GP100 / GP102 | NVLink, HBM2 (GP100), 16nm | 16 nm | P100 |
| Volta | 2017 | GV100, 64 FP32 | 1st Tensor Core, independent thread scheduling | 12 nm | V100 |
| Turing | 2018 | TU102, 64 FP32 | RT Core + 2nd Tensor to graphics | 12 nm | RTX 2080 Ti |
| Ampere | 2020 | GA100, 64 FP32 | 3rd Tensor (TF32/sparsity), MIG | 7 nm | A100 |
| Ada | 2022 | AD102, 128 FP32 | 4th Tensor, 3rd RT, SER | 4 nm | RTX 4090 |
| Hopper | 2022 | GH100, 128 FP32 | Transformer Engine (FP8), TMA, clusters | 4 nm | H100 |
| Blackwell | 2024 | 2 dies, 208B | FP4, TMEM, 5th NVLink, scale-first | TSMC 4NP | B200 / GB200 |
| Rubin | 2026 | 2 dies, 336B | HBM4, NVLink 6; Vera Rubin = rack platform | preliminary per NVIDIA | Rubin / Vera Rubin NVL72 |

## Per-Generation SM Specifications

If the family tree above is the one-line summary of each generation, the tables below show how the SM is actually organized inside, using NVIDIA whitepaper figures. **Bold values mark what changed from the previous generation of the same line.** Scan a column vertically and you can see exactly when each property changed.

First, the shared trunk and the datacenter line:

| Chip (year) | Partitions | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | Schedulers/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G80 (2006) | monolithic | 8 | shared w/ FP32 | — | — | 1 | 16 KB (shared only, no L1) | 32 KB |
| GF100 (2010) | monolithic | **32** | 〃 | **16 FMA/clk** | — | **2** | **64 KB (shared/L1 combined, 48+16 split)** | **128 KB** |
| GK110 (2012) | monolithic | **192** | 〃 | **64** | — | **4** | 64 KB combined **+ 48 KB read-only** | **256 KB** |
| GM200 (2014) | **4 × 32 (first split)** | **128** | 〃 | **4** | — | 4 | **96 KB (dedicated shared, separate L1)** | 256 KB |
| GP100 (2016) | **2 × 32** | **64** | 〃 | **32** | — | **2** | **64 KB (dedicated shared)** | 256 KB |
| GV100 (2017) | **4 × 16** | 64 | **64 (separate datapath)** | 32 | **8 (1st gen, FP16)** | **4** | **128 KB (unified shared+L1)** | 256 KB |
| GA100 (2020) | 4 × 16 | 64 | 64 | 32 | **4 (3rd gen, TF32/BF16, larger units)** | 4 | **192 KB unified (shared up to 164 KB)** | 256 KB |
| GH100 (2022) | **4 × 32** | **128** | 64 | **64** | **4 (4th gen, FP8, wgmma)** | 4 | **256 KB unified (shared up to 228 KB)** | 256 KB |
| B200 (2024) | 4 × 32 | 128 | **128** | 64 | **4 (5th gen, FP4, tcgen05)** | 4 | 256 KB | 256 KB **+ TMEM 256 KB** |

Then the graphics line (forking at Turing):

| Chip (year) | Partitions | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT Core/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TU102 (2018) | 4 × 16 | 64 | 64 (concurrent issue) | 2 | 8 (2nd gen, INT8/4) | 1 (1st gen) | 96 KB unified | 256 KB |
| GA102 (2020) | **4 × 32** | **128 (64 dedicated + 64 shared w/ INT)** | 64 | 2 | **4 (3rd gen)** | 1 **(2nd gen)** | **128 KB unified** | 256 KB |
| AD102 (2022) | 4 × 32 | 128 | 64 | 2 | 4 **(4th gen, FP8)** | 1 **(3rd gen)** | 128 KB unified | 256 KB |
| GB202 (2024) | 4 × 32 | **128 (all cores unified FP32/INT32)** | **128** | 2 | 4 **(5th gen, FP4)** | 1 **(4th gen)** | 128 KB unified | 256 KB |

A few trends the tables make visible. The FP32 count wandered — 8 → 32 → 192 → 128 → 64 → 128 — but the register file has not moved from 256 KB since Kepler, and the partition structure has held the "warp-sized multiple × 4" shape since Maxwell. Shared memory, meanwhile, grew steadily from 16 KB to 228 KB, and Blackwell added an entirely new storage pool in TMEM. The thesis of this post — compute stays, investment goes to data supply — shows up directly in the numbers.

A few footnotes. GF100's FP64 is listed as per-clock FMA throughput (16 DFMA/clk) because that is how NVIDIA disclosed it, rather than as a unit count. The drop from 8 to 4 Tensor Cores at GA100 is not a regression — each unit's tile got larger (see the Tensor Core section). At the time of writing, no SM-level whitepaper for datacenter Blackwell has been published, so the B200 row uses figures from NVIDIA's technical blog and public educational material. Rubin is omitted because its SM details are not public.

## Rubin: A Rack-Scale Platform

The newest entry in the genealogy is easy to misread. "Blackwell" is still the name of a GPU you can point at. "Rubin" is mostly the name of a system. NVIDIA's own description of Vera Rubin is a rack — the NVL72: 72 Rubin GPUs and 36 Vera CPUs in a single liquid-cooled NVLink-6 domain, delivering roughly 3.6 EFLOPS of FP4 inference and 20.7 TB of HBM4. The Vera CPU is its own chip, with 88 custom Olympus Arm cores. NVIDIA's current Vera Rubin page describes the platform as a seven-chip platform spanning compute, networking, storage, and switching. This is why trying to summarize Rubin with a single SM diagram is already the wrong level of abstraction.

![Rubin and Vera Rubin platform diagram](./images/rubin-platform.svg?v=2)
*Rubin is a platform: GPU, Vera CPU, NVLink switches, DPU, and Ethernet in one rack.*

So keep the two names separate. The **Rubin GPU** is a microarchitecture; compare it to GB100. **Vera Rubin** is a co-designed rack-scale computer; compare it to GB200 NVL72. The endpoint of the genealogy is not a faster chip — it is the admission that the interesting unit is now the rack.

![NVIDIA architecture snapshots, Tesla to Rubin](./images/architecture-snapshots.svg?v=2)
*Architecture snapshots: the shared trunk moves from graphics to compute; the upper branch is datacenter AI, the lower branch RTX graphics.*

## Synthesis: Three Trajectories

Three trajectories run the full length of the lineage. First, **specialization increases** — the move away from doing every computation on one general-purpose core, toward dedicated hardware for specific operations. The SM keeps its general-purpose cores while stacking Tensor Cores, RT Cores, the Transformer Engine, and dedicated Tensor Memory around them. Second, **precision decreases** — from FP32 to FP4 — because AI can pay for throughput in bits. Third, **the unit of design grows**: a chip, then two dies, then a rack.

The one thing that never changes is where the bottleneck sits. From memory coalescing in the CUDA C post to Hopper's TMA and Blackwell's TMEM, every generation spends most of its new hardware budget not on raw FLOPs but on moving data and hiding latency. Compute has been cheap for a decade; feeding it has not. The single thread running through the whole family tree is the memory wall.

## References

- [Fabien Sanglard, A history of NVidia Stream Multiprocessor](https://fabiensanglard.net/cuda/): the narrative and SM design changes from Tesla through Turing.
- [SemiAnalysis, NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell): the precision, asynchrony, and tile-size argument.
- [Chips and Cheese, Blackwell: NVIDIA's Massive GPU](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu): the scale-over-specialization microarchitecture reading.
- NVIDIA primary architecture documents: [Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf), [Kepler GK110](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf), [Maxwell tuning](https://docs.nvidia.com/cuda/maxwell-tuning-guide/), [Pascal GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [Volta GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [Ampere A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [Ampere GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [H100 whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf), [GeForce RTX Blackwell](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) and [Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/): primary figures for the latest generations.
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) and the [NVIDIA Rubin platform technical blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/): Rubin GPU, NVLink 6, NVL72, and the preliminary-spec caveat.
- [Cornell Virtual Workshop, B200 SM](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm): B200 SM configuration figures.
