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

> Primary: NVIDIA architecture whitepapers and product pages. Narrative and microarchitecture: Fabien Sanglard, Chips and Cheese, SemiAnalysis.

## Overview: How to Read the Genealogy

The [primer post](../cuda-0-gpu-architecture/) used the 2006 Tesla chip as an anchor for CUDA's vocabulary. This post follows that anchor forward. The goal is narrow: place any NVIDIA GPU on one timeline and state in a sentence what it changed and why.

![NVIDIA GPU architecture family tree, Tesla to Rubin](./images/timeline.svg?v=1)
*The family tree: a shared trunk through Pascal, then a fork at Volta into a datacenter line (top) and a graphics line (bottom).*

The whole line rewards one framing. Two things almost never change, and everything else is a response to two pressures.

What stays: the SIMT execution model, the warp of 32 threads, a thread block that runs to completion on one SM, and the memory hierarchy of registers, shared memory, and global DRAM. Learn those once, in the [CUDA C post](../cuda-c-basics/), and they hold from G80 to Rubin.

What changes is driven by two pressures. First, the workload moved from graphics to AI, so the SM kept bolting on specialized units (Tensor Cores, then RT Cores, then a Transformer Engine) around the same general-purpose core. Second, a single die stopped being enough, so the unit of design grew from a chip to two dies to a rack. The pattern is a general SM that stays put while accelerators pile up around it and the package keeps getting bigger.

This genealogy mostly follows the SM, because CUDA programs are scheduled onto SMs. The rest of the GPU (L2, memory controllers, ROPs, copy engines, the host interface, and the fabric) matters for performance and system design, but the SM is the unit where warp execution, registers, shared memory, Tensor Cores, RT Cores, and TMA/TMEM changes show up.

![Anatomy of a GPU die, where the SM sits](./images/gpu-anatomy.svg?v=1)
*A GPU die is an array of SMs wrapped in L2, memory controllers, DRAM, a graphics-only fixed-function path, and host or fabric interfaces. Every per-generation diagram below zooms into one SM.*

## The Unified Era: Tesla and Fermi

**Tesla (2006, G80).** The starting point, covered in the primer. Its one idea, replacing fixed vertex and pixel pipelines with a single unified array of programmable cores, is what made a GPU a general compute device and made CUDA possible. 8 SP per SM, one warp scheduler, SIMT, 90 nm.

![Tesla SM component diagram](./images/sm-tesla.svg?v=1)
*Tesla SM (G80): 8 scalar SP, one scheduler, 16 KB shared. The origin.*

**Fermi (2010, GF100).** The generation that turned a graphics chip into a compute chip on purpose. Fermi added the things a real programming target needs: a proper L1 data cache and L2, ECC, a fused multiply-add, full IEEE double precision, and C++ support. The SM grew to 32 CUDA cores with two warp schedulers, and texture units moved inside the SM. If Tesla proved the GPU could compute, Fermi made it something you would actually build a numerical library on.

![Fermi SM component diagram](./images/sm-fermi.svg?v=1)
*Fermi SM (GF100): 32 cores, two schedulers, the first L1 data cache.*

## The Efficiency Era: Kepler, Maxwell, Pascal

**Kepler (2012, GK110).** The throughput bet. Kepler widened the SM enormously into the SMX, 192 CUDA cores, and moved instruction scheduling out of hardware and into the compiler to save power. The bet was that raw core count and a simpler scheduler would win perf-per-watt at lower clocks. It half worked: Kepler was efficient in aggregate but hard to keep fed, and per-core utilization suffered. It is the generation people point to when they say a wider SM is not automatically a faster one.

![Kepler SMX component diagram](./images/sm-kepler.svg?v=3)
*Kepler SMX (GK110): 192 cores, four schedulers, compiler scheduling.*

**Maxwell (2014, GM200).** The correction. Maxwell narrowed the SM back to 128 cores and partitioned it into four processing blocks of 32, each with its own scheduler and register file, so the hardware maps cleanly onto warps again. No new process node, just a cleaner design, and it delivered one of NVIDIA's largest efficiency jumps. Maxwell is the case study for why architectural tidiness can beat brute width.

**Pascal (2016).** The split becomes visible. The consumer parts (GP102, GTX 1080 Ti) were essentially Maxwell moved to 16 nm with faster GDDR5X, a process-and-bandwidth generation. The datacenter part (GP100, P100) was a different animal: 64 FP32 lanes per SM but with FP64 and, for the first time, NVLink and HBM2. Pascal is where the consumer and datacenter designs stopped being the same chip with a different bin.

![Maxwell and Pascal SM component diagram](./images/sm-maxwell-pascal.svg?v=2)
*Maxwell and Pascal: the SM cut into four warp-sized partitions.*

## The AI Pivot: Volta and Turing

**Volta (2017, GV100).** The hinge of the whole genealogy. Volta added the first Tensor Core, a unit that does a small matrix multiply and accumulate in one instruction, because doing matmul with ordinary FP instructions wastes most of its power on instruction overhead rather than arithmetic. It also introduced independent thread scheduling: every thread got its own program counter, which is why the CUDA C post has to caveat warp lockstep and reach for `__syncwarp()`. Volta had no consumer part; it was datacenter only. Everything about modern AI hardware starts here.

![Volta SM component diagram](./images/sm-volta.svg?v=1)
*Volta SM (GV100): the first Tensor Core joins the CUDA cores.*

**Turing (2018, TU102).** Volta's ideas reach the graphics line. Turing put a second-generation Tensor Core and a new RT Core for ray tracing into a consumer GPU, and split the datapath so an SM could issue FP32 and INT32 at once. It is the moment the graphics line stopped being purely graphics and started carrying AI and ray-tracing accelerators, which is how you get DLSS.

![Turing and Ada SM component diagram](./images/sm-turing-ada.svg?v=1)
*Turing and Ada: an RT Core and a graphics Tensor Core enter the SM.*

## The Datacenter Arms Race: Ampere and Hopper

**Ampere (2020, GA100).** Scale plus format. The third-generation Tensor Core added TF32 (FP32 range, reduced mantissa, a drop-in for training) and BF16, plus structured sparsity for a 2x throughput claim. Just as important, `cp.async` let a thread copy global to shared memory without staging through registers, relieving the register pressure that limits Tensor Core kernels. Ampere also brought MIG, slicing an A100 into isolated instances. The A100 ran 64 FP32 lanes per SM; the consumer RTX 30 parts ran 128.

![Ampere SM component diagram](./images/sm-ampere.svg?v=1)
*Ampere SM (GA100): 3rd-gen Tensor and cp.async feeding shared memory.*

**Hopper (2022, GH100).** The Transformer Engine generation. The fourth-generation Tensor Core added FP8 (E4M3 and E5M2), and Hopper wrapped it in machinery aimed squarely at large language models: warpgroup-level asynchronous MMA (`wgmma`), the Tensor Memory Accelerator (TMA) for bulk async copies driven by a single thread, and thread block clusters with distributed shared memory so SMs can share data directly. The motivating problem, in SemiAnalysis's framing, is that Tensor Core throughput doubled every generation while global memory latency did not, so Hopper spent its budget on hiding and feeding, not on more raw FLOPs. H100, HBM3, NVLink 4 at 900 GB/s.

![Hopper SM component diagram](./images/sm-hopper.svg?v=2)
*Hopper SM (GH100): FP8 Tensor, TMA, wgmma, thread block clusters.*

## The Scale Era: Ada, Blackwell, Rubin

**Ada (2022, AD102).** The graphics-line counterpart to Hopper. Fourth-generation Tensor Cores, third-generation RT Cores, Shader Execution Reordering to claw back ray-tracing divergence, and the DLSS 3 frame-generation stack. RTX 4090, TSMC 4 nm.

**Blackwell (2024).** Two chips, one name. The datacenter part (B200) is where the die stopped being one die: two reticle-limited dies joined by a 10 TB/s link and presented to software as a single GPU, 208 billion transistors together, on HBM3e. The consumer part (GB202, RTX 5090) is instead a single die near 750 mm2 with 192 SMs on GDDR7, and it is the one Chips and Cheese reads as scale over specialization, a 64-bank L2 at roughly 8.7 TB/s chosen for bandwidth over latency, winning by sheer core density rather than per-core cleverness. Both share the fifth-generation Tensor Core, which added FP4 (NVFP4 and the microscaling MXFP formats), a dedicated Tensor Memory (TMEM) so operands live outside the register file, and CTA-pair MMA where two SMs cooperate on one matrix op. GB200 pairs two datacenter Blackwell GPUs with a Grace CPU; GB200 NVL72 wires 72 of them into one NVLink domain that behaves as a single rack-scale GPU.

![Blackwell SM component diagram](./images/sm-blackwell.svg?v=2)
*Blackwell SM (B200): FP4 Tensor, dedicated TMEM, CTA-pair MMA.*

**Rubin (2026).** The current generation, and the point where the unit of design finishes moving from chip to rack. NVIDIA's Rubin materials list a 336 billion transistor Rubin GPU with 288 GB of HBM4 at 22 TB/s and NVLink 6 at 3.6 TB/s. NVIDIA marks the public specifications as preliminary, and it does not frame the product as just "a Rubin GPU." It frames Vera Rubin as a platform, which is the distinction the next section makes.

## Tensor Core Evolution

Of all the threads through the genealogy, the Tensor Core is the one that decided where the transistors and the R&D went. It is a matrix-multiply-accumulate unit, and its five generations move along two axes: precision and asynchrony.

Precision fell every generation because AI tolerates it: FP16 (Volta), INT8/INT4 (Turing), TF32 and BF16 (Ampere), FP8 (Hopper), FP4 (Blackwell). Lower precision means more math per transistor and per byte moved, and the Transformer Engine exists to switch precision per layer automatically so accuracy survives.

The subtler half is that the Tensor Core grew by tile size, not by count. A matrix multiply does on the order of $N^3$ arithmetic operations while moving on the order of $N^2$ data, so its arithmetic intensity rises with the tile edge:

$$I \sim \frac{N^3}{N^2} = N$$

Bigger tiles amortize data movement better, so each generation made the instruction compute a larger matrix (4x4x4, then 8x8x4, then 16x8x16 and beyond) rather than stamping down more small units. And because Tensor throughput kept doubling while memory latency did not, the execution model went from synchronous warp-level MMA (Volta) to asynchronous warpgroup MMA (Hopper's `wgmma`) to fully asynchronous single-thread MMA with operands in dedicated Tensor Memory (Blackwell). Across the whole arc, the bottleneck is feeding the compute, not the compute itself.

## Two Lines: Consumer and Datacenter

From Volta onward the family runs on two branches that share DNA but optimize for different things.

The datacenter line (GV100, GA100, GH100, GB200, Rubin) maximizes AI throughput and interconnect: fewer FP32 lanes per SM but more INT/FP64/Tensor, HBM instead of GDDR, NVLink and now whole-rack fabrics, features like MIG and thread block clusters. The graphics line (Turing, Ada, the consumer Ampere and Blackwell parts) keeps enough Tensor Cores for DLSS and adds RT Cores and rendering features, on GDDR. Ampere and Blackwell exist on both branches under one name, which is why "an Ampere GPU" can mean an A100 or an RTX 3090 with quite different SMs. Naming a generation is not enough; you have to say which line.

## Rubin as a Rack-Scale Platform

The newest entry is easy to misread. "Blackwell" still names a GPU you can point at. "Rubin" mostly names a system. NVIDIA's own description of Vera Rubin is a rack, the NVL72: 72 Rubin GPUs and 36 Vera CPUs in one liquid-cooled, NVLink-6 domain, delivering on the order of 3.6 EFLOPS of FP4 inference with 20.7 TB of HBM4. The Vera CPU is its own chip, 88 custom Olympus Arm cores. NVIDIA's latest Vera Rubin page frames the platform as seven chips across compute, networking, storage, and switching, so treating Rubin as a lone SM diagram is already the wrong abstraction.

![Rubin and Vera Rubin platform diagram](./images/rubin-platform.svg?v=2)
*Rubin is a platform: GPU, Vera CPU, NVLink switch, DPU, and Ethernet in one rack.*

So keep two names apart. The Rubin GPU is a microarchitecture, comparable to GB100. Vera Rubin is a co-designed rack-scale computer, comparable to GB200 NVL72. The genealogy's endpoint is not a faster chip; it is the admission that the interesting unit is now the rack.

![NVIDIA architecture snapshots, Tesla to Rubin](./images/architecture-snapshots.svg?v=2)
*Architecture snapshots: the trunk is graphics-to-compute; the upper branch is datacenter AI; the lower branch is RTX graphics.*

## The Family Tree

| Generation | Year | SM / codename | Defining change | Node | Flagship |
| --- | --- | --- | --- | --- | --- |
| Tesla | 2006 | SM, 8 SP (G80) | unified shader, SIMT, CUDA | 90 nm | 8800 GTX |
| Fermi | 2010 | SM, 32 (GF100) | L1 data cache, FMA, FP64, C++ | 40 nm | GTX 480 |
| Kepler | 2012 | SMX, 192 (GK110) | software scheduler, wide SM | 28 nm | K20 |
| Maxwell | 2014 | SMM, 128 (GM200) | efficiency, 4x32 partitions | 28 nm | GTX 980 Ti |
| Pascal | 2016 | GP100 / GP102 | NVLink, HBM2 (GP100), 16 nm | 16 nm | P100 |
| Volta | 2017 | GV100, 64 FP32 | 1st Tensor Core, ITS | 12 nm | V100 |
| Turing | 2018 | TU102, 64 FP32 | RT Core + 2nd Tensor to graphics | 12 nm | RTX 2080 Ti |
| Ampere | 2020 | GA100, 64 FP32 | 3rd Tensor (TF32/sparsity), MIG | 7 nm | A100 |
| Ada | 2022 | AD102, 128 FP32 | 4th Tensor, 3rd RT, SER | 4 nm | RTX 4090 |
| Hopper | 2022 | GH100, 128 FP32 | Transformer Engine (FP8), TMA, clusters | 4 nm | H100 |
| Blackwell | 2024 | 2 dies, 208 B | FP4, TMEM, 5th NVLink, scale-first | TSMC 4NP | B200 / GB200 |
| Rubin | 2026 | 2 dies, 336 B | HBM4, NVLink 6; Vera Rubin = rack platform | not finalized in NVIDIA public specs | Rubin / Vera Rubin NVL72 |

## Synthesis: Three Trajectories

Three trajectories run the whole length. Specialization increases: the SM keeps a general core and accretes Tensor Cores, RT Cores, a Transformer Engine, and dedicated Tensor Memory around it. Precision decreases: FP32 down to FP4, because AI pays for throughput with bits. And the unit of design grows: chip, then two dies, then a rack.

The one thing that does not change is the bottleneck. From coalescing in the CUDA C post to Hopper's TMA to Blackwell's TMEM, every generation spends most of its new hardware on moving and hiding data, not on raw FLOPs. Compute has been cheap for a decade; feeding it has not. The through-line of the whole family tree is the memory wall.

## References

- [Fabien Sanglard, A history of NVidia Stream Multiprocessor](https://fabiensanglard.net/cuda/): the Tesla-to-Turing narrative and SM design changes.
- [SemiAnalysis, NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell): precision, asynchrony, and the tile-size argument.
- [Chips and Cheese, Blackwell: NVIDIA's Massive GPU](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu): the scale-over-specialization microarchitecture read.
- NVIDIA primary architecture docs: [Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf), [Kepler GK110](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf), [Maxwell tuning](https://docs.nvidia.com/cuda/maxwell-tuning-guide/), [Pascal GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [Volta GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [Ampere A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [Ampere GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), and [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) and [Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/): primary numbers for the newest generations.
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) and [NVIDIA Rubin platform technical blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/): Rubin GPU, NVLink 6, NVL72, and preliminary-spec caveats.
