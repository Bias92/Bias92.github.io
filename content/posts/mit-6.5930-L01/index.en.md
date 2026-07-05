---
title: "6.5930 L01 - Introduction and Applications"
date: 2026-03-07
tags: ["MLSys", "DNN-Accelerator", "MIT-6.5930"]
draft: false
---

## A Word from Ilya Sutskever (L01-3)

![Ilya Sutskever Quote](images/L01-3-ilya-quote.png)

Before we begin, let me quote Ilya Sutskever (co-founder of OpenAI) on why you should take this course.

He said this at the 50th anniversary event for the ACM Turing Award in 2017: "Compute has been the oxygen of deep learning." The point is that no matter how good your algorithm is, you cannot run it without compute. Read the other way around, it means the person who makes compute more efficient is the one who resolves the bottleneck for progress in deep learning.

This is also precisely why MLSys is an AI-resistant career. No matter how much the AI models change (CNN → Transformer → MoE → Mamba...), compute is ultimately inevitable. The people who build models get reset every time a new architecture appears, but the people who work on the hardware and system infrastructure that runs those models efficiently stay in demand.

## AI Ingredients: The Three Elements Behind the AI Explosion (L01-2)

![AI Ingredients](images/L01-2-ai-ingredients.png)

1) Big Data. Facebook processes 350 million images a day, YouTube gets 300 hours of video per minute, and Walmart handles 2.5PB per hour. Because this data exists, we can train models like LLMs.

2) GPU Acceleration. A GPU like the Tesla T4 makes MatMul possible, which a CPU cannot run quickly. It performs parallel computation across hundreds to thousands of cores, so deep learning training that used to be impractical now fits within realistic time budgets.

3) New ML Techniques. Just as AlexNet opened the CNN era and LLMs/Transformers dominate today, hardware requirements keep changing as algorithms advance.

## GPU Evolution into DNN-Specific Hardware (L01-4)

![GPU Evolution](images/L01-4-gpu-evolution.png)

After the Pascal architecture was born in 2016, general-purpose CUDA cores could perform MatMul (FLOP), and a year later in 2017 the V100 Tensor Core appeared (a unit friendly to assembly instructions like FMA). At the same chip area as general-purpose CUDA cores, throughput jumped by several multiples, and FP16 precision support made computation dramatically faster.

In 2020, the A100 Tensor Core added sparsity support. This is a technique that raises throughput by skipping when 50% of the weights are zero and computing only the valid values.

> Note: A100 sparsity does not work with just any sparse matrix. Only 2:4 structured sparsity (exactly 2 zeros among 4 consecutive elements) is recognized and accelerated by the hardware. Because of this constraint, the pruning algorithm and the hardware sparsity pattern have to match up together, and this is the core of algorithm-hardware co-design.

## Software Companies Building HW (L01-5)

![Software Companies Building HW](images/L01-5-sw-companies-hw.png)

Most of the devices mentioned so far are NVIDIA products. So why do Google and Amazon build their own chips?

The bottom line is cost plus optimization. An NVIDIA GPU is general-purpose, designed to run everything beyond DNNs, but from Google's perspective most datacenter work is DNN inference/training, so building a chip tuned exactly for that yields far better performance per watt and per area (TOPS/W, TOPS/mm²).

- Google TPU. v1 (2016) was inference-only, specialized for matrix multiplication with a 256×256 systolic array. v4 also supports training.
- Amazon Inferentia/Trainium. Built to reduce the inference/training cost of the models running on AWS.
- Domestically, companies like FuriosaAI and Rebellions are designing in the same direction.

## Cerebras WSE (L01-6)

![Cerebras WSE](images/L01-6-cerebras-wse.png)

An interesting extreme case is the Cerebras WSE. A normal chip is a single small die cut from a wafer, but Cerebras uses the entire wafer as one chip. As a result it boasts a whopping 18GB of SRAM (the A100 is around 20MB, and even a TPU is within 32MB).

That said, in practice yield, cooling, and price problems are so severe that it is not widely used. Think of it as an example that shows how far domain-specific design can go.

## Mobile SoCs for DNNs (L01-7)

![Mobile SOCs](images/L01-7-mobile-soc.png)

DNN-specific units go not only into datacenters but also into phone and laptop chips. This signals that the era of Edge Device AI is arriving.

- Apple A11 (2017). The first to carry the ANE (Apple Neural Engine). For on-device inference such as FaceID and Animoji.
- Apple M2 (2022). 16 Neural Engine cores. 26x faster than the A11. The fact that the speed is 26x with the same core count means the core architecture itself improved significantly from generation to generation.
- Conditions under which the ANE runs well: a small, integer-quantized, pruned model. These are the practical constraints of on-device inference.

## Growth in Computing Energy Consumption (L01-8)

![alt text](image-1.png)

The forecast is that datacenters will grow from 3% of total U.S. electricity (2022) to 8% (projected 2030) [Goldman Sachs, April 2024]. There is also a prediction that ICT as a whole will account for 20.9% of the world's electricity.

Model sizes keep growing, but power is not infinite, so doing the same computation with less energy has become the core challenge.

An important fact here: data movement eats far more energy than computation. A DRAM access consumes roughly 200x the energy of an ALU operation. In other words, the core of accelerator design is not how fast you compute but how little you move data.

Approaches that address this problem:
- Expanding on-chip SRAM. Cerebras WSE packing in 18GB of SRAM is also about reducing DRAM accesses.
- FlashAttention. Instead of building the full attention matrix in HBM, it tiles into sizes that fit in SRAM and completes the computation inside the SM. This is a method that reduces memory accesses without accuracy loss to raise speed.
- Memory hierarchy design. Reusing data as much as possible from nearby memory in the order register file → local buffer → global buffer → DRAM (covered in detail in the later part of this course).

In the end the common principle is one: keep data as close to compute as possible, and minimize the number of accesses to distant memory.

## Computing Cost of ChatGPT (L01-12)

GPT-3 has 96 layers, 175B parameters, and the total floating-point operations needed for training is 3.14×10²³ FLOPS.
- Running it on a single V100 takes 355 years
- Running it in the cloud still costs about $4.6M (roughly 6 billion KRW)
- GPT-4 is estimated at $100M+ (roughly 130 billion KRW or more)

## Changing Trends: DeepSeek (L01-13)

![](2026-03-07-14-59-08.png)

DeepSeek accomplished for about 8 billion KRW the training that cost GPT-4 more than 130 billion. As a result, market competition is heating up around making existing models far cheaper and more competitive.

DeepSeek's core technique is the MoE (Mixture of Experts) architecture. It has 671B parameters but activates only 37B per token. Rather than running all parameters, it selectively runs only the experts it needs, dramatically raising compute efficiency.

## Training vs Inference (L01-15)

![](2026-03-07-15-05-37.png)

DNN computation splits broadly into two kinds:

- Training. Expensive per run but done occasionally (once you train the model once and obtain appropriate weights, you are done).
- Inference. Cheap per run but runs trillions of times every day (send a question Q to an agent and get an answer A, and one single-turn inference occurs).

In other words, the cumulative cost of inference vastly exceeds that of training. Because of this, the LLM Inference Optimizing job field is in the spotlight.

## Advantages of On-Device (L01-18)

![](2026-03-07-15-09-09.png)

On-Device has three advantages over Cloud:

1) Communication. Environments with no network or an unstable one (developing countries, remote areas, etc.). If you cannot connect to the cloud, you cannot use AI.
2) Privacy. Sensitive information like medical (EMR, EHR) or defense (Palantir) data must not be sent to the cloud. Processing it directly on the device means the data never leaks out.
3) Latency. Cases requiring real-time response, like autonomous driving. Cloud round-trip delay (tens to hundreds of ms) is fatal. It must be processed right at the edge device.

## Self-Driving Cars (L01-19)

Autonomous driving is a representative case of Edge Inference:

- Cameras plus radar generate roughly 6GB of data about every 30 seconds
- One self-driving car = DNN × 60Hz × 10 cameras = 21.6M inferences per hour
- At a scale of one million cars = 21.6 trillion inferences per hour
- The prototype consumes 2,500W on compute alone. Air cooling is impossible; liquid cooling or immersion cooling is required
- Sending it to the cloud risks accidents due to latency

## Moore's Law Slowdown and the Need for Domain-Specific HW (L01-22)

![](2026-03-07-15-23-37.png)

The fundamental reason MLSys exists:

- Moore's Law slowdown. Transistor counts still rise, but speed has slowed. Transistors per dollar have also stalled.
- End of Dennard Scaling. In the past, as transistors shrank, power dropped proportionally. This broke around 2005. On the graph you can see clock speed (blue) and TDP (gray) go flat.

Since the performance and efficiency of general-purpose processors no longer rise automatically, domain-specific hardware is needed.

General-purpose hardware (e.g. CPU) can do everything: word processing, games, web browsers, DNNs, and so on, but is not specialized for any one of them. Domain-specific hardware is optimized for a particular domain:

- NVIDIA Tensor Core. Matrix multiplication only
- Google TPU. DNN inference/training
- Apple ANE. On-device DNN inference
- FuriosaAI NPU. AI inference

## L01-23 ~ L01-25: The Evolution of the CPU Pipeline

These are pipeline diagrams showing how a CPU (general-purpose hardware) works. The reason these slides appear is to set up the point that "a general-purpose CPU has to get this complicated to raise utilization, but for DNNs all of this complexity is pure waste."

### Simple In-Order Pipeline (L01-23)

![](2026-03-07-16-09-22.png)

This is a structure that splits the ID stage of a 5-stage pipeline (IF→ID→EX→MEM→WB) into Decode and Reg Read, making it 6 stages. With less work per stage, you can raise the clock frequency. Since the width is greater than 1, you can call it superscalar.

Each stage:

Fetch. It fetches an instruction from the Icache (instruction cache) at the address the PC points to. The next PC is decided by branch prediction. It predicts the branch outcome in advance so it can keep fetching without stalls.

Decode. It interprets the fetched instruction, parsing the opcode, rs, rd, and immediate values.

Reg Read. It reads the source operand values from the register file (Regs). It reads several operands simultaneously, up to the superscalar width.

Execute. The ALU (the red chevron symbol in the figure) performs the actual computation. Three chevrons means three functional units (for example, a combination of an integer ALU, an FP/Multiply unit, and a branch unit). The superscalar width is designed to match this functional-unit count, and in this figure the width = 3.

Dcache/Store Buffer. The memory-access stage. On a load it reads data from the Dcache, and on a store it writes to the Store Buffer and reflects it into the cache later.

Reg Write. It writes the computation result back into the register file (the Write-Back of the 5-stage pipeline).

### Basic Out-of-Order Pipeline (L01-24)

![alt text](image-2.png)

What changed going from In-Order to Out-of-Order:

The limit of the in-order superscalar is clear. Even with three functional units (width=3), if the leading instruction is blocked by a cache miss, all the following instructions stall too. No matter how much you increase the superscalar width, throughput does not go up.

To solve this, OoO added a Priority Queue (Issue Queue). Even if one instruction is blocked by a cache miss, the remaining instructions whose operands are ready can be sent to Execute first regardless of order, so execution can continue.

### SMT: Simultaneous Multi-Threading (L01-25)

> Note: this is not SIMT (GPU). SMT ≠ SIMT.

![alt text](image-3.png)

The Red/Yellow/Green in the figure denote different threads. The key point is that it does not replicate the pipeline hardware; rather, multiple threads share a single pipeline. (The place that uses this remarkably and absurdly well is Intel, which has something called Hyper-Threading.)

SMT looks hard, but it just adds Thread Choosing to the out-of-order pipeline so that multiple threads share one pipeline and fill empty slots, thereby increasing throughput.

Summary of pipeline evolution:

| Structure | Characteristics |
|------|------|
| Scalar | 5-stage, width=1, one instruction per cycle |
| Superscalar (In-Order) | width>1, several per cycle, in order |
| OoO + Superscalar | width>1, several per cycle, ready ones first |
| OoO + SMT | the above + multiple threads sharing the pipeline |

All of this control logic (branch prediction, register renaming, issue queue, retire, thread choosing...) is overhead for the sake of generality. Since DNNs have a regular computation pattern (repeated MatMul), all of this is unnecessary, so an accelerator strips it all out and keeps only the PE array + memory hierarchy.

## Every Accelerator is Unique (L01-26)

![alt text](image-4.png)

Now for the main topic. We have learned that the CPU is useless for a DNN accelerator.

There are also many kinds of DNN accelerators. The designs in the picture above are all tensor-operation accelerators, yet they all look different:

- Eyeriss [JSSC2017]. For CNN inference, a spatial array with a local scratchpad at each PE
- Eyeriss V2 [JETCAS2019]. An improved Eyeriss, adding a flexible NoC
- SCNN [ISCA2017]. Dedicated to sparse CNNs, a structure that skips zero values
- ExTensor [MICRO2019]. Accelerates sparse tensor operations
- Gamma [ASPLOS2021]. Specialized for sparse matrix multiplication
- spZip [ISCA2021]. Specialized for compressing/decompressing sparse data

The conclusion is one: every accelerator is designed differently. A CPU has a standardized pipeline structure, but a DNN accelerator has completely different PE placement, memory hierarchy, and dataflow depending on the target workload (dense/sparse, CNN/Transformer, etc.). In this course you learn a framework for analyzing these design differences.

> A PE (Processing Element) is a single compute unit. What the ALU does in a CPU is called a PE in a DNN accelerator. Inside it holds a multiplier plus an adder plus a small register file, so it performs the MAC (Multiply-Accumulate) operation.

## TeAAL Pyramid of Concerns & FuseMax (L01-28 ~ L01-34)

> From here on, unfamiliar terms get thrown around a lot. It is fine to just skip ahead. The next lecture covers it in detail.

### TeAAL Pyramid of Concerns (L01-28)

![alt text](image-6.png)

This is a framework that decomposes accelerator design into four layers. The Architecture on the left imposes constraints on each layer, and the decisions get finer-grained as you go up. Just as the OSI 7 Layer model separates the network by layer, this pyramid separates accelerator design decisions by layer.

| Layer | Meaning | Course Link |
|------|------|----------|
| Compute (top) | What operation you do (MatMul, Attention) | Lab 1: Einsum |
| Mapping | How you place the operation onto the HW (tiling, dataflow) | Lab 2, 3 |
| Format | How data is stored (dense, sparse, etc.) | Lab 4: Sparsity |
| Binding (bottom) | Assignment to physical HW (PE, Buffer) | Lab 2, 3 |

This pyramid is the framework that runs through the entire course.

### FuseMax: A Real Application of the Pyramid (L01-29)

![alt text](image-7.png)

FuseMax [Nayak et al., MICRO 2024] is an accelerator design for Transformer Attention. By improving one thing at each layer, it raised PE utilization from 0% to about 90%.

| Step | Layer | Improvement | Result |
|------|------|----------|------|
| Cascade | Compute | Fuse MatMul+Softmax, remove data movement | util rises slightly |
| Arch change | Architecture → Mapping | Modify the architecture to enable a new mapping | util rises meaningfully |
| Binding improvement | Binding | Efficient assignment to PEs, maximize resource use | util ~90% |

> Terminology:
> - Unfused = running each stage of Attention (Q×K → Softmax → ×V) separately. Each time it writes the intermediate result to memory and reads it back.
> - FLAT = the existing partial-fuse approach (prior work).
> - Cascade = FuseMax's complete fuse. It avoids writing intermediate results to memory.

Takeaway: accelerator performance comes not from a single factor but only when every layer of the pyramid is optimized together.

### Changes in PE Utilization (L01-30 ~ L01-34)

![alt text](image-8.png)

L01-30 graph: In the baseline state, PE utilization for BERT, TrXL, T5, and XLM is all below 0.25 across every model and every sequence length (1K~1M). It means you laid down hundreds of PEs, yet most of them are idle.

- x-axis = sequence length (1K ~ 1M)
- y-axis = PE utilization (0 ~ 1.0)
- red = Unfused (separated operations), orange = FLAT (the existing fused approach). Both are dismal.

After this, in L01-31~33, applying the Cascade → Architecture → Binding improvements in sequence raises utilization up to about 90%.

## L01-43

![alt text](image-9.png)

This is the typical structure of a DNN accelerator. It strips out all of the CPU's complex pipeline (branch prediction, OoO, SMT...) and keeps only the PE array + memory hierarchy.

Structure (left → right):

DRAM → Global Buffer (100~500kB) → PE array (200~1000 units) → inside each PE a Reg File (<1kB) + ALU + Control

Each PE is a small, simple unit that only does the MAC (Multiply-Accumulate) operation. Because it lacks control logic like the CPU's branch prediction or register renaming, you can pack far more compute units into the same chip area.

Key point: Normalized Energy Cost (65nm basis)

| Where the data comes from | Energy cost |
|---|---|
| The ALU operation itself | 1× (baseline) |
| Reg File (0.5~1kB) → ALU | 1× |
| PE-to-PE NoC → ALU | 2× |
| Global Buffer (100~500kB) → ALU | 6× |
| DRAM → ALU | 200× |

The energy to fetch data once from DRAM is 200x that of the actual computation. This is the concrete number behind the "data movement > computation" point made in L01-8.

So the core goal of accelerator design is to keep data as close to the PE as possible (Reg File, Global Buffer) and minimize the number of DRAM accesses. FlashAttention tiling in SRAM instead of HBM, and Cerebras packing in 18GB of SRAM, are all this same principle.

> "Farther and larger memories consume more power." This one line is the core principle of this course.

## Accelerator Design Decisions (L01-44)

![alt text](image-10.png)

These are the items a designer has to decide in the structure above, and each maps directly to a Lab in this course:

| Design decision | Specifics | Lab |
|----------|----------|-----|
| PE array | Number of PEs, how PEs connect (NoC) | Lab 2, 3 |
| Memory hierarchy | How many levels, capacity of each level, data placement | Lab 2, 3 |
| Scheduling | mapping (dataflow, tiling), parallelism, fusion | Lab 2, 3 |
| Sparsity handling | gating, skipping, compression format | Lab 4 |
| Implementation technology | RRAM, optical, superconductors, etc. | Lab 5 |

Think of it as each layer of the TeAAL Pyramid (Compute, Mapping, Format, Binding) coming down here into concrete design decisions.

## Roofline Analysis of Accelerator Inefficiency (L01-45)

![alt text](image-11.png)

The Roofline Model is a framework for determining whether an accelerator's performance bottleneck is compute-bound or memory-bound. You use it directly starting from Lab 1.

Axis interpretation:
- x-axis = Compute Intensity (MAC/data). How many operations per data element. The higher it is, the more compute-heavy.
- y-axis = performance (MAC/cycle). The number of operations actually processed per cycle.
- slope = memory bandwidth

The core of the graph:
- If you are on the left slope, you are memory-bound (fetching data is the bottleneck)
- If you are on the right flat region, you are compute-bound (the compute units are the bottleneck)

Steps 1~7 show, stepwise, "why performance drops from the theoretical peak to the actual result":

| Step | Constraining factor | Lab |
|------|---------|-----|
| Step 1 | The workload's own maximum parallelism (the full Einsum iteration space) | Lab 1 |
| Step 2 | Maximum parallelism the dataflow allows (parallel_for limit) | Lab 2, 3 |
| Number of PEs | Theoretical Peak Performance (the hardware limit) | none |
| Step 3 | Finite PE array size | Lab 2, 3 |
| Step 4 | PE array dimension constraint (a 2D array limits both sides) | Lab 2, 3 |
| Step 5 | Finite storage capacity (space for stationary data) | Lab 2, 3 |
| Step 6 | Insufficient average bandwidth | Lab 2, 3 |
| Step 7 | Insufficient instantaneous bandwidth (ramp up/down) | none |

Going from top to bottom, the roofline tightens. Starting from the theoretical peak and reflecting the realistic constraints one by one gives you the actually achievable performance.

### Measured Roofline: Nsight Compute

![alt text](image-12.png)

Above is a roofline from profiling the FlashAttention forward kernel with Nsight Compute. This is how the concept from L01-45 looks on a real GPU:

- x-axis = HW Arithmetic Intensity (FLOP/byte)
- y-axis = HW Performance (FLOP/s)
- blue lines = the bandwidth roofline of each memory level (DRAM, L2, L1)
- blue dashed line (horizontal) = compute peak (the theoretical maximum compute performance of this HW)
- colored dot = the actual measured value of the kernel. Being below the roofline means there is that much inefficiency.

Compute (SM) Throughput of 25.30% and DRAM Throughput of 4.46%, both low, means there is still plenty of room for optimization. The Step 1~7 analysis you learn in this course is exactly the tool that systematically diagnoses "why it is this low relative to peak."

> The difference between L01-45 and Nsight Compute: L01-45 is from the DNN accelerator (PE array) viewpoint, while Nsight Compute is from the GPU (SM/warp) viewpoint. The framework is the same, but the hardware terminology differs. PE ↔ SM, Reg File ↔ Shared Memory, Global Buffer ↔ L2 Cache, DRAM ↔ HBM.
