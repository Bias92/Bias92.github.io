---
title: "05 CUDA Concurrency: Streams, Async Copies, and Overlap"
date: 2026-08-22T00:00:00+09:00
draft: false
tags: ["CUDA", "GPU Programming", "CUDA Streams", "Asynchronous Execution", "Pinned Memory", "Nsight Systems"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "How data copies and kernel execution are placed in the same time window, explained in the order host and device memory, pinned memory, cudaMemcpyAsync, stream, and chunk."
---

> Source: [07 Concurrency](https://www.youtube.com/watch?v=D3LU_Jz_ar8)

A CUDA program uses a CPU and a GPU together. The CPU side is called the host and the GPU side the device; host memory is the system RAM the CPU uses, and device memory is the memory attached to the GPU. The data to compute on starts out in host memory, so for the GPU to process it the data has to be moved into device memory. The [basic flow]({{< relref "/posts/cuda-c-basics" >}}#host-device-data-flow) of a CUDA program is therefore three steps: copy the input from host memory to device memory, run the computation on the GPU, and bring the result back to host memory. The total time differs between finishing these three steps one after another and running steps that belong to different data in the same time window, and the mechanism that decides that arrangement is the stream.

## Host Memory and Device Memory

Allocation is the act of reserving a memory region for the program to use and receiving its starting address as a pointer. A pointer is a variable that holds a memory address. Host memory is allocated with `malloc` and released with `free`. Device memory is then allocated with `cudaMalloc` and released with `cudaFree`, and the pointer returned points to a region the GPU accesses. Because the two pointers point to different memories, a value the CPU wrote into the `malloc` region has to be copied before the GPU can read it.

```cpp
float *h_x = (float *)malloc(bytes);   // host memory
float *d_x = nullptr;
cudaMalloc(&d_x, bytes);               // device memory
```

## H2D Copy, Kernel Launch, D2H Copy

A copy from host memory to device memory is an H2D (Host to Device) copy, and the opposite direction is a D2H (Device to Host) copy. `cudaMemcpy` is the function that performs this copy, with the direction given in its last argument. Next, the function to run on the GPU, the kernel, is launched with the `<<<grid, block>>>` syntax, and this call is the kernel launch. Here a thread is the GPU's unit of work that executes the kernel, a block is a bundle of threads placed together, and grid is the number of blocks. Once the kernel has written its result into device memory, a D2H copy brings the result back to host memory.

```cpp
cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);   // H2D copy
transform<<<grid, block>>>(d_x, d_y, N);                // kernel launch
cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);   // D2H copy
```

These three lines have a fixed order. The kernel must run after the input has arrived in device memory, and the D2H copy must start after the kernel has finished writing the result. So when one piece of data is processed whole, the H2D copy, kernel, and D2H copy line up one after another, and the total time is the sum of the H2D copy time $T_H$, the kernel time $T_K$, and the D2H copy time $T_D$.

$$
T_{\text{serial}} = T_H + T_K + T_D
$$

All three steps are necessary, but if steps that belong to different data are slotted in between them, the GPU's copy unit and compute unit can work in the same time window. The way to do that starts from how host memory is managed inside the operating system.

## Page and Pageable Memory

The operating system manages host memory in fixed-size units called pages. A common page size is 4KB. Both the address space the program sees and the actual RAM are divided into this size, and the operating system keeps a table of which RAM location each page of the program is placed in. When `malloc` is called, only the pages of the program's address space are reserved at first, and RAM is attached when that address is first read or written. Host memory whose connection to RAM is made when needed, and may later change, is called pageable memory.

### Page Fault

When the program reads or writes a page that is not yet in RAM, a page fault occurs. A page fault is the signal for the operating system to step in and attach RAM to that page. If RAM is short at that point, the operating system writes the contents of a page that has not been used for a while out to disk and frees that spot; the disk area that holds these evicted pages is called swap or the page file. Reading an evicted page again raises another page fault, and the operating system brings it back from disk into RAM. Because of this, pageable memory can handle data larger than RAM, but which pages are in RAM changes from moment to moment, and the operating system has to step in every time one is missing.

### Pinned Memory

The GPU has a piece of hardware dedicated to copying between host memory and device memory, the copy engine. Once the CPU has issued a copy, the copy engine moves the data on its own without the CPU, and for that the RAM location of the host-side data must not change while the move is in progress. Pageable memory gives no such guarantee. So CUDA keeps a separate kind of host memory for which it has asked the operating system to leave the pages in RAM and not move them, and this host memory fixed in RAM is called pinned memory. Since its pages never go out to disk, pinned memory is also called non-pageable memory.

Pinned memory is allocated with `cudaHostAlloc` and released with `cudaFreeHost`. `cudaHostAlloc` is an allocation function that returns host memory just like `malloc`, differing only in that the returned pointer points to pinned memory. It is not a function that copies data, and it does not replace `cudaMalloc`, which creates device memory. So even after switching host memory to pinned memory, device memory is still created separately with `cudaMalloc`.

| Function | Region created |
|---|---|
| `malloc` / `free` | pageable host memory |
| `cudaHostAlloc` / `cudaFreeHost` | pinned host memory |
| `cudaMalloc` / `cudaFree` | device memory |

```cpp
float *h_x = nullptr;
cudaHostAlloc(&h_x, bytes, cudaHostAllocDefault);   // pinned host memory
float *d_x = nullptr;
cudaMalloc(&d_x, bytes);                            // device memory

// ... work that uses h_x and d_x ...

cudaFreeHost(h_x);
cudaFree(d_x);
```

To pin a region that was already created with `malloc`, use `cudaHostRegister`; to unpin it while keeping the region, use `cudaHostUnregister`.

Pinned memory occupies that much real RAM, so no more of it can be created than the RAM size, and past that limit `cudaHostAlloc` returns an out-of-memory error. And pinning a large share of RAM leaves the operating system with less RAM to work with and slows down the host side, so only the regions that exchange data with the GPU are made pinned.

![Pinned memory](images/pinned-memory-chart.svg)

## Asynchronous Calls and cudaMemcpyAsync

An asynchronous call is a call where the CPU moves on to the next line without waiting for the GPU work to complete. A kernel launch is asynchronous to begin with, so the CPU runs the next code before the kernel finishes. `cudaMemcpy`, by contrast, starts its copy only after all previously submitted GPU work has finished and holds the CPU until the copy is done. The function that removes this hold is `cudaMemcpyAsync`. Its arguments are the same as `cudaMemcpy` with one stream appended at the end; the CPU returns right after the call and the copy engine performs the copy later. The host-side pointer must be pinned memory in this case, because the copy engine can start at any moment only if the host-side pages do not move.

An asynchronous call only means the CPU does not wait; it does not mean two GPU operations actually run at the same time. Even when the call returns early, the two operations may run one after another inside the GPU. Which operations run in which order is decided by the stream.

## Stream

A stream is a sequence of operations that execute on the GPU in the order they were submitted. There are two rules. First, two operations submitted to the same stream execute in submission order, and the later one does not start before the earlier one has finished. Second, between two operations submitted to different streams, CUDA imposes no order at all. So an operation in stream 1 may run before, at the same time as, or after an operation in stream 2. To run two operations in the same time window they must be placed in different streams, and in the same stream there is no possibility of overlap. Being in different streams is a necessary condition for overlap, not a sufficient one: when the GPU has no resources to spare, operations in different streams also run one after another.

A stream is declared as a variable of type `cudaStream_t` and created with `cudaStreamCreate`. The created stream goes into the last argument of `cudaMemcpyAsync` and the fourth argument of the kernel launch's `<<<>>>`. In `<<<grid, block, 0, stream>>>`, the third value is the amount of [shared memory]({{< relref "/posts/cuda-3-shared-memory" >}}), the small memory inside the GPU that the threads of a block use together, to reserve additionally at run time, and `0` means no extra space.

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_x, h_x, bytes, cudaMemcpyHostToDevice, stream);
transform<<<grid, block, 0, stream>>>(d_x, d_y, N);
cudaMemcpyAsync(h_y, d_y, bytes, cudaMemcpyDeviceToHost, stream);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

None of the three calls holds the CPU, but because they enter the same stream, the kernel runs after the H2D copy finishes and the D2H copy starts after the kernel finishes. So the H2D copy → kernel → D2H copy order for the same data is kept by the stream. `cudaStreamSynchronize` is the function that makes the CPU wait until all work in that stream has finished, and `cudaStreamQuery` only reports whether the stream is empty without waiting. A stream that is no longer needed is removed with `cudaStreamDestroy`.

![Stream ordering](images/stream-ordering-chart.svg)

## Chunk

When a large array is processed in one piece, the whole H2D copy must finish before the kernel starts, and the whole kernel must finish before the D2H copy starts. For this reason the array is divided into several ranges, and one such piece is called a chunk. If each output element depends only on the input at the same position, chunks have no reason to wait for one another. The H2D copy, kernel, and D2H copy of one chunk are placed in the same stream so the first rule keeps their order, and the next chunk is placed in a different stream so the second rule leaves room for overlap. As a result, while the kernel of chunk 1 is running, the H2D copy of chunk 2 and the D2H copy of chunk 0 can proceed.

To put this structure in code, several streams are created and rotated across chunks. Device memory is allocated once at the full array size, and only the start of each chunk is moved with `offset`.

```cpp
constexpr int streamCount = 4;
constexpr size_t chunkElements = 1 << 20;

cudaStream_t streams[streamCount];
for (int i = 0; i < streamCount; ++i) {
    cudaStreamCreate(&streams[i]);
}

for (size_t chunk = 0, offset = 0; offset < N;
     ++chunk, offset += chunkElements) {
    const size_t count = std::min(chunkElements, N - offset);
    const size_t chunkBytes = count * sizeof(float);
    cudaStream_t stream = streams[chunk % streamCount];

    cudaMemcpyAsync(d_x + offset, h_x + offset, chunkBytes,
                    cudaMemcpyHostToDevice, stream);

    const int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    transform<<<grid, block, 0, stream>>>(
        d_x + offset, d_y + offset, count);

    cudaMemcpyAsync(h_y + offset, d_y + offset, chunkBytes,
                    cudaMemcpyDeviceToHost, stream);
}

cudaDeviceSynchronize();

for (int i = 0; i < streamCount; ++i) {
    cudaStreamDestroy(streams[i]);
}
```

Each trip through the loop handles one chunk, submitting its H2D copy, kernel, and D2H copy back to back into the same stream before moving to the next chunk. Submitting all three stages of one chunk first is called depth-first issue order. The opposite, submitting every chunk's H2D copy first and then the kernels and D2H copies grouped by kind, is breadth-first issue order. Both can produce the same overlap, but when one kind of operation piles up at the front it can exceed what the GPU can hold and reduce the overlap, so depth-first issue order is the more reliable one.

When a stream is reused, the new work attaches behind the earlier work in that stream. So chunk 4, which reuses `streams[0]`, runs after the D2H copy of chunk 0 has finished, and since the first rule keeps this order no extra control is needed. Memory allocation and stream creation, on the other hand, are finished before the loop. `cudaMalloc` and `cudaStreamCreate` take no stream argument, so placing them inside the loop breaks the overlap between the work before and after them. The loop therefore keeps only the copies and kernel launches that take a stream argument, and keeps using the resources made in advance.

The actual shape of the overlap depends on how much data each chunk copies and how long the kernel runs. If the kernel is a very short operation, the main gain comes from overlapping the H2D copy with the D2H copy rather than with the kernel.

![Chunk pipeline](images/chunk-pipeline-chart.svg)

## Default Stream

A kernel launch or `cudaMemcpy` with no stream specified goes into the default stream. In the default configuration this is the legacy default stream: work submitted to it starts only after every piece of work submitted before it, in whatever stream, has finished, and every piece of work submitted after it waits for it to finish. So if even one default-stream operation slips into the middle of the chunk loop, the overlap between the streams before and after it is cut. For this reason every copy and kernel launch in the section that builds the overlap names a stream created explicitly.

With the compiler option `nvcc --default-stream per-thread`, the default stream behaves like an ordinary stream without this waiting and one is created per CPU thread. This option is used to mix code that was already written to use the default stream with other streams without rewriting it.

![Default stream](images/default-stream-chart.svg)

## Putting a Host Function into a Stream

`cudaLaunchHostFunc` is the function that submits a function to be run on the CPU into a stream. The submitted function is called when the stream's execution reaches that position, so it runs after a kernel placed earlier in the same stream has finished. No CUDA calls such as kernel launches or `cudaMalloc` are made inside this function. When the CPU has to continue processing a kernel's result, this function orders the two without stopping everything with `cudaDeviceSynchronize`, and the older `cudaStreamAddCallback` that played the same role has been replaced by it.

## CUDA Event

A CUDA event is a marker placed inside a stream. Putting it into a stream with `cudaEventRecord` records it, and it completes when the stream's execution reaches that position. So `cudaEventSynchronize` can make the CPU wait until a given event completes, and `cudaEventElapsedTime` can read the time between two events.

Events are also used to create an order between two streams. `cudaStreamWaitEvent` makes the later work of one stream wait until an event recorded in another stream completes. This is a way for the programmer to break the second rule only at the point where it is needed, so it is used only where a wait is required.

```cpp
cudaEvent_t ready;
cudaEventCreate(&ready);

produce<<<grid, block, 0, stream0>>>(data);
cudaEventRecord(ready, stream0);

cudaStreamWaitEvent(stream1, ready, 0);
consume<<<grid, block, 0, stream1>>>(data);

cudaStreamSynchronize(stream1);
cudaEventDestroy(ready);
```

In the code above, `consume` runs after `produce` has finished. As a result, only the needed order between the two streams is created without stopping the CPU or the whole device.

![CUDA event](images/event-wait-chart.svg)

## Running Several Kernels at Once

Not only copies and kernels but also independent kernels can be placed in different streams, which creates the possibility that they run at the same time on the same GPU. But when the GPU hands out blocks it places the blocks of the kernel submitted first before those of the next, so if the first kernel fills nearly all of the GPU's compute resources, the next kernel waits until a slot opens. To observe concurrent execution, therefore, kernels are needed that use few resources and run for a long time.

If a single kernel can fill the GPU, that one kernel is the fastest. Running several kernels at once is meaningful when work arrives in small units that are hard to merge into one kernel.

Stream priority is the priority the GPU consults when deciding which stream's kernel to take the next block from. A stream with a priority is created with `cudaStreamCreateWithPriority`, and the available range is read with `cudaDeviceGetStreamPriorityRange`. This priority does not interrupt blocks that are already running, and if the lower-priority kernel finishes first the difference does not show.

## Streams on Multiple GPUs

The same stream rules carry over when there are several GPUs. `cudaGetDeviceCount` reads the number of GPUs, and `cudaSetDevice` selects the GPU that subsequent CUDA calls will target. The selected GPU is the current device. Streams and events are bound to the current device at the time they are created, so submitting a kernel to that stream while a different device is selected fails.

```cpp
cudaSetDevice(0);
cudaStreamCreate(&stream0);   // stream bound to GPU 0

cudaSetDevice(1);
cudaStreamCreate(&stream1);   // stream bound to GPU 1
```

When the CPU switches devices and submits a kernel to each GPU's stream, the kernels on the two GPUs can run at the same time. To move data between GPUs, peer access can be used. Peer access is the ability of one GPU to read and write another GPU's memory directly, and it requires the two GPUs to be on the same interconnect such as PCIe or NVLink. `cudaDeviceCanAccessPeer` checks whether it is supported; if copies go in both directions, `cudaDeviceEnablePeerAccess` is called on both sides, and then `cudaMemcpyPeerAsync` performs the copy. The data then moves straight from one GPU's memory to the other's without passing through host memory.

![Multi GPU](images/multi-gpu-chart.svg)

## Unified Memory and Prefetch

The stream rules are the same with [Unified Memory](/posts/cuda-4-unified-memory/). In that case `cudaMemPrefetchAsync` instead of `cudaMemcpyAsync` puts the migration of a managed allocation into the stream, and a kernel in the same stream runs after the migration has finished. The migration happens page by page and has to update the page records on both the CPU and the GPU, so it has more to do than `cudaMemcpyAsync`, and this can leave empty gaps on the execution timeline.

## Checking in Nsight Systems

Whether concurrent execution actually happened is checked on the GPU execution timeline. Nsight Systems is a tool that records the CPU's CUDA calls and the GPU's copies and kernel executions on the same timeline while the program runs.

```bash
nsys profile --stats=true ./overlap
```

This command saves the run as a report file and prints a summary of CUDA calls, kernels, and copies. Opening the report file in the Nsight Systems window shows the CPU-side calls on top and the GPU-side copies and kernels below. In the serial code the H2D copy, kernel, and D2H copy appear in a single line; in the code with several streams the rows split per stream and the kernel of one chunk appears in the same time range as the copy of another chunk.

In the end, concurrency is not a technique for removing dependencies. The H2D copy → kernel → D2H copy order of the same data is kept with the same stream, and only independent chunks are split across different streams. On top of that, whether the times actually overlap is decided by pinned memory and by the resources the GPU has left.

## References

1. [OLCF CUDA Training Series: CUDA Concurrency](https://www.olcf.ornl.gov/cuda-training-series/)
2. [CUDA Concurrency slides](https://www.olcf.ornl.gov/wp-content/uploads/2020/07/07_Concurrency.pdf)
3. [OLCF CUDA Training Series: HW7](https://github.com/olcf/cuda-training-series/tree/master/exercises/hw7)
4. [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
5. [CUDA C++ Best Practices Guide: Asynchronous and Overlapping Transfers with Computation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)
6. [CUDA Runtime API: API Synchronization Behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
7. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
