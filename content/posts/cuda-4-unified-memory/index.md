---
title: "04 CUDA Unified Memory: Virtual Address, Placement, and Coherence"
date: 2026-08-13
draft: false
tags: ["CUDA", "GPU Programming", "Unified Memory", "Managed Memory", "Heterogeneous Memory", "Jetson"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "Unified Memory가 통합하는 것은 물리 메모리가 아니라 CPU와 GPU의 접근 모델이다. 가상 주소, page, migration, coherence, limited/full 지원을 정리하고 Jetson AGX Orin의 실제 장치 속성에 적용한다."
---

CUDA 프로그램은 CPU와 GPU라는 서로 다른 processor를 함께 사용한다. CPU를 **Host**, GPU를 **Device**라고 부른다. 두 processor는 명령을 실행하는 방식뿐 아니라 memory에 접근하는 방식도 다르다. 이런 구조를 **heterogeneous system**이라고 한다.

[Host-Device 데이터 흐름]({{< relref "/posts/cuda-c-basics" >}}#host-device-데이터-흐름)에서는 CPU용 `h_data`와 GPU용 `d_data`를 따로 만들고, 계산 전후에 `cudaMemcpy`를 호출했다. 위치와 이동 시점이 코드에 그대로 보이는 explicit memory management다. 대신 자료구조가 복잡해질수록 host pointer, device pointer, copy 방향, 수명을 모두 개발자가 맞춰야 한다.

Unified Memory는 CPU와 GPU가 하나의 memory allocation을 CUDA가 정한 접근 규칙 아래 사용하게 한다. CPU DRAM과 GPU memory를 하나의 physical memory로 합치는 기능은 아니다. 이 글은 virtual address와 physical frame, synchronization과 cache coherence를 차례로 설명한다. 마지막에는 Jetson AGX Orin의 실제 device attribute를 이용해 shared DRAM이라는 hardware topology와 limited Unified Memory라는 access model을 구분한다.

## CPU Memory와 GPU Memory

**Allocation**은 프로그램이 사용할 address range와 그 memory의 관리 규칙을 확보하는 일이다. **Allocator**는 요청받은 크기와 alignment에 맞는 range를 찾아 allocation을 만드는 software component다. **Pointer**는 object나 memory 위치를 가리키는 address이며, allocation API가 반환한 pointer는 프로그램이 사용할 수 있는 range의 시작을 가리킨다.

`malloc`은 CPU process의 heap allocator를 사용하고, `cudaMalloc`은 CUDA Runtime과 driver를 거쳐 device allocation을 만든다. 두 allocation은 서로 다른 memory domain에 놓일 수 있다.

Allocator는 여러 thread의 allocation 요청을 동시에 받을 수 있으므로 free block 목록이나 arena 같은 내부 상태를 **mutex** 또는 **atomic operation**으로 보호할 수 있다. Mutex는 한 번에 한 thread만 해당 상태를 수정하게 하고, atomic operation은 값을 중간에 끼어드는 변경 없이 하나의 연산으로 갱신하는 CPU instruction이다. 경쟁이 없으면 mutex의 빠른 경로가 atomic instruction만으로 끝날 수 있고, 기다려야 하면 operating system이 thread를 재우고 깨운다. **Semaphore**는 사용할 수 있는 resource의 개수를 counter로 관리하는 별도의 synchronization primitive다. 따라서 mutex와 semaphore는 allocator보다 아래에 있는 memory 계층이 아니라, allocator 구현이 필요에 따라 사용하는 synchronization 도구다.

[Google TCMalloc의 공식 설계](https://google.github.io/tcmalloc/design.html)는 이 구분을 실제 allocator 구조로 보여 준다. Front-end의 per-thread 또는 per-CPU cache는 한 실행 주체만 접근하므로 대부분의 작은 allocation을 lock 없이 처리한다. Local cache가 비면 shared middle-end에서 object를 보충하는데, 이곳의 Transfer cache와 Central free list는 size class별 mutex로 보호된다. 즉 모든 `malloc`이 하나의 global lock을 잡는 것이 아니라, local fast path와 synchronization이 필요한 shared path가 나뉜다. 아래 그림은 해당 문서의 [Apache 2.0](https://github.com/google/tcmalloc/blob/master/LICENSE) 원본이다.

![Google TCMalloc의 local front-end cache, shared middle-end, OS page heap 구조](images/tcmalloc-internals.png#medium)

Discrete GPU가 달린 일반적인 PC에서는 CPU DRAM과 GPU 전용 memory(VRAM)가 물리적으로 분리돼 있다. 앞 글처럼 CPU의 `malloc` allocation과 GPU의 `cudaMalloc` allocation을 따로 쓰는 방식에서는 계산 전 H2D(Host to Device) copy가 필요하고, GPU 결과를 CPU에서 읽기 전 D2H(Device to Host) copy가 필요하다.

Integrated GPU는 CPU와 GPU가 같은 physical system memory를 공유할 수 있다는 점에서 discrete GPU와 다르다. 이 경우에는 CPU DRAM과 별도 VRAM 사이를 건너는 copy가 필요하지 않다. 그러나 같은 DRAM을 써도 heterogeneous한 두 processor가 virtual address를 변환하는 방식과, 자주 쓰는 data의 cached copy를 관리하는 방식은 서로 다를 수 있다. 따라서 shared DRAM이라는 사실만으로 CPU pointer를 GPU가 곧바로 사용할 수 있다거나, 한쪽이 쓴 최신 값이 다른 쪽에 자동으로 보인다고 결론낼 수 없다. **Physical topology**는 memory가 hardware에 어떻게 연결됐는지를 설명하고, **programming model**은 software가 그 memory에 어떻게 접근할 수 있는지를 정한다.

그래서 CUDA는 이 간극을 다루는 programming model 중 하나로 Unified Memory를 제공한다. Unified Memory는 hardware 구조를 바꾸는 기능이 아니다. 대신 CPU code와 GPU kernel이 같은 allocation을 가리키고, 허용된 순서로 접근했을 때 다음 processor가 최신 값을 보게 한다. 이를 위해 현재 system의 Runtime, driver, hardware가 support model에 맞춰 address mapping, data placement, cache 상태를 관리한다.

## Virtual Address와 Physical Memory

이 글에서 다루는 일반적인 CUDA process에서 pointer에 담기는 값은 **virtual address**다. 이 값은 DRAM이나 VRAM chip의 물리적 위치를 직접 나타내는 physical address가 아니다.

Virtual memory는 virtual address space를 보통 **page**라는 일정 크기의 단위로 나눈다. Physical memory는 같은 크기의 **frame**(page frame)으로 나뉜다. **Page table**은 virtual page가 어느 physical frame에 연결됐는지와 어떤 접근이 허용되는지 기록한다. 이 연결을 **mapping**이라고 한다. 아직 유효한 mapping이 없는 virtual page도 있다.

CPU나 GPU가 pointer를 읽거나 쓸 때 각 processor의 주소 변환 장치인 **MMU**(Memory Management Unit)가 virtual address를 현재 접근 가능한 physical address로 변환한다. Virtual address는 virtual page number와 page 안의 위치를 나타내는 offset으로 나뉜다. 주소 변환은 virtual page number를 physical frame number로 바꾸지만 offset은 그대로 유지한다.

MMU는 먼저 **TLB**(Translation Lookaside Buffer)에서 최근의 virtual-page-to-physical-frame 변환 결과를 찾는다. TLB는 data가 아니라 주소 변환 결과를 보관하는 cache다. TLB에 결과가 없으면 page table을 조회한다. 따라서 **TLB miss는 page fault가 아니다**. Page table에도 유효한 mapping이 없거나 접근 권한이 맞지 않을 때 fault가 발생할 수 있다.

![CPU의 virtual address가 MMU와 TLB를 거쳐 physical address로 변환되는 구조](images/address-translation.png?v=4#medium)

변환된 physical address는 cache와 physical memory로 이어지는 memory hierarchy에서 사용된다. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-and-system-memory)는 heterogeneous system에 여러 physical memory가 있으며, CUDA가 data의 allocation, **placement**, migration을 관리한다고 설명한다. [세부 allocator 표](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#overview-of-memory-allocators-for-unified-memory)에서도 `Placement Policy`를 별도 항목으로 두고, full-support 환경의 `cudaMallocManaged`를 `First touch/hint`로 분류한다. First touch는 보통 data를 처음 접근한 processor 쪽 memory에 초기 배치한다는 뜻이다. Hint는 driver의 배치 결정을 유도할 뿐, 즉시 migration하거나 실제 placement를 보장하지 않는다.

이 공식 용례에 따라 이 글에서 **placement**는 managed allocation의 각 page 또는 memory range를 backing하는 data가 현재 어느 physical memory에 저장돼 있는지를 뜻한다. **Mapping**이 `virtual page → physical frame`의 주소 연결이라면, placement는 그 backing data를 담은 physical memory의 위치다. Discrete system의 후보는 CPU의 system DRAM과 각 GPU의 memory다. CPU와 iGPU가 system DRAM을 공유하는 구조에서는 둘 사이에 별도 DRAM↔VRAM placement가 없다.

GPU가 CPU memory의 data를 remote access할 수 있도록 mapping만 추가하면 placement는 유지된다. 반면 **migration**은 data의 최신 내용을 다른 physical memory로 옮기므로 placement를 바꾼다. 처리 단위는 support model에 따라 다르다. Software-coherent full model은 보통 page 단위로 이동하고, hardware-coherent model은 cache-line 단위 접근도 가능하며, limited model은 virtual page보다 큰 단위로 이동할 수 있다.

최신 값이 매 순간 DRAM이나 VRAM에만 있는 것도 아니다. CPU나 GPU가 값을 쓰면 변경된 값이 한동안 cache에 남을 수 있다. 다음 processor가 최신 값을 보려면 올바른 접근 순서와 cache 상태가 함께 보장돼야 한다.

CUDA의 **UVA**(Unified Virtual Addressing)는 한 process 안의 CPU memory와 각 GPU memory를 하나의 virtual address space에 배치한다. CPU와 GPU는 서로 다른 page table을 사용할 수 있으므로 같은 pointer 값을 쓰더라도 각 processor에 유효한 mapping이 필요하다. UVA 자체는 `cudaMalloc`로 만든 device allocation을 CPU가 읽게 만들지 않으며, data placement나 cache 상태도 관리하지 않는다.

Unified Memory는 그 다음 층이다. CUDA가 CPU와 GPU의 접근을 관리하는 **managed allocation**을 제공하고, system이 지원하는 방식으로 mapping, placement, 최신 값의 visibility를 관리한다. 정리하면 UVA는 **주소 체계**, Unified Memory는 **접근과 관리 규칙**이다.

## Unified Memory와 Managed Allocation

**Unified Memory**는 CUDA가 정한 접근 순서를 지킬 때 CPU와 GPU 양쪽 코드가 사용할 수 있는 **managed memory**를 제공한다. `cudaMallocManaged`는 managed allocation을 명시적으로 만드는 기본 Runtime API다. **CUDA Runtime**은 application이 호출하는 host API를 제공하는 library다.

```cpp
int *x = nullptr;
cudaMallocManaged(&x, sizeof(*x));
```

`sizeof(*x)`만큼의 공간을 확보하고, 그 시작 주소를 pointer 변수 `x`에 기록한다. 함수가 `x`의 값을 바꿔야 하므로 첫 번째 인자는 `x`가 아니라 `&x`다. 이 allocation은 `cudaFree(x)`로 해제한다.

Explicit-copy 방식에서는 CPU용 `h_x`, GPU용 `d_x`, H2D, D2H가 필요했다. Managed 방식에서는 CPU와 GPU가 `x` 하나를 사용하며 소스에 두 copy를 적지 않는다. **Driver**는 GPU와 memory 상태를 제어하는 system software다. Runtime, driver, hardware가 현재 system의 지원 방식에 따라 address mapping, physical placement, 최신 값의 가시성을 구현한다.

여기까지가 `cudaMallocManaged`의 보장이다. 다음 내용은 보장하지 않는다.

- 동일한 내용의 완전한 physical copy가 CPU memory와 GPU memory 양쪽에 항상 존재한다.
- 물리적인 data copy가 전혀 없다.
- CPU와 GPU가 synchronization 없이 같은 위치를 동시에 읽고 쓸 수 있다.
- Explicit copy보다 항상 빠르다.

아래 코드는 같은 managed allocation을 CPU와 GPU가 차례로 사용하는 기본 형태다. `41`은 값이 바뀌었는지 확인하려고 고른 임의의 수다. GPU thread 하나가 한 번 `1`을 더하므로 예상 결과는 `42`다.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_one(int *x) {
    *x += 1;
}

int main() {
    int *x = nullptr;
    cudaMallocManaged(&x, sizeof(*x));

    *x = 41;
    std::printf("before kernel: %d\n", *x);

    add_one<<<1, 1>>>(x);
    cudaDeviceSynchronize();

    std::printf("after kernel:  %d\n", *x);
    cudaFree(x);
}
```

`__global__`은 GPU에서 실행되는 kernel을 선언한다. `<<<1, 1>>>`은 block 하나에 thread 하나를 배치한다. Kernel launch는 CPU에 대해 asynchronous하므로 GPU write와 CPU read 사이에 `cudaDeviceSynchronize()`를 뒀다.

## Synchronization과 Coherence

앞 코드에는 순서와 값의 가시성이라는 두 층이 있다. **Synchronization**은 CPU와 GPU 작업의 순서를 정한다. `cudaDeviceSynchronize()`는 현재 device에 앞서 제출한 작업이 끝날 때까지 CPU thread를 기다리게 한다. 이 함수가 성공한 뒤에야 CPU가 `x`를 읽는다. Application이 별도의 cache-flush 함수를 호출할 필요는 없다.

CPU와 GPU는 DRAM 접근을 줄이려고 최근 data를 각자의 **cache**에 보관한다. Cache는 보통 **cache line**이라는 연속된 byte 묶음으로 data를 가져온다. Processor가 cache line을 고친 뒤 변경된 값이 아직 아래 memory에 반영되지 않은 상태를 **dirty**라고 한다. 다른 cache에 남은 이전 사본은 **stale**하다.

**Cache coherence**는 올바르게 synchronization된 접근에서 다음 processor가 최신 값을 보도록 cached copy의 상태를 관리하는 규칙이다. Write-back은 변경된 값을 아래 memory에 반영하고, invalidation은 오래된 cached copy를 무효화한다. Hardware가 processor 사이의 cache 상태 전달을 지원할 수도 있고, driver가 synchronization 경계에서 필요한 cache maintenance를 수행할 수도 있다.

Synchronization과 coherence는 별도 API 두 개를 뜻하지 않는다. Program은 `cudaDeviceSynchronize()`로 실행 순서를 만들고, CUDA의 memory model과 현재 system의 driver 또는 hardware가 그 경계에서 값의 가시성을 보장한다. Placement는 backing data가 어느 physical memory에 있는가의 문제이고, coherence는 다음 processor가 최신 값을 보는가의 문제다.

Coherence는 data race를 허용하는 기능이 아니다. CPU와 GPU가 synchronization 없이 같은 위치를 동시에 읽고 쓰면 full Unified Memory에서도 결과를 보장할 수 없다. 앞 예제의 `41 → 42`는 순차 접근과 값의 가시성을 확인할 뿐, 내부 cache operation의 종류와 비용까지 측정하지 않는다.

## Limited와 Full Unified Memory

Physical topology와 Unified Memory support model은 서로 다른 축이다.

| 질문 | 가능한 구조 |
|---|---|
| CPU와 GPU의 physical memory가 분리됐는가 | CPU DRAM + GPU VRAM / shared SoC DRAM |
| Unified Memory를 어떤 방식으로 지원하는가 | limited / full |

Shared DRAM을 쓴다고 full Unified Memory인 것은 아니다. 반대로 physical memory가 분리된 discrete GPU도 operating system과 driver 조합에 따라 full Unified Memory를 지원할 수 있다.

**Limited Unified Memory**에서는 CPU와 GPU의 접근 시점이 크게 나뉜다. 기본 규칙에서는 GPU kernel이 실행되는 동안 CPU가 managed memory에 접근하면 안 된다. Kernel launch와 completion synchronization이 접근 주체를 넘기는 경계가 된다. **Oversubscription**, 즉 GPU memory보다 큰 managed allocation을 사용하는 기능도 지원하지 않는다.

**Full Unified Memory**에서는 GPU의 실제 접근 시점에 page migration이나 remote-access mapping으로 접근을 처리할 수 있고, CPU와 GPU가 서로 다른 managed 위치를 동시에 사용할 수 있다. GPU memory보다 큰 managed allocation도 만들 수 있다. Discrete system에서는 필요한 pages를 GPU memory에 두고 나머지는 system memory에 둘 수도 있다. 그래도 synchronization 없이 같은 위치를 동시에 수정하는 data race까지 허용되는 것은 아니다.

현재 장치가 어느 model인지 GPU 이름이나 compute capability만으로 판단하면 안 된다. Operating system, kernel, driver, GPU, CPU-GPU interconnect의 조합이 결과를 바꾼다. CUDA는 `cudaDeviceGetAttribute`로 현재 환경을 직접 조회하게 한다.

`managedMemory`는 explicit managed allocation을 만들 수 있는지 알려 준다. 그다음 세 attribute는 아래 순서로 읽는다.

1. `concurrentManagedAccess`가 `0`이면 limited Unified Memory다.
2. 그 값이 `1`이면 full support이며, `pageableMemoryAccess`가 `0`일 때는 CUDA API로 명시적으로 만든 managed allocation만 full model을 사용한다.
3. 두 값이 모두 `1`이면 `malloc`, `new`, `mmap` 같은 system allocation까지 Unified Memory 범위에 들어간다. 이때만 `pageableMemoryAccessUsesHostPageTables`를 읽는다. `0`은 software coherence, `1`은 host page table을 사용하는 hardware coherence를 뜻한다.

Limited도 `cudaMallocManaged`로 pointer 하나를 만들고 CPU → GPU → CPU 순서로 사용할 수 있다. “Limited”는 API가 동작하지 않는다는 뜻이 아니다. GPU 실행 중 CPU의 managed-memory 접근, fine-grained on-demand migration, oversubscription을 지원하지 않는다는 뜻이다.

## Page Fault와 Migration

Page fault와 migration은 full Unified Memory의 automatic placement를 이해하기 위해 필요한 개념이다. **Software coherence**는 hardware protocol 대신 memory manager와 driver가 mapping과 migration을 관리하는 방식이다. 다음 설명은 CPU DRAM과 GPU memory가 분리된 software-coherent full model의 한 경로다. 뒤에서 다룰 Orin의 실제 경로가 아니다.

CPU가 managed allocation을 먼저 쓰면 해당 virtual page를 backing하는 physical frame이 CPU memory에 놓일 수 있다. GPU가 그 virtual address를 처음 읽을 때 GPU page table에 유효한 mapping이 없거나 GPU가 현재 backing frame에 직접 접근할 수 없다면 **page fault**가 발생한다. 이 문맥의 fault는 program crash가 아니라 “현재 상태로 이 memory access를 바로 완료할 수 없다”는 event다.

Memory manager와 CUDA driver는 GPU memory에 physical frame을 준비하고 virtual page의 최신 내용을 옮긴 뒤 GPU mapping을 설치한다. 멈췄던 GPU instruction은 그다음 재개된다. Virtual page의 내용을 한 memory domain의 physical frame에서 다른 memory domain의 frame으로 옮기는 작업이 **migration**이다. Pointer 값은 바뀌지 않는다.

![Software-coherent full Unified Memory의 page fault와 migration](images/demand-paging.svg)

Page fault와 migration은 동의어가 아니다. 그림의 경로에서는 fault가 처리를 시작하게 만든 event이고 migration이 그 해결 방법이다. 예를 들어 GPU가 interconnect를 통해 CPU memory에 있는 page를 remote access하는 full-support system은 page를 복사하지 않고 mapping만 설치할 수 있다.

CPU와 GPU가 같은 pages를 번갈아 수정하면 양방향 migration이 반복되는 **page ping-pong**이 생길 수 있다. Managed code가 짧아도 data movement 비용은 커질 수 있다는 뜻이다. `cudaMemPrefetchAsync`는 지정한 range를 destination processor 가까이 populate하거나 migrate하도록 요청하는 performance hint이며, correctness를 보장하는 synchronization 함수는 아니다.

**HMM**(Heterogeneous Memory Management)은 Linux kernel이 CPU와 GPU의 page-table 변경, device fault, page migration을 연결하는 infrastructure다. 호환되는 Linux PCIe system에서는 HMM이 system allocation을 지원하는 software-coherent full model을 구현할 수 있다. 앞의 attribute 조합만으로 HMM 사용을 확정할 수는 없으며, 실제 addressing mode는 `nvidia-smi -q`에서 확인한다.

## Jetson AGX Orin: Shared DRAM과 Limited Unified Memory

앞의 개념을 실제 장치에 적용했다. 환경은 Jetson AGX Orin Developer Kit, L4T R36.5.0, JetPack 6.2.2, CUDA 12.6이다. Orin은 CPU와 GPU를 하나의 chip에 넣은 **SoC**(System on Chip)다. Device 0은 compute capability 8.7인 **iGPU**(integrated GPU)였다. Compute capability는 GPU가 지원하는 CUDA hardware 기능 세대를 나타낸다.

```text
device=0 name=Orin cc=8.7 integrated=1
managedMemory=1
concurrentManagedAccess=0
pageableMemoryAccess=0
```

이 출력에서 `managedMemory=1`은 explicit managed allocation을 지원한다는 뜻이다. `concurrentManagedAccess=0`에서 support model은 limited로 판정된다. `pageableMemoryAccess=0`이므로 plain `malloc`이나 `new` allocation은 Unified Memory 대상이 아니다. 앞 절의 HMM 경로도 아니다. `pageableMemoryAccessUsesHostPageTables`는 앞의 두 조건이 모두 `1`일 때만 해석하므로 이 Orin 판정에는 사용하지 않는다.

여기까지는 실제 장치 출력으로 판정한 결과다. 다음 shared-memory와 cache 설명은 이번 probe에서 내부 operation을 관측한 결과가 아니라 NVIDIA의 Tegra memory model을 이 Orin에 적용한 것이다.

Tegra 문서에 따르면 Tegra의 CPU와 iGPU는 SoC DRAM을 공유하며 device memory, host memory, unified memory가 같은 physical SoC DRAM에 할당된다. `integrated=1`이라는 실제 출력도 이 장치가 host memory system과 통합된 GPU임을 확인한다. 따라서 Orin의 managed allocation을 CPU DRAM에서 별도 VRAM으로 PCIe migration한다고 설명하면 틀린다.

Tegra에서 `concurrentManagedAccess=0`인 Unified Memory는 CPU와 iGPU 양쪽에서 cached된다. Orin에는 별도 VRAM copy가 없지만 CPU cache와 GPU cache가 하나로 합쳐진 것은 아니다. 같은 SoC DRAM 위의 cached copy가 어느 processor의 최신 값인지 맞추는 일이 남는다.

Orin은 **I/O coherency**, 즉 one-way coherency를 지원한다. GPU는 CPU cache의 최신 update를 읽을 수 있으므로 application이 CPU cache를 직접 clean할 필요가 없다. 반대 방향까지 hardware가 대칭으로 처리하는 full coherency는 아니다. GPU cache의 최신 값을 CPU가 읽게 만드는 데 필요한 GPU cache-management operation은 CUDA driver가 managed memory 내부에서 처리한다.

또한 Tegra 문서는 `concurrentManagedAccess=0`인 환경에서 kernel launch와 synchronization에 추가 coherency·cache-maintenance operation이 필요하다고 명시한다. 이 경계의 추가 작업은 latency를 늘릴 수 있다. 정확히 어느 cache line이 write-back 또는 invalidation됐는지와 그 비용은 이번 실행에서 측정하지 않았다.

![Jetson AGX Orin의 one-way I/O coherency와 driver-managed GPU cache](images/orin-shared-dram.svg)

실제로 `managed_add.cu`를 Orin GPU architecture인 `sm_87` 대상으로 빌드해 실행한 결과는 다음과 같았다.

```text
before kernel: 41
after kernel:  42
```

실측으로 확인한 것은 `integrated=1`, `concurrentManagedAccess=0`, 그리고 synchronization 뒤의 `41 → 42`다. 이 값들로 Orin을 **shared physical DRAM + limited Unified Memory**로 판정했다. NVIDIA Tegra 문서상 `concurrentManagedAccess=0`인 장치는 GPU를 destination으로 하는 `cudaMemPrefetchAsync`를 지원하지 않으므로 probe에서도 실행하지 않았다.

실행 가능한 전체 코드는 [managed_add.cu](/code/cuda-04/managed_add.cu), attribute 조회 코드는 [orin_um_probe.cu](/code/cuda-04/orin_um_probe.cu), 실제 출력은 [Orin observation](/code/cuda-04/orin-jetpack-6.2.2.txt)에 있다.

## 결론

이번 Orin 실행은 CPU write → GPU write → CPU read의 순서와 값의 가시성만 확인했다. Cache-maintenance 비용과 explicit copy 대비 성능은 측정하지 않았으므로 빠르거나 느리다는 결론은 내리지 않는다.

Unified Memory가 통합하는 것은 physical memory chip이 아니다. CPU와 GPU가 같은 allocation을 가리키고, 허용된 순서에서 최신 값을 볼 수 있게 관리하는 규칙이다. 이 Orin에서 backing storage는 shared SoC DRAM이다. Limited model이므로 CPU와 GPU의 managed-memory 접근 시점이 분리되고, driver는 kernel launch와 synchronization 경계에서 coherence와 cache maintenance를 처리한다.

## 참고

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html): UVA, Unified Memory, limited/full model, device attributes, prefetch, HMM.
- [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html): page fault, migration, coherence, performance behavior의 상세 설명.
- [CUDA for Tegra: Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management): Tegra의 shared SoC DRAM, cache coherence, limited Unified Memory 지침.
