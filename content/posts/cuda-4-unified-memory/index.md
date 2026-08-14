---
title: "04 CUDA Unified Memory: Virtual Address, Placement, and Coherence"
date: 2026-08-13
draft: false
tags: ["CUDA", "GPU Programming", "Unified Memory", "Managed Memory", "Heterogeneous Memory", "Jetson"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "CPU와 GPU가 하나의 managed allocation을 사용하는 원리를 가상 주소, 데이터 배치, migration, cache coherence로 설명하고 Jetson AGX Orin의 실제 장치 속성에 적용한다."
---

CUDA 프로그램은 CPU와 GPU라는 서로 다른 processor를 함께 사용한다. CPU를 Host, GPU를 Device라고 부른다. 두 processor는 명령을 실행하는 방식뿐 아니라 memory에 접근하는 방식도 다르다. 이런 구조를 heterogeneous system이라고 한다.

[Host-Device 데이터 흐름]({{< relref "/posts/cuda-c-basics" >}}#host-device-데이터-흐름)에서는 CPU용 `h_data`와 GPU용 `d_data`를 따로 만들고, 계산 전후에 `cudaMemcpy`를 호출했다. 위치와 이동 시점이 코드에 그대로 보이는 explicit memory management다. 대신 자료구조가 복잡해질수록 host pointer, device pointer, copy 방향, 수명을 모두 개발자가 맞춰야 한다.

Unified Memory는 기존 CPU·GPU 메모리 구조 위에서 두 처리 장치가 하나의 managed allocation을 사용하게 한다. 이 글은 가상 주소와 데이터 배치, 동기화와 캐시 일관성을 설명한 뒤 Jetson AGX Orin의 실제 장치 속성에 적용한다.

## CPU Memory와 GPU Memory

Allocation은 프로그램이 사용할 memory 영역을 확보하는 일이다. Allocator는 요청한 크기의 영역을 마련하고 시작 address를 반환하는 software다. Pointer는 object나 memory 위치를 가리키는 address이며, allocation API가 반환한 pointer는 확보한 영역의 시작을 가리킨다.

`malloc`은 CPU code가 사용할 allocation을 만든다. `cudaMalloc`은 GPU가 접근하도록 CUDA가 관리하는 device allocation을 만든다. 두 API가 반환한 pointer는 각각 다른 memory 영역을 가리킬 수 있다.

discrete GPU가 달린 일반적인 PC에서는 CPU DRAM과 GPU 전용 memory(VRAM)가 물리적으로 분리돼 있다. 앞 글처럼 CPU의 `malloc` allocation과 GPU의 `cudaMalloc` allocation을 따로 쓰는 방식에서는 계산 전 H2D(Host to Device) copy가 필요하고, GPU 결과를 CPU에서 읽기 전 D2H(Device to Host) copy가 필요하다.

integrated GPU에서는 CPU와 GPU가 같은 시스템 DRAM을 사용한다. 두 처리 장치는 주소 변환 장치와 캐시를 각각 사용하므로 GPU에 유효한 주소 연결과 CPU·GPU 사이의 접근 순서가 필요하다.

Unified Memory는 application이 synchronization으로 정한 접근 순서 아래에서 CUDA가 주소 연결, 데이터 배치, 값의 가시성을 관리하는 방식이다. CUDA 런타임과 드라이버, 하드웨어가 현재 시스템의 지원 수준에 맞춰 이 작업을 나눠 맡는다.

## Virtual Address와 Physical Memory

### 주소 변환

CUDA process의 pointer에는 virtual address가 담긴다. MMU는 이 주소를 DRAM이나 VRAM의 physical address로 변환한다.

Virtual memory는 virtual address space를 보통 page라는 일정 크기의 단위로 나눈다. Physical memory는 같은 크기의 frame(page frame)으로 나뉜다. Page table은 virtual page가 어느 physical frame에 연결됐는지와 어떤 접근이 허용되는지 기록한다. 이 연결을 mapping이라고 한다. 아직 유효한 mapping이 없는 virtual page도 있다.

CPU나 GPU가 pointer를 읽거나 쓸 때 각 processor의 주소 변환 장치인 MMU(Memory Management Unit)가 virtual address를 현재 접근 가능한 physical address로 변환한다. Virtual address는 virtual page number와 page 안의 위치를 나타내는 offset으로 나뉜다. 주소 변환은 virtual page number를 physical frame number로 바꾸지만 offset은 그대로 유지한다.

MMU는 먼저 TLB(Translation Lookaside Buffer)에서 최근의 가상 페이지→물리 프레임 변환 결과를 찾는다. 캐시는 자주 쓰는 것을 가까운 곳에 작게 복사해 두고 먼저 찾아보는 저장소다. TLB는 데이터가 아니라 주소 변환 결과를 보관하는 캐시다.

![CPU의 virtual address가 MMU와 TLB를 거쳐 physical address로 변환되는 구조](images/address-translation.png?v=4#medium)

변환된 물리 주소는 캐시와 물리 메모리로 이어지는 메모리 계층에서 사용된다. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-and-system-memory)는 이기종 시스템에 여러 물리 메모리가 있으며, CUDA가 데이터의 할당과 배치, 이동을 관리한다고 설명한다.

### 배치와 이동

여기서 관리형 할당(managed allocation)은 `cudaMallocManaged`로 만든 할당을 말한다. CUDA 런타임과 드라이버가 저장 위치와 이동 시점, 처리 장치별 매핑을 관리한다. `cudaMalloc`은 device allocation을 만들며, host와 데이터를 주고받을 때 application이 `cudaMemcpy` 같은 explicit transfer를 요청한다.

CUDA 문서는 이 셋을 구분해서 쓴다. 아래는 discrete GPU에서 `x`가 가리키는 데이터가 시스템 DRAM에서 VRAM으로 이동하는 한 경우다. 포인터 `x`에는 가상 주소 `V`가 들어 있고, `*x`의 값은 `41`이라고 가정한다.

| 용어 | 무엇을 가리키나 | `x`를 따라간 예시 |
|---|---|---|
| mapping (매핑) | 가상 페이지를 물리 프레임에 연결하는 주소 관계 | 이동 전에는 `V`가 시스템 DRAM의 프레임 A에 연결된다. 이동 후 GPU에서는 같은 `V`가 VRAM의 프레임 B에 연결된다. 포인터에 든 `V`는 바뀌지 않는다. |
| placement (배치) | 데이터가 현재 어느 물리 메모리에 저장돼 있는가 | `41`이 시스템 DRAM의 프레임 A에 있으면 배치는 시스템 DRAM이다. 이동 후 프레임 B에 있으면 배치는 VRAM이다. |
| migration (이동) | 데이터를 다른 물리 메모리로 옮겨 배치를 바꾸는 일 | `41`이 든 페이지를 시스템 DRAM의 프레임 A에서 VRAM의 프레임 B로 복사하고, GPU의 주소 연결을 프레임 B로 바꾼다. |

![값 41이 든 관리형 페이지가 시스템 DRAM에서 메모리 컨트롤러와 PCIe를 거쳐 discrete GPU의 VRAM으로 이동하는 경로](images/migration-placement.gif?v=3#compact)

이 그림은 CPU DRAM과 VRAM이 PCIe로 분리된 discrete GPU 경로다. 배치는 관리형 할당의 각 페이지마다 정해지며, 가능한 위치는 하드웨어 구조에 따라 달라진다.

| | 관리형 데이터가 놓이는 곳 | 별도 VRAM으로 옮기는 과정 |
|---|---|---|
| discrete GPU | 시스템 DRAM 또는 VRAM | 있음 |
| integrated GPU | 공유 시스템 DRAM | 없음 |

최신 값은 그 주소에 가장 마지막으로 쓴 값이다. 이 값은 CPU나 GPU 캐시에 머물 수 있으므로 다음 처리 장치의 접근 순서와 캐시 상태를 함께 관리해야 한다.

### UVA와 Unified Memory

CUDA의 UVA(Unified Virtual Addressing)는 한 프로세스 안의 CPU 메모리와 각 GPU 메모리를 하나의 가상 주소 공간에 배치한다. CPU와 GPU는 각자 유효한 매핑을 사용한다. UVA는 메모리를 구분하는 주소 체계를 제공한다. `cudaMalloc` allocation의 접근 주체는 GPU다. Unified Memory는 CPU와 GPU가 함께 사용하는 managed allocation의 접근과 배치, 값의 가시성을 관리한다.

## Unified Memory와 Managed Allocation

Unified Memory는 CUDA가 정한 접근 순서를 지킬 때 CPU와 GPU 양쪽 코드가 사용할 수 있는 managed memory를 제공한다. `cudaMallocManaged`는 managed allocation을 명시적으로 만드는 기본 Runtime API다. CUDA Runtime은 application이 호출하는 host API를 제공하는 library다.

```cpp
int *x = nullptr;
cudaMallocManaged(&x, sizeof(*x));
```

`sizeof(*x)`만큼의 공간을 확보하고, 그 시작 주소를 pointer 변수 `x`에 기록한다. 함수가 `x`의 값을 바꿔야 하므로 첫 번째 인자는 `x`가 아니라 `&x`다. 이 allocation은 `cudaFree(x)`로 해제한다.

Explicit-copy 방식에서는 CPU용 `h_data`, GPU용 `d_data`, H2D, D2H가 필요했다. Managed 방식에서는 CPU와 GPU가 `x` 하나를 사용하며 소스에 두 copy를 적지 않는다. Runtime, driver, hardware가 현재 system의 지원 수준에 따라 주소 연결, 데이터 배치, 값의 가시성을 구현한다.

### CPU가 쓴 값을 GPU가 수정하기

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

앞 코드에는 실행 순서와 최신 값의 가시성이라는 두 층이 있다. 우선 `cudaDeviceSynchronize()`는 현재 device에 제출한 작업이 끝날 때까지 CPU thread를 기다리게 하여 GPU write를 CPU read보다 앞에 둔다.

CPU와 GPU는 DRAM 접근을 줄이려고 최근 데이터를 각자의 캐시에 보관한다. 이렇게 캐시에 올라간 값을 캐시 사본이라고 한다. 캐시는 보통 캐시 라인이라는 연속된 바이트 묶음 단위로 데이터를 가져온다. 처리 장치가 캐시 라인을 고쳤는데 그 값이 아직 아래 메모리에 반영되지 않은 상태를 dirty(더러움, 미반영), 다른 캐시에 남은 옛 사본을 stale(낡음)이라고 한다.

Synchronization은 write와 다음 read의 순서를 만든다. CUDA memory model은 synchronization 뒤 다음 처리 장치가 최신 값을 보도록 값의 가시성을 정한다. Cache coherence는 이 순서에 맞춰 cache 상태를 관리한다. 이때 구현은 지원 모델에 따라 hardware coherence 또는 driver cache maintenance를 사용한다. Write-back은 변경된 값을 아래 메모리에 반영하고, invalidation은 낡은 cache entry를 무효화한다.

다만 같은 위치를 동시에 수정하는 접근에는 synchronization이 필요하다. 실제로 앞 예제의 `41 → 42`는 GPU write 뒤 CPU read가 최신 값을 본 결과다. 정리하면 placement는 data의 물리적 위치를, coherence는 다음 처리 장치에 보이는 값을 다룬다.

## Unified Memory 지원 모델

CUDA는 managed memory의 접근 방식을 `Full model`과 `Limited model`로 나눈다. 분류 기준은 GPU가 managed memory를 언제 준비하는지와 GPU가 실행되는 동안 CPU 접근을 허용하는지다.

### Full model

`concurrentManagedAccess=1`인 지원 방식이다. GPU가 실제로 접근한 page를 그때 처리할 수 있고, CPU와 GPU는 같은 managed allocation의 서로 다른 주소를 동시에 사용할 수 있다.

### Limited model

`concurrentManagedAccess=0`인 지원 방식이다. CUDA는 kernel을 시작하기 전에 managed memory를 GPU가 사용할 수 있게 준비하고, synchronization 뒤 CPU 접근을 다시 연다.

| 비교 항목 | Full model | Limited model |
|---|---|---|
| `cudaMallocManaged` | 사용할 수 있음 | 사용할 수 있음 |
| GPU가 data를 준비하는 시점 | GPU가 실제로 접근할 때 필요한 page를 처리할 수 있음 | Kernel을 시작하기 전에 CUDA가 준비함 |
| GPU 실행 중 CPU의 managed-memory 접근 | 가능함. 같은 주소의 충돌은 synchronization이 필요함 | 허용되지 않음 |
| GPU가 사용할 수 있는 physical memory보다 큰 managed allocation | 사용할 수 있음 | 사용할 수 없음 |

Jetson AGX Orin은 shared DRAM과 `Limited model`을 함께 사용한다. 반면 다른 시스템에서는 CPU DRAM과 GPU VRAM이 분리된 discrete GPU가 `Full model`로 동작할 수 있다.

CUDA 공식 [메모리 할당 방식 표](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#overview-of-memory-allocators-for-unified-memory)의 `Placement Policy` 열은 allocation API별 데이터 배치 기준을 나타낸다. `cudaMallocManaged`의 `First touch/hint`는 처음 접근한 처리 장치와 프로그램이 전달한 성능 정보를 배치에 반영한다. Preferred-location hint는 선호 위치를 제시하며 실제 이동 시점과 최종 배치는 Runtime이 정한다.

현재 지원 모델은 operating system, kernel, driver, GPU, CPU-GPU interconnect의 조합에서 결정된다. 따라서 현재 환경의 지원 값은 `cudaDeviceGetAttribute`로 확인한다.

`managedMemory`는 explicit managed allocation을 만들 수 있는지 알려 준다. 그다음 세 attribute는 아래 순서로 읽는다.

1. `concurrentManagedAccess`가 `0`이면 `Limited model`이다.
2. 그 값이 `1`이면 `Full model`이며, `pageableMemoryAccess`가 `0`일 때는 CUDA API로 명시적으로 만든 managed allocation만 이 모델을 사용한다.
3. 두 값이 모두 `1`이면 `malloc`, `new`, `mmap` 같은 system allocation까지 Unified Memory 범위에 들어간다. 이때만 `pageableMemoryAccessUsesHostPageTables`를 읽는다. `0`은 software coherence, 즉 driver가 mapping과 migration을 관리해 앞의 cache coherence를 달성하는 방식이다. `1`은 hardware coherence로, CPU와 GPU가 같은 host page table을 쓰고 cache 상태를 hardware가 직접 맞춘다.

## Discrete GPU의 Page Fault와 Migration

다음은 CPU DRAM과 GPU memory가 분리된 software-coherent [`Full model`](#full-model)에서 GPU page fault를 migration으로 처리하는 경우다.

CPU가 managed allocation을 먼저 쓰면 해당 virtual page의 physical frame이 CPU memory에 놓일 수 있다. 이 상태에서 GPU가 그 virtual address를 처음 읽으면 유효한 GPU mapping이 없으므로 page fault가 발생한다. 이 fault는 mapping 준비가 끝날 때까지 GPU memory access를 중단시키는 recoverable event다.

그러면 memory manager와 CUDA driver는 GPU memory에 physical frame을 준비하고 page의 최신 내용을 옮긴 뒤 GPU mapping을 설치한다. 이어서 멈췄던 GPU instruction이 재개된다. Migration은 page 내용을 다른 physical memory로 옮기며 virtual address는 유지한다.

![Software coherence를 사용하는 Full model의 page fault와 migration](images/demand-paging.svg)

즉 fault는 처리를 시작시키고, migration은 가능한 해결 방법 중 하나다. 예를 들어 remote mapping을 사용하는 환경에서는 page가 CPU memory에 머문 상태로 GPU mapping이 설치된다.

CPU와 GPU가 같은 pages를 번갈아 수정하면 양방향 migration이 반복되는 page ping-pong이 생길 수 있다. 이럴 때 `cudaMemPrefetchAsync`로 지정한 범위의 데이터를 미리 옮겨 placement 시점을 앞당길 수 있다. 다만 실행 순서는 여전히 CUDA synchronization으로 정한다.

한편 HMM(Heterogeneous Memory Management)은 Linux kernel이 CPU와 GPU의 page-table 변경, device fault, page migration을 연결하는 infrastructure다. 호환되는 Linux PCIe system에서는 HMM이 system allocation을 지원하는 software-coherent `Full model`을 구현한다. Device attributes는 지원 모델을 분류하고, `nvidia-smi -q`의 Addressing Mode가 현재 HMM 사용 여부를 보여 준다.

## Jetson AGX Orin: Shared DRAM과 Limited model

앞의 개념을 실제 장치에 적용했다. 환경은 Jetson AGX Orin Developer Kit, L4T R36.5.0, JetPack 6.2.2, CUDA 12.6이다. Orin은 CPU와 GPU를 하나의 chip에 넣은 SoC(System on Chip)다. Device 0은 compute capability 8.7인 integrated GPU였다. Compute capability는 GPU가 지원하는 CUDA hardware 기능 세대를 나타낸다.

```text
device=0 name=Orin cc=8.7 integrated=1
managedMemory=1
concurrentManagedAccess=0
pageableMemoryAccess=0
```

### 판정

`managedMemory=1`은 explicit managed allocation 지원을, `concurrentManagedAccess=0`은 `Limited model`을 뜻한다. `pageableMemoryAccess=0`은 Unified Memory의 범위를 `cudaMallocManaged` 같은 explicit managed allocation으로 제한한다.

이 Orin은 `concurrentManagedAccess=0`이므로 GPU 대상 `cudaMemPrefetchAsync`를 지원하지 않는다. 그래서 측정 프로그램은 해당 호출을 건너뛰었다.

`Limited model` 판정은 위 장치 출력에 근거한다. 반면 shared SoC DRAM과 cache 동작은 NVIDIA Tegra memory model을 이 장치에 적용한 결과다.

### 공유 DRAM과 캐시

Tegra 문서에 따르면 Tegra의 CPU와 integrated GPU는 SoC DRAM을 공유하며 device memory, host memory, unified memory가 같은 physical SoC DRAM에 할당된다. 실제 출력의 `integrated=1`도 Orin GPU의 integrated 구조를 확인한다.

Orin의 managed allocation은 shared SoC DRAM에 놓인다. 하지만 CPU와 GPU는 managed data를 각각의 cache에 저장할 수 있다. 그래서 다음 처리 장치가 최신 값을 읽도록 cache 상태를 맞추는 과정이 필요하고, 이 과정이 coherence다.

Orin의 one-way I/O coherency는 GPU가 CPU cache의 최신 update를 읽게 한다. 반대로 GPU가 쓴 값을 CPU가 읽는 방향은 CUDA driver가 synchronization 경계에서 GPU cache를 관리한다.

또한 Tegra 문서는 `concurrentManagedAccess=0`인 환경의 kernel launch와 synchronization에 coherency·cache-maintenance operation이 추가되며, 이 작업이 latency를 늘릴 수 있다고 설명한다.

![Jetson AGX Orin의 one-way I/O coherency와 driver-managed GPU cache](images/orin-shared-dram.svg)

실제로 `managed_add.cu`를 Orin GPU architecture인 `sm_87` 대상으로 빌드해 실행한 결과는 다음과 같았다.

```text
before kernel: 41
after kernel:  42
```

`41 → 42`는 synchronization 뒤 CPU가 GPU의 최신 값을 읽었음을 확인한다. 다만 이번 실행은 device attributes와 값의 가시성을 기록했을 뿐 timing은 측정하지 않았다.

실행 가능한 전체 코드는 [managed_add.cu](/code/cuda-04/managed_add.cu), attribute 조회 코드는 [orin_um_probe.cu](/code/cuda-04/orin_um_probe.cu), 실제 출력은 [Orin observation](/code/cuda-04/orin-jetpack-6.2.2.txt)에 있다.

## 참고

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html): UVA, Unified Memory 지원 모델, device attributes, prefetch, HMM.
- [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html): page fault, migration, coherence, performance behavior의 상세 설명.
- [CUDA for Tegra: Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management): Tegra의 shared SoC DRAM, cache coherence, `Limited model` 지침.
