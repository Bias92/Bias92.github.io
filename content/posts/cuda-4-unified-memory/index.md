---
title: "04 CUDA Unified Memory: Virtual Address, Placement, and Coherence"
date: 2026-08-13
draft: false
tags: ["CUDA", "GPU Programming", "Unified Memory", "Managed Memory", "Heterogeneous Memory", "Jetson"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "CPU와 GPU가 하나의 메모리 할당을 함께 사용하는 원리를 가상 주소, 데이터 배치와 이동, 동기화, 캐시 일관성으로 설명하고 Jetson AGX Orin의 실제 장치 속성에 적용한다."
---

CUDA 프로그램은 CPU와 GPU라는 서로 다른 processor를 함께 사용한다. CPU를 Host, GPU를 Device라고 부른다. 두 processor는 명령 실행 방식과 memory 접근 방식이 다르다. 이런 구조를 heterogeneous system이라고 한다.

[Host-Device 데이터 흐름]({{< relref "/posts/cuda-c-basics" >}}#host-device-데이터-흐름)에서는 CPU용 `h_data`와 GPU용 `d_data`를 따로 만들고, 두 memory 사이의 데이터를 `cudaMemcpy`로 복사했다. 위치와 이동 시점이 코드에 그대로 보이는 explicit memory management다. 자료구조가 복잡해질수록 두 memory 영역과 copy 방향, 수명을 모두 개발자가 맞춘다.

Unified Memory에서는 `cudaMallocManaged`로 CPU와 GPU가 함께 사용할 메모리 영역을 만든다. 이렇게 CUDA가 관리하는 영역을 managed allocation이라고 한다. 이 글은 가상 주소와 데이터 배치, 동기화와 캐시 일관성을 차례로 설명한 뒤 Jetson AGX Orin의 실제 장치 속성에 적용한다.

## CPU Memory와 GPU Memory

Allocation은 프로그램이 사용할 memory 영역을 확보하는 일이다. Allocation API는 요청한 크기의 영역을 마련하고 시작 address를 pointer로 반환한다.

`malloc`은 CPU code가 사용할 allocation을 만든다. `cudaMalloc`은 GPU가 접근하도록 CUDA가 관리하는 device allocation을 만든다. 두 API가 반환한 pointer는 각각 다른 memory 영역을 가리킬 수 있다.

discrete GPU가 달린 일반적인 PC에서는 CPU의 주 memory인 system DRAM과 GPU 전용 memory인 VRAM이 물리적으로 분리돼 있다. 두 memory는 PCIe라는 연결 통로를 통해 데이터를 주고받는다. 앞 글처럼 CPU의 `malloc` allocation과 GPU의 `cudaMalloc` allocation을 따로 쓰면 계산 전에 H2D(Host to Device) copy를 하고, GPU 결과를 CPU에서 읽기 전에 D2H(Device to Host) copy를 한다.

integrated GPU에서는 CPU와 GPU가 같은 system DRAM을 사용한다. 두 처리 장치는 주소 변환 장치와 자주 쓰는 data를 임시 보관하는 cache를 각각 사용한다. 따라서 같은 DRAM을 사용하더라도 각 처리 장치에 맞는 주소 연결과 CPU·GPU 사이의 접근 순서가 필요하다.

프로그램은 GPU 작업이 끝난 뒤 CPU가 읽는 식으로 접근 순서를 정한다. CUDA Runtime은 프로그램이 호출하는 API를 제공하는 library이고, CUDA driver는 GPU 실행과 주소 연결을 제어하는 system software다. Runtime과 driver, hardware가 주소 연결과 데이터 배치, cache 상태를 나눠 관리한다.

## Virtual Address와 Physical Memory

### 주소 변환

CUDA process의 pointer에는 virtual address가 담긴다. MMU(Memory Management Unit)는 이 주소를 DRAM이나 VRAM의 physical address로 변환하는 장치다.

Virtual address space는 한 process가 사용할 수 있는 virtual address의 전체 범위다. 이 범위는 보통 page라는 일정 크기의 단위로 나뉜다. Physical memory는 같은 크기의 frame(page frame)으로 나뉜다. Page table은 virtual page가 어느 physical frame에 연결됐는지와 어떤 접근이 허용되는지 기록한다. 이 연결을 mapping이라고 한다. 아직 physical frame에 연결되지 않은 virtual page도 있다.

CPU나 GPU가 pointer를 읽거나 쓸 때 각 처리 장치의 MMU가 virtual address를 physical address로 변환한다. Virtual address는 virtual page number와 page 안의 위치를 나타내는 offset으로 나뉜다. 주소 변환은 virtual page number를 physical frame number로 바꾸고 offset은 그대로 사용한다.

MMU는 먼저 TLB(Translation Lookaside Buffer)에서 최근의 가상 페이지→물리 프레임 변환 결과를 찾는다. 캐시는 자주 쓰는 것을 가까운 곳에 작게 복사해 두고 먼저 찾아보는 저장소다. TLB는 주소 변환 결과를 보관하는 캐시다.

![CPU의 virtual address가 MMU와 TLB를 거쳐 physical address로 변환되는 구조](images/address-translation.png?v=4#medium)

변환된 physical address는 실제 데이터가 있는 system DRAM이나 VRAM의 위치를 가리킨다. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-and-system-memory)는 CUDA가 이런 여러 physical memory 사이의 데이터 배치와 이동을 관리한다고 설명한다.

### 배치와 이동

Managed allocation은 `cudaMallocManaged`로 만든 memory 영역이다. CUDA Runtime과 driver가 이 영역의 저장 위치와 이동 시점, 처리 장치별 mapping을 관리한다. `cudaMalloc`은 device allocation을 만들며, host와 데이터를 주고받을 때 프로그램이 `cudaMemcpy`를 요청한다.

CUDA 문서는 이 셋을 구분해서 쓴다. 아래는 discrete GPU에서 `x`가 가리키는 데이터가 시스템 DRAM에서 VRAM으로 이동하는 한 경우다. 포인터 `x`에는 가상 주소 `V`가 들어 있고, `*x`의 값은 `41`이라고 가정한다.

| 용어 | 무엇을 가리키나 | `x`를 따라간 예시 |
|---|---|---|
| mapping (매핑) | 가상 페이지를 물리 프레임에 연결하는 주소 관계 | 이동 전에는 `V`가 시스템 DRAM의 프레임 A에 연결된다. 이동 후 GPU에서는 같은 `V`가 VRAM의 프레임 B에 연결된다. 이동 전후의 포인터에는 같은 `V`가 들어 있다. |
| placement (배치) | 데이터가 현재 어느 물리 메모리에 저장돼 있는가 | `41`이 시스템 DRAM의 프레임 A에 있으면 배치는 시스템 DRAM이다. 이동 후 프레임 B에 있으면 배치는 VRAM이다. |
| migration (이동) | 데이터를 다른 물리 메모리로 옮겨 배치를 바꾸는 일 | `41`이 든 페이지를 시스템 DRAM의 프레임 A에서 VRAM의 프레임 B로 복사하고, GPU의 주소 연결을 프레임 B로 바꾼다. |

![값 41이 든 관리형 페이지가 시스템 DRAM에서 메모리 컨트롤러와 PCIe를 거쳐 discrete GPU의 VRAM으로 이동하는 경로](images/migration-placement.gif?v=3#compact)

이 그림은 CPU DRAM과 VRAM이 PCIe로 분리된 discrete GPU 경로다. 배치는 관리형 할당의 각 페이지마다 정해지며, 가능한 위치는 하드웨어 구조에 따라 달라진다.

| | 관리형 데이터가 놓이는 곳 | 별도 VRAM으로 옮기는 과정 |
|---|---|---|
| discrete GPU | 시스템 DRAM 또는 VRAM | 있음 |
| integrated GPU | 공유 system DRAM | 공유 DRAM 안에서 접근 |

최신 값은 그 주소에 가장 마지막으로 쓴 값이다. 이 값은 CPU나 GPU 캐시에 머물 수 있으므로 다음 처리 장치의 접근 순서와 캐시 상태를 함께 관리해야 한다.

### UVA와 Unified Memory

CUDA의 UVA(Unified Virtual Addressing)는 한 프로세스 안의 CPU 메모리와 각 GPU 메모리를 하나의 가상 주소 공간에 배치한다. CPU와 GPU는 각자 유효한 매핑을 사용한다. UVA는 메모리를 구분하는 주소 체계를 제공한다. `cudaMalloc` allocation의 접근 주체는 GPU다. Unified Memory는 managed allocation의 접근과 배치를 관리하고, 앞선 처리 장치가 쓴 최신 값을 다음 처리 장치가 읽게 한다.

## Unified Memory와 Managed Allocation

Unified Memory는 CPU와 GPU 양쪽 코드가 사용할 수 있는 managed allocation을 제공한다. `cudaMallocManaged`는 이 allocation을 만드는 기본 Runtime API다.

```cpp
int *x = nullptr;
cudaMallocManaged(&x, sizeof(*x));
```

`sizeof(*x)`만큼의 공간을 확보하고, 그 시작 주소를 pointer 변수 `x`에 기록한다. `&x`는 pointer 변수 `x` 자체의 주소를 함수에 전달하므로 함수가 `x`에 시작 주소를 기록할 수 있게 한다. 이 allocation은 `cudaFree(x)`로 해제한다.

Explicit-copy 방식은 CPU용 `h_data`, GPU용 `d_data`, H2D, D2H를 코드에 둔다. Managed 방식은 CPU와 GPU가 `x` 하나를 사용하고, 데이터 이동은 Runtime과 driver, hardware가 현재 system의 지원 수준에 맞춰 처리한다.

### CPU가 쓴 값을 GPU가 수정하기

아래 코드는 같은 managed allocation을 CPU와 GPU가 차례로 사용하는 기본 형태다. Kernel은 GPU에서 실행되는 함수이고, thread는 그 함수를 실행하는 작업 단위다. 이 코드는 GPU thread 하나가 `41`에 한 번 `1`을 더하므로 `42`를 만든다.

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

`__global__`은 kernel을 선언한다. Block은 함께 배치되는 GPU thread의 묶음이다. `<<<1, 1>>>`은 block 하나에 thread 하나를 배치한다. Kernel launch 직후 CPU는 다음 코드를 계속 실행하므로 GPU write와 CPU read 사이에 `cudaDeviceSynchronize()`를 뒀다.

## Synchronization과 Cache Coherence

앞 예제는 CPU가 `41`을 쓰고, GPU가 `42`로 바꾼 뒤, CPU가 그 값을 읽는 순서다. 여기에는 두 가지 일이 필요하다. CPU는 GPU 작업이 끝난 뒤 읽어야 하고, 그때 GPU가 만든 최신 값 `42`를 읽어야 한다.

첫 번째는 synchronization이다. `cudaDeviceSynchronize()`는 앞서 제출한 GPU 작업이 끝날 때까지 CPU thread를 기다리게 한다. 따라서 GPU write가 끝난 뒤 CPU read가 시작된다.

두 번째는 cache coherence다. CPU와 GPU는 DRAM 접근을 줄이려고 최근 데이터를 각자의 cache에 보관한다. Cache는 cache line이라는 연속된 byte 묶음 단위로 데이터를 가져온다. GPU가 `42`를 cache에 쓴 뒤 CPU의 다음 read가 `42`를 얻도록 cache 상태를 맞춘다.

이 cache coherence를 달성하는 방식은 둘로 나뉜다. Hardware가 processor 사이의 cache 상태를 직접 맞추면 hardware coherence다. Driver가 access와 synchronization 경계에서 주소 연결, 데이터 이동, cache 상태를 조정해 다음 처리 장치가 최신 값을 읽게 하면 software coherence다. 이 가운데 cache의 변경값을 다른 처리 장치에 보이게 하거나 이전 cache 사본을 폐기하는 작업을 cache maintenance라고 한다.

Synchronization은 CPU가 언제 읽는지를 정하고, cache coherence는 그때 어떤 값이 보이는지를 정한다. Placement는 data가 놓인 physical memory를 나타낸다. 같은 위치를 여러 처리 장치가 수정할 때는 synchronization으로 접근 순서를 정한다.

## Unified Memory 지원 모델

CUDA는 managed allocation의 접근 방식을 `Full model`과 `Limited model`로 나눈다. 분류 기준은 GPU가 managed allocation을 언제 준비하는지와 GPU가 실행되는 동안 CPU 접근을 허용하는지다.

### Full model

`concurrentManagedAccess`는 CPU와 GPU가 managed allocation을 동시에 사용할 수 있는지를 나타내는 device attribute다. 이 값이 `1`이면 `Full model`이다. GPU가 실제로 접근한 page를 그때 준비할 수 있고, CPU와 GPU는 같은 managed allocation의 서로 다른 주소를 동시에 사용할 수 있다.

### Limited model

`concurrentManagedAccess=0`이면 `Limited model`이다. CUDA는 kernel launch 경계에서 managed memory를 GPU가 사용할 수 있게 준비하고, synchronization 뒤 CPU 접근을 다시 연다.

| 비교 항목 | Full model | Limited model |
|---|---|---|
| `cudaMallocManaged` | 사용할 수 있음 | 사용할 수 있음 |
| GPU가 data를 준비하는 시점 | GPU가 실제로 접근할 때 필요한 page를 처리할 수 있음 | Kernel launch 경계에서 CUDA가 준비함 |
| GPU 실행 중 CPU의 managed-memory 접근 | 서로 다른 주소에 접근할 수 있음 | GPU 작업을 synchronization한 뒤 CPU가 접근함 |
| GPU가 사용할 수 있는 physical memory보다 큰 managed allocation | GPU memory보다 큰 allocation도 사용함 | GPU가 사용할 수 있는 physical memory 용량 안에서 사용함 |

Jetson AGX Orin은 shared DRAM과 `Limited model`을 함께 사용한다. CPU DRAM과 GPU VRAM이 분리된 discrete GPU가 `Full model`로 동작하는 시스템도 있다.

`Full model`에서 `cudaMallocManaged`로 만든 page는 보통 그 page를 처음 읽거나 쓴 처리 장치 쪽에 배치된다. CUDA 문서는 이를 `First touch`라고 부른다. 프로그램은 `cudaMemAdvise`로 선호 위치를 driver에 알려 줄 수 있다. 이 정보를 `hint`라고 하며, driver는 이를 이후 배치 결정에 사용한다.

현재 지원 모델은 operating system, operating-system kernel, CUDA driver, GPU, CPU–GPU 연결 구조의 조합에서 결정된다. 따라서 현재 환경의 지원 값은 `cudaDeviceGetAttribute`로 확인한다.

`managedMemory`는 explicit managed allocation을 만들 수 있는지 알려 준다. 그다음 세 attribute는 아래 순서로 읽는다.

1. `concurrentManagedAccess`가 `0`이면 `Limited model`이다.
2. 그 값이 `1`이면 `Full model`이며, `pageableMemoryAccess`가 `0`일 때는 CUDA API로 명시적으로 만든 managed allocation만 이 모델을 사용한다.
3. 두 값이 모두 `1`이면 `malloc`, `new`, `mmap` 같은 system allocation까지 Unified Memory 범위에 들어간다. 이때만 `pageableMemoryAccessUsesHostPageTables`를 읽는다. `0`은 software coherence, 즉 driver가 mapping과 migration을 관리해 앞의 cache coherence를 달성하는 방식이다. `1`은 hardware coherence로, CPU와 GPU가 같은 host page table을 쓰고 cache 상태를 hardware가 직접 맞춘다.

## Discrete GPU의 Page Fault와 Migration

다음은 CPU DRAM과 GPU memory가 분리된 software-coherent [`Full model`](#full-model)에서 GPU page fault를 migration으로 처리하는 경우다. Software coherence는 CPU와 GPU의 주소 연결과 데이터 이동을 driver가 관리하는 방식이다.

초기 상태에는 CPU memory의 managed page와 CPU 쪽 mapping이 있다. GPU가 그 virtual address를 처음 읽으면 page fault가 발생하고 memory access가 멈춘다. Page fault는 해당 virtual page에 사용할 GPU mapping을 준비하라는 신호다.

Fault를 처리하는 방법에는 migration과 remote mapping이 있다. 아래 그림은 migration 경로다. Page table과 physical frame을 관리하는 operating-system memory manager가 CUDA driver와 함께 GPU memory에 physical frame을 준비하고 page 내용을 옮긴 뒤 GPU mapping을 설치한다. Mapping이 준비되면 멈췄던 GPU instruction이 재개된다.

![Software coherence를 사용하는 Full model의 page fault와 migration](images/demand-paging.svg)

Remote mapping 경로에서는 page를 CPU memory에 둔 채 GPU mapping을 그 physical frame에 연결한다. Migration은 data placement를 바꾸고, remote mapping은 placement를 유지한다.

CPU와 GPU가 같은 pages를 번갈아 수정하면 양방향 migration이 반복되는 page ping-pong이 생길 수 있다. `cudaMemPrefetchAsync`는 지정한 범위의 데이터를 미리 옮겨 placement 시점을 앞당긴다. CUDA synchronization은 CPU와 GPU의 실행 순서를 정한다.

### HMM

HMM(Heterogeneous Memory Management)은 Linux kernel에서 CPU page table의 변경, GPU fault, page migration을 연결하는 subsystem이다. HMM을 사용하는 `Full model`에서는 `malloc`, `new`, `mmap`으로 만든 system allocation도 GPU가 사용할 수 있다. Device attributes는 이 지원 범위를 분류하고, `nvidia-smi -q`의 `Addressing Mode`는 현재 HMM 사용 여부를 보여 준다.

## Jetson AGX Orin: Shared DRAM과 Limited model

앞의 개념을 실제 장치에 적용했다. 환경은 Jetson AGX Orin Developer Kit, Jetson Linux 배포판인 L4T R36.5.0, CUDA 개발 도구를 묶은 JetPack 6.2.2, CUDA 12.6이다. Orin은 CPU와 GPU를 하나의 chip에 넣은 SoC(System on Chip)다. Device 0은 compute capability 8.7인 integrated GPU였다. Compute capability는 GPU가 지원하는 CUDA hardware 기능 세대를 나타낸다.

```text
device=0 name=Orin cc=8.7 integrated=1
managedMemory=1
concurrentManagedAccess=0
pageableMemoryAccess=0
```

### 판정

`managedMemory=1`은 explicit managed allocation 지원을, `concurrentManagedAccess=0`은 `Limited model`을 뜻한다. `pageableMemoryAccess=0`은 Unified Memory의 범위를 `cudaMallocManaged` 같은 explicit managed allocation으로 제한한다.

위 장치 출력은 `Limited model`을 판정한다. NVIDIA Tegra memory model은 shared SoC DRAM과 cache 동작을 설명한다.

### 공유 DRAM과 캐시

Tegra 문서에 따르면 Tegra의 CPU와 integrated GPU는 SoC DRAM을 공유하며 device memory, host memory, unified memory가 같은 physical SoC DRAM에 할당된다. 실제 출력의 `integrated=1`도 Orin GPU의 integrated 구조를 확인한다.

Orin의 managed allocation은 shared SoC DRAM에 놓인다. Shared DRAM 위에서 CPU와 GPU는 managed data를 각각의 cache에 저장할 수 있다. 다음 처리 장치가 최신 값을 읽도록 cache 상태를 맞추는 과정이 cache coherence다.

Orin의 one-way I/O coherency는 GPU가 CPU cache의 최신 변경값을 읽게 한다. GPU가 쓴 값을 CPU가 읽는 방향에서는 CUDA driver가 synchronization 경계에서 GPU cache 상태를 관리한다.

Tegra 문서는 `concurrentManagedAccess=0`인 환경의 kernel launch와 synchronization에 cache maintenance 작업이 추가되며, 이 작업이 실행 지연 시간을 늘릴 수 있다고 설명한다.

![Jetson AGX Orin의 one-way I/O coherency와 driver-managed GPU cache](images/orin-shared-dram.svg)

실제로 `managed_add.cu`를 compute capability 8.7의 compile target인 `sm_87`로 빌드해 실행한 결과는 다음과 같았다.

```text
before kernel: 41
after kernel:  42
```

출력의 `41 → 42`는 GPU 작업이 끝난 뒤 CPU가 최신 값 `42`를 읽은 결과다. 함께 기록한 device attributes는 이 Orin의 Unified Memory 지원 범위를 보여 준다.

실행 가능한 전체 코드는 [managed_add.cu](/code/cuda-04/managed_add.cu), attribute 조회 코드는 [orin_um_probe.cu](/code/cuda-04/orin_um_probe.cu), 실제 출력은 [Orin observation](/code/cuda-04/orin-jetpack-6.2.2.txt)에 있다.

## 참고

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html): UVA, Unified Memory 지원 모델, device attributes, prefetch, HMM.
- [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html): page fault, migration, coherence, performance behavior의 상세 설명.
- [CUDA for Tegra: Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management): Tegra의 shared SoC DRAM, cache coherence, `Limited model` 지침.
