---
title: "04 CUDA Unified Memory: Virtual Address, Placement, and Coherence"
date: 2026-08-13
draft: false
tags: ["CUDA", "GPU Programming", "Unified Memory", "Managed Memory", "Heterogeneous Memory", "Jetson"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "Unified Memory가 통합하는 것은 물리 메모리가 아니라 CPU와 GPU의 접근 모델이다. 가상 주소, page, migration, coherence, 두 가지 지원 모델을 정리하고 Jetson AGX Orin의 실제 장치 속성에 적용한다."
---

CUDA 프로그램은 CPU와 GPU라는 서로 다른 processor를 함께 사용한다. CPU를 Host, GPU를 Device라고 부른다. 두 processor는 명령을 실행하는 방식뿐 아니라 memory에 접근하는 방식도 다르다. 이런 구조를 heterogeneous system이라고 한다.

[Host-Device 데이터 흐름]({{< relref "/posts/cuda-c-basics" >}}#host-device-데이터-흐름)에서는 CPU용 `h_data`와 GPU용 `d_data`를 따로 만들고, 계산 전후에 `cudaMemcpy`를 호출했다. 위치와 이동 시점이 코드에 그대로 보이는 explicit memory management다. 대신 자료구조가 복잡해질수록 host pointer, device pointer, copy 방향, 수명을 모두 개발자가 맞춰야 한다.

Unified Memory는 CPU와 GPU가 하나의 메모리 할당을 CUDA가 정한 접근 규칙 아래 사용하게 한다. CPU DRAM과 GPU 메모리를 하나의 물리 메모리로 합치는 기능은 아니다. 이 글은 가상 주소와 물리 프레임, 동기화와 캐시 일관성을 차례로 설명한다. 마지막에는 Jetson AGX Orin의 실제 장치 속성을 이용해, CPU와 GPU가 DRAM을 공유하더라도 관리형 메모리의 접근 방식에는 제한이 있음을 확인한다.

## CPU Memory와 GPU Memory

Allocation은 프로그램이 사용할 memory 영역을 확보하는 일이다. Allocator는 요청한 크기의 영역을 마련하고 시작 address를 반환하는 software다. Pointer는 object나 memory 위치를 가리키는 address이며, allocation API가 반환한 pointer는 확보한 영역의 시작을 가리킨다.

`malloc`은 CPU code가 사용할 allocation을 만든다. `cudaMalloc`은 GPU가 접근하도록 CUDA가 관리하는 device allocation을 만든다. 두 API가 반환한 pointer는 각각 다른 memory 영역을 가리킬 수 있다.

discrete GPU가 달린 일반적인 PC에서는 CPU DRAM과 GPU 전용 memory(VRAM)가 물리적으로 분리돼 있다. 앞 글처럼 CPU의 `malloc` allocation과 GPU의 `cudaMalloc` allocation을 따로 쓰는 방식에서는 계산 전 H2D(Host to Device) copy가 필요하고, GPU 결과를 CPU에서 읽기 전 D2H(Device to Host) copy가 필요하다.

integrated GPU에서는 CPU와 GPU가 같은 시스템 DRAM을 사용하므로 별도 VRAM을 오가는 복사가 없다. 그러나 두 처리 장치는 주소 변환 장치와 캐시를 각각 사용한다. 따라서 CPU 포인터를 GPU가 아무 조건 없이 사용할 수 있는 것은 아니다. 한쪽이 쓴 최신 값을 다른 쪽이 보려면 두 처리 장치에 맞는 주소 연결과 올바른 접근 순서가 필요하다.

Unified Memory는 이 주소 연결과 접근 순서를 CUDA가 관리하는 방식이다. 하드웨어 구조 자체를 바꾸지는 않는다. CPU 코드와 GPU 커널이 같은 할당 영역을 가리키고 정해진 순서로 접근하면 최신 값을 볼 수 있게 한다. CUDA 런타임과 드라이버, 하드웨어가 현재 시스템의 지원 방식에 맞춰 주소 연결과 데이터 배치, 캐시 상태를 관리한다.

## Virtual Address와 Physical Memory

### 주소 변환

이 글에서 다루는 일반적인 CUDA process에서 pointer에 담기는 값은 virtual address다. 이 값은 DRAM이나 VRAM chip의 물리적 위치를 직접 나타내는 physical address가 아니다.

Virtual memory는 virtual address space를 보통 page라는 일정 크기의 단위로 나눈다. Physical memory는 같은 크기의 frame(page frame)으로 나뉜다. Page table은 virtual page가 어느 physical frame에 연결됐는지와 어떤 접근이 허용되는지 기록한다. 이 연결을 mapping이라고 한다. 아직 유효한 mapping이 없는 virtual page도 있다.

CPU나 GPU가 pointer를 읽거나 쓸 때 각 processor의 주소 변환 장치인 MMU(Memory Management Unit)가 virtual address를 현재 접근 가능한 physical address로 변환한다. Virtual address는 virtual page number와 page 안의 위치를 나타내는 offset으로 나뉜다. 주소 변환은 virtual page number를 physical frame number로 바꾸지만 offset은 그대로 유지한다.

MMU는 먼저 TLB(Translation Lookaside Buffer)에서 최근의 가상 페이지→물리 프레임 변환 결과를 찾는다. 캐시는 자주 쓰는 것을 가까운 곳에 작게 복사해 두고 먼저 찾아보는 저장소다. TLB는 데이터가 아니라 주소 변환 결과를 보관하는 캐시다.

![CPU의 virtual address가 MMU와 TLB를 거쳐 physical address로 변환되는 구조](images/address-translation.png?v=4#medium)

변환된 물리 주소는 캐시와 물리 메모리로 이어지는 메모리 계층에서 사용된다. [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#unified-and-system-memory)는 이기종 시스템에 여러 물리 메모리가 있으며, CUDA가 데이터의 할당과 배치, 이동을 관리한다고 설명한다.

### 배치와 이동

여기서 관리형 할당(managed allocation)은 `cudaMallocManaged`로 만든 할당을 말한다. 어느 메모리에 둘지, 언제 옮길지, 어느 처리 장치에 매핑을 걸지를 개발자가 아니라 CUDA 런타임과 드라이버가 정하기 때문에 관리형이라고 부른다. `cudaMalloc`으로 만든 할당은 개발자가 `cudaMemcpy`로 직접 옮기므로 여기에 해당하지 않는다.

CUDA 문서는 이 셋을 구분해서 쓴다. 아래는 discrete GPU에서 `x`가 가리키는 데이터가 시스템 DRAM에서 VRAM으로 이동하는 한 경우다. 포인터 `x`에는 가상 주소 `V`가 들어 있고, `*x`의 값은 `41`이라고 가정한다.

| 용어 | 무엇을 가리키나 | `x`를 따라간 예시 |
|---|---|---|
| mapping (매핑) | 가상 페이지를 물리 프레임에 연결하는 주소 관계 | 이동 전에는 `V`가 시스템 DRAM의 프레임 A에 연결된다. 이동 후 GPU에서는 같은 `V`가 VRAM의 프레임 B에 연결된다. 포인터에 든 `V`는 바뀌지 않는다. |
| placement (배치) | 데이터가 현재 어느 물리 메모리에 저장돼 있는가 | `41`이 시스템 DRAM의 프레임 A에 있으면 배치는 시스템 DRAM이다. 이동 후 프레임 B에 있으면 배치는 VRAM이다. |
| migration (이동) | 데이터를 다른 물리 메모리로 옮겨 배치를 바꾸는 일 | `41`이 든 페이지를 시스템 DRAM의 프레임 A에서 VRAM의 프레임 B로 복사하고, GPU의 주소 연결을 프레임 B로 바꾼다. |

배치(placement)는 관리형 할당의 각 페이지마다 따로 정해진다.

배치가 가능한 위치는 하드웨어 구조에 따라 다르다.

| | 관리형 데이터가 놓이는 곳 | 별도 VRAM으로 옮기는 과정 |
|---|---|---|
| discrete GPU | 시스템 DRAM 또는 VRAM | 있음 |
| integrated GPU | 공유 시스템 DRAM | 없음 |

최신 값이 언제나 DRAM이나 VRAM에만 있는 것도 아니다. CPU나 GPU가 값을 쓰면 변경된 값이 한동안 캐시에 남을 수 있다. 다음 처리 장치가 최신 값을 보려면 올바른 접근 순서와 캐시 상태가 함께 보장돼야 한다.

### UVA와 Unified Memory

CUDA의 UVA(Unified Virtual Addressing)는 한 프로세스 안의 CPU 메모리와 각 GPU 메모리를 하나의 가상 주소 공간에 배치한다. CPU와 GPU는 서로 다른 페이지 테이블을 사용할 수 있으므로 같은 포인터 값을 쓰더라도 각 처리 장치에 유효한 매핑이 필요하다. UVA 자체는 `cudaMalloc`로 만든 장치 메모리 할당을 CPU가 읽게 만들지 않으며, 데이터 배치나 캐시 상태도 관리하지 않는다.

Unified Memory는 그 다음 층이다. CUDA는 CPU와 GPU의 접근을 관리하는 관리형 할당(managed allocation)을 제공하고, 시스템이 지원하는 방식으로 매핑과 배치, 최신 값이 보이도록 관리한다. 정리하면 UVA는 주소 체계, Unified Memory는 접근과 관리 규칙이다.

## Unified Memory와 Managed Allocation

Unified Memory는 CUDA가 정한 접근 순서를 지킬 때 CPU와 GPU 양쪽 코드가 사용할 수 있는 managed memory를 제공한다. `cudaMallocManaged`는 managed allocation을 명시적으로 만드는 기본 Runtime API다. CUDA Runtime은 application이 호출하는 host API를 제공하는 library다.

```cpp
int *x = nullptr;
cudaMallocManaged(&x, sizeof(*x));
```

`sizeof(*x)`만큼의 공간을 확보하고, 그 시작 주소를 pointer 변수 `x`에 기록한다. 함수가 `x`의 값을 바꿔야 하므로 첫 번째 인자는 `x`가 아니라 `&x`다. 이 allocation은 `cudaFree(x)`로 해제한다.

Explicit-copy 방식에서는 CPU용 `h_data`, GPU용 `d_data`, H2D, D2H가 필요했다. Managed 방식에서는 CPU와 GPU가 `x` 하나를 사용하며 소스에 두 copy를 적지 않는다. Driver는 GPU와 memory 상태를 제어하는 system software다. Runtime, driver, hardware가 현재 system의 지원 방식에 따라 address mapping, physical placement, 최신 값의 가시성을 구현한다.

### 보장하는 것과 하지 않는 것

여기까지가 `cudaMallocManaged`의 보장이다. 다음 내용은 보장하지 않는다.

- 동일한 내용의 완전한 physical copy가 CPU memory와 GPU memory 양쪽에 항상 존재한다.
- 물리적인 data copy가 전혀 없다.
- CPU와 GPU가 synchronization 없이 같은 위치를 동시에 읽고 쓸 수 있다.
- Explicit copy보다 항상 빠르다.

### 최소 예제

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

앞 코드에는 순서와 값의 가시성이라는 두 층이 있다. Synchronization은 CPU와 GPU 작업의 순서를 정한다. `cudaDeviceSynchronize()`는 현재 device에 앞서 제출한 작업이 끝날 때까지 CPU thread를 기다리게 한다. 이 함수가 성공한 뒤에야 CPU가 `x`를 읽는다. Application이 별도의 cache-flush 함수를 호출할 필요는 없다.

CPU와 GPU는 DRAM 접근을 줄이려고 최근 데이터를 각자의 캐시에 보관한다. 이렇게 캐시에 올라간 값을 캐시 사본이라고 한다. 캐시는 보통 캐시 라인이라는 연속된 바이트 묶음 단위로 데이터를 가져온다. 처리 장치가 캐시 라인을 고쳤는데 그 값이 아직 아래 메모리에 반영되지 않은 상태를 dirty(더러움, 미반영), 다른 캐시에 남은 옛 사본을 stale(낡음)이라고 한다.

Cache coherence는 올바르게 synchronization된 접근에서 다음 처리 장치가 최신 값을 보도록 캐시 사본의 상태를 관리하는 규칙이다. Write-back은 변경된 값을 아래 메모리에 반영하고, invalidation은 낡은 캐시 사본을 무효화한다. Hardware가 processor 사이의 cache 상태 전달을 지원할 수도 있고, driver가 synchronization 경계에서 필요한 cache maintenance를 수행할 수도 있다.

Synchronization과 coherence는 별도 API 두 개를 뜻하지 않는다. Program은 `cudaDeviceSynchronize()`로 실행 순서를 만들고, CUDA의 memory model과 현재 system의 driver 또는 hardware가 그 경계에서 값의 가시성을 보장한다. Placement는 실제 data가 어느 physical memory에 있는가의 문제이고, coherence는 다음 processor가 최신 값을 보는가의 문제다.

Coherence는 data race를 허용하는 기능이 아니다. CPU와 GPU가 synchronization 없이 같은 위치를 동시에 읽고 쓰면 Unified Memory에서도 결과를 보장할 수 없다. 앞 예제의 `41 → 42`는 순차 접근과 값의 가시성을 확인할 뿐, 내부 cache operation의 종류와 비용까지 측정하지 않는다.

## Unified Memory 지원 모델

- `Full model`: GPU가 managed memory를 실제로 접근할 때 필요한 page를 옮기거나, 데이터가 있는 memory를 직접 읽도록 주소를 연결할 수 있다. CPU와 GPU가 같은 managed allocation의 서로 다른 주소를 동시에 사용할 수 있으며 GPU memory 용량보다 큰 managed allocation도 사용할 수 있다. 다만 synchronization 없이 같은 위치를 동시에 수정하는 data race는 여전히 허용되지 않는다.
- `Limited model`: `cudaMallocManaged`는 사용할 수 있지만 CPU와 GPU의 접근 시점이 분리된다. CUDA는 kernel을 시작하기 전에 GPU가 managed memory를 사용할 수 있도록 준비하고, synchronization이 끝난 뒤 CPU가 다시 접근할 수 있게 한다. GPU가 실행되는 동안 CPU는 managed memory에 접근할 수 없다. GPU가 필요한 page만 접근 시점에 가져오는 방식과 GPU memory 용량보다 큰 managed allocation도 지원하지 않는다.

이 분류는 CPU와 GPU가 물리 메모리를 공유하는지와 별개다. Shared DRAM을 사용하는 장치도 `Limited model`일 수 있고, CPU DRAM과 GPU VRAM이 분리된 discrete GPU도 `Full model`일 수 있다.

CUDA 공식 [메모리 할당 방식 표](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#overview-of-memory-allocators-for-unified-memory)의 `Placement Policy`는 새 기능 이름이 아니라, 메모리 할당 API마다 데이터의 저장 위치를 어떻게 정하는지 비교하는 열 제목이다. `cudaMallocManaged` 행의 `First touch/hint`는 첫 접근과 권고 정보라는 두 배치 기준을 묶어 쓴 것이다. `First touch`는 각 페이지를 CPU나 GPU가 처음 읽거나 쓸 때, 보통 그 처리 장치의 메모리에 데이터를 두는 방식이다. `hint`는 프로그램이 CUDA에 예상 접근 방식이나 선호 위치를 알려 주는 정보이며 정확성이 아니라 성능에만 영향을 준다. 이 가운데 선호 위치를 알려 주는 정보는 데이터를 즉시 옮기지 않으며, 실제 위치를 그곳으로 고정하지도 않는다.

현재 장치가 어느 지원 모델인지 GPU 이름이나 compute capability만으로 판단하면 안 된다. Operating system, kernel, driver, GPU, CPU-GPU interconnect의 조합이 결과를 바꾼다. CUDA는 `cudaDeviceGetAttribute`로 현재 환경을 직접 조회하게 한다.

`managedMemory`는 explicit managed allocation을 만들 수 있는지 알려 준다. 그다음 세 attribute는 아래 순서로 읽는다.

1. `concurrentManagedAccess`가 `0`이면 `Limited model`이다.
2. 그 값이 `1`이면 `Full model`이며, `pageableMemoryAccess`가 `0`일 때는 CUDA API로 명시적으로 만든 managed allocation만 이 모델을 사용한다.
3. 두 값이 모두 `1`이면 `malloc`, `new`, `mmap` 같은 system allocation까지 Unified Memory 범위에 들어간다. 이때만 `pageableMemoryAccessUsesHostPageTables`를 읽는다. `0`은 software coherence, 즉 driver가 mapping과 migration을 관리해 앞의 cache coherence를 달성하는 방식이다. `1`은 hardware coherence로, CPU와 GPU가 같은 host page table을 쓰고 cache 상태를 hardware가 직접 맞춘다.

## Page Fault와 Migration

Page fault와 migration은 `Full model`의 자동 배치를 이해하기 위해 필요한 개념이다. Software coherence는 hardware protocol 대신 memory manager와 driver가 mapping과 migration을 관리하는 방식이다. 다음 설명은 CPU DRAM과 GPU memory가 분리되고 software coherence를 사용하는 `Full model`의 한 경로다. 뒤에서 다룰 Orin의 실제 경로가 아니다.

CPU가 managed allocation을 먼저 쓰면 해당 virtual page에 대응하는 physical frame이 CPU memory에 놓일 수 있다. GPU가 그 virtual address를 처음 읽을 때 GPU page table에 유효한 mapping이 없거나 GPU가 현재 그 physical frame에 직접 접근할 수 없다면 page fault가 발생한다. 이 문맥의 fault는 program crash가 아니라 “현재 상태로 이 memory access를 바로 완료할 수 없다”는 event다.

Memory manager와 CUDA driver는 GPU memory에 physical frame을 준비하고 virtual page의 최신 내용을 옮긴 뒤 GPU mapping을 설치한다. 멈췄던 GPU instruction은 그다음 재개된다. Virtual page의 내용을 한 memory domain의 physical frame에서 다른 memory domain의 frame으로 옮기는 작업이 migration이다. Pointer 값은 바뀌지 않는다.

![Software coherence를 사용하는 Full model의 page fault와 migration](images/demand-paging.svg)

Page fault와 migration은 동의어가 아니다. 그림의 경로에서는 fault가 처리를 시작하게 만든 event이고 migration이 그 해결 방법이다. 예를 들어 GPU가 interconnect를 통해 CPU memory에 있는 page를 직접 읽을 수 있는 `Full model` 환경에서는 page를 복사하지 않고 mapping만 설치할 수 있다.

CPU와 GPU가 같은 pages를 번갈아 수정하면 양방향 migration이 반복되는 page ping-pong이 생길 수 있다. Managed code가 짧아도 data movement 비용은 커질 수 있다는 뜻이다. `cudaMemPrefetchAsync`는 지정한 범위의 데이터를 특정 처리 장치 가까운 메모리로 미리 옮기도록 요청한다. 이 호출은 성능을 조정할 뿐, 올바른 실행 순서를 보장하는 synchronization 함수가 아니다.

HMM(Heterogeneous Memory Management)은 Linux kernel이 CPU와 GPU의 page-table 변경, device fault, page migration을 연결하는 infrastructure다. 호환되는 Linux PCIe system에서는 HMM이 system allocation을 지원하고 software coherence를 사용하는 `Full model`을 구현할 수 있다. 앞의 attribute 조합만으로 HMM 사용을 확정할 수는 없으며, 실제 addressing mode는 `nvidia-smi -q`에서 확인한다.

## Jetson AGX Orin: Shared DRAM과 Limited model

앞의 개념을 실제 장치에 적용했다. 환경은 Jetson AGX Orin Developer Kit, L4T R36.5.0, JetPack 6.2.2, CUDA 12.6이다. Orin은 CPU와 GPU를 하나의 chip에 넣은 SoC(System on Chip)다. Device 0은 compute capability 8.7인 integrated GPU였다. Compute capability는 GPU가 지원하는 CUDA hardware 기능 세대를 나타낸다.

```text
device=0 name=Orin cc=8.7 integrated=1
managedMemory=1
concurrentManagedAccess=0
pageableMemoryAccess=0
```

### 판정

이 출력에서 `managedMemory=1`은 명시적으로 만든 관리형 할당을 지원한다는 뜻이다. `concurrentManagedAccess=0`이므로 지원 방식은 `Limited model`로 판정된다. `pageableMemoryAccess=0`이므로 일반 `malloc`이나 `new` 할당은 Unified Memory 대상이 아니다. 앞 절의 HMM 경로도 아니다. `pageableMemoryAccessUsesHostPageTables`는 앞의 두 조건이 모두 `1`일 때만 해석하므로 이 Orin 판정에는 사용하지 않는다.

여기까지는 실제 장치 출력으로 판정한 결과다. 다음 shared-memory와 cache 설명은 이번 probe에서 내부 operation을 관측한 결과가 아니라 NVIDIA의 Tegra memory model을 이 Orin에 적용한 것이다.

### 공유 DRAM과 캐시

Tegra 문서에 따르면 Tegra의 CPU와 integrated GPU는 SoC DRAM을 공유하며 device memory, host memory, unified memory가 같은 physical SoC DRAM에 할당된다. `integrated=1`이라는 실제 출력도 이 장치가 integrated GPU임을 확인한다. 따라서 Orin의 managed allocation을 CPU DRAM에서 별도 VRAM으로 PCIe migration한다고 설명하면 틀린다.

Tegra에서 `concurrentManagedAccess=0`인 Unified Memory는 CPU와 integrated GPU 양쪽 캐시에 올라간다. Orin에는 별도 VRAM 사본이 없지만 CPU 캐시와 GPU 캐시가 하나로 합쳐진 것은 아니다. 같은 SoC DRAM 위의 캐시 사본 중 어느 쪽이 최신 값인지 맞추는 일이 남는다.

Orin은 I/O coherency, 즉 one-way coherency를 지원한다. GPU는 CPU cache의 최신 update를 읽을 수 있으므로 application이 CPU cache를 직접 clean할 필요가 없다. 반대 방향까지 hardware가 대칭으로 처리하는 full coherency는 아니다. GPU cache의 최신 값을 CPU가 읽게 만드는 데 필요한 GPU cache-management operation은 CUDA driver가 managed memory 내부에서 처리한다.

또한 Tegra 문서는 `concurrentManagedAccess=0`인 환경에서 kernel launch와 synchronization에 추가 coherency·cache-maintenance operation이 필요하다고 명시한다. 이 경계의 추가 작업은 latency를 늘릴 수 있다. 정확히 어느 캐시 라인이 write-back 또는 invalidation됐는지와 그 비용은 이번 실행에서 측정하지 않았다.

![Jetson AGX Orin의 one-way I/O coherency와 driver-managed GPU cache](images/orin-shared-dram.svg)

실제로 `managed_add.cu`를 Orin GPU architecture인 `sm_87` 대상으로 빌드해 실행한 결과는 다음과 같았다.

```text
before kernel: 41
after kernel:  42
```

실측으로 확인한 것은 `integrated=1`, `concurrentManagedAccess=0`, 그리고 synchronization 뒤의 `41 → 42`다. 이 값들로 Orin을 shared physical DRAM + `Limited model`로 판정했다. NVIDIA Tegra 문서상 `concurrentManagedAccess=0`인 장치는 GPU를 destination으로 하는 `cudaMemPrefetchAsync`를 지원하지 않으므로 probe에서도 실행하지 않았다.

실행 가능한 전체 코드는 [managed_add.cu](/code/cuda-04/managed_add.cu), attribute 조회 코드는 [orin_um_probe.cu](/code/cuda-04/orin_um_probe.cu), 실제 출력은 [Orin observation](/code/cuda-04/orin-jetpack-6.2.2.txt)에 있다.

## 결론

이번 Orin 실행은 CPU write → GPU write → CPU read의 순서와 값의 가시성만 확인했다. Cache-maintenance 비용과 explicit copy 대비 성능은 측정하지 않았으므로 빠르거나 느리다는 결론은 내리지 않는다.

Unified Memory가 통합하는 것은 physical memory chip이 아니다. CPU와 GPU가 같은 allocation을 가리키고, 허용된 순서에서 최신 값을 볼 수 있게 관리하는 규칙이다. 이 Orin의 managed allocation은 shared SoC DRAM에 저장된다. `Limited model`이므로 CPU와 GPU의 managed-memory 접근 시점이 분리되고, driver는 kernel launch와 synchronization 경계에서 coherence와 cache maintenance를 처리한다.

## 참고

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html): UVA, Unified Memory 지원 모델, device attributes, prefetch, HMM.
- [CUDA Programming Guide: Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html): page fault, migration, coherence, performance behavior의 상세 설명.
- [CUDA for Tegra: Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management): Tegra의 shared SoC DRAM, cache coherence, `Limited model` 지침.
