---
title: "04 CUDA Unified Memory: cudaMallocManaged와 Jetson AGX Orin"
date: 2026-08-13
draft: false
tags: ["CUDA", "GPU Programming", "Unified Memory", "Managed Memory", "Jetson"]
categories: ["CUDA"]
series: ["CUDA C"]
summary: "cudaMallocManaged가 무엇을 할당하는지 explicit copy와 비교한다. CPU가 쓴 41을 Orin GPU가 42로 바꾼 결과와 장치 속성으로 이 환경의 Unified Memory 지원 범위를 확인한다."
---

[Host-Device 데이터 흐름]({{< relref "/posts/cuda-c-basics" >}}#host-device-데이터-흐름)에서는 CPU용 `h_data`와 GPU용 `d_data`를 따로 만들었다. 계산 전에는 H2D(Host to Device), 계산 후에는 D2H(Device to Host) `cudaMemcpy`를 호출했다. CPU memory와 GPU 전용 memory(VRAM)가 분리된 그래픽카드의 기본 방식이다.

**할당(allocation)**은 프로그램이 사용할 메모리 공간을 확보하고 그 시작 주소를 포인터로 받는 일이다. CUDA의 **Unified Memory**는 CPU 코드와 GPU kernel이 모두 접근할 수 있도록 CUDA가 관리하는 메모리 공간을 제공한다. 그 공간을 만드는 가장 기본적인 함수가 `cudaMallocManaged`다.

이 글은 `cudaMallocManaged` 한 줄의 의미부터 시작한다. CPU가 정수 하나를 쓰고, Jetson AGX Orin의 GPU가 1을 더하고, CPU가 결과를 다시 읽는다. 실제로 확인한 코드와 장치 속성만 다룬다.

## cudaMallocManaged

명시적으로 복사하면 CPU의 값과 GPU의 값을 따로 둔다.

```cpp
int h_x = 41;                         // CPU memory
int *d_x = nullptr;
cudaMalloc(&d_x, sizeof(*d_x));       // GPU memory

cudaMemcpy(d_x, &h_x, sizeof(h_x), cudaMemcpyHostToDevice);
add_one<<<1, 1>>>(d_x);
cudaMemcpy(&h_x, d_x, sizeof(h_x), cudaMemcpyDeviceToHost);
```

`cudaMallocManaged`를 쓰면 포인터를 하나만 만든다.

```cpp
int *x = nullptr;
cudaMallocManaged(&x, sizeof(*x));
```

첫 번째 인자 `&x`는 포인터 변수 `x`의 주소다. CUDA 함수를 제공하는 런타임은 새로 확보한 메모리의 시작 주소를 여기에 기록한다. 두 번째 인자 `sizeof(*x)`는 `int` 하나만큼의 바이트를 요청한다는 뜻이다.

이렇게 만든 공간을 **managed allocation**이라고 부른다. CPU 코드는 `x`를 역참조할 수 있고, GPU kernel에도 같은 `x`를 넘길 수 있다. 소스 코드에서는 `h_x`, `d_x`와 두 `cudaMemcpy`가 사라진다.

여기서 “managed”는 아무 때나 CPU와 GPU가 동시에 써도 된다는 뜻이 아니다. 어느 쪽이 언제 접근할 수 있는지는 장치의 지원 범위와 동기화에 따라 달라진다. 아래 예제는 CPU와 GPU가 같은 `x`를 **차례로** 사용한다.

## CPU의 41, GPU의 42

`41`은 CUDA에서 특별한 숫자가 아니다. CPU가 쓴 시작값과 GPU가 고친 결과를 눈으로 구분하려고 고른 값이다. GPU thread 하나가 정확히 한 번 `1`을 더하므로 예상 결과는 `42`다.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_one(int *x) {
    *x += 1;
}

int main() {
    int *x = nullptr;
    cudaMallocManaged(&x, sizeof(*x));

    *x = 41;                       // CPU가 41을 쓴다.
    std::printf("before kernel: %d\n", *x);

    add_one<<<1, 1>>>(x);          // GPU가 한 번 1을 더한다.
    cudaDeviceSynchronize();       // GPU가 끝날 때까지 CPU가 기다린다.

    std::printf("after kernel:  %d\n", *x);
    cudaFree(x);
}
```

`__global__`이 붙은 `add_one`은 GPU에서 실행되는 함수, 즉 kernel이다. `<<<1, 1>>>`은 block 하나와 thread 하나로 kernel을 실행한다. 병렬 성능을 내려는 구성이 아니라 `*x += 1`을 한 번만 실행하려는 구성이다.

커널 실행 요청은 비동기다. CPU는 GPU 작업이 끝나기 전에 다음 줄로 진행할 수 있다. `cudaDeviceSynchronize()`는 앞서 요청한 GPU 작업이 끝날 때까지 CPU를 기다리게 한다. 복사 함수가 아니다. 이 호출이 끝난 뒤 CPU가 `x`를 읽으므로 GPU가 쓴 최신 값이 보여야 한다.

본문에서는 흐름만 보이도록 오류 검사를 생략했다. 실행 가능한 전체 코드는 [managed_add.cu](/code/cuda-04/managed_add.cu)에 있다.

## Jetson AGX Orin의 실행 결과

Jetson AGX Orin은 CPU와 NVIDIA GPU를 한 칩에 넣은 SoC(System on Chip)다. 별도 그래픽카드처럼 CPU DRAM과 GPU VRAM을 PCIe로 연결한 구조가 아니다. NVIDIA의 Tegra 문서에 따르면 Orin의 host memory, device memory, unified memory는 같은 physical SoC DRAM에 할당된다. 그래서 이 장치에서는 “managed memory가 RAM과 VRAM 사이를 복사했다”라고 설명하면 틀린다.

JetPack 6.2.2, L4T R36.5.0, CUDA 12.6 환경에서 전체 소스를 다음처럼 빌드했다. `compute capability`는 CUDA가 구분하는 GPU architecture version이다. Orin의 8.7에 맞춰 `sm_87`을 지정했다.

```bash
nvcc -O2 -arch=sm_87 managed_add.cu -o managed_add
./managed_add
```

실제 출력은 다음과 같았다.

```text
before kernel: 41
after kernel:  42
```

CPU가 쓴 41을 GPU가 읽어 42로 바꿨고, 동기화가 끝난 뒤 CPU가 42를 읽었다. 이 결과로 확인한 것은 그 순차 접근이 정상적으로 동작했다는 사실이다.

같은 장치에서 CUDA 런타임에 장치 속성도 물었다. 장치 속성은 현재 GPU와 소프트웨어 환경이 어떤 기능을 지원하는지 나타내는 값이다. Unified Memory와 직접 관련된 네 값만 추리면 다음과 같다.

```text
device=0 name=Orin cc=8.7 integrated=1
managedMemory=1
concurrentManagedAccess=0
pageableMemoryAccess=0
```

`integrated=1`은 이 GPU가 host memory system과 통합된 GPU임을 나타낸다. Orin의 구조와 함께 보면 CPU와 GPU가 같은 physical SoC DRAM을 사용한다.

`managedMemory=1`은 이 장치가 `cudaMallocManaged`를 지원한다는 뜻이다.

`concurrentManagedAccess=0`은 **limited Unified Memory**라는 뜻이다. 여기서 limited는 GPU 작업 중 CPU가 managed allocation에 동시에 접근하는 방식을 지원하지 않는다는 의미다. 이 예제는 GPU가 끝난 뒤에만 CPU가 `x`를 읽으므로 그 제한을 지킨다. 또한 `cudaDeviceSynchronize()`가 비동기 kernel의 완료를 보장하므로 CPU read와 GPU write가 겹치지 않는다.

`pageableMemoryAccess=0`은 일반적인 `malloc`이나 `new`로 확보한 메모리를 등록 없이 managed allocation처럼 직접 쓸 수 없다는 뜻이다. 따라서 이 예제의 Unified Memory 공간은 `cudaMallocManaged`로 만든다.

따라서 이 환경의 결론은 **shared physical DRAM을 사용하는 limited Unified Memory**다. `cudaMallocManaged`는 CPU용 포인터와 GPU용 포인터를 하나로 줄였고, CPU와 GPU는 그 allocation을 정해진 순서로 사용했다. 전체 장치 출력과 환경 정보는 [Orin observation](/code/cuda-04/orin-jetpack-6.2.2.txt)에 남겼다.

## 참고

- [CUDA Programming Guide: Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html): Unified Memory와 device attribute의 정의.
- [CUDA for Tegra: Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management): Tegra의 shared SoC DRAM과 Unified Memory 동작.
