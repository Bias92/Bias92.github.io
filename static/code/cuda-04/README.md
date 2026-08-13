# CUDA 04 examples

두 소스는 서로 다른 목적의 독립 실행 파일이다.

## Files

| 파일 | 역할 | 출력의 성격 |
|---|---|---|
| `managed_add.cu` | 본문의 CPU 41 → GPU +1 → CPU 42 최소 예제 | 정답 확인용, benchmark 아님 |
| `prefetch_demo.cu` | device attributes, prefetch, range attributes 진단 | 환경 의존 진단용, 성능표용 실측 아님 |

## Build

### Windows — Git Bash / RTX 4060 Ti

Visual Studio x64 Developer 환경을 먼저 초기화한 Git Bash에서 실행한다. 03 글과 같은 CUDA 13.0, MSVC 19.42 환경이며, MSVC minor version이 다르면 `CCBIN`을 설치된 `Hostx64/x64` 경로로 바꾼다.

```bash
CCBIN="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"

nvcc -O2 -arch=sm_89 -ccbin "$CCBIN" -Xcompiler -wd4819 \
  -o managed_add.exe managed_add.cu

nvcc -O2 -arch=sm_89 -std=c++17 -ccbin "$CCBIN" -Xcompiler -wd4819 \
  -o prefetch_demo.exe prefetch_demo.cu
```

### Linux

RTX 4060 Ti는 `sm_89`, Jetson AGX Orin은 `sm_87`을 사용한다.

```bash
CUDA_TARGET_ARCH=sm_89

nvcc -O2 -arch="$CUDA_TARGET_ARCH" \
  -o managed_add managed_add.cu

nvcc -O2 -arch="$CUDA_TARGET_ARCH" -std=c++17 \
  -o prefetch_demo prefetch_demo.cu
```

## Run

```bash
./managed_add.exe
./prefetch_demo.exe              # 기본값: 64M floats, 배열당 256 MiB
./prefetch_demo.exe 134217728    # N override
```

Linux에서는 `.exe`를 뺀 이름으로 실행한다.

`managed_add`의 예상 정답은 고정이다.

```text
before kernel: 41
after kernel:  42
```

`prefetch_demo`의 첫 줄은 `CUDART_VERSION`, `managedMemory`, `concurrentManagedAccess`, `pageableMemoryAccess`, `usesHostPageTables`를 출력한다. 값은 실행 환경에 따라 달라진다. 끝까지 실행되는 환경의 correctness gate는 `checksum error = 0`이다.

## CUDA 12.2–12.x / 13.x API

CUDA 13.0에서 unsuffixed `cudaMemPrefetchAsync`와 `cudaMemAdvise`의 location 인자가 `int device`에서 `cudaMemLocation`으로 바뀌었다. `cudaMemPrefetchAsync`에는 현재 0이어야 하는 `flags`도 추가됐다. `prefetch_demo.cu`는 `CUDART_VERSION`으로 두 API를 분기한다.

```cpp
#if CUDART_VERSION >= 13000
cudaMemLocation gpu{};
gpu.type = cudaMemLocationTypeDevice;
gpu.id = dev;
cudaMemPrefetchAsync(ptr, bytes, gpu, 0, stream);
#else
cudaMemPrefetchAsync(ptr, bytes, dev, stream);
#endif
```

GPU destination prefetch는 destination GPU와 stream-associated device 모두 `concurrentManagedAccess != 0`을 요구한다. GPU를 location으로 지정하는 `cudaMemAdviseSetPreferredLocation`·`cudaMemAdviseSetAccessedBy`도 target device에서 같은 조건을 요구한다. `prefetch_demo`는 이 조건을 미리 건너뛰지 않고 limited 환경의 API 실패를 출력한다. native Windows와 WSL, 일부 Tegra에서 그 실패는 build 오류가 아니라 capability 차이다. kernel-only time은 warm-up 뒤 반복 median이 아니므로 benchmark 수치로 사용하지 않는다.

## Optional: Linux Workstation fault trace

```bash
nsys profile --trace=cuda \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  -o managed_add ./managed_add

nsys stats managed_add.nsys-rep
```

fault trace는 overhead가 크며 Embedded Platforms Edition에서는 해당 두 옵션을 지원하지 않는다.
