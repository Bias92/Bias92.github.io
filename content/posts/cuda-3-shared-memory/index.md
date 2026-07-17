---
title: "03 CUDA Shared Memory: Tiling, Bank Conflicts, and Reduction"
date: 2026-07-14
draft: false
tags: ["CUDA", "GPU Programming", "Shared Memory", "Warp Divergence", "Parallel Programming", "Reduction", "Nsight Compute"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "RTX 4060 Ti에서 tiled GEMM, transpose, reduction을 구현하고 측정한다. coalescing sector, bank conflict, warp divergence와 predication, occupancy 한도, reduction 4단계 개선과 CUB 비교."
---

이 글은 shared memory를 쓰는 커널 세 개를 구현하고 측정한다. tiled GEMM, transpose, reduction이다. shared memory는 cache와 달리 적재와 교체를 커널이 직접 관리한다. 그래서 얻는 것(재사용 제어)과 새로 생기는 문제(barrier 규칙, bank conflict, occupancy 감소)를 코드와 Nsight Compute 카운터로 하나씩 확인한다. [CUDA C 글](../cuda-c-basics/)의 roofline과 occupancy 공식을 그대로 이어서 쓴다. 전체 소스와 재현 명령은 글 끝에 있다.

## 측정 조건

| 항목 | 값 |
| --- | --- |
| GPU | RTX 4060 Ti 8GB (AD106, cc 8.9, SM 34개) |
| L2 캐시 | 32MB |
| DRAM | GDDR6, 이론 대역폭 288 GB/s |
| FP32 peak | 약 22 TFLOP/s (공칭 부스트 2.54GHz 기준) |
| 툴체인 | CUDA 13.0, `nvcc -O3 -arch=sm_89`, 호스트 MSVC 19.42 |
| 프로파일러 | Nsight Compute 2025.3.1 |

시간은 `cudaEvent`로 재고 warm-up 후 반복 측정의 median이다(reduction·transpose는 10회 후 50회, GEMM은 5회 후 30회). 표의 시간은 kernel 구간만이다. 결과 검증은 reduction이 CPU double 참조합, GEMM이 임의 64개 원소 재계산, transpose가 임의 1000개 비교고, 실패하면 벤치가 non-zero로 종료한다. ncu 계측 중에는 커널이 느려지므로 시간은 전부 계측 없는 median이고 ncu에서는 카운터만 가져왔다.

표의 GB/s는 논리적으로 읽고 쓴 바이트를 시간으로 나눈 effective bandwidth고, 비교 기준은 이론 피크 288 GB/s다. DRAM이 실제로 옮긴 바이트는 `dram__bytes`로 따로 잰다. 절대값은 세션 클럭 상태에 따라 수 % 흔들리므로 결론은 버전 간 비율에 둔다.

이 카드의 ridge point는 22 TFLOP/s ÷ 288 GB/s ≈ 77 FLOP/B다. A100의 9.75보다 한참 오른쪽이다. 소비자용 카드는 FLOP 대비 대역폭이 약해서 같은 커널이라도 더 깊은 memory-bound에서 출발한다.

## Coalescing

cc 6.0 이상에서 한 warp의 global memory 접근은 active lane들이 건드린 32바이트 sector 수만큼 트랜잭션으로 합쳐진다. full warp가 연속·정렬된 `float` 32개를 읽으면 128바이트 범위라 sector 4개로 끝난다. 시작 주소가 `float` 하나 어긋나면 범위가 sector 경계를 하나 더 넘어 5개가 되고, lane 간격이 32바이트 이상이면 lane마다 다른 sector를 건드려 32개까지 벌어진다.

![Coalescing: 정렬된 연속 접근은 sector 4개, 한 word 어긋나면 5개, 흩어지면 32개가 필요하다](images/coalescing.svg?v=2)

4는 보편적인 합격선이 아니라 "active lane 32개가 `float` 하나씩"이라는 조건에서 나온 값이다. lane 수나 데이터 폭이 바뀌면 최소 sector 수도 바뀌고, 기준은 언제나 실제로 건드린 서로 다른 32바이트 sector 수다. `cudaMalloc`이 돌려준 포인터는 충분히 정렬돼 있지만 offset을 더한 subview는 어긋날 수 있다. 이웃 warp가 앞 warp의 남는 sector를 cache에서 재사용할 수 있으므로 sectors/request가 4에서 5로 늘었다고 시간이 정확히 25% 늘지는 않는다. sectors/request 값은 ncu의 `l1tex__t_sectors...sum ÷ l1tex__t_requests...sum`으로 읽는다. 아래 transpose에서 이 값이 32와 4로 찍힌다.

## Tiled GEMM

행렬곱 C = A×B (N×N)에서 C 원소 하나에 A 한 행과 B 한 열이 필요하다. naive 커널은 thread 하나가 C 원소 하나를 맡고 필요한 값을 매번 global memory에서 읽는다.

```cpp
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}
```

내부 루프 한 바퀴에 thread가 요청하는 게 8바이트, FLOP이 2개다. 이 기준의 arithmetic intensity는 0.25 FLOP/B고, 캐시를 무시하면 global 읽기 요청이 전체 2N³회다. 요청이 그대로 트랜잭션이 되지는 않는다. warp 안에서 같은 주소는 broadcast로 합쳐지고 연속 주소는 coalescing으로 묶여 sector가 되고, 그중 일부만 L2를 지나 DRAM까지 간다. 아래 표들은 이 세 층(요청, sector, DRAM)을 따로 센다.

A의 같은 행은 C의 같은 행 N개 계산에 전부 재사용된다. naive는 그 재사용을 캐시에 맡기고, tiling은 shared memory에서 직접 관리한다. block의 thread들이 A와 B의 T×T 타일을 협동해서 복사하고, 타일 안에서 나올 수 있는 부분곱을 전부 누적한 뒤 다음 타일로 넘어간다.

![Tiling: global에서 shared로 타일 복사 1회, 부분곱 T회 재사용](images/tiling.svg?v=2)

```cpp
#define T 32

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < N / T; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * T + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * T + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < T; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}
```

첫 `__syncthreads()`는 덜 채워진 타일을 읽는 걸 막고, 두 번째는 아직 쓰이는 타일을 다음 반복이 덮는 걸 막는다. barrier에는 규칙이 있다. block의 일부 thread만 early return한 채 `__syncthreads()`를 만나면 실행이 정의되지 않는다. 위 코드는 N이 T의 배수라 전원이 같은 흐름을 지나지만, 임의 크기를 받는 코드는 경계 밖 thread를 return시키는 대신 loop와 barrier에 그대로 참여시키고 load 값만 0으로 채운 뒤 마지막 C store에만 범위 검사를 건다. `__syncwarp`는 warp 하나짜리 보장이라 여러 warp가 타일을 채우는 이 구조를 대체하지 못한다.

트래픽을 요청 기준으로 세면 타일 원소 하나가 T개의 부분곱에 재사용되므로 global 읽기 요청은 2N³에서 2N³/T로 준다. block 하나가 타일 쌍을 N/T번 복사하며 8NT 바이트를 읽고 T²개 thread가 2N FLOP씩 하므로

$$
I_{\text{tiled}} = \frac{2NT^2}{8NT} = \frac{T}{4}\ \text{FLOP/B}
$$

T = 32면 8 FLOP/B다. 타일 로드는 완전히 coalesced라 요청한 바이트와 sector가 운반한 바이트가 같고, sector가 전부 DRAM까지 간다는 최악 가정에서의 상한은 8 × 288 GB/s ≈ 2.3 TFLOP/s다.

N = 2048과 4096에서 측정한 결과다.

| N | 커널 | median | GFLOP/s | DRAM 트래픽 (ncu) |
| --- | --- | --- | --- | --- |
| 2048 | naive | 11.556 ms | 1487 | 75~145 MB (요동) |
| 2048 | tiled T=16 | 8.897 ms | 1931 | 70~150 MB (요동) |
| 2048 | tiled T=32 | 9.044 ms | 1900 | 60~100 MB (요동) |
| 4096 | naive | 119.921 ms | 1146 | 17.95 GB |
| 4096 | tiled T=16 | 85.891 ms | 1600 | 17.9 GB |
| 4096 | tiled T=32 | 82.993 ms | 1656 | 9.42 GB |

| 커널 (N=2048) | global load sectors | sector 트래픽 | naive 대비 |
| --- | --- | --- | --- |
| naive | 1,073,741,824 | 34.4 GB | 1× |
| tiled T=16 | 134,217,728 | 4.29 GB | 8× |
| tiled T=32 | 67,108,864 | 2.15 GB | 16× |

![GEMM N=2048: 요청, sector, DRAM 세 층에서 실측한 트래픽](images/traffic-layers.svg)

sector 절감은 요청 기준 계산(T=32에서 32×)의 절반인 8×와 16×다. block이 `dim3(16,16)`이라 warp가 두 행에 걸치는데, B의 주소에는 row가 들어가지 않아 두 half-warp가 같은 16개 주소를 읽고, A는 행당 한 주소를 half-warp 전체가 broadcast로 받는다. naive도 warp 안에서 이미 2배를 절약한다. sector 기준 naive의 intensity는 34.4GB에 0.5 FLOP/B고, 2N³/T의 32×는 요청 기준 상한이다.

N = 2048의 `dram__bytes`는 실행마다 요동해서 표에 범위로 적었다. 입력 행렬 둘이 합쳐 32MB로 L2 크기와 같아 경계에 걸리고, C 쓰기와 프로파일러 replay가 캐시 상태를 바꾼다. 어느 실행에서도 sector 34.4GB의 0.5% 미만만 DRAM으로 내려갔다. 이 크기에서는 캐시 계층이 naive의 반복 읽기를 거의 다 흡수하고, tiling의 시간 이득도 1.3배에 그친다. 커널 간 비교에는 요동하는 DRAM MB 대신 재현되는 sector 수와 시간을 쓴다.

L2를 넘는 N = 4096(행렬당 64MB)에서는 naive의 DRAM이 17.95GB로 뛰고 tiled T=32(9.42GB)와 1.9배 벌어진다. tiled T=16의 DRAM은 17.9GB로 naive와 거의 같지만 시간은 1.4배 빠르다. T=16은 sector 트래픽이 T=32의 두 배라 DRAM으로 새는 양도 두 배고, 그 값이 naive의 DRAM과 겹친 것이다. 그래도 naive보다 빠른 이유는 sector 층에 있다. naive의 sector 트래픽은 tiled T=16의 8배다. DRAM 바이트가 같아도 L1과 L2 사이에서 오간 양이 다르고 그 차이가 시간으로 나타난다.

tiled의 절대 성능은 낮다. N = 4096 tiled T=32는 DRAM 9.42GB에 2N³ FLOP이므로 실측 DRAM intensity 14.6 FLOP/B, 해당 roofline은 4.2 TFLOP/s인데 실제는 1.66 TFLOP/s다. DRAM 처리율도 113 GB/s로 피크의 39%뿐이다. 병목은 DRAM이 아니라 SM 내부다. thread당 출력이 1개라 FMA마다 shared load가 두 번 붙는 구조가 유력한 원인이고, stall 단위의 분해는 이 글 범위 밖이다. thread 하나가 C 원소 여러 개를 레지스터에 놓고 누적하는 register tiling이 다음 단계며, 그 경로는 Boehm의 워크로그가 naive부터 warptiling까지 다룬다.

## Bank Conflict

shared memory는 32개 bank로 나뉘고 연속된 4바이트 워드가 bank 0, 1, ..., 31, 0, 1, ... 순서로 순환 배정된다. 한 사이클에 32개 lane이 서로 다른 bank를 건드리면 동시에 처리되고, 같은 bank의 서로 다른 주소를 건드리면 직렬화된다. n개가 겹치면 n-way conflict다. 여러 lane이 같은 주소를 읽는 경우는 broadcast라 conflict가 아니다. 같은 주소에 동시에 쓰는 건 합쳐지는 연산이 아니며 어느 lane의 값이 남을지 정해져 있지 않다.

전형적인 conflict 지점이 2차원 타일의 열 방향 접근이고, shared memory transpose가 그 예다.

```cpp
__shared__ float tile[32][32];

tile[threadIdx.y][threadIdx.x] = in[...];   // 행 방향 쓰기: bank 분산
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y];  // 열 방향 읽기: 32-way conflict
```

열 방향 읽기에서 lane들이 읽는 `tile[0][c], tile[1][c], ...`는 32워드 간격이라 전부 bank c에 떨어진다. 해법은 둘이다.

padding. 행 길이를 33으로 늘리면 열 방향 bank가 (r + c) mod 32로 행마다 하나씩 어긋난다.

```cpp
__shared__ float tile[32][33];
```

일반화하면 stride S로 읽을 때 conflict 차수는 gcd(S, 32)다. stride 32는 32-way, 33은 conflict-free, 2와 4는 각각 2-way와 4-way. padding은 stride와 bank 수의 공약수를 1로 만드는 조작이다(단일 `float` load 기준. 64비트나 vector 접근은 lane 하나가 bank 여러 개에 걸친다). 비용은 행당 4바이트 낭비다.

swizzle. 저장 위치의 열 인덱스를 행 번호와 XOR하면 메모리를 더 쓰지 않고 같은 효과가 난다.

```cpp
__shared__ float tile[32][32];

tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[...];   // 쓰기: bank 분산
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];  // 읽기도 분산
```

행 안에서 열들이 XOR로 재배열되면 행 방향 접근도 열 방향 접근도 bank 32개에 정확히 한 번씩 떨어진다. 비용은 인덱스 계산과 타일 폭이 2의 거듭제곱이어야 한다는 제약이다. CUTLASS가 layout 수준으로 일반화해 둔 기법이다.

![Bank conflict: stride 32의 충돌을 padding과 XOR swizzle로 푸는 두 방법](images/bank-conflict.svg?v=3)

배치 자체를 바꾸는 방법도 있다. lane과 행·열의 대응을 뒤집어 shared 접근을 연속으로 만들 수 있지만, 이 transpose에서는 그 순간 global 접근이 다시 strided가 된다. 한쪽의 conflict를 다른 쪽의 uncoalesced access로 옮긴 것이라 해법이 아니다. 교환 범위가 warp 하나 안이고 lane마다 scalar 몇 개뿐이라면 `__shfl_sync()`로 register 값을 직접 넘겨 shared memory를 빼는 방법도 있으나, shuffle은 warp 밖으로 값을 보낼 수 없고 32×32 타일을 보관하지도 못하므로 block 전체가 협동하는 이 transpose를 대체하지 못한다.

4096×4096 float transpose로 네 버전을 측정했다. naive는 shared memory 없이 읽기 coalesced, 쓰기 strided. 나머지는 32×32 타일을 거쳐 양쪽을 coalesced로 만들고 conflict 처리만 다르다.

| 커널 | median | effective GB/s | store sectors/request | shared load conflicts |
| --- | --- | --- | --- | --- |
| naive (strided write) | 1.039 ms | 129.2 | 32.0 | 0 |
| tile[32][32] | 0.605 ms | 221.8 | 4.0 | 16,278,462 |
| tile[32][33] padded | 0.550 ms | 244.2 | 4.0 | 33,035 |
| tile[32][32] swizzled | 0.549 ms | 244.5 | 4.0 | 33,208 |

naive의 store는 request당 sector 32개로 최악이고, 타일을 거치면 4개로 줄며 시간이 1.72배 빨라진다. 남은 격차가 bank conflict다. shared load request 524,288개에 conflict 16,278,462개, request당 약 31개로 거의 모든 load가 32-way로 쪼개진다. padding은 이를 33,035개로, swizzle은 33,208개로 줄인다. 99.8% 감소이며 잔여분은 request당 약 0.063개로 작지만, 이 카운터만으로 그 원인까지 특정할 수는 없다. 두 해법은 시간도 카운터도 동급이고, 차이는 비용의 종류다. padding은 행당 4바이트, swizzle은 XOR 한 번과 2의 거듭제곱 제약이다.

naive가 sector를 8배 쓰고도 1.72배만 느린 이유는 층 구분에 있다. 네 버전의 DRAM 트래픽은 122MB, 119MB, 123MB, 124MB로 사실상 같고, 논리적으로 옮겨야 하는 128MB(읽기 64 + 쓰기 64) 근처다. naive가 낭비한 sector는 DRAM에 도달하기 전에 L2에서 흡수되고, 비용은 DRAM 대역폭이 아니라 L1과 L2 사이의 트랜잭션 수와 지연으로 나타난다. GEMM에서 T=16이 naive와 같은 DRAM을 쓰고도 빨랐던 것과 같은 구도다. conflict 제거의 시간 이득은 10%지만 카운터로는 16백만 대 3만이다. effective bandwidth가 이미 피크 근처(244 GB/s, 85%)라 시간 차이가 작게 보일 뿐이다.

위 GEMM tiled 커널에는 이 문제가 없다. `Bs[k][threadIdx.x]`는 행 방향이라 분산되고 `As[threadIdx.y][k]`는 broadcast다.

## Occupancy와 Block 크기

N=2048에서 T=16과 T=32의 성능은 비슷하지만 SM 상주 조건은 다르다.

| 커널 (N=2048) | achieved occupancy | shared/block | registers/thread |
| --- | --- | --- | --- |
| tiled T=16 (256 threads) | 99.5% | 2.05 KB | 36 |
| tiled T=32 (1024 threads) | 66.7% | 8.19 KB | 36 |

![SM 상주: T=16은 256-thread block 6개가 SM을 채우고, T=32는 1024-thread block 하나만 올라가 512 슬롯이 빈다](images/occupancy-residency.svg)

T=32는 block이 1024 thread인데 cc 8.9의 SM당 상주 thread 한도가 1536이라 block이 SM에 하나만 올라간다. 1024/1536 = 66.7%로 실측과 일치한다. [CUDA C 글](../cuda-c-basics/)의 occupancy 공식에서 thread 한도 항이 바닥난 경우다. 이 사례는 thread 한도지만 커널에 따라 register나 shared memory 항이 먼저 바닥나고, register를 억지로 줄이면 spill이 생길 수 있다. 타일을 키우면 재사용은 늘지만 block이 커져 상주 병렬성이 준다. 재사용을 더 쌓는 길이 block 확대가 아니라 thread당 일 늘리기(register tiling)인 이유가 여기에도 있다.

| 커널 (N=2048) | achieved occupancy | eligible warps/cycle | issue active |
| --- | --- | --- | --- |
| tiled T=16 | 99.45% | 1.26 | 25.38% |
| tiled T=32 | 66.65% | 0.98 | 21.53% |

occupancy가 33%p 낮아졌는데 성능 차이는 그보다 훨씬 작다. T=32는 sector가 절반이라 잃은 병렬성을 일부 상쇄한다. occupancy는 scheduler가 고를 수 있는 warp 수의 상한이지 실행 시간의 비율이 아니고, 보편적인 목표치도 없다. 아래 reduction의 v2는 76%로 제일 낮은데 제일 빠르다. 튜닝 순서는 eligible warp와 issue slot을 먼저 보고, 모자랄 때 무엇이 resident warp를 제한하는지 찾는 쪽이다.

## Warp Divergence

warp는 32개 lane을 한 묶음으로 발행한다. lane 번호를 \(\ell \in \{0,\ldots,31\}\), branch 조건을 \(p_\ell\)이라 하자. 조건이 참인 lane과 거짓인 lane의 집합은

$$
A = \{\ell \mid p_\ell = 1\}, \qquad
B = \{\ell \mid p_\ell = 0\}
$$

이다. \(B\)는 warp에서 \(A\)의 여집합이므로 divergence 조건은

$$
0 < |A| < 32
\quad\Longleftrightarrow\quad
A \neq \varnothing \ \land\ B \neq \varnothing
$$

이다. 소스 코드 수준에서는 warp-nonuniform branch다. 실제 기계어에도 branch가 남으면 warp는 A 경로를 A의 active mask로, B 경로를 B의 active mask로 실행한다.

```cpp
int lane = threadIdx.x & 31;

if (lane < 16)
    A();
else
    B();
```

모든 warp에서 lane 0~15는 A, lane 16~31은 B를 고른다. A를 실행할 때의 mask는 `0x0000ffff`, B에서는 `0xffff0000`이다.

A와 B 경로가 각각 \(n_A\), \(n_B\)개의 warp instruction으로 컴파일됐다고 하자. branch와 reconvergence 명령은 일단 제외한다. 한 경로만 고르면 해당 구간의 발행 수는 \(n_A\), 두 경로가 모두 선택되면

$$
I_{\text{diverged}} = n_A + n_B
$$

다. 증명은 active mask의 정의에서 바로 나온다. A가 비어 있지 않으므로 A 경로의 \(n_A\)개 명령이 한 번씩 발행되고, B도 비어 있지 않으므로 B 경로의 \(n_B\)개 명령이 한 번씩 발행된다. lane 수는 명령의 발행 횟수를 곱하지 않는다. 따라서 16:16과 31:1은 \(n_A\), \(n_B\)가 같다면 warp instruction 발행 수가 같다.

발행된 lane slot 중 실제로 켜진 비율을 \(\eta\)라 두면

$$
\eta =
\frac{|A|n_A + |B|n_B}
     {32(n_A+n_B)}
$$

다. 두 경로 길이가 같아 \(n_A=n_B=n\)이면

$$
\eta =
\frac{(|A|+|B|)n}{64n}
= \frac{32n}{64n}
= \frac{1}{2}
$$

이다. 16:16뿐 아니라 31:1도 두 경로 길이가 같으면 이 구간의 평균 lane utilization은 50%다.

warp 경계에 맞춰 조건을 만들면 다른 warp가 다른 코드를 실행해도 divergence가 아니다.

```cpp
int warp = threadIdx.x >> 5;

if ((warp & 1) == 0)
    A();
else
    B();
```

짝수 warp는 32개 lane이 전부 A, 홀수 warp는 전부 B를 고른다. 각 warp 안에서는 조건이 uniform이다.

\(T_A\)와 \(T_B\)를 각 경로의 warp instruction sequence가 발행되는 시간으로 두자. 다른 warp에 의한 latency hiding과 경로 사이의 memory overlap을 무시하면 두 경로가 active mask를 바꿔 직렬 실행되므로

$$
T_{\text{diverged}}
\approx T_{\text{branch}} + T_A + T_B + T_{\text{reconverge}}
$$

모든 lane이 A를 고르는 warp는

$$
T_{\text{uniform}}
\approx T_{\text{branch}} + T_A
$$

이고, 같은 A 경로를 기준으로 한 추가 시간은

$$
\Delta T
= T_{\text{diverged}} - T_{\text{uniform}}
\approx T_B + T_{\text{reconverge}}
$$

다.

실제 branch가 남고 A와 B가 모두 선택되면 해당 warp는 A 경로의 명령과 B 경로의 명령을 모두 발행한다. 한쪽 경로를 실행하는 동안 다른 쪽 lane은 active mask에서 빠진다. 여기서 늘어나는 것은 warp가 발행해야 할 동적 명령 수다. 고정된 "divergence penalty cycle"이 따로 붙는 구조가 아니며, 실제 시간은 각 경로의 명령 종류, dependency, memory stall, 그리고 다른 warp가 latency를 얼마나 숨기는지에 따라 달라진다.

소스의 `if`가 갈린다고 실제 branch divergence가 반드시 생기는 것도 아니다. 본문이 짧으면 compiler가 branch를 없애고 predicated instruction으로 바꿀 수 있다.

```cpp
float y = x;
if (lane < 16)
    y = 2.0f * x;
```

개념적인 SASS 형태는 아래와 같다. 정확한 opcode와 register는 architecture와 compiler version에 따라 달라진다.

```text
ISETP.LT ... P0, lane, 16
@P0 FMUL  y, x, 2.0
```

이 경우 warp는 A와 B라는 두 program counter로 갈라지지 않는다. `FMUL`은 한 번 발행되고 predicate가 참인 lane만 결과를 쓴다. control-flow divergence는 없지만 predicate가 꺼진 lane은 그 명령에서 유효한 일을 하지 않는다. divergent branch와 predication은 둘 다 active lane utilization을 낮출 수 있지만 같은 현상은 아니다.

위 코드의 `FMUL`에서는 \(|A|=16\)이고 명령이 하나이므로 predicated lane utilization은

$$
\eta_{\text{predicated}} = \frac{|A|}{32} = \frac{16}{32} = \frac{1}{2}
$$

다.

판별은 소스의 `if` 개수가 아니라 생성된 SASS와 instruction별 counter로 한다. SASS에 lane별 branch target이 남았는지 확인하고, Nsight Compute Source page의 `Divergent Branches` (`smsp__branch_targets_threads_divergent`)와 `Avg. Predicated-On Threads Executed`를 해당 instruction에서 본다. kernel 전체 평균은 짧은 divergent 구간을 나머지 full-warp 실행에 섞어 희석할 수 있다.

## Reduction

reduction은 배열 N개를 값 하나로 줄이는 연산이다. 합, 최댓값, 평균이 여기 속하고 softmax의 최댓값과 분모 합, layernorm의 평균과 분산으로 ML 커널 안에 계속 나온다.

트리로 접으면 매 단계 절반의 thread가 두 값을 합치고 critical path 깊이는 log₂N이다. 총 덧셈은 여전히 N-1개다. 병렬화는 일의 양이 아니라 깊이를 줄인다. 대신 단계마다 이전 쓰기가 끝났다는 보장이 필요하고, 이 동기화 비용을 어디까지 줄이느냐가 아래 네 버전의 차이다.

![Reduction: sequential addressing으로 8개가 3단계에 접힌다](images/reduction-tree.svg?v=2)

공통 구조는 multi-pass다. 각 block이 shared memory에서 부분합을 만들고, 부분합 배열에 같은 커널을 다시 돌린다. 2²⁴개(64MiB) 입력에 block 256이면 65,536개 → 256개 → 1개, 세 pass다. 뒤 두 pass의 입력은 전체의 0.4%라 effective bandwidth는 원본 64MiB 기준으로 계산한다.

버전 0, 트리의 직역.

```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        buf[tid] += buf[tid + s];
    __syncthreads();
}
```

느린 요인이 둘이다. 첫째는 흩어진 active lane이다. 조건은 s=1에서 `0x55555555`, s=2에서 `0x11111111`이 되고, 일하는 lane이 16, 8, 4, ...개로 줄어도 해당 warp 명령은 발행된다. compiler가 이 짧은 본문을 predication으로 바꾸면 실제 branch divergence는 없지만 낮은 lane utilization은 그대로다. 둘째는 `%` 연산이다. 제수 `2 * s`가 loop마다 바뀌어 compiler가 상수 bit mask로 접지 못하고, sm_89 SASS에 `IABS → I2F → MUFU.RCP → F2I → IMAD.HI`로 이어지는 나머지 계산이 남는다(`cuobjdump --dump-sass`로 확인). 버전 1의 SASS에는 이 명령 묶음이 없다.

버전 1, sequential addressing.

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        buf[tid] += buf[tid + s];
    __syncthreads();
}
```

두 버전 모두 256개 원소를 여덟 단계로 줄인다. 단계 \(j \in \{0,\ldots,7\}\)에서 덧셈을 수행하는 thread 수는

$$
a_j = \frac{256}{2^{j+1}}
$$

이고, 전체 덧셈 수는

$$
\sum_{j=0}^{7} a_j
= 128 + 64 + \cdots + 1
= 255
$$

다. 두 버전의 산술량은 같다. 차이는 \(a_j\)개의 active thread를 몇 개 warp에 배치하느냐다.

v0는 active thread가 block 전체에 흩어진다. 참인 predicate를 하나 이상 가진 warp 수는

$$
W_j^{(0)} = \min(8, a_j)
$$

이므로

$$
\left(W_j^{(0)}\right)_{j=0}^{7}
= (8,8,8,8,8,4,2,1),
\qquad
\sum_{j=0}^{7} W_j^{(0)} = 47
$$

이다. v1은 active thread를 block 앞에 연속으로 모으므로

$$
W_j^{(1)} = \left\lceil \frac{a_j}{32} \right\rceil
$$

이고

$$
\left(W_j^{(1)}\right)_{j=0}^{7}
= (4,2,1,1,1,1,1,1),
\qquad
\sum_{j=0}^{7} W_j^{(1)} = 12
$$

다. 실제 branch가 남고 body가 \(m\)개의 warp instruction이라면 body 발행량은 각각 \(47m\)과 \(12m\)이다. predication으로 바뀌면 이 계산은 active thread의 공간적 분포만 증명하며, 실제 발행량은 SASS에서 확인해야 한다.

v1의 s=128에서는 앞 warp 4개가 통째로 일하고 s=64에서 2개, s=32에서 1개다. 이 세 단계의 조건은 warp 단위로 uniform이다. s=16부터는 첫 warp 안에서 조건이 갈리지만 active lane이 앞에서부터 연속된다. `buf[tid]`와 `buf[tid + s]`도 연속 주소라 bank conflict가 없고 `%`도 사라진다. barrier 수는 v0와 같은데 1.63배 빠르다. 실제 비율이 \(47/12\)가 아닌 이유는 global load, loop guard, barrier, modulo 계산이 함께 들어가고 branch가 predication으로 바뀔 수도 있기 때문이다.

![v0은 활성 lane이 warp 안에 흩어지고, v1은 warp 단위로 앞에 몰린다](images/reduction-lanes.svg)

버전 2, warp shuffle. s=16부터는 첫 warp 안에서 조건이 갈리고 active lane 수가 절반씩 준다. 그 경계부터는 shared memory와 `__syncthreads()` 없이 레지스터에서 접는다.

```cpp
if (tid < 32) {
    float x = buf[tid] + buf[tid + 32];
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, off);
    if (tid == 0) out[blockIdx.x] = x;
}
```

`__shfl_down_sync`는 warp 안에서 레지스터 값을 lane 사이에 직접 전달한다. 마지막 여섯 단계의 shared 왕복과 block barrier가 빠진다. full mask `0xffffffffu`는 첫 warp 전체가 `tid < 32`를 만족해서 유효한 것이고, 일부 lane만 참여하는 코드라면 `__ballot_sync`로 mask를 만들어 참여 lane 전원이 같은 mask로 같은 intrinsic을 실행해야 한다. source lane이 mask에 없으면 반환값이 정의되지 않는다.

버전 3, block당 atomic 한 번. multi-pass 대신 각 block의 lane 0이 `atomicAdd(out, x)`를 한 번 실행한다. atomic은 같은 주소에 대해 직렬화된다. 원소마다 atomic이면 같은 주소에 16,777,216번 접근하지만 block reduction 뒤에는 65,536번이고, 이 벤치에서는 64MiB load 뒤에 가려진다. 입력 크기, block 수, 아키텍처의 atomic 처리율이 바뀌면 결과도 달라질 수 있다. 버전 3은 실행 전에 결과 scalar를 0으로 만드는 4바이트 `cudaMemset`이 필요하다. 이는 측정 구간 밖이므로 v3의 end-to-end 비용에는 초기화가 추가되며, 표는 kernel 구간만 비교한다. float atomic은 덧셈 순서가 실행마다 달라 비트 재현성이 없다.

2²⁴개(64MB) 합산 측정이다. CUB의 `DeviceReduce::Sum`을 기준선으로 넣었다.

| 버전 | median | effective GB/s | 이론 피크 대비 | achieved occupancy |
| --- | --- | --- | --- | --- |
| v0 interleaved + modulo | 0.625 ms | 107.3 | 37% | 93.6% |
| v1 sequential | 0.383 ms | 175.2 | 61% | 88.9% |
| v2 + warp shuffle | 0.256 ms | 262.1 | 91% | 76.3% |
| v3 + block당 atomic 1회 | 0.252 ms | 266.4 | 92% | 78.1% |
| CUB DeviceReduce | 0.254 ms | 264.3 | 92% | - |

v2부터는 effective bandwidth가 이론 피크의 90%대라 더 줄일 게 거의 없다. 원소당 FLOP이 1개뿐인 순수 대역폭 문제고, 잘 쓴 reduction의 상한은 memcpy 속도다. v3는 CUB와 같은 구간에 온다. CUB는 임의 타입과 크기에서 저 성능을 내는 라이브러리고 v3는 float 큰 배열 하나에 고정된 커널이니, 같은 조건에서 같은 대역폭 구간이라는 비교로 읽어야 한다.

카운터 관련 주의 하나. `smsp__average_thread_inst_executed_per_inst_executed`는 v0 = 31.93, v1 = 31.81로 거의 같다. 이 값은 predicate의 참·거짓과 무관하게 실행된 thread instruction을 세므로 실제 branch divergence를 입증하는 counter가 아니다. 실행의 대부분이 full-warp global load라는 점도 짧은 트리 구간을 kernel 평균에서 희석한다. 실제 branch 여부는 위에서 설명한 instruction별 divergent target counter와 SASS로 분리하고, 이 실험의 v0 → v1 결론은 시간과 `%` 명령 묶음의 제거에 둔다.

커널 경계를 넘는 fusion과 stream을 이용한 전송·연산 겹치기는 다음 글에서, Tensor Core와 CUTLASS는 그다음 글에서 다룬다.

## 소스 코드

세 벤치의 전체 소스는 저장소에 있다. [gemm_bench.cu](/code/cuda-03/gemm_bench.cu), [transpose_bench.cu](/code/cuda-03/transpose_bench.cu), [reduce_bench.cu](/code/cuda-03/reduce_bench.cu). 결과 검증에 실패하면 non-zero로 종료한다.

```bash
nvcc -O3 -arch=sm_89 -o gemm_bench gemm_bench.cu
nvcc -O3 -arch=sm_89 -o transpose_bench transpose_bench.cu
nvcc -O3 -arch=sm_89 -std=c++17 -o reduce_bench reduce_bench.cu   # CUB 사용

./gemm_bench          # N=2048
./gemm_bench 4096
./transpose_bench
./reduce_bench
```

본문의 카운터는 Nsight Compute CLI로 수집했다. 커널별 전체 ncu 명령과 알려진 측정 불안정성(N=2048 `dram__bytes` 요동)은 [README](/code/cuda-03/README.md)에 있다.

## 참고

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, shared memory, bank conflict, occupancy, branch predication의 기준 문서
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): SIMT divergence, 동기화, atomics, warp intrinsic의 정확한 의미
- [Mark Harris, An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/): coalescing, shared tile, padding을 한 실험으로 보여주는 표준 사례
- [Andreas Holt, Shared-Memory Tiled Matrix Multiplication](https://andreasholt.com/posts/shared-tiled-matmul/): tiled GEMM의 그림과 경계 처리까지 붙인 설명
- [Lei Mao, CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/): bank 주소 매핑의 상세
- [Lei Mao, CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/): swizzle 주소 매핑의 상세
- [Fabian Schütze, Visualizing Bank Conflicts](https://fabianschuetze.github.io/bankconflictscuda.html): 현대 아키텍처의 bank 동작 보충
- [Mark Harris, Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf): reduction을 일곱 단계로 개선하는 고전. 오래된 자료라 warp-synchronous 코드를 그대로 복사하면 안 된다
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/): shuffle과 계층적 atomic. 코드는 현대식 `__shfl_down_sync()`로 바꿔 읽어야 한다
- [Lei Mao, CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/): batched reduction 구현과 실행 결과 중심의 정리
- [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html): threadblock/warp/thread tiling과 register 재사용, double buffering으로 이어지는 상위 레퍼런스
- [Simon Boehm, How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM): 이 글이 멈춘 지점(register tiling)부터 warptiling까지 가는 워크로그
- [CUB](https://nvidia.github.io/cccl/cub/): 직접 짠 reduction과 비교해 볼 production 구현. `WarpReduce → BlockReduce → DeviceReduce` 계층
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/): 이 글의 카운터들이 정확히 뭘 세는지
