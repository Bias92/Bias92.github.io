---
title: "00 GPU Architecture Primer: The Tesla Foundation"
date: 2026-05-15
draft: false
tags: ["CUDA", "GPU Architecture", "Tesla", "SIMT", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "CUDA C 이전: 2008 IEEE Micro 논문의 Tesla 통합 아키텍처(G80/GT200). 그래픽에서 연산으로 이어지는 데이터 경로(Input Assembler, work distribution, SPA, TPC, SM/SP, SFU/LSU, ROP, DRAM), warp와 SIMT가 하드웨어에서 태어난 지점, clock domain, 그리고 각 구조가 어떻게 CUDA 용어가 되는지."
---

> Source: Lindholm, Nickolls, Oberman, Montrym, *"NVIDIA Tesla: A Unified Graphics and Computing Architecture,"* IEEE Micro 28(2), 2008.

## 동기: CUDA 이전의 하드웨어

[CUDA C 글](../cuda-c-basics/)은 block, warp, SM, occupancy, memory coalescing을 마치 언어 기능처럼 다룬다. 아니다. 이건 NVIDIA가 2006년 11월 G80(GeForce 8800 GTX)로 내놓고 2008년 GT200(GTX 280)으로 다듬은 하드웨어 아키텍처의, 소프트웨어에서 보이는 이름들이다. NVIDIA는 그 아키텍처를 Tesla라 부르고, 근거는 위의 IEEE Micro 논문이다.

이 계층을 건너뛰고 CUDA C를 배우면 `warp = 32`나 "접근을 coalesce해라"는 외워야 할 규칙이 된다. 하드웨어를 먼저 보면 그건 결과다. 이 글은 Tesla 데이터 경로를 위에서 아래로 한 번 훑어서, [CUDA C 글](../cuda-c-basics/)의 CUDA 추상들이 물리적인 무언가 위에 얹히게 만든다.

먼저 한 가지 단서: 여기서 Tesla(G80/GT200)는 역사적 기준점이지 현대 GPU의 스냅샷이 아니다. SM은 2006년 이후 여러 번 다시 지어졌다. SM당 FP32 lane이 8개에서 128개로 늘었고, warp scheduler 하나가 넷이 됐고, shared memory가 16KB에서 200KB 넘게 커졌고, 그때 없던 유닛들(범용 L1 데이터 캐시, tensor core, async copy)이 붙었다. 살아남은 건 이름의 족보와 멘탈 모델이다. SPA → TPC → SM → SP가 CUDA의 block, warp, SM 스케줄링 뒤에 있는 계보다. 아래 숫자들은 오늘의 스펙시트가 아니라 2008년 스냅샷으로 읽고, 개념을 오래 가는 부분으로 받아들여라.

## 통합 셰이더 아키텍처

Tesla 이전 GPU는 특화된 프로세서들이 고정된 파이프라인으로 늘어선 구조였다. vertex shader, 그다음 rasterization, 그다음 pixel(fragment) shader. 각각 명령어 집합도 실리콘도 다른 별개 유닛이다. vertex 작업과 pixel 작업의 비율은 설계 시점에 고정되므로, vertex가 몰리거나 pixel이 몰리는 프레임에서는 칩 절반이 논다.

Tesla는 이걸 갈아엎었다. 별개의 shader 단계들을, 모든 shader 타입이 시분할로 나눠 쓰는 동일한 프로그래머블 프로세서 하나의 배열로 바꿨다. vertex, geometry, pixel이 전부 같은 코어에서 돌고, 하드웨어는 그저 일이 있는 단계로 배열을 다시 겨눈다. NVIDIA는 이 배열을 SPA(Streaming Processor Array)라 부른다.

이 결정 하나가 GPGPU를 만들었다. 자기 메모리 시스템을 가진 범용 프로그래머블 프로세서 배열이 생기면, 그걸 비(非)그래픽 연산에 겨누는 건 노출하기 나름이고, 그게 바로 CUDA가 하는 일이다. CUDA C는 그래픽 칩에 덧붙인 게 아니라, 같은 통합 배열에 대한 두 번째 프런트엔드다.

![NVIDIA Tesla (G80) unified architecture](./images/tesla1.svg?v=2)
*G80 기준: 8 TPC × 2 SM × 8 SP = 128 SP, DRAM 파티션 6개. SPA는 이 8개 TPC 배열이고, compute work distribution(마젠타)이 CUDA가 모는 경로다.*

## 데이터 경로

메모리에서 출발해 칩을 통과하고 다시 메모리로 돌아오는 일감을 따라가 보자. 그래픽 모드의 경로는 이렇다.

1. **Input Assembler.** DRAM에서 vertex 인덱스와 속성을 읽어 primitive(점, 선, 삼각형)로 조립한다. 정문이다.
2. **Work distribution.** Tesla에는 vertex, pixel, compute 각각의 분배기가 따로 있다. 각 분배기가 일감 묶음을 SPA에 넘기고 프로세서들 사이에서 load-balance한다. compute 모드에서는 *compute work distribution* 유닛이 thread block을 SM에 하나씩, SM이 비는 대로 넘기는 부분이다. 이 load-balancer가 CUDA transparent scalability의 물리적 근원이다.
3. **SPA.** 프로세서 배열이 shader(또는 커널)를 실행한다. 그래픽에선 vertex shading을 하고 나중에 pixel shading을, CUDA에선 커널을 돌린다. 연산 엔진이고 나머지는 전부 여기에 먹이를 주거나 결과를 빼낸다.
4. **Setup / raster (그래픽 전용).** vertex 작업과 pixel 작업 사이에서 고정 기능 유닛이 clip, 삼각형 setup, rasterize를 해서 fragment를 만들고, pixel 분배기가 그걸 다시 SPA로 넣는다.
5. **ROP (Raster Operations Processor).** pixel shading 뒤에 ROP가 고정 기능 백엔드를 처리한다. depth·stencil 테스트, color blend, antialiasing, 그리고 framebuffer로의 최종 쓰기다. ROP 하나는 DRAM 파티션 하나에 묶여 있다.
6. **DRAM.** 모든 게 시작하고 끝나는 메모리 파티션들.

CUDA에서는 그래픽 전용 단계(setup, raster, ROP)가 놀고, 경로가 이렇게 줄어든다. host가 grid를 launch하고, compute work distributor가 block을 SM들에 뿌리고, SPA가 커널을 돌리고, load/store가 DRAM과 데이터를 주고받는다. 같은 실리콘, 켜지는 단계만 적다.

## 연산 계층: SPA → TPC → SM → SP

SPA는 코어가 평평하게 깔린 풀이 아니다. 3단 계층이고, 각 단이 CUDA 개념에 대응된다.

- **TPC (Texture / Processor Cluster).** SPA는 TPC들로 나뉜다. TPC 하나는 texture 유닛과, 그걸 공유하는 소수의 SM을 묶는다. G80은 SM 2개짜리 TPC 8개, GT200은 SM 3개짜리 TPC 10개다.
- **SM (Streaming Multiprocessor).** 실제로 thread를 돌리는 유닛. SM 하나에 SP 8개, SFU 2개, multithreaded 명령 fetch/issue 유닛, register file, 그리고 16KB shared memory가 있다. CUDA thread block이 내려앉아 머무는 곳이 SM이다.
- **SP (Streaming Processor).** thread 하나의 부동소수점·정수 연산(주로 MAD, 곱셈-덧셈)을 실행하는 scalar ALU. SM당 8개. 훗날 마케팅이 "CUDA core"로 개명한 유닛이다.

총합은 계층에서 바로 떨어진다.

$$
\text{G80: } 8\ \text{TPC} \times 2\ \text{SM} \times 8\ \text{SP} = 128\ \text{SP}
\qquad
\text{GT200: } 10 \times 3 \times 8 = 240\ \text{SP}
$$

SM당 실행 유닛 2종이 CUDA에서 더 중요하다.

- **SFU (Special Function Unit).** SM당 2개. transcendental(reciprocal, reciprocal-sqrt, sin, cos, log, exp)을 계산하고, 그래픽에선 pixel 속성을 보간한다. CUDA 커널이 `__sinf`나 `rsqrtf`를 부르면 이 유닛이다.
- **LSU (Load/Store Unit).** global·local memory로의 load/store를 메모리 파이프라인을 통해 발행하는 경로. warp 하나가 이걸 어떻게 쓰느냐가 [CUDA C 글](../cuda-c-basics/) coalescing의 전부다.

## SIMT와 warp

CUDA 추상이 하드웨어로 문자 그대로 정의되는 지점이 여기다. SM의 명령 유닛은 thread를 하나씩 추적하지 않는다. 32개씩 묶은 warp 단위로 생성·관리·스케줄·실행한다. Tesla 논문이 이걸 SIMT(Single-Instruction, Multiple-Thread)라 명명했다. SM이 warp 하나에 명령 하나를 issue하면 32 thread가 전부 그걸 실행하되, 각자 자기 데이터와 자기 레지스터로 실행한다.

왜 32이고, 왜 warp 명령 하나에 한 사이클 넘게 걸릴까? lane을 세어 보면 된다. SM에는 SP가 8개인데 warp는 32 thread이므로, SM은 warp를 8개 SP 위에 빠른 사이클 4개에 걸쳐 흘려보낸다.

$$
\frac{32\ \text{threads/warp}}{8\ \text{SP}} = 4\ \text{shader clocks per warp instruction}
$$

즉 물리적 SIMD 폭은 8인데 프로그래머가 보는 아키텍처 폭은 32다. NVIDIA는 여기서 warp를 32로 못박고 이후로 한 번도 안 바꿨다. 그래서 한 세대 뒤 CUDA 코드도 여전히 32를 가정한다. warp가 분기하면(divergence) 그 분기를 안 타는 thread는 해당 사이클 동안 마스킹되는데, 그게 CUDA 글이 말한 비용이다.

thread도 공짜로 떠다니지 않는다. SM마다 register file과 16KB shared memory가 고정돼 있고, 상주하는 warp마다 자기 레지스터와 shared memory를 그 풀에서 잘라 간다. G80 SM은 한 번에 최대 24 warp(768 thread)를 상주시키는데, 실제로 몇 개가 들어가느냐는 thread 하나가 얼마나 탐욕스러운지에 달렸다. 상주 warp 대 thread당 자원의 이 맞바꿈이 바로 occupancy이고, 이 풀들이 SM 위의 유한한 하드웨어라서 존재한다.

![SPA to TPC to SM to SP hierarchy](./images/tesla2.svg?v=1)

## Clock domain

Tesla GPU는 한 주파수로 안 돈다. 별개의 clock domain들이 있고, 이걸 헷갈리면 어떤 성능 추정도 어긋난다.

- core(graphics) clock은 프런트엔드, setup, raster, ROP를 돌린다.
- shader clock은 SP를 돌리고, core clock보다 훨씬 빠르다. 8800 GTX는 core가 575MHz인데 shader는 1.35GHz로, 약 2.35배다.
- memory clock은 또 별개로 GDDR3 인터페이스를 돌린다.

shader를 일부러 뜨겁게 돌린다. throughput이 SP 배열에서 나오니까 공정이 허락하는 만큼 높게 클럭하고 나머지 칩은 느리고 시원하게 둔다. FLOP 추정에 core clock이 아니라 shader clock을 쓰는 이유가 이것이다. 8800 GTX에서 SP 하나가 shader clock당 MAD 하나(2 FLOP)를 한다고 세면,

$$
128\ \text{SP} \times 1.35\ \text{GHz} \times 2\ \text{FLOP} \approx 346\ \text{GFLOP/s}
$$

NVIDIA는 더 높은 수치(518 GFLOP/s)를 제시했는데, 같은 사이클에 SFU가 co-issue할 수 있는 MUL까지 세서다. 실제 커널은 그걸 좀처럼 지속하지 못한다. 마케팅 피크와 달성 가능한 피크 사이의 이 간극은 습관으로 새겨둘 만하다. 어떤 숫자든 어느 clock과 어떤 명령 조합을 가정한 건지 항상 물어라.

## 메모리 서브시스템: DRAM 파티션과 coalescing

Tesla의 DRAM은 하나의 덩어리가 아니다. 독립된 파티션들로 쪼개져 있고, 각 파티션은 자기 memory controller와 자기 ROP를 갖는다. G80은 64-bit짜리 파티션 6개(합쳐 384-bit 버스), GT200은 8개(512-bit)다. 주소는 파티션들에 걸쳐 interleave돼서, 순차 메모리가 모든 controller를 병렬로 훑고 지나가고, 총 대역폭은 파티션들의 합이다.

8800 GTX 기준, 384-bit 버스에 GDDR3 900MHz(double data rate, 핀당 1.8 Gb/s)면,

$$
\frac{384\ \text{bit}}{8} \times 1.8 \times 10^{9}\ \text{s}^{-1} = 48\ \text{B} \times 1.8\ \text{GT/s} \approx 86.4\ \text{GB/s}
$$

이 파티션·interleave·넓은 메모리가 CUDA에서 coalescing이 규칙으로 존재하는 이유다. warp의 32 lane이 load를 발행하면 LSU가 그걸 파티션들이 처리할 메모리 트랜잭션으로 바꾼다. 32개 주소가 연속이면 몇 개의 넓은 트랜잭션으로 묶여 controller들에 퍼지고 버스가 꽉 찬다. 흩어져 있으면 주소마다 자기 트랜잭션을 끌고 오고, 가져온 라인 대부분이 버려진다. CUDA 레벨의 "warp 접근을 연속으로 만들어라"는 조언은, "파티션 메모리 시스템이 만들어진 목적인 넓고 정렬된 트랜잭션을 먹여라"일 뿐이다.

## 계승과 변화

위는 전부 2008년 칩이다. A100(2020)이나 H100(2022)의 SM을 열면 숫자는 못 알아볼 정도지만 뼈대는 같다. 개념은 안정적이고 규모만 요동친다는 이 분리가, Tesla를 읽을 가치의 전부다.

| | G80 (2006) | A100 (2020) | H100 (2022) |
| --- | --- | --- | --- |
| SM당 FP32 lane | 8 | 64 | 128 |
| SM당 warp scheduler | 1 | 4 | 4 |
| warp 크기 | 32 | 32 | 32 |
| SM당 shared memory | 16 KB | 최대 164 KB | 최대 228 KB |
| SM당 32-bit 레지스터 | 8,192 | 65,536 | 65,536 |
| 이후 추가 | 기준선 | L1 데이터 캐시, tensor core, ITS, async copy | + TMA, thread block cluster |

안 바뀐 것: warp는 여전히 32 thread, block은 여전히 SM 하나에서 돌고, SM은 여전히 warp를 스케줄해서 그 사이를 오가며 지연을 숨기고, shared memory는 여전히 SM 위에 있고, global memory는 여전히 파티션돼 있어서 coalesce된 접근을 원한다. 이게 CUDA가 올라선 불변식이다. 바뀐 것: SM이 넓어지고(lane·scheduler 증가), 깊어지고(범용 L1 데이터 캐시, 큰 register file), 특수화됐다(tensor core, TMA). 그러니 모델과 어휘는 Tesla로 잡고, 최적화 대상이 될 숫자는 현재 아키텍처 whitepaper로 잡아라.

## Tesla 하드웨어 → CUDA 소프트웨어

CUDA C 글의 모든 추상은 이 칩 위 무언가의 이름이다.

| Tesla 하드웨어 | CUDA C 개념 |
| --- | --- |
| Compute work distributor | grid의 block이 SM들에 퍼지는 방식 (transparent scalability) |
| SM | thread block 하나가 처음부터 끝까지 도는 곳 |
| Warp (32 thread, 8 SP에 4 clock) | SIMT 실행·스케줄 단위 |
| SP (scalar MAD ALU) | "CUDA core" |
| SFU | `__sinf`, `rsqrtf` 등 intrinsic |
| SM당 register file + 16KB shared memory | occupancy가 맞바꾸는 자원 |
| Shared memory | `__shared__` |
| LSU + interleave된 DRAM 파티션 | coalescing과 대역폭이 중요한 이유 |

이 글 뒤에 CUDA C 글을 읽으면 추상이 더는 임의적이지 않다. `warp = 32`는 SM이 32 thread를 8 SP에 흘리는 것이고, occupancy는 SM의 유한한 register file과 shared memory이며, coalescing은 넓은 트랜잭션을 원하는 DRAM 파티션들이다. CUDA는 이걸 발명한 게 아니라 노출했다.

## 참고

- Lindholm, Nickolls, Oberman, Montrym, [*"NVIDIA Tesla: A Unified Graphics and Computing Architecture"*](https://ieeexplore.ieee.org/document/4523358): SPA/TPC/SM/SP 계층, SIMT, 그래픽 데이터 경로의 1차 출처.
- [NVIDIA GeForce 8800 GPU Architecture Technical Brief](https://www.nvidia.com/): G80 클럭, 파티션, register/shared memory 크기.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): 소프트웨어 모델이 현재 아키텍처 하드웨어에 어떻게 매핑되는지.
- [NVIDIA Hopper (H100) Architecture](https://www.nvidia.com/en-us/data-center/h100/): Tesla 계보가 자라난 현대 SM(128 FP32 lane, 4세대 tensor core, TMA).
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, occupancy, pinned memory, host-device 전송을 오늘날 하드웨어에 적용.
