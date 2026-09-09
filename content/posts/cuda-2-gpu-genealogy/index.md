---
title: "01 NVIDIA GPU Architecture Genealogy: Tesla to Rubin"
date: 2026-05-22
draft: false
tags: ["CUDA", "GPU Architecture", "Tensor Core", "NVIDIA", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "Tesla(2006)부터 Rubin(2026)까지 NVIDIA GPU의 계보를 따라가요. SIMT·warp=32·SM당 block은 유지하면서 어떤 전용 연산기를 더했는지, Tensor Core 다섯 세대와 consumer·datacenter 라인의 분기를 살펴보고 Rubin GPU와 Vera Rubin platform도 구분해볼게요."
---

> NVIDIA의 아키텍처 whitepaper[^genealogy-whitepaper]와 공식 제품 페이지를 1차 자료로 삼았어요. 세대별 흐름과 microarchitecture 해석에는[^genealogy-microarchitecture] Fabien Sanglard, Chips and Cheese, SemiAnalysis를 참고했고요. 전체 출처는 글 끝에 모아두었답니다.

## 개요

안녕하세요ㅎㅎ 오늘은 NVIDIA GPU[^genealogy-gpu] 이름들을 조금 길게 따라가보려고 해요. 2006년 Tesla에서 2026년 Rubin까지, 20년 동안 아키텍처가 십수 번 바뀌었거든요. 이름만 봐도 벌써 제법 많죠.. 어느 세대가 무엇을 왜 바꿨는지 하나씩 연결해두면, 낯선 제품을 만났을 때도 타임라인에서 자리를 찾기 쉬워요.

![눈을 반짝이며 기대하는 문](/images/naver-moon/moon-4.png)

출발점은 [이전 글](../cuda-0-gpu-architecture/)[^genealogy-cuda]에서 본 2006년 Tesla 칩이에요. 그 구조를 기준으로 놓고, 다음 세대에서는 무엇이 달라졌는지 살펴볼게요.

![NVIDIA GPU architecture family tree, Tesla to Rubin](./images/timeline.svg?v=1)
*Pascal까지는 하나의 줄기를 공유하고, Volta부터 위쪽 datacenter 라인과 아래쪽 graphics 라인으로[^genealogy-lines] 갈라져요.*

먼저 **계속 유지된 구조를 알아두고, 작업의 종류와 규모가 달라질 때 무엇을 추가했는지** 보시면 좋아요. 세대 이름을 하나씩 외울 때보다 연결되는 부분이 많답니다.

이어지는 토대는 실행 모델과 메모리 계층이에요. 명령 하나를 thread 32개 묶음인 warp[^genealogy-thread]에 실행하는 SIMT(Single Instruction, Multiple Threads), 하나의 SM 위에서 끝까지 실행되는 thread block[^genealogy-block], register → shared memory → global DRAM[^genealogy-memory]으로 이어지는 메모리 계층을 말해요. SM은 Streaming Multiprocessor의 약자로, 연산 유닛·스케줄러·shared memory[^genealogy-scheduler]를 묶어놓은 GPU의 기본 구성 단위고요. [CUDA C 글](../cuda-c-basics/)에서 익힌 이 구조는 G80부터 Rubin까지 이어져요. 오래전에 작성한 CUDA 코드도 지금 GPU를 대상으로 컴파일할[^genealogy-compile] 수 있는 기반이 여기에 있는 거예요.

그 위의 변화는 먼저 **워크로드의 이동**을 따라갔어요. GPU를 많이 쓰는 작업이 그래픽에서 AI로 옮겨가면서, SM의 범용 코어 옆에 특정 연산을 맡는 유닛이 더해졌거든요. Tensor Core, RT Core, Transformer Engine[^genealogy-specialized]이 차례로 등장해요. 또 하나는 **규모의 압력**이에요. 다이 하나로[^genealogy-die] 수요를 감당하기 어려워지니 설계 단위가 칩 하나에서 다이 2개로, 다시 랙 전체로[^genealogy-rack] 커졌답니다. 범용 SM을 바탕으로 전용 accelerator를 더하고 전체 규모도 키워온 흐름을 볼 거예요.

여기서는 주로 SM을 확대해서 비교할게요. CUDA 프로그램의 block과 warp가 배정되고 스케줄되는 곳이니까요. L2 캐시, memory controller[^genealogy-cache-controller], ROP, copy engine, host interface, fabric[^genealogy-peripherals]도 성능과 시스템 설계에 중요해요. 그중 프로그래머가 접하는 warp 실행, register, shared memory, Tensor Core, RT Core, TMA/TMEM의[^genealogy-tma-tmem] 변화를 SM 중심으로 놓으면 비교하기 편하답니다.

세대별로 SM 도식과 제원표도 붙여두었어요. **굵은 값은 같은 라인의 직전 세대와 달라진 항목**이에요. 수치는 NVIDIA whitepaper를 기준으로 하고, 다른 자료를 사용한 곳은 따로 표시할게요. 숫자가 많으니 바뀐 부분부터 보셔도 괜찮아요~

![Anatomy of a GPU die, where the SM sits](./images/gpu-anatomy.svg?v=1)
*GPU 다이에서는 SM 배열 주변에 L2·memory controller·DRAM 연결, graphics 전용 고정기능과 host/fabric 인터페이스가 자리해요. 아래 세대별 도식은 그중 SM 하나를 확대해서 보여준답니다.*

## Tesla (2006, G80)

첫 번째는 Tesla예요. 이전 GPU에서는 vertex 처리와 pixel 처리를[^genealogy-graphics] 각각 전담하는 고정 파이프라인이[^genealogy-pipeline] 있었어요. Tesla는 이를 하나의 프로그래머블 코어 배열로 통합했답니다. 같은 코어에 그래픽 외의 계산도 맡길 수 있게 되면서 CUDA라는 프로그래밍 모델이 가능해졌어요.

이때는 90nm 공정에[^genealogy-process] SM당 scalar processor(SP) 8개[^genealogy-sp], warp scheduler 1개였어요. 요즘 숫자를 보다가 돌아오면 소박해 보이기도 하죠ㅎㅎ 그래도 이후 계보가 이어지는 출발점이에요.

![Tesla SM component diagram](./images/sm-tesla.svg?v=1)
*Tesla SM(G80)에는 scalar SP 8개, scheduler 1개, shared memory 16 KB가 있어요. 여기서부터 비교해볼게요.*

| 칩 | Partition[^genealogy-table] | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G80 | 단일 | 8 (SP) | — | — | 1 | 16 KB (shared 전용, L1 없음) | 32 KB |

## Fermi (2010, GF100)

GPU에 계산을 맡기기 시작했으니, 이제 수치 라이브러리를 올려서 본격적으로 사용할 기반도 필요하겠죠? Fermi에서는 범용 L1 데이터 캐시와 L2 캐시, ECC 메모리[^genealogy-ecc], fused multiply-add(FMA)[^genealogy-fma], IEEE 표준을 완전히 따르는[^genealogy-ieee] 배정밀도(FP64) 연산, C++ 지원이 갖춰졌어요. 그래픽 칩을 연산용 프로그래밍 대상으로 쓰기 위해 필요한 것들이 들어온 거예요.

SM도 CUDA 코어 32개와 warp scheduler 2개로 커졌고, texture unit[^genealogy-texture]이 SM 안으로 들어왔답니다. Tesla에서 열린 GPU 연산의 가능성을 실제 수치 계산용으로 다듬은 세대로 보시면 돼요.

![Fermi SM component diagram](./images/sm-fermi.svg?v=1)
*Fermi SM(GF100)은 코어 32개와 scheduler 2개를 갖춰요. 이 계보에서 L1 데이터 캐시가 처음 들어온 세대랍니다.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GF100 | 단일 | **32** | **16 FMA/clk**[^genealogy-clock]¹ | — | **2** | **64 KB (shared/L1 겸용, 48+16 분할)** | **128 KB** |

¹ GF100의 FP64는 전용 유닛 개수가 아니라 클럭당 FMA 처리량으로 공개되어 있어요. 표에도 그 기준으로 적었답니다.

## Kepler (2012, GK110)

Kepler에서는 SM을 SMX라는 이름으로 넓히면서 CUDA 코어를 192개까지 늘렸어요. 대신 명령 스케줄링의 상당 부분은 하드웨어에서 컴파일러로 옮겼고요. 단순한 스케줄러와 많은 코어를 낮은 클럭으로 돌려 전력 대비 성능(perf-per-watt)을 높이려는 설계였답니다.

전체 처리량을[^genealogy-throughput] 기준으로 보면 효율을 얻었어요. 다만 코어 192개에 쉬지 않고 일을 공급하기가 어려워서 코어당 활용률은 떨어졌어요. 코어를 늘려놨는데 다 쓰기가 어렵다니.. 참 아쉽죠. SM의 폭이 넓어져도 그만큼 자동으로 빨라지지는 않는다는 사례로 Kepler가 자주 언급되는 이유예요.

![Kepler SMX component diagram](./images/sm-kepler.svg?v=3)
*Kepler SMX(GK110)는 코어 192개, scheduler 4개를 갖추고 컴파일러 주도 스케줄링을 사용해요.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK110 | 단일 | **192** | **64** | — | **4** | 64 KB 겸용 **+ 48 KB read-only[^genealogy-read-only]** | **256 KB** |

## Maxwell (2014, GM200)

다음 Maxwell에서는 코어 수를 128개로 줄이고 SM 내부를 다시 나눴어요. 코어 32개, 전용 scheduler, 전용 register file을 가진 processing block을 4개 두는 구성이에요. 32는 warp 크기와 같으니, 각 processing block을 warp의 실행 폭에 맞춘 셈이죠.

새로운 공정으로 옮기지 않고 설계를 정리하는 것만으로도 큰 효율 향상을 얻었답니다. 코어를 조금 줄였는데 오히려 잘 쓰게 됐네요ㅎㅎ Maxwell이 효율적인 SM 구성의 사례로 꼽히는 이유예요. 이때 자리 잡은 SM 내부의 partition 구조는 뒤의 세대에도 이어져요.

![Maxwell and Pascal SM component diagram](./images/sm-maxwell-pascal.svg?v=2)
*Maxwell과 Pascal의 이 구성에서는 SM을 warp 크기의 partition 4개로 나눠요. 이런 분할이 Maxwell에서 자리 잡았답니다.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM200 | **4 × 32 (첫 분할)** | **128** | **4** | — | 4 | **96 KB (shared 전용, L1 분리)** | 256 KB |

## Pascal (2016, GP100 / GP102)

Pascal부터는 같은 세대 안에서도 consumer와 datacenter의 구성이 뚜렷하게 달라져요. consumer 쪽 GP102(GTX 1080 Ti)는 Maxwell 설계를 16nm 공정으로 옮기고 GDDR5X[^genealogy-gddr]를 붙여서 공정과 대역폭을[^genealogy-bandwidth] 개선했어요.

datacenter 쪽 GP100(P100)은 SM당 FP32 lane[^genealogy-lane]을 64개로 줄이면서 강력한 FP64 유닛을 갖췄고요. GPU 사이의 고속 연결인 NVLink와 고대역폭 메모리 HBM2[^genealogy-hbm]도 여기서 처음 들어왔답니다. 같은 Pascal이라고 한꺼번에 외워두면 나중에 헷갈리겠죠? 두 제품이 달라지는 부분을 같이 봐주세요~

여기서 말하는 차이는 같은 다이를 수율이나 성능에 따라[^genealogy-yield] 등급만 나누어 파는 binning과는 달라요. consumer와 datacenter 칩 자체의 구성이 갈라지는 지점이 Pascal인 거예요.

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GP100 (datacenter) | **2 × 32** | **64** | **32** | — | **2** | **64 KB (shared 전용)** | 256 KB |
| GP102 (consumer) | 4 × 32 | 128 | 4 | — | 4 | 96 KB | 256 KB |

GP102는 표에 적힌 Maxwell 구성을 사실상 유지해서, 해당 행에는 굵게 표시한 값이 없어요.

## Volta (2017, GV100)

Volta에서는 첫 Tensor Core가 등장해요! 작은 행렬의 곱셈-누산(matrix multiply-accumulate, 이하 MMA)[^genealogy-mma]을 명령 하나로 처리하는 전용 유닛이랍니다.

일반 FP 명령으로 행렬곱을 하면 계산 하나하나에 명령을 발행해야 해요. 이때 실제 연산보다 명령을 fetch하고 decode하고 schedule하는 오버헤드에[^genealogy-instruction-work] 전력이 많이 들어가거든요. 작은 행렬 연산을 명령 하나로 묶으면 개별 연산마다 반복되던 그 부담을 줄일 수 있어요. 계산을 시키는 데도 비용이 꽤 들었던 거네요..

![기세 좋게 주먹을 든 문](/images/naver-moon/moon-114.png)

또 하나 기억할 변화는 independent thread scheduling이에요[^genealogy-its]. 이때부터 warp 안의 thread가 각자 program counter[^genealogy-pc]를 갖게 됐어요. warp 전체가 같은 명령을 늘 같은 박자로 실행한다는 lockstep 가정을 그대로 쓸 수 없게 된 거예요. CUDA C 글에서 warp lockstep에 단서를 붙이고 `__syncwarp()`[^genealogy-syncwarp]를 따로 설명하는 이유가 여기 있답니다.

Volta는 이 계보에서 consumer 파트를 따로 두지 않고 datacenter 중심으로 나온 세대예요. 이후 AI 하드웨어를 설명할 때 계속 등장하는 Tensor Core와 새로운 thread 스케줄링을 여기서 만나게 돼요.

![Volta SM component diagram](./images/sm-volta.svg?v=1)
*Volta SM(GV100)에서는 CUDA 코어 옆에 첫 Tensor Core가 들어와요.*

| 칩 | Partition | FP32/SM | INT32/SM[^genealogy-int32] | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GV100 | **4 × 16** | **64** | **64 (전용 datapath 분리)** | **32** | **8 (1세대, FP16)**[^genealogy-fp16] | **4** | **128 KB (shared+L1 통합)** | 256 KB |

표에 INT32 열이 새로 생겼죠? Volta 이전에는 정수 연산이 FP32 코어와 datapath를 공유했는데, 여기서 전용 경로가 분리됐기 때문이에요.

## Turing (2018, TU102)

Turing에서는 Volta의 아이디어가 그래픽 라인으로도 이어져요. consumer GPU에 2세대 Tensor Core가 들어오고, ray tracing 연산을[^genealogy-ray-tracing] 전담하는 RT Core도 새로 추가됐답니다.

datapath를 나누면서 SM에서 FP32와 INT32 연산을 동시에 발행할 수 있게 됐어요. 실제 워크로드에는 부동소수점 계산 사이에 주소 계산 같은 정수 연산도 섞여 나오니, 이 경로 분리가 도움이 돼요. 그래픽용 GPU에도 AI와 ray tracing 전용 accelerator가 함께 들어간 거죠.

DLSS도 이 하드웨어를 활용해요. 낮은 해상도로 렌더링한 프레임을[^genealogy-rendering] 신경망으로 업스케일해서 성능을 확보하는 기능이랍니다. Tensor Core가 그래픽 라인에 들어온 이유가 여기서도 연결되네요~

![Turing and Ada SM component diagram](./images/sm-turing-ada.svg?v=1)
*Turing과 Ada의 SM 도식이에요. RT Core와 그래픽용 Tensor Core도 함께 볼 수 있어요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TU102 | 4 × 16 | 64 | 64 (동시 발행) | **2** | 8 **(2세대, INT8/4)**[^genealogy-int8] | **1 (1세대)** | **96 KB 통합** | 256 KB |

이 표는 갈라진 그래픽 라인의 시작점이라, 굵은 값은 Volta(GV100)와 비교해서 표시했어요.

## Ampere (2020, GA100 / GA102)

Ampere의 3세대 Tensor Core에는 TF32와 BF16 포맷이[^genealogy-formats] 추가됐어요. TF32는 FP32의 지수 범위를 유지하면서 mantissa를 줄인 포맷으로, 학습 코드에서 별도 변경 없이 활용할 수 있게 마련됐답니다. structured sparsity[^genealogy-sparsity]도 들어왔는데요. 가중치의 절반을 정해진 패턴에 맞춰 0으로 만들고 그 0의 계산을 건너뛰는 방식이에요. 이 조건을 맞추면 2배 처리량을 얻을 수 있어요. 아무 행렬이나 넣고 두 배를 기대하시면 안 되겠죠ㅎㅎ

데이터를 옮기는 쪽에서는 `cp.async` 명령을[^genealogy-async] 봐주세요. 이전에 global memory에서 shared memory로 복사할 때는 register를 거쳐야 했어요. `cp.async`는 그 중간 register를 거치지 않고 복사할 수 있어서, Tensor Core 커널에서 부담이 되던 register 사용량을 덜어줘요. 사용할 자리가 빠듯할 때는 이런 경로 하나도 반갑답니다.

![활짝 웃으며 좋아하는 문](/images/naver-moon/moon-5.png)

MIG(Multi-Instance GPU)도 추가됐어요. A100 하나를 서로 격리된 여러 GPU 인스턴스로 나누어 사용할 수 있는 기능이에요.

참, Ampere라는 이름은 datacenter와 consumer 양쪽에서 사용해요. 아래 표처럼 SM당 FP32가 datacenter A100에서는 64개, consumer RTX 30에서는 128개예요. 같은 이름이어도 표의 어느 행을 보는지 확인해주셔야 해요.

![Ampere SM component diagram](./images/sm-ampere.svg?v=1)
*Ampere SM(GA100)에는 3세대 Tensor Core와, register를 거치지 않고 shared memory로 복사하는 cp.async가 있어요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA100 (datacenter) | 4 × 16 | 64 | 64 | 32 | **4 (3세대, TF32/BF16)**² | — | **192 KB 통합 (shared 최대 164 KB)** | 256 KB |
| GA102 (consumer) | **4 × 32** | **128 (64 전용 + 64 INT 겸용)** | 64 | 2 | **4 (3세대)** | 1 **(2세대)** | **128 KB 통합** | 256 KB |

² Tensor Core 수가 8개에서 4개로 줄었지만, 유닛 하나가 처리하는 타일 크기는[^genealogy-tile] 커졌어요. 아래 [Tensor Core의 진화](#tensor-core의-진화)에서 다시 볼게요. 굵은 값은 GA100의 경우 GV100과, GA102의 경우 TU102와 비교한 표시예요.

## Hopper (2022, GH100)

Hopper에서는 Transformer Engine이 등장해요. layer마다 FP8과 FP16 중 적절한 정밀도를[^genealogy-layer] 자동으로 고르는 하드웨어와 소프트웨어의 조합이에요. 낮은 정밀도로 처리량을 얻으면서도 정확도를 유지하도록 돕는 장치랍니다. 4세대 Tensor Core에는 FP8의 E4M3, E5M2 포맷이[^genealogy-fp8-formats] 추가됐고요.

LLM 워크로드에[^genealogy-llm] 데이터를 공급하고 연산을 이어가기 위한 기능도 함께 들어와요. `wgmma`는 warp 4개를 묶은 warpgroup 단위로 실행하는 비동기 행렬 명령이에요. TMA(Tensor Memory Accelerator)는 thread 하나가 복사를 개시하면 하드웨어가 대량의 비동기 복사를 수행해주는 엔진이고요. thread block cluster와 distributed shared memory로[^genealogy-cluster] 여러 SM 사이에서 shared memory 데이터를 직접 주고받는 길도 생겼어요.

SemiAnalysis는 이 흐름을 Tensor Core 처리량은 세대마다 2배로 늘어나는데 global memory 지연은 줄지 않는 문제로 설명해요. 계산하는 쪽은 빨라졌는데 데이터가 아직 오는 중이면.. 기다려야 하니까요ㅠㅠ Hopper가 지연을 숨기고 데이터를 공급하는[^genealogy-latency-hide] 하드웨어에 투자한 배경이에요. 대표 제품 H100은 HBM3와 900 GB/s NVLink 4를 갖췄답니다.

![Hopper SM component diagram](./images/sm-hopper.svg?v=2)
*Hopper SM(GH100)에서는 FP8 Tensor Core, TMA, wgmma와 thread block cluster를 함께 보시면 돼요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GH100 | **4 × 32** | **128** | 64 | **64** | 4 **(4세대, FP8, wgmma)** | — | **256 KB 통합 (shared 최대 228 KB)** | 256 KB |

## Ada (2022, AD102)

같은 2022년의 그래픽 라인에는 Ada가 있어요. 4세대 Tensor Core와 3세대 RT Core를 탑재했고, Shader Execution Reordering(SER)도 들어왔답니다. ray tracing 중 생기는 thread divergence[^genealogy-divergence]에 대응해 실행을 재정렬하는 기능이에요. DLSS 3 frame generation 스택도[^genealogy-frame-generation] 추가됐고요. 대표 제품은 TSMC 4nm 공정의 RTX 4090이에요.

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD102 | 4 × 32 | 128 | 64 | 2 | 4 **(4세대, FP8)** | 1 **(3세대)** | 128 KB 통합 | 256 KB |

## Blackwell (2024, B200 / GB202)

Blackwell의 datacenter 파트 B200부터는 다이가 2개예요. reticle 한계, 즉 노광 장비가 한 번에 찍을 수 있는 최대 크기까지 키운 다이 두 개를 10 TB/s 링크로 연결했어요. 소프트웨어에는 하나의 GPU로 보이고요. 합쳐서 2080억 트랜지스터에[^genealogy-transistor] HBM3e를 사용해요. 칩 하나도 크다고 했는데 이제 둘을 붙였네요..

![입을 벌리고 놀라는 문](/images/naver-moon/moon-3.png)

consumer 파트인 GB202 계열(RTX 5090)은 750mm² 근처의 단일 다이와 GDDR7을 사용해요. GB202 풀다이는 SM 192개 규모이고, 출하되는 RTX 5090은 그중 170 SM, 즉 CUDA core 21,760개를 활성화한 구성이랍니다.

Chips and Cheese는 GB202를 특화보다 규모를 더 밀어붙인 설계로 해석해요. 약 8.7 TB/s 대역폭을 가진 64뱅크 L2 캐시도[^genealogy-bank] 지연보다 대역폭을 택한 구성으로 보고요. 코어 하나의 복잡도를 높이기보다 높은 코어 밀도로 처리량을 얻는다는 설명이에요.

두 파트에는 모두 5세대 Tensor Core가 들어가요. 이 세대에서는 FP4(NVFP4와 microscaling MXFP 포맷)[^genealogy-fp4], Tensor Memory(TMEM), CTA-pair MMA를 살펴볼 거예요. TMEM은 행렬 operand[^genealogy-operand]를 register file 밖의 전용 저장소에 두는 기능이에요. CTA-pair MMA는 SM 두 개가 하나의 행렬 연산을 협력해서 수행하는 방식이고요. 여기서 CTA는 thread block을 하드웨어 쪽에서 부르는 이름이랍니다.

시스템 구성도 함께 커졌어요. GB200은 datacenter Blackwell GPU 2개와 Grace CPU 1개를 한 모듈로 묶어요. GB200 NVL72는 이 모듈들을 GPU 72개 규모로 연결해 하나의 NVLink 도메인을[^genealogy-nvlink-domain] 만들고, 랙 전체가 하나의 큰 GPU처럼 협력하게 한 구성이에요.

![Blackwell SM component diagram](./images/sm-blackwell.svg?v=2)
*Blackwell SM(B200)에는 FP4 Tensor Core, 전용 TMEM, CTA-pair MMA가 들어와요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B200 (datacenter)³ | 4 × 32 | 128 | **128** | 64 | 4 **(5세대, FP4, tcgen05)**[^genealogy-tcgen05] | — | 256 KB | 256 KB **+ TMEM 256 KB** |
| GB202 (consumer) | 4 × 32 | **128 (전 코어 FP32/INT32 통합)** | **128** | 2 | 4 **(5세대, FP4)** | 1 **(4세대)** | 128 KB 통합 | 256 KB |

³ 이 글 작성 시점에는 datacenter Blackwell의 SM 단위 whitepaper가 공개되지 않았어요. 그래서 B200 행은 NVIDIA 기술 블로그와 공개 교육 자료의 수치로 적었답니다. 굵은 값은 B200의 경우 GH100과, GB202의 경우 AD102와 비교했어요.

## Rubin (2026)

이 글에서 마지막으로 볼 세대는 Rubin이에요. 칩에서 랙으로 설계 단위가 커지는 흐름을 여기까지 따라가볼게요. NVIDIA의 공개 자료에는 Rubin GPU가 3360억 트랜지스터, HBM4 288GB와 22 TB/s 대역폭, NVLink 6 3.6 TB/s로 소개돼 있어요.

다만 이 공개 spec에는 **preliminary, 즉 잠정 수치라는 표시**가 있어요. SM 세부도 공개되지 않은 상태라 이 절에는 앞에서 보던 SM 제원표를 넣지 않았답니다. 빈칸이 궁금하더라도 아직 확정된 숫자처럼 채워두면 안 되겠죠~

![진땀을 흘리며 조심하는 문](/images/naver-moon/moon-115.png)

여기서는 이름을 쓸 때도 조금 주의해야 해요. Rubin GPU라는 칩과 Vera Rubin이라는 시스템을 구분해서 보는 거예요. NVIDIA가 Vera Rubin을 소개할 때는 NVL72라는 랙 구성을 기준으로 설명하거든요. Rubin GPU 72개와 Vera CPU 36개를 액체냉각 NVLink-6 도메인 하나로 연결해서, 약 3.6 EFLOPS의 FP4 추론 성능과[^genealogy-inference-flops] 20.7 TB의 HBM4를 제공하는 구성이에요.

Vera CPU는 커스텀 Olympus Arm 코어 88개를[^genealogy-arm] 가진 별도의 칩이에요. 이 글에서 참고한 NVIDIA의 Vera Rubin 페이지는 compute, networking, storage, switching[^genealogy-platform]을 아우르는 seven-chip platform으로 전체를 설명해요. SM 하나의 도식만으로는 CPU와 네트워크까지 함께 설계한 이 구성을 담기 어렵답니다.

![Rubin and Vera Rubin platform diagram](./images/rubin-platform.svg?v=2)
*Vera Rubin platform에서는 GPU, Vera CPU, NVLink switch, DPU, Ethernet[^genealogy-network]이 한 랙을 이루어요.*

비교할 때도 **Rubin GPU**의 microarchitecture는 GB100과, 함께 설계된 rack-scale 컴퓨터인 **Vera Rubin**은 GB200 NVL72와 나란히 놓으면 돼요. GPU 하나에서 시작했는데 어느새 랙 전체를 보고 있네요ㅎㅎ 이제는 그 단위까지 살펴봐야 세대의 변화를 설명할 수 있어요.

![NVIDIA architecture snapshots, Tesla to Rubin](./images/architecture-snapshots.svg?v=2)
*Architecture snapshots를 모아봤어요. 공통 줄기에서 graphics와 compute의 통합을 거친 뒤, 위쪽은 datacenter AI로, 아래쪽은 RTX graphics로 이어져요.*

## Tensor Core의 진화

세대별 표를 봤으니 Tensor Core만 따로 이어서 볼까요? 행렬 곱셈-누산을 맡는 이 유닛은, 트랜지스터와 R&D 예산이 어디에 쓰였는지 보여주는 중요한 부분이에요. Volta부터 Blackwell까지 다섯 세대를 비교할 때는 **정밀도**와 **비동기성**을 함께 보시면 돼요.

추가된 포맷은 FP16(Volta) → INT8/INT4(Turing) → TF32와 BF16(Ampere) → FP8(Hopper) → FP4(Blackwell)로 이어져요. AI 워크로드가 더 낮은 정밀도를 활용할 수 있어서 가능한 흐름이에요. 정밀도를 절반으로 낮추면 같은 트랜지스터와 메모리에서 옮긴 바이트당 더 많은 연산을 할 수 있고, 처리량을 두 배로 높일 여지가 생겨요. 앞에서 본 Transformer Engine은 이 과정에서 적절한 정밀도를 골라 정확도를 유지하도록 돕는 역할이었죠.

그런데 표에서 Tensor Core 개수만 세면 조금 이상해 보이는 부분이 있어요. **유닛 하나가 처리하는 타일 크기**도 같이 봐야 하거든요. 행렬 곱은 대략 $N^3$번 연산하면서 $N^2$만큼 데이터를 옮겨요. 따라서 데이터 이동량에 대한 연산량, 즉 arithmetic intensity는 타일의 한 변 길이에 비례해서 커진답니다.

$$I \sim \frac{N^3}{N^2} = N$$

타일이 커지면 한 번 옮긴 데이터로 더 많은 계산을 하니, 데이터 이동 비용을 나누어 부담할 수 있어요. NVIDIA도 명령 하나가 계산하는 행렬 크기를 4×4×4 → 8×8×4 → 16×8×16[^genealogy-mma-shape], 그 이상으로 키워왔어요. Volta에서 SM당 8개였던 Tensor Core가 Ampere 이후 4개로 줄어든 것도 이렇게 유닛 하나가 커졌기 때문이랍니다. 세대마다 처리량은 두 배로 늘어났고요.

숫자만 보고 줄었다고 서운해할 뻔했네요ㅎㅎ 개수와 함께 무엇을 얼마나 처리하는지 봐야겠어요.

![능청스럽게 웃는 문](/images/naver-moon/moon-10.png)

실행 방식에도 같은 문제가 이어져요. Tensor 처리량은 계속 2배로 늘어나는데 메모리 지연이 줄지 않으면, 계산하는 동안 다음 데이터를 옮길 수 있어야 해요. 그래서 Volta의 동기식 warp-level MMA[^genealogy-mma-execution]에서 Hopper의 비동기 warpgroup MMA인 `wgmma`로, 다시 Blackwell의 전용 Tensor Memory에 operand를 두는 완전 비동기 single-thread MMA로 바뀌었답니다. **빨라진 연산기에 데이터를 제때 공급하는 일**이 계속 중요해지는 거예요.

## 두 라인: consumer와 datacenter

이번에는 Volta 이후의 두 라인을 나란히 놓아볼게요. 공통된 설계를 이어받으면서도 주로 처리할 작업에 따라 구성이 달라져요.

**Datacenter 라인**(GV100 → GA100 → GH100 → B200 → Rubin)은 AI 처리량과 interconnect에 힘을 줘요. SM당 FP32 lane 수보다는 INT32·FP64·Tensor Core의 비중을 크게 두고, GDDR 대신 HBM을 사용해요. NVLink에서 출발한 연결은 이제 랙 전체를 잇는 fabric으로 커졌고요. fabric은 칩과 노드를 묶어주는 통신망을 말해요. MIG와 thread block cluster 같은 datacenter용 기능도 이 라인에서 만나게 된답니다.

**Graphics 라인**(TU102 → GA102 → AD102 → GB202)은 DLSS에 필요한 Tensor Core와 함께 RT Core, 렌더링 기능을 갖춰요. 메모리도 GDDR을 쓰고요. 그래픽 작업에서 사용할 구성에 맞추어 발전해온 쪽이에요.

아까 Ampere에서 봤던 이름 문제도 다시 기억해주세요. Ampere와 Blackwell은 양쪽 라인에 모두 있는 이름이에요. "Ampere GPU"라고만 하면 A100인지 RTX 3090인지 아직 모르죠? 두 제품은 SM당 FP32 lane부터 64개와 128개로 달라요. 세대 이름에 어느 라인인지까지 덧붙여야 정확하게 비교할 수 있답니다.

## 가계도

| 세대 | 연도 | SM / 코드네임 | 정의적 변화 | 공정 | 대표 |
| --- | --- | --- | --- | --- | --- |
| Tesla | 2006 | SM, 8 SP (G80) | unified shader[^genealogy-shader], SIMT, CUDA | 90 nm | 8800 GTX |
| Fermi | 2010 | SM, 32 (GF100) | L1 데이터캐시, FMA, FP64, C++ | 40 nm | GTX 480 |
| Kepler | 2012 | SMX, 192 (GK110) | 컴파일러 스케줄러, wide SM | 28 nm | K20 |
| Maxwell | 2014 | SMM, 128 (GM200)[^genealogy-smm] | 효율, 4x32 partition | 28 nm | GTX 980 Ti |
| Pascal | 2016 | GP100 / GP102 | NVLink, HBM2(GP100), 16nm | 16 nm | P100 |
| Volta | 2017 | GV100, 64 FP32 | 1st Tensor Core, 독립 thread 스케줄링 | 12 nm | V100 |
| Turing | 2018 | TU102, 64 FP32 | RT Core + 2nd Tensor를 그래픽으로 | 12 nm | RTX 2080 Ti |
| Ampere | 2020 | GA100, 64 FP32 | 3rd Tensor(TF32/sparsity), MIG | 7 nm | A100 |
| Ada | 2022 | AD102, 128 FP32 | 4th Tensor, 3rd RT, SER | 4 nm | RTX 4090 |
| Hopper | 2022 | GH100, 128 FP32 | Transformer Engine(FP8), TMA, cluster | 4 nm | H100 |
| Blackwell | 2024 | 2 dies, 208B[^genealogy-billion] | FP4, TMEM, 5th NVLink, scale-first[^genealogy-scale] | TSMC 4NP | B200 / GB200 |
| Rubin | 2026 | 2 dies, 336B | HBM4, NVLink 6; Vera Rubin = rack platform | NVIDIA 공개 spec상 미확정 | Rubin / Vera Rubin NVL72 |

## 정리: 세 가지 흐름

Tesla부터 Rubin까지 따라오면서 이름을 꽤 많이 만났네요. 마지막으로 변화의 방향을 다시 짚어볼게요~

먼저 **specialization, 즉 특화가 늘어났어요.** SM의 범용 코어는 유지하면서 특정 연산을 맡는 Tensor Core와 RT Core, Transformer Engine, 전용 Tensor Memory를 더해왔죠. **사용하는 정밀도도 낮아졌어요.** FP32에서 FP4까지, AI 작업에서 허용할 수 있는 정밀도 범위 안에서 비트 수를 줄여 처리량을 얻는 방향이에요. 그리고 **설계 단위가 커졌답니다.** 칩 하나를 보던 데서 다이 2개를 연결하고, 이제는 랙 전체를 함께 설계하게 됐어요.

이렇게 바뀌는 동안에도 데이터를 옮기고 지연을 숨기는 문제는 계속 남아 있어요. CUDA C 글의 memory coalescing[^genealogy-coalescing], Hopper의 TMA, Blackwell의 TMEM을 이어보면 왜 새 하드웨어에서 데이터 공급에 많은 자원을 쓰는지 보인답니다. 제원표에서도 register file은 Kepler 이후 256 KB로 유지되지만 shared memory는 처음의 16 KB에서 228 KB까지 늘어났죠. Blackwell에서는 TMEM이라는 저장소도 따로 생겼고요.

지난 10년간 연산 자체의 비용은 낮아졌지만, 그 연산에 필요한 데이터를 가져오는 비용은 여전히 크게 남아 있어요. 이 계보를 따라 계속 만나게 되는 memory wall이에요. 계산은 빨라졌는데 기다릴 일이 아직 남아 있네요.. 다음에 새 GPU의 사양표를 보실 때는 연산 수치와 함께 데이터를 어떻게 공급하는지도 살펴봐주세요ㅎㅎ

## 참고

- [Fabien Sanglard, A history of NVidia Stream Multiprocessor](https://fabiensanglard.net/cuda/): Tesla부터 Turing까지의 서사와 SM 설계 변화.
- [SemiAnalysis, NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell): 정밀도, 비동기성, 타일 크기 논증.
- [Chips and Cheese, Blackwell: NVIDIA's Massive GPU](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu): scale-over-specialization microarchitecture 해석.
- NVIDIA 1차 architecture 문서: [Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf), [Kepler GK110](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf), [Maxwell tuning](https://docs.nvidia.com/cuda/maxwell-tuning-guide/), [Pascal GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [Volta GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [Ampere A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [Ampere GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [H100 whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf), [GeForce RTX Blackwell](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)와 [Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/): 최신 세대의 1차 수치.
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)와 [NVIDIA Rubin platform technical blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/): Rubin GPU, NVLink 6, NVL72, preliminary spec caveat.
- [Cornell Virtual Workshop, B200 SM](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm): B200 SM 구성 수치.
- 스티커: LINE [Moon & James](https://store.line.me/stickershop/product/1/en).

[^genealogy-whitepaper]: 아키텍처는 연산 장치와 메모리, 그 연결과 실행 방식을 정한 설계예요. whitepaper는 제조사가 이런 설계와 기능·제원을 자세히 설명해 공개하는 기술 문서고요.

[^genealogy-microarchitecture]: 명령을 실제로 실행하기 위해 칩 내부의 연산기·저장 공간·스케줄러를 어떻게 구성했는지 다루는 말이에요. 같은 종류의 명령을 지원해도 이 내부 구성에 따라 성능과 전력 사용이 달라져요.

[^genealogy-gpu]: GPU(Graphics Processing Unit)는 많은 데이터에 비슷한 계산을 병렬로 수행하도록 만든 프로세서예요. 그래픽 처리뿐 아니라 행렬 계산 같은 수치 연산에도 사용해요.

[^genealogy-cuda]: NVIDIA GPU에 계산을 맡길 수 있게 하는 프로그래밍 환경이에요. 이 글에서는 CUDA 프로그램의 실행 단위와 실제 GPU 하드웨어가 어떻게 이어지는지 살펴봐요.

[^genealogy-lines]: datacenter 라인은 서버에서 AI·과학 계산 등을 주로 처리하는 제품군이에요. graphics 라인은 화면을 만드는 그래픽 작업에 초점을 둔 제품군이고, 뒤에서 말하는 consumer는 개인 사용자용 시장을 가리켜요.

[^genealogy-thread]: thread는 자기 번호와 실행 상태를 갖고 코드를 수행하는 단위예요. warp는 같은 block의 thread 32개를 하드웨어가 함께 관리하는 실행 묶음이에요.

[^genealogy-block]: 여러 thread를 묶어 한 SM에 배정하는 단위예요. 같은 block의 thread들은 shared memory로 값을 나누고, 서로 지정한 지점에 도착할 때까지 기다릴 수 있어요.

[^genealogy-memory]: register는 각 thread가 계산 중인 값을 두는 가까운 저장 공간이고, shared memory는 같은 block의 thread들이 함께 쓰는 SM 내부 저장 공간이에요. global DRAM은 모든 SM에서 접근할 수 있는 큰 메모리예요. register를 모아둔 SM의 저장소는 register file이라고 불러요.

[^genealogy-scheduler]: SM에 올라온 warp 중 다음 명령을 실행할 준비가 된 것을 고르고 연산기에 일을 내보내는 장치예요. 데이터를 기다리는 warp 대신 다른 warp를 실행하도록 선택할 수 있어요.

[^genealogy-compile]: 사람이 작성한 소스 코드를 대상 프로세서가 실행할 수 있는 명령으로 바꾸는 일이에요. 이를 수행하는 프로그램이 컴파일러예요.

[^genealogy-specialized]: Tensor Core는 행렬의 곱셈과 결과 누적을, RT Core는 빛의 경로와 물체의 충돌 위치를 찾는 계산을 돕는 전용 연산기예요. Transformer Engine은 AI 계산에서 숫자를 표현하는 비트 수를 조절해 속도와 정확도를 함께 챙기는 하드웨어·소프트웨어 조합이에요.

[^genealogy-die]: 반도체 웨이퍼에서 잘라낸 회로 조각 하나예요. 하나의 GPU 제품에 다이 하나를 넣을 수도 있고 여러 다이를 연결해 넣을 수도 있어요.

[^genealogy-rack]: 서버와 통신 장비, 전원·냉각 장치를 여러 층으로 꽂아 구성하는 장비함이에요. 랙 전체로 설계 단위를 넓힌다는 말은 그 안의 여러 CPU와 GPU, 연결망까지 함께 맞춘다는 뜻이에요.

[^genealogy-cache-controller]: 캐시는 자주 쓰거나 최근에 가져온 데이터를 가까이에 보관하는 저장소예요. L2는 여러 SM이 공유하는 캐시 단계이고, memory controller는 메모리 읽기·쓰기 요청의 처리 순서와 신호를 조정하는 장치예요.

[^genealogy-peripherals]: ROP는 화면에 쓸 색·깊이 등의 최종 그래픽 처리를, copy engine은 데이터 복사를 맡는 장치예요. host interface는 CPU 쪽과 GPU를 잇는 접점이고, fabric은 칩과 장비들을 연결하는 통신망이에요.

[^genealogy-tma-tmem]: TMA(Tensor Memory Accelerator)는 큰 데이터 묶음을 주로 global memory와 shared memory 사이에서 옮기는 장치예요. 복사를 요청한 thread는 완료를 바로 기다리지 않고 다른 일을 진행할 수 있어요. TMEM(Tensor Memory)은 Tensor Core가 행렬 계산에 사용할 값과 결과를 두는 전용 저장 공간으로, thread의 register file과는 별개예요.

[^genealogy-graphics]: vertex는 삼각형 같은 도형을 이루는 꼭짓점이고, pixel은 화면을 이루는 작은 점이에요. 그래픽 작업은 꼭짓점의 위치 등을 계산한 뒤 화면의 각 점에 들어갈 색을 계산하는 단계를 거쳐요.

[^genealogy-pipeline]: 파이프라인은 앞 단계의 결과를 다음 단계에 넘기는 처리 구조예요. 여기서는 vertex와 pixel용 처리 장치를 따로 배정해두었다는 뜻이고, 프로그래머블 코어는 사용자가 작성한 프로그램에 따라 계산 내용을 바꿀 수 있는 코어예요.

[^genealogy-process]: 공정은 트랜지스터와 배선을 반도체 위에 만드는 제조 기술이에요. nm는 나노미터지만, 특히 현대 공정 이름의 숫자를 회로 모든 부분의 실제 길이로 읽으면 안 돼요.

[^genealogy-sp]: SP는 곱셈·덧셈 같은 연산을 값 하나씩 처리하는 기본 연산기예요. 이후 제품 설명에서 CUDA core라고 부르게 되는 유닛이에요.

[^genealogy-clock]: 뒤의 표에서 clk는 clock cycle, 즉 회로가 동작을 진행하는 기준 박자 한 번이에요. 클럭당 처리량은 한 박자에 얼마나 계산할 수 있는지를 나타내요.

[^genealogy-table]: Partition은 SM 내부를 나눈 실행 구역이며, /SM은 SM 하나당 수치라는 뜻이에요. FP32와 FP64는 실수를 각각 32비트·64비트로 표현하는 형식으로, FP64가 일반적으로 더 세밀한 값을 표현해요. L1은 SM 가까이에 있는 첫 번째 데이터 캐시이고, Shared+L1은 shared memory와 그 캐시의 용량 구성을 함께 보여줘요.

[^genealogy-ecc]: Error-Correcting Code의 약자로, 메모리에 추가 검사 정보를 저장해 데이터의 비트 오류를 찾아내고 일부 오류를 바로잡는 기능이에요. 오래 실행되는 수치 계산에서 잘못된 결과를 줄이는 데 도움을 줘요.

[^genealogy-fma]: 곱셈 결과에 다른 값을 더하는 계산을 하나의 연산으로 처리하고 마지막에 한 번만 반올림하는 방식이에요. 곱셈과 덧셈을 따로 반올림할 때와 결과가 달라질 수 있어요.

[^genealogy-ieee]: 부동소수점 숫자의 표현과 반올림, 무한대·계산 불가능한 값 등의 처리를 정한 표준이에요. 여기서는 숫자 계산이 어떤 규칙을 따라야 하는지에 관한 말이에요.

[^genealogy-texture]: texture는 표면에 입힐 그림 같은 데이터를 담은 것이에요. texture unit은 좌표에 맞는 값을 읽고 주변 값을 섞어 부드럽게 만드는 처리를 맡아요.

[^genealogy-throughput]: 일정 시간에 끝내는 작업의 양이에요. 요청 하나가 끝날 때까지 걸리는 시간인 지연(latency)과는 구분하며, 개별 요청이 오래 걸려도 많은 요청을 동시에 처리하면 전체 처리량은 높을 수 있어요.

[^genealogy-read-only]: 커널이 읽기만 하는 데이터를 보관하는 캐시예요. 다시 읽는 데이터를 DRAM까지 가지 않고 가까운 저장소에서 꺼내도록 도와줘요.

[^genealogy-gddr]: GPU에서 많은 데이터를 빠르게 주고받도록 설계한 DRAM 규격이에요. GDDR5X·GDDR7은 그 규격의 다른 세대이고, 뒤에서 나오는 HBM과는 메모리 구성 방식이 달라요.

[^genealogy-bandwidth]: 초당 옮길 수 있는 데이터 양이에요. GB/s는 초당 십억 바이트, TB/s는 초당 일조 바이트를 뜻하고, 메모리 요청 하나의 대기 시간인 지연과는 다른 값이에요.

[^genealogy-lane]: 여기서 lane은 FP32 계산을 병렬로 수행하는 물리적인 연산 경로를 말해요. warp 안에서 thread 하나의 자리를 뜻하는 논리적인 lane과는 구분해요.

[^genealogy-hbm]: 여러 DRAM 층을 쌓고 GPU 가까이에서 아주 넓은 연결로 데이터를 주고받는 메모리예요. HBM2·HBM3·HBM3e·HBM4는 그 세대와 개선 버전을 가리키며, GDDR과 함께 GPU의 큰 데이터 저장소로 사용돼요.

[^genealogy-yield]: 만들어진 칩 중 정해진 동작 조건을 만족하는 칩의 비율을 말해요. 결함이 있는 부분을 끄고 낮은 사양 제품으로 판매하는 등 실제 동작 상태에 따라 제품을 나누기도 해요.

[^genealogy-mma]: 행렬 둘을 곱한 결과를 이미 있던 행렬 값에 더하는 계산이에요. 누산은 이렇게 계산한 값을 기존 합계에 계속 더해 모으는 것을 뜻해요.

[^genealogy-instruction-work]: fetch는 명령을 가져오는 일, decode는 무슨 동작인지 해석하는 일, schedule은 언제 어느 실행 장치에 보낼지 정하는 일이에요. 오버헤드는 원하는 계산 자체 외에 이런 준비와 관리에 드는 비용을 말해요.

[^genealogy-its]: warp 안의 thread마다 실행 위치와 대기 상태를 따로 관리하는 방식이에요. 같은 warp라는 이유만으로 모든 thread가 같은 순간에 같은 위치에 도달했다고 가정할 수는 없어요.

[^genealogy-pc]: thread가 다음에 실행할 명령의 위치를 기록하는 상태예요. thread마다 이 값을 관리하면 분기하거나 기다린 thread의 실행 위치를 따로 추적할 수 있어요.

[^genealogy-syncwarp]: 지정한 warp의 thread들이 이 지점에 도착할 때까지 기다리게 하는 CUDA 함수예요. 참여한 thread들의 메모리 접근 순서도 맞춰줘서, 다른 thread가 쓴 값을 사용할 때 필요한 동기화에 쓰여요.

[^genealogy-int32]: INT32는 32비트 정수 계산을 뜻해요. datapath는 값이 연산 장치를 지나며 처리되는 하드웨어 경로로, 전용 경로가 있으면 FP32 계산과 별개로 정수 계산을 진행할 수 있어요.

[^genealogy-fp16]: 실수를 16비트로 나타내는 부동소수점 형식이에요. FP32보다 적은 공간을 쓰지만 표현할 수 있는 값의 범위와 세밀함이 줄어들어요.

[^genealogy-ray-tracing]: 빛의 경로를 광선으로 표현하고 어떤 물체와 만나는지 추적해 그림자·반사 등을 계산하는 방식이에요. RT Core는 광선과 장면 구조를 비교하며 만나는 위치를 찾는 일을 가속해요.

[^genealogy-rendering]: 렌더링은 도형·재질·조명 등의 장면 정보로 화면 이미지를 만드는 일이에요. 해상도는 이미지의 가로·세로 픽셀 수이고, 업스케일은 작은 이미지를 더 큰 해상도로 복원하는 처리예요.

[^genealogy-int8]: 정수를 각각 8비트와 4비트로 표현하는 형식이에요. 적은 비트로 많은 값을 담고 계산할 수 있지만, AI에 사용할 때는 실수 값을 이 범위에 맞게 바꾸는 과정이 필요해요.

[^genealogy-formats]: 부동소수점은 수의 크기 범위를 정하는 지수와 세밀한 숫자 부분인 가수(mantissa)를 나누어 저장해요. TF32는 FP32의 넓은 지수 범위를 쓰되 계산에 쓰는 가수 비트를 줄인 Tensor Core용 형식이고, BF16은 FP32와 같은 지수 비트 수를 쓰는 16비트 형식이에요.

[^genealogy-sparsity]: 가중치는 AI 모델이 학습한 계산 계수예요. 여기서 structured sparsity는 행렬에서 정해진 방향으로 연속한 값 4개 중 2개를 0으로 두는 2:4 형태로, 하드웨어가 그 0에 대한 계산을 건너뛸 수 있게 해요.

[^genealogy-async]: 비동기는 작업을 요청한 뒤 완료를 바로 기다리지 않고 다음 일을 진행할 수 있다는 뜻이에요. 이 명령으로 복사하는 동안 독립된 계산을 할 수 있지만, 복사한 데이터를 사용하기 전에는 복사가 끝났는지 확인해야 해요.

[^genealogy-tile]: 큰 행렬을 계산하기 편한 작은 직사각형 조각으로 나눈 것이에요. 행렬을 통째로 한 번에 처리하는 대신 이런 조각 단위로 가져와 계산하고 재사용해요.

[^genealogy-layer]: layer는 신경망에서 입력을 받아 정해진 계산을 하고 다음 단계에 넘기는 층이에요. 정밀도는 숫자를 얼마나 세밀하게 표현하는지를 말하며, FP8은 실수를 8비트로 저장해 FP16보다 공간과 계산 부담을 줄이는 형식이에요.

[^genealogy-fp8-formats]: E 뒤 숫자는 지수 비트 수, M 뒤 숫자는 가수 비트 수예요. 둘 다 부호 1비트를 더해 8비트가 되며, E5M2는 더 넓은 수의 범위에, E4M3는 범위 안의 값을 더 세밀하게 표현하는 데 비중을 둬요.

[^genealogy-llm]: Large Language Model, 즉 대규모 언어 모델이에요. 많은 글을 학습해 문맥에 맞는 다음 토큰(텍스트를 잘게 나눈 단위)을 예측하며, 학습과 실행 과정에서 큰 행렬 계산이 반복돼요.

[^genealogy-cluster]: cluster는 가까운 SM들에 여러 block을 함께 배정해 서로 기다리고 데이터를 나누도록 한 묶음이에요. distributed shared memory는 같은 cluster 안에서 다른 block의 shared memory에도 접근할 수 있는 기능이에요.

[^genealogy-latency-hide]: 데이터를 기다리는 동안 독립된 다른 계산이나 다음 데이터의 복사를 진행하는 방식이에요. 메모리 자체의 응답 시간이 줄지 않아도 기다림과 유용한 작업을 겹쳐 연산기가 쉬는 시간을 줄일 수 있어요.

[^genealogy-divergence]: 같은 warp 안의 thread들이 조건문 등에 따라 서로 다른 코드 경로로 갈라지는 상황이에요. 한 경로를 처리하는 동안 다른 경로의 thread는 그 명령에 참여하지 못해 연산기 활용이 떨어질 수 있어요.

[^genealogy-frame-generation]: 앞뒤 프레임과 움직임 정보 등을 이용해 그 사이에 표시할 새 프레임을 만드는 기능이에요. 여기서 스택은 이를 수행하는 하드웨어·소프트웨어 기능들을 함께 부르는 말이에요.

[^genealogy-transistor]: 전기 신호를 제어하는 아주 작은 소자로, 연산 회로와 저장 회로를 만드는 기본 재료예요. 개수가 많다고 성능이 그대로 비례하는 것은 아니고 어디에 배치해 어떤 기능을 맡겼는지가 중요해요.

[^genealogy-bank]: 캐시를 나누어 요청을 처리할 수 있게 한 내부 저장 구역이에요. 요청이 여러 bank에 잘 나뉘면 동시에 더 많은 데이터를 처리할 수 있어요.

[^genealogy-fp4]: FP4는 값 하나를 4비트 부동소수점으로 표현하는 형식이에요. microscaling은 작은 값 묶음마다 공통 배율을 함께 저장해 표현 범위를 조절하는 방식이고, NVFP4와 MXFP4는 묶음 크기와 배율 표현 등이 달라요.

[^genealogy-operand]: 연산에 넣는 입력 값, 즉 피연산자예요. 행렬곱에서는 곱할 두 행렬과 누적에 사용할 값 등이 해당해요.

[^genealogy-nvlink-domain]: NVLink 연결을 통해 서로 데이터를 주고받으며 협력할 수 있도록 묶인 GPU들의 범위예요. 여러 GPU가 연결돼 있다는 뜻이지 CUDA에서 자동으로 단일 GPU 하나로 합쳐진다는 뜻은 아니에요.

[^genealogy-tcgen05]: Blackwell의 5세대 Tensor Core 연산과 전용 Tensor Memory 관리를 표현하는 NVIDIA PTX 명령 계열 이름이에요. PTX는 CUDA 컴파일 과정에서 사용하는 GPU용 가상 명령어 체계예요.

[^genealogy-inference-flops]: FLOPS는 초당 부동소수점 연산 수이고, EFLOPS는 초당 10의 18제곱 번을 뜻해요. 추론은 학습된 AI 모델에 새 입력을 넣어 결과를 얻는 과정이며, 여기의 성능 수치는 FP4 계산 조건에서의 값이에요.

[^genealogy-arm]: Arm은 CPU가 실행할 명령과 동작 규칙을 정의하는 아키텍처 계열이에요. 커스텀 코어는 그 명령 규칙을 지원하면서 내부 연산기와 실행 구조를 제조사가 직접 설계했다는 뜻이에요.

[^genealogy-platform]: 순서대로 계산, 장비 간 데이터 통신, 데이터 저장, 통신 경로를 골라 전달하는 기능을 말해요. 이들을 함께 구성하므로 GPU 칩만의 설계와 시스템 전체 설계를 구분해서 보는 거예요.

[^genealogy-network]: NVLink switch는 여러 GPU 사이의 NVLink 통신을 연결해주는 장치예요. DPU는 네트워크·저장장치 같은 시스템 데이터 처리를 CPU 대신 맡는 프로세서이고, Ethernet은 장비들을 연결하는 대표적인 네트워크 규격이에요.

[^genealogy-mma-shape]: 이 세 숫자는 보통 출력 행렬의 행 수 M, 열 수 N, 두 입력 행렬이 곱하며 합치는 길이 K를 뜻해요. 예를 들어 M×K 행렬과 K×N 행렬을 곱하면 M×N 결과가 나와요.

[^genealogy-mma-execution]: warp-level 명령은 warp의 thread들이 함께 참여해 실행해요. 뒤의 single-thread MMA는 thread 하나가 전용 행렬 연산기에 작업을 요청한다는 뜻으로, 행렬 전체를 범용 코어 하나가 순서대로 계산한다는 뜻은 아니에요.

[^genealogy-smm]: Maxwell 세대의 SM을 가리키는 이름이에요. Kepler의 SMX, Maxwell의 SMM처럼 같은 역할의 처리 장치를 세대별로 구분해서 부르기도 해요.

[^genealogy-coalescing]: warp의 여러 thread가 요청한 메모리 접근을 가능한 적은 수의 전송으로 합치는 것이에요. 이웃한 thread가 이웃한 주소를 읽으면 데이터를 효율적으로 공급하기 좋아요.

[^genealogy-shader]: shader는 꼭짓점이나 화면의 색 등을 계산하는 GPU 프로그램이에요. unified shader는 shader 종류별로 연산기를 따로 두던 구성을 합쳐 같은 프로세서 배열에서 실행하게 한 설계를 뜻해요.

[^genealogy-billion]: B는 billion, 즉 십억을 줄인 표시예요. 여기서 208B는 트랜지스터 2080억 개를 뜻하고, 메모리 용량의 바이트를 뜻하는 B와는 문맥이 달라요.

[^genealogy-scale]: 여기서는 코어·캐시·연결의 전체 규모를 키워 더 많은 일을 처리하는 데 비중을 둔다는 설계 해석이에요. 새로운 CUDA 기능이나 명령 이름은 아니에요.
