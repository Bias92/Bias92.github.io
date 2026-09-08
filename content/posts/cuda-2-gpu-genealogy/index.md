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

> NVIDIA의 아키텍처 whitepaper와 공식 제품 페이지를 1차 자료로 삼았어요. 세대별 흐름과 microarchitecture 해석에는 Fabien Sanglard, Chips and Cheese, SemiAnalysis를 참고했고요. 전체 출처는 글 끝에 모아두었답니다.

## 개요

안녕하세요ㅎㅎ 오늘은 NVIDIA GPU 이름들을 조금 길게 따라가보려고 해요. 2006년 Tesla에서 2026년 Rubin까지, 20년 동안 아키텍처가 십수 번 바뀌었거든요. 이름만 봐도 벌써 제법 많죠.. 어느 세대가 무엇을 왜 바꿨는지 하나씩 연결해두면, 낯선 제품을 만났을 때도 타임라인에서 자리를 찾기 쉬워요.

![눈을 반짝이며 기대하는 문](/images/naver-moon/moon-4.png)

출발점은 [primer 글(CUDA 0)](../cuda-0-gpu-architecture/)에서 본 2006년 Tesla 칩이에요. 그 구조를 기준으로 놓고, 다음 세대에서는 무엇이 달라졌는지 살펴볼게요.

![NVIDIA GPU architecture family tree, Tesla to Rubin](./images/timeline.svg?v=1)
*Pascal까지는 하나의 줄기를 공유하고, Volta부터 위쪽 datacenter 라인과 아래쪽 graphics 라인으로 갈라져요.*

먼저 **계속 유지된 구조를 알아두고, 작업의 종류와 규모가 달라질 때 무엇을 추가했는지** 보시면 좋아요. 세대 이름을 하나씩 외울 때보다 연결되는 부분이 많답니다.

이어지는 토대는 실행 모델과 메모리 계층이에요. 명령 하나를 thread 32개 묶음인 warp에 실행하는 SIMT(Single Instruction, Multiple Threads), 하나의 SM 위에서 끝까지 실행되는 thread block, register → shared memory → global DRAM으로 이어지는 메모리 계층을 말해요. SM은 Streaming Multiprocessor의 약자로, 연산 유닛·스케줄러·shared memory를 묶어놓은 GPU의 기본 구성 단위고요. [CUDA C 글](../cuda-c-basics/)에서 익힌 이 구조는 G80부터 Rubin까지 이어져요. 오래전에 작성한 CUDA 코드도 지금 GPU를 대상으로 컴파일할 수 있는 기반이 여기에 있는 거예요.

그 위의 변화는 먼저 **워크로드의 이동**을 따라갔어요. GPU를 많이 쓰는 작업이 그래픽에서 AI로 옮겨가면서, SM의 범용 코어 옆에 특정 연산을 맡는 유닛이 더해졌거든요. Tensor Core, RT Core, Transformer Engine이 차례로 등장해요. 또 하나는 **규모의 압력**이에요. 다이 하나로 수요를 감당하기 어려워지니 설계 단위가 칩 하나에서 다이 2개로, 다시 랙 전체로 커졌답니다. 범용 SM을 바탕으로 전용 accelerator를 더하고 전체 규모도 키워온 흐름을 볼 거예요.

여기서는 주로 SM을 확대해서 비교할게요. CUDA 프로그램의 block과 warp가 배정되고 스케줄되는 곳이니까요. L2 캐시, memory controller, ROP, copy engine, host interface, fabric도 성능과 시스템 설계에 중요해요. 그중 프로그래머가 접하는 warp 실행, register, shared memory, Tensor Core, RT Core, TMA/TMEM의 변화를 SM 중심으로 놓으면 비교하기 편하답니다.

세대별로 SM 도식과 제원표도 붙여두었어요. **굵은 값은 같은 라인의 직전 세대와 달라진 항목**이에요. 수치는 NVIDIA whitepaper를 기준으로 하고, 다른 자료를 사용한 곳은 따로 표시할게요. 숫자가 많으니 바뀐 부분부터 보셔도 괜찮아요~

![Anatomy of a GPU die, where the SM sits](./images/gpu-anatomy.svg?v=1)
*GPU 다이에서는 SM 배열 주변에 L2·memory controller·DRAM 연결, graphics 전용 고정기능과 host/fabric 인터페이스가 자리해요. 아래 세대별 도식은 그중 SM 하나를 확대해서 보여준답니다.*

## Tesla (2006, G80)

첫 번째는 Tesla예요. 이전 GPU에서는 vertex 처리와 pixel 처리를 각각 전담하는 고정 파이프라인이 있었어요. Tesla는 이를 하나의 프로그래머블 코어 배열로 통합했답니다. 같은 코어에 그래픽 외의 계산도 맡길 수 있게 되면서 CUDA라는 프로그래밍 모델이 가능해졌어요.

이때는 90nm 공정에 SM당 scalar processor(SP) 8개, warp scheduler 1개였어요. 요즘 숫자를 보다가 돌아오면 소박해 보이기도 하죠ㅎㅎ 그래도 이후 계보가 이어지는 출발점이에요.

![Tesla SM component diagram](./images/sm-tesla.svg?v=1)
*Tesla SM(G80)에는 scalar SP 8개, scheduler 1개, shared memory 16 KB가 있어요. 여기서부터 비교해볼게요.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G80 | 단일 | 8 (SP) | — | — | 1 | 16 KB (shared 전용, L1 없음) | 32 KB |

## Fermi (2010, GF100)

GPU에 계산을 맡기기 시작했으니, 이제 수치 라이브러리를 올려서 본격적으로 사용할 기반도 필요하겠죠? Fermi에서는 범용 L1 데이터 캐시와 L2 캐시, ECC 메모리, fused multiply-add(FMA), IEEE 표준을 완전히 따르는 배정밀도(FP64) 연산, C++ 지원이 갖춰졌어요. 그래픽 칩을 연산용 프로그래밍 대상으로 쓰기 위해 필요한 것들이 들어온 거예요.

SM도 CUDA 코어 32개와 warp scheduler 2개로 커졌고, texture unit이 SM 안으로 들어왔답니다. Tesla에서 열린 GPU 연산의 가능성을 실제 수치 계산용으로 다듬은 세대로 보시면 돼요.

![Fermi SM component diagram](./images/sm-fermi.svg?v=1)
*Fermi SM(GF100)은 코어 32개와 scheduler 2개를 갖춰요. 이 계보에서 L1 데이터 캐시가 처음 들어온 세대랍니다.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GF100 | 단일 | **32** | **16 FMA/clk**¹ | — | **2** | **64 KB (shared/L1 겸용, 48+16 분할)** | **128 KB** |

¹ GF100의 FP64는 전용 유닛 개수가 아니라 클럭당 FMA 처리량으로 공개되어 있어요. 표에도 그 기준으로 적었답니다.

## Kepler (2012, GK110)

Kepler에서는 SM을 SMX라는 이름으로 넓히면서 CUDA 코어를 192개까지 늘렸어요. 대신 명령 스케줄링의 상당 부분은 하드웨어에서 컴파일러로 옮겼고요. 단순한 스케줄러와 많은 코어를 낮은 클럭으로 돌려 전력 대비 성능(perf-per-watt)을 높이려는 설계였답니다.

전체 처리량을 기준으로 보면 효율을 얻었어요. 다만 코어 192개에 쉬지 않고 일을 공급하기가 어려워서 코어당 활용률은 떨어졌어요. 코어를 늘려놨는데 다 쓰기가 어렵다니.. 참 아쉽죠. SM의 폭이 넓어져도 그만큼 자동으로 빨라지지는 않는다는 사례로 Kepler가 자주 언급되는 이유예요.

![비구름 아래에서 아쉬워하는 문](/images/naver-moon/moon-9.png)

![Kepler SMX component diagram](./images/sm-kepler.svg?v=3)
*Kepler SMX(GK110)는 코어 192개, scheduler 4개를 갖추고 컴파일러 주도 스케줄링을 사용해요.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GK110 | 단일 | **192** | **64** | — | **4** | 64 KB 겸용 **+ 48 KB read-only** | **256 KB** |

## Maxwell (2014, GM200)

다음 Maxwell에서는 코어 수를 128개로 줄이고 SM 내부를 다시 나눴어요. 코어 32개, 전용 scheduler, 전용 register file을 가진 processing block을 4개 두는 구성이에요. 32는 warp 크기와 같으니, 각 processing block을 warp의 실행 폭에 맞춘 셈이죠.

새로운 공정으로 옮기지 않고 설계를 정리하는 것만으로도 큰 효율 향상을 얻었답니다. 코어를 조금 줄였는데 오히려 잘 쓰게 됐네요ㅎㅎ Maxwell이 효율적인 SM 구성의 사례로 꼽히는 이유예요. 이때 자리 잡은 SM 내부의 partition 구조는 뒤의 세대에도 이어져요.

![Maxwell and Pascal SM component diagram](./images/sm-maxwell-pascal.svg?v=2)
*Maxwell과 Pascal의 이 구성에서는 SM을 warp 크기의 partition 4개로 나눠요. 이런 분할이 Maxwell에서 자리 잡았답니다.*

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM200 | **4 × 32 (첫 분할)** | **128** | **4** | — | 4 | **96 KB (shared 전용, L1 분리)** | 256 KB |

## Pascal (2016, GP100 / GP102)

Pascal부터는 같은 세대 안에서도 consumer와 datacenter의 구성이 뚜렷하게 달라져요. consumer 쪽 GP102(GTX 1080 Ti)는 Maxwell 설계를 16nm 공정으로 옮기고 GDDR5X를 붙여서 공정과 대역폭을 개선했어요.

datacenter 쪽 GP100(P100)은 SM당 FP32 lane을 64개로 줄이면서 강력한 FP64 유닛을 갖췄고요. GPU 사이의 고속 연결인 NVLink와 고대역폭 메모리 HBM2도 여기서 처음 들어왔답니다. 같은 Pascal이라고 한꺼번에 외워두면 나중에 헷갈리겠죠? 두 제품이 달라지는 부분을 같이 봐주세요~

여기서 말하는 차이는 같은 다이를 수율이나 성능에 따라 등급만 나누어 파는 binning과는 달라요. consumer와 datacenter 칩 자체의 구성이 갈라지는 지점이 Pascal인 거예요.

| 칩 | Partition | FP32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GP100 (datacenter) | **2 × 32** | **64** | **32** | — | **2** | **64 KB (shared 전용)** | 256 KB |
| GP102 (consumer) | 4 × 32 | 128 | 4 | — | 4 | 96 KB | 256 KB |

GP102는 표에 적힌 Maxwell 구성을 사실상 유지해서, 해당 행에는 굵게 표시한 값이 없어요.

## Volta (2017, GV100)

Volta에서는 첫 Tensor Core가 등장해요! 작은 행렬의 곱셈-누산(matrix multiply-accumulate, 이하 MMA)을 명령 하나로 처리하는 전용 유닛이랍니다.

일반 FP 명령으로 행렬곱을 하면 계산 하나하나에 명령을 발행해야 해요. 이때 실제 연산보다 명령을 fetch하고 decode하고 schedule하는 오버헤드에 전력이 많이 들어가거든요. 작은 행렬 연산을 명령 하나로 묶으면 개별 연산마다 반복되던 그 부담을 줄일 수 있어요. 계산을 시키는 데도 비용이 꽤 들었던 거네요..

![기세 좋게 주먹을 든 문](/images/naver-moon/moon-114.png)

또 하나 기억할 변화는 independent thread scheduling이에요. 이때부터 warp 안의 thread가 각자 program counter를 갖게 됐어요. warp 전체가 같은 명령을 늘 같은 박자로 실행한다는 lockstep 가정을 그대로 쓸 수 없게 된 거예요. CUDA C 글에서 warp lockstep에 단서를 붙이고 `__syncwarp()`를 따로 설명하는 이유가 여기 있답니다.

Volta는 이 계보에서 consumer 파트를 따로 두지 않고 datacenter 중심으로 나온 세대예요. 이후 AI 하드웨어를 설명할 때 계속 등장하는 Tensor Core와 새로운 thread 스케줄링을 여기서 만나게 돼요.

![Volta SM component diagram](./images/sm-volta.svg?v=1)
*Volta SM(GV100)에서는 CUDA 코어 옆에 첫 Tensor Core가 들어와요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | 스케줄러/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GV100 | **4 × 16** | **64** | **64 (전용 datapath 분리)** | **32** | **8 (1세대, FP16)** | **4** | **128 KB (shared+L1 통합)** | 256 KB |

표에 INT32 열이 새로 생겼죠? Volta 이전에는 정수 연산이 FP32 코어와 datapath를 공유했는데, 여기서 전용 경로가 분리됐기 때문이에요.

## Turing (2018, TU102)

Turing에서는 Volta의 아이디어가 그래픽 라인으로도 이어져요. consumer GPU에 2세대 Tensor Core가 들어오고, ray tracing 연산을 전담하는 RT Core도 새로 추가됐답니다.

datapath를 나누면서 SM에서 FP32와 INT32 연산을 동시에 발행할 수 있게 됐어요. 실제 워크로드에는 부동소수점 계산 사이에 주소 계산 같은 정수 연산도 섞여 나오니, 이 경로 분리가 도움이 돼요. 그래픽용 GPU에도 AI와 ray tracing 전용 accelerator가 함께 들어간 거죠.

DLSS도 이 하드웨어를 활용해요. 낮은 해상도로 렌더링한 프레임을 신경망으로 업스케일해서 성능을 확보하는 기능이랍니다. Tensor Core가 그래픽 라인에 들어온 이유가 여기서도 연결되네요~

![Turing and Ada SM component diagram](./images/sm-turing-ada.svg?v=1)
*Turing과 Ada의 SM 도식이에요. RT Core와 그래픽용 Tensor Core도 함께 볼 수 있어요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TU102 | 4 × 16 | 64 | 64 (동시 발행) | **2** | 8 **(2세대, INT8/4)** | **1 (1세대)** | **96 KB 통합** | 256 KB |

이 표는 갈라진 그래픽 라인의 시작점이라, 굵은 값은 Volta(GV100)와 비교해서 표시했어요.

## Ampere (2020, GA100 / GA102)

Ampere의 3세대 Tensor Core에는 TF32와 BF16 포맷이 추가됐어요. TF32는 FP32의 지수 범위를 유지하면서 mantissa를 줄인 포맷으로, 학습 코드에서 별도 변경 없이 활용할 수 있게 마련됐답니다. structured sparsity도 들어왔는데요. 가중치의 절반을 정해진 패턴에 맞춰 0으로 만들고 그 0의 계산을 건너뛰는 방식이에요. 이 조건을 맞추면 2배 처리량을 얻을 수 있어요. 아무 행렬이나 넣고 두 배를 기대하시면 안 되겠죠ㅎㅎ

데이터를 옮기는 쪽에서는 `cp.async` 명령을 봐주세요. 이전에 global memory에서 shared memory로 복사할 때는 register를 거쳐야 했어요. `cp.async`는 그 중간 register를 거치지 않고 복사할 수 있어서, Tensor Core 커널에서 부담이 되던 register 사용량을 덜어줘요. 사용할 자리가 빠듯할 때는 이런 경로 하나도 반갑답니다.

![활짝 웃으며 좋아하는 문](/images/naver-moon/moon-5.png)

MIG(Multi-Instance GPU)도 추가됐어요. A100 하나를 서로 격리된 여러 GPU 인스턴스로 나누어 사용할 수 있는 기능이에요.

참, Ampere라는 이름은 datacenter와 consumer 양쪽에서 사용해요. 아래 표처럼 SM당 FP32가 datacenter A100에서는 64개, consumer RTX 30에서는 128개예요. 같은 이름이어도 표의 어느 행을 보는지 확인해주셔야 해요.

![Ampere SM component diagram](./images/sm-ampere.svg?v=1)
*Ampere SM(GA100)에는 3세대 Tensor Core와, register를 거치지 않고 shared memory로 복사하는 cp.async가 있어요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA100 (datacenter) | 4 × 16 | 64 | 64 | 32 | **4 (3세대, TF32/BF16)**² | — | **192 KB 통합 (shared 최대 164 KB)** | 256 KB |
| GA102 (consumer) | **4 × 32** | **128 (64 전용 + 64 INT 겸용)** | 64 | 2 | **4 (3세대)** | 1 **(2세대)** | **128 KB 통합** | 256 KB |

² Tensor Core 수가 8개에서 4개로 줄었지만, 유닛 하나가 처리하는 타일 크기는 커졌어요. 아래 [Tensor Core의 진화](#tensor-core의-진화)에서 다시 볼게요. 굵은 값은 GA100의 경우 GV100과, GA102의 경우 TU102와 비교한 표시예요.

## Hopper (2022, GH100)

Hopper에서는 Transformer Engine이 등장해요. layer마다 FP8과 FP16 중 적절한 정밀도를 자동으로 고르는 하드웨어와 소프트웨어의 조합이에요. 낮은 정밀도로 처리량을 얻으면서도 정확도를 유지하도록 돕는 장치랍니다. 4세대 Tensor Core에는 FP8의 E4M3, E5M2 포맷이 추가됐고요.

LLM 워크로드에 데이터를 공급하고 연산을 이어가기 위한 기능도 함께 들어와요. `wgmma`는 warp 4개를 묶은 warpgroup 단위로 실행하는 비동기 행렬 명령이에요. TMA(Tensor Memory Accelerator)는 thread 하나가 복사를 개시하면 하드웨어가 대량의 비동기 복사를 수행해주는 엔진이고요. thread block cluster와 distributed shared memory로 여러 SM 사이에서 shared memory 데이터를 직접 주고받는 길도 생겼어요.

SemiAnalysis는 이 흐름을 Tensor Core 처리량은 세대마다 2배로 늘어나는데 global memory 지연은 줄지 않는 문제로 설명해요. 계산하는 쪽은 빨라졌는데 데이터가 아직 오는 중이면.. 기다려야 하니까요ㅠㅠ Hopper가 지연을 숨기고 데이터를 공급하는 하드웨어에 투자한 배경이에요. 대표 제품 H100은 HBM3와 900 GB/s NVLink 4를 갖췄답니다.

![Hopper SM component diagram](./images/sm-hopper.svg?v=2)
*Hopper SM(GH100)에서는 FP8 Tensor Core, TMA, wgmma와 thread block cluster를 함께 보시면 돼요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GH100 | **4 × 32** | **128** | 64 | **64** | 4 **(4세대, FP8, wgmma)** | — | **256 KB 통합 (shared 최대 228 KB)** | 256 KB |

## Ada (2022, AD102)

같은 2022년의 그래픽 라인에는 Ada가 있어요. 4세대 Tensor Core와 3세대 RT Core를 탑재했고, Shader Execution Reordering(SER)도 들어왔답니다. ray tracing 중 생기는 thread divergence에 대응해 실행을 재정렬하는 기능이에요. DLSS 3 frame generation 스택도 추가됐고요. 대표 제품은 TSMC 4nm 공정의 RTX 4090이에요.

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD102 | 4 × 32 | 128 | 64 | 2 | 4 **(4세대, FP8)** | 1 **(3세대)** | 128 KB 통합 | 256 KB |

## Blackwell (2024, B200 / GB202)

Blackwell의 datacenter 파트 B200부터는 다이가 2개예요. reticle 한계, 즉 노광 장비가 한 번에 찍을 수 있는 최대 크기까지 키운 다이 두 개를 10 TB/s 링크로 연결했어요. 소프트웨어에는 하나의 GPU로 보이고요. 합쳐서 2080억 트랜지스터에 HBM3e를 사용해요. 칩 하나도 크다고 했는데 이제 둘을 붙였네요..

![입을 벌리고 놀라는 문](/images/naver-moon/moon-3.png)

consumer 파트인 GB202 계열(RTX 5090)은 750mm² 근처의 단일 다이와 GDDR7을 사용해요. GB202 풀다이는 SM 192개 규모이고, 출하되는 RTX 5090은 그중 170 SM, 즉 CUDA core 21,760개를 활성화한 구성이랍니다.

Chips and Cheese는 GB202를 특화보다 규모를 더 밀어붙인 설계로 해석해요. 약 8.7 TB/s 대역폭을 가진 64뱅크 L2 캐시도 지연보다 대역폭을 택한 구성으로 보고요. 코어 하나의 복잡도를 높이기보다 높은 코어 밀도로 처리량을 얻는다는 설명이에요.

두 파트에는 모두 5세대 Tensor Core가 들어가요. 이 세대에서는 FP4(NVFP4와 microscaling MXFP 포맷), Tensor Memory(TMEM), CTA-pair MMA를 살펴볼 거예요. TMEM은 행렬 operand를 register file 밖의 전용 저장소에 두는 기능이에요. CTA-pair MMA는 SM 두 개가 하나의 행렬 연산을 협력해서 수행하는 방식이고요. 여기서 CTA는 thread block을 하드웨어 쪽에서 부르는 이름이랍니다.

시스템 구성도 함께 커졌어요. GB200은 datacenter Blackwell GPU 2개와 Grace CPU 1개를 한 모듈로 묶어요. GB200 NVL72는 이 모듈들을 GPU 72개 규모로 연결해 하나의 NVLink 도메인을 만들고, 랙 전체가 하나의 큰 GPU처럼 협력하게 한 구성이에요.

![Blackwell SM component diagram](./images/sm-blackwell.svg?v=2)
*Blackwell SM(B200)에는 FP4 Tensor Core, 전용 TMEM, CTA-pair MMA가 들어와요.*

| 칩 | Partition | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B200 (datacenter)³ | 4 × 32 | 128 | **128** | 64 | 4 **(5세대, FP4, tcgen05)** | — | 256 KB | 256 KB **+ TMEM 256 KB** |
| GB202 (consumer) | 4 × 32 | **128 (전 코어 FP32/INT32 통합)** | **128** | 2 | 4 **(5세대, FP4)** | 1 **(4세대)** | 128 KB 통합 | 256 KB |

³ 이 글 작성 시점에는 datacenter Blackwell의 SM 단위 whitepaper가 공개되지 않았어요. 그래서 B200 행은 NVIDIA 기술 블로그와 공개 교육 자료의 수치로 적었답니다. 굵은 값은 B200의 경우 GH100과, GB202의 경우 AD102와 비교했어요.

## Rubin (2026)

이 글에서 마지막으로 볼 세대는 Rubin이에요. 칩에서 랙으로 설계 단위가 커지는 흐름을 여기까지 따라가볼게요. NVIDIA의 공개 자료에는 Rubin GPU가 3360억 트랜지스터, HBM4 288GB와 22 TB/s 대역폭, NVLink 6 3.6 TB/s로 소개돼 있어요.

다만 이 공개 spec에는 **preliminary, 즉 잠정 수치라는 표시**가 있어요. SM 세부도 공개되지 않은 상태라 이 절에는 앞에서 보던 SM 제원표를 넣지 않았답니다. 빈칸이 궁금하더라도 아직 확정된 숫자처럼 채워두면 안 되겠죠~

![진땀을 흘리며 조심하는 문](/images/naver-moon/moon-115.png)

여기서는 이름을 쓸 때도 조금 주의해야 해요. Rubin GPU라는 칩과 Vera Rubin이라는 시스템을 구분해서 보는 거예요. NVIDIA가 Vera Rubin을 소개할 때는 NVL72라는 랙 구성을 기준으로 설명하거든요. Rubin GPU 72개와 Vera CPU 36개를 액체냉각 NVLink-6 도메인 하나로 연결해서, 약 3.6 EFLOPS의 FP4 추론 성능과 20.7 TB의 HBM4를 제공하는 구성이에요.

Vera CPU는 커스텀 Olympus Arm 코어 88개를 가진 별도의 칩이에요. 이 글에서 참고한 NVIDIA의 Vera Rubin 페이지는 compute, networking, storage, switching을 아우르는 seven-chip platform으로 전체를 설명해요. SM 하나의 도식만으로는 CPU와 네트워크까지 함께 설계한 이 구성을 담기 어렵답니다.

![Rubin and Vera Rubin platform diagram](./images/rubin-platform.svg?v=2)
*Vera Rubin platform에서는 GPU, Vera CPU, NVLink switch, DPU, Ethernet이 한 랙을 이루어요.*

비교할 때도 **Rubin GPU**의 microarchitecture는 GB100과, 함께 설계된 rack-scale 컴퓨터인 **Vera Rubin**은 GB200 NVL72와 나란히 놓으면 돼요. GPU 하나에서 시작했는데 어느새 랙 전체를 보고 있네요ㅎㅎ 이제는 그 단위까지 살펴봐야 세대의 변화를 설명할 수 있어요.

![NVIDIA architecture snapshots, Tesla to Rubin](./images/architecture-snapshots.svg?v=2)
*Architecture snapshots를 모아봤어요. 공통 줄기에서 graphics와 compute의 통합을 거친 뒤, 위쪽은 datacenter AI로, 아래쪽은 RTX graphics로 이어져요.*

## Tensor Core의 진화

세대별 표를 봤으니 Tensor Core만 따로 이어서 볼까요? 행렬 곱셈-누산을 맡는 이 유닛은, 트랜지스터와 R&D 예산이 어디에 쓰였는지 보여주는 중요한 부분이에요. Volta부터 Blackwell까지 다섯 세대를 비교할 때는 **정밀도**와 **비동기성**을 함께 보시면 돼요.

추가된 포맷은 FP16(Volta) → INT8/INT4(Turing) → TF32와 BF16(Ampere) → FP8(Hopper) → FP4(Blackwell)로 이어져요. AI 워크로드가 더 낮은 정밀도를 활용할 수 있어서 가능한 흐름이에요. 정밀도를 절반으로 낮추면 같은 트랜지스터와 메모리에서 옮긴 바이트당 더 많은 연산을 할 수 있고, 처리량을 두 배로 높일 여지가 생겨요. 앞에서 본 Transformer Engine은 이 과정에서 적절한 정밀도를 골라 정확도를 유지하도록 돕는 역할이었죠.

그런데 표에서 Tensor Core 개수만 세면 조금 이상해 보이는 부분이 있어요. **유닛 하나가 처리하는 타일 크기**도 같이 봐야 하거든요. 행렬 곱은 대략 $N^3$번 연산하면서 $N^2$만큼 데이터를 옮겨요. 따라서 데이터 이동량에 대한 연산량, 즉 arithmetic intensity는 타일의 한 변 길이에 비례해서 커진답니다.

$$I \sim \frac{N^3}{N^2} = N$$

타일이 커지면 한 번 옮긴 데이터로 더 많은 계산을 하니, 데이터 이동 비용을 나누어 부담할 수 있어요. NVIDIA도 명령 하나가 계산하는 행렬 크기를 4×4×4 → 8×8×4 → 16×8×16, 그 이상으로 키워왔어요. Volta에서 SM당 8개였던 Tensor Core가 Ampere 이후 4개로 줄어든 것도 이렇게 유닛 하나가 커졌기 때문이랍니다. 세대마다 처리량은 두 배로 늘어났고요.

숫자만 보고 줄었다고 서운해할 뻔했네요ㅎㅎ 개수와 함께 무엇을 얼마나 처리하는지 봐야겠어요.

![능청스럽게 웃는 문](/images/naver-moon/moon-10.png)

실행 방식에도 같은 문제가 이어져요. Tensor 처리량은 계속 2배로 늘어나는데 메모리 지연이 줄지 않으면, 계산하는 동안 다음 데이터를 옮길 수 있어야 해요. 그래서 Volta의 동기식 warp-level MMA에서 Hopper의 비동기 warpgroup MMA인 `wgmma`로, 다시 Blackwell의 전용 Tensor Memory에 operand를 두는 완전 비동기 single-thread MMA로 바뀌었답니다. **빨라진 연산기에 데이터를 제때 공급하는 일**이 계속 중요해지는 거예요.

## 두 라인: consumer와 datacenter

이번에는 Volta 이후의 두 라인을 나란히 놓아볼게요. 공통된 설계를 이어받으면서도 주로 처리할 작업에 따라 구성이 달라져요.

**Datacenter 라인**(GV100 → GA100 → GH100 → B200 → Rubin)은 AI 처리량과 interconnect에 힘을 줘요. SM당 FP32 lane 수보다는 INT32·FP64·Tensor Core의 비중을 크게 두고, GDDR 대신 HBM을 사용해요. NVLink에서 출발한 연결은 이제 랙 전체를 잇는 fabric으로 커졌고요. fabric은 칩과 노드를 묶어주는 통신망을 말해요. MIG와 thread block cluster 같은 datacenter용 기능도 이 라인에서 만나게 된답니다.

**Graphics 라인**(TU102 → GA102 → AD102 → GB202)은 DLSS에 필요한 Tensor Core와 함께 RT Core, 렌더링 기능을 갖춰요. 메모리도 GDDR을 쓰고요. 그래픽 작업에서 사용할 구성에 맞추어 발전해온 쪽이에요.

아까 Ampere에서 봤던 이름 문제도 다시 기억해주세요. Ampere와 Blackwell은 양쪽 라인에 모두 있는 이름이에요. "Ampere GPU"라고만 하면 A100인지 RTX 3090인지 아직 모르죠? 두 제품은 SM당 FP32 lane부터 64개와 128개로 달라요. 세대 이름에 어느 라인인지까지 덧붙여야 정확하게 비교할 수 있답니다.

## 가계도

| 세대 | 연도 | SM / 코드네임 | 정의적 변화 | 공정 | 대표 |
| --- | --- | --- | --- | --- | --- |
| Tesla | 2006 | SM, 8 SP (G80) | unified shader, SIMT, CUDA | 90 nm | 8800 GTX |
| Fermi | 2010 | SM, 32 (GF100) | L1 데이터캐시, FMA, FP64, C++ | 40 nm | GTX 480 |
| Kepler | 2012 | SMX, 192 (GK110) | 컴파일러 스케줄러, wide SM | 28 nm | K20 |
| Maxwell | 2014 | SMM, 128 (GM200) | 효율, 4x32 partition | 28 nm | GTX 980 Ti |
| Pascal | 2016 | GP100 / GP102 | NVLink, HBM2(GP100), 16nm | 16 nm | P100 |
| Volta | 2017 | GV100, 64 FP32 | 1st Tensor Core, 독립 thread 스케줄링 | 12 nm | V100 |
| Turing | 2018 | TU102, 64 FP32 | RT Core + 2nd Tensor를 그래픽으로 | 12 nm | RTX 2080 Ti |
| Ampere | 2020 | GA100, 64 FP32 | 3rd Tensor(TF32/sparsity), MIG | 7 nm | A100 |
| Ada | 2022 | AD102, 128 FP32 | 4th Tensor, 3rd RT, SER | 4 nm | RTX 4090 |
| Hopper | 2022 | GH100, 128 FP32 | Transformer Engine(FP8), TMA, cluster | 4 nm | H100 |
| Blackwell | 2024 | 2 dies, 208B | FP4, TMEM, 5th NVLink, scale-first | TSMC 4NP | B200 / GB200 |
| Rubin | 2026 | 2 dies, 336B | HBM4, NVLink 6; Vera Rubin = rack platform | NVIDIA 공개 spec상 미확정 | Rubin / Vera Rubin NVL72 |

## 정리: 세 가지 흐름

Tesla부터 Rubin까지 따라오면서 이름을 꽤 많이 만났네요. 마지막으로 변화의 방향을 다시 짚어볼게요~

먼저 **specialization, 즉 특화가 늘어났어요.** SM의 범용 코어는 유지하면서 특정 연산을 맡는 Tensor Core와 RT Core, Transformer Engine, 전용 Tensor Memory를 더해왔죠. **사용하는 정밀도도 낮아졌어요.** FP32에서 FP4까지, AI 작업에서 허용할 수 있는 정밀도 범위 안에서 비트 수를 줄여 처리량을 얻는 방향이에요. 그리고 **설계 단위가 커졌답니다.** 칩 하나를 보던 데서 다이 2개를 연결하고, 이제는 랙 전체를 함께 설계하게 됐어요.

이렇게 바뀌는 동안에도 데이터를 옮기고 지연을 숨기는 문제는 계속 남아 있어요. CUDA C 글의 memory coalescing, Hopper의 TMA, Blackwell의 TMEM을 이어보면 왜 새 하드웨어에서 데이터 공급에 많은 자원을 쓰는지 보인답니다. 제원표에서도 register file은 Kepler 이후 256 KB로 유지되지만 shared memory는 처음의 16 KB에서 228 KB까지 늘어났죠. Blackwell에서는 TMEM이라는 저장소도 따로 생겼고요.

지난 10년간 연산 자체의 비용은 낮아졌지만, 그 연산에 필요한 데이터를 가져오는 비용은 여전히 크게 남아 있어요. 이 계보를 따라 계속 만나게 되는 memory wall이에요. 계산은 빨라졌는데 기다릴 일이 아직 남아 있네요.. 다음에 새 GPU의 사양표를 보실 때는 연산 수치와 함께 데이터를 어떻게 공급하는지도 살펴봐주세요ㅎㅎ

![반짝이는 엄지척을 보내는 문](/images/naver-moon/moon-13.png)

## 참고

- [Fabien Sanglard, A history of NVidia Stream Multiprocessor](https://fabiensanglard.net/cuda/): Tesla부터 Turing까지의 서사와 SM 설계 변화.
- [SemiAnalysis, NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell): 정밀도, 비동기성, 타일 크기 논증.
- [Chips and Cheese, Blackwell: NVIDIA's Massive GPU](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu): scale-over-specialization microarchitecture 해석.
- NVIDIA 1차 architecture 문서: [Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf), [Kepler GK110](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf), [Maxwell tuning](https://docs.nvidia.com/cuda/maxwell-tuning-guide/), [Pascal GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [Volta GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [Ampere A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [Ampere GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [H100 whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf), [GeForce RTX Blackwell](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)와 [Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/): 최신 세대의 1차 수치.
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)와 [NVIDIA Rubin platform technical blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/): Rubin GPU, NVLink 6, NVL72, preliminary spec caveat.
- [Cornell Virtual Workshop, B200 SM](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm): B200 SM 구성 수치.
- 스티커: LINE [Moon & James](https://store.line.me/stickershop/product/1/en).
