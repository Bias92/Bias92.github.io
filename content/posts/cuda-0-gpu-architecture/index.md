---
title: "00 GPU Architecture Primer: The Tesla Foundation"
date: 2026-05-15
draft: false
tags: ["CUDA", "GPU Architecture", "Tesla", "SIMT", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "CUDA C에 나오는 warp, SM, coalescing은 어디서 왔을까요? 2008 IEEE Micro 논문의 Tesla(G80/GT200)를 따라 Input Assembler부터 SPA·TPC·SM/SP·SFU/LSU, ROP와 DRAM까지 살펴봐요. 그래픽과 연산의 데이터 경로, SIMT와 clock domain을 CUDA 용어와 연결해볼게요."
---

> Source: Lindholm, Nickolls, Oberman, Montrym, *"NVIDIA Tesla: A Unified Graphics and Computing Architecture,"* IEEE Micro 28(2), 2008.

## 동기: CUDA 이전의 하드웨어

안녕하세요~ 오늘은 CUDA C[^tesla-cuda]를 보기 전에 GPU[^tesla-gpu] 안쪽부터 잠깐 들여다보려고 해요ㅎㅎ

[CUDA C 글](../cuda-c-basics/)에는 block(함께 실행할 thread 묶음)[^tesla-block], warp(32개 thread의 실행 묶음), SM(GPU 내부 처리 장치)[^tesla-warp-sm], occupancy[^tesla-occupancy], memory coalescing[^tesla-coalescing] 같은 말이 나와요. 코드와 함께 보다 보면 언어에서 정해놓은 기능 같기도 한데요. 이 이름들은 실제 하드웨어 구조를 소프트웨어 쪽에서 부르는 말이랍니다. NVIDIA가 2006년 11월 G80(GeForce 8800 GTX)으로 내놓고, 2008년 GT200(GTX 280)으로 다듬은 아키텍처[^tesla-architecture]가 그 출발점이에요. 이름은 Tesla이고, 위에 적어둔 IEEE Micro 논문에서 구조를 확인할 수 있어요.

하드웨어를 건너뛰면 `warp = 32`부터 외우게 되죠. 메모리 접근은 coalesce하라고 하고요. 외울 것도 참 많네요.. 그런데 데이터가 칩 안에서 어떻게 움직이는지 먼저 보면, 왜 그런 규칙이 생겼는지 연결할 수 있어요. 그래서 이번에는 Tesla의 데이터 경로를 위에서 아래로 따라가면서 [CUDA C 글](../cuda-c-basics/)의 용어들이 실제로 어느 부분을 가리키는지 짚어볼게요.

![눈을 반짝이며 기다리는 문](/images/naver-moon/moon-4.png)

참, 여기서 볼 Tesla(G80/GT200)는 2008년 무렵의 기준점이에요. 이 숫자를 그대로 요즘 GPU에 적용하시면 곤란하답니다. SM은 2006년 이후 여러 번 바뀌었거든요. SM당 FP32 lane[^tesla-fp32-lane]은 8개에서 128개로, warp scheduler[^tesla-scheduler]는 하나에서 넷으로 늘었고, 16KB였던 shared memory[^tesla-shared]도 200KB를 넘겼어요. 범용 L1 데이터 캐시[^tesla-cache], tensor core, async copy[^tesla-tensor-async]처럼 당시에는 없던 기능도 들어왔고요.

그동안 이어져 온 것은 구조를 이해하는 방식과 이름들이에요. SPA → TPC → SM → SP[^tesla-hierarchy]를 따라가면 CUDA의 block, warp, SM 스케줄링이 어디서 왔는지 보인답니다. 아래 숫자는 당시 칩의 모습으로 봐주시고, 이후에도 이어지는 개념을 같이 챙겨가시면 좋겠어요~

## 통합 셰이더 아키텍처

Tesla 이전 GPU에서는 특화된 프로세서들이 정해진 파이프라인을[^tesla-pipeline] 이루고 있었어요. vertex shader[^tesla-vertex]를 거치고, rasterization[^tesla-rasterization]을 하고, 그다음 pixel(fragment) shader로[^tesla-pixel] 가는 식이죠. 각 유닛은 명령어 집합도[^tesla-isa] 다르고 실리콘도 따로였답니다. vertex와 pixel에 얼마만큼의 하드웨어를 배정할지는 설계할 때 정해져요. 그러다 한쪽 작업만 몰리는 프레임이[^tesla-frame] 나오면, 다른 쪽은 할 일이 없어서 칩 절반이 놀기도 하는 거예요.

한쪽은 바쁜데 바로 옆은 쉬고 있다니.. 조금 아깝죠ㅎㅎ

![못마땅하게 돌아보는 문](/images/naver-moon/moon-17.png)

Tesla에서는 shader 종류마다 따로 두었던 프로세서를 하나의 통합 배열로 바꿨어요. 동일한 프로그래머블 프로세서들을 모든 shader 타입이 시간을 나누어 쓰는 방식이에요. vertex, geometry, pixel 작업이[^tesla-geometry] 모두 같은 코어에서 돌고, 하드웨어가 일이 있는 단계에 배열을 배정해주는 거죠. 이 배열이 SPA(Streaming Processor Array)예요.

자기 메모리 시스템을 갖춘 범용 프로세서 배열이 생기니, 여기에 그래픽 이외의 연산을 맡길 길도 열렸어요. 그 배열을 프로그래머가 계산에 사용할 수 있게 내놓은 것이 CUDA랍니다. CUDA C로 들어온 커널도[^tesla-kernel] 같은 배열에서 실행돼요. 그래픽 작업과 별도로 그 통합 배열에 일을 넣는 두 번째 프론트엔드라고[^tesla-frontend] 보시면 돼요.

![NVIDIA Tesla (G80) unified architecture](./images/tesla1.svg?v=2)
*G80은 8 TPC × 2 SM × 8 SP = 128 SP이고, DRAM 파티션은[^tesla-dram] 6개예요. TPC 8개의 배열이 SPA이며, 마젠타색 compute work distribution이 CUDA에서 사용하는 경로랍니다.*

## 데이터 경로

그러면 일감이 실제로 어디를 지나가는지 볼까요? 메모리에서 출발해서 칩을 통과한 뒤 다시 메모리로 돌아올 때까지, 그래픽 모드에서는 다음 순서로 움직여요.

1. Input Assembler. DRAM에서 vertex 인덱스와 속성을[^tesla-attributes] 읽어서 primitive(점, 선, 삼각형)로 조립해요. 그래픽 데이터가 들어오는 입구랍니다.
2. Work distribution. Tesla에는 vertex, pixel, compute 분배기가 각각 있어요. 일감 묶음을 SPA에 넘기면서 프로세서 사이의 작업량을 맞춰주는, 즉 load-balance를 하는 부분이에요. compute 모드에서는 *compute work distribution* 유닛이 SM의 자리가 나는 대로 thread block을 하나씩 배정해요. CUDA의 transparent scalability[^tesla-scalability]도 실제로는 이 load-balancer가 받쳐주는 거예요.
3. SPA. 프로세서 배열이 shader나 커널을 실행해요. 그래픽에서는 vertex shading을 하고 나중에 pixel shading을, CUDA에서는 커널을 돌린답니다. 이곳이 연산을 맡고, 주변 유닛들은 필요한 데이터를 공급하거나 결과를 받아가요.
4. Setup / raster (그래픽 전용). vertex 작업과 pixel 작업 사이를 이어줘요. 고정 기능 유닛이[^tesla-fixed] clip, 삼각형 setup, rasterize[^tesla-setup]를 해서 fragment를 만들면, pixel 분배기가 이를 다시 SPA에 넣어요.
5. ROP (Raster Operations Processor). pixel shading이 끝난 뒤의 고정 기능 처리를 맡아요. depth·stencil 테스트[^tesla-tests], color blend, antialiasing[^tesla-blend]을 하고 framebuffer[^tesla-framebuffer]에 최종 결과를 쓰는 곳이죠. ROP 하나는 DRAM 파티션 하나에 묶여 있어요.
6. DRAM. 입력 데이터를 읽기 시작한 곳이자, 처리한 결과가 돌아가는 메모리 파티션들이에요.

CUDA에서는 그래픽 전용인 setup, raster, ROP를 사용하지 않아요. host가 grid를 launch하면[^tesla-launch] compute work distributor가 block을 SM들에 배정하고, SPA가 커널을 실행해요. 데이터는 load/store[^tesla-load-store]를 통해 DRAM과 주고받고요. 같은 칩에서 필요한 단계만 거치니 경로가 조금 짧아졌네요~

## 연산 계층: SPA → TPC → SM → SP

SPA 안쪽에도 묶음이 있어요. 코어가 한데 모여 있는 것으로만 생각하면 block이나 SM의 위치를 놓치기 쉬운데요. 아래의 3단 계층을 차례로 보면 CUDA 개념과 연결할 수 있답니다.

- TPC (Texture / Processor Cluster). SPA를 나누는 묶음이에요. TPC 하나에는 texture 유닛과[^tesla-texture], 이를 공유하는 몇 개의 SM이 들어 있어요. G80은 SM 2개씩 들어간 TPC가 8개, GT200은 SM 3개씩 들어간 TPC가 10개예요.
- SM (Streaming Multiprocessor). thread가 실제로 실행되는 유닛이에요. SM 하나에 SP 8개, SFU[^tesla-sfu] 2개, multithreaded 명령 fetch/issue 유닛[^tesla-fetch-issue], register file[^tesla-registers], 16KB shared memory가 들어 있어요. CUDA thread block이 배정된 뒤 머무는 곳도 이 SM이랍니다.
- SP (Streaming Processor). thread 하나의 부동소수점·정수 연산을 실행하는 scalar ALU예요[^tesla-alu]. 주로 MAD, 즉 곱셈-덧셈을 처리하고 SM마다 8개가 있어요. 나중에 제품 소개에서 "CUDA core"라고 부르게 된 유닛이 바로 이것이에요.

이제 TPC 수, TPC당 SM 수, SM당 SP 수를 곱해보면 전체 SP 수가 나와요.

$$
\text{G80: } 8\ \text{TPC} \times 2\ \text{SM} \times 8\ \text{SP} = 128\ \text{SP}
\qquad
\text{GT200: } 10 \times 3 \times 8 = 240\ \text{SP}
$$

숫자로 세어보니 G80은 128개, GT200은 240개네요. 이름이 많아서 복잡해 보였는데, 묶음별로 세면 확인할 수 있어요ㅎㅎ

![알겠다는 듯 경례하는 문](/images/naver-moon/moon-106.png)

CUDA를 볼 때는 SM의 다음 두 실행 유닛도 함께 알아두시면 좋아요.

- SFU (Special Function Unit). SM마다 2개가 있어요. reciprocal, reciprocal-sqrt, sin, cos, log, exp 같은[^tesla-functions] 특수 함수를 계산하고, 그래픽에서는 pixel 속성을 보간해요[^tesla-interpolation]. CUDA 커널의 `__sinf`나 `rsqrtf` 호출도[^tesla-intrinsic] 이 유닛으로 이어진답니다.
- LSU (Load/Store Unit). global·local memory[^tesla-memory-spaces]에 대한 load/store를 메모리 파이프라인으로 발행하는 경로예요. warp가 이 경로를 어떻게 사용하는지 따라가면 [CUDA C 글](../cuda-c-basics/)의 coalescing도 이해할 수 있어요.

## SIMT와 warp

이번에는 익숙한 warp가 나올 차례예요. SM의 명령 유닛은 thread 32개를 묶어서 warp 단위로 생성하고, 관리하고, 스케줄해서 실행해요. Tesla 논문에서는 이 방식을 SIMT(Single-Instruction, Multiple-Thread)라고 불렀어요. SM이 warp에 명령 하나를 issue하면 32개 thread가 그 명령을 실행하되, 사용하는 데이터와 레지스터는 thread마다 자기 것이에요.

그런데 아까 SM에 SP가 8개라고 했죠? thread는 32개인데 한 번에 들어갈 자리는 8개네요. 그래서 Tesla SM은 warp 하나를 8개 SP에서 빠른 shader clock[^tesla-clock] 4개에 걸쳐 실행한답니다.

$$
\frac{32\ \text{threads/warp}}{8\ \text{SP}} = 4\ \text{shader clocks per warp instruction}
$$

물리적인 SIMD 폭은[^tesla-simd] 8이고, 프로그래머에게 보이는 warp의 폭은 32인 거예요. NVIDIA는 이때 정한 warp 크기 32를 이후 세대에도 유지했어요. 그래서 뒤의 세대에서 쓰는 CUDA 코드도 여전히 32개를 기준으로 생각하는 거랍니다. warp 안에서 분기(divergence)가 갈리면[^tesla-divergence] 해당 분기로 가지 않는 thread는 그 실행 동안 마스킹돼요. CUDA 글에서 분기 비용을 이야기하는 이유도 여기에 있어요.

thread를 많이 올려놓으려면 자리도 필요해요. SM마다 register file과 16KB shared memory라는 정해진 자원이 있고, 상주하는 warp의 레지스터와 그 thread들이 속한 block의 shared memory를 여기서 배정받거든요. G80 SM에는 최대 24 warp(768 thread)가 상주할 수 있지만, thread당 사용하는 자원이 많아지면 실제로 들어가는 warp 수는 줄어요. 이 관계를 다루는 것이 occupancy예요. 마음 같아서는 다 올리고 싶어도.. SM 안의 저장 공간이 무한하지는 않으니까요ㅠㅠ

![SPA to TPC to SM to SP hierarchy](./images/tesla2.svg?v=1)

## Clock domain

성능을 계산하기 전에 clock도 확인하고 갈게요. Tesla GPU의 각 부분은 서로 다른 clock domain[^tesla-clock-domain]으로 움직여요. 사양표에서 주파수 하나만 골라 계산하면 결과가 어긋날 수 있답니다.

- core(graphics) clock은 프론트엔드, setup, raster, ROP에 적용돼요.
- shader clock은 SP에 적용되고 core clock보다 훨씬 빨라요. 8800 GTX의 core는 575MHz, shader는 1.35GHz로 약 2.35배 차이가 나요.
- memory clock은 GDDR3 인터페이스를[^tesla-gddr] 위한 별도의 clock이에요.

연산 처리량을 내는 SP 배열에는 공정[^tesla-process]이 허용하는 범위에서 높은 clock을 주고, 나머지 부분은 더 낮은 clock으로 돌려 전력과 발열을 줄이는 구성이에요. FLOP을 계산할[^tesla-flop] 때 shader clock을 넣어야 하는 이유가 이것이랍니다. 8800 GTX에서 SP 하나가 shader clock마다 MAD 하나, 즉 2 FLOP을 처리한다고 세면 다음과 같아요.

$$
128\ \text{SP} \times 1.35\ \text{GHz} \times 2\ \text{FLOP} \approx 346\ \text{GFLOP/s}
$$

그런데 NVIDIA가 제시한 수치는 518 GFLOP/s로 더 높아요. 같은 사이클에 SFU가 co-issue할 수 있는 MUL[^tesla-coissue]까지 포함했기 때문이에요. 실제 커널에서 이 조합을 계속 유지하기는 어렵고요.

숫자가 더 크다고 바로 그 속도가 나오는 건 아니었네요 ^^;; 제품에 적힌 피크와 실제로 달성할 수 있는 피크를 비교할 때는, 어느 clock에 어떤 명령 조합을 가정했는지까지 확인해주시면 좋아요.

![진땀을 흘리는 문](/images/naver-moon/moon-115.png)

## 메모리 서브시스템: DRAM 파티션과 coalescing

메모리 쪽으로도 내려가 볼게요. Tesla의 DRAM은 여러 독립 파티션으로 나뉘어 있고, 각 파티션에 memory controller와 ROP[^tesla-controller]가 하나씩 있어요. G80에는 64-bit 파티션이 6개라 버스 폭이 총 384-bit예요[^tesla-bus]. GT200은 8개라 512-bit이고요.

주소는 여러 파티션에 걸쳐 interleave돼요. 연속된 메모리 주소를 접근할 때 여러 controller가 병렬로 처리할 수 있도록 나누어 놓은 거예요. 그래서 전체 대역폭도[^tesla-bandwidth] 각 파티션의 대역폭을 합한 값이 된답니다.

8800 GTX의 384-bit 버스와 GDDR3 900MHz를 넣어서 계산해볼게요. double data rate라 핀당 전송률은 1.8 Gb/s예요[^tesla-ddr].

$$
\frac{384\ \text{bit}}{8} \times 1.8 \times 10^{9}\ \text{s}^{-1} = 48\ \text{B} \times 1.8\ \text{GT/s} \approx 86.4\ \text{GB/s}
$$

이렇게 폭이 넓고 여러 파티션으로 나뉜 메모리를 효율적으로 쓰려면, 데이터를 가져오는 방식도 맞아야겠죠? warp의 32 lane[^tesla-warp-lane]이 load를 발행하면 LSU가 그 요청들을 파티션에서 처리할 메모리 트랜잭션으로[^tesla-transaction] 바꿔요. 32개 주소가 연속되고 정렬 조건도 맞으면 몇 개의 넓은 트랜잭션으로 묶여 여러 controller가 처리할 수 있어요. 버스 폭을 잘 쓰게 되는 거죠.

반대로 주소가 흩어져 있으면 따로 처리할 트랜잭션이 늘어나고, 힘들게 가져온 메모리 라인에서[^tesla-line] 실제로 쓰는 부분은 조금일 수 있어요. 나머지는 그냥 버리는 셈이네요.. CUDA에서 warp의 접근을 연속으로 만들라고 하는 coalescing 조언은, 이 메모리 시스템에 맞게 넓고 정렬된 트랜잭션을 만들어주자는 뜻이에요.

## 계승과 변화

여기까지 본 모습은 2008년 칩을 기준으로 했어요. A100(2020)이나 H100(2022)과 나란히 놓으면 숫자가 꽤 달라져요. 그래도 warp를 실행하고 자원을 나누어 쓰는 기본 구조는 이어져 있답니다. Tesla에서 배운 개념을 가지고 다음 세대도 읽을 수 있는 이유예요.

| | G80 (2006) | A100 (2020) | H100 (2022) |
| --- | --- | --- | --- |
| SM당 FP32 lane | 8 | 64 | 128 |
| SM당 warp scheduler | 1 | 4 | 4 |
| warp 크기 | 32 | 32 | 32 |
| SM당 shared memory | 16 KB | 최대 164 KB | 최대 228 KB |
| SM당 32-bit 레지스터 | 8,192 | 65,536 | 65,536 |
| 이후 추가 | 기준선 | L1 데이터 캐시, tensor core, ITS, async copy[^tesla-its] | + TMA, thread block cluster[^tesla-tma-cluster] |

표에서도 warp 크기는 계속 32네요ㅎㅎ block 하나가 SM 하나에서 실행된다는 점도 같아요. SM은 여러 warp를 번갈아 스케줄하며 지연을 숨기고[^tesla-latency], shared memory는 SM 안에 있으며, 파티션으로 나뉜 global memory는 coalesce된 접근을 효율적으로 처리해요. CUDA를 이해할 때 계속 가져갈 수 있는 부분들이죠.

![반짝이는 엄지척을 보내는 문](/images/naver-moon/moon-13.png)

그 위에서 lane과 scheduler가 늘고, 범용 L1 데이터 캐시와 큰 register file이 들어왔어요. tensor core나 TMA처럼 특정 일을 맡는 유닛도 추가됐고요. 구조와 용어는 Tesla에서 익혀두시되, 실제로 최적화할 때 쓸 자원 크기와 처리량은 대상 아키텍처의 whitepaper[^tesla-whitepaper]에서 확인해주세요~

## Tesla 하드웨어 → CUDA 소프트웨어

이제 CUDA C에서 보던 말들을 Tesla의 하드웨어와 하나씩 연결해볼게요. 아래 표를 보시면 소프트웨어에서 말하는 기능이 칩의 어느 부분으로 이어지는지 찾을 수 있어요.

| Tesla 하드웨어 | CUDA C 개념 |
| --- | --- |
| Compute work distributor | grid의 block이 SM들에 퍼지는 방식 (transparent scalability) |
| SM | thread block 하나가 처음부터 끝까지 도는 곳 |
| Warp (32 thread, 8 SP에 4 clock) | SIMT 실행·스케줄 단위 |
| SP (scalar MAD ALU) | "CUDA core" |
| SFU | `__sinf`, `rsqrtf` 등 intrinsic |
| SM당 register file + 16KB shared memory | occupancy가 맞바꾸는 자원 |
| Shared memory | `__shared__`[^tesla-shared-keyword] |
| LSU + interleave된 DRAM 파티션 | coalescing과 대역폭이 중요한 이유 |

CUDA C 글로 돌아가면 `warp = 32`를 볼 때 Tesla SM이 32 thread를 8 SP에서 나누어 실행하던 모습을 떠올려보세요. occupancy에서는 SM의 한정된 register file과 shared memory를, coalescing에서는 넓은 트랜잭션을 처리하는 DRAM 파티션을 생각하시면 되고요. CUDA가 하드웨어의 이런 구조를 프로그래머에게 보여주는 거랍니다.

오늘은 코드보다 칩 안을 오래 구경했네요ㅎㅎ 다음에 CUDA 용어를 만나면 이름만 외우기보다, 어느 자리를 말하는지 함께 떠올려주시면 좋겠어요~

## 참고

- Lindholm, Nickolls, Oberman, Montrym, [*"NVIDIA Tesla: A Unified Graphics and Computing Architecture"*](https://ieeexplore.ieee.org/document/4523358): SPA/TPC/SM/SP 계층, SIMT, 그래픽 데이터 경로의 1차 출처.
- [NVIDIA GeForce 8800 GPU Architecture Technical Brief](https://www.nvidia.com/): G80 클럭, 파티션, register/shared memory 크기.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): 소프트웨어 모델이 현재 아키텍처 하드웨어에 어떻게 매핑되는지.
- [NVIDIA Hopper (H100) Architecture](https://www.nvidia.com/en-us/data-center/h100/): Tesla 계보가 자라난 현대 SM(128 FP32 lane, 4세대 tensor core, TMA).
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, occupancy, pinned memory, host-device 전송을 오늘날 하드웨어에 적용.
- 스티커: LINE [Moon & James](https://store.line.me/stickershop/product/1/en), [LINE Characters in Love!](https://store.line.me/stickershop/product/1252/en).

[^tesla-cuda]: CUDA는 NVIDIA GPU에 그래픽 이외의 계산도 맡길 수 있게 하는 프로그래밍 환경이에요. CUDA C는 C/C++에서 GPU용 함수와 실행 설정을 표현하도록 확장한 언어를 가리켜요.

[^tesla-gpu]: 많은 데이터에 비슷한 계산을 병렬로 수행하도록 만든 프로세서예요. 원래 그래픽 처리를 맡았고, CUDA에서는 수치 계산에도 사용해요.

[^tesla-block]: thread는 자기 번호와 실행 상태를 갖고 코드를 수행하는 단위예요. block은 함께 실행되며 데이터를 나누고 서로 기다릴 수 있도록 묶은 thread 그룹이고요.

[^tesla-warp-sm]: warp는 같은 block의 thread 32개를 묶은 실행 단위예요. SM(Streaming Multiprocessor)은 여러 warp를 실행하는 연산기와 저장 공간을 갖춘 GPU 내부의 처리 장치예요.

[^tesla-occupancy]: SM에 현재 올라가 있는 warp 수를 그 SM이 지원하는 최대 warp 수로 나눈 비율이에요. 연산기가 실제로 몇 퍼센트 바쁘게 일하는지를 직접 나타내는 값은 아니랍니다.

[^tesla-coalescing]: warp 안의 여러 thread가 요청한 메모리 접근을 가능한 적은 수의 전송으로 합치는 것이에요. 이웃한 thread가 이웃한 주소를 읽으면 데이터를 효율적으로 가져오기 좋아요.

[^tesla-architecture]: 여기서는 연산 장치와 메모리, 그 사이 연결을 어떻게 구성하고 어떤 방식으로 실행하게 했는지를 뜻해요. 제품 이름보다 안쪽의 설계를 보는 말이에요.

[^tesla-fp32-lane]: FP32는 실수를 32비트로 나타내는 부동소수점 형식이에요. 여기서 lane 수는 그 계산을 수행하는 물리적인 연산 경로 수이고, 뒤에서 나올 warp 안의 thread 번호인 lane과는 구분해요.

[^tesla-scheduler]: SM에 올라와 있는 warp 중 다음 명령을 실행할 준비가 된 warp를 고르는 장치예요. 한 warp가 데이터를 기다리는 동안 다른 warp에 일을 시킬 수 있어요.

[^tesla-shared]: 한 block의 thread들이 함께 읽고 쓸 수 있는 SM 내부의 빠른 저장 공간이에요. 다시 쓸 데이터를 여기에 두거나 thread 사이에서 값을 전달할 때 사용해요.

[^tesla-cache]: 캐시는 메모리에서 가져온 데이터 일부를 가까이에 보관해 다시 요청할 때 빨리 꺼내주는 저장소예요. L1은 연산 장치에 가까운 첫 번째 캐시 단계이며, 필요한 데이터를 하드웨어가 자동으로 보관해요.

[^tesla-tensor-async]: Tensor Core는 작은 행렬의 곱셈과 결과 누적을 한꺼번에 처리하는 전용 연산기예요. async copy는 복사 완료를 바로 기다리지 않고 다른 일을 진행할 수 있는 비동기 복사로, 복사된 데이터를 쓰기 전에는 완료를 확인해야 해요.

[^tesla-hierarchy]: SPA는 전체 연산 프로세서 배열, TPC는 그 안에서 몇 개의 SM과 그래픽용 데이터 처리 장치를 묶은 그룹이에요. SM 안에는 실제로 숫자를 계산하는 SP가 들어가고요.

[^tesla-pipeline]: 작업을 여러 처리 단계로 나누고 앞 단계의 결과를 다음 단계로 넘기는 구조예요. 서로 다른 단계가 서로 다른 데이터를 동시에 처리할 수 있어요.

[^tesla-vertex]: vertex는 삼각형 같은 도형을 이루는 꼭짓점으로 위치와 색 같은 속성을 가져요. vertex shader는 이 꼭짓점의 좌표 등을 계산하는 GPU 프로그램이에요.

[^tesla-rasterization]: 도형이 화면의 어느 위치를 덮는지 찾아 색과 깊이를 계산할 대상인 fragment를 만드는 과정이에요. fragment는 아직 최종 화면에 기록되기 전의 픽셀 후보라고 보시면 돼요.

[^tesla-pixel]: pixel은 화면을 이루는 작은 점이고, pixel 또는 fragment shader는 그 위치에 들어갈 색 등을 계산하는 프로그램이에요. 계산 결과가 뒤의 검사에서 버려질 수도 있어서 fragment가 곧 최종 pixel인 것은 아니에요.

[^tesla-isa]: 프로세서가 알아듣고 실행할 수 있는 명령의 목록과 규칙이에요. 덧셈이나 메모리 읽기 같은 동작을 어떤 명령으로 표현하는지가 여기에 들어가요.

[^tesla-frame]: 영상이나 게임 화면에서 한 시점에 보여주는 이미지 한 장이에요. 프레임마다 등장하는 도형과 효과가 달라지면 GPU 각 단계의 일감도 달라져요.

[^tesla-geometry]: 이 중 geometry shader는 꼭짓점 하나가 아니라 점·선·삼각형 같은 도형 단위로 입력을 받아 도형을 추가하거나 바꾸는 프로그램이에요.

[^tesla-kernel]: CUDA에서 GPU에 실행시키는 함수를 커널(kernel)이라고 해요. 같은 함수 코드를 많은 thread가 실행하면서 자기 번호에 해당하는 데이터를 처리해요.

[^tesla-frontend]: 프로세서 배열에 작업을 넘기기 전에 입력과 실행 요청을 받아 정리하는 앞단이에요. 여기서는 그래픽 명령을 받는 경로와 CUDA 계산 요청을 받는 경로를 구분하는 말이에요.

[^tesla-dram]: DRAM은 GPU가 처리할 입력과 결과를 대량으로 보관하는 메모리예요. 파티션은 그 메모리를 독립적으로 요청을 처리할 수 있는 구역으로 나눈 것으로, 여러 구역을 함께 쓰면 더 많은 데이터를 옮길 수 있어요.

[^tesla-attributes]: 인덱스는 어떤 꼭짓점을 사용할지 가리키는 번호예요. 속성은 그 꼭짓점의 위치·색·표면 그림을 읽을 좌표 등이며, 인덱스를 따라 속성을 읽어 도형을 만들어요.

[^tesla-scalability]: SM이 적은 GPU와 많은 GPU에서 같은 block 구성을 실행할 수 있다는 뜻이에요. 각 block을 어느 SM에 배정할지는 하드웨어가 맡으므로 GPU마다 일일이 배치를 다시 정하지 않아도 돼요.

[^tesla-fixed]: 사용자가 올린 프로그램을 실행하는 대신 정해진 종류의 작업을 회로로 수행하는 장치예요. 삼각형을 화면의 픽셀 후보로 바꾸는 처리 등이 해당해요.

[^tesla-setup]: clip은 보일 수 있는 영역 밖의 도형 부분을 잘라내요. setup은 삼각형의 변과 속성을 화면에서 계산하기 위한 값을 준비하고, rasterize는 이를 이용해 어느 위치에 fragment를 만들지 정해요.

[^tesla-tests]: depth 테스트는 깊이 값을 비교해 앞의 물체에 가려진 부분을 걸러내요. stencil 테스트는 화면 위치별로 저장한 표시 값을 검사해 특정 영역만 그리도록 제한해요.

[^tesla-blend]: color blend는 새로 계산한 색을 이미 저장된 색과 섞는 처리로 반투명 표현 등에 사용해요. antialiasing은 도형 경계가 계단처럼 보이는 현상을 줄이는 처리예요.

[^tesla-framebuffer]: 그리는 화면의 색과 깊이 같은 결과를 보관하는 메모리 영역이에요. 완성된 색 이미지는 화면에 표시할 때 사용해요.

[^tesla-launch]: host는 GPU에 일을 요청하는 CPU 쪽이에요. grid는 커널 한 번을 실행할 때 생성되는 전체 block 묶음이고, launch는 그 실행을 요청하는 동작이에요.

[^tesla-load-store]: load는 메모리의 값을 실행 장치 쪽으로 읽는 동작이고, store는 계산한 값을 메모리에 쓰는 동작이에요.

[^tesla-texture]: texture는 표면에 입힐 그림 같은 데이터를 저장한 것이에요. texture 유닛은 좌표에 맞는 값을 읽고 주변 값들을 섞어 부드러운 결과를 만드는 처리를 맡아요.

[^tesla-fetch-issue]: fetch는 실행할 명령을 가져오는 일, issue는 준비된 명령을 실행 장치에 내보내는 일이에요. multithreaded라는 말은 여러 thread 묶음의 실행을 관리한다는 뜻이에요.

[^tesla-registers]: 레지스터는 thread가 계산 중인 값과 중간 결과를 보관하는 가까운 저장 공간이에요. SM의 레지스터 전체를 register file이라고 부르며, 여러 thread가 이 한정된 자원을 나누어 배정받아요.

[^tesla-alu]: ALU(Arithmetic Logic Unit)는 덧셈·곱셈이나 값 비교 같은 계산을 하는 회로예요. scalar는 값을 하나씩 다룬다는 뜻으로, 여러 값을 한꺼번에 다루는 벡터 연산과 구분해요.

[^tesla-functions]: 순서대로 역수, 제곱근의 역수, 사인, 코사인, 로그, 지수 함수를 말해요. 예를 들어 reciprocal-sqrt는 입력이 4이면 1/2을 계산하는 함수예요.

[^tesla-interpolation]: 알고 있는 위치의 값들로 그 사이 위치의 값을 구하는 일이에요. 삼각형 꼭짓점에 주어진 색으로 삼각형 안쪽의 색을 계산하는 경우가 한 예예요.

[^tesla-intrinsic]: 단정밀도는 FP32(32비트 실수)를 말하며, `__sinf`는 빠른 사인 계산, `rsqrtf`는 제곱근의 역수 계산에 쓰는 CUDA 함수예요. 컴파일러는 소스 코드를 프로세서 명령으로 번역하는 프로그램이고, intrinsic은 이 프로그램이 특별히 알아보고 특정 명령이나 짧은 명령 묶음으로 바꾸는 내장 함수예요. 빠른 함수는 정확도 조건도 함께 확인해야 해요.

[^tesla-memory-spaces]: global memory는 모든 thread가 접근할 수 있는 큰 메모리 공간이에요. local memory는 thread 하나만 접근하지만 실제 저장은 보통 같은 DRAM 계통을 사용하므로, 이름이 local이라고 SM 내부의 빠른 메모리라는 뜻은 아니에요.

[^tesla-clock]: clock은 회로가 동작을 진행할 기준 박자이고, 그 한 번을 cycle이라고 해요. shader clock은 SP 연산 장치가 따르는 박자예요. MHz는 초당 백만 번, GHz는 초당 십억 번의 박자를 뜻해요.

[^tesla-simd]: SIMD(Single Instruction, Multiple Data)는 명령 하나로 여러 데이터에 같은 연산을 적용하는 방식이에요. 여기서 폭 8은 그 연산을 병렬로 수행하는 하드웨어 경로가 8개라는 뜻이에요.

[^tesla-divergence]: 같은 warp 안에서 조건문의 참·거짓이 달라 thread들이 다른 코드 경로로 나뉘는 상황이에요. 한 경로를 실행할 때 그 경로에 속하지 않은 thread를 잠시 참여시키지 않는 것이 마스킹이에요.

[^tesla-clock-domain]: 같은 clock 신호를 기준으로 움직이는 회로들의 영역이에요. GPU 내부에 이런 영역을 여러 개 두면 연산기와 메모리를 서로 다른 주파수로 돌릴 수 있어요.

[^tesla-gddr]: GDDR(Graphics Double Data Rate)은 GPU용으로 많은 데이터를 빠르게 주고받도록 설계한 DRAM 규격이고, 뒤 숫자는 규격의 세대예요. 인터페이스는 메모리와 GPU를 연결해 신호를 주고받는 접점과 규칙이에요.

[^tesla-flop]: FLOP은 부동소수점 덧셈이나 곱셈 한 번을 세는 단위예요. FLOP/s는 초당 연산 수이고, GFLOP/s는 초당 십억 번이에요. 곱하고 더하는 MAD는 보통 2 FLOP으로 계산해요.

[^tesla-coissue]: co-issue는 같은 cycle에 둘 이상의 실행 경로로 명령을 함께 내보내는 것이에요. MUL은 곱셈 명령이며, 여기서는 SP의 MAD와 별도로 SFU의 곱셈까지 동시에 활용할 수 있다는 계산이에요.

[^tesla-controller]: 메모리 읽기·쓰기 요청을 받아 DRAM이 처리할 순서와 신호를 조정하는 장치예요.

[^tesla-bus]: 버스는 장치 사이에서 데이터를 전달하는 연결이고, 폭은 한 번에 병렬로 전달할 수 있는 비트 수예요. 8비트가 1바이트이므로 384비트 폭은 한 번에 48바이트에 해당해요.

[^tesla-bandwidth]: 초당 옮길 수 있는 데이터 양이에요. GB/s는 초당 십억 바이트를 뜻하며, 요청 하나를 끝내는 데 걸리는 시간인 지연과는 다른 값이에요.

[^tesla-ddr]: double data rate는 clock 신호가 올라갈 때와 내려갈 때 모두 데이터를 보내는 방식이에요. Gb/s의 소문자 b는 비트, GB/s의 대문자 B는 바이트를 뜻하고, GT/s는 초당 십억 번의 전송을 세는 단위예요.

[^tesla-warp-lane]: 여기서 lane은 warp 안에서 thread 하나가 차지하는 논리적인 자리로 번호는 0부터 31까지예요. 앞의 FP32 lane처럼 물리적인 연산기 수를 세는 말과는 문맥이 달라요.

[^tesla-transaction]: 메모리 시스템이 요청들을 묶어서 처리하는 실제 전송 단위예요. 정렬은 주소가 정해진 크기의 경계에 맞아 있다는 뜻으로, 경계를 걸치면 같은 데이터 양에도 전송이 더 필요할 수 있어요.

[^tesla-line]: 일정 크기의 연속된 데이터를 묶어 가져오는 단위를 가리켜요. 그중 한 값만 써도 묶음 전체를 가져와야 한다면, 옮긴 데이터에 비해 실제 사용한 데이터가 적어져요.

[^tesla-its]: ITS(Independent Thread Scheduling)는 warp 안의 thread마다 다음에 실행할 위치 등을 따로 관리하는 방식이에요. Volta부터 도입됐으며 thread들이 항상 정확히 같은 박자로 진행한다고 가정해서는 안 돼요.

[^tesla-tma-cluster]: TMA(Tensor Memory Accelerator)는 큰 데이터 묶음을 주로 global memory와 shared memory 사이에서 비동기로 옮기는 장치예요. thread block cluster는 가까운 SM들에 여러 block을 함께 배정해 서로 기다리거나 각 block의 shared memory를 주고받을 수 있게 한 묶음이에요.

[^tesla-latency]: 메모리나 연산 결과를 기다리는 동안 다른 준비된 warp를 실행해 연산기가 쉬는 시간을 줄이는 방식이에요. 요청 자체가 빨라지는 것은 아니지만 기다리는 시간을 다른 일과 겹칠 수 있어요.

[^tesla-whitepaper]: 제조사가 구조와 기능, 제원을 자세히 설명해 공개하는 기술 문서예요. 여기서는 특정 GPU 세대의 실제 자원 크기를 확인하는 자료로 사용해요.

[^tesla-shared-keyword]: CUDA C에서 변수를 block의 shared memory에 두겠다고 표시하는 선언이에요. 그 block 안의 thread들이 같은 저장 공간에 접근해요.

[^tesla-sfu]: SFU(Special Function Unit)는 역수나 삼각함수 같은 특수 계산을 맡는 실행 장치예요. Tesla에서는 그래픽의 픽셀 속성 보간에도 사용되며, 아래에서 지원하는 함수들을 살펴볼게요.

[^tesla-process]: 반도체 위에 작은 소자와 배선을 만드는 제조 기술이에요. 같은 설계라도 이 기술과 전력·발열 조건에 따라 안정적으로 동작할 수 있는 clock의 범위가 달라져요.
