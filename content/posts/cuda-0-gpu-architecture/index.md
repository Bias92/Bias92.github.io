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

안녕하세요~ 오늘은 CUDA C를 보기 전에 GPU 안쪽부터 잠깐 들여다보려고 해요ㅎㅎ

[CUDA C 글](../cuda-c-basics/)에는 block, warp, SM, occupancy, memory coalescing 같은 말이 나와요. 코드와 함께 보다 보면 언어에서 정해놓은 기능 같기도 한데요. 이 이름들은 실제 하드웨어 구조를 소프트웨어 쪽에서 부르는 말이랍니다. NVIDIA가 2006년 11월 G80(GeForce 8800 GTX)으로 내놓고, 2008년 GT200(GTX 280)으로 다듬은 아키텍처가 그 출발점이에요. 이름은 Tesla이고, 위에 적어둔 IEEE Micro 논문에서 구조를 확인할 수 있어요.

하드웨어를 건너뛰면 `warp = 32`부터 외우게 되죠. 메모리 접근은 coalesce하라고 하고요. 외울 것도 참 많네요.. 그런데 데이터가 칩 안에서 어떻게 움직이는지 먼저 보면, 왜 그런 규칙이 생겼는지 연결할 수 있어요. 그래서 이번에는 Tesla의 데이터 경로를 위에서 아래로 따라가면서 [CUDA C 글](../cuda-c-basics/)의 용어들이 실제로 어느 부분을 가리키는지 짚어볼게요.

![눈을 반짝이며 기다리는 문](/images/naver-moon/moon-4.png)

참, 여기서 볼 Tesla(G80/GT200)는 2008년 무렵의 기준점이에요. 이 숫자를 그대로 요즘 GPU에 적용하시면 곤란하답니다. SM은 2006년 이후 여러 번 바뀌었거든요. SM당 FP32 lane은 8개에서 128개로, warp scheduler는 하나에서 넷으로 늘었고, 16KB였던 shared memory도 200KB를 넘겼어요. 범용 L1 데이터 캐시, tensor core, async copy처럼 당시에는 없던 기능도 들어왔고요.

그동안 이어져 온 것은 구조를 이해하는 방식과 이름들이에요. SPA → TPC → SM → SP를 따라가면 CUDA의 block, warp, SM 스케줄링이 어디서 왔는지 보인답니다. 아래 숫자는 당시 칩의 모습으로 봐주시고, 이후에도 이어지는 개념을 같이 챙겨가시면 좋겠어요~

## 통합 셰이더 아키텍처

Tesla 이전 GPU에서는 특화된 프로세서들이 정해진 파이프라인을 이루고 있었어요. vertex shader를 거치고, rasterization을 하고, 그다음 pixel(fragment) shader로 가는 식이죠. 각 유닛은 명령어 집합도 다르고 실리콘도 따로였답니다. vertex와 pixel에 얼마만큼의 하드웨어를 배정할지는 설계할 때 정해져요. 그러다 한쪽 작업만 몰리는 프레임이 나오면, 다른 쪽은 할 일이 없어서 칩 절반이 놀기도 하는 거예요.

한쪽은 바쁜데 바로 옆은 쉬고 있다니.. 조금 아깝죠ㅎㅎ

![못마땅하게 돌아보는 문](/images/naver-moon/moon-17.png)

Tesla에서는 shader 종류마다 따로 두었던 프로세서를 하나의 통합 배열로 바꿨어요. 동일한 프로그래머블 프로세서들을 모든 shader 타입이 시간을 나누어 쓰는 방식이에요. vertex, geometry, pixel 작업이 모두 같은 코어에서 돌고, 하드웨어가 일이 있는 단계에 배열을 배정해주는 거죠. 이 배열이 SPA(Streaming Processor Array)예요.

자기 메모리 시스템을 갖춘 범용 프로세서 배열이 생기니, 여기에 그래픽 이외의 연산을 맡길 길도 열렸어요. 그 배열을 프로그래머가 계산에 사용할 수 있게 내놓은 것이 CUDA랍니다. CUDA C로 들어온 커널도 같은 배열에서 실행돼요. 그래픽 작업과 별도로 그 통합 배열에 일을 넣는 두 번째 프론트엔드라고 보시면 돼요.

![NVIDIA Tesla (G80) unified architecture](./images/tesla1.svg?v=2)
*G80은 8 TPC × 2 SM × 8 SP = 128 SP이고, DRAM 파티션은 6개예요. TPC 8개의 배열이 SPA이며, 마젠타색 compute work distribution이 CUDA에서 사용하는 경로랍니다.*

## 데이터 경로

그러면 일감이 실제로 어디를 지나가는지 볼까요? 메모리에서 출발해서 칩을 통과한 뒤 다시 메모리로 돌아올 때까지, 그래픽 모드에서는 다음 순서로 움직여요.

1. Input Assembler. DRAM에서 vertex 인덱스와 속성을 읽어서 primitive(점, 선, 삼각형)로 조립해요. 그래픽 데이터가 들어오는 입구랍니다.
2. Work distribution. Tesla에는 vertex, pixel, compute 분배기가 각각 있어요. 일감 묶음을 SPA에 넘기면서 프로세서 사이의 작업량을 맞춰주는, 즉 load-balance를 하는 부분이에요. compute 모드에서는 *compute work distribution* 유닛이 SM의 자리가 나는 대로 thread block을 하나씩 배정해요. CUDA의 transparent scalability도 실제로는 이 load-balancer가 받쳐주는 거예요.
3. SPA. 프로세서 배열이 shader나 커널을 실행해요. 그래픽에서는 vertex shading을 하고 나중에 pixel shading을, CUDA에서는 커널을 돌린답니다. 이곳이 연산을 맡고, 주변 유닛들은 필요한 데이터를 공급하거나 결과를 받아가요.
4. Setup / raster (그래픽 전용). vertex 작업과 pixel 작업 사이를 이어줘요. 고정 기능 유닛이 clip, 삼각형 setup, rasterize를 해서 fragment를 만들면, pixel 분배기가 이를 다시 SPA에 넣어요.
5. ROP (Raster Operations Processor). pixel shading이 끝난 뒤의 고정 기능 처리를 맡아요. depth·stencil 테스트, color blend, antialiasing을 하고 framebuffer에 최종 결과를 쓰는 곳이죠. ROP 하나는 DRAM 파티션 하나에 묶여 있어요.
6. DRAM. 입력 데이터를 읽기 시작한 곳이자, 처리한 결과가 돌아가는 메모리 파티션들이에요.

CUDA에서는 그래픽 전용인 setup, raster, ROP를 사용하지 않아요. host가 grid를 launch하면 compute work distributor가 block을 SM들에 배정하고, SPA가 커널을 실행해요. 데이터는 load/store를 통해 DRAM과 주고받고요. 같은 칩에서 필요한 단계만 거치니 경로가 조금 짧아졌네요~

## 연산 계층: SPA → TPC → SM → SP

SPA 안쪽에도 묶음이 있어요. 코어가 한데 모여 있는 것으로만 생각하면 block이나 SM의 위치를 놓치기 쉬운데요. 아래의 3단 계층을 차례로 보면 CUDA 개념과 연결할 수 있답니다.

- TPC (Texture / Processor Cluster). SPA를 나누는 묶음이에요. TPC 하나에는 texture 유닛과, 이를 공유하는 몇 개의 SM이 들어 있어요. G80은 SM 2개씩 들어간 TPC가 8개, GT200은 SM 3개씩 들어간 TPC가 10개예요.
- SM (Streaming Multiprocessor). thread가 실제로 실행되는 유닛이에요. SM 하나에 SP 8개, SFU 2개, multithreaded 명령 fetch/issue 유닛, register file, 16KB shared memory가 들어 있어요. CUDA thread block이 배정된 뒤 머무는 곳도 이 SM이랍니다.
- SP (Streaming Processor). thread 하나의 부동소수점·정수 연산을 실행하는 scalar ALU예요. 주로 MAD, 즉 곱셈-덧셈을 처리하고 SM마다 8개가 있어요. 나중에 제품 소개에서 "CUDA core"라고 부르게 된 유닛이 바로 이것이에요.

이제 TPC 수, TPC당 SM 수, SM당 SP 수를 곱해보면 전체 SP 수가 나와요.

$$
\text{G80: } 8\ \text{TPC} \times 2\ \text{SM} \times 8\ \text{SP} = 128\ \text{SP}
\qquad
\text{GT200: } 10 \times 3 \times 8 = 240\ \text{SP}
$$

숫자로 세어보니 G80은 128개, GT200은 240개네요. 이름이 많아서 복잡해 보였는데, 묶음별로 세면 확인할 수 있어요ㅎㅎ

![알겠다는 듯 경례하는 문](/images/naver-moon/moon-106.png)

CUDA를 볼 때는 SM의 다음 두 실행 유닛도 함께 알아두시면 좋아요.

- SFU (Special Function Unit). SM마다 2개가 있어요. reciprocal, reciprocal-sqrt, sin, cos, log, exp 같은 특수 함수를 계산하고, 그래픽에서는 pixel 속성을 보간해요. CUDA 커널의 `__sinf`나 `rsqrtf` 호출도 이 유닛으로 이어진답니다.
- LSU (Load/Store Unit). global·local memory에 대한 load/store를 메모리 파이프라인으로 발행하는 경로예요. warp가 이 경로를 어떻게 사용하는지 따라가면 [CUDA C 글](../cuda-c-basics/)의 coalescing도 이해할 수 있어요.

## SIMT와 warp

이번에는 익숙한 warp가 나올 차례예요. SM의 명령 유닛은 thread 32개를 묶어서 warp 단위로 생성하고, 관리하고, 스케줄해서 실행해요. Tesla 논문에서는 이 방식을 SIMT(Single-Instruction, Multiple-Thread)라고 불렀어요. SM이 warp에 명령 하나를 issue하면 32개 thread가 그 명령을 실행하되, 사용하는 데이터와 레지스터는 thread마다 자기 것이에요.

그런데 아까 SM에 SP가 8개라고 했죠? thread는 32개인데 한 번에 들어갈 자리는 8개네요. 그래서 Tesla SM은 warp 하나를 8개 SP에서 빠른 shader clock 4개에 걸쳐 실행한답니다.

$$
\frac{32\ \text{threads/warp}}{8\ \text{SP}} = 4\ \text{shader clocks per warp instruction}
$$

물리적인 SIMD 폭은 8이고, 프로그래머에게 보이는 warp의 폭은 32인 거예요. NVIDIA는 이때 정한 warp 크기 32를 이후 세대에도 유지했어요. 그래서 뒤의 세대에서 쓰는 CUDA 코드도 여전히 32개를 기준으로 생각하는 거랍니다. warp 안에서 분기(divergence)가 갈리면 해당 분기로 가지 않는 thread는 그 실행 동안 마스킹돼요. CUDA 글에서 분기 비용을 이야기하는 이유도 여기에 있어요.

thread를 많이 올려놓으려면 자리도 필요해요. SM마다 register file과 16KB shared memory라는 정해진 자원이 있고, 상주하는 warp의 레지스터와 그 thread들이 속한 block의 shared memory를 여기서 배정받거든요. G80 SM에는 최대 24 warp(768 thread)가 상주할 수 있지만, thread당 사용하는 자원이 많아지면 실제로 들어가는 warp 수는 줄어요. 이 관계를 다루는 것이 occupancy예요. 마음 같아서는 다 올리고 싶어도.. SM 안의 저장 공간이 무한하지는 않으니까요ㅠㅠ

![비구름 아래에서 우는 문](/images/naver-moon/moon-9.png)

![SPA to TPC to SM to SP hierarchy](./images/tesla2.svg?v=1)

## Clock domain

성능을 계산하기 전에 clock도 확인하고 갈게요. Tesla GPU의 각 부분은 서로 다른 clock domain으로 움직여요. 사양표에서 주파수 하나만 골라 계산하면 결과가 어긋날 수 있답니다.

- core(graphics) clock은 프론트엔드, setup, raster, ROP에 적용돼요.
- shader clock은 SP에 적용되고 core clock보다 훨씬 빨라요. 8800 GTX의 core는 575MHz, shader는 1.35GHz로 약 2.35배 차이가 나요.
- memory clock은 GDDR3 인터페이스를 위한 별도의 clock이에요.

연산 처리량을 내는 SP 배열에는 공정이 허용하는 범위에서 높은 clock을 주고, 나머지 부분은 더 낮은 clock으로 돌려 전력과 발열을 줄이는 구성이에요. FLOP을 계산할 때 shader clock을 넣어야 하는 이유가 이것이랍니다. 8800 GTX에서 SP 하나가 shader clock마다 MAD 하나, 즉 2 FLOP을 처리한다고 세면 다음과 같아요.

$$
128\ \text{SP} \times 1.35\ \text{GHz} \times 2\ \text{FLOP} \approx 346\ \text{GFLOP/s}
$$

그런데 NVIDIA가 제시한 수치는 518 GFLOP/s로 더 높아요. 같은 사이클에 SFU가 co-issue할 수 있는 MUL까지 포함했기 때문이에요. 실제 커널에서 이 조합을 계속 유지하기는 어렵고요.

숫자가 더 크다고 바로 그 속도가 나오는 건 아니었네요 ^^;; 제품에 적힌 피크와 실제로 달성할 수 있는 피크를 비교할 때는, 어느 clock에 어떤 명령 조합을 가정했는지까지 확인해주시면 좋아요.

![진땀을 흘리는 문](/images/naver-moon/moon-115.png)

## 메모리 서브시스템: DRAM 파티션과 coalescing

메모리 쪽으로도 내려가 볼게요. Tesla의 DRAM은 여러 독립 파티션으로 나뉘어 있고, 각 파티션에 memory controller와 ROP가 하나씩 있어요. G80에는 64-bit 파티션이 6개라 버스 폭이 총 384-bit예요. GT200은 8개라 512-bit이고요.

주소는 여러 파티션에 걸쳐 interleave돼요. 연속된 메모리 주소를 접근할 때 여러 controller가 병렬로 처리할 수 있도록 나누어 놓은 거예요. 그래서 전체 대역폭도 각 파티션의 대역폭을 합한 값이 된답니다.

8800 GTX의 384-bit 버스와 GDDR3 900MHz를 넣어서 계산해볼게요. double data rate라 핀당 전송률은 1.8 Gb/s예요.

$$
\frac{384\ \text{bit}}{8} \times 1.8 \times 10^{9}\ \text{s}^{-1} = 48\ \text{B} \times 1.8\ \text{GT/s} \approx 86.4\ \text{GB/s}
$$

이렇게 폭이 넓고 여러 파티션으로 나뉜 메모리를 효율적으로 쓰려면, 데이터를 가져오는 방식도 맞아야겠죠? warp의 32 lane이 load를 발행하면 LSU가 그 요청들을 파티션에서 처리할 메모리 트랜잭션으로 바꿔요. 32개 주소가 연속되고 정렬 조건도 맞으면 몇 개의 넓은 트랜잭션으로 묶여 여러 controller가 처리할 수 있어요. 버스 폭을 잘 쓰게 되는 거죠.

반대로 주소가 흩어져 있으면 따로 처리할 트랜잭션이 늘어나고, 힘들게 가져온 메모리 라인에서 실제로 쓰는 부분은 조금일 수 있어요. 나머지는 그냥 버리는 셈이네요.. CUDA에서 warp의 접근을 연속으로 만들라고 하는 coalescing 조언은, 이 메모리 시스템에 맞게 넓고 정렬된 트랜잭션을 만들어주자는 뜻이에요.

## 계승과 변화

여기까지 본 모습은 2008년 칩을 기준으로 했어요. A100(2020)이나 H100(2022)과 나란히 놓으면 숫자가 꽤 달라져요. 그래도 warp를 실행하고 자원을 나누어 쓰는 기본 구조는 이어져 있답니다. Tesla에서 배운 개념을 가지고 다음 세대도 읽을 수 있는 이유예요.

| | G80 (2006) | A100 (2020) | H100 (2022) |
| --- | --- | --- | --- |
| SM당 FP32 lane | 8 | 64 | 128 |
| SM당 warp scheduler | 1 | 4 | 4 |
| warp 크기 | 32 | 32 | 32 |
| SM당 shared memory | 16 KB | 최대 164 KB | 최대 228 KB |
| SM당 32-bit 레지스터 | 8,192 | 65,536 | 65,536 |
| 이후 추가 | 기준선 | L1 데이터 캐시, tensor core, ITS, async copy | + TMA, thread block cluster |

표에서도 warp 크기는 계속 32네요ㅎㅎ block 하나가 SM 하나에서 실행된다는 점도 같아요. SM은 여러 warp를 번갈아 스케줄하며 지연을 숨기고, shared memory는 SM 안에 있으며, 파티션으로 나뉜 global memory는 coalesce된 접근을 효율적으로 처리해요. CUDA를 이해할 때 계속 가져갈 수 있는 부분들이죠.

![반짝이는 엄지척을 보내는 문](/images/naver-moon/moon-13.png)

그 위에서 lane과 scheduler가 늘고, 범용 L1 데이터 캐시와 큰 register file이 들어왔어요. tensor core나 TMA처럼 특정 일을 맡는 유닛도 추가됐고요. 구조와 용어는 Tesla에서 익혀두시되, 실제로 최적화할 때 쓸 자원 크기와 처리량은 대상 아키텍처의 whitepaper에서 확인해주세요~

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
| Shared memory | `__shared__` |
| LSU + interleave된 DRAM 파티션 | coalescing과 대역폭이 중요한 이유 |

CUDA C 글로 돌아가면 `warp = 32`를 볼 때 Tesla SM이 32 thread를 8 SP에서 나누어 실행하던 모습을 떠올려보세요. occupancy에서는 SM의 한정된 register file과 shared memory를, coalescing에서는 넓은 트랜잭션을 처리하는 DRAM 파티션을 생각하시면 되고요. CUDA가 하드웨어의 이런 구조를 프로그래머에게 보여주는 거랍니다.

오늘은 코드보다 칩 안을 오래 구경했네요ㅎㅎ 다음에 CUDA 용어를 만나면 이름만 외우기보다, 어느 자리를 말하는지 함께 떠올려주시면 좋겠어요~

![양손으로 하트를 보내는 문](/images/naver-moon/moon-22085.png)

## 참고

- Lindholm, Nickolls, Oberman, Montrym, [*"NVIDIA Tesla: A Unified Graphics and Computing Architecture"*](https://ieeexplore.ieee.org/document/4523358): SPA/TPC/SM/SP 계층, SIMT, 그래픽 데이터 경로의 1차 출처.
- [NVIDIA GeForce 8800 GPU Architecture Technical Brief](https://www.nvidia.com/): G80 클럭, 파티션, register/shared memory 크기.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): 소프트웨어 모델이 현재 아키텍처 하드웨어에 어떻게 매핑되는지.
- [NVIDIA Hopper (H100) Architecture](https://www.nvidia.com/en-us/data-center/h100/): Tesla 계보가 자라난 현대 SM(128 FP32 lane, 4세대 tensor core, TMA).
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, occupancy, pinned memory, host-device 전송을 오늘날 하드웨어에 적용.
- 스티커: LINE [Moon & James](https://store.line.me/stickershop/product/1/en), [LINE Characters in Love!](https://store.line.me/stickershop/product/1252/en).
