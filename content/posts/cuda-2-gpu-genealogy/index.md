---
title: "01 NVIDIA GPU Architecture Genealogy: Tesla to Rubin"
date: 2026-05-22
draft: false
tags: ["CUDA", "GPU Architecture", "Tensor Core", "NVIDIA", "Video Notes"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "Tesla(2006)에서 Rubin(2026)까지 한 줄기: NVIDIA SM이 무엇을 그대로 유지하고(SIMT, warp=32, SM당 block) 그 위에 어떤 specialized accelerator를 쌓아왔는지, Tensor Core가 다섯 세대에 걸쳐 어떻게 진화했는지, consumer와 datacenter 라인이 왜 갈라졌는지, 그리고 왜 'Rubin'은 GPU가 아니라 platform인지."
---

> 이 글의 1차 자료는 NVIDIA의 아키텍처 whitepaper와 공식 제품 페이지다. 서사와 microarchitecture 해석은 Fabien Sanglard, Chips and Cheese, SemiAnalysis를 참고했다. 전체 출처는 글 끝의 참고 목록에 있다.

## 개요: 계보를 읽는 법

2006년의 Tesla부터 2026년의 Rubin까지, NVIDIA GPU는 20년 동안 십수 개의 아키텍처를 거쳤다. 이름만 나열하면 외울 것이 많아 보이지만, 실제로는 하나의 단순한 프레임으로 전부 정리된다. 이 글의 목표는 그 프레임을 만드는 것이다. 어떤 NVIDIA GPU든 타임라인 위에 올려놓고 "이 세대는 무엇을 바꿨고 왜 바꿨는지"를 한 문장으로 말할 수 있게 되는 것.

이 글은 [primer 글(CUDA 0)](../cuda-0-gpu-architecture/)에서 정리한 2006년 Tesla 칩을 기준점으로 삼아, 그 이후의 변화를 따라간다.

![NVIDIA GPU architecture family tree, Tesla to Rubin](./images/timeline.svg?v=1)
*가계도: Pascal까지는 하나의 줄기를 공유하다가, Volta에서 datacenter 라인(위)과 graphics 라인(아래)으로 갈라진다.*

프레임은 이렇다. **거의 바뀌지 않는 토대가 있고, 나머지 모든 변화는 두 가지 압력에 대한 반응이다.**

바뀌지 않는 토대는 실행 모델과 메모리 계층이다. 명령 하나를 스레드 32개 묶음(warp)이 동시에 실행하는 SIMT(Single Instruction, Multiple Threads) 모델, 하나의 SM(Streaming Multiprocessor — 연산 유닛·스케줄러·shared memory를 묶은, GPU를 구성하는 기본 블록) 위에서 끝까지 실행되는 thread block, 그리고 register → shared memory → global DRAM으로 이어지는 메모리 계층. 이 구조는 [CUDA C 글](../cuda-c-basics/)에서 한 번 배우면 G80부터 Rubin까지 그대로 적용된다. 20년 전에 쓴 CUDA 코드가 지금 GPU에서도 컴파일되는 이유다.

변화를 미는 두 압력은 다음과 같다. 첫째는 **워크로드의 이동**이다. GPU의 주 고객이 그래픽에서 AI로 옮겨가면서, SM은 범용 코어를 유지한 채 그 주변에 특화 유닛을 계속 덧붙였다. Tensor Core가 먼저 왔고, RT Core가 뒤따랐고, Transformer Engine이 그 다음이었다. 둘째는 **규모의 압력**이다. 다이 하나로는 수요를 감당할 수 없게 되면서, 설계 단위가 칩 하나 → 다이 2개 → 랙 전체로 커졌다. 요약하면 계보의 패턴은 이것이다: 범용 SM은 그대로 있고, 그 주변에 accelerator가 쌓이고, 패키지는 계속 커진다.

이 글이 SM을 중심으로 계보를 따라가는 이유는 단순하다. CUDA 프로그램이 스케줄되는 단위가 SM이기 때문이다. GPU의 나머지 부분(L2 캐시, memory controller, ROP, copy engine, host interface, fabric)도 성능과 시스템 설계에 중요하지만, warp 실행·register·shared memory·Tensor Core·RT Core·TMA/TMEM 같은 프로그래머가 체감하는 변화가 드러나는 단위는 SM이다.

![Anatomy of a GPU die, where the SM sits](./images/gpu-anatomy.svg?v=1)
*GPU 다이는 SM 배열을 L2·memory controller·DRAM·graphics 전용 고정기능·host/fabric 인터페이스가 감싼 구조다. 아래의 세대별 도식은 모두 그중 SM 하나를 확대한 것이다.*

## 통합의 시대: Tesla와 Fermi

**Tesla (2006, G80)** 는 이 계보의 출발점이다. 그 이전의 GPU는 vertex 처리와 pixel 처리를 각각 전담하는 고정 파이프라인을 갖고 있었다. Tesla는 이 둘을 프로그래머블 코어의 단일 통합 배열로 교체했다. 이 하나의 결정이 GPU를 그래픽 전용 장치에서 범용 연산 장치로 바꿨고, CUDA라는 프로그래밍 모델을 가능하게 했다. 구성은 오늘날 기준으로는 소박하다. SM당 scalar processor(SP) 8개, warp scheduler 1개, 90nm 공정.

![Tesla SM component diagram](./images/sm-tesla.svg?v=1)
*Tesla SM (G80): scalar SP 8개, scheduler 1개, shared memory 16 KB. 모든 것의 원점.*

**Fermi (2010, GF100)** 는 그래픽 칩을 본격적인 연산 칩으로 바꾼 세대다. "GPU로도 계산이 된다"를 넘어 "GPU를 진지한 프로그래밍 타깃으로 쓸 수 있다"로 가려면 필요한 것들이 있었고, Fermi가 그것들을 붙였다. 제대로 된 L1 데이터 캐시와 L2 캐시, ECC 메모리, fused multiply-add(FMA), IEEE 표준을 완전히 따르는 배정밀도(FP64) 연산, 그리고 C++ 지원. SM 자체도 CUDA 코어 32개, warp scheduler 2개로 커졌고, texture unit이 SM 내부로 들어왔다. Tesla가 GPU 연산의 가능성을 증명했다면, Fermi는 그 위에 수치 라이브러리를 얹을 만한 물건을 만들었다.

![Fermi SM component diagram](./images/sm-fermi.svg?v=1)
*Fermi SM (GF100): 코어 32개, scheduler 2개, GPU 최초의 L1 데이터 캐시.*

## 효율의 시대: Kepler, Maxwell, Pascal

**Kepler (2012, GK110)** 는 처리량에 베팅한 세대다. SM을 SMX라는 이름으로 대폭 넓혀 CUDA 코어를 192개까지 늘리는 대신, 명령 스케줄링의 상당 부분을 하드웨어에서 컴파일러로 옮겨 전력을 아꼈다. 많은 코어와 단순한 스케줄러를 낮은 클럭으로 돌려 전력 대비 성능(perf-per-watt)에서 이긴다는 계산이었다. 결과는 절반의 성공이었다. 총량 기준으로는 효율적이었지만, 192개의 코어를 쉬지 않고 먹여 살리기가 어려워 코어당 활용률이 떨어졌다. 지금도 "SM이 넓다고 자동으로 빠른 것은 아니다"라는 교훈을 이야기할 때 사람들이 가리키는 세대가 Kepler다.

![Kepler SMX component diagram](./images/sm-kepler.svg?v=3)
*Kepler SMX (GK110): 코어 192개, scheduler 4개, 컴파일러 주도 스케줄링.*

**Maxwell (2014, GM200)** 은 Kepler의 과욕을 교정한 세대다. SM을 코어 128개로 다시 좁히고, 이를 각각 32개 코어·전용 scheduler·전용 register file을 가진 4개의 processing block으로 분할했다. 32는 warp의 크기이므로, 이 분할로 하드웨어가 warp에 다시 깔끔하게 1:1로 매핑됐다. 새로운 공정 없이 설계 정리만으로 NVIDIA 역사상 손꼽히는 효율 도약을 만들어냈다. 정갈한 아키텍처가 무식한 폭을 이길 수 있다는 사례로 Maxwell이 인용되는 이유다. 이때 자리잡은 "SM = warp 크기 partition 4개" 구조는 이후 세대에도 그대로 이어진다.

**Pascal (2016)** 부터는 라인 분기가 눈에 보이기 시작한다. consumer 파트(GP102, GTX 1080 Ti)는 사실상 Maxwell 설계를 16nm 공정으로 옮기고 GDDR5X를 붙인 공정·대역폭 업그레이드였다. 반면 datacenter 파트(GP100, P100)는 다른 물건이었다. SM당 FP32 lane은 64개로 줄었지만 강력한 FP64 유닛을 갖췄고, GPU 간 고속 연결인 NVLink와 고대역폭 메모리 HBM2가 처음으로 들어왔다. consumer와 datacenter 칩이 "같은 칩의 다른 bin"(bin: 같은 다이를 수율·성능에 따라 등급을 나눠 서로 다른 제품으로 파는 것)이기를 멈춘 지점이 Pascal이다.

![Maxwell and Pascal SM component diagram](./images/sm-maxwell-pascal.svg?v=2)
*Maxwell과 Pascal: SM을 warp 크기 partition 4개로 분할한 구조가 여기서 자리잡는다.*

## AI로의 선회: Volta와 Turing

**Volta (2017, GV100)** 는 계보 전체의 경첩이다. 여기서 첫 Tensor Core가 등장한다. Tensor Core는 작은 행렬의 곱셈-누산(matrix multiply-accumulate, 이하 MMA)을 명령 하나로 처리하는 전용 유닛이다. 이것이 필요했던 이유는, 일반 FP 명령으로 행렬곱을 수행하면 전력 대부분이 실제 연산이 아니라 명령 fetch/decode/schedule 오버헤드로 새기 때문이다. 연산 하나하나를 명령으로 발행하는 대신 행렬 단위로 묶어버리면 이 오버헤드가 사라진다.

Volta의 두 번째 유산은 independent thread scheduling이다. 이때부터 warp 안의 thread가 각자의 program counter를 갖게 됐다. warp의 모든 스레드가 같은 명령을 같은 박자로 실행한다는 lockstep 가정이 이때 깨졌고, CUDA C 글에서 warp lockstep에 단서를 달고 `__syncwarp()`를 설명해야 하는 이유가 바로 이것이다. Volta는 consumer 파트 없이 datacenter 전용으로만 나왔다. 현대 AI 하드웨어의 모든 것이 이 세대에서 시작한다.

![Volta SM component diagram](./images/sm-volta.svg?v=1)
*Volta SM (GV100): 첫 Tensor Core가 CUDA 코어 옆에 합류한다.*

**Turing (2018, TU102)** 은 Volta의 아이디어가 그래픽 라인에 도달한 세대다. 2세대 Tensor Core와, ray tracing 연산을 전담하는 새로운 RT Core를 consumer GPU에 넣었다. datapath도 분리되어 SM이 FP32 연산과 INT32 연산을 동시에 발행할 수 있게 됐다(주소 계산 같은 정수 연산이 FP 연산과 섞여 나오는 실제 워크로드에서 효과가 크다). 그래픽 라인이 순수 그래픽이기를 멈추고 AI·ray tracing accelerator를 싣기 시작한 순간이며, DLSS(저해상도로 렌더링한 프레임을 신경망으로 업스케일해 성능을 버는 기능) 같은 것이 가능해진 것도 이 하드웨어 덕분이다.

![Turing and Ada SM component diagram](./images/sm-turing-ada.svg?v=1)
*Turing과 Ada: RT Core와 그래픽용 Tensor Core가 SM에 들어온다.*

## Datacenter 군비경쟁: Ampere와 Hopper

**Ampere (2020, GA100)** 의 키워드는 규모와 포맷이다. 3세대 Tensor Core는 TF32(FP32의 지수 범위를 유지하되 mantissa를 줄인, 학습 코드가 수정 없이 쓸 수 있는 포맷)와 BF16을 추가했고, structured sparsity(가중치를 일정 패턴으로 절반을 0으로 만든 뒤 그 0들을 건너뛰어 계산하는 기법)로 조건부 2배 처리량을 제공했다. 포맷만큼 중요한 것이 `cp.async` 명령이다. 이전에는 global memory에서 shared memory로 데이터를 옮기려면 register를 경유해야 했는데, `cp.async`는 이 복사를 register를 거치지 않고 수행한다. Tensor Core 커널의 고질적 병목이던 register 압박을 덜어준 것이다. 이 외에도 MIG(Multi-Instance GPU)가 추가되어 A100 하나를 완전히 격리된 여러 GPU 인스턴스로 나눌 수 있게 됐다. 참고로 같은 Ampere라도 datacenter A100은 SM당 FP32 lane 64개, consumer RTX 30 시리즈는 128개로 SM 구성이 다르다. 이 차이는 아래 [두 라인](#두-라인-consumer와-datacenter) 절에서 다시 다룬다.

![Ampere SM component diagram](./images/sm-ampere.svg?v=1)
*Ampere SM (GA100): 3세대 Tensor Core, 그리고 shared memory에 직접 급식하는 cp.async.*

**Hopper (2022, GH100)** 는 Transformer Engine 세대다. Transformer Engine은 layer마다 FP8과 FP16 중 적정 정밀도를 자동으로 골라주는 하드웨어+소프트웨어 조합으로, 낮은 정밀도의 속도를 취하되 정확도 붕괴를 막는 안전장치다. 4세대 Tensor Core가 FP8 포맷(E4M3, E5M2)을 추가했고, Hopper는 그것을 LLM 워크로드를 정조준한 기계장치로 감쌌다. warpgroup(warp 4개 묶음) 단위로 실행되는 비동기 행렬 명령 `wgmma`, 단일 thread가 개시하면 하드웨어가 알아서 수행하는 대량 비동기 복사 엔진 TMA(Tensor Memory Accelerator), 그리고 여러 SM이 shared memory를 직접 주고받을 수 있게 하는 thread block cluster와 distributed shared memory. SemiAnalysis의 프레이밍을 빌리면, 이 모든 것의 동기는 "Tensor Core 처리량은 매 세대 2배가 되는데 global memory 지연은 줄지 않는다"는 문제다. 그래서 Hopper는 트랜지스터 예산을 raw FLOP이 아니라 지연 은닉과 데이터 공급에 썼다. 대표 제품은 H100으로, HBM3와 900 GB/s NVLink 4를 갖췄다.

![Hopper SM component diagram](./images/sm-hopper.svg?v=2)
*Hopper SM (GH100): FP8 Tensor Core, TMA, wgmma, thread block cluster.*

## 규모의 시대: Ada, Blackwell, Rubin

**Ada (2022, AD102)** 는 Hopper와 같은 해에 나온 그래픽 라인의 짝이다. 4세대 Tensor Core와 3세대 RT Core를 실었고, ray tracing에서 발생하는 thread divergence를 하드웨어가 재정렬해 회복하는 Shader Execution Reordering(SER), 그리고 DLSS 3 frame generation 스택이 추가됐다. 대표 제품은 RTX 4090, TSMC 4nm 공정이다.

**Blackwell (2024)** 은 "칩 두 개, 이름 하나"로 요약된다. datacenter 파트(B200)는 GPU가 다이 하나이기를 멈춘 지점이다. reticle 한계(노광 장비가 한 번에 찍을 수 있는 최대 크기)까지 키운 다이 2개를 10 TB/s 링크로 묶어, 소프트웨어에는 단일 GPU로 보이게 했다. 합산 2080억 트랜지스터에 HBM3e를 쓴다. 반면 consumer 파트(GB202 계열, RTX 5090)는 정반대 접근으로, 750mm² 근처의 단일 다이에 GDDR7을 쓴다. GB202 풀다이는 192 SM 규모이며, RTX 5090의 출하 설정은 그중 170 SM(21,760 CUDA core)을 활성화한 것이다. Chips and Cheese는 이 설계를 "specialization보다 scale"로 읽는다. 지연보다 대역폭을 택한 64뱅크 L2 캐시(약 8.7 TB/s)를 갖추고, 코어당 영리함이 아니라 순전한 코어 밀도로 승부한다는 해석이다.

두 파트는 5세대 Tensor Core를 공유한다. 5세대의 추가분은 FP4(NVFP4와 microscaling MXFP 포맷), 행렬 operand를 register file 바깥의 전용 저장소에 두는 Tensor Memory(TMEM), 그리고 SM 두 개가 하나의 행렬 연산을 협력 수행하는 CTA-pair MMA(CTA는 thread block의 하드웨어 쪽 이름)다. 시스템 쪽에서는 GB200이 datacenter Blackwell GPU 2개와 Grace CPU 1개를 한 모듈로 묶고, GB200 NVL72는 그 모듈을 72 GPU 규모로 엮어 단일 rack-scale GPU처럼 동작하는 NVLink 도메인을 만든다.

![Blackwell SM component diagram](./images/sm-blackwell.svg?v=2)
*Blackwell SM (B200): FP4 Tensor Core, 전용 TMEM, CTA-pair MMA.*

**Rubin (2026)** 은 현재 세대이자, 설계 단위가 칩에서 랙으로 이동하는 과정을 완성하는 지점이다. NVIDIA의 공개 자료는 Rubin GPU를 3360억 트랜지스터, HBM4 288GB 22 TB/s, NVLink 6 3.6 TB/s로 제시한다. 단, NVIDIA는 이 공개 spec을 preliminary(잠정)로 표시하고 있고, 제품을 "Rubin GPU" 단일 칩으로 설명하지도 않는다. Vera Rubin이라는 platform으로 설명한다. 이 구분이 왜 중요한지는 [아래 절](#rubin-랙-스케일-platform)에서 다룬다.

## Tensor Core의 진화

계보의 여러 줄기 중에서 트랜지스터와 R&D 예산이 실제로 어디로 갔는지를 결정한 것은 Tensor Core다. Tensor Core는 행렬 곱셈-누산 유닛이며, Volta부터 Blackwell까지 다섯 세대의 진화는 **정밀도**와 **비동기성**이라는 두 축으로 정리된다.

정밀도는 매 세대 내려갔다. FP16(Volta) → INT8/INT4(Turing) → TF32와 BF16(Ampere) → FP8(Hopper) → FP4(Blackwell). AI 워크로드가 낮은 정밀도를 견디기 때문에 가능한 이동이다. 정밀도가 절반이 되면 트랜지스터당, 그리고 메모리에서 옮긴 바이트당 연산량이 배로 늘어난다. Hopper 절에서 소개한 Transformer Engine이 바로 이 이동의 안전장치다.

놓치기 쉬운 절반은 Tensor Core가 **개수가 아니라 타일 크기로 커졌다**는 점이다. 행렬 곱은 대략 $N^3$번의 연산을 하면서 $N^2$만큼의 데이터를 옮기므로, 데이터 이동 1회당 연산량(arithmetic intensity)이 타일 변 길이에 비례해 오른다.

$$I \sim \frac{N^3}{N^2} = N$$

즉 타일이 클수록 데이터 이동 비용이 잘 상각된다. 그래서 NVIDIA는 작은 유닛을 더 찍어내는 대신, 명령 하나가 계산하는 행렬을 매 세대 키웠다(4×4×4 → 8×8×4 → 16×8×16, 그 이상). 실제로 SM당 Tensor Core 개수는 Volta의 8개에서 Ampere 이후 4개로 오히려 줄었는데, 유닛 하나가 훨씬 커졌기 때문에 처리량은 세대마다 배가 됐다.

실행 모델도 같은 이유로 진화했다. Tensor 처리량은 계속 2배가 되는데 메모리 지연은 줄지 않으니, 연산과 데이터 이동을 겹치는 능력이 관건이 된다. 그래서 동기식 warp-level MMA(Volta)에서 비동기 warpgroup MMA(Hopper의 `wgmma`)로, 다시 operand가 전용 Tensor Memory에 상주하는 완전 비동기 single-thread MMA(Blackwell)로 이동했다. 계보 전체를 관통하는 명제는 이것이다: **병목은 연산이 아니라, 연산을 먹이는 일이다.**

## 두 라인: consumer와 datacenter

Volta 이후 이 계보는 DNA를 공유하되 서로 다른 것을 최적화하는 두 갈래로 달린다.

**Datacenter 라인**(GV100 → GA100 → GH100 → B200 → Rubin)은 AI 처리량과 interconnect를 극대화한다. SM당 FP32 lane은 적은 대신 INT32·FP64·Tensor Core 비중이 크고, 메모리는 GDDR 대신 HBM을 쓰며, NVLink에서 시작해 이제는 랙 전체를 잇는 fabric(칩과 노드들을 하나로 묶는 통신망)을 갖추고, MIG나 thread block cluster 같은 datacenter 전용 기능을 싣는다.

**Graphics 라인**(TU102 → GA102 → AD102 → GB202)은 DLSS를 돌릴 만큼의 Tensor Core를 남기고, RT Core와 렌더링 기능을 더하며, GDDR 메모리를 쓴다.

주의할 점: Ampere와 Blackwell이라는 이름은 두 갈래 모두에 존재한다. 그래서 "Ampere GPU"라는 말은 A100을 가리킬 수도, RTX 3090을 가리킬 수도 있으며, 이 둘의 SM은 상당히 다르다(FP32 lane 64 vs 128). 세대 이름만으로는 부족하고, 어느 라인인지 함께 말해야 정확하다.

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

## 세대별 SM 제원표

위 가계도가 "각 세대의 한 줄 요약"이라면, 아래 표는 SM 내부가 실제로 어떻게 구성되어 있는지를 NVIDIA whitepaper 수치로 정리한 것이다. **굵은 값은 같은 라인의 직전 세대 대비 바뀐 항목**이다. 표를 세로로 훑으면 어떤 항목이 언제 바뀌었는지가 바로 보인다.

먼저 공유 줄기와 datacenter 라인:

| 칩 (연도) | Partition 구성 | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | Scheduler/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G80 (2006) | 단일 | 8 | FP32와 겸용 | — | — | 1 | 16 KB (shared 전용, L1 없음) | 32 KB |
| GF100 (2010) | 단일 | **32** | 〃 | **16 FMA/clk** | — | **2** | **64 KB (shared/L1 겸용, 48+16 분할)** | **128 KB** |
| GK110 (2012) | 단일 | **192** | 〃 | **64** | — | **4** | 64 KB 겸용 **+ 48 KB read-only** | **256 KB** |
| GM200 (2014) | **4 × 32 (첫 분할)** | **128** | 〃 | **4** | — | 4 | **96 KB (shared 전용, L1 분리)** | 256 KB |
| GP100 (2016) | **2 × 32** | **64** | 〃 | **32** | — | **2** | **64 KB (shared 전용)** | 256 KB |
| GV100 (2017) | **4 × 16** | 64 | **64 (전용 datapath 분리)** | 32 | **8 (1세대, FP16)** | **4** | **128 KB (shared+L1 통합)** | 256 KB |
| GA100 (2020) | 4 × 16 | 64 | 64 | 32 | **4 (3세대, TF32/BF16, 유닛 대형화)** | 4 | **192 KB 통합 (shared 최대 164 KB)** | 256 KB |
| GH100 (2022) | **4 × 32** | **128** | 64 | **64** | **4 (4세대, FP8, wgmma)** | 4 | **256 KB 통합 (shared 최대 228 KB)** | 256 KB |
| B200 (2024) | 4 × 32 | 128 | **128** | 64 | **4 (5세대, FP4, tcgen05)** | 4 | 256 KB | 256 KB **+ TMEM 256 KB** |

다음으로 graphics 라인 (Turing에서 분기):

| 칩 (연도) | Partition 구성 | FP32/SM | INT32/SM | FP64/SM | Tensor/SM | RT Core/SM | Shared+L1 | Register file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TU102 (2018) | 4 × 16 | 64 | 64 (동시 발행) | 2 | 8 (2세대, INT8/4) | 1 (1세대) | 96 KB 통합 | 256 KB |
| GA102 (2020) | **4 × 32** | **128 (64 전용 + 64 INT 겸용)** | 64 | 2 | **4 (3세대)** | 1 **(2세대)** | **128 KB 통합** | 256 KB |
| AD102 (2022) | 4 × 32 | 128 | 64 | 2 | 4 **(4세대, FP8)** | 1 **(3세대)** | 128 KB 통합 | 256 KB |
| GB202 (2024) | 4 × 32 | **128 (전 코어 FP32/INT32 통합)** | **128** | 2 | 4 **(5세대, FP4)** | 1 **(4세대)** | 128 KB 통합 | 256 KB |

표에서 읽을 수 있는 흐름 몇 가지. FP32 개수는 8 → 32 → 192 → 128 → 64 → 128로 오르내렸지만, register file은 Kepler 이후 256 KB에서 한 번도 바뀌지 않았고 partition 구조는 Maxwell 이후 "warp 크기 배수 × 4" 틀을 유지한다. 반면 shared memory는 16 KB에서 228 KB까지 꾸준히 늘었고, Blackwell은 아예 TMEM이라는 새 저장소를 추가했다. 본문의 명제("연산은 그대로, 데이터 공급에 투자")가 표의 숫자로도 확인되는 셈이다.

몇 가지 각주. GF100의 FP64는 전용 유닛 개수 대신 클럭당 FMA 처리량(16 DFMA/clk)으로 공개되어 그렇게 적었다. GA100에서 Tensor Core가 8개에서 4개로 줄어든 것은 퇴보가 아니라 유닛 하나의 타일 크기가 커진 결과다(본문 Tensor Core 절 참고). B200은 이 글 작성 시점 기준 datacenter Blackwell의 SM 단위 whitepaper가 공개되지 않아, NVIDIA 기술 블로그와 공개 교육 자료의 수치를 썼다. Rubin은 SM 세부가 미공개라 표에서 제외했다.

## Rubin: 랙-스케일 platform

계보의 최신 항목은 오독하기 쉽다. "Blackwell"은 아직 손가락으로 가리킬 수 있는 GPU의 이름이다. 반면 "Rubin"은 대체로 시스템의 이름이다. NVIDIA 스스로가 Vera Rubin을 설명할 때 쓰는 단위는 랙, 즉 NVL72다. Rubin GPU 72개와 Vera CPU 36개를 하나의 액체냉각 NVLink-6 도메인에 넣어 약 3.6 EFLOPS의 FP4 추론 성능과 20.7 TB의 HBM4를 제공한다. Vera CPU는 88개의 커스텀 Olympus Arm 코어를 가진 독립 칩이다. NVIDIA의 최신 Vera Rubin 페이지는 이 platform을 compute, networking, storage, switching을 아우르는 seven-chip platform으로 설명한다. Rubin을 SM 다이어그램 하나로 요약하려는 시도 자체가 이미 잘못된 abstraction 레벨인 이유다.

![Rubin and Vera Rubin platform diagram](./images/rubin-platform.svg?v=2)
*Rubin은 platform이다: GPU, Vera CPU, NVLink switch, DPU, Ethernet이 한 랙에 들어간다.*

그러니 두 이름을 구분해서 쓰자. **Rubin GPU**는 microarchitecture이며, 비교 대상은 GB100이다. **Vera Rubin**은 co-design된 rack-scale 컴퓨터이며, 비교 대상은 GB200 NVL72다. 이 계보의 종착점은 더 빠른 칩이 아니라, 이제 흥미로운 단위가 랙이라는 인정이다.

![NVIDIA architecture snapshots, Tesla to Rubin](./images/architecture-snapshots.svg?v=2)
*Architecture snapshots: 공유 줄기는 graphics에서 compute로 이동하고, 위쪽 가지는 datacenter AI, 아래쪽 가지는 RTX graphics로 갈라진다.*

## 종합: 세 가지 궤적

계보 전체를 관통하는 궤적은 세 개다. 첫째, **specialization(특화)은 증가한다.** 모든 연산을 범용 코어 하나로 처리하는 대신, 특정 연산 전용 하드웨어를 따로 두는 방향으로의 이동이다. SM은 범용 코어를 유지하면서 Tensor Core, RT Core, Transformer Engine, 전용 Tensor Memory를 주변에 쌓아왔다. 둘째, **정밀도는 감소한다.** FP32에서 FP4까지. AI가 처리량의 대가를 비트로 지불할 수 있기 때문이다. 셋째, **설계 단위는 커진다.** 칩에서 다이 2개로, 다시 랙으로.

그리고 바뀌지 않는 단 하나는 병목의 위치다. CUDA C 글의 memory coalescing부터 Hopper의 TMA, Blackwell의 TMEM까지, 매 세대는 새 하드웨어 예산의 대부분을 raw FLOP이 아니라 데이터를 옮기고 지연을 은닉하는 데 썼다. 연산은 10년째 싸다. 그 연산을 먹이는 일은 여전히 비싸다. 가계도 전체를 관통하는 한 줄기는 memory wall이다.

## 참고

- [Fabien Sanglard, A history of NVidia Stream Multiprocessor](https://fabiensanglard.net/cuda/): Tesla부터 Turing까지의 서사와 SM 설계 변화.
- [SemiAnalysis, NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell): 정밀도, 비동기성, 타일 크기 논증.
- [Chips and Cheese, Blackwell: NVIDIA's Massive GPU](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu): scale-over-specialization microarchitecture 해석.
- NVIDIA 1차 architecture 문서: [Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf), [Kepler GK110](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf), [Maxwell tuning](https://docs.nvidia.com/cuda/maxwell-tuning-guide/), [Pascal GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [Volta GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [Ampere A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [Ampere GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [H100 whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf), [GeForce RTX Blackwell](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)와 [Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/): 최신 세대의 1차 수치.
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)와 [NVIDIA Rubin platform technical blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/): Rubin GPU, NVLink 6, NVL72, preliminary spec caveat.
- [Cornell Virtual Workshop, B200 SM](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm): B200 SM 구성 수치.
