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

> Primary: NVIDIA 아키텍처 whitepaper와 제품 페이지. 서사·microarchitecture 해석: Fabien Sanglard, Chips and Cheese, SemiAnalysis.

## 왜 전체 줄기를 훑는가

[primer 글(CUDA 0)](../cuda-0-gpu-architecture/)은 2006년 Tesla 칩을 CUDA 용어의 기준점으로 썼다. 이 글은 그 기준점을 앞으로 따라간다. NVIDIA GPU 아무거나 하나를 한 타임라인 위에 올려놓고, 그게 뭘 바꿨고 왜 바꿨는지 한 문장으로 말할 수 있으면 된다. 이 글이 만들려는 게 그거다.

![NVIDIA GPU architecture family tree, Tesla to Rubin](./images/timeline.svg?v=1)
*가계도: Pascal까지 공유 줄기, Volta에서 datacenter 라인(위)과 graphics 라인(아래)으로 분기.*

전체 줄기는 한 가지 프레이밍으로 보면 깔끔하다. 거의 안 바뀌는 두 가지가 있고, 나머지는 전부 두 가지 압력에 대한 반응이다.

**안 바뀌는 것:** SIMT 실행 모델, 32 thread짜리 warp, SM 하나에서 끝까지 도는 thread block, 그리고 register·shared memory·global DRAM의 메모리 계층. [CUDA C 글](../cuda-c-basics/)에서 이거 한 번 배우면 G80부터 Rubin까지 그대로 간다.

**바뀌는 것은 두 압력이 민다.** 첫째, 워크로드가 그래픽에서 AI로 옮겨가면서 SM은 같은 범용 코어 주변에 specialized unit(Tensor Core, 그다음 RT Core, 그다음 Transformer Engine)을 계속 덧붙였다. 둘째, 다이 하나로는 부족해지면서 설계 단위가 칩 → 2개 다이 → 랙으로 커졌다. 계보의 패턴은 이렇다. 범용 SM은 그대로 있고, 그 주변에 accelerator가 쌓이고, 패키지가 계속 커진다.

이 계보는 대체로 SM을 따라간다. CUDA 프로그램이 SM 위에 스케줄되기 때문이다. GPU의 나머지(L2, memory controller, ROP, copy engine, host interface, fabric)도 성능과 시스템 설계에는 중요하지만, warp 실행·register·shared memory·Tensor Core·RT Core·TMA/TMEM 변화가 드러나는 단위는 SM이다.

![Anatomy of a GPU die, where the SM sits](./images/gpu-anatomy.svg?v=1)
*GPU 다이는 SM 배열을 L2·memory controller·DRAM·graphics 전용 고정기능·host/fabric 인터페이스가 감싼 것이다. 아래 세대별 도식은 전부 그중 SM 하나로 줌인한다.*

## 통합의 시대: Tesla와 Fermi

**Tesla (2006, G80).** 출발점, primer에서 다뤘다. 고정된 vertex/pixel 파이프라인을 프로그래머블 코어의 단일 통합 배열로 바꾼 그 한 아이디어가 GPU를 범용 연산장치로 만들고 CUDA를 가능하게 했다. SM당 SP 8개, warp scheduler 1개, SIMT, 90nm.

![Tesla SM component diagram](./images/sm-tesla.svg?v=1)
*Tesla SM (G80): scalar SP 8개, scheduler 1개, shared 16 KB. 원점.*

**Fermi (2010, GF100).** 그래픽 칩을 작정하고 연산 칩으로 바꾼 세대. Fermi는 진짜 프로그래밍 타깃에 필요한 것들을 붙였다. 제대로 된 L1 데이터 캐시와 L2, ECC, fused multiply-add, 완전한 IEEE 배정밀도, 그리고 C++ 지원. SM은 CUDA 코어 32개에 warp scheduler 2개로 커졌고, texture unit이 SM 안으로 들어왔다. Tesla가 GPU도 연산할 수 있음을 증명했다면, Fermi는 그 위에 수치 라이브러리를 얹을 만한 물건으로 만들었다.

![Fermi SM component diagram](./images/sm-fermi.svg?v=1)
*Fermi SM (GF100): 코어 32개, scheduler 2개, 첫 L1 데이터 캐시.*

## 효율의 시대: Kepler, Maxwell, Pascal

**Kepler (2012, GK110).** 처리량 베팅. Kepler는 SM을 SMX로 대폭 넓혀(CUDA 코어 192개) 명령 스케줄링을 하드웨어에서 컴파일러로 옮겨 전력을 아꼈다. 코어 수와 단순한 스케줄러로 낮은 클럭에서 perf-per-watt를 이긴다는 베팅이었다. 반쯤 맞았다. Kepler는 총량으론 효율적이었지만 먹여 살리기 어려웠고 코어당 활용률이 떨어졌다. "SM이 넓다고 자동으로 빠른 건 아니다"를 말할 때 사람들이 가리키는 세대다.

![Kepler SMX component diagram](./images/sm-kepler.svg?v=3)
*Kepler SMX (GK110): 코어 192개, scheduler 4개, 컴파일러 스케줄링.*

**Maxwell (2014, GM200).** 교정. Maxwell은 SM을 코어 128개로 다시 좁히고 4개의 processing block(각 32개, 자기 scheduler와 register file)으로 분할해서 하드웨어가 warp에 다시 깔끔하게 매핑되게 했다. 새 공정 없이 설계만 정리했는데 NVIDIA 최대 효율 도약 중 하나를 냈다. 정갈한 아키텍처가 무식한 폭을 이길 수 있다는 사례가 Maxwell이다.

**Pascal (2016).** 분기가 눈에 보이기 시작. consumer 파트(GP102, GTX 1080 Ti)는 사실상 Maxwell을 16nm로 옮기고 GDDR5X를 붙인, 공정+대역폭 세대였다. datacenter 파트(GP100, P100)는 다른 물건이었다. SM당 FP32 lane 64개지만 FP64를 갖고, 처음으로 NVLink와 HBM2가 들어왔다. Pascal은 consumer와 datacenter 설계가 "같은 칩 다른 bin"이길 멈춘 지점이다.

![Maxwell and Pascal SM component diagram](./images/sm-maxwell-pascal.svg?v=2)
*Maxwell과 Pascal: SM을 warp 크기 partition 4개로 쪼갬.*

## AI로의 선회: Volta와 Turing

**Volta (2017, GV100).** 계보 전체의 경첩. Volta는 첫 Tensor Core를 붙였다. 작은 행렬 곱셈-누산을 명령 하나로 하는 유닛인데, 일반 FP 명령으로 matmul을 하면 전력 대부분이 연산이 아니라 명령 오버헤드에 새기 때문이다. 그리고 independent thread scheduling을 도입했다. thread마다 자기 program counter를 갖게 됐고, 그래서 CUDA C 글이 warp lockstep에 단서를 달고 `__syncwarp()`를 꺼내야 하는 것이다. Volta는 consumer 파트가 없는 datacenter 전용이었다. 현대 AI 하드웨어의 모든 게 여기서 시작한다.

![Volta SM component diagram](./images/sm-volta.svg?v=1)
*Volta SM (GV100): 첫 Tensor Core가 CUDA 코어 옆에 합류.*

**Turing (2018, TU102).** Volta의 아이디어가 그래픽 라인에 도달. Turing은 2세대 Tensor Core와 ray tracing용 새 RT Core를 consumer GPU에 넣고, datapath를 쪼개 SM이 FP32와 INT32를 동시에 issue할 수 있게 했다. 그래픽 라인이 순수 그래픽이길 멈추고 AI·ray tracing accelerator를 싣기 시작한 순간이고, 그게 DLSS가 나오는 방식이다.

![Turing and Ada SM component diagram](./images/sm-turing-ada.svg?v=1)
*Turing과 Ada: RT Core와 그래픽용 Tensor Core가 SM에 진입.*

## Datacenter 군비경쟁: Ampere와 Hopper

**Ampere (2020, GA100).** 규모와 포맷. 3세대 Tensor Core가 TF32(FP32 범위, 줄인 mantissa, 학습용 drop-in)와 BF16을 더했고, structured sparsity로 2배 처리량을 주장했다. 그만큼 중요한 게 `cp.async`인데, thread가 global에서 shared memory로 register를 거치지 않고 복사하게 해서 Tensor Core 커널을 제한하는 register 압박을 덜었다. Ampere는 MIG도 가져와 A100을 격리된 인스턴스로 쪼갰다. A100은 SM당 FP32 lane 64개, consumer RTX 30은 128개였다.

![Ampere SM component diagram](./images/sm-ampere.svg?v=1)
*Ampere SM (GA100): 3세대 Tensor, cp.async가 shared memory에 급식.*

**Hopper (2022, GH100).** Transformer Engine 세대. 4세대 Tensor Core가 FP8(E4M3, E5M2)을 더했고, Hopper는 그걸 LLM을 정조준한 기계장치로 감쌌다. warpgroup 단위 비동기 MMA(`wgmma`), 단일 thread가 개시하는 대량 비동기 복사기 TMA(Tensor Memory Accelerator), 그리고 SM끼리 직접 데이터를 나누는 distributed shared memory를 가진 thread block cluster. SemiAnalysis의 프레이밍으로, 동기가 되는 문제는 "Tensor Core 처리량은 매 세대 2배인데 global memory 지연은 안 줄었다"는 것이고, 그래서 Hopper는 예산을 raw FLOP이 아니라 은닉과 급식에 썼다. H100, HBM3, NVLink 4 900 GB/s.

![Hopper SM component diagram](./images/sm-hopper.svg?v=2)
*Hopper SM (GH100): FP8 Tensor, TMA, wgmma, thread block cluster.*

## 규모의 시대: Ada, Blackwell, Rubin

**Ada (2022, AD102).** Hopper의 그래픽 라인 짝. 4세대 Tensor Core, 3세대 RT Core, ray tracing divergence를 되찾는 Shader Execution Reordering, 그리고 DLSS 3 frame generation 스택. RTX 4090, TSMC 4nm.

**Blackwell (2024).** 칩 두 개, 이름 하나. datacenter 파트(B200)가 다이 하나이길 멈춘 지점이다. reticle 한계 다이 2개를 10 TB/s 링크로 묶어 소프트웨어엔 GPU 하나로 보이게 했고, 합쳐 2080억 트랜지스터, HBM3e다. consumer 파트(GB202, RTX 5090)는 반대로 750mm² 근처 단일 다이에 SM 192개, GDDR7이고, Chips and Cheese가 "specialization보다 scale"로 읽는 게 이쪽이다. 지연보다 대역폭을 택한 64뱅크 L2(약 8.7 TB/s)로, 코어당 영리함이 아니라 순전한 코어 밀도로 이긴다. 둘 다 5세대 Tensor Core를 공유하는데, FP4(NVFP4와 microscaling MXFP 포맷), operand가 register file 바깥에 사는 전용 Tensor Memory(TMEM), 그리고 SM 둘이 한 행렬 연산을 협력하는 CTA-pair MMA를 더했다. GB200은 datacenter Blackwell GPU 둘에 Grace CPU를 짝짓고, GB200 NVL72는 그런 걸 72개 묶어 단일 rack-scale GPU처럼 동작하는 NVLink 도메인을 만든다.

![Blackwell SM component diagram](./images/sm-blackwell.svg?v=2)
*Blackwell SM (B200): FP4 Tensor, 전용 TMEM, CTA-pair MMA.*

**Rubin (2026).** 현재 세대이자, 설계 단위가 칩에서 랙으로 이동을 완성하는 지점. NVIDIA의 Rubin 자료는 Rubin GPU를 3360억 트랜지스터, HBM4 288GB 22 TB/s, NVLink 6 3.6 TB/s로 제시한다. NVIDIA는 이 공개 spec을 preliminary라고 표시하고, 제품을 단순히 "Rubin GPU" 하나로 설명하지 않는다. platform인 Vera Rubin으로 설명한다. 그 구분을 다음 절에서 한다.

## Tensor Core, 하나의 줄기

계보의 여러 줄기 중, 트랜지스터와 R&D가 실제로 어디로 갔는지 결정한 게 Tensor Core다. 행렬 곱셈-누산 유닛이고, 다섯 세대는 정밀도와 비동기성 두 축으로 움직인다.

정밀도는 매 세대 떨어졌다. AI가 그걸 견디니까. FP16(Volta), INT8/INT4(Turing), TF32와 BF16(Ampere), FP8(Hopper), FP4(Blackwell). 정밀도가 낮으면 트랜지스터당·옮긴 바이트당 연산이 늘고, Transformer Engine은 layer마다 정밀도를 자동으로 바꿔 정확도를 지키려고 존재한다.

미묘한 절반은 Tensor Core가 개수가 아니라 타일 크기로 커졌다는 것이다. 행렬 곱은 대략 $N^3$번 연산하면서 대략 $N^2$만큼 데이터를 옮기므로, arithmetic intensity가 타일 변 길이에 비례해 오른다.

$$I \sim \frac{N^3}{N^2} = N$$

큰 타일일수록 데이터 이동을 잘 amortize하니까, 매 세대 명령이 더 큰 행렬(4x4x4, 그다음 8x8x4, 그다음 16x8x16 그 이상)을 계산하게 만들었지 작은 유닛을 더 찍어내지 않았다. 그리고 Tensor 처리량은 계속 2배가 되는데 메모리 지연은 안 줄어서, 실행 모델이 동기 warp-level MMA(Volta) → 비동기 warpgroup MMA(Hopper의 `wgmma`) → operand가 전용 Tensor Memory에 있는 완전 비동기 single-thread MMA(Blackwell)로 갔다. 줄기 전체에서 병목은 연산이 아니라 연산을 먹이는 일이다.

## 두 라인: consumer와 datacenter

Volta 이후 이 계열은 DNA를 공유하되 다른 걸 최적화하는 두 갈래로 달린다.

**datacenter 라인**(GV100, GA100, GH100, GB200, Rubin)은 AI 처리량과 interconnect를 극대화한다. SM당 FP32 lane은 적지만 INT/FP64/Tensor는 더 많고, GDDR 대신 HBM, NVLink에 이제 랙 전체 fabric, MIG나 thread block cluster 같은 기능. **그래픽 라인**(Turing, Ada, consumer Ampere·Blackwell)은 DLSS용 Tensor Core를 충분히 남기고 RT Core와 렌더링 기능을 더하며 GDDR을 쓴다. Ampere와 Blackwell은 한 이름 아래 두 갈래에 다 존재하고, 그래서 "Ampere GPU"가 A100일 수도 RTX 3090일 수도 있으며 SM이 꽤 다르다. 세대 이름만으론 부족하다. 어느 라인인지 말해야 한다.

## Rubin은 GPU가 아니라 platform이다

최신 항목은 오독하기 쉽다. "Blackwell"은 아직 가리킬 수 있는 GPU 이름이다. "Rubin"은 대체로 시스템 이름이다. NVIDIA 자신의 Vera Rubin 설명은 랙, 즉 NVL72다. Rubin GPU 72개와 Vera CPU 36개를 하나의 액체냉각 NVLink-6 도메인에 넣어 대략 3.6 EFLOPS FP4 추론과 20.7 TB HBM4를 낸다. Vera CPU는 자체 칩으로, 88개의 커스텀 Olympus Arm 코어다. NVIDIA 최신 Vera Rubin 페이지는 이 platform을 compute, networking, storage, switching을 아우르는 seven-chip platform으로 설명한다. 그러니 Rubin을 SM 그림 하나로 끝내려는 사고방식 자체가 이미 틀린 abstraction이다.

![Rubin and Vera Rubin platform diagram](./images/rubin-platform.svg?v=2)
*Rubin은 platform: GPU, Vera CPU, NVLink switch, DPU, Ethernet를 한 랙에.*

그러니 두 이름을 구분해라. Rubin GPU는 microarchitecture로, GB100에 비교할 것이다. Vera Rubin은 co-design된 rack-scale 컴퓨터로, GB200 NVL72에 비교할 것이다. 계보의 종착점은 더 빠른 칩이 아니라, 흥미로운 단위가 이제 랙이라는 인정이다.

![NVIDIA architecture snapshots, Tesla to Rubin](./images/architecture-snapshots.svg?v=2)
*Architecture snapshots: 공유 줄기는 graphics에서 compute로, 위쪽은 datacenter AI, 아래쪽은 RTX graphics.*

## 가계도

| 세대 | 연도 | SM / 코드네임 | 정의적 변화 | 공정 | 대표 |
| --- | --- | --- | --- | --- | --- |
| Tesla | 2006 | SM, 8 SP (G80) | unified shader, SIMT, CUDA | 90 nm | 8800 GTX |
| Fermi | 2010 | SM, 32 (GF100) | L1 데이터캐시, FMA, FP64, C++ | 40 nm | GTX 480 |
| Kepler | 2012 | SMX, 192 (GK110) | 컴파일러 스케줄러, wide SM | 28 nm | K20 |
| Maxwell | 2014 | SMM, 128 (GM200) | 효율, 4x32 partition | 28 nm | GTX 980 Ti |
| Pascal | 2016 | GP100 / GP102 | NVLink, HBM2(GP100), 16nm | 16 nm | P100 |
| Volta | 2017 | GV100, 64 FP32 | 1st Tensor Core, ITS | 12 nm | V100 |
| Turing | 2018 | TU102, 64 FP32 | RT Core + 2nd Tensor를 그래픽으로 | 12 nm | RTX 2080 Ti |
| Ampere | 2020 | GA100, 64 FP32 | 3rd Tensor(TF32/sparsity), MIG | 7 nm | A100 |
| Ada | 2022 | AD102, 128 FP32 | 4th Tensor, 3rd RT, SER | 4 nm | RTX 4090 |
| Hopper | 2022 | GH100, 128 FP32 | Transformer Engine(FP8), TMA, cluster | 4 nm | H100 |
| Blackwell | 2024 | 2 dies, 208B | FP4, TMEM, 5th NVLink, scale-first | TSMC 4NP | B200 / GB200 |
| Rubin | 2026 | 2 dies, 336B | HBM4, NVLink 6; Vera Rubin = rack platform | NVIDIA 공개 spec상 미확정 | Rubin / Vera Rubin NVL72 |

## 계보가 말해주는 것

세 궤적이 전체 길이를 관통한다. **specialization은 증가한다.** SM은 범용 코어를 유지하며 Tensor Core, RT Core, Transformer Engine, 전용 Tensor Memory를 그 주변에 쌓는다. **정밀도는 감소한다.** FP32에서 FP4로. AI가 처리량을 비트로 지불하니까. 그리고 **설계 단위는 커진다.** 칩, 그다음 다이 2개, 그다음 랙.

안 바뀌는 단 하나는 병목이다. CUDA C 글의 coalescing부터 Hopper의 TMA, Blackwell의 TMEM까지, 매 세대가 새 하드웨어 대부분을 raw FLOP이 아니라 데이터를 옮기고 은닉하는 데 쓴다. 연산은 10년째 싸고, 그걸 먹이는 건 안 그렇다. 가계도 전체를 관통하는 한 줄기는 memory wall이다.

## 참고

- [Fabien Sanglard, A history of NVidia Stream Multiprocessor](https://fabiensanglard.net/cuda/): Tesla부터 Turing까지의 서사와 SM 설계 변화.
- [SemiAnalysis, NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell): 정밀도, 비동기성, 타일 크기 논증.
- [Chips and Cheese, Blackwell: NVIDIA's Massive GPU](https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu): scale-over-specialization microarchitecture 해석.
- NVIDIA 1차 architecture 문서: [Fermi](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf), [Kepler GK110](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf), [Maxwell tuning](https://docs.nvidia.com/cuda/maxwell-tuning-guide/), [Pascal GP100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf), [Volta GV100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [Ampere A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), [Ampere GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)와 [Vera Rubin Platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/): 최신 세대의 1차 수치.
- [NVIDIA Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)와 [NVIDIA Rubin platform technical blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/): Rubin GPU, NVLink 6, NVL72, preliminary spec caveat.
