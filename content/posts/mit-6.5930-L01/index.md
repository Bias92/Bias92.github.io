---
title: "6.5930 L01 - Introduction and Applications"
date: 2026-03-07
tags: ["MLSys", "DNN-Accelerator", "MIT-6.5930"]
draft: false
---

## Ilya Sutskever의 한 마디 (L01-3)

![Ilya Sutskever Quote](images/L01-3-ilya-quote.png)

강의 초반에 Ilya Sutskever의 말을 하나 가져온다.

"Compute has been the oxygen of deep learning."

2017년 ACM 튜링상 50주년 행사에서 나온 말이다. 좋은 알고리즘만으로는 아무것도 못 돌린다. 결국 연산 자원이 받쳐줘야 한다. 나는 이 문장을 MLSys가 왜 필요한지에 대한 가장 짧은 답으로 읽었다.

모델은 계속 바뀐다. CNN에서 Transformer로 넘어왔고, 이제는 MoE나 Mamba 같은 구조도 흔하다. 그때마다 모델 쪽 지식은 크게 흔들리지만, 모델을 실제 장비에서 효율적으로 돌리는 문제는 없어지지 않는다. AI-resistant한 커리어라는 표현이 조금 거창해 보여도, MLSys가 오래 갈 분야라는 얘기에는 동의한다.

## AI Ingredients: 3대 요소 (L01-2)

![AI Ingredients](images/L01-2-ai-ingredients.png)

강의는 지금의 AI를 만든 재료를 Big Data, GPU Acceleration, New ML Techniques 세 가지로 묶는다.

1. Big Data. Facebook에는 하루 3.5억 장의 이미지가 올라오고, YouTube에는 분당 300시간 분량의 영상이 쌓이며, Walmart는 시간당 2.5PB의 데이터를 처리한다. 모델이 먹을 데이터가 이 정도로 쌓였기 때문에 대규모 학습도 가능해졌다.

2. GPU Acceleration. Tesla T4 같은 GPU는 행렬곱을 대량으로 병렬 처리한다. CPU만으로는 감당하기 어려웠던 학습량을 현실적인 시간 안으로 끌어왔다.

3. New ML Techniques. AlexNet이 CNN 시대를 열었고 Transformer가 지금의 LLM을 만들었다. 알고리즘이 바뀌면 잘 맞는 하드웨어의 조건도 같이 바뀐다.

셋 중 하나만 빠져도 지금 규모의 AI는 나오기 어렵다. 이 수업은 그중 compute 쪽을 파고든다.

## GPU의 DNN 특화 진화 (L01-4)

![GPU Evolution](images/L01-4-gpu-evolution.png)

Pascal 시기까지 행렬곱은 주로 범용 CUDA Core가 맡았다. 2017년 V100에는 Tensor Core가 들어갔다. CUDA Core에서 스칼라 FMA를 반복하는 대신 작은 행렬의 multiply-accumulate를 전용 유닛에서 처리하는 방식이다. 같은 면적에서 DNN 연산 처리량을 훨씬 높일 수 있었고, FP16 지원도 이득을 키웠다.

2020년 A100에서는 Tensor Core가 2:4 structured sparsity를 지원하기 시작했다. 연속된 네 원소 중 두 개가 0인 패턴을 맞추면 0인 항을 건너뛰고 유효한 값만 계산한다.

여기서 아무 sparse matrix나 빨라지는 건 아니다. pruning 결과가 하드웨어가 이해하는 2:4 패턴에 맞아야 한다. 알고리즘과 하드웨어를 따로 설계할 수 없는 이유가 이런 데서 나온다.

## SW 기업의 자체 HW (L01-5)

![Software Companies Building HW](images/L01-5-sw-companies-hw.png)

Google이나 Amazon은 NVIDIA GPU를 대량으로 사는 회사이면서 동시에 직접 칩도 만든다. 이유는 비용과 최적화다.

GPU는 그래픽부터 과학 계산까지 다 받아야 하는 범용 장치다. 반대로 데이터센터에서 돌릴 작업이 DNN inference와 training으로 좁혀져 있다면 그 연산에만 맞춘 칩을 만드는 편이 전력과 면적 면에서 유리할 수 있다.

Google TPU v1은 inference용으로 나왔고 256×256 systolic array를 사용했다. v2부터는 training도 지원했다. Amazon은 AWS의 inference와 training 비용을 낮추려고 Inferentia와 Trainium을 만들었다. 국내의 FuriosaAI와 Rebellions도 같은 문제를 각자 다른 방식으로 풀고 있다.

## Cerebras WSE (L01-6)

![Cerebras WSE](images/L01-6-cerebras-wse.png)

Cerebras WSE는 이 방향을 끝까지 밀어붙인 사례다. 보통 칩은 웨이퍼에서 작은 die를 잘라 만들지만, Cerebras는 웨이퍼 전체를 칩 하나로 쓴다. 덕분에 온칩 SRAM만 18GB다. A100의 온칩 SRAM이 20MB 안팎이고 TPU도 32MB 이내라는 점을 생각하면 규모가 다르다.

물론 공짜는 아니다. 수율, 냉각, 가격 문제가 따라온다. 널리 쓰이는 정답이라기보다는 특정 워크로드를 위해 어디까지 과감해질 수 있는지 보여주는 설계에 가깝다.

## Mobile SoCs for DNNs (L01-7)

![Mobile SOCs](images/L01-7-mobile-soc.png)

DNN 전용 유닛은 데이터센터에만 있지 않다. 폰과 노트북에도 이미 들어가 있다.

Apple A11은 2017년에 처음으로 2코어 ANE를 탑재했다. Face ID와 Animoji 같은 on-device inference가 주된 용도였다. 2022년 M2의 ANE는 16코어다. A11 대비 26배라는 성능 향상에는 코어 수 증가와 세대별 아키텍처 개선이 함께 들어간다.

모바일에서는 모델 크기, 정수 양자화, pruning 가능성이 특히 중요하다. 서버처럼 전력과 메모리를 넉넉하게 쓸 수 없어서다.

## 컴퓨팅 에너지 소비 증가 (L01-8)

![alt text](image-1.png)

슬라이드는 미국 데이터센터의 전력 비중이 2022년 3%에서 2030년 8%까지 늘어날 수 있다는 Goldman Sachs의 2024년 전망을 인용한다. ICT 전체가 세계 전력의 20.9%를 차지할 수 있다는 예측도 함께 나온다.

모델은 계속 커지는데 전력은 무한하지 않다. 같은 연산을 더 적은 에너지로 끝내는 게 시스템 문제로 올라온 이유다.

여기서 연산 자체보다 data movement가 비싸다는 사실이 중요하다. 슬라이드의 65nm 기준 수치에서는 DRAM 접근 한 번이 ALU 연산보다 약 200배 많은 에너지를 먹는다. 연산기를 빠르게 만드는 것만으로는 부족하다. 데이터를 멀리 보내는 횟수부터 줄여야 한다.

그래서 Cerebras는 온칩 SRAM을 크게 만들었고, FlashAttention은 전체 attention matrix를 HBM에 쓰지 않도록 타일 단위로 계산한다. 일반적인 accelerator도 register file, local buffer, global buffer, DRAM 순서로 메모리 계층을 두고 가까운 곳의 데이터를 최대한 재사용한다.

## Computing Cost of ChatGPT (L01-12)

GPT-3는 96개 레이어와 175B 파라미터를 가진다. 학습에 필요한 부동소수점 연산은 총 3.14×10²³ FLOPs로 잡는다.

슬라이드의 계산으로는 V100 한 장에서 355년이 걸리고, 클라우드 비용은 약 4.6M 달러다. 한화로는 약 60억 원이다. GPT-4의 학습 비용은 100M 달러 이상으로 추정한다.

숫자의 정확한 범위보다 중요한 건 규모다. 모델 크기가 한 세대 오를 때 연산 비용도 사람이 감당하기 어려운 속도로 커졌다.

## Changing Trends: DeepSeek (L01-13)

![](2026-03-07-14-59-08.png)

슬라이드는 GPT-4의 추정 학습비와 DeepSeek의 공개 비용을 나란히 놓는다. GPT-4는 1,300억 원 이상, DeepSeek는 약 80억 원이라는 숫자다.

다만 둘은 같은 항목이 아니다. DeepSeek 쪽은 최종 학습 런의 GPU 렌탈비에 가깝고, GPT-4 쪽은 훨씬 넓은 범위를 포함한 총비용 추정치다. 그대로 나눠서 몇 배 싸다고 말하면 비교가 틀어진다.

DeepSeek의 구조에서 눈에 띄는 건 MoE다. 전체 파라미터는 671B지만 토큰 하나를 처리할 때는 37B만 활성화한다. 모든 expert를 매번 돌리지 않고 필요한 일부만 골라 계산량을 줄인다.

## Training vs Inference (L01-15)

![](2026-03-07-15-05-37.png)

Training은 한 번의 비용이 크지만 모델 하나를 기준으로 실행 횟수는 적다. Inference는 한 번의 비용이 상대적으로 작아도 서비스가 살아 있는 동안 계속 호출된다.

챗봇에 질문 하나를 보내 답을 받는 것도 inference 한 번이다. 이런 요청이 매일 수조 번 쌓이면 누적 비용은 training보다 더 큰 문제가 된다. LLM inference 최적화 직군이 따로 생기는 이유도 여기 있다.

## On-Device의 장점 (L01-18)

![](2026-03-07-15-09-09.png)

클라우드 대신 디바이스에서 직접 추론할 이유는 세 가지다.

1. Communication. 네트워크가 없거나 불안정한 곳에서는 클라우드 모델을 호출할 수 없다.

2. Privacy. 의료 기록이나 국방 데이터처럼 밖으로 보내기 곤란한 정보는 디바이스 안에서 처리하는 편이 안전하다.

3. Latency. 자율주행처럼 즉시 반응해야 하는 시스템에서는 수십에서 수백 ms의 왕복 지연도 치명적이다.

## Self-Driving Cars (L01-19)

자율주행은 edge inference의 요구가 한꺼번에 드러나는 예다.

카메라와 레이더는 약 30초마다 6GB의 데이터를 만든다. 자동차 한 대가 10개 DNN을 60Hz로, 카메라 10대에 대해 돌린다고 가정하면 시간당 inference 횟수는 21.6M이다. 차량이 100만 대면 시간당 21.6조 번이다.

프로토타입 컴퓨팅 장비의 소비전력이 2,500W에 이르면 공랭만으로 버티기 어렵다. 그렇다고 센서 데이터를 전부 클라우드로 보낼 수도 없다. 통신량과 지연이 바로 안전 문제로 이어진다.

## Moore's Law 둔화와 Domain-Specific HW의 필요성 (L01-22)

![](2026-03-07-15-23-37.png)

MLSys가 독립된 분야가 된 배경에는 Moore's Law와 Dennard Scaling의 둔화가 있다.

트랜지스터 수는 여전히 늘지만 예전 같은 속도는 아니다. 달러당 트랜지스터 수도 정체되고 있다. 트랜지스터가 작아질수록 전력도 함께 줄던 Dennard Scaling은 2005년 전후로 깨졌다. 슬라이드에서 clock speed와 TDP가 더 이상 크게 오르지 않는 이유다.

범용 프로세서의 성능과 효율이 공정만 바꾼다고 자동으로 오르지 않으니 특정 연산에 맞춘 하드웨어가 필요해졌다. Tensor Core는 행렬곱, Google TPU는 DNN training과 inference, Apple ANE는 on-device inference, FuriosaAI NPU는 AI inference에 초점을 둔다.

## L01-23 ~ L01-25: CPU Pipeline의 진화

이 부분은 범용 CPU가 높은 utilization을 얻기 위해 얼마나 많은 제어 로직을 들고 있는지 보여준다. DNN처럼 규칙적인 행렬 연산에서는 그 복잡성이 그대로 이득이 되지 않는다.

### Simple In-Order Pipeline (L01-23)

![](2026-03-07-16-09-22.png)

슬라이드의 파이프라인은 IF, ID, EX, MEM, WB로 이어지는 5단계 구조에서 ID를 Decode와 Reg Read로 나눈 6단계 구조다. 단계별 작업을 줄이면 클럭을 높일 여지가 생긴다. 한 사이클에 명령어를 여러 개 처리하므로 superscalar 구조이기도 하다.

Fetch에서는 PC가 가리키는 주소의 명령어를 I-cache에서 가져온다. 다음 PC는 branch predictor가 고른다.

Decode에서는 opcode, rs, rd, immediate를 해석한다. Reg Read에서는 필요한 operand를 레지스터 파일에서 읽는다.

Execute에서는 ALU가 실제 연산을 수행한다. 그림의 빨간 chevron이 세 개라면 동시에 쓸 수 있는 functional unit이 세 개라는 뜻이다. Integer ALU, FP 또는 Multiply Unit, Branch Unit 같은 조합을 생각하면 된다.

D-cache와 Store Buffer는 load와 store를 처리하고, Reg Write에서 결과를 레지스터 파일에 돌려놓는다.

### Basic Out-of-Order Pipeline (L01-24)

![alt text](image-2.png)

In-order superscalar는 앞 명령어가 cache miss에 걸리면 뒤에 준비된 명령어까지 같이 멈춘다. functional unit을 세 개 깔아도 일이 도착하지 않으면 놀 수밖에 없다.

Out-of-order 구조는 issue queue를 두고 operand가 준비된 명령어부터 먼저 보낸다. 앞의 load가 메모리를 기다리는 동안 독립적인 뒤 명령어를 실행해 빈 슬롯을 채운다.

### SMT: Simultaneous Multi-Threading (L01-25)

여기서 SMT는 GPU의 SIMT와 다른 개념이다.

![alt text](image-3.png)

그림의 빨강, 노랑, 초록은 서로 다른 스레드다. 파이프라인을 스레드 수만큼 복제한 게 아니라 하나의 파이프라인을 여러 스레드가 나눠 쓴다. Intel이 Hyper-Threading이라는 이름으로 유별나게 잘 써먹던 방식이다.

한 스레드가 막혔을 때 다른 스레드의 준비된 명령어로 빈 슬롯을 채워 throughput을 올린다.

| 구조 | 특징 |
|------|------|
| Scalar | 5-stage, width=1, 한 사이클에 명령어 1개 |
| Superscalar (In-Order) | width>1, 한 사이클에 여러 개를 순서대로 처리 |
| OoO + Superscalar | width>1, 준비된 명령어부터 처리 |
| OoO + SMT | 여러 스레드가 하나의 파이프라인을 공유 |

Branch Prediction, Register Renaming, Issue Queue, Retire, Thread Choosing은 모두 범용성을 위해 들어간 제어 로직이다. DNN의 반복적인 MatMul에는 이 복잡성이 과하다. accelerator가 복잡한 CPU pipeline 대신 PE array와 memory hierarchy를 앞세우는 이유다.

## Accelerator 다양성 (L01-26)

![alt text](image-4.png)

DNN accelerator라고 해서 다 같은 모양은 아니다. 타깃 워크로드가 달라지면 PE 배치, 메모리 계층, 데이터 흐름도 달라진다.

1. Eyeriss는 CNN inference를 위한 spatial array이고 각 PE에 local scratchpad를 둔다.

2. Eyeriss V2는 더 유연한 NoC를 넣었다.

3. SCNN은 sparse CNN에서 0인 값을 건너뛰는 데 초점을 맞췄다.

4. ExTensor와 Gamma는 sparse tensor와 sparse matrix 연산을 각각 겨냥한다.

5. spZip은 sparse 데이터의 압축과 해제를 다룬다.

CPU는 파이프라인의 큰 틀이 꽤 표준화되어 있다. accelerator는 dense인지 sparse인지, CNN인지 Transformer인지에 따라 설계부터 갈라진다. 이 차이를 같은 언어로 분석하는 법이 수업의 본론이다.

PE는 Processing Element의 약자다. CPU의 ALU에 가까운 작은 연산 유닛이며, 보통 multiplier와 adder, 작은 register file을 묶어 MAC을 처리한다.

## TeAAL Pyramid of Concerns & FuseMax (L01-28 ~ L01-34)

여기부터 용어가 갑자기 많아진다. 첫 강의에서는 이름을 다 외우기보다 계층을 나눠 본다는 감각만 챙기면 된다.

### TeAAL Pyramid of Concerns (L01-28)

![alt text](image-6.png)

TeAAL은 accelerator 설계를 네 계층으로 나눈다. 위로 갈수록 결정이 세밀해지고, architecture는 각 계층이 가능한 범위를 제한한다.

| 계층 | 의미 | 수업 연결 |
|------|------|----------|
| Compute | 어떤 연산을 하는가. MatMul, Attention 등 | Lab 1: Einsum |
| Mapping | 연산을 하드웨어에 어떻게 배치하는가. tiling, dataflow 등 | Lab 2, 3 |
| Format | 데이터를 어떤 형태로 저장하는가. dense, sparse 등 | Lab 4: Sparsity |
| Binding | 연산과 데이터를 실제 PE와 buffer에 어떻게 할당하는가 | Lab 2, 3 |

수업 뒤쪽에서 나오는 설계 문제 대부분을 이 네 칸에 다시 놓아볼 수 있다.

### FuseMax: 피라미드 실제 적용 사례 (L01-29)

![alt text](image-7.png)

FuseMax는 Transformer attention을 가속하는 설계다. compute, mapping, binding을 차례로 손보면서 낮았던 PE utilization을 약 90%까지 올린다.

| 단계 | 계층 | 바꾼 것 | 결과 |
|------|------|----------|------|
| Cascade | Compute | MatMul과 Softmax를 fuse해 중간 data movement를 줄임 | utilization 소폭 상승 |
| Architecture 변경 | Architecture와 Mapping | 새로운 mapping이 가능하도록 구조를 바꿈 | utilization 추가 상승 |
| Binding 개선 | Binding | 작업을 PE에 더 고르게 할당 | utilization 약 90% |

Unfused는 Q×K, Softmax, ×V를 따로 실행해 단계마다 중간 결과를 메모리에 쓴다. FLAT은 기존의 부분적인 fusion 방식이다. Cascade는 중간 결과를 메모리에 내보내지 않도록 더 강하게 묶는다.

어느 한 단계만 잘 만든다고 성능이 나오지는 않는다. 연산을 묶는 방식, 하드웨어 구조, 실제 자원 배치가 같이 맞아야 한다.

### PE Utilization 변화 (L01-30 ~ L01-34)

![alt text](image-8.png)

L01-30의 baseline에서는 BERT, TrXL, T5, XLM 모두 시퀀스 길이 1K부터 1M까지 PE utilization이 0.25 아래에 머문다. PE를 수백 개 만들어 놓고도 대부분을 놀리는 상태다.

그래프의 x축은 sequence length, y축은 PE utilization이다. 빨간색 Unfused와 주황색 FLAT 모두 낮게 깔린다. L01-31부터 L01-33까지 Cascade, Architecture, Binding을 차례로 적용하면 utilization이 약 90%까지 올라간다.

## DNN Accelerator의 기본 구조 (L01-43)

![alt text](image-9.png)

전형적인 DNN accelerator는 복잡한 CPU pipeline 대신 PE array와 memory hierarchy를 둔다.

데이터는 DRAM에서 Global Buffer로 들어오고 PE array로 전달된다. 각 PE 안에는 1kB보다 작은 Reg File과 ALU, 간단한 Control이 있다. PE 수는 수백 개에서 수천 개 수준이고 Global Buffer는 대략 100kB에서 500kB 규모다.

각 PE는 MAC처럼 단순하고 반복적인 연산에 집중한다. Branch Prediction이나 Register Renaming에 면적을 쓰지 않으니 같은 칩 면적에 연산 유닛을 더 많이 넣을 수 있다.

슬라이드가 제시한 65nm 기준 normalized energy cost는 다음과 같다.

| 데이터를 가져오는 위치 | 에너지 비용 |
|---|---|
| ALU 연산 자체 | 1× |
| Reg File에서 ALU | 1× |
| PE 사이 NoC에서 ALU | 2× |
| Global Buffer에서 ALU | 6× |
| DRAM에서 ALU | 200× |

DRAM 접근 한 번이 실제 연산보다 200배 비싸다. FlashAttention이 HBM 대신 SRAM에서 타일링하고, Cerebras가 18GB SRAM을 넣은 이유가 같은 표에서 나온다.

"Farther and larger memories consume more power."

이 문장을 기억해두면 수업 뒤쪽의 dataflow와 tiling 얘기가 훨씬 쉽게 읽힌다.

## Accelerator 설계 결정 변수 (L01-44)

![alt text](image-10.png)

설계자가 직접 골라야 하는 항목은 수업의 Lab과 이어진다.

| 설계 결정 | 구체적 내용 | Lab |
|----------|----------|-----|
| PE array | PE 개수와 PE 사이 연결 방식 | Lab 2, 3 |
| Memory hierarchy | 계층 수, 용량, 데이터 배치 | Lab 2, 3 |
| Scheduling | mapping, dataflow, tiling, parallelism, fusion | Lab 2, 3 |
| Sparsity 처리 | gating, skipping, 압축 포맷 | Lab 4 |
| 구현 기술 | RRAM, optical, superconductors 등 | Lab 5 |

TeAAL의 Compute, Mapping, Format, Binding이 여기서 실제 설계 변수로 내려온다.

## Roofline 기반 비효율 분석 (L01-45)

![alt text](image-11.png)

Roofline Model은 성능이 연산량에 막혔는지 메모리 대역폭에 막혔는지 보는 틀이다. x축은 compute intensity, y축은 성능이다. x축 값이 클수록 데이터 하나를 가져와 더 많은 연산을 한다. 경사진 선은 memory bandwidth가 만드는 한계고, 오른쪽의 평평한 선은 compute peak가 만드는 한계다.

왼쪽 경사면에 붙으면 memory-bound이고 오른쪽 평탄면에 붙으면 compute-bound다.

슬라이드의 Step 1부터 Step 7은 이론적인 peak가 현실의 제약을 만날 때 얼마나 깎이는지 보여준다.

| Step | 제약 요인 | Lab |
|------|---------|-----|
| Step 1 | 워크로드 자체의 최대 병렬성 | Lab 1 |
| Step 2 | dataflow가 허용하는 최대 병렬성 | Lab 2, 3 |
| PE 개수 | 하드웨어의 theoretical peak | 없음 |
| Step 3 | 유한한 PE array 크기 | Lab 2, 3 |
| Step 4 | PE array 차원 제약 | Lab 2, 3 |
| Step 5 | 유한한 storage 용량 | Lab 2, 3 |
| Step 6 | 평균 bandwidth 부족 | Lab 2, 3 |
| Step 7 | 순간 bandwidth 부족 | 없음 |

제약을 하나씩 반영할수록 roofline이 아래로 조여진다. 처음부터 실제 성능을 하나의 숫자로 찍는 대신 어디에서 얼마가 깎였는지 분리해서 볼 수 있다.

### 로컬 FlashAttention 커널을 Nsight Compute로 재봤다

수업 슬라이드만 옮겨 적으면 여기서 글이 끝난다. 실제 GPU에서도 같은 식으로 읽히는지 확인하려고 로컬의 FlashAttention forward 커널을 다시 프로파일링했다.

장비는 RTX 4060 Ti 8GB, compute capability 8.9다. CUDA 13.0으로 `sm_89` 타깃을 빌드했고 Nsight Compute 2025.3.1의 detailed set을 사용했다. 입력은 `batch_heads=4`, `N=1024`, `D=128`이다. block은 128 threads, grid는 64 blocks다.

| 측정 항목 | 값 |
|----------|----|
| 프로파일러 없이 잰 평균 kernel time | 17.25 ms |
| Nsight 계측 중 kernel duration | 20.56 ms |
| Compute (SM) Throughput | 8.23% |
| DRAM Throughput | 1.10% |
| L1/TEX Cache Throughput | 91.33% |
| Achieved Occupancy | 8.33% |
| Dynamic Shared Memory | 98.82 KB/block |
| Registers | 40/thread |

DRAM throughput이 1.10%라고 해서 이 커널이 잘 최적화됐다는 뜻은 아니다. SM throughput도 8.23%에 그쳤고 FP32 peak의 약 1%만 썼다. 대신 L1/TEX 쪽은 91.33%까지 찼다.

occupancy가 낮은 이유도 바로 보였다. block 하나가 dynamic shared memory를 98.82KB 써서 SM당 block limit이 1로 걸렸다. 그 결과 SM에 active warp가 4개만 남았고 achieved occupancy가 8.33%에 멈췄다.

Nsight는 shared memory access에서 excessive wavefront가 전체의 95%라고 경고했다. 지금 커널의 병목은 HBM 대역폭이 아니라 shared memory 용량과 access pattern이다. DRAM 숫자 하나만 보고 FlashAttention답게 IO를 줄였다고 끝냈으면 완전히 반만 본 셈이다.

여기서 수업의 roofline과 실제 프로파일링이 연결된다. 수업은 PE array와 buffer 관점에서 제약을 좁혀가고, Nsight Compute는 SM, warp, shared memory, cache 관점에서 같은 작업을 한다. 이름은 달라도 peak에서 실제 성능까지 무엇이 깎아먹는지 하나씩 찾는다는 점은 같다.
