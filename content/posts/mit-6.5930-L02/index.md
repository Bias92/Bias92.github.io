---
title: "6.5930 L02 - From Einsum to DNN Workloads"
date: 2026-07-18T00:00:00+09:00
tags: ["MLSys", "DNN-Accelerator", "Einsum", "CNN", "MIT-6.5930"]
categories: ["MIT 6.5930"]
series: ["MIT 6.5930"]
math: true
summary: "Einsum으로 workload를 쓰고 iteration space, memory traffic, compute intensity를 계산한다. 같은 수식도 mapping에 따라 왜 3배 가까이 달라지는지 본 뒤 CNN과 FC를 tensor 연산으로 내린다."
draft: false
---

> 기준 자료: MIT 6.5930/1 Spring 2026, [L02 - Overview of Deep Neural Network Components](https://csg.csail.mit.edu/6.5930/lectures/L02-Overview_on_DNN_components.pdf)

L01에서 가장 자주 나온 말은 data movement였다. DRAM 접근이 비싸다, 가까운 메모리에서 재사용해야 한다, PE를 많이 넣어도 utilization이 안 나오면 소용없다. 전부 맞는 말인데 아직은 구호에 가깝다. 얼마나 비싼지, 어떤 구현이 더 나은지는 계산하지 않았다.

L02는 작은 matrix-vector multiply 하나를 붙잡고 그 계산을 한다. 수식은 같은데 loop order를 정하는 순간 memory traffic이 달라지고, best-case compute intensity가 0.99인데 실제 구현은 0.33까지 떨어진다. 이 차이가 앞으로 나올 mapping과 dataflow의 출발점이다.

강의 후반의 CNN, fully connected layer 설명도 같은 맥락으로 읽으면 덜 교과서적이다. CNN 구조를 복습하려는 게 아니라, 모델의 layer를 accelerator가 다룰 수 있는 **tensor 식과 iteration space**로 번역하는 과정이다.

## Workload와 hardware 사이에 있는 것

Accelerator 설계 과정을 아주 거칠게 적으면 다음과 같다.

1. hardware architecture를 정한다. PE, register, buffer, DRAM과 연결 구조를 고른다.
2. workload를 tensor 식으로 쓴다.
3. 그 식을 어떤 순서로 계산하고, 데이터는 어떤 format으로 저장하며, 실제 자원 어디에 붙일지 정한다.
4. compute 수와 traffic, throughput을 계산한다.
5. 다른 구현과 비교하면서 병목을 옮긴다.

두 번째와 세 번째 사이가 생각보다 넓다. 같은 matrix multiplication도 loop 순서, tile 크기, spatial parallelism에 따라 전혀 다른 하드웨어 동작이 된다.

![TeAAL separation of concerns](images/L02-08-separation-of-concerns.png)
*Einsum은 계산의 내용만 쓴다. 아래로 내려갈수록 구현 결정이 붙는다.*

TeAAL은 이걸 Cascade of Einsums, Mapping, Format, Binding으로 나눈다. 위쪽은 짧고 추상적이다. 아래로 갈수록 실제 구현에 가까워진다.

- **Cascade of Einsums**: 어떤 tensor 계산들을 어떤 의존관계로 실행하는가
- **Mapping**: iteration space를 어떤 순서로 순회하고, tile과 병렬 처리를 어떻게 잡는가
- **Format**: dense인지 sparse인지, sparse라면 어떤 압축 표현을 쓰는가
- **Binding**: 계산과 데이터를 구체적인 PE, register, buffer, network에 어떻게 할당하는가

Einsum 하나만 보고 accelerator 성능을 말할 수 없는 이유가 이 피라미드에 다 들어 있다.

## Einsum은 수식이면서 iteration space다

Tensor는 다차원 배열이다. scalar는 0차원, vector는 1차원, matrix는 2차원이다. 이 수업은 각 차원을 `rank`라고 부른다. 선형대수에서 말하는 matrix rank와는 다른 용법이라 처음엔 조금 거슬린다. 여기서는 그냥 axis 또는 dimension으로 읽으면 된다.

Matrix-vector multiply를 예로 잡자.

$$
Z_m = A_{k,m} \times B_k
$$

오른쪽에는 \(k\)가 있는데 왼쪽에는 없다. 이 rank는 reduction 대상이다. 합 기호를 생략했을 뿐 실제 뜻은 아래와 같다.

$$
Z_m = \sum_k A_{k,m} B_k
$$

표기가 짧다는 것보다 더 중요한 성질이 있다. 이 식은 **계산 순서를 정하지 않는다.**

![Operational definition of an Einsum and its iteration space](images/L02-18-iteration-space.png)
*식이 만드는 공간은 \(K \times M\). 어느 방향부터 훑을지는 아직 비어 있다.*

TeAAL의 operational definition으로 Einsum은 세 가지를 정한다.

1. input과 output tensor, 그리고 각 tensor가 쓰는 rank
2. 수행해야 할 모든 계산점으로 이뤄진 iteration space
3. 각 점에서 실행할 연산

위 식의 iteration space는 \(K \times M\)이다. 점 \((k,m)\) 하나를 방문하면 \(A_{k,m}\)과 \(B_k\)를 읽어 곱하고 \(Z_m\)에 더한다. 모든 점을 한 번씩 처리하면 계산이 끝난다. \(k\)부터 돌든 \(m\)부터 돌든, tile로 잘라 병렬 PE에 뿌리든 수학적으로는 같은 답이다.

그 자유도가 mapping의 설계 공간이다.

## 계산량은 식에서 바로 나온다

\(K \times M\)개의 iteration point마다 multiply를 한 번 하므로 multiplication 수는 \(KM\)이다. \(Z_m\) 하나를 만들 때 \(K\)개 값을 더하니 addition은 정확히 세면 \((K-1)M\)번이다.

여기까지는 구현과 무관하다. 순서를 바꿔도 필요한 유효 연산 수는 같다. 달라지는 건 데이터를 몇 번 움직이느냐다.

### 이론적으로 가장 적게 옮긴다면

![Best-case compute intensity and the memory hierarchy](images/L02-23-compute-intensity.png)

강의는 보통의 FLOPs/byte 대신 `multiplications/value`로 compute intensity를 정의한다.

$$
\text{CI} = \frac{\text{number of multiplications}}
{\text{number of values moved}}
$$

MAC을 연산 한 번으로 셀지 두 번으로 셀지, 값 하나가 FP32인지 INT8인지 같은 문제를 잠시 치워두려는 정의다. 실제 Roofline 분석에서는 FLOPs/byte로 다시 단위를 맞추면 된다.

각 값을 필요한 최소 횟수만 DRAM에서 가져온다고 해보자.

- \(A\): 원소가 \(KM\)개이므로 \(KM\) loads
- \(B\): 원소가 \(K\)개이므로 \(K\) loads
- \(Z\): 결과가 \(M\)개이므로 \(M\) stores

따라서 best-case traffic과 CI는

$$
T_{\text{best}} = KM + K + M
$$

$$
\text{CI}_{\text{best}} = \frac{KM}{KM + K + M}
$$

이다. \(A\)는 계산마다 다른 원소를 쓰므로 재사용이 없다. 대신 \(B_k\) 하나는 모든 \(m\)에서 재사용하고, \(Z_m\)은 \(k\) 방향 reduction이 끝날 때까지 가까운 저장소에 붙잡아둬야 이 traffic이 나온다.

그런데 `붙잡아둔다`는 말에 이미 hardware 조건이 숨어 있다. \(B\)와 \(Z\)의 live set을 담을 register나 SRAM이 있어야 하고, mapping도 그 reuse를 깨지 않는 순서여야 한다.

## Loop를 쓰는 순간 reuse가 결정된다

Einsum에 순서를 하나 부여해보자.

```python
for k in range(K):
    b_reg = B[k]
    for m in range(M):
        a_reg = A[k, m]
        z_reg = Z[m]
        z_reg += a_reg * b_reg
        Z[m] = z_reg
```

`k`를 바깥에 두면 \(B_k\)는 register에 한 번 올린 뒤 \(M\)번 쓸 수 있다. \(B\) stationary mapping이다. 대신 \(Z_m\)은 \(k\)가 바뀔 때마다 다시 읽고 써야 한다. 한 개짜리 `z_reg`가 모든 \(M\)개의 partial sum을 다음 \(k\)까지 보관할 수는 없기 때문이다.

이 구현의 traffic은 다음처럼 나온다.

$$
\begin{aligned}
A\text{ loads} &= KM \\
B\text{ loads} &= K \\
Z\text{ loads} &= (K-1)M \\
Z\text{ stores} &= KM
\end{aligned}
$$

첫 번째 \(k\)에서는 \(Z\)를 0으로 시작한다고 보면 load가 필요 없어서 \((K-1)M\)이다. 전부 더하면

$$
T_{\text{achieved}} = 3KM - M + K
$$

이고,

$$
\text{CI}_{\text{achieved}} =
\frac{KM}{3KM - M + K}
$$

가 된다.

![Best-case and achieved compute intensity](images/L02-41-best-vs-achieved-ci.png)
*\(K=250, M=100\)에서 0.99 대 0.33. 연산식은 한 글자도 안 바뀌었다.*

\(K=250\), \(M=100\)을 넣으면 best case는 약 0.99 multiplications/value, 이 mapping은 약 0.33이다. 같은 \(KM=25{,}000\)번의 multiplication을 하면서 데이터를 거의 세 배 옮긴다.

`m`을 바깥 loop로 바꾸면 반대 trade-off가 생긴다. \(Z_m\) partial sum은 register에 오래 둘 수 있지만, 이번에는 \(B_k\)를 \(m\)마다 다시 읽는다. 둘 다 충분히 재사용하려면 더 큰 local storage, tiling, 여러 PE 사이의 multicast 같은 구조가 필요하다. loop interchange 한 줄에서 시작한 문제가 그대로 accelerator architecture 문제가 된다.

이 예제가 L02의 중심이다. **best-case CI는 workload의 잠재력이고, achieved CI는 architecture와 mapping이 실제로 건져낸 reuse다.**

## Roofline에서 오른쪽으로 가는 법

![Roofline model](images/L02-42-roofline.png)

Roofline의 식은 간단하다.

$$
\text{Throughput} \le
\min(P_{\text{peak}},\; BW \times \text{CI})
$$

\(P_{\text{peak}}\)는 연산기가 낼 수 있는 최대 throughput, \(BW\)는 memory bandwidth다. CI가 낮으면 \(BW \times \text{CI}\)가 먼저 한계가 되어 그래프의 사선 구간에 놓인다. memory-bound다. CI가 충분히 높아지면 평평한 compute roof에 닿고, 그때부터 compute-bound가 된다.

이 그림이 유용한 이유는 최적화 방향을 바로 거르기 때문이다.

Memory-bound인 점에서 MAC lane을 두 배로 늘리면 평평한 지붕만 올라간다. 현재 점은 그대로다. 반대로 reuse를 살려 CI를 높이면 점이 오른쪽으로 움직이고, bandwidth를 늘리면 사선의 기울기가 커진다. “병렬성을 더 주면 빨라진다”는 말은 compute-bound 근처에서만 온전히 맞는다.

앞의 matrix-vector multiply에서 0.33을 0.99로 끌어올리는 일은 단순히 traffic을 줄이는 데서 끝나지 않는다. Roofline 위에서는 같은 bandwidth로 낼 수 있는 throughput을 세 배 가까이 키우는 이동이다.

## CNN을 tensor 식으로 내리기

강의 후반은 CNN 구성요소를 빠르게 훑는다. convolution, activation, optional한 pooling과 normalization, 마지막의 fully connected layer. 고전적인 vision CNN에서는 convolution이 전체 계산의 90% 이상을 차지하는 경우가 흔해서 accelerator 분석도 CONV부터 시작한다.

![A conventional deep CNN](images/L02-47-cnn-overview.png)

초기 layer는 edge나 corner 같은 local feature를 잡고, 깊은 layer로 갈수록 더 넓은 receptive field를 보며 고수준 feature를 만든다. ML 관점에서는 익숙한 얘기다. Hardware 쪽에서 중요한 건 각 layer가 정확히 몇 번 곱하고, 어떤 activation과 weight를 반복해 쓰는가다.

### 먼저 2D 한 장

![2D convolution example with a 3 by 3 filter](images/L02-57-conv-example.png)

\(H \times W\) input에 \(R \times S\) filter를 stride \(U\)로 움직이면, padding이 없을 때 output 크기는

$$
P = \left\lfloor \frac{H-R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W-S}{U} \right\rfloor + 1
$$

이다.

슬라이드의 5×5 input과 3×3 filter에서 stride 1이면 output은 3×3이다. output point 하나마다 9번 곱하므로 총 81 multiplications다. Filter에 0이 있어도 sparse skipping을 지원하지 않는 구현은 그대로 9번 계산한다.

![Stride as downsampling](images/L02-71-stride.png)

Stride 2면 output이 2×2로 줄고 36번, stride 3이면 1×1과 9번이 된다. Stride를 키우는 건 stride 1 결과를 일정 간격으로 downsample하는 것과 같다. Padding \(D\)를 양쪽에 주면 식은

$$
P = \left\lfloor \frac{H+2D-R}{U} \right\rfloor + 1
$$

로 바뀐다. 3×3 filter, stride 1에서 \(D=1\)을 주는 흔한 `same` convolution은 input과 output의 공간 크기를 유지한다.

참고로 딥러닝 프레임워크가 `convolution`이라고 부르는 연산은 보통 filter를 뒤집지 않는 cross-correlation이다. 슬라이드의 index 식과 naïve loop도 그 형태다. 학습이 filter 값을 알아서 맞추므로 모델 사용에서는 문제가 없지만, 신호처리의 convolution 정의와 코드를 비교할 때는 구분해야 한다.

### Batch와 channel까지 펼치면 7중 loop가 된다

![Input channels, filters, and output channels](images/L02-77-channels.png)

실제 CONV에는 input channel과 output channel, batch가 붙는다.

| 기호 | 뜻 |
| --- | --- |
| \(N\) | batch size |
| \(C\) | input channels |
| \(H,W\) | input height, width |
| \(R,S\) | filter height, width |
| \(M\) | output channels, 즉 filter 수 |
| \(P,Q\) | output height, width |
| \(U\) | stride |

Weight tensor는 \(F[M,C,R,S]\), input activation은 \(I[N,C,H,W]\), output은 \(O[N,M,P,Q]\)다. 식 하나로 적으면

$$
O_{n,m,p,q}
= B_m +
I_{n,c,Up+r,Uq+s}
\times F_{m,c,r,s}
$$

가 된다. \(c,r,s\)가 오른쪽에만 있으므로 셋 모두 reduction rank다.

![Convolution written as an Einsum](images/L02-82-conv-einsum.png)
*이 한 줄은 계산 내용은 정하지만 \(n,m,p,q,c,r,s\)를 어떤 순서로 돌지는 말하지 않는다.*

Iteration space 크기와 multiplication 수는

$$
N \times M \times P \times Q \times C \times R \times S
$$

다. 이 식을 그대로 naïve loop로 옮기면 7중 loop가 나온다.

![Naive seven-loop implementation of convolution](images/L02-83-conv-loop-nest.png)

```python
for n in range(N):
    for m in range(M):
        for q in range(Q):
            for p in range(P):
                O[n, m, p, q] = B[m]
                for c in range(C):
                    for r in range(R):
                        for s in range(S):
                            O[n, m, p, q] += (
                                I[n, c, U*p+r, U*q+s] * F[m, c, r, s]
                            )
```

Einsum과 loop nest의 차이는 수학 표기 취향이 아니다. 위 코드는 `s → r → c → p → q → m → n`이라는 처리 순서를 이미 박아 넣었다. 어떤 값을 register에 오래 둘지, 연속된 memory access가 어느 rank인지, 병렬 PE에 무엇을 나눠줄지가 이 순서에 묶인다.

Einsum은 그 결정을 미룬다. 이후 mapping 단계에서 loop를 reorder하고, tile로 쪼개고, 여러 rank를 한 PE array에 spatial하게 펼친다. 하나의 CONV 식에서 weight-stationary, output-stationary, row-stationary 같은 서로 다른 dataflow가 나오는 자리다.

## Fully connected는 특별한 연산이 아니다

Fully connected layer를 CONV 관점에서 보면 filter의 공간 크기가 input 전체와 같은 경우다.

$$
R=H,\qquad S=W,\qquad P=Q=1
$$

Batch가 하나라면

$$
O_m = I_{c,h,w} \times F_{m,c,h,w}
$$

이고 \(c,h,w\)를 하나의 \(k=CHW\) rank로 flatten하면

$$
O_m = I_k \times F_{m,k}
$$

가 된다. Matrix-vector multiply, 즉 GEMV다.

Batch \(N\)을 넣으면 input이 \(N \times K\) matrix가 되면서

$$
O_{n,m} = I_{n,k} \times F_{m,k}
$$

형태의 matrix-matrix multiply, GEMM으로 바뀐다.

![A batched fully connected layer becomes matrix-matrix multiplication](images/L02-101-fc-matmul.png)

이건 단순한 표기 정리가 아니다. Batch가 커지면 같은 weight matrix를 여러 input이 공유해 arithmetic intensity를 높일 수 있다. GEMV는 weight를 읽어 한 vector에 쓰고 끝나서 memory-bound가 되기 쉽지만, GEMM은 불러온 weight tile을 batch 방향으로 재사용한다. GPU와 DNN accelerator가 GEMM에 그렇게 많은 하드웨어와 소프트웨어를 투자하는 이유가 여기서 보인다.

CONV도 `im2col`로 input window를 펼치면 GEMM으로 바꿀 수 있다. 다만 실제로 펼친 matrix를 메모리에 만들면 중복 데이터가 커진다. 그래서 고성능 library는 대개 layout과 tile 안에서 암묵적으로 변환하는 implicit GEMM이나 convolution 전용 kernel을 쓴다. 수학적으로 GEMM과 같다는 말과, 메모리에서도 공짜로 GEMM이 된다는 말은 다르다.

## L02를 덮고 남는 것

처음엔 CNN 기초가 절반이라 쉬어가는 강의처럼 보였다. 다시 보면 목적은 꽤 명확하다.

- DNN layer를 Einsum으로 적으면 계산의 내용과 iteration space가 드러난다.
- 식만으로는 처리 순서가 정해지지 않는다. 그 빈칸이 mapping이다.
- 계산량이 같아도 mapping과 storage에 따라 traffic이 달라진다.
- Compute intensity는 workload가 가진 reuse와 구현이 실제로 살린 reuse를 구분한다.
- Roofline은 현재 구현이 compute와 memory 중 어디에 막혔는지 보여준다.

여기까지 오면 accelerator diagram을 볼 때 PE 개수부터 세는 습관이 조금 바뀐다. 먼저 식을 찾고, reduction rank가 무엇인지, 어떤 tensor가 어느 방향으로 재사용되는지, 그 값을 붙잡아둘 메모리가 있는지 보게 된다. 다음 강의부터 나오는 dataflow와 partitioning도 결국 이 질문을 더 큰 iteration space에 적용하는 일이다.

## 참고

- [MIT 6.5930/1 Spring 2026 L02 slides](https://csg.csail.mit.edu/6.5930/lectures/L02-Overview_on_DNN_components.pdf)
- [TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators, MICRO 2023](https://people.csail.mit.edu/emer/media/papers/2023.10.micro.teaal.pdf)
- [TeAAL and HiFiber tutorial](https://teaal.csail.mit.edu/tutorials/2024.micro-teaal/index.html)
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)
- [Efficient Processing of Deep Neural Networks, Sze et al.](https://doi.org/10.1007/978-3-031-01766-7)
