---
title: "03 CUDA Shared Memory: Tiling, Bank Conflicts, and Reduction"
date: 2026-07-14
draft: false
tags: ["CUDA", "GPU Programming", "Shared Memory", "Warp Divergence", "Parallel Programming", "Reduction"]
categories: ["CUDA"]
series: ["CUDA C"]
math: true
summary: "global memory를 읽어 shared memory에 재사용할 때 무엇을 챙겨야 할까요? Coalescing과 tiling부터 bank conflict를 푸는 padding·swizzle, occupancy, warp divergence와 predication을 살펴보고, reduction을 네 단계로 개선하는 코드까지 따라가봐요."
---

안녕하세요~ 오늘은 같은 block 안에서 memory를 같이 쓰는 이야기를 해보려고 해요ㅎㅎ

먼저 자리를 좀 잡고 갈게요. GPU에서 실행되는 함수가 kernel이고, 그 kernel을 실행하는 작업 단위가 thread예요. Thread들은 block으로 묶여서 배치되고, block 하나는 GPU의 실행 장치인 SM(Streaming Multiprocessor) 하나에서 끝까지 실행돼요.

이 thread들이 쓰는 memory를 보면, 모든 thread가 접근할 수 있는 큰 global memory가 있고 SM 안에서 같은 block끼리 함께 쓰는 작은 shared memory가 있어요. Shared memory 쪽이 훨씬 빠르긴 한데요. 어떤 데이터를 언제 올리고 비울지는 kernel 코드에서 직접 정해줘야 한답니다. Hardware가 알아서 채워주는 cache처럼 두고 쓰면 되는 건 아니에요.

그 수고를 들이는 이유는 재사용이에요. Global memory에서 한 번 읽은 데이터를 block 안에서 여러 번 쓸 수 있거든요. 다만 같이 쓰기 시작하면 순서도 맞춰야겠죠? 쓰기와 읽기를 맞추는 barrier, 같은 저장 장치에 접근이 몰릴 때의 bank conflict, block이 커져 SM에 올릴 thread가 줄어드는 occupancy 문제까지 따라와요.

빠른 memory 하나 썼을 뿐인데 챙길 일이 제법 생기네요..ㅎㅎ

![기대하다가 살짝 땀을 흘리는 문](/images/naver-moon/moon-115.png)

행렬곱, transpose, reduction을 보면서 이 문제들이 코드 어디에서 나오는지 짚어볼게요. 우선 shared memory에 올릴 데이터도 global memory에서 잘 읽어와야 하니, 그쪽부터 시작해요.

## Global Memory와 Coalescing

GPU는 thread를 32개씩 묶어서 같은 명령을 실행해요. 이 묶음이 warp이고, 그 안의 thread 자리가 lane이랍니다. Warp가 global memory를 읽을 때면 32개 lane이 저마다 읽을 주소를 내놓는데요.

Memory 쪽에서는 필요한 byte만 낱개로 보내주지 않아요. Sector라는 32바이트 덩어리로 보내요. 경계도 주소 0부터 32바이트 간격으로 정해져 있어서, 그중 1바이트만 필요해도 sector 하나를 통째로 받아야 해요. 조금만 쓰고 나머지는 남기면 아깝겠죠ㅎㅎ

여러 lane의 접근을 가능한 적은 sector 전송으로 모으는 걸 coalescing이라고 해요. 그래서 global memory 접근 비용을 볼 때는 lane이 몇 개인지보다 서로 다른 sector를 몇 개 건드렸는지 보셔야 해요.

`float` 하나는 4바이트예요. 32개 lane이 연속된 `float` 32개를 읽으면 필요한 범위가 128바이트이고, 시작 주소가 sector 경계인 32의 배수에 맞으면 sector 4개로 끝나요.

그런데 시작점을 `float` 하나만큼만 옮겨볼까요? 같은 128바이트가 경계 하나를 더 걸쳐서 sector가 5개 필요해져요. Lane 사이 간격을 32바이트 이상 벌리면 각자 다른 sector를 건드리니 32개까지 늘어나고요. 쓸 데이터는 세 경우 모두 128바이트인데, 실제 전송량은 128, 160, 1024바이트로 달라진답니다.

![Coalescing](images/coalescing.svg?v=2)

`cudaMalloc`이 돌려주는 pointer는 충분히 정렬돼 있어요. 다만 그 pointer에 offset을 더해서 부분 영역을 만들면 시작 주소가 어긋날 수 있으니 그때는 다시 봐주셔야 해요.

참, sector 4개라는 숫자에도 조건이 붙어요. Lane 32개가 `float`를 하나씩 읽는 경우예요. Lane 수나 데이터 폭이 달라지면 최소 sector 수도 함께 달라진답니다.

## Shared Memory와 Tiling

이제 행렬곱 $C = A \times B$로 넘어가볼게요. $C$의 원소 하나를 계산하려면 $A$의 한 행과 $B$의 한 열을 읽어야 하죠. 가장 단순하게는 thread 하나에 $C$ 원소 하나를 맡기고, 필요한 값을 그때그때 global memory에서 가져오면 돼요.

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

코드의 `threadIdx`는 block 안에서의 thread 번호, `blockIdx`는 grid 안에서의 block 번호예요. `blockDim`은 block 한 변의 thread 수고요. 안쪽 반복문이 한 바퀴 돌 때마다 thread는 8바이트를 읽어서 곱셈과 덧셈, 두 번의 연산을 해요.

그런데 $A$의 같은 행은 $C$의 같은 행에 있는 $N$개 원소를 계산할 때 계속 필요해요. 이대로면 이미 읽었던 값을 global memory에서 또 가져오게 되는 거예요. 자주 쓸 걸 알면서 매번 가지러 가네요..ㅎㅎ

이 재사용을 shared memory에서 직접 관리하는 방법이 tiling이에요. 행렬을 $T \times T$ 크기로 나눈 정사각 조각을 tile이라고 부르는데요. Block의 thread들이 힘을 나눠서 $A$와 $B$의 tile 하나씩을 shared memory에 복사해둬요.

그 tile로 만들 수 있는 부분곱을 모두 누적한 다음에 다음 tile을 가져오는 식이에요. 덕분에 tile 원소 하나를 global memory에서 한 번 읽고, shared memory에서는 $T$번 재사용할 수 있답니다.

![만족스럽게 웃는 문](/images/naver-moon/moon-10.png)

![Tiling](images/tiling.svg?v=2)

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

`__shared__`는 이 변수를 shared memory에 두겠다는 선언이에요. 중간에 두 번 나오는 `__syncthreads()`도 봐주세요. Block의 모든 thread가 이 줄에 도착할 때까지 먼저 온 thread를 기다리게 하는 barrier예요.

첫 번째는 아직 다른 thread가 채우지도 않은 tile을 읽지 않도록, 두 번째는 다른 thread가 읽는 중인 tile에 다음 데이터를 덮어쓰지 않도록 기다려요. 먼저 끝났다고 혼자 다음 일을 시작하면 곤란하겠죠? ^^;;

Barrier에 참여하는 조건도 맞춰야 해요. Block의 일부 thread가 먼저 return하고 나머지만 `__syncthreads()`에 도착하면 실행이 정의되지 않아요. 여기서는 $N$이 $T$의 배수라 모두 같은 흐름을 지나는데요. 임의 크기를 받으려면 경계 밖 thread도 return시키지 말고 반복문과 barrier에 참여시켜주세요. 읽는 값은 0으로 채우고, 마지막 `C` 저장에만 범위 검사를 걸면 돼요.

얼마나 재사용하는지는 arithmetic intensity로 계산해볼 수 있어요. Global memory에서 읽은 바이트당 연산을 몇 번 했는지 나타내는 값이에요. Block 하나가 tile 쌍을 $N/T$번 가져오면 읽는 양은 $8NT$바이트예요. $T^2$개 thread가 각각 $2N$번 연산하니, 둘을 나눠볼게요.

$$
I_{\text{tiled}} = \frac{2NT^2}{8NT} = \frac{T}{4}\ \text{FLOP/B}
$$

$T = 32$를 넣으면 8 FLOP/B가 나와요. Tile 없이 매번 읽던 0.25 FLOP/B와 비교하면 32배예요. 가져온 값 하나를 여러 번 쓰는 효과가 숫자로도 보이네요ㅎㅎ Tile을 복사할 때는 행 방향의 연속된 주소를 읽으니, 앞에서 본 coalescing 조건도 만족해요.

## Bank Conflict

Shared memory에 가져왔으니 이제 편하게 읽으면 될까요~ 그런데 여기에도 접근이 몰리는 자리가 있어요.

Shared memory는 bank라는 독립된 저장 장치 32개로 나뉘어요. 연속된 4바이트 word를 bank 0, 1, ..., 31, 0, 1, ... 순서로 배정하는 구조예요. Warp의 32개 lane이 서로 다른 bank에 접근하면 한 번에 처리할 수 있지만, 같은 bank의 서로 다른 주소에 몰리면 차례로 처리해야 해요. 이렇게 $n$개 접근이 겹쳐 직렬화되는 게 $n$-way bank conflict랍니다.

여러 lane이 아예 같은 주소를 읽는 경우는 따로 봐야 해요. 그때는 값 하나를 모두에게 나눠주는 broadcast라서 conflict가 아니에요.

2차원 tile을 열 방향으로 읽을 때 이 conflict가 잘 드러나요. 행과 열을 뒤바꾸는 transpose를 보시면 되는데요. Global memory에서 행 방향으로 읽은 tile을 shared memory에 놓고, 열 방향으로 꺼내 쓰면 global memory의 읽기와 쓰기를 모두 coalescing할 수 있어요.

```cpp
__shared__ float tile[32][32];

tile[threadIdx.y][threadIdx.x] = in[...];   // 행 방향 쓰기: bank 분산
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y];  // 열 방향 읽기: 32-way conflict
```

문제는 그 열 방향 읽기예요. Lane들이 가져갈 `tile[0][c], tile[1][c], ...`는 32 word씩 떨어져 있어서 전부 bank `c`로 모여요. Bank는 32개나 있는데 한 군데에서 기다리게 생겼네요ㅠㅠ

![못마땅한 눈으로 땀을 흘리는 문](/images/naver-moon/moon-8.png)

주소 배치를 조금 바꿔서 이 겹침을 풀어볼게요. Padding과 swizzle, 두 방법이 있어요.

먼저 padding이에요. 행 길이를 32에서 33으로 늘려줘요. 그러면 열 방향으로 내려갈 때 bank 번호가 행마다 하나씩 어긋나서 같은 bank에 몰리지 않아요.

```cpp
__shared__ float tile[32][33];
```

일정한 stride $S$로 읽는 경우를 식으로 적으면 conflict 차수는 $\gcd(S, 32)$예요. Stride 32에서는 32-way, 33에서는 conflict가 없고, 2와 4에서는 각각 2-way와 4-way가 돼요.

Padding은 이렇게 stride와 bank 수의 공약수를 1로 만들어주는 방법이에요. 대신 행마다 4바이트씩 더 써야 한답니다. 한 칸 비워두는 데도 이유가 있었네요ㅎㅎ

두 번째는 swizzle이에요. 이번에는 저장할 열 번호에 행 번호를 XOR해줄 거예요. Memory를 더 쓰지 않고도 bank 접근을 분산할 수 있어요.

```cpp
__shared__ float tile[32][32];

tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[...];   // 쓰기: bank 분산
__syncthreads();
out[...] = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];  // 읽기도 분산
```

행 안의 열 위치를 XOR로 재배열하면 행 방향으로 접근할 때도, 열 방향으로 접근할 때도 32개 bank에 정확히 한 번씩 들어가요. 대신 index 계산이 한 번 들어가고, tile 폭은 2의 거듭제곱이어야 해요. 공간을 아낀 만큼 주소 계산 쪽에서 챙길 게 생기는 셈이에요.

![Bank conflict](images/bank-conflict.svg?v=3)

Lane과 행·열의 대응 자체를 뒤집어서 shared memory 접근을 연속으로 만드는 방법도 떠올릴 수 있어요. 다만 그렇게 하면 global memory 접근이 다시 strided가 돼요. 여기서 기다리는 걸 해결해놓고 저쪽에서 더 읽게 되면.. 참 난감하죠. 두 memory의 접근을 같이 봐야 해요.

앞의 tiled 행렬곱은 이 문제가 없어요. `Bs[k][threadIdx.x]`는 행 방향으로 읽어서 bank가 분산되고, `As[threadIdx.y][k]`는 warp 안의 lane들이 같은 주소를 읽는 broadcast이기 때문이에요.

## Occupancy와 Block 크기

Occupancy는 SM에 실제로 올라간 warp 수를, 그 SM이 동시에 올릴 수 있는 최대 warp 수로 나눈 값이에요. 한 warp가 global memory를 기다릴 때 같은 SM에 있는 다른 warp를 실행할 수 있어서, 올라온 warp가 많으면 그 기다림을 다른 작업으로 채울 여지가 생기죠.

얼마나 올릴 수 있는지는 thread 수, block 슬롯, register, shared memory 가운데 먼저 부족해지는 자원이 정해요. 여기서 register는 thread가 계산 중인 값을 두는 SM 안의 가장 빠른 저장소랍니다.

Tiled 행렬곱에 $T = 32$를 넣으면 block 하나가 1024 thread예요. GPU의 CUDA hardware 기능 세대를 나타내는 번호를 compute capability라고 하는데요. Compute capability 8.9에서는 SM에 상주할 수 있는 thread가 최대 1536개라, 이 block을 하나 올리고 나면 두 번째는 들어갈 수 없어요.

결국 1536개 자리 중 1024개만 써서 occupancy가 66.7%가 돼요. $T = 16$이면 block당 256 thread라 여섯 block으로 1536개를 채울 수 있고요. 자리는 남았는데 block째로는 안 들어가네요..^^;;

![난처하게 땀을 흘리는 문](/images/naver-moon/moon-3.png)

![Occupancy](images/occupancy-residency.svg)

Tile을 키우면 재사용은 늘지만, block까지 커져 SM에 올릴 수 있는 병렬성이 줄어들어요. 그래서 재사용을 더 늘리고 싶을 때는 thread 하나가 $C$의 원소 여러 개를 register에 두고 누적하는 register tiling으로 이어져요. Block 크기만 계속 늘리는 데에는 한계가 있는 거죠.

참, occupancy를 실행 시간의 비율로 읽지는 말아주세요. SM이 고를 수 있는 warp 수의 상한을 보는 값이에요. Occupancy가 낮아도 warp마다 하는 일이 많으면 빠를 수 있답니다.

## Warp Divergence

이번에는 같은 warp 안에서 갈 길이 달라지는 경우예요. Warp는 32개 lane에 같은 명령을 한 번에 발행하는데, `if` 조건이 lane마다 다르면 일부는 참 경로로, 나머지는 거짓 경로로 가야 하죠. 이때 생기는 게 warp divergence예요.

Lane 번호를 $\ell \in \{0,\ldots,31\}$, branch 조건을 $p_\ell$로 놓고, 참인 lane의 집합을 $A$, 거짓인 lane의 집합을 $B$로 적어볼게요.

$$
A = \{\ell \mid p_\ell = 1\}, \qquad
B = \{\ell \mid p_\ell = 0\}
$$

두 집합이 모두 비어 있지 않으면 한 warp 안에서 갈 길이 나뉜 거예요. 아래 코드는 lane 번호로 딱 절반을 나눠요.

```cpp
int lane = threadIdx.x & 31;

if (lane < 16)
    A();
else
    B();
```

모든 warp에서 lane 0~15는 `A`, lane 16~31은 `B`를 골라요. 두 경로를 동시에 실행할 수는 없으니, 먼저 lane 0~15만 켜고 `A`를 실행한 뒤 lane 16~31만 켜고 `B`를 실행해요. 현재 명령의 결과를 쓸 lane을 표시하는 32비트 값이 active mask예요.

다 같이 들어왔는데 절반씩 차례를 기다리네요.

`A`와 `B`가 각각 $n_A$, $n_B$개의 warp 명령으로 컴파일됐다고 해볼게요. 한 경로만 선택하면 그 구간에 $n_A$번 발행하면 되지만, 두 경로가 모두 선택되면 $n_A + n_B$번이 필요해요.

이 발행 횟수에 lane 수를 또 곱하지는 않아요. 16:16으로 갈리든 31:1로 갈리든, 두 경로의 길이가 같으면 발행되는 명령 수도 같답니다. 발행된 lane 자리 중 실제로 켜진 비율을 $\eta$로 두고 계산하면 이렇게 돼요.

$$
\eta =
\frac{|A|n_A + |B|n_B}
     {32(n_A+n_B)}
$$

두 경로의 길이가 같아서 $n_A = n_B$이면 $\eta = 1/2$예요. 한쪽에 한 명만 남았다고 그 경로를 생략해줄 수는 없으니까요..ㅎㅎ

![고개를 돌려 못마땅하게 보는 문](/images/naver-moon/moon-17.png)

이 예제에서 divergence를 피하려면 warp 경계에 맞춰 조건을 나누면 돼요. 이번에는 lane이 아닌 warp 번호를 볼게요.

```cpp
int warp = threadIdx.x >> 5;

if ((warp & 1) == 0)
    A();
else
    B();
```

짝수 warp는 32개 lane이 모두 `A`를, 홀수 warp는 모두 `B`를 선택해요. 각 warp 안에서는 조건이 같아졌죠? 서로 다른 warp끼리 다른 코드를 실행하는 건 divergence가 아니에요.

그런데 소스에 `if`가 있다고 실제 branch까지 꼭 생기는 건 아니에요. 본문이 짧으면 compiler가 branch를 없애고 predicated instruction으로 바꾸기도 하거든요. Predication은 모든 lane에 명령을 한 번 발행하되, 조건이 참인 lane만 결과를 쓰게 하는 방식이에요.

```cpp
float y = x;
if (lane < 16)
    y = 2.0f * x;
```

GPU 기계어인 SASS로 보면 개념적으로는 아래와 같아요. 정확한 명령 이름이나 register는 architecture와 compiler 버전에 따라 달라지니, 여기서는 조건을 만든 뒤 어떻게 쓰는지 봐주세요.

```text
ISETP.LT ... P0, lane, 16
@P0 FMUL  y, x, 2.0
```

여기서는 warp가 두 경로로 갈라지지 않아요. `FMUL`을 한 번 발행하고, predicate `P0`가 참인 lane만 결과를 써요. Branch divergence는 없지만 꺼진 lane이 그 명령에서 유효한 일을 하는 것도 아니고요.

Divergent branch와 predication 모두 켜진 lane의 비율을 낮출 수 있어요. 다만 실제로 경로를 나눠 실행했는지는 구분해야 한답니다. `if`만 보고 결론을 내리기에는 compiler도 하는 일이 있네요ㅎㅎ

## Reduction

마지막으로 reduction을 볼게요. 배열 $N$개를 값 하나로 줄이는 연산이에요. 합, 최댓값, 평균이 여기 들어가고, ML kernel에서도 softmax의 최댓값과 분모 합, layernorm의 평균과 분산을 구할 때 계속 만나게 돼요.

![주먹을 들고 힘내는 문](/images/naver-moon/moon-114.png)

트리처럼 접어가면 단계마다 절반의 thread가 두 값을 합쳐요. 단계 수는 $\log_2 N$이지만 총 덧셈은 여전히 $N - 1$개예요. 덧셈할 양은 그대로 두고 병렬로 처리해서 깊이를 줄이는 거죠.

다만 다음 단계로 갈 때는 앞 단계의 쓰기가 끝났어야 해요. 아래 네 버전은 이 동기화 비용을 어디까지 줄일 수 있는지 따라가보는 과정이랍니다.

![Reduction tree](images/reduction-tree.svg?v=2)

기본 구조는 multi-pass예요. Block마다 shared memory에서 자기 몫의 부분합을 만든 뒤, 그 부분합 배열에 같은 kernel을 다시 실행해요. 값 하나가 남을 때까지 반복하는 거죠. 예를 들어 $2^{24}$개 입력을 block 256으로 처리하면 65,536개 → 256개 → 1개로, 세 번 실행하게 돼요.

버전 0은 트리 모양을 그대로 코드로 옮긴 형태예요. `tid`는 block 안의 thread 번호이고, `buf`는 shared memory에 올려둔 입력이에요.

```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        buf[tid] += buf[tid + s];
    __syncthreads();
}
```

이 코드는 두 군데에서 비용을 써요. 먼저 active lane이 흩어져 있어요. `s = 1`일 때는 짝수 번호, `s = 2`일 때는 4의 배수만 조건을 만족하죠. 일하는 lane이 16, 8, 4, ...개로 줄어도 그 warp의 명령은 계속 발행해야 해요.

`%` 연산도 남아 있어요. 제수인 `2 * s`가 반복마다 바뀌어서 compiler가 비트 연산으로 바꾸지 못하고, 나눗셈 명령 묶음이 남는 거예요. 더하는 코드는 짧은데 그 앞뒤로 일이 붙네요ㅠㅠ

버전 1에서는 일할 thread를 block 앞쪽에 연속으로 모아줄게요. Sequential addressing이라고 하는 방식이에요.

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        buf[tid] += buf[tid + s];
    __syncthreads();
}
```

두 버전 모두 256개 원소를 여덟 단계로 줄여요. 단계 $j \in \{0,\ldots,7\}$에서 덧셈하는 thread 수도 $a_j = 256 / 2^{j+1}$로 같고요. 달라지는 건 그 $a_j$개 thread가 몇 개 warp에 걸쳐 있느냐예요.

버전 0은 active thread가 block 전체에 흩어져 있어요. 조건이 참인 thread를 하나라도 가진 warp가 $\min(8, a_j)$개이고, 여덟 단계를 합하면 47이에요. 버전 1은 앞에서부터 모으니 $\lceil a_j / 32 \rceil$개면 되고, 합하면 12로 줄어요.

`s = 128`에서는 앞 warp 4개가 통째로 일하고, `s = 64`에서는 2개, `s = 32`에서는 1개가 일해요. 이 세 단계는 warp 안의 조건이 모두 같아요. `s = 16`부터는 첫 warp 안에서 조건이 나뉘지만, active lane은 여전히 앞에서부터 연속이고요.

`buf[tid]`와 `buf[tid + s]`도 연속 주소라 bank conflict가 없어요. `%`까지 사라졌네요. 일할 사람들 자리만 모았는데 정리가 꽤 됐죠ㅎㅎ

![엄지를 들어 보이는 문](/images/naver-moon/moon-13.png)

![Reduction lanes](images/reduction-lanes.svg)

버전 2는 warp shuffle을 써요. `s = 16`부터는 첫 warp 하나만 일하니, 그 경계에서는 shared memory를 오가며 `__syncthreads()`로 기다릴 필요 없이 register에서 접어갈 수 있어요.

```cpp
if (tid < 32) {
    float x = buf[tid] + buf[tid + 32];
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, off);
    if (tid == 0) out[blockIdx.x] = x;
}
```

`__shfl_down_sync`는 warp 안에서 register 값을 lane끼리 직접 전달하는 함수예요. 첫 인자는 참여 lane을 표시하는 mask이고, 세 번째 인자는 몇 lane 뒤의 값을 가져올지 정해요. 이렇게 하면 마지막 여섯 단계의 shared memory 왕복과 block barrier가 빠진답니다.

여기서 mask `0xffffffffu`를 쓸 수 있는 건 첫 warp 전체가 `tid < 32`를 만족하기 때문이에요. 일부 lane만 참여하는 코드라면 참여 lane 전원이 같은 mask로 같은 함수를 실행해야 해요. Mask 숫자만 그대로 가져다 쓰면 되는 건 아니니 이 부분은 꼭 같이 봐주세요.

버전 3에서는 block마다 atomic을 한 번만 할 거예요. Atomic은 여러 thread가 같은 주소를 동시에 고쳐도 한 번에 하나씩 적용되도록 보장하는 연산이에요.

Multi-pass 대신 각 block의 lane 0이 `atomicAdd(out, x)`를 한 번 실행해서, 구해둔 부분합을 결과에 바로 더해요. 원소마다 atomic을 쓰면 같은 주소에 $N$번 접근하지만, block reduction을 먼저 하면 block 수만큼만 접근하면 되죠.

실행 전에는 `cudaMemset`으로 결과 변수를 0으로 만들어주세요. 그리고 `float` atomic은 실행마다 덧셈 순서가 달라질 수 있어요. 마지막 비트까지 똑같은 결과가 나온다고 보장하지는 않는답니다.

원소당 연산이 한 번뿐인 reduction은 memory 대역폭에 묶여요. 그래서 잘 구현한 reduction이 어디까지 빨라질 수 있는지 볼 때는 같은 크기의 memcpy 속도를 상한으로 삼아요.

CUDA에 포함된 CUB의 `DeviceReduce::Sum`은 임의 타입과 크기를 처리하면서 이 구간에 도달하는 구현이에요. 위 버전 3은 `float` 배열 하나로 범위를 정해서 그 구조를 보여준 것이고요.

다 더해서 값 하나 만드는 동안에도 memory를 읽고, 기다리고, 값을 전달하는 비용을 꽤 챙겨야 하네요. 아래 전체 소스에서는 각 단계가 어떻게 이어지는지 보실 수 있어요~

![하트를 만들어 인사하는 문](/images/naver-moon/moon-22085.png)

## 소스 코드

세 kernel의 전체 소스는 [gemm_bench.cu](/code/cuda-03/gemm_bench.cu), [transpose_bench.cu](/code/cuda-03/transpose_bench.cu), [reduce_bench.cu](/code/cuda-03/reduce_bench.cu)에 있어요. 컴파일 명령도 함께 적어둘게요.

```bash
nvcc -O3 -arch=sm_89 -o gemm_bench gemm_bench.cu
nvcc -O3 -arch=sm_89 -o transpose_bench transpose_bench.cu
nvcc -O3 -arch=sm_89 -std=c++17 -o reduce_bench reduce_bench.cu
```

## 참고

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): coalescing, shared memory, bank conflict, occupancy, branch predication의 기준 문서
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): SIMT divergence, 동기화, atomics, warp intrinsic의 정확한 의미
- [Mark Harris, An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/): coalescing, shared tile, padding을 한 예제로 보여주는 표준 사례
- [Andreas Holt, Shared-Memory Tiled Matrix Multiplication](https://andreasholt.com/posts/shared-tiled-matmul/): tiled GEMM의 그림과 경계 처리까지 붙인 설명
- [Lei Mao, CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/): bank 주소 매핑의 상세
- [Lei Mao, CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/): swizzle 주소 매핑의 상세
- [Fabian Schütze, Visualizing Bank Conflicts](https://fabianschuetze.github.io/bankconflictscuda.html): 현대 아키텍처의 bank 동작 보충
- [Mark Harris, Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf): reduction을 일곱 단계로 개선하는 고전이에요. 오래된 자료라 warp-synchronous 코드를 그대로 복사해서 쓰시면 안 돼요.
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/): shuffle과 계층적 atomic을 설명해요. 코드는 현대식 `__shfl_down_sync()`로 바꿔 읽어주세요.
- [Lei Mao, CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/): batched reduction 구현 중심의 정리
- [CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html): threadblock/warp/thread tiling과 register 재사용, double buffering으로 이어지는 상위 레퍼런스
- [Simon Boehm, How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM): register tiling부터 warptiling까지 가는 워크로그
- [CUB](https://nvidia.github.io/cccl/cub/): `WarpReduce → BlockReduce → DeviceReduce` 계층의 production 구현

스티커: LINE Moon · [Moon & James](https://store.line.me/stickershop/product/1/en) · [LINE Characters in Love!](https://store.line.me/stickershop/product/1252/en)
