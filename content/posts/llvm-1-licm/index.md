---
title: "01 Loop-Invariant Code Motion: From Loop Body to Preheader"
date: 2026-08-31
draft: false
math: false
tags: ["LLVM", "Compiler", "IR", "Optimization", "LICM"]
categories: ["Compiler"]
series: ["LLVM"]
summary: "LLVM LICM은 반복마다 같은 곱셈과 덧셈을 반복문 본문에서 preheader로 옮긴다. 명령의 operand가 모두 반복 불변이어도 메모리 효과나 0회 실행 경로의 안전성이 증명되지 않으면 반복문 안에 남는다."
---

LICM(loop-invariant code motion)은 반복문 안의 반복 불변 명령을 반복문 밖으로 옮기는 LLVM [최적화 pass]({{< relref "/posts/llvm-0-ir-pipeline" >}}#pass와-최적화-레벨)다. 반복 불변 명령은 반복문이 도는 동안 같은 결과를 만드는 명령이다. LLVM은 명령을 옮긴 뒤에도 프로그램의 동작이 같다고 증명할 수 있을 때만 LICM을 적용한다.

반복문 앞쪽으로 명령을 옮기는 변환을 hoisting(호이스팅)이라고 한다. 다음 `transform` 함수에서는 `factor`를 만드는 곱셈과 덧셈이 hoisting 대상이다.

## 반복마다 같은 계산

`unsigned`는 0 이상의 정수를 나타내는 C 타입이다. `input`과 `output`은 각각 `count`개의 `unsigned` 원소를 가리키고, `count`는 0 이상으로 둔다. `i`는 `count`가 양수일 때 0부터 `count - 1`까지 움직이는 반복 변수다. `scale`과 `offset`은 함수 인자이고, `factor`는 `scale * scale + offset`의 결과다.

```c
// licm.c
void transform(const unsigned *input, unsigned *output,
               int count, unsigned scale, unsigned offset) {
  for (int i = 0; i < count; ++i) {
    unsigned factor = scale * scale + offset;
    output[i] = input[i] * factor;
  }
}
```

operand(피연산자)는 명령이 입력으로 받는 값이다. `scale`과 `offset`은 반복문 안에서 바뀌지 않는다. 그래서 두 값을 operand로 쓰는 `scale * scale`과 `scale * scale + offset`도 반복 불변이다.

`i`는 반복할 때마다 증가한다. `input[i]`와 `output[i]`의 주소도 `i`에 따라 바뀐다. 최적화 전 IR에서는 `factor` 계산이 `for.body`에 있어 반복마다 한 번 실행된다.

## LICM 직전의 IR

`clang`은 [C를 LLVM IR로 바꾸는]({{< relref "/posts/llvm-0-ir-pipeline" >}}#c에서-ir로) 프론트엔드이고, `opt`는 IR에 pass를 실행하는 도구다. 두 도구는 Homebrew LLVM 22.1.8[^version]을 사용한다. 명령의 `LLVM_BIN`은 해당 LLVM 실행 파일이 든 디렉터리 경로다. `-fno-discard-value-names`는 C 식별자를 IR 이름에 남기고, `-Xclang`은 뒤따르는 옵션을 clang 내부 프론트엔드에 전달한다. `-disable-O0-optnone`은 [optnone]({{< relref "/posts/llvm-0-ir-pipeline" >}}#optnone)을 붙이지 않는다.

LLVM IR의 지역 이름은 SSA(static single assignment) 형식을 따른다. SSA에서는 가상 레지스터 하나를 한 번만 정의한다. `phi`는 여러 제어 흐름이 합쳐질 때 어느 이전 basic block의 값을 받을지 고르는 명령이다. basic block은 중간에 분기하지 않고 위에서 아래로 실행되는 IR 명령 묶음이다.

header는 반복문 바깥에서 반복문 안으로 들어갈 때 가장 먼저 거치는 basic block이다. preheader는 반복문 바깥에서 header로만 이어지는 단일 진입 basic block이다. backedge는 반복문 안에서 header로 돌아가는 제어 흐름이고, latch는 backedge가 시작되는 basic block이다. exit block은 반복문을 빠져나온 뒤 도착하는 basic block이다.

LICM 전에 `mem2reg`, `loop-simplify`, `lcssa`를 실행한다.

| pass | 역할 |
|---|---|
| `mem2reg` | `alloca`로 만든 지역 변수 중 승격 가능한 대상을 SSA 값 흐름으로 바꿈 |
| `loop-simplify` | preheader, 하나의 backedge, 반복문 안에서만 들어오는 전용 exit block들을 갖춘 Loop Simplify Form을 보장 |
| `lcssa`(loop-closed SSA) | 반복문 안에서 정의하고 밖에서 쓰는 값이 exit block의 `phi`를 거치게 함 |

loop pass는 반복문을 하나씩 처리한다. LLVM은 loop pass 앞에 `loop-simplify`와 `lcssa`를 자동으로 실행한다. `raw.ll`은 clang이 만든 IR이고, `before.ll`은 세 준비 pass를 실행한 IR이다. 준비 pass는 `before.ll`을 읽기 쉬운 형태로 만들고 이후 비교에 LICM의 변화만 남긴다.

```bash
LLVM_BIN=/opt/homebrew/opt/llvm/bin

"$LLVM_BIN/clang" -O0 -Xclang -disable-O0-optnone \
  -fno-discard-value-names \
  -S -emit-llvm licm.c -o raw.ll

"$LLVM_BIN/opt" -S \
  -passes='mem2reg,loop-simplify,lcssa' \
  raw.ll -o before.ll
```

`transform`은 반복문 안에서 정의한 값을 반복문 밖에서 쓰지 않는다. 그래서 `lcssa`는 `before.ll`에 새 `phi`를 추가하지 않는다.

파일 수준 설정, 매개변수와 함수의 attributes, metadata를 덜어낸 `transform`은 다음과 같다.

`before.ll`의 target triple은 `arm64-apple-macosx15.0.0`이다. `arm64-apple-macosx15.0.0`에서 pointer는 64비트이고 `int`와 `unsigned`는 32비트다.

```llvm
define void @transform(ptr %input, ptr %output,
                       i32 %count, i32 %scale, i32 %offset) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %count
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %mul = mul i32 %scale, %scale
  %add = add i32 %mul, %offset
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i32, ptr %input, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %mul1 = mul i32 %0, %add
  %idxprom2 = sext i32 %i.0 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %output, i64 %idxprom2
  store i32 %mul1, ptr %arrayidx3, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:
  ret void
}
```

다섯 basic block의 역할은 다음과 같다.

| basic block | 역할 |
|---|---|
| `entry` | 함수 시작점이자 preheader. 반복문 바깥에서 `for.cond`로 들어가는 유일한 block |
| `for.cond` | header. `i < count`를 검사하고 `for.body` 또는 `for.end`로 분기 |
| `for.body` | 배열 원소를 읽고 `factor`를 곱한 뒤 결과를 씀 |
| `for.inc` | latch. `i`를 증가시키고 backedge로 `for.cond`에 돌아감 |
| `for.end` | 반복문을 빠져나온 뒤 도착하는 exit block |

IR 명령은 C의 반복문을 다음처럼 나눈다.

| IR | 뜻 |
|---|---|
| `ptr`, `i1`, `i32`, `i64` | 각각 pointer, 1비트 정수, 32비트 정수, 64비트 정수 타입 |
| `br label`, `br i1` | 다른 basic block으로 무조건 분기하거나 `i1` 조건에 따라 분기 |
| `%i.0 = phi ...` | 첫 반복에는 0, 이후 반복에는 `%inc`를 `i`로 선택 |
| `%cmp = icmp slt ...` | signed less-than 비교로 `i < count`를 계산 |
| `%mul`, `%add`, `%mul1` | 각각 `scale * scale`, `%mul + offset`, `input[i] * factor`를 계산 |
| `%idxprom`, `%idxprom2` | 각각 `input`과 `output`의 주소에 쓸 `i`를 `sext`로 `i32`에서 `i64`로 부호 확장 |
| `%arrayidx`, `%arrayidx3` | `getelementptr inbounds`로 각각 `input[i]`와 `output[i]`의 주소를 계산. `inbounds`는 주소가 같은 할당 객체의 허용 범위 안에 있다는 약속 |
| `%0 = load`, `store`, `align 4` | `%arrayidx`에서 `input[i]`를 읽어 `%0`에 정의하고 `%arrayidx3`에 `%mul1`을 씀. 32비트 `unsigned`의 주소가 4바이트 경계에 맞는다고 표시 |
| `%inc = add nsw ...` | `i + 1`을 계산. `nsw`는 signed overflow가 없다는 약속이며 0 이상인 `count`와 `i < count`가 이 조건을 보장 |

## LICM 실행

`-passes='licm'`은 변환 pass로 LICM을 지정한다. LLVM 22의 opt는 Loop Simplify Form, LCSSA, MemorySSA[^mssa]를 자동으로 준비한다. MemorySSA는 각 메모리 사용이 어느 정의의 값을 받는지 나타내는 use-def 관계와 어떤 쓰기가 읽은 값을 바꿀 수 있는지를 기록한 분석 결과다.

`after.ll`은 `before.ll`에 LICM을 실행한 결과다. `diff -u`는 두 텍스트 파일을 줄 단위로 비교한다. `-I '^; ModuleID'`는 입력 파일 이름만 담은 `ModuleID` 주석을 비교에서 제외한다.

```bash
"$LLVM_BIN/opt" -S \
  -passes='licm' \
  before.ll -o after.ll

diff -u -I '^; ModuleID' before.ll after.ll
```

함수 본문에 생긴 차이는 네 줄이다. `-`는 `before.ll`에서 빠진 줄이고, `+`는 `after.ll`에 들어간 줄이다.

```diff
 entry:
+  %mul = mul i32 %scale, %scale
+  %add = add i32 %mul, %offset
   br label %for.cond

 for.body:
-  %mul = mul i32 %scale, %scale
-  %add = add i32 %mul, %offset
   %idxprom = sext i32 %i.0 to i64
```

`%mul`과 `%add`의 위치가 `for.body`에서 `entry`로 바뀌었다. 함수 시작점에서 `for.body`로 가는 모든 경로는 `entry`를 지난다. 이런 관계를 dominance(지배 관계)라고 하며, `entry`가 `for.body`를 지배한다. 그래서 `entry`로 이동한 `%add`의 정의는 `%mul1`이 `%add`를 사용하기 전에 실행된다.

![LICM 전후의 동일한 제어 흐름과 for.body에서 entry로 이동한 mul, add 명령](images/licm-hoisting.svg)

`count`가 0이면 `for.body`는 실행되지 않지만, hoisting된 `%mul`과 `%add`는 `entry`에서 한 번 실행된다. 두 명령은 32비트 결과의 하위 32비트를 남기고 메모리를 읽거나 쓰지 않는다. 모든 `scale`과 `offset` 값에 결과가 정의되므로 speculative execution(결과가 필요한지 결정되기 전에 실행하는 방식)도 안전하다.

명령을 그대로 preheader로 옮기는 LICM 경로는 다음 세 조건을 검사한다.

1. 명령의 모든 operand가 반복 불변이어야 한다.
2. 이동 전후의 메모리 읽기와 쓰기, 함수 호출처럼 외부에서 볼 수 있는 동작이 같아야 한다.
3. 원래 실행되지 않던 경로에서 speculative execution이 생겨도 안전해야 한다.

`%mul`과 `%add`는 세 조건을 모두 만족한다.

## 반복 불변이어도 남는 나눗셈

`transform_div`는 `numerator / denominator`를 각 배열 원소에 곱한다. `numerator`는 나눠지는 수이고, `denominator`는 나누는 수다. `input`과 `output`은 각각 `count`개의 `unsigned` 원소를 가리키며, `count`는 0 이상이다. `count`가 양수일 때 `i`의 범위는 0부터 `count - 1`까지다.

```c
// licm_div.c
void transform_div(const unsigned *input, unsigned *output,
                   int count, unsigned numerator,
                   unsigned denominator) {
  for (int i = 0; i < count; ++i) {
    unsigned factor = numerator / denominator;
    output[i] = input[i] * factor;
  }
}
```

`transform_div`에도 `mem2reg`, `loop-simplify`, `lcssa`, `licm`을 차례로 적용한다. LICM 후에도 `%div`는 `for.body`에 남는다. `udiv`는 두 `unsigned` 정수의 몫을 계산하고 그 결과를 `%div`에 정의하는 IR 명령이다.

```llvm
for.body:
  %div = udiv i32 %numerator, %denominator
  ; input[i]를 읽고 %div를 곱한 뒤 output[i]에 쓰는 명령
```

`count`가 0이면 `for.body`가 실행되지 않는다. 이 경로에서는 `denominator`가 0이어도 나눗셈이 없다. `%div`를 `entry`로 옮기면 0으로 나누는 실행이 새로 생길 수 있다. LLVM IR의 `udiv`는 0으로 나눌 때 undefined behavior(결과와 이후 동작을 LLVM이 정의하지 않는 실행)를 만든다. 두 operand는 반복 불변이지만 speculative execution이 안전하지 않아서 LICM은 `%div`를 옮기지 않는다.

메모리 명령은 주소 외의 조건도 필요하다. alias는 서로 다른 포인터가 같은 메모리 위치를 가리킬 가능성이다. 반복 불변 주소를 읽는 `load`라도 반복문 안의 `store`나 함수 호출이 alias된 위치를 바꿀 수 있으면 읽은 값이 달라진다. LLVM은 alias 분석과 MemorySSA를 이용해 충돌하는 쓰기를 찾고, 반복문이 0회 실행되는 경로에서도 해당 주소를 안전하게 읽을 수 있는지 따로 검사한다.

volatile 접근은 컴파일러가 생략하거나 합치면 안 되는 메모리 접근이다. 실행 횟수 자체가 외부에서 보이는 동작이므로 hoisting할 수 없다. 일반 함수 호출은 호출의 의미, attributes, 분석 결과가 메모리 효과와 다른 부작용이 이동을 허용한다고 증명해야 한다. 그다음에야 나머지 조건을 검사한다.

## 참고

- [LLVM's Analysis and Transform Passes: LICM](https://releases.llvm.org/22.1.0/docs/Passes.html#licm-loop-invariant-code-motion): LICM의 동작과 메모리 명령 이동 조건.
- [LLVM Loop Terminology](https://releases.llvm.org/22.1.0/docs/LoopTerminology.html#loop-simplify-form): header, preheader, latch, exit과 Loop Simplify Form의 정의.
- [Using the New Pass Manager](https://releases.llvm.org/22.1.0/docs/NewPassManager.html#invoking-opt): `opt -passes` 문법과 자동으로 선택되는 pass 실행 범위.
- [MemorySSA](https://releases.llvm.org/22.1.0/docs/MemorySSA.html): 메모리 use-def 관계와 alias 분석을 이용한 충돌 판정.
- [LLVM Language Reference: `udiv`](https://releases.llvm.org/22.1.0/docs/LangRef.html#udiv-instruction): unsigned 나눗셈과 0으로 나누는 경우의 의미.
- [LLVM 22.1.0 LICM implementation](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.0/llvm/lib/Transforms/Scalar/LICM.cpp): 실제 hoisting 안전성 검사와 MemorySSA 사용.

[^version]: 명령과 IR은 `/opt/homebrew/opt/llvm/bin`의 Homebrew LLVM 22.1.8로 생성했다. 다른 LLVM 버전에서는 IR 표기와 pass 실행 구조가 달라질 수 있다.

[^mssa]: LLVM 22.1.8에서 `-passes='licm' -print-pipeline-passes`는 `function(loop-mssa(licm<allowspeculation>)),verify`를 출력한다. `loop-mssa`는 LICM에 MemorySSA를 제공하고, `function(...)`은 모듈(한 IR 파일)의 각 함수에 반복문 pass를 적용한다. `allowspeculation`은 안전성 검사를 통과한 명령의 speculative execution을 허용하는 LICM 설정이고, `verify`는 변환 뒤 IR의 구조를 검사한다.
