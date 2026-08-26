---
title: "00 LLVM IR and the Compilation Pipeline: From C Code to Machine Code"
date: 2026-08-25
draft: false
math: true
tags: ["LLVM", "Compiler", "IR", "clang", "Assembly"]
categories: ["Compiler"]
series: ["LLVM"]
summary: "인터프리터, JIT, AOT의 차이에서 시작해 clang으로 C 코드를 LLVM IR로 뽑아 해독하고, llc로 어셈블리를 거쳐 실행 파일까지 내려가는 컴파일 파이프라인 전체를 다룬다. 최적화 pass가 IR을 바꾸는 과정을 -O0와 -O1의 diff로 관찰하고, opt가 조용히 아무것도 안 하게 만드는 optnone을 기록한다."
---

컴파일러는 사람이 읽고 쓰는 소스 코드를 CPU가 실행하는 기계어로 번역하는 프로그램이다. CPU는 기계어만 실행하므로, C처럼 기계어로 번역되는 언어로 쓴 코드는 실행 전에 이 번역을 거친다.

번역을 언제 하느냐에 따라 프로그램을 실행하는 방식이 셋으로 갈린다.

| 방식 | 번역 시점 | 실행 사이에 남는 것 | 예 |
|---|---|---|---|
| 인터프리터 | 번역하지 않고 실행할 때마다 소스를 해석 | 없음 | CPython |
| JIT(just-in-time) | 실행 중에, 자주 실행되는 부분만 | 메모리에 둔 기계어 | Java JVM, 브라우저의 V8 |
| AOT(ahead-of-time) | 실행 전에 전부 | 실행 파일 | clang, GCC |

이 차이는 같은 함수를 반복 호출하면 시간으로 드러난다. 아래는 동일한 정수 반복 계산을 CPython(인터프리터), numba(JIT), `clang -O2`로 미리 컴파일한 C 라이브러리(AOT)로 12번씩 호출해 기록한 호출별 시간이다. 반환값은 셋 다 같다.

![Per-call time of the same function under an interpreter, a JIT, and an AOT build](images/jit_aot_interp.gif?v=2#medium)

인터프리터는 호출마다 같은 해석을 되풀이해 매번 같은 비용을 낸다. JIT는 첫 호출에 컴파일 비용을 몰아 내고(warmup) 그다음부터 번역된 기계어를 재사용하며, AOT는 그 비용을 실행 전에 내서 첫 호출부터 빠르다.

LLVM은 C/C++ 같은 native 언어(가상 머신 없이 CPU가 직접 실행하는 기계어로 번역되는 언어)를 위한 AOT 컴파일러다.

같은 구조가 [CUDA C 기초]({{< relref "/posts/cuda-c-basics" >}}#nvcc-컴파일-파이프라인)에 이미 나왔다. NVIDIA의 CUDA 컴파일러인 nvcc는 `.cu` 파일을 CPU에서 실행할 host 코드와 GPU에서 실행할 device 코드로 나누어 처리하는데, device 코드를 PTX라는 중간 명령어로 바꾸고, PTX를 GPU 기계어인 SASS로 내린다. CPU 쪽 컴파일러도 같은 방식으로 중간 단계를 거쳐 내려가며, 그 대표가 LLVM이다. 실제로 nvcc의 device 코드 컴파일러인 cicc가 LLVM 위에서 만들어져 있어서, 두 스택은 층별로 그대로 대응한다.

![The LLVM stack and the nvcc stack from the CUDA C post, layer by layer](images/cpu-gpu-stack.svg)

## 파이프라인

![C 소스가 clang, opt, llc를 거쳐 실행 파일이 되는 파이프라인 필기 그림](images/pipeline.svg?v=2)

- **프런트엔드(clang)**: `Test.c` → `Test.ll`. 파싱과 타입 검사.
- **미들엔드(opt)**: `Test.ll` → `Test.ll'`. pass들이 IR을 고쳐 쓰는 최적화.
- **백엔드(llc, as, ld)**: `Test.ll'` → `Test.s` → `Test.o` → `a.out`. 기계어 생성과 조립.

세 구간 사이를 흐르는 것은 IR(intermediate representation, 소스와 기계어 사이의 중간 언어) 하나다. 다른 컴파일러의 IR은 메모리 안에만 존재하는 객체 구조라 파일로 적을 수 없지만, LLVM IR은 문법이 공개된 텍스트 규격이라 저장하고, 손으로 고치고, 다시 컴파일러에 입력할 수 있다.

## C에서 IR로

예제는 두 함수짜리 C 파일이다.

```c
// Test.c
int func1(void) { int a = 4; return a; }
int main(void)  { return 0; }
```

```bash
clang -emit-llvm -S Test.c   # → Test.ll
```

`-emit-llvm -S`는 기계어까지 가지 않고 사람이 읽을 수 있는 IR에서 멈추라는 옵션이다. 나온 `Test.ll`의 func1은 다음과 같다.

```llvm
define i32 @func1() #0 {
  %1 = alloca i32, align 4
  store i32 4, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}
```

| IR | 뜻 | 대응 C 코드 |
|---|---|---|
| `define i32 @func1() #0` | i32(32비트 정수)를 반환하는 함수 func1 정의. `#0`은 attributes 묶음 참조 | `int func1(void)` |
| `%1 = alloca i32, align 4` | 스택에 i32 한 칸 확보. 그 주소에 `%1`이라는 이름표 | `int a`의 자리 |
| `store i32 4, ptr %1` | 그 주소에 4 저장 | `a = 4` |
| `%2 = load i32, ptr %1` | 그 주소에서 읽어 `%2`에 담기 | `return a`의 a 읽기 |
| `ret i32 %2` | `%2` 반환 | `return` |

기호는 네 개만 알면 본문이 읽힌다.

| 기호 | 뜻 |
|---|---|
| `i32` | 32비트 정수 타입. i1, i8, i64도 있다 |
| `%이름` | 지역 이름표. 가상 레지스터라 개수 제한이 없다 |
| `@이름` | 전역 이름표. 함수 이름이 여기 속한다 |
| `;` | 주석 |

파일 머리의 `target triple`과 꼬리의 `attributes`, `!` 메타데이터는 환경 설정이라 본문 해독에는 필요 없다. 그리고 옛 LLVM(14 이전)은 `ptr` 대신 `i32*`로 타입까지 표기했는데, LLVM 15부터 opaque pointer로 통일됐다. 문법 세대가 다를 뿐 뜻은 같다.

## IR에서 어셈블리로

```bash
llc Test.ll -o Test.s
```

`llc`는 백엔드다. llc는 IR을 target triple(IR 파일 첫머리에 적힌 대상 CPU와 운영체제 명세)이 가리키는 CPU의 어셈블리로 내린다. 그래서 출력되는 어셈블리는 대상 CPU마다 다르다. x86-64를 지정하면 x86-64 어셈블리가, ARM64를 지정하면 ARM64 어셈블리가, RISC-V를 지정하면 RISC-V 어셈블리가 나오고, `llc -mtriple=x86_64-pc-linux-gnu Test.ll`처럼 옵션으로 대상을 바꿀 수 있다. IR은 하나인데 백엔드가 대상별로 갈라지는 이 구조가 파이프라인 절에서 말한 IR의 역할이다.

다음은 ARM64 대상의 출력에서 func1을 대응시킨 표다.

| Test.ll (가상) | Test.s (ARM 실물) | 뜻 |
|---|---|---|
| `%1 = alloca i32` | `sub sp, sp, #16` | 스택 프레임 확보. `%1`의 실체는 `sp+12` 칸 |
| `store i32 4, ptr %1` | `mov w8, #4` → `str w8, [sp, #12]` | 4를 레지스터에 담아 스택에 저장 |
| `%2 = load i32, ptr %1` | `ldr w0, [sp, #12]` | 스택에서 읽어 w0에. `%2`의 실체는 w0 |
| `ret i32 %2` | `add sp, sp, #16` → `ret` | 프레임 반납 후 복귀. w0가 반환값 |

가상 이름표(%N)를 실물 장소(레지스터, 스택 칸)에 배정하는 것이 백엔드의 일이다. 어셈블리 파일의 줄은 세 종류로 나뉜다. `.`으로 시작하는 줄은 어셈블러 지시자라 건너뛰고, `이름:`은 라벨, 들여쓰인 줄만 실제 CPU 명령이다.

`Test.ll`의 `store i32 4`를 `store i32 9`로 바꾸고 `llc`를 다시 돌리면 `mov w8, #9`가 나온다. IR 텍스트 자체가 컴파일러의 입력이라, 프런트엔드를 거치지 않고 IR을 고치는 것만으로 프로그램이 바뀐다.

## 어셈블리에서 실행 파일로

`Test.s`는 텍스트 파일이다. `mov w8, #4`가 문자 그대로 적혀 있을 뿐이고, CPU는 문자를 실행하지 못한다. CPU가 실행하는 것은 명령마다 정해진 비트 패턴이고, 어셈블러(as)가 문자 표기를 그 비트 패턴으로 바꾼 것이 object 파일 `Test.o`다.

프로그램 하나는 보통 여러 소스 파일로 만들고, 소스 파일 하나가 object 파일 하나가 된다. 여기에 라이브러리(printf처럼 자주 쓰는 함수를 미리 컴파일해 둔 object 파일 묶음)가 더해진다. Object 파일은 기계어지만 혼자서는 실행되지 않는다. 자기가 호출하는 다른 object 파일이나 라이브러리 속 함수의 주소가 아직 비어 있기 때문이다. 링커(ld)가 이 object 파일들과 라이브러리를 모아 빈 주소를 채우고 하나로 이어 붙이면 실행 파일이 된다.

```bash
clang -c Test.s -o Test.o   # assemble
clang Test.o -o a.out       # link
```

두 단계 모두 clang 명령으로 부를 수 있고, clang이 내부에서 as와 ld를 호출한다.

## pass와 최적화 레벨

pass는 LLVM에 내장된 작은 프로그램으로, IR 전체를 한 차례 훑으며 정해진 한 가지 분석 또는 변환을 수행한다. pass마다 담당하는 변환이 하나씩 정해져 있다.

| pass | 담당하는 한 가지 |
|---|---|
| mem2reg | 지역 변수의 메모리 왕복(alloca, store, load)을 값 전달로 바꾼다 |
| instcombine | 명령 조합을 같은 결과의 더 짧은 조합으로 바꾼다 |
| simplifycfg | 도달할 수 없는 블록을 지우고 분기를 단순화한다 |
| dce | 결과에 영향을 주지 않는 명령을 지운다 (dead code elimination) |
| licm | 반복문 안에서 반복마다 값이 같은 계산을 반복문 앞으로 옮긴다 (loop invariant code motion) |

실행 방법은 둘이다. `opt`는 pass를 낱개로 골라 실행하는 도구이고, clang의 `-O1` `-O2` `-O3` 옵션은 미리 정해둔 pass 목록을 순서대로 실행한다. 목록은 포함 관계다.

$$ O_0(0) \subset O_1(98) \subset O_2(115) \subset O_3(118) $$

레벨을 올리는 대가는 컴파일 시간이고, `-O3`의 추가분은 코드 크기를 늘려서라도 속도를 우선하는 pass들이라 배포 빌드는 보통 `-O2`에서 멈춘다.

괄호 안 숫자는 다음 명령이 출력하는 목록의 항목 수를 센 것으로(LLVM 22 기준), 버전에 따라 달라진다. 위 표의 pass들이 전부 이 목록 안에 들어 있다.

```bash
opt -passes='default<O1>' -print-pipeline-passes Test.ll -S -o /dev/null
```

## 최적화를 diff로 관찰하기

읽을 수 있는 IR의 장점은 최적화 전후를 비교할 때 나온다.

```bash
clang -O1 -emit-llvm -S Test.c -o Test_O1.ll
```

`-O0`(기본값)로 뽑은 `Test.ll`(좌측)과 `-O1`로 뽑은 `Test_O1.ll`(우측)을 VS Code의 diff 편집기로 비교하면 다음과 같다.

![Test.ll and Test_O1.ll compared in the VS Code diff editor](images/opt-diff.png?v=3)

func1의 본문이 어떻게 바뀌었는지 줄 단위로 대응시키면 다음과 같다.

| 왼쪽 `-O0` (최적화 전) | 오른쪽 `-O1` (최적화 후) | 변화 |
|---|---|---|
| `%1 = alloca i32` | 없음 | 변수 a가 쓸 스택 자리가 삭제됨 |
| `store i32 4, ptr %1` | 없음 | 4를 메모리에 쓰는 명령이 삭제됨 |
| `%2 = load i32, ptr %1` | 없음 | 메모리에서 도로 읽는 명령이 삭제됨 |
| `ret i32 %2` | `ret i32 4` | 변수에서 읽은 값 대신 상수 4를 바로 반환 |

최적화 pass가 이 함수는 4를 메모리에 넣었다가 곧바로 꺼내 반환하므로 답이 항상 4라는 것을 증명하고 변수의 존재 자체를 지웠다. C 코드는 그대로인데 프로그램이 네 줄에서 한 줄이 됐고, 그 과정이 텍스트 diff로 남는다.

## optnone

mem2reg는 지역 변수를 메모리에서 레지스터로 승격시키는 pass다. `opt`로 이 pass 하나만 적용할 수 있다.

```bash
opt -passes=mem2reg Test.ll -S -o Test_m2r.ll
```

그런데 `-O0`로 뽑은 IR에 적용하면 출력이 입력과 똑같이 나온다. 원인은 attributes에 있다. `-O0`로 IR을 뽑으면 clang이 모든 함수에 `optnone`이라는 최적화 금지 표식을 붙이고, `opt`는 그 표식을 존중해 pass를 건너뛴다. attributes가 pass의 실행 여부까지 제어하는 것이다.

표식 없이 뽑으면 정상 동작한다.

```bash
clang -Xclang -disable-O0-optnone -emit-llvm -S Test.c -o Test_noopt.ll
opt -passes=mem2reg Test_noopt.ll -S
```

func1이 `ret i32 4` 한 줄로 접힌다. IR로 pass 실험을 할 때는 이 옵션으로 뽑는다.

## 참고

- [LLVM for Grad Students](https://www.cs.cornell.edu/~asampson/blog/llvm.html): What is LLVM? / The Pieces / Understanding LLVM IR 세 챕터.
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html): IR 문법의 공식 정의.
- [The Architecture of Open Source Applications: LLVM](https://aosabook.org/en/v1/llvm.html): Chris Lattner가 쓴 LLVM 설계 배경.
