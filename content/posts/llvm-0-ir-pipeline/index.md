---
title: "LLVM 00: C 코드가 기계어가 되기까지 — IR와 컴파일 파이프라인"
date: 2026-08-25
draft: false
tags: ["LLVM", "Compiler", "IR", "clang", "Assembly"]
categories: ["Compiler"]
series: ["LLVM"]
summary: "clang으로 C 코드를 LLVM IR로 뽑고, IR을 한 줄씩 해독하고, llc로 ARM 어셈블리까지 내려 세 층을 직접 대응시킨다. 최적화 pass가 코드를 바꾸는 과정을 -O0와 -O1의 diff로 관찰하고, opt가 조용히 아무것도 안 하게 만드는 optnone 함정을 기록한다."
---

컴파일러 공부를 시작했다. 커리큘럼은 llvm → licm → opencl → tvm 순서다. 이 글은 그 첫 기록이다. 목표는 하나다. C 코드 한 조각이 기계어가 되기까지의 층을 직접 눈으로 확인하는 것.

읽은 자료는 [LLVM for Grad Students](https://www.cs.cornell.edu/~asampson/blog/llvm.html)의 앞 세 챕터다. 저자는 LLVM을 이렇게 정의한다. C/C++ 같은 native 언어를 위한, hackable한 ahead-of-time 컴파일러. ahead-of-time(AOT)은 실행 전에 전부 기계어로 번역해 두는 방식이다. 실행 중에 번역하는 JIT, 번역 없이 매번 해석하는 인터프리터와 대비된다.

## 전체 그림

```
Test.c ──[clang 프런트엔드]──> Test.ll ──[pass들]──> Test.ll' ──[llc 백엔드]──> Test.s ──> 실행 파일
(C 소스)  파싱→AST→타입검사      (IR)     IR→IR 변환   (최적화된 IR)            (어셈블리)
```

모든 단계 사이를 흐르는 것은 IR 하나다. LLVM의 특징이 여기서 나온다. 다른 컴파일러의 IR은 실행 중인 컴파일러의 메모리 안에만 존재하는 객체 그물이라 파일로 적을 수 없다. LLVM IR은 문법이 공개된 텍스트 규격이다. 파일로 저장하고, 손으로 고치고, 다시 컴파일러에 입력할 수 있다. 이 글에서 그 세 가지를 전부 해 본다.

## 준비

macOS 기본 clang에는 `llc`, `opt`가 없다. Homebrew로 LLVM을 설치했다.

```bash
brew install llvm
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
```

LLVM은 keg-only라 PATH를 직접 등록해야 한다. 시스템 clang과의 충돌을 피하기 위한 Homebrew의 정책이다.

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

`-emit-llvm -S`는 기계어까지 가지 말고 사람이 읽을 수 있는 IR에서 멈추라는 뜻이다. 나온 `Test.ll`의 func1 부분에 주석을 달아 해독했다.

```llvm
define i32 @func1() #0 {          ; i32(32비트 정수)를 반환하는 함수 func1. #0은 attributes 묶음 참조
  %1 = alloca i32, align 4        ; 스택에 i32 한 칸 확보. 그 주소에 %1이라는 이름표
  store i32 4, ptr %1, align 4    ; 그 주소에 4 저장            ← int a = 4;
  %2 = load i32, ptr %1, align 4  ; 그 주소에서 읽어 %2에 담기   ← return a의 a 읽기
  ret i32 %2                      ; %2 반환
}
```

기호는 네 개만 알면 본문이 읽힌다.

| 기호 | 뜻 |
|---|---|
| `i32` | 32비트 정수 타입. i1, i8, i64도 있다 |
| `%이름` | 지역 이름표. 가상 레지스터라 개수 제한이 없다 |
| `@이름` | 전역 이름표. 함수 이름이 여기 속한다 |
| `;` | 주석 |

파일 머리의 `target triple`과 꼬리의 `attributes`, `!` 메타데이터는 환경 설정이다. 본문 해독에는 필요 없어 건너뛴다. 한 가지만 적어 두면, 옛 LLVM(14 이전)은 `ptr` 대신 `i32*`로 타입까지 표기했다. LLVM 15부터 opaque pointer로 통일됐다. 문법 세대 차이일 뿐 뜻은 같다.

## IR에서 어셈블리로

```bash
llc Test.ll -o Test.s
```

`llc`는 백엔드다. IR을 타겟 CPU의 어셈블리로 내린다. 이 기계는 Apple Silicon이라 ARM64 어셈블리가 나온다. func1의 대응은 다음과 같다.

| Test.ll (가상) | Test.s (ARM 실물) | 뜻 |
|---|---|---|
| `%1 = alloca i32` | `sub sp, sp, #16` | 스택 프레임 확보. `%1`의 실체는 `sp+12` 칸 |
| `store i32 4, ptr %1` | `mov w8, #4` → `str w8, [sp, #12]` | 4를 레지스터에 담아 스택에 저장 |
| `%2 = load i32, ptr %1` | `ldr w0, [sp, #12]` | 스택에서 읽어 w0에. `%2`의 실체는 w0 |
| `ret i32 %2` | `add sp, sp, #16` → `ret` | 프레임 반납 후 복귀. w0가 반환값 |

가상 이름표(%N)를 실물 장소(레지스터, 스택 칸)에 배정하는 것이 백엔드의 일이라는 게 표에서 그대로 보인다. 어셈블리 파일의 줄은 세 종류다. `.`으로 시작하는 줄은 어셈블러 지시자라 건너뛰고, `이름:`은 라벨, 들여쓰인 줄만 실제 CPU 명령이다.

IR이 텍스트라는 것도 여기서 확인했다. `Test.ll`의 `store i32 4`를 손으로 `store i32 9`로 고치고 다시 `llc`를 돌리면 `mov w8, #9`가 나온다. 컴파일러를 거치지 않고 IR을 직접 편집해 프로그램을 바꾼 것이다.

## 최적화를 diff로 관찰하기

읽을 수 있는 IR의 진짜 장점은 최적화 전후를 비교할 때 나온다.

```bash
clang -O1 -emit-llvm -S Test.c -o Test_O1.ll
```

`-O0`(기본값)의 func1은 alloca → store → load → ret 네 줄이었다. `-O1`은 한 줄이다.

```llvm
define noundef i32 @func1() local_unnamed_addr #0 {
  ret i32 4
}
```

최적화 pass가 "이 함수는 4를 메모리에 넣었다가 곧바로 꺼내 반환한다, 답은 항상 4다"를 증명하고 변수의 존재 자체를 지웠다. C 코드는 그대로인데 프로그램이 4줄에서 1줄이 됐고, 그 과정이 텍스트 diff로 남는다.

pass는 컴파일러가 프로그램 전체(IR)를 한 차례 훑으면서 정해진 한 가지 분석 또는 변환을 수행하는 작업 단위다. `-O1`은 pass 여러 개를 정해진 순서로 묶어 돌리는 clang 옵션이고, `opt`는 pass를 낱개로 골라 돌리는 도구다.

## 함정: optnone

`opt`로 mem2reg(지역 변수를 메모리에서 레지스터로 승격시키는 pass) 하나만 돌려 봤다.

```bash
opt -passes=mem2reg Test.ll -S -o Test_m2r.ll
```

결과가 입력과 똑같았다. 원인은 attributes에 있었다. `-O0`로 IR을 뽑으면 clang이 모든 함수에 `optnone`이라는 "최적화 금지" 표식을 붙이고, `opt`는 그 표식을 존중해 pass를 건너뛴다. attributes가 장식이 아니라 pass의 실행 여부를 제어하는 스위치라는 것을 여기서 배웠다.

표식 없이 뽑으면 정상 동작한다.

```bash
clang -Xclang -disable-O0-optnone -emit-llvm -S Test.c -o Test_noopt.ll
opt -passes=mem2reg Test_noopt.ll -S
```

func1이 `ret i32 4` 한 줄로 접힌다. IR로 pass 실험을 할 때는 이 옵션으로 뽑는 것이 표준이다.

## 정리

- 층은 셋이다. C 소스(사람) → IR(컴파일러 내부의 공개 초안, CPU 무관) → 어셈블리(CPU 실물 명령)
- clang은 프런트엔드이자 파이프라인 전체의 지휘자다. llc는 백엔드, opt는 pass 낱개 실행기다
- LLVM IR은 저장·수정·재입력이 되는 정식 언어다. 최적화가 한 일을 diff로 잡을 수 있는 이유다
- `-O0` IR에는 optnone이 붙는다. opt가 조용히 아무것도 안 하면 이것부터 의심한다

다음 글은 licm(loop invariant code motion)이다. 반복문 안의 불변 계산을 밖으로 빼는 변환 pass 하나를 골라, 이번과 같은 방식으로 전후 IR을 비교한다.
