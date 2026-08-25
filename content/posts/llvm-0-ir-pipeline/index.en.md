---
title: "LLVM 00: From C Code to Machine Code"
date: 2026-08-25
draft: false
tags: ["LLVM", "Compiler", "IR", "clang", "Assembly"]
categories: ["Compiler"]
series: ["LLVM"]
summary: "Emit LLVM IR from C with clang, decode the IR line by line, and lower it to ARM assembly with llc to map all three layers by hand. Observe how optimization passes rewrite code via an -O0 vs -O1 diff, and record the optnone trap that makes opt silently do nothing."
---

I started studying compilers. The curriculum goes llvm → licm → opencl → tvm. This is the first entry. The goal is simple: watch a piece of C code travel through every layer on its way to machine code.

The reading was the first three chapters of [LLVM for Grad Students](https://www.cs.cornell.edu/~asampson/blog/llvm.html). The author defines LLVM as a nice, hackable, ahead-of-time compiler for native languages like C and C++. Ahead-of-time (AOT) means translating everything to machine code before the program runs. JIT translates during execution instead, and interpreters never translate at all, re-interpreting every time.

## The Big Picture

```
Test.c ──[clang frontend]──> Test.ll ──[passes]──> Test.ll' ──[llc backend]──> Test.s ──> executable
(C source)  parse→AST→typecheck  (IR)    IR→IR       (optimized IR)            (assembly)
```

One IR flows between every stage. This is where LLVM differs from other compilers. Most compilers keep their IR as a web of in-memory objects that cannot be written down. LLVM IR is a text format with a published grammar. You can save it to a file, edit it by hand, and feed it back to the compiler. This post does all three.

## Setup

The stock macOS clang ships without `llc` and `opt`, so I installed LLVM via Homebrew.

```bash
brew install llvm
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
```

LLVM is keg-only; you register the PATH yourself. That is Homebrew's policy to avoid clashing with the system clang.

## From C to IR

The example is a two-function C file.

```c
// Test.c
int func1(void) { int a = 4; return a; }
int main(void)  { return 0; }
```

```bash
clang -emit-llvm -S Test.c   # → Test.ll
```

`-emit-llvm -S` means: stop before machine code, at human-readable IR. I annotated func1 in the resulting `Test.ll`:

```llvm
define i32 @func1() #0 {          ; function func1 returning i32 (32-bit int). #0 refers to an attribute group
  %1 = alloca i32, align 4        ; reserve one i32 slot on the stack; its address is named %1
  store i32 4, ptr %1, align 4    ; store 4 at that address            ← int a = 4;
  %2 = load i32, ptr %1, align 4  ; load from that address into %2     ← reading a in "return a"
  ret i32 %2                      ; return %2
}
```

Four symbols are enough to read the body.

| Symbol | Meaning |
|---|---|
| `i32` | 32-bit integer type. i1, i8, i64 also exist |
| `%name` | local name. A virtual register, so there is no limit on how many |
| `@name` | global name. Functions live here |
| `;` | comment |

The `target triple` at the top and the `attributes` / `!` metadata at the bottom are environment configuration; they are not needed to decode the body. One note: older LLVM (pre-15) wrote typed pointers like `i32*` instead of `ptr`. Opaque pointers unified this in LLVM 15. Same meaning, different generation of syntax.

## From IR to Assembly

```bash
llc Test.ll -o Test.s
```

`llc` is the backend: it lowers IR to the target CPU's assembly. On Apple Silicon that is ARM64. The mapping for func1:

| Test.ll (virtual) | Test.s (ARM, physical) | What happens |
|---|---|---|
| `%1 = alloca i32` | `sub sp, sp, #16` | reserve the stack frame; `%1` becomes the slot at `sp+12` |
| `store i32 4, ptr %1` | `mov w8, #4` → `str w8, [sp, #12]` | put 4 in a register, store it to the stack |
| `%2 = load i32, ptr %1` | `ldr w0, [sp, #12]` | load from the stack into w0; `%2` becomes w0 |
| `ret i32 %2` | `add sp, sp, #16` → `ret` | release the frame and return; w0 carries the return value |

The table shows exactly what a backend does: assign virtual names (%N) to physical places (registers, stack slots). An assembly file has three kinds of lines: lines starting with `.` are assembler directives (skip them), `name:` lines are labels, and only the indented lines are actual CPU instructions.

The text-ness of IR is easy to verify here. Hand-edit `store i32 4` into `store i32 9` in `Test.ll`, run `llc` again, and the output shows `mov w8, #9`. The program changed without going through the compiler frontend at all.

## Watching Optimization as a Diff

The real payoff of readable IR shows up when comparing before and after optimization.

```bash
clang -O1 -emit-llvm -S Test.c -o Test_O1.ll
```

At `-O0` (the default), func1 was four lines: alloca → store → load → ret. At `-O1` it is one:

```llvm
define noundef i32 @func1() local_unnamed_addr #0 {
  ret i32 4
}
```

The optimization passes proved "this function stores 4 to memory and immediately loads it back, so the answer is always 4" and erased the variable's existence. The C code did not change, the program went from four lines to one, and the whole event is captured in a text diff.

A pass is a unit of work in which the compiler sweeps over the entire program (IR) once, performing one predetermined analysis or transformation. `-O1` is a clang option that runs many passes in a fixed order; `opt` is the tool that runs a single pass of your choosing.

## The Trap: optnone

I tried running just mem2reg (the pass that promotes local variables from memory to registers) with `opt`:

```bash
opt -passes=mem2reg Test.ll -S -o Test_m2r.ll
```

The output was identical to the input. The culprit was in the attributes. When clang emits IR at `-O0`, it stamps every function with `optnone`, a "do not optimize" marker, and `opt` respects it by skipping the function entirely. This is where I learned that attributes control whether passes run at all.

Emit without the marker and it works:

```bash
clang -Xclang -disable-O0-optnone -emit-llvm -S Test.c -o Test_noopt.ll
opt -passes=mem2reg Test_noopt.ll -S
```

func1 folds to a single `ret i32 4`. When emitting IR for pass experiments, this option is the standard way.

## Recap

- There are three layers: C source (for humans) → IR (the compiler's public draft, CPU-agnostic) → assembly (physical CPU instructions)
- clang is the frontend and also the driver of the whole pipeline; llc is the backend; opt runs passes one at a time
- LLVM IR is a real language you can save, edit, and feed back. That is why optimization becomes observable as a diff
- `-O0` IR carries optnone. When opt silently does nothing, suspect it first

Next up is licm (loop invariant code motion): pick one transformation pass that hoists loop-invariant computation out of loops, and compare the IR before and after, the same way as here.
