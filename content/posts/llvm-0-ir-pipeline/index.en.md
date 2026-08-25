---
title: "00 LLVM IR and the Compilation Pipeline: From C Code to Machine Code"
date: 2026-08-25
draft: false
tags: ["LLVM", "Compiler", "IR", "clang", "Assembly"]
categories: ["Compiler"]
series: ["LLVM"]
summary: "Emit LLVM IR from C with clang, decode the IR line by line, and lower it to ARM assembly with llc to map the three layers. Observe how optimization passes rewrite code via an -O0 vs -O1 diff, and record the optnone attribute that makes opt silently do nothing."
---

A compiler is a program that translates source code, which humans read and write, into the machine code a CPU executes. A CPU executes only machine code, so code written in a language like C goes through this translation before it runs.

Depending on when the translation happens, there are three ways to run a program.

| Method | When it translates | What persists between runs | Examples |
|---|---|---|---|
| Interpreter | never; re-interprets the source on every run | nothing | CPython |
| JIT (just-in-time) | during execution, only for hot code | machine code kept in memory | Java JVM, V8 in browsers |
| AOT (ahead-of-time) | everything, before the program runs | an executable file | clang, GCC |

The difference shows up as time once the same function is called repeatedly. Below are per-call times for the same integer loop called twelve times as CPython (interpreter), numba (JIT), and a C library compiled ahead of time with `clang -O2` (AOT). All three return the same value.

![Per-call time of the same function under an interpreter, a JIT, and an AOT build](images/jit_aot_interp.gif?v=2#medium)

The interpreter repeats the same interpretation on every call and pays the same cost each time. The JIT pays its compilation cost on the first call (warmup) and reuses the translated machine code afterwards, while the AOT build pays that cost before the program runs and is fast from the first call.

LLVM is an AOT compiler for native languages (languages that compile to machine code the CPU executes directly, with no virtual machine) like C and C++.

The same structure appeared in [CUDA C Basics]({{< relref "/posts/cuda-c-basics" >}}#the-nvcc-compilation-pipeline). nvcc, NVIDIA's CUDA compiler, splits a `.cu` file into host code that runs on the CPU and device code that runs on the GPU, lowering the device code to PTX, an intermediate instruction set, and then to SASS, the GPU's machine code. CPU-side compilers step down through intermediate stages the same way, and LLVM is the representative one. In fact cicc, the device code compiler inside nvcc, is built on LLVM, so the two stacks correspond layer by layer.

![The LLVM stack and the nvcc stack from the CUDA C post, layer by layer](images/cpu-gpu-stack.svg)

## The Pipeline

![Hand-drawn pipeline from C source through clang, opt, and llc to an executable](images/pipeline.svg?v=2)

- **Frontend (clang)**: `Test.c` → `Test.ll`. Parsing and type checking.
- **Middle-end (opt)**: `Test.ll` → `Test.ll'`. Passes rewriting the IR, which is optimization.
- **Backend (llc, as, ld)**: `Test.ll'` → `Test.s` → `Test.o` → `a.out`. Code generation and assembly.

One IR (intermediate representation, the language between source and machine code) flows between the three stages. Other compilers keep their IR as an in-memory object structure that cannot be written down, while LLVM IR is a text format with a published grammar, so it can be saved, edited by hand, and fed back to the compiler.

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

`-emit-llvm -S` tells clang to stop at human-readable IR instead of going all the way to machine code. func1 in the resulting `Test.ll` reads as follows.

```llvm
define i32 @func1() #0 {
  %1 = alloca i32, align 4
  store i32 4, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}
```

| IR | Meaning | C counterpart |
|---|---|---|
| `define i32 @func1() #0` | define function func1 returning i32 (32-bit int); `#0` refers to an attribute group | `int func1(void)` |
| `%1 = alloca i32, align 4` | reserve one i32 slot on the stack; its address is named `%1` | the slot for `int a` |
| `store i32 4, ptr %1` | store 4 at that address | `a = 4` |
| `%2 = load i32, ptr %1` | load from that address into `%2` | reading a in `return a` |
| `ret i32 %2` | return `%2` | `return` |

Four symbols are enough to read the body.

| Symbol | Meaning |
|---|---|
| `i32` | 32-bit integer type. i1, i8, i64 also exist |
| `%name` | local name. A virtual register, so there is no limit on how many |
| `@name` | global name. Functions live here |
| `;` | comment |

The `target triple` at the top and the `attributes` and `!` metadata at the bottom are environment configuration, not needed for decoding the body. Older LLVM (before 15) wrote typed pointers like `i32*` instead of `ptr`; opaque pointers unified this in LLVM 15. Different generation of syntax, same meaning.

## From IR to Assembly

```bash
llc Test.ll -o Test.s
```

`llc` is the backend. It lowers IR to the target CPU's assembly, which on Apple Silicon is ARM64. The mapping for func1:

| Test.ll (virtual) | Test.s (ARM, physical) | What happens |
|---|---|---|
| `%1 = alloca i32` | `sub sp, sp, #16` | reserve the stack frame; `%1` becomes the slot at `sp+12` |
| `store i32 4, ptr %1` | `mov w8, #4` → `str w8, [sp, #12]` | put 4 in a register, store it to the stack |
| `%2 = load i32, ptr %1` | `ldr w0, [sp, #12]` | load from the stack into w0; `%2` becomes w0 |
| `ret i32 %2` | `add sp, sp, #16` → `ret` | release the frame and return; w0 carries the return value |

Assigning virtual names (%N) to physical places (registers, stack slots) is the backend's job. An assembly file has three kinds of lines: lines starting with `.` are assembler directives and can be skipped, `name:` lines are labels, and only the indented lines are actual CPU instructions.

Change `store i32 4` to `store i32 9` in `Test.ll`, run `llc` again, and the output shows `mov w8, #9`. The IR text itself is the compiler input, so editing the IR alone changes the program without going through the frontend.

## From Assembly to an Executable

`Test.s` is still text. The assembler (as) turns that text into the bits a CPU executes, producing the object file `Test.o`. An object file is machine code but not runnable on its own: it is a fragment whose references to functions in other fragments or libraries are still unresolved. The linker (ld) collects the fragments, fills in those addresses, and joins them into an executable.

```bash
clang -c Test.s -o Test.o   # assemble
clang Test.o -o a.out       # link
```

Both steps can be invoked through the clang command, which calls as and ld internally.

## Watching Optimization as a Diff

The payoff of readable IR shows up when comparing before and after optimization.

```bash
clang -O1 -emit-llvm -S Test.c -o Test_O1.ll
```

At `-O0` (the default), func1 is four lines: alloca → store → load → ret. At `-O1` it is one.

```llvm
define noundef i32 @func1() local_unnamed_addr #0 {
  ret i32 4
}
```

The optimization passes proved that this function stores 4 to memory and immediately loads it back, so the answer is always 4, and erased the variable's existence. The C code did not change, the program went from four lines to one, and the whole event is captured in a text diff.

A pass is a unit of work in which the compiler sweeps over the entire program (IR) once, performing one predetermined analysis or transformation. `-O1` is a clang option that runs many passes in a fixed order; `opt` is the tool that runs a single pass of your choosing.

## optnone

mem2reg is the pass that promotes local variables from memory to registers, and `opt` can apply this single pass on its own.

```bash
opt -passes=mem2reg Test.ll -S -o Test_m2r.ll
```

Applied to IR emitted at `-O0`, however, the output comes back identical to the input. The cause is in the attributes. When clang emits IR at `-O0`, it stamps every function with `optnone`, a do-not-optimize marker, and `opt` respects it by skipping the function entirely. Attributes control whether passes run at all.

Emitting without the marker restores normal behavior.

```bash
clang -Xclang -disable-O0-optnone -emit-llvm -S Test.c -o Test_noopt.ll
opt -passes=mem2reg Test_noopt.ll -S
```

func1 folds to a single `ret i32 4`. This option is the standard way to emit IR for pass experiments.

## References

- [LLVM for Grad Students](https://www.cs.cornell.edu/~asampson/blog/llvm.html): the What is LLVM? / The Pieces / Understanding LLVM IR chapters.
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html): the official definition of IR syntax.
- [The Architecture of Open Source Applications: LLVM](https://aosabook.org/en/v1/llvm.html): Chris Lattner on the design background of LLVM.
