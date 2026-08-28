---
title: "00 LLVM IR and the Compilation Pipeline: From C Code to Machine Code"
date: 2026-08-25
draft: false
math: true
tags: ["LLVM", "Compiler", "IR", "clang", "Assembly"]
categories: ["Compiler"]
series: ["LLVM"]
summary: "Starting from the difference between interpreters, JITs, and AOT compilers, emit LLVM IR from C with clang, decode it, and follow the compilation pipeline down through assembly to an executable. Observe how optimization passes rewrite the IR via an -O0 vs -O1 diff, and record the optnone attribute that makes opt silently do nothing."
---

A compiler is a program that translates the source code humans read and write into machine code so the CPU can run it. A CPU can natively execute only machine code, so code written in a language like C must go through this translation once before it runs.

Depending on when the translation happens, the ways to run a program split broadly into three.

| Method | When it translates | What persists between runs | Examples[^ex] |
|---|---|---|---|
| Interpreter | never; re-interprets the source on every run | nothing | CPython |
| JIT (just-in-time) | during execution, only for hot code | machine code kept in memory | Java JVM, V8 in browsers |
| AOT (ahead-of-time) | everything, before the program runs | an executable file | clang, GCC |

The difference between the three methods shows up as time once the same function is called repeatedly. The graph below shows per-call times for the same integer loop, called twelve times each as CPython (interpreter), numba[^numba] (JIT), and a C library compiled ahead of time with `clang -O2` (AOT).

![Per-call time of the same function under an interpreter, a JIT, and an AOT build](images/jit_aot_interp.gif?v=2#medium)

The interpreter repeats the same interpretation on every call and pays the same cost each time. The JIT pays its compilation cost on the first call (warmup[^warmup]) and reuses the translated machine code afterwards, while the AOT build pays that cost before the program runs and is fast from the first call.

LLVM[^llvm] is an AOT compiler for native languages (languages that compile to machine code the CPU executes directly, with no virtual machine[^vm]) like C and C++.

The structure of stepping down through an intermediate language instead of translating source straight to machine code already appeared with nvcc in [CUDA C Basics]({{< relref "/posts/cuda-c-basics" >}}#the-nvcc-compilation-pipeline). nvcc, NVIDIA's CUDA compiler, splits a `.cu` file into host code that runs on the CPU and device code that runs on the GPU, lowering the device code to PTX[^ptx], an intermediate instruction set, and then to SASS, the GPU's machine code. CPU-side compilers step down through intermediate stages the same way, and LLVM is the representative one. In fact cicc, the device code compiler inside nvcc, is built on LLVM, so the two stacks correspond layer by layer.

![The LLVM stack and the nvcc stack from the CUDA C post, layer by layer](images/cpu-gpu-stack.svg)

## The Pipeline

![Hand-drawn pipeline from C source through clang, opt, and llc to an executable](images/pipeline.svg?v=2)

- **Frontend (clang)**: `Test.c` → `Test.ll`. Parsing[^parse] and type checking.
- **Middle-end (opt)**: `Test.ll` → `Test.ll'`. Passes rewriting the IR, which is optimization.
- **Backend (llc, as, ld)**: `Test.ll'` → `Test.s` → `Test.o` → `a.out`. Code generation and assembly.

A single IR (intermediate representation, the language between source and machine code) travels through all three stages: frontend, middle-end, and backend. Other compilers hold their IR only as in-memory objects, so it cannot be pulled out into a file. LLVM IR is text with a published grammar, so you can save it to a file, edit it by hand, and feed it back to the compiler.

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
| `%1 = alloca i32, align 4` | reserve one i32 slot on the stack[^stack]; its address is named `%1`[^align] | the slot for `int a` |
| `store i32 4, ptr %1` | store 4 at that address | `a = 4` |
| `%2 = load i32, ptr %1` | load from that address into `%2` | reading a in `return a` |
| `ret i32 %2` | return `%2` | `return` |

These are the 4 symbols that appear in the body.

| Symbol | Meaning |
|---|---|
| `i32` | 32-bit integer type. i1, i8, i64 also exist |
| `%name` | local name. A virtual register[^reg], so there is no limit on how many |
| `@name` | global name. Functions live here |
| `;` | comment |

The target triple[^triple] at the top, the attributes[^attr] at the bottom, and the `!` metadata are environment configuration, not needed for decoding the body.[^ptr14]

## From IR to Assembly

```bash
llc Test.ll -o Test.s
```

`llc` is the backend tool that lowers IR to the assembly[^asm] of the CPU named by the target triple[^triple], so the output assembly differs per target CPU: targeting x86-64 produces x86-64 assembly, ARM64 produces ARM64 assembly, RISC-V produces RISC-V assembly, and an option such as `llc -mtriple=x86_64-pc-linux-gnu Test.ll` switches the target. One IR fanning out to per-target backends is exactly the role of IR described in the pipeline section.

The following table maps func1 in the ARM64-target output.[^armregs]

| Test.ll (virtual) | Test.s (ARM, physical) | What happens |
|---|---|---|
| `%1 = alloca i32` | `sub sp, sp, #16` | reserve the stack frame; `%1` becomes the slot at `sp+12` |
| `store i32 4, ptr %1` | `mov w8, #4` → `str w8, [sp, #12]` | put 4 in a register, store it to the stack |
| `%2 = load i32, ptr %1` | `ldr w0, [sp, #12]` | load from the stack into w0; `%2` becomes w0 |
| `ret i32 %2` | `add sp, sp, #16` → `ret` | release the frame and return; w0 carries the return value |

The backend assigns virtual names (%N) to physical places (registers, stack slots). An assembly file has 3 kinds of lines.

| Line shape | What it is | When reading |
|---|---|---|
| starts with `.` | assembler directive[^directive] | skip |
| `name:` | label[^label] | position marker |
| indented | CPU instruction | what to read |

Change `store i32 4` to `store i32 9` in `Test.ll`, run `llc` again, and the output shows `mov w8, #9`. The IR text itself is the compiler input, so editing the IR alone changes the program without going through the frontend.

## From Assembly to an Executable

`Test.s` is a text file holding instructions like `mov w8, #4` as characters. A CPU, however, is a circuit that acts only when the fixed bit pattern of an instruction arrives, so it cannot execute character data like 'm', 'o', 'v' as instructions. The assembler (as) therefore turns the text notation into instruction bit patterns, and the result is the object file `Test.o`.

A program is usually built from several source files, each becoming one object file, plus libraries (bundles of precompiled object files holding common functions such as printf). An object file is machine code but not runnable on its own, because the addresses of functions it calls in other object files or libraries are still unresolved. The linker (ld) collects those object files and libraries, fills in the missing addresses, and joins them into an executable. This process is called linking.

```bash
clang -c Test.s -o Test.o   # assemble
clang Test.o -o a.out       # link
```

Both steps can be invoked through the clang command, which calls as and ld internally.

## Passes and Optimization Levels

A pass is a small program built into LLVM that sweeps over the entire IR once, performing one predetermined analysis or transformation. Each pass owns exactly one transformation.

| pass | The one thing it does |
|---|---|
| mem2reg | replaces the memory round trip of local variables (alloca, store, load) with direct value flow |
| instcombine | rewrites instruction combinations into shorter ones with the same result |
| simplifycfg | deletes unreachable blocks[^block] and simplifies branches |
| dce | deletes instructions that do not affect the result (dead code elimination) |
| licm | moves computations whose value is the same on every iteration out of the loop (loop invariant code motion) |

Running passes splits into two ways. With `opt` you pick and run a single pass of your choosing, and with clang's `-O1` `-O2` `-O3` options a predefined list of passes runs in order. And the three options' lists nest inside one another.[^1]

$$ O_0(0) \subset O_1(98) \subset O_2(115) \subset O_3(118) $$

The price of a higher level is compile time, and the passes `-O3` adds trade code size for speed, which is why release builds usually stop at `-O2`.

The following command prints the actual contents of a list. Every pass in the table above is inside it.

```bash
opt -passes='default<O1>' -print-pipeline-passes Test.ll -S -o /dev/null
```

## Watching Optimization as a Diff

The payoff of readable IR shows up when comparing before and after optimization.

```bash
clang -O1 -emit-llvm -S Test.c -o Test_O1.ll
```

Comparing `Test.ll` emitted at `-O0` (the default, left) with `Test_O1.ll` emitted at `-O1` (right) in the VS Code diff editor[^diffed] gives the following.

![Test.ll and Test_O1.ll compared in the VS Code diff editor](images/opt-diff.png?v=3)

Mapping the body of func1 line by line:

| Left, `-O0` (before) | Right, `-O1` (after) | Change |
|---|---|---|
| `%1 = alloca i32` | gone | the stack slot for variable a is removed |
| `store i32 4, ptr %1` | gone | the instruction writing 4 to memory is removed |
| `%2 = load i32, ptr %1` | gone | the instruction reading it back from memory is removed |
| `ret i32 %2` | `ret i32 4` | the constant 4 is returned directly instead of a value read from the variable |

The optimization passes proved that this function stores 4 to memory and immediately loads it back, so the answer is always 4, and erased the variable's existence. The C code did not change, the program went from four lines to one, and the whole event is captured in a text diff.

## optnone

optnone is a do-not-optimize marker that clang attaches to the attributes[^attr] of every function when it emits IR at `-O0`. The marker shows itself as follows when experimenting with passes.

Apply mem2reg from the previous section on its own with `opt`.

```bash
opt -passes=mem2reg Test.ll -S -o Test_m2r.ll
```

The output comes back identical to the input. Before processing a function, `opt` reads its attributes and skips the function when it sees optnone. Attributes control whether passes run at all.

Emitting without the marker restores normal behavior.[^xclang]

```bash
clang -Xclang -disable-O0-optnone -emit-llvm -S Test.c -o Test_noopt.ll
opt -passes=mem2reg Test_noopt.ll -S
```

func1 folds to a single `ret i32 4`. This option is the standard way to emit IR for pass experiments.

## References

- [LLVM for Grad Students](https://www.cs.cornell.edu/~asampson/blog/llvm.html): the What is LLVM? / The Pieces / Understanding LLVM IR chapters.
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html): the official definition of IR syntax.
- [The Architecture of Open Source Applications: LLVM](https://aosabook.org/en/v1/llvm.html): Chris Lattner on the design background of LLVM.

[^1]: The numbers in parentheses count the passes in the printed pipeline of LLVM 22 and vary by version.

[^ex]: CPython is the standard Python interpreter, the Java JVM runs Java, V8 is the JavaScript engine in Chrome, and GCC is another C/C++ compiler in the same role as clang.
[^numba]: A JIT library that compiles Python functions to machine code during execution.
[^warmup]: The name for the slow early calls in which a JIT pays its translation cost.
[^vm]: A program that executes intermediate code instead of machine code; the Java JVM is the canonical example.
[^parse]: Breaking source characters down by grammar rules into a structured tree.
[^stack]: The stack is the memory region where a function's local data accumulates; it grows on function entry and shrinks on return.
[^align]: align 4 tells the compiler to place the address at a multiple of four bytes.
[^reg]: A register is one of a CPU's few fast internal storage slots, fixed in number. A virtual register is a name the IR can mint without limit; the backend assigns it to a physical register.
[^asm]: Assembly is a notation that writes CPU instructions as human-readable text. Its relation to machine-code bits is covered in the next section.
[^armregs]: In the table, sp is the register holding the current top address of the stack (the stack pointer); w8 and w0 are ARM general-purpose registers, and w0 carries the return value.
[^directive]: A directive is an instruction to the assembler, not a CPU instruction.
[^label]: A label is a name attached to a position in the code; branches and calls refer to positions by these names.
[^diffed]: A view showing the differences between two files side by side: deleted lines in red on the left, added lines in green on the right.
[^xclang]: In the command, -Xclang makes clang pass the following option straight to its internal frontend, and -disable-O0-optnone is the option that suppresses the marker.
[^block]: A block (basic block) is a run of instructions executed strictly top to bottom with no branches inside.

[^attr]: attributes is the list collecting the properties applied to a function: the `attributes #0 = {...}` line near the bottom of the IR file, which the `#0` on a function definition points to.

[^llvm]: LLVM began as an acronym for Low Level Virtual Machine, but the project has long outgrown virtual machines and the name now stands on its own.

[^ptx]: Short for Parallel Thread Execution.

[^ptr14]: Older LLVM (before 15) wrote typed pointers like `i32*` instead of `ptr`. LLVM 15 unified this as opaque pointers (a notation that does not spell out the pointed-to type); different generation of syntax, same meaning.

[^triple]: The target triple is the target specification written at the top of the IR file, made of three parts as the name says (CPU-vendor-OS). Example: `arm64-apple-macosx15.0.0`.
