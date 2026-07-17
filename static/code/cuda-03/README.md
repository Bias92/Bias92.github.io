# 03 post benchmarks (RTX 4060 Ti, CUDA 13.0)

Validation gates: every binary exits non-zero if result verification fails
(reduce: relerr > 1e-5 vs CPU double reference; gemm: 64-sample maxrel > 1e-2;
transpose: 1000-sample mismatch). Kernel launches are error-checked.

## Build (git bash)

```
CCBIN="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"
nvcc -O3 -arch=sm_89 -ccbin "$CCBIN" -Xcompiler -wd4819 -o transpose_bench.exe transpose_bench.cu
nvcc -O3 -arch=sm_89 -ccbin "$CCBIN" -Xcompiler -wd4819 -o gemm_bench.exe gemm_bench.cu
nvcc -O3 -arch=sm_89 -std=c++17 -ccbin "$CCBIN" -Xcompiler -wd4819 -o reduce_bench.exe reduce_bench.cu
```

## Timing tables (cudaEvent, median)

```
./reduce_bench.exe          # reduction table: v0/v1/v2/v3/CUB
./gemm_bench.exe            # GEMM N=2048: naive / tiled16 / tiled32
./gemm_bench.exe 4096       # GEMM N=4096
./transpose_bench.exe       # transpose table: naive / tile32 / tile33 / swizzle
```

Session-to-session clock state moves absolute numbers by a few percent;
the post's conclusions are ratios between variants.

## Nsight Compute counters used in the post

```
NCU="/c/Program Files/NVIDIA Corporation/Nsight Compute 2025.3.1/target/windows-desktop-win7-x64/ncu.exe"
```

Reduction (occupancy, predicate-agnostic thread-instruction average, duration;
run per kernel):

```
for k in reduce_v0 reduce_v1 reduce_v2; do
  "$NCU" -k $k --launch-skip 0 --launch-count 1 --metrics \
gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__average_thread_inst_executed_per_inst_executed.ratio,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
./reduce_bench.exe
done
```

Transpose (bank conflicts, store sectors/requests, DRAM; run per kernel):

```
for k in transpose_naive transpose_tile32 transpose_tile33 transpose_swizzle; do
  "$NCU" -k $k --launch-skip 0 --launch-count 1 --metrics \
gpu__time_duration.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,dram__bytes.sum \
./transpose_bench.exe
done
```

GEMM N=2048. The tiled kernels are templates, so select them by launch order
on the "tiled" regex: skip 0 = first tiled16 launch, skip 35 (5 warmup + 30
reps) = first tiled32 launch.

```
M="gpu__time_duration.sum,dram__bytes.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block_static"
"$NCU" -k matmul_naive   --launch-skip 0  --launch-count 1 --metrics $M ./gemm_bench.exe
"$NCU" -k "regex:tiled"  --launch-skip 0  --launch-count 1 --metrics $M ./gemm_bench.exe
"$NCU" -k "regex:tiled"  --launch-skip 35 --launch-count 1 --metrics $M ./gemm_bench.exe
```

GEMM N=4096 (DRAM and sector traffic across the L2 boundary; the same tiled
launch-skip mapping applies):

```
M4096="dram__bytes.sum,gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
"$NCU" -k matmul_naive  --launch-skip 0  --launch-count 1 --metrics $M4096 ./gemm_bench.exe 4096
"$NCU" -k "regex:tiled" --launch-skip 0  --launch-count 1 --metrics $M4096 ./gemm_bench.exe 4096
"$NCU" -k "regex:tiled" --launch-skip 35 --launch-count 1 --metrics $M4096 ./gemm_bench.exe 4096
```

Known instability: dram__bytes.sum for all three GEMM variants at N=2048
swings across fresh profiler runs (roughly 75-145 MB for naive, 70-150 MB for
tiled16, and 60-100 MB for tiled32 in the observed sessions). The A+B input
working set alone sits at the 32 MB L2 boundary; C writes, cache state, and
profiler replay change how much leaks to DRAM. The sector counters stay exact.
The post reports DRAM as ranges and bases the traffic argument on N=4096.

Reduction v0 modulo check in SASS:

```
cuobjdump --dump-sass reduce_bench.exe
```

In `_Z9reduce_v0PKfPfi`, the changing `tid % (2*s)` divisor lowers to the
`IABS -> I2F -> MUFU.RCP -> F2I -> IMAD.HI` remainder sequence rather than a
single constant bit mask.

The source condition is warp-nonuniform, but that alone does not prove that the
short body remained an actual divergent branch after compilation. Inspect SASS
and the Nsight Compute Source-page `Divergent Branches` counter
(`smsp__branch_targets_threads_divergent`) to distinguish a branch from
predicated instructions.
