import time, ctypes, subprocess, json, pathlib

D = pathlib.Path(__file__).parent
N = 2_000_000
CALLS = 12

# ---------- workload: same LCG loop in all three ----------
def lcg_py(n):
    s = 1
    for _ in range(n):
        s = (s * 1664525 + 1013904223) % 4294967296
    return s

from numba import njit

@njit(cache=False)
def lcg_jit(n):
    s = 1
    for _ in range(n):
        s = (s * 1664525 + 1013904223) % 4294967296
    return s

C_SRC = r"""
#include <stdint.h>
uint32_t lcg(long n) {
    uint32_t s = 1;
    for (long i = 0; i < n; i++) s = s * 1664525u + 1013904223u;
    return s;
}
"""

# ---------- AOT: compile BEFORE any call, measure build time ----------
(D / "lcg.c").write_text(C_SRC)
t0 = time.perf_counter()
subprocess.run(["clang", "-O2", "-shared", "-o", str(D / "lcg.dylib"), str(D / "lcg.c")], check=True)
aot_build = time.perf_counter() - t0
lib = ctypes.CDLL(str(D / "lcg.dylib"))
lib.lcg.restype = ctypes.c_uint32
lib.lcg.argtypes = [ctypes.c_long]

def timed(fn):
    out = []
    for _ in range(CALLS):
        t = time.perf_counter()
        r = fn(N)
        out.append(time.perf_counter() - t)
    return r, out

r1, t_py  = timed(lcg_py)
r2, t_jit = timed(lcg_jit)          # call 1 includes numba's runtime compile
r3, t_aot = timed(lambda n: lib.lcg(n))

assert r1 == r2 == r3, (r1, r2, r3)
json.dump({"aot_build": aot_build, "py": t_py, "jit": t_jit, "aot": t_aot},
          open(D / "result.json", "w"))
print("same answer:", r1)
print("AOT build (before run): %.3fs" % aot_build)
for name, ts in [("interpreter", t_py), ("JIT", t_jit), ("AOT", t_aot)]:
    print("%-11s call1=%.4fs  call2=%.6fs  median(2..)=%.6fs" %
          (name, ts[0], ts[1], sorted(ts[1:])[len(ts[1:]) // 2]))
