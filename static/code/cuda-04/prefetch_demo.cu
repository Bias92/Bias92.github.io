// prefetch_demo.cu
// Builds under CUDA 13.x (new cudaMemLocation form) and CUDA 12.2-12.9 (legacy int form).
//   nvcc -arch=sm_89 -O2 prefetch_demo.cu -o prefetch_demo.exe        (RTX 4060 Ti / AD106)
//   nvcc -arch=sm_87 -O2 prefetch_demo.cu -o prefetch_demo            (Jetson AGX Orin)
// MSVC host: nvcc -arch=sm_89 -O2 -Xcompiler "/std:c++17" prefetch_demo.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x)                                                                        \
    do {                                                                                \
        cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "%s:%d  %s -> %s (%d)\n", __FILE__, __LINE__, #x,      \
                         cudaGetErrorString(e_), (int)e_);                              \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Portable wrappers. CUDA 13.0 changed the parameter "int device" to
// "cudaMemLocation location" on cudaMemPrefetchAsync / cudaMemAdvise.
// CUDART_VERSION is 13000 for 13.0, 12080 for 12.8, etc.
// ---------------------------------------------------------------------------
#if CUDART_VERSION >= 13000
#define UM_HAS_MEMLOCATION 1
#else
#define UM_HAS_MEMLOCATION 0
#endif

static cudaError_t prefetchToDevice(const void* p, size_t n, int dev, cudaStream_t s)
{
#if UM_HAS_MEMLOCATION
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    return cudaMemPrefetchAsync(p, n, loc, /*flags=*/0u, s);
#else
    return cudaMemPrefetchAsync(p, n, dev, s);
#endif
}

static cudaError_t prefetchToHost(const void* p, size_t n, cudaStream_t s)
{
#if UM_HAS_MEMLOCATION
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeHost;   // id is ignored for this type
    loc.id   = 0;
    return cudaMemPrefetchAsync(p, n, loc, /*flags=*/0u, s);
#else
    return cudaMemPrefetchAsync(p, n, cudaCpuDeviceId, s);
#endif
}

static cudaError_t adviseDevice(const void* p, size_t n, cudaMemoryAdvise adv, int dev)
{
#if UM_HAS_MEMLOCATION
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    return cudaMemAdvise(p, n, adv, loc);
#else
    return cudaMemAdvise(p, n, adv, dev);
#endif
}

// ---------------------------------------------------------------------------

__global__ void saxpy(float* __restrict__ y, const float* __restrict__ x,
                      float a, size_t n)
{
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

static void dumpRangeAttrs(const void* p, size_t bytes)
{
    int   readMostly = -1, prefLoc = -1, lastPrefetch = -1;
    int   accessedBy[4] = {-1, -1, -1, -1};

    CHECK(cudaMemRangeGetAttribute(&readMostly, sizeof(int),
                                   cudaMemRangeAttributeReadMostly, p, bytes));
    CHECK(cudaMemRangeGetAttribute(&prefLoc, sizeof(int),
                                   cudaMemRangeAttributePreferredLocation, p, bytes));
    CHECK(cudaMemRangeGetAttribute(&lastPrefetch, sizeof(int),
                                   cudaMemRangeAttributeLastPrefetchLocation, p, bytes));
    CHECK(cudaMemRangeGetAttribute(accessedBy, sizeof(accessedBy),
                                   cudaMemRangeAttributeAccessedBy, p, bytes));

    // Batched form: same info in one call.
    cudaMemLocationType prefType = cudaMemLocationTypeInvalid;
    int                 prefId   = -1;
    void*  data[2]      = { &prefType, &prefId };
    size_t sizes[2]     = { sizeof(prefType), sizeof(prefId) };
    cudaMemRangeAttribute attrs[2] = { cudaMemRangeAttributePreferredLocationType,
                                       cudaMemRangeAttributePreferredLocationId };
    CHECK(cudaMemRangeGetAttributes(data, sizes, attrs, 2, p, bytes));

    std::printf("  ReadMostly=%d  PreferredLocation=%d  LastPrefetchLocation=%d\n",
                readMostly, prefLoc, lastPrefetch);
    std::printf("  AccessedBy[0..3]={%d,%d,%d,%d}  (cudaInvalidDeviceId=%d, cudaCpuDeviceId=%d)\n",
                accessedBy[0], accessedBy[1], accessedBy[2], accessedBy[3],
                (int)cudaInvalidDeviceId, (int)cudaCpuDeviceId);
    std::printf("  PreferredLocationType=%d  PreferredLocationId=%d\n",
                (int)prefType, prefId);
}

int main(int argc, char** argv)
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    int cma = 0, pma = 0, hpt = 0, managed = 0;
    CHECK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev));
    CHECK(cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, dev));
    CHECK(cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, dev));
    CHECK(cudaDeviceGetAttribute(&hpt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev));
    std::printf("CUDART_VERSION=%d  managedMemory=%d  concurrentManagedAccess=%d  "
                "pageableMemoryAccess=%d  usesHostPageTables=%d\n",
                CUDART_VERSION, managed, cma, pma, hpt);
    if (!cma) {
        std::printf("SKIP: concurrentManagedAccess==0.\n"
                    "      GPU-target prefetch and GPU-target memory advice in this "
                    "demo require a non-zero value.\n");
        return 0;
    }

    const size_t N     = (argc > 1) ? (size_t)std::atoll(argv[1]) : (64ull << 20); // 64M floats
    const size_t bytes = N * sizeof(float);
    std::printf("N=%zu  bytes=%.2f MiB per array\n", N, bytes / (1024.0 * 1024.0));

    float *x = nullptr, *y = nullptr;
    CHECK(cudaMallocManaged(&x, bytes));
    CHECK(cudaMallocManaged(&y, bytes));

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    const int threads = 256;
    const int blocks  = 1024;

    // ---------------- Pass A: fault-driven (no prefetch) ----------------
    for (size_t i = 0; i < N; ++i) { x[i] = 1.0f; y[i] = 2.0f; }   // CPU-resident now
    CHECK(cudaEventRecord(t0, s));
    saxpy<<<blocks, threads, 0, s>>>(y, x, 3.0f, N);
    CHECK(cudaEventRecord(t1, s));
    CHECK(cudaStreamSynchronize(s));
    float msFault = 0.f;
    CHECK(cudaEventElapsedTime(&msFault, t0, t1));

    // Push everything back to the host so pass B starts from the same state.
    CHECK(prefetchToHost(x, bytes, s));
    CHECK(prefetchToHost(y, bytes, s));
    CHECK(cudaStreamSynchronize(s));

    // ---------------- Pass B: explicit prefetch ----------------
    for (size_t i = 0; i < N; ++i) { x[i] = 1.0f; y[i] = 2.0f; }
    cudaError_t pe1 = prefetchToDevice(x, bytes, dev, s);
    cudaError_t pe2 = prefetchToDevice(y, bytes, dev, s);
    if (pe1 != cudaSuccess || pe2 != cudaSuccess) {
        std::printf("prefetch to device failed: %s / %s\n",
                    cudaGetErrorString(pe1), cudaGetErrorString(pe2));
    }
    CHECK(cudaEventRecord(t0, s));
    saxpy<<<blocks, threads, 0, s>>>(y, x, 3.0f, N);
    CHECK(cudaEventRecord(t1, s));
    CHECK(cudaStreamSynchronize(s));
    float msPrefetch = 0.f;
    CHECK(cudaEventElapsedTime(&msPrefetch, t0, t1));

    std::printf("kernel-only time:  fault-driven %.3f ms | prefetched %.3f ms  (%.2fx)\n",
                msFault, msPrefetch, msFault / msPrefetch);
    std::printf("  (kernel time alone hides the migration cost of pass A only if you "
                "measure wall time around the whole region)\n");

    // ---------------- Advice + attribute query ----------------
    std::printf("attributes after prefetch:\n");
    dumpRangeAttrs(x, bytes);

    CHECK(adviseDevice(x, bytes, cudaMemAdviseSetReadMostly, dev));
    CHECK(adviseDevice(x, bytes, cudaMemAdviseSetPreferredLocation, dev));
    CHECK(adviseDevice(x, bytes, cudaMemAdviseSetAccessedBy, dev));
    std::printf("attributes after SetReadMostly + SetPreferredLocation + SetAccessedBy:\n");
    dumpRangeAttrs(x, bytes);

    CHECK(adviseDevice(x, bytes, cudaMemAdviseUnsetReadMostly, dev));
    CHECK(adviseDevice(x, bytes, cudaMemAdviseUnsetPreferredLocation, dev));
    CHECK(adviseDevice(x, bytes, cudaMemAdviseUnsetAccessedBy, dev));

    // ---------------- verify ----------------
    CHECK(prefetchToHost(y, bytes, s));
    CHECK(cudaStreamSynchronize(s));
    double err = 0.0;
    for (size_t i = 0; i < N; i += 4096) err += (double)(y[i] - 5.0f);
    std::printf("checksum error = %g\n", err);

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHECK(cudaStreamDestroy(s));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
    return 0;
}
