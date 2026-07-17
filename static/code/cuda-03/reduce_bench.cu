#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

constexpr int BS = 256;

// v0: interleaved addressing with a warp-nonuniform modulo predicate
__global__ void reduce_v0(const float* in, float* out, int n) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = buf[0];
}

// v1: sequential addressing
__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = buf[0];
}

// v2: sequential addressing down to one warp, then warp shuffle
__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float x = buf[tid] + buf[tid + 32];
        for (int off = 16; off > 0; off >>= 1)
            x += __shfl_down_sync(0xffffffffu, x, off);
        if (tid == 0) out[blockIdx.x] = x;
    }
}

// v3: v2 block reduction + one atomicAdd per block (single kernel, single pass)
__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float x = buf[tid] + buf[tid + 32];
        for (int off = 16; off > 0; off >>= 1)
            x += __shfl_down_sync(0xffffffffu, x, off);
        if (tid == 0) atomicAdd(out, x);
    }
}

typedef void (*Kernel)(const float*, float*, int);

// run multi-pass reduction (v0/v1/v2 protocol): reduce until one value remains.
// returns pointer to the device scalar holding the result (no D2H inside).
const float* run_multipass(Kernel k, const float* d_in, float* d_a, float* d_b, int n) {
    int count = n;
    const float* src = d_in;
    float* dst = d_a;
    float* other = d_b;
    while (count > 1) {
        int blocks = (count + BS - 1) / BS;
        k<<<blocks, BS>>>(src, dst, count);
        count = blocks;
        src = dst;
        float* t = dst == d_a ? other : d_a;
        other = dst;
        dst = t;
    }
    CHECK(cudaGetLastError());
    return src;
}

float median_ms(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    const int N = 1 << 24;   // 16,777,216 floats = 64 MB
    const size_t bytes = (size_t)N * sizeof(float);
    const int WARMUP = 10, REPS = 50;

    float* h = (float*)malloc(bytes);
    srand(42);
    double ref = 0.0;
    for (int i = 0; i < N; i++) { h[i] = (float)rand() / RAND_MAX; ref += h[i]; }

    float *d_in, *d_a, *d_b, *d_scalar;
    CHECK(cudaMalloc(&d_in, bytes));
    int maxblocks = (N + BS - 1) / BS;
    CHECK(cudaMalloc(&d_a, maxblocks * sizeof(float)));
    CHECK(cudaMalloc(&d_b, ((maxblocks + BS - 1) / BS) * sizeof(float)));
    CHECK(cudaMalloc(&d_scalar, sizeof(float)));
    CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    struct { const char* name; Kernel k; } vs[] = {
        {"v0_interleaved", reduce_v0},
        {"v1_sequential", reduce_v1},
        {"v2_shuffle   ", reduce_v2},
    };

    int failures = 0;
    auto check_err = [&](double relerr) { if (relerr > 1e-5) failures++; };

    printf("%-16s %10s %10s %12s %s\n", "variant", "ms", "GB/s", "result", "relerr");
    for (auto& v : vs) {
        const float* d_r = nullptr;
        for (int w = 0; w < WARMUP; w++) d_r = run_multipass(v.k, d_in, d_a, d_b, N);
        std::vector<float> ts;
        for (int it = 0; it < REPS; it++) {
            CHECK(cudaEventRecord(t0));
            d_r = run_multipass(v.k, d_in, d_a, d_b, N);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            ts.push_back(ms);
        }
        float r; CHECK(cudaMemcpy(&r, d_r, sizeof(float), cudaMemcpyDeviceToHost));
        float ms = median_ms(ts);
        printf("%-16s %10.3f %10.1f %12.1f %.2e\n", v.name, ms, bytes / ms / 1e6, r, fabs(r - ref) / ref);
        check_err(fabs(r - ref) / ref);
    }

    // v3: single pass + atomic
    {
        float r = 0;
        for (int w = 0; w < WARMUP; w++) {
            CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
            reduce_v3<<<maxblocks, BS>>>(d_in, d_scalar, N);
        }
        std::vector<float> ts;
        for (int it = 0; it < REPS; it++) {
            CHECK(cudaMemset(d_scalar, 0, sizeof(float)));
            CHECK(cudaEventRecord(t0));
            reduce_v3<<<maxblocks, BS>>>(d_in, d_scalar, N);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            ts.push_back(ms);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(&r, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
        float ms = median_ms(ts);
        printf("%-16s %10.3f %10.1f %12.1f %.2e\n", "v3_atomic1/blk", ms, bytes / ms / 1e6, r, fabs(r - ref) / ref);
        check_err(fabs(r - ref) / ref);
    }

    // CUB DeviceReduce reference
    {
        void* d_tmp = nullptr; size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_in, d_scalar, N);
        CHECK(cudaMalloc(&d_tmp, tmp_bytes));
        float r = 0;
        for (int w = 0; w < WARMUP; w++) cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_in, d_scalar, N);
        std::vector<float> ts;
        for (int it = 0; it < REPS; it++) {
            CHECK(cudaEventRecord(t0));
            cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_in, d_scalar, N);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            ts.push_back(ms);
        }
        CHECK(cudaMemcpy(&r, d_scalar, sizeof(float), cudaMemcpyDeviceToHost));
        float ms = median_ms(ts);
        printf("%-16s %10.3f %10.1f %12.1f %.2e\n", "cub_device   ", ms, bytes / ms / 1e6, r, fabs(r - ref) / ref);
        check_err(fabs(r - ref) / ref);
    }

    free(h);
    return failures ? 1 : 0;
}
