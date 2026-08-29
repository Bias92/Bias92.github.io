// Measures a serial H2D -> kernel -> D2H run against a chunked, multi-stream run
// of the same work. Build: nvcc -O3 -arch=sm_80 -o overlap_bench overlap_bench.cu
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CK(call) do {                                                      \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
        std::printf("CUDA error %s at line %d\n",                          \
                    cudaGetErrorString(err_), __LINE__);                   \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

__global__ void transform(const float *x, float *y, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        y[i] = x[i] * 2.0f;
    }
}

constexpr size_t N             = 1ULL << 24;   // 16,777,216 elements
constexpr size_t chunkElements = 1ULL << 20;   // 1,048,576 elements
constexpr int    streamCount   = 4;
constexpr int    block         = 256;
constexpr int    WARMUP        = 5;
constexpr int    REPEAT        = 30;

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

static float run_serial(float *h_x, float *h_y, float *d_x, float *d_y, size_t bytes) {
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    const int grid = static_cast<int>((N + block - 1) / block);

    CK(cudaEventRecord(start));
    CK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    transform<<<grid, block>>>(d_x, d_y, N);
    CK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    return ms;
}

static float run_streamed(float *h_x, float *h_y, float *d_x, float *d_y,
                          cudaStream_t *streams) {
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    CK(cudaEventRecord(start));
    for (size_t chunk = 0, offset = 0; offset < N;
         ++chunk, offset += chunkElements) {
        const size_t count = std::min(chunkElements, N - offset);
        const size_t chunkBytes = count * sizeof(float);
        cudaStream_t stream = streams[chunk % streamCount];

        CK(cudaMemcpyAsync(d_x + offset, h_x + offset, chunkBytes,
                           cudaMemcpyHostToDevice, stream));
        const int grid = static_cast<int>((count + block - 1) / block);
        transform<<<grid, block, 0, stream>>>(d_x + offset, d_y + offset, count);
        CK(cudaMemcpyAsync(h_y + offset, d_y + offset, chunkBytes,
                           cudaMemcpyDeviceToHost, stream));
    }
    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));
    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    return ms;
}

static bool verify(const float *h_x, const float *h_y) {
    for (size_t i = 0; i < N; i += 9973) {
        if (std::fabs(h_y[i] - h_x[i] * 2.0f) > 1e-5f) {
            std::printf("verify failed at %zu: got %f, want %f\n",
                        i, h_y[i], h_x[i] * 2.0f);
            return false;
        }
    }
    return true;
}

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("gpu=%s  asyncEngineCount=%d  concurrentKernels=%d\n",
                prop.name, prop.asyncEngineCount, prop.concurrentKernels);
    std::printf("N=%zu  chunkElements=%zu  chunks=%zu  streams=%d  repeat=%d\n",
                N, chunkElements, (N + chunkElements - 1) / chunkElements,
                streamCount, REPEAT);

    const size_t bytes = N * sizeof(float);
    float *h_x = nullptr, *h_y = nullptr, *d_x = nullptr, *d_y = nullptr;
    CK(cudaHostAlloc(&h_x, bytes, cudaHostAllocDefault));
    CK(cudaHostAlloc(&h_y, bytes, cudaHostAllocDefault));
    CK(cudaMalloc(&d_x, bytes));
    CK(cudaMalloc(&d_y, bytes));
    for (size_t i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i % 1000);
    }

    cudaStream_t streams[streamCount];
    for (int i = 0; i < streamCount; ++i) {
        CK(cudaStreamCreate(&streams[i]));
    }

    for (int i = 0; i < WARMUP; ++i) {
        run_serial(h_x, h_y, d_x, d_y, bytes);
        run_streamed(h_x, h_y, d_x, d_y, streams);
    }

    std::vector<float> ser, str;
    for (int i = 0; i < REPEAT; ++i) {
        std::fill(h_y, h_y + N, 0.0f);
        ser.push_back(run_serial(h_x, h_y, d_x, d_y, bytes));
        if (!verify(h_x, h_y)) return 1;

        std::fill(h_y, h_y + N, 0.0f);
        str.push_back(run_streamed(h_x, h_y, d_x, d_y, streams));
        if (!verify(h_x, h_y)) return 1;
    }

    const float ms_ser = median(ser);
    const float ms_str = median(str);
    std::printf("serial  median = %.3f ms  (min %.3f, max %.3f)\n",
                ms_ser, *std::min_element(ser.begin(), ser.end()),
                *std::max_element(ser.begin(), ser.end()));
    std::printf("stream  median = %.3f ms  (min %.3f, max %.3f)\n",
                ms_str, *std::min_element(str.begin(), str.end()),
                *std::max_element(str.begin(), str.end()));
    std::printf("speedup = %.2fx\n", ms_ser / ms_str);

    for (int i = 0; i < streamCount; ++i) {
        CK(cudaStreamDestroy(streams[i]));
    }
    CK(cudaFreeHost(h_x));
    CK(cudaFreeHost(h_y));
    CK(cudaFree(d_x));
    CK(cudaFree(d_y));
    return 0;
}
