#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

// naive: coalesced read, strided write
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    out[(size_t)x * n + y] = in[(size_t)y * n + x];
}

// shared tile without padding: 32-way bank conflict on the strided side
__global__ void transpose_tile32(const float* in, float* out, int n) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[(size_t)y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[(size_t)y * n + x] = tile[threadIdx.x][threadIdx.y];
}

// padded tile: conflict-free
__global__ void transpose_tile33(const float* in, float* out, int n) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[(size_t)y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[(size_t)y * n + x] = tile[threadIdx.x][threadIdx.y];
}

// XOR-swizzled tile: conflict-free without padding
__global__ void transpose_swizzle(const float* in, float* out, int n) {
    __shared__ float tile[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x ^ threadIdx.y] = in[(size_t)y * n + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    out[(size_t)y * n + x] = tile[threadIdx.x][threadIdx.y ^ threadIdx.x];
}

float median_ms(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

int main() {
    const int N = 4096;
    const size_t bytes = (size_t)N * N * sizeof(float);
    const int WARMUP = 10, REPS = 50;

    float* h = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)N * N; i++) h[i] = (float)(i % 1000);
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes)); CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    dim3 b(32, 32), g(N/32, N/32);

    struct { const char* name; void (*k)(const float*, float*, int); } vs[] = {
        {"naive (strided write)", transpose_naive},
        {"tile[32][32]         ", transpose_tile32},
        {"tile[32][33] padded  ", transpose_tile33},
        {"tile[32][32] swizzled", transpose_swizzle},
    };

    float* hout = (float*)malloc(bytes);
    int failures = 0;
    for (auto& v : vs) {
        for (int w = 0; w < WARMUP; w++) v.k<<<g, b>>>(d_in, d_out, N);
        CHECK(cudaGetLastError());
        std::vector<float> ts;
        for (int it = 0; it < REPS; it++) {
            CHECK(cudaEventRecord(t0));
            v.k<<<g, b>>>(d_in, d_out, N);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            ts.push_back(ms);
        }
        CHECK(cudaMemcpy(hout, d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int s = 0; s < 1000 && ok; s++) {
            int i = rand() % N, j = rand() % N;
            if (hout[(size_t)i * N + j] != h[(size_t)j * N + i]) ok = false;
        }
        float ms = median_ms(ts);
        printf("%-24s %8.3f ms %9.1f GB/s   %s\n", v.name, ms, 2.0 * bytes / ms / 1e6, ok ? "OK" : "WRONG");
        if (!ok) failures++;
    }

    free(h); free(hout);
    return failures ? 1 : 0;
}
