#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < N; k++)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

template<int T>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < N / T; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * T + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * T + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < T; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

float median_ms(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }

int main(int argc, char** argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 2048;
    printf("N = %d\n", N);
    const size_t bytes = (size_t)N * N * sizeof(float);
    const double flops = 2.0 * N * (double)N * N;
    const int WARMUP = 5, REPS = 30;

    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    srand(7);
    for (size_t i = 0; i < (size_t)N * N; i++) { hA[i] = (float)rand()/RAND_MAX - 0.5f; hB[i] = (float)rand()/RAND_MAX - 0.5f; }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes)); CHECK(cudaMalloc(&dB, bytes)); CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));

    int failures = 0;
    auto bench = [&](const char* name, void (*launch)(const float*, const float*, float*, int)) {
        for (int w = 0; w < WARMUP; w++) launch(dA, dB, dC, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<float> ts;
        for (int it = 0; it < REPS; it++) {
            CHECK(cudaEventRecord(t0));
            launch(dA, dB, dC, N);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
            ts.push_back(ms);
        }
        float ms = median_ms(ts);
        // sample-validate 64 random entries on CPU
        CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        double maxrel = 0.0;
        for (int s = 0; s < 64; s++) {
            int i = rand() % N, j = rand() % N;
            double acc = 0.0;
            for (int k = 0; k < N; k++) acc += (double)hA[i*(size_t)N+k] * hB[k*(size_t)N+j];
            double rel = fabs(hC[i*(size_t)N+j] - acc) / (fabs(acc) + 1e-9);
            maxrel = std::max(maxrel, rel);
        }
        bool ok = maxrel < 1e-2;
        printf("%-14s %10.3f ms %10.1f GFLOP/s   maxrel %.2e  %s\n", name, ms, flops / ms / 1e6, maxrel, ok ? "OK" : "VALIDATION FAIL");
        if (!ok) failures++;
    };

    bench("naive", [](const float* A, const float* B, float* C, int N) {
        dim3 b(16, 16); dim3 g(N/16, N/16);
        matmul_naive<<<g, b>>>(A, B, C, N);
    });
    bench("tiled T=16", [](const float* A, const float* B, float* C, int N) {
        dim3 b(16, 16); dim3 g(N/16, N/16);
        matmul_tiled<16><<<g, b>>>(A, B, C, N);
    });
    bench("tiled T=32", [](const float* A, const float* B, float* C, int N) {
        dim3 b(32, 32); dim3 g(N/32, N/32);
        matmul_tiled<32><<<g, b>>>(A, B, C, N);
    });

    free(hA); free(hB); free(hC);
    return failures ? 1 : 0;
}
