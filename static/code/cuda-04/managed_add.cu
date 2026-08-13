#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::fprintf(stderr, "%s:%d: %s\n",                   \
                     __FILE__, __LINE__,                         \
                     cudaGetErrorString(err));                   \
        return 1;                                                \
    }                                                            \
} while (0)

__global__ void add_one(int *x) {
    *x += 1;
}

int main() {
    int *x = nullptr;
    CUDA_CHECK(cudaMallocManaged(&x, sizeof(*x)));

    *x = 41;
    std::printf("before kernel: %d\n", *x);

    add_one<<<1, 1>>>(x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("after kernel:  %d\n", *x);
    CUDA_CHECK(cudaFree(x));
}
