#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void show_attr(int dev, const char *name, cudaDeviceAttr attr) {
    int value = -1;
    cudaError_t e = cudaDeviceGetAttribute(&value, attr, dev);
    if (e == cudaSuccess) {
        std::printf("%s=%d\n", name, value);
    } else {
        std::printf("%s=UNAVAILABLE (%s)\n", name, cudaGetErrorString(e));
        cudaGetLastError();
    }
}

int main(int argc, char **argv) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count == 0) {
        std::fprintf(stderr, "cudaGetDeviceCount: %s, count=%d\n",
                     cudaGetErrorString(e), count);
        return 1;
    }

    std::printf("CUDART_VERSION=%d\n", CUDART_VERSION);
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    std::printf("runtimeVersion=%d driverVersion=%d deviceCount=%d\n",
                runtime_version, driver_version, count);

    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp p{};
        e = cudaGetDeviceProperties(&p, dev);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "cudaGetDeviceProperties(%d): %s\n",
                         dev, cudaGetErrorString(e));
            return 1;
        }

        std::printf("device=%d name=%s cc=%d.%d integrated=%d\n",
                    dev, p.name, p.major, p.minor, p.integrated);
    }

    int dev = argc == 2 ? std::atoi(argv[1]) : 0;
    if (dev < 0 || dev >= count) {
        std::fprintf(stderr, "invalid selected device: %d\n", dev);
        return 2;
    }

    cudaDeviceProp selected{};
    if (cudaGetDeviceProperties(&selected, dev) != cudaSuccess) return 3;
    std::printf("selectedDevice=%d name=%s cc=%d.%d integrated=%d\n",
                dev, selected.name, selected.major, selected.minor,
                selected.integrated);
    if (selected.integrated != 1 || selected.major != 8 || selected.minor != 7) {
        std::fprintf(stderr,
                     "selected device is not the expected Orin iGPU; stop here\n");
        return 4;
    }

    if (cudaSetDevice(dev) != cudaSuccess) return 5;
    show_attr(dev, "managedMemory", cudaDevAttrManagedMemory);
    show_attr(dev, "concurrentManagedAccess", cudaDevAttrConcurrentManagedAccess);
    show_attr(dev, "pageableMemoryAccess", cudaDevAttrPageableMemoryAccess);
    show_attr(dev, "usesHostPageTables",
              cudaDevAttrPageableMemoryAccessUsesHostPageTables);
    show_attr(dev, "directManagedMemAccessFromHost",
              cudaDevAttrDirectManagedMemAccessFromHost);
    show_attr(dev, "hostRegisterSupported",
              cudaDevAttrHostRegisterSupported);
}
