#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void addKernel(__half* a, const __half* b, int64_t n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pairs = n / 2;
    if (index < pairs) {
        reinterpret_cast<__half2*>(a)[index] = __hadd2(
            reinterpret_cast<const __half2*>(a)[index],
            reinterpret_cast<const __half2*>(b)[index]);
    } else if (index == pairs && (n & 1)) {
        a[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

extern "C" void run_kernel(__half* a, const __half* b, int64_t n) {
    constexpr int blockSize = 256;
    const int64_t workItems = n / 2 + (n & 1);
    addKernel<<<(workItems + blockSize - 1) / blockSize, blockSize>>>(a, b, n);
}
