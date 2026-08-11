#include <cuda_runtime.h>
#include <cufft.h>
#include <dlfcn.h>

#include <cstdint>

namespace {

using PlanMany = cufftResult (*)(
    cufftHandle*, int, int*, int*, int, int, int*, int, int, cufftType, int);
using SetStream = cufftResult (*)(cufftHandle, cudaStream_t);
using ExecC2C = cufftResult (*)(cufftHandle, cufftComplex*, cufftComplex*, int);

struct FFTState {
    void* library = nullptr;
    PlanMany plan_many = nullptr;
    SetStream set_stream = nullptr;
    ExecC2C exec_c2c = nullptr;
    cufftHandle plan = 0;
    cufftComplex* buffer = nullptr;
    int64_t batch = 0;
    int64_t n = 0;
    bool ready = false;
};

FFTState state;

template <typename T>
T load_symbol(void* library, const char* name) {
    return reinterpret_cast<T>(dlsym(library, name));
}

bool initialize(int64_t batch, int64_t n) {
    if (state.ready) return state.batch == batch && state.n == n;
    state.library = dlopen("libcufft.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!state.library) state.library = dlopen("libcufft.so", RTLD_NOW | RTLD_LOCAL);
    if (!state.library) return false;

    state.plan_many = load_symbol<PlanMany>(state.library, "cufftPlanMany");
    state.set_stream = load_symbol<SetStream>(state.library, "cufftSetStream");
    state.exec_c2c = load_symbol<ExecC2C>(state.library, "cufftExecC2C");
    if (!state.plan_many || !state.set_stream || !state.exec_c2c) return false;

    const int64_t total = batch * n;
    if (cudaMalloc(&state.buffer, static_cast<size_t>(total) * sizeof(cufftComplex)) != cudaSuccess) {
        return false;
    }
    int transform_size = static_cast<int>(n);
    if (state.plan_many(
            &state.plan, 1, &transform_size,
            nullptr, 1, transform_size,
            nullptr, 1, transform_size,
            CUFFT_C2C, static_cast<int>(batch)) != CUFFT_SUCCESS) {
        return false;
    }
    if (state.set_stream(state.plan, nullptr) != CUFFT_SUCCESS) return false;
    state.batch = batch;
    state.n = n;
    state.ready = true;
    return true;
}

__global__ void pack_complex(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    cufftComplex* __restrict__ packed,
    int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) packed[index] = make_cuFloatComplex(real[index], imag[index]);
}

__global__ void unpack_complex(
    const cufftComplex* __restrict__ packed,
    float* __restrict__ real,
    float* __restrict__ imag,
    int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const cufftComplex value = packed[index];
        real[index] = value.x;
        imag[index] = value.y;
    }
}

}  // namespace

extern "C" void run_kernel(
    const float* x_real,
    const float* x_imag,
    float* y_real,
    float* y_imag,
    int64_t batch,
    int64_t n) {
    if (!initialize(batch, n)) return;
    constexpr int kBlockSize = 256;
    const int64_t total = batch * n;
    const int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    pack_complex<<<grid, kBlockSize>>>(x_real, x_imag, state.buffer, total);
    if (state.exec_c2c(state.plan, state.buffer, state.buffer, CUFFT_FORWARD) != CUFFT_SUCCESS) return;
    unpack_complex<<<grid, kBlockSize>>>(state.buffer, y_real, y_imag, total);
}
