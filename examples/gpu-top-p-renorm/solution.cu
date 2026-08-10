#include <cuda_runtime.h>
#include <stdint.h>

namespace {

template <int BlockThreads, int RadixBits>
struct SharedStorage {
    static constexpr int kWarps = BlockThreads / 32;
    static constexpr int kBins = 1 << RadixBits;
    float bin_sums[kWarps][kBins];
    uint32_t prefix;
    float accepted_sum;
};

template <int BlockThreads, int RadixBits>
__global__ void top_p_kernel(
    const float* __restrict__ probs,
    const float* __restrict__ top_p,
    float* __restrict__ renorm_probs,
    int64_t batch_size,
    int64_t num_classes
) {
    if (blockIdx.x >= static_cast<unsigned int>(batch_size)) return;
    using Storage = SharedStorage<BlockThreads, RadixBits>;
    __shared__ Storage shared;
    constexpr int kWarps = Storage::kWarps;
    constexpr int kBins = Storage::kBins;
    constexpr int kPasses = 32 / RadixBits;
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * num_classes;

    if (thread == 0) {
        shared.prefix = 0;
        shared.accepted_sum = 0.0f;
    }
    __syncthreads();

    // Positive IEEE-754 floats have the same unsigned-integer and numeric
    // ordering. Select the first excluded value one radix digit at a time.
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        for (int index = thread; index < kWarps * kBins; index += BlockThreads) {
            reinterpret_cast<float*>(shared.bin_sums)[index] = 0.0f;
        }
        __syncthreads();

        const int shift = 32 - RadixBits * (pass + 1);
        const uint32_t mask = pass == 0 ? 0u : 0xffffffffu << (shift + RadixBits);
        const uint32_t prefix = shared.prefix;
        for (int64_t index = thread; index < num_classes; index += BlockThreads) {
            const float value = probs[row_offset + index];
            const uint32_t bits = __float_as_uint(value);
            if ((bits & mask) == prefix) {
                atomicAdd(&shared.bin_sums[warp][(bits >> shift) & (kBins - 1)], value);
            }
        }
        __syncthreads();

        for (int bin = thread; bin < kBins; bin += BlockThreads) {
            float sum = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kWarps; ++source_warp) {
                sum += shared.bin_sums[source_warp][bin];
            }
            shared.bin_sums[0][bin] = sum;
        }
        __syncthreads();

        if (thread == 0) {
            float accepted = shared.accepted_sum;
            const float threshold = top_p[blockIdx.x];
            for (int bin = kBins - 1; bin >= 0; --bin) {
                const float candidate = accepted + shared.bin_sums[0][bin];
                if (candidate <= threshold) {
                    accepted = candidate;
                } else {
                    shared.prefix = prefix | (static_cast<uint32_t>(bin) << shift);
                    break;
                }
            }
            shared.accepted_sum = accepted;
        }
        __syncthreads();
    }

    const uint32_t excluded = shared.prefix;
    float denominator = shared.accepted_sum;
    const bool force_maximum = denominator <= 0.0f;
    if (force_maximum) denominator = __uint_as_float(excluded);
    for (int64_t index = thread; index < num_classes; index += BlockThreads) {
        const float probability = probs[row_offset + index];
        const uint32_t bits = __float_as_uint(probability);
        const bool retained = force_maximum ? bits == excluded : bits > excluded;
        renorm_probs[row_offset + index] = retained ? probability / denominator : 0.0f;
    }
}

} // namespace

extern "C" void run_kernel(
    const float* probs,
    const float* top_p,
    float* renorm_probs,
    int64_t batch_size,
    int64_t num_classes
) {
    if (batch_size <= 0 || num_classes <= 0) return;
    const unsigned int blocks = static_cast<unsigned int>(batch_size);
    if (num_classes <= 2048) {
        top_p_kernel<128, 4><<<blocks, 128>>>(
            probs, top_p, renorm_probs, batch_size, num_classes
        );
    } else if (num_classes <= 4096) {
        top_p_kernel<256, 4><<<blocks, 256>>>(
            probs, top_p, renorm_probs, batch_size, num_classes
        );
    } else {
        top_p_kernel<256, 8><<<blocks, 256>>>(
            probs, top_p, renorm_probs, batch_size, num_classes
        );
    }
}
