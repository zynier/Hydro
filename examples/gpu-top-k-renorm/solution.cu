#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

__global__ void topKRenormKernel(
    const float* __restrict__ probs,
    const int32_t* __restrict__ top_k,
    float* __restrict__ output,
    int64_t num_classes) {
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    using BlockScan = cub::BlockScan<int, kBlockSize>;
    union TempStorage {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage scan;
    };
    __shared__ TempStorage temp;
    __shared__ unsigned int histogram[256];
    __shared__ unsigned int prefix;
    __shared__ unsigned int prefix_mask;
    __shared__ unsigned int threshold;
    __shared__ int remaining_rank;
    __shared__ float denominator;

    const int64_t row = blockIdx.x;
    const int64_t row_offset = row * num_classes;
    if (threadIdx.x == 0) {
        prefix = 0;
        prefix_mask = 0;
        remaining_rank = top_k[row];
    }
    __syncthreads();

#pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0;
        __syncthreads();

        const unsigned int current_prefix = prefix;
        const unsigned int current_mask = prefix_mask;
        for (int64_t column = threadIdx.x; column < num_classes; column += kBlockSize) {
            const unsigned int bits = __float_as_uint(probs[row_offset + column]);
            if ((bits & current_mask) == current_prefix) {
                atomicAdd(&histogram[(bits >> shift) & 255U], 1U);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int rank = remaining_rank;
            int selected_bucket = 255;
            for (; selected_bucket >= 0; --selected_bucket) {
                const int count = static_cast<int>(histogram[selected_bucket]);
                if (rank <= count) break;
                rank -= count;
            }
            prefix |= static_cast<unsigned int>(selected_bucket) << shift;
            prefix_mask |= 255U << shift;
            remaining_rank = rank;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) threshold = prefix;
    __syncthreads();

    float greater_sum = 0.0f;
    int equal_count = 0;
    for (int64_t column = threadIdx.x; column < num_classes; column += kBlockSize) {
        const float value = probs[row_offset + column];
        const unsigned int bits = __float_as_uint(value);
        if (bits > threshold) greater_sum += value;
        else if (bits == threshold) ++equal_count;
    }
    const float block_sum = BlockReduce(temp.reduce).Sum(greater_sum);
    if (threadIdx.x == 0) {
        denominator = block_sum + __uint_as_float(threshold) * remaining_rank;
    }
    __syncthreads();

    int equal_offset;
    BlockScan(temp.scan).ExclusiveSum(equal_count, equal_offset);
    __syncthreads();

    const float inverse_sum = 1.0f / denominator;
    int equal_seen = 0;
    for (int64_t column = threadIdx.x; column < num_classes; column += kBlockSize) {
        const float value = probs[row_offset + column];
        const unsigned int bits = __float_as_uint(value);
        bool selected = bits > threshold;
        if (bits == threshold) {
            selected = equal_offset + equal_seen < remaining_rank;
            ++equal_seen;
        }
        output[row_offset + column] = selected ? value * inverse_sum : 0.0f;
    }
}

}  // namespace

extern "C" void run_kernel(
    const float* probs,
    const int32_t* top_k,
    float* renorm_probs,
    int64_t batch_size,
    int64_t num_classes) {
    topKRenormKernel<<<batch_size, kBlockSize>>>(
        probs, top_k, renorm_probs, num_classes);
}
