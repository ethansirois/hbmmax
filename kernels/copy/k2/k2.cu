#include "kernels/copy/linear_config.cuh"
#include "kernels/common/mem_ops.cuh"
constexpr int GROUP_SIZE = 256;

template <int... J>
__device__ __forceinline__
void copy_group(const float4* dA, float4* dB, std::integer_sequence<int, J...>) {
    (store<J * SMALL_BYTE_STEP>(dB, load<J * SMALL_BYTE_STEP>(dA)), ...);
}

extern "C"
__global__ void 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    copy_tail<(SMALL_NUM_STEPS / GROUP_SIZE) * GROUP_SIZE * SMALL_STEP>(dA, dB);

    const int step = BLOCKS_PER_SM * SMALL_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    dA += idx;
    dB += idx;

    for (int i = 0; i + GROUP_SIZE <= SMALL_NUM_STEPS; i += GROUP_SIZE) {
        copy_group(dA, dB, std::make_integer_sequence<int, GROUP_SIZE>{});
        dA += GROUP_SIZE * SMALL_STEP;
        dB += GROUP_SIZE * SMALL_STEP;
    }

}
