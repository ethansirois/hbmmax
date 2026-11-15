#include "kernels/copy/linear_config.cuh"
constexpr int GROUP_SIZE = 512;

extern "C"
__global__ void 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    copy_tail<(SMALL_NUM_STEPS / GROUP_SIZE) * GROUP_SIZE * SMALL_STEP>(dA, dB);

    const int step = BLOCKS_PER_SM * SMALL_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    // can support immediate of -0x800000 to 0x7fc000
    // so then we want to add 0x800000 << 1 in terms of addr
    // so then we want 2^23, so then group size should be
    // 2^23 / (step * 16) -> 2^23 / (4 * 256 * 16) -> 23 - (2+8+4) -> 23 - 14 = 9
    for (int i = 0; i + GROUP_SIZE <= SMALL_NUM_STEPS; i += GROUP_SIZE) {
        #pragma unroll
        for (int j = 0; j < GROUP_SIZE; ++j) {
            dB[j * step + idx] = dA[j * step + idx];
        }
        dA += GROUP_SIZE * step;
        dB += GROUP_SIZE * step;
    }

}
