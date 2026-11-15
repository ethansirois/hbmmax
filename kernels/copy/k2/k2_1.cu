#include "kernels/copy/linear_config.cuh"
constexpr int GROUP_SIZE = 512;

/*
 * We can use a template parameter because a template parameter
 * must be a known at compile-time 
 */
template <const int offset>
__device__ __forceinline__
float4 load(const float4* p) {
    float4 v;
    asm volatile("ld.global.cg.nc.v4.f32 {%0, %1, %2, %3}, [%4 + %5];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p), "n"(offset));
    return v;
}

template <const int offset>
__device__ __forceinline__
void store(float4* p, float4 v) {
    asm volatile("st.global.cg.v4.f32 [%0 + %1], {%2, %3, %4, %5};"
        :: "l"(p), "n"(offset), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory");
}

extern "C"
__global__ void 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    copy_tail<(SMALL_NUM_STEPS / GROUP_SIZE) * GROUP_SIZE * SMALL_STEP>(dA, dB);

    const int step = BLOCKS_PER_SM * SMALL_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    for (int i = 0; i + GROUP_SIZE <= SMALL_NUM_STEPS; i += GROUP_SIZE) {
        #pragma unroll
        for (int j = 0; j < GROUP_SIZE; ++j) {
            constexpr int offset = 0;
            const float4 value = load<offset>(&dA[j * step + idx]);
            store<offset>(&dB[j * step + idx], value);
        }
        dA += GROUP_SIZE * step;
        dB += GROUP_SIZE * step;
    }

}
