#include "kernels/copy/async_config.cuh"
#include <cuda_pipeline.h>
#include "kernels/common/mem_ops.cuh"
extern "C"
__global__ void 
__launch_bounds__(THREADS_PER_SM, BLOCKS_PER_SM)
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    copy_tail<PIPELINE_BULK_VECTORS>(dA, dB);
    if constexpr (PIPELINE_BULK_VECTORS == 0) return;

    extern __shared__ char raw_smem[];
    float4* smem = reinterpret_cast<float4*>(raw_smem);

    const int step = BLOCKS_PER_SM * SMALL_SMs_USED * THREADS_PER_SM;
    const int idx = blockIdx.x * THREADS_PER_SM + threadIdx.x;
    smem += threadIdx.x;

    float4 regs[MAX_REGS];

    dA += idx;
    dB += idx;

    // prologue: load 60 g -> r0, copy 40 g -> s0
    #pragma unroll
    for (int i = 0; i < MAX_REGS; ++i) {
        regs[i] = dA[i * step];
    }

    #pragma unroll
    for (int i = 0; i < SMEM_LEN; ++i) {
        __pipeline_memcpy_async(&smem[i * THREADS_PER_SM], &dA[(MAX_REGS + i) * SMALL_STEP], sizeof(float4));
    }
    __pipeline_commit();

    //// dA loaded MAX_REGS + SMEM_LEN
    //// dA = original_dA + MAX_REGS

    // store 60 r0 -> g
    store_group<SMALL_BYTE_STEP, MAX_REGS>(dB, regs);

    //// dA loaded MAX_REGS + SMEM_LEN
    //// dA = original_dA + MAX_REGS
    //// dB stored MAX_REGS 
    //// dB = original_dB + MAX_REGS

    // load 20 g -> r1  (regs[SMEM_LEN..MAX_REGS))
    #pragma unroll
    for (int i = SMEM_LEN; i < MAX_REGS; ++i) {
        regs[i] = dA[(MAX_REGS + i) * step];
    }

    dA += 2 * MAX_REGS * SMALL_STEP;
    dB += MAX_REGS * SMALL_STEP;
    //// dA loaded 2*MAX_REGS
    //// dA = original_dA + 2*MAX_REGS
    //// dB stored MAX_REGS
    //// dB = original_dB + MAX_REGS

    // main loop
    for (int it = 0; it < NUM_MAIN_LOOPS; ++it) {
        __pipeline_wait_prior(0);

        #pragma unroll
        for (int i = 0; i < SMEM_LEN; ++i) {
            // s_load 40 s -> r_big
            regs[i] = smem[i * THREADS_PER_SM];
            __pipeline_memcpy_async(&smem[i * THREADS_PER_SM], &dA[(i + SMEM_LEN) * SMALL_STEP], sizeof(float4));
        }
        // store 40 r_big -> g
        store_group<SMALL_BYTE_STEP, SMEM_LEN>(dB, regs);
        
        // load 40 g -> r_big
        for (int i = 0; i < SMEM_LEN; ++i) {
            regs[i] = dA[i * step];
        }

        //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SHARED) + [0..2*SMEM_LEN]
        //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
        //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
        //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN)

        // store 20 r_small -> g
        store_group<SMALL_BYTE_STEP, SMALL_REGS, SMEM_LEN>(dB, regs + SMEM_LEN);
        // load 20 g -> r_small
        #pragma unroll
        for (int i = SMEM_LEN; i < MAX_REGS; ++i) {
            // load 20 g -> r_small
            regs[i] = dA[(i + SMEM_LEN) * step];
        }
        //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS + SMEM_LEN
        //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
        //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS
        //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS

        // store 40 r_big -> g
        store_group<SMALL_BYTE_STEP, SMEM_LEN, MAX_REGS>(dB, regs);
        
        __pipeline_commit();
        dA += (MAX_REGS + SMEM_LEN) * SMALL_STEP;
        dB += (MAX_REGS + SMEM_LEN) * SMALL_STEP;

        //// dA loaded 2*MAX_REGS + (it + 1)*(MAX_REGS + SMEM_LEN)
        //// dA = original_dA + 2*MAX_REGS + (it + 1)*(MAX_REGS + SMEM_LEN)
        //// dB stored MAX_REGS + (it + 1)*(MAX_REGS + SMEM_LEN)
        //// dB = original_dB + MAX_REGS + (it + 1)*(MAX_REGS + SMEM_LEN)
    }
    
    // epilogue: drain in-flight smem + 20 regs
    __pipeline_wait_prior(0);

    #pragma unroll
    for (int i = 0; i < SMEM_LEN; ++i) {
        // s_load 40 s -> r_big
        regs[i] = smem[i * THREADS_PER_SM];
    }
    // store 40 r_big -> g
    store_group<SMALL_BYTE_STEP, SMEM_LEN>(dB, regs);
    // load 40 g -> r_big
    load_group<SMALL_BYTE_STEP, SMEM_LEN>(dA, regs);

    //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SHARED) + [0..2*SMEM_LEN]
    //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
    //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
    //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN)

    // store 20 r_small -> g
    store_group<SMALL_BYTE_STEP, SMALL_REGS, SMEM_LEN>(dB, regs + SMEM_LEN);

    //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS + SMEM_LEN
    //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
    //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS
    //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS

    // store 40 r_big -> g
    store_group<SMALL_BYTE_STEP, SMEM_LEN, MAX_REGS>(dB, regs);

    //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
    //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
    //// dB stored 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
    //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN)
}
