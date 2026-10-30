#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


/// Update per machine /// 
constexpr int NUM_SMs = 48;

// let's just use a multiple of 2, will fully saturate it anyways
constexpr int LARGE_SMs_USED = 32; 
constexpr int SMALL_SMs_USED = 4;

constexpr int N_its = 100;
constexpr int BLOCKS_PER_SM = 1;
constexpr int NUM_MAIN_LOOPS = 512;
constexpr int SMEM_LEN = 20;
constexpr int MAX_REGS = 61;
constexpr int THREADS_PER_SM = 256;
constexpr long SMALL_NUMEL = SMALL_SMs_USED * THREADS_PER_SM * 4 * (NUM_MAIN_LOOPS * (MAX_REGS + SMEM_LEN) + (2 * MAX_REGS + SMEM_LEN));
constexpr int LARGE_NUMEL = LARGE_SMs_USED * THREADS_PER_SM * 4 * (NUM_MAIN_LOOPS * (MAX_REGS + SMEM_LEN) + (2*MAX_REGS));

// test
// constexpr int SMALL_NUM_STEPS = NUM_THREAD_BLOCKS / SMALL_SMs_USED;

// max
constexpr int LARGE_NUM_STEPS = LARGE_NUMEL / LARGE_SMs_USED / THREADS_PER_SM / 4;



extern "C"
__global__ void 
__launch_bounds__(THREADS_PER_SM, BLOCKS_PER_SM)
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
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
        dB[i * step] = dA[i * step];
        if (i < SMEM_LEN) {
            __pipeline_memcpy_async(&smem[i * THREADS_PER_SM], &dA[(MAX_REGS + i) * step], sizeof(float4));
        } else {
            regs[i] = dA[(MAX_REGS + i) * step];
        }
    }

    __pipeline_commit();


    dA += 2 * MAX_REGS * step;
    dB += MAX_REGS * step;
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
            dB[i * step] = smem[i * THREADS_PER_SM];
            // copy 40 g -> s
            __pipeline_memcpy_async(&smem[i * THREADS_PER_SM], &dA[(i + SMEM_LEN) * step], sizeof(float4));
            // store 40 r_big -> g
            // load 40 g -> r_big
            regs[i] = dA[i * step];
        }
        __pipeline_commit();

        //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SHARED) + [0..2*SMEM_LEN]
        //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
        //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
        //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN)

        #pragma unroll
        for (int i = SMEM_LEN; i < MAX_REGS; ++i) {
            // store 20 r_small -> g
            dB[(i) * step] = regs[i];
            // load 20 g -> r_small
            regs[i] = dA[(i + SMEM_LEN) * step];
        }

        //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS + SMEM_LEN
        //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
        //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS
        //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS

        // store 40 r_big -> g
        #pragma unroll
        for (int i = 0; i < SMEM_LEN; ++i) {
            dB[(MAX_REGS + i) * step] = regs[i];
        }
        dA += (MAX_REGS + SMEM_LEN) * step;
        dB += (MAX_REGS + SMEM_LEN) * step;

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
        // store 40 r_big -> g
        dB[i * step] = regs[i];
        // load 40 g -> r_big
        regs[i] = dA[i * step];
    }

    //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SHARED) + [0..2*SMEM_LEN]
    //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
    //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
    //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN)

    #pragma unroll
    for (int i = SMEM_LEN; i < MAX_REGS; ++i) {
        // store 20 r_small -> g
        dB[(i) * step] = regs[i];
        // load 20 g -> r_small
    }

    //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS + SMEM_LEN
    //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
    //// dB stored MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS
    //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN) + MAX_REGS

    // store 40 r_big -> g
    #pragma unroll
    for (int i = 0; i < SMEM_LEN; ++i) {
        dB[(MAX_REGS + i) * step] = regs[i];
    }

    //// dA loaded 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
    //// dA = original_dA + 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN)
    //// dB stored 2*MAX_REGS + it*(MAX_REGS + SMEM_LEN) + SMEM_LEN
    //// dB = original_dB + MAX_REGS + it*(MAX_REGS + SMEM_LEN)
}


__global__ void large_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    const int step = BLOCKS_PER_SM * LARGE_SMs_USED * THREADS_PER_SM;
    const int idx = blockIdx.x * THREADS_PER_SM + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < LARGE_NUM_STEPS; ++i) {
        dB[i * step + idx] = dA[i * step + idx];
    }
}


bool check_equal(const float* A, const float* B, const int n) {
    for (int i = 0; i < n; ++i) {
        if (A[i] != B[i]) {
            printf("NOT Equal at i: %i, with A[i]: %f and B[i]: %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void init_random(float* A, const int n) {
    for (int i = 0; i < n; ++i) {
        A[i] = rand() / (float)RAND_MAX;
    }
}


int main() {
	int devCount;
	cudaGetDeviceCount(&devCount);
	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printf("dev %i has:\n", i);
		printf("%i SMs\n", devProp.multiProcessorCount);
		printf("%i maxThreadsPerBlock\n", devProp.maxThreadsPerBlock);
        printf("%i maxThreadsPerSM\n", devProp.maxThreadsPerMultiProcessor);
        printf("%i maxDynamicSharedMemorySize\n", devProp.sharedMemPerBlockOptin);
        printf("%i max x, %i max y, %i max z\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
		printf("%i regsPerBlock\n", devProp.regsPerBlock);
        printf("%i regsPerMultiprocessor\n", devProp.regsPerMultiprocessor);

	}


    printf("SMALL_SMs_USED: %i\n", SMALL_SMs_USED);
    printf("LARGE_SMs_USED: %i\n", LARGE_SMs_USED);
	

	float* A = new float[SMALL_NUMEL];
    init_random(A, SMALL_NUMEL);
    float* B = new float[SMALL_NUMEL];


    float4* dA;
    cudaMalloc(&dA, SMALL_NUMEL * sizeof(float));
    cudaMemcpy(dA, A, SMALL_NUMEL * sizeof(float), cudaMemcpyHostToDevice);
    
    float4* dB;
    cudaMalloc(&dB, SMALL_NUMEL * sizeof(float));

    float4* dA2;
    cudaMalloc(&dA2, SMALL_NUMEL * sizeof(float));
    cudaMemcpy(dA2, A, SMALL_NUMEL * sizeof(float), cudaMemcpyHostToDevice);
    
    float4* dB2;
    cudaMalloc(&dB2, SMALL_NUMEL * sizeof(float));


    int* which_block;
    cudaMalloc(&which_block, sizeof(int));
    cudaMemset(which_block, 0, sizeof(int));
    
    const int small_smem_bytes = THREADS_PER_SM * SMEM_LEN * (int)sizeof(float4);
    cudaFuncSetAttribute(
        small_SMs,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        small_smem_bytes);
    cudaCheckErrors("set dynamic smem");

    // setup bench
    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // warmup
    small_SMs<<<SMALL_SMs_USED, THREADS_PER_SM, small_smem_bytes>>>(dA, dB);
    cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();
    cudaCheckErrors("dev sync");

    // launch kernel
    cudaEventRecord(start);
    for (int i = 0; i < N_its; ++i) {
        small_SMs<<<SMALL_SMs_USED, THREADS_PER_SM, small_smem_bytes>>>(dA, dB);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);



    float ms;
    cudaEventElapsedTime(&ms, start, end);
    printf("ms/N_its: %f\n", ms/N_its);
    printf("bandwidth (GB/S): %f\n", sizeof(float) * 2 * double(static_cast<double>(SMALL_NUMEL) / 1000.0 / 1000.0) / (ms / double(N_its)));


    cudaMemcpy(B, dB, SMALL_NUMEL * sizeof(float), cudaMemcpyDeviceToHost);

    printf("equal: %i\n", check_equal(A, B, SMALL_NUMEL));


    /*

    /// Baseline ///
    // reset B
    std::fill_n(B, LARGE_NUMEL, 0.0f);
    
    // setup bench
    cudaEvent_t mstart;
    cudaEvent_t mend;

    cudaEventCreate(&mstart);
    cudaEventCreate(&mend);

    // warmup
    large_SMs<<<LARGE_SMs_USED, THREADS_PER_SM>>>(dA2, dB2);
    // cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();
    // cudaCheckErrors("dev sync");

    // launch kernel
    cudaEventRecord(mstart);
    for (int i = 0; i < N_its; ++i) {
        large_SMs<<<LARGE_SMs_USED, THREADS_PER_SM>>>(dA2, dB2);
    }
    cudaEventRecord(mend);
    cudaEventSynchronize(mend);



    float mms;
    cudaEventElapsedTime(&mms, mstart, mend);
    printf("mms/N_its: %f\n", mms/N_its);

    cudaMemcpy(B, dB2, LARGE_NUMEL * sizeof(float), cudaMemcpyDeviceToHost);
    printf("equal: %i\n", check_equal(A, B, LARGE_NUMEL));

    */


    cudaFree(dA);
    cudaFree(dB);

	delete[] A;
	delete[] B;
	
	return 0;
}
