#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>


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
constexpr int NUMEL = (1 << 24);
constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_THREAD_BLOCKS = (NUMEL / BLOCKS_PER_SM) / THREADS_PER_BLOCK / 4;

// test
constexpr int SMALL_NUM_STEPS = NUM_THREAD_BLOCKS / SMALL_SMs_USED;

// max
constexpr int LARGE_NUM_STEPS = NUM_THREAD_BLOCKS / LARGE_SMs_USED;


extern "C"
__global__ void 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
small_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    const int step = BLOCKS_PER_SM * SMALL_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < SMALL_NUM_STEPS; ++i) {
        dB[idx] = dA[idx];
        dA += step;
        dB += step;
    }
}


__global__ void large_SMs(const float4* __restrict__ dA, float4* __restrict__ dB) {
    const int step = BLOCKS_PER_SM * LARGE_SMs_USED * THREADS_PER_BLOCK;
    const int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

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
        printf("%i max x, %i max y, %i max z\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
		printf("%i regsPerBlock\n", devProp.regsPerBlock);
        printf("%i regsPerMultiprocessor\n", devProp.regsPerMultiprocessor);

	}


	// size_t freeMem;
	// size_t totalMem;
	// cudaMemGetInfo(&freeMem, &totalMem);
	// printf("freeMem: %llu,\ntotalMem: %llu\n", static_cast<unsigned long long>(freeMem >> 30), static_cast<unsigned long long>(totalMem >> 30));

    printf("NUM BYTES: %i\n", NUMEL * sizeof(float));
    printf("SMALL_SMs_USED: %i\n", SMALL_SMs_USED);
    printf("LARGE_SMs_USED: %i\n", LARGE_SMs_USED);
	

	float* A = new float[NUMEL];
    init_random(A, NUMEL);
    float* B = new float[NUMEL];

    float4* dA;
    cudaMalloc(&dA, NUMEL * sizeof(float));
    cudaMemcpy(dA, A, NUMEL * sizeof(float), cudaMemcpyHostToDevice);
    
    float4* dB;
    cudaMalloc(&dB, NUMEL * sizeof(float));

    float4* dA2;
    cudaMalloc(&dA2, NUMEL * sizeof(float));
    cudaMemcpy(dA2, A, NUMEL * sizeof(float), cudaMemcpyHostToDevice);
    
    float4* dB2;
    cudaMalloc(&dB2, NUMEL * sizeof(float));


    int* which_block;
    cudaMalloc(&which_block, sizeof(int));
    cudaMemset(which_block, 0, sizeof(int));
    

    // setup bench
    cudaEvent_t start;
    cudaEvent_t end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // warmup
    small_SMs<<<SMALL_SMs_USED, THREADS_PER_BLOCK>>>(dA, dB);
    // cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();
    // cudaCheckErrors("dev sync");

    // launch kernel
    cudaEventRecord(start);
    for (int i = 0; i < N_its; ++i) {
        small_SMs<<<SMALL_SMs_USED, THREADS_PER_BLOCK>>>(dA, dB);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);



    float ms;
    cudaEventElapsedTime(&ms, start, end);
    printf("ms/N_its: %f\n", ms/N_its);
    printf("bandwidth (GB/S): %f\n", sizeof(float) * 2 * double(static_cast<double>(NUMEL) / 1000.0 / 1000.0) / (ms / double(N_its)));

    cudaMemcpy(B, dB, NUMEL * sizeof(float), cudaMemcpyDeviceToHost);

    printf("equal: %i\n", check_equal(A, B, NUMEL));
    printf("bandwidth: %f\n", float(NUMEL * sizeof(float) * 2) / ms / 1000000.0);



    /// Baseline ///
    // reset B
    std::fill_n(B, NUMEL, 0.0f);
    
    // setup bench
    cudaEvent_t mstart;
    cudaEvent_t mend;

    cudaEventCreate(&mstart);
    cudaEventCreate(&mend);

    // warmup
    large_SMs<<<LARGE_SMs_USED, THREADS_PER_BLOCK>>>(dA2, dB2);
    // cudaCheckErrors("warmup launch");
    cudaDeviceSynchronize();
    // cudaCheckErrors("dev sync");

    // launch kernel
    cudaEventRecord(mstart);
    for (int i = 0; i < N_its; ++i) {
        large_SMs<<<LARGE_SMs_USED, THREADS_PER_BLOCK>>>(dA2, dB2);
    }
    cudaEventRecord(mend);
    cudaEventSynchronize(mend);



    float mms;
    cudaEventElapsedTime(&mms, mstart, mend);
    printf("mms/N_its: %f\n", mms/N_its);
    printf("bandwidth (GB/S): %f\n", sizeof(float) * 2 * double(static_cast<double>(NUMEL) / 1000.0 / 1000.0) / (mms / double(N_its)));

    cudaMemcpy(B, dB2, NUMEL * sizeof(float), cudaMemcpyDeviceToHost);
    printf("equal: %i\n", check_equal(A, B, NUMEL));



    cudaFree(dA);
    cudaFree(dB);

	delete[] A;
	delete[] B;
	
	return 0;
}
