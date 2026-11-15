#pragma once

#include "bench/config.hpp"
#include "bench/cuda_utils.cuh"

template <class Launch>
float time_kernel(Launch launch, const RunConfig& config) {
    for (int i = 0; i < config.warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaEvent start, end;
    CUDA_CHECK(cudaEventRecord(start.handle));
    for (int i = 0; i < config.iterations; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end.handle));
    CUDA_CHECK(cudaEventSynchronize(end.handle));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.handle, end.handle));
    return ms / config.iterations;
}
