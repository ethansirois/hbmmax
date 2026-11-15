#pragma once

#include <cstdio>
#include <random>
#include <vector>
#include "bench/cuda_utils.cuh"

// The driver uses this interface; future problems can own different buffers and checks.
class Problem {
    std::vector<float> input_, output_;
    DeviceBuffer<float4> input_device_, output_device_;
    CudaModule module_;
    CUfunction small_, peak_;
public:
    explicit Problem(const char* cubin) : input_(NUMEL), output_(NUMEL),
                input_device_((NUMEL + 3) / 4), output_device_((NUMEL + 3) / 4), module_(cubin) {
        DRIVER_CHECK(cuModuleGetFunction(&small_, module_.handle, "small_SMs"));
        DRIVER_CHECK(cuModuleGetFunction(&peak_, module_.handle, "large_SMs"));
        std::mt19937 generator(0);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for (auto& value : input_) value = distribution(generator);
        CUDA_CHECK(cudaMemcpy(input_device_.data, input_.data(), NUMEL * sizeof(float),
                              cudaMemcpyHostToDevice));
        if constexpr (SMEM_LEN > 0) {
            DRIVER_CHECK(cuFuncSetAttribute(small_,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_LEN * THREADS_PER_BLOCK * sizeof(float4)));
        }
    }

    static constexpr size_t elements() { return NUMEL; }
    static constexpr size_t bytes_processed() { return 2 * size_t(NUMEL) * sizeof(float); }
    void reset() {
        // A non-finite sentinel makes unwritten output fail validation.
        CUDA_CHECK(cudaMemset(output_device_.data, 0xff, NUMEL * sizeof(float)));
    }
    void launch(bool peak) {
        void* arguments[] = {&input_device_.data, &output_device_.data};
        const unsigned shared = peak ? 0 : SMEM_LEN * THREADS_PER_BLOCK * sizeof(float4);
        DRIVER_CHECK(cuLaunchKernel(peak ? peak_ : small_,
            peak ? LARGE_SMs_USED : SMALL_SMs_USED, 1, 1, THREADS_PER_BLOCK, 1, 1,
            shared, nullptr, arguments, nullptr));
    }
    bool validate() {
        CUDA_CHECK(cudaMemcpy(output_.data(), output_device_.data, NUMEL * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < input_.size(); ++i) {
            if (input_[i] != output_[i]) {
                std::fprintf(stderr, "Mismatch at %zu: expected %f, got %f\n",
                             i, input_[i], output_[i]);
                return false;
            }
        }
        return true;
    }
};
