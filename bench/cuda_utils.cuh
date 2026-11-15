#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(expression) cuda_check((expression), #expression)

inline void driver_check(CUresult status, const char* expression) {
    if (status != CUDA_SUCCESS) {
        const char* message = "unknown CUDA driver error";
        cuGetErrorString(status, &message);
        throw std::runtime_error(std::string(expression) + ": " + message);
    }
}
#define DRIVER_CHECK(expression) driver_check((expression), #expression)

class CudaModule {
public:
    CUmodule handle;
    explicit CudaModule(const char* path) { DRIVER_CHECK(cuModuleLoad(&handle, path)); }
    ~CudaModule() { cuModuleUnload(handle); }
    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;
};

template <class T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { CUDA_CHECK(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

class CudaEvent {
public:
    cudaEvent_t handle;
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&handle)); }
    ~CudaEvent() { cudaEventDestroy(handle); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};
