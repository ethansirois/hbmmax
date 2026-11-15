#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include "bench/config.hpp"
#include "bench/timing.cuh"
#include HBM_CONFIG_HEADER
#include HBM_PROBLEM_HEADER

int main(int argc, char** argv) {
    try {
        RunConfig config;
        std::string cubin = (std::filesystem::absolute(argv[0]).parent_path() / "kernel.cubin").string();
        std::string json_path;
        bool peak_only = false, kernel_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::puts("Options: --device N --warmup N --iterations N --cubin PATH --json PATH --peak-only --kernel-only");
                return 0;
            }
            if (option == "--peak-only") { peak_only = true; continue; }
            if (option == "--kernel-only") { kernel_only = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--cubin") { cubin = value; continue; }
            if (option == "--json") { json_path = value; continue; }
            size_t parsed;
            const int number = std::stoi(value, &parsed);
            if (parsed != value.size()) throw std::runtime_error("Invalid number: " + value);
            if (option == "--device") config.device = number;
            else if (option == "--warmup") config.warmup = number;
            else if (option == "--iterations") config.iterations = number;
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (config.device < 0 || config.warmup < 0 || config.iterations <= 0 || (peak_only && kernel_only))
            throw std::runtime_error("Invalid run configuration");
        CUDA_CHECK(cudaSetDevice(config.device));
        // Initialize the runtime's primary context before loading the assembled cubin.
        CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, config.device));
        if (SMALL_SMs_USED > properties.multiProcessorCount || LARGE_SMs_USED > properties.multiProcessorCount)
            throw std::runtime_error("Requested SM budget exceeds the device SM count");
        if (THREADS_PER_BLOCK > properties.maxThreadsPerBlock)
            throw std::runtime_error("Requested block size exceeds the device limit");
        std::printf("device=%s numel=%zu sms=%d peak_sms=%d\n", properties.name,
                    Problem::elements(), SMALL_SMs_USED, LARGE_SMs_USED);
        Problem problem(cubin.c_str());
        bool valid = true;
        double bandwidth[2] = {0, 0};
        float milliseconds[2] = {0, 0};
        for (int mode = 0; mode < 2; ++mode) {
            const bool peak = mode == 1;
            if ((peak_only && !peak) || (kernel_only && peak)) continue;
            problem.reset();
            problem.launch(peak);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (!problem.validate()) { valid = false; continue; }
            const float ms = time_kernel([&] { problem.launch(peak); }, config);
            const bool correct = problem.validate();
            valid = valid && correct;
            milliseconds[mode] = ms;
            bandwidth[mode] = double(Problem::bytes_processed()) / (ms * 1e6);
            std::printf("%s: %.3f GB/s (%.6f ms) valid=%s\n", peak ? "peak" : "kernel",
                        bandwidth[mode], ms, correct ? "yes" : "no");
        }
        if (!json_path.empty()) {
            std::ofstream report(json_path);
            if (!report) throw std::runtime_error("Cannot write " + json_path);
            report << std::setprecision(12)
                   << "{\"device\":" << std::quoted(properties.name)
                   << ",\"available_sms\":" << properties.multiProcessorCount
                   << ",\"numel\":" << Problem::elements()
                   << ",\"sms\":" << SMALL_SMs_USED << ",\"peak_sms\":" << LARGE_SMs_USED
                   << ",\"iterations\":" << config.iterations << ",\"warmup\":" << config.warmup
                   << ",\"valid\":" << (valid ? "true" : "false")
                   << ",\"kernel_gbps\":" << bandwidth[0] << ",\"peak_gbps\":" << bandwidth[1]
                   << ",\"kernel_ms\":" << milliseconds[0] << ",\"peak_ms\":" << milliseconds[1] << "}\n";
            if (!report) throw std::runtime_error("Failed writing " + json_path);
        }
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
