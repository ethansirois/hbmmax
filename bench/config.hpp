#pragma once

#include "kernels/common/config.cuh"

struct RunConfig {
    int device = 0;
    int warmup = 5;
    int iterations = 100;
};
