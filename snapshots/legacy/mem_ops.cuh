#pragma once

#include <utility>

template <int offset>
__device__ __forceinline__
float4 load(const float4* p) {
    float4 v;
    asm volatile("ld.global.cg.nc.v4.f32 {%0, %1, %2, %3}, [%4 + %5];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p), "n"(offset));
    return v;
}

template <int offset>
__device__ __forceinline__
void store(float4* p, float4 v) {
    asm volatile("st.global.cg.v4.f32 [%0 + %1], {%2, %3, %4, %5};"
        :: "l"(p), "n"(offset), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
        : "memory");
}

template <int ByteStride, int BaseElem, int... J>
__device__ __forceinline__
void load_group(const float4* p, float4* regs, std::integer_sequence<int, J...>) {
    ((regs[J] = load<(BaseElem + J) * ByteStride>(p)), ...);
}

template <int ByteStride, int BaseElem, int... J>
__device__ __forceinline__
void store_group(float4* p, const float4* regs, std::integer_sequence<int, J...>) {
    (store<(BaseElem + J) * ByteStride>(p, regs[J]), ...);
}

template <int ByteStride, int Count, int BaseElem = 0>
__device__ __forceinline__
void load_group(const float4* p, float4* regs) {
    load_group<ByteStride, BaseElem>(p, regs, std::make_integer_sequence<int, Count>{});
}

template <int ByteStride, int Count, int BaseElem = 0>
__device__ __forceinline__
void store_group(float4* p, const float4* regs) {
    store_group<ByteStride, BaseElem>(p, regs, std::make_integer_sequence<int, Count>{});
}
