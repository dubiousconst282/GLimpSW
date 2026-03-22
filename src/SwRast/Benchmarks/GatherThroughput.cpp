#include <cstdint>
#include <immintrin.h>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

constexpr uint32_t BLOCK_SIZE = 1024;

[[gnu::noinline]]
uint32_t LoadScalar32(const uint32_t* src) {
    uint32_t sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 4) {
        // NOTE: no diff between LEA addressing modes:
        // 03 47 1C                      add    eax, dword ptr [rdi + 0x1c]
        // 03 44 8F 0C                   add    eax, dword ptr [rdi + 4*rcx + 0xc]
        sum1 += src[i + 0];
        sum2 += src[i + 1];
        sum3 += src[i + 2];
        sum4 += src[i + 3];
    }
    return sum1+sum2+sum3+sum4;
}

using v8_uint = uint32_t [[clang::ext_vector_type(8)]];
using v16_uint = uint32_t [[clang::ext_vector_type(16)]];

template<typename T>
T load(const void* ptr) {
    struct [[gnu::may_alias, gnu::packed]] block { T value; };
    return ((block*)ptr)->value;
}

[[gnu::noinline]]
uint32_t LoadAVX2(const uint32_t* src) {
    v8_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 16) {
        sum1 += load<v8_uint>(&src[i + 0]);
        sum2 += load<v8_uint>(&src[i + 8]);
    }
    return __builtin_reduce_add(sum1 + sum2);
}

[[gnu::noinline]]
uint32_t LoadAVX512(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        sum1 += load<v16_uint>(&src[i + 0]);
        sum2 += load<v16_uint>(&src[i + 16]);
    }
    return __builtin_reduce_add(sum1 + sum2);
}

[[gnu::noinline]]
uint32_t Gather_AVX512(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        v16_uint idx = v16_uint{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        sum1 += (v16_uint)_mm512_i32gather_epi32(idx, &src[i + 0], 4);
        sum2 += (v16_uint)_mm512_i32gather_epi32(idx, &src[i + 16], 4);
    }
    return __builtin_reduce_add(sum1 + sum2);
}

[[gnu::noinline]]
uint32_t Gather64_AVX512(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        v16_uint idx = v16_uint{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        sum1 += (v16_uint)_mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx * 2, 0), &src[i + 0], 4);
        sum2 += (v16_uint)_mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx * 2, 0), &src[i + 16], 4);
    }
    return __builtin_reduce_add(sum1 + sum2);
}

uint64_t splitmix64(uint64_t& s) {
    uint64_t result = (s += 0x9E3779B97f4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

int main() {
    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.epochs(30);

    alignas(64) uint32_t src[BLOCK_SIZE];

    uint64_t seed = 12345;
    for (int i = 0; i < BLOCK_SIZE; i++) src[i] = splitmix64(seed);

    ctx.relative();

    ctx.run("Load_Scalar_32", [&]() {
        auto res = LoadScalar32(src);
        ctx.doNotOptimizeAway(res * res);
    });
    ctx.run("Load_AVX2", [&]() {
        auto res = LoadAVX2(src);
        ctx.doNotOptimizeAway(res * res);
    });
    ctx.run("Load_AVX512", [&]() {
        auto res = LoadAVX512(src);
        ctx.doNotOptimizeAway(res * res);
    });
    ctx.run("Gather32_AVX512", [&]() {
        auto res = Gather_AVX512(src);
        ctx.doNotOptimizeAway(res * res);
    });
    ctx.run("Gather64_AVX512", [&]() {
        auto res = Gather64_AVX512(src);
        ctx.doNotOptimizeAway(res * res);
    });
}


/*
TigerLake

GDS mitigation = on
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|              121.69 |        8,217,937.67 |    1.3% |        1,230.00 |          302.42 |  4.067 |          66.00 |    1.5% |      1.64 | `Load_Scalar_32`
|               18.70 |       53,467,269.08 |    0.7% |          171.00 |           46.36 |  3.688 |          10.00 |    0.0% |      1.67 | `Load_AVX2`
|               18.69 |       53,495,847.79 |    0.3% |           97.00 |           46.46 |  2.088 |           6.00 |    0.0% |      1.65 | `Load_AVX512`
|              410.49 |        2,436,102.25 |    0.6% |          294.00 |        1,018.32 |  0.289 |           6.00 |    0.0% |      1.64 | `Gather_AVX512`
Process exited with code 0.

kernel mitigations = off
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|              120.67 |        8,287,178.49 |    0.6% |        1,230.00 |          299.41 |  4.108 |          66.00 |    1.5% |      1.56 | `Load_Scalar_32`
|               30.60 |       32,684,184.90 |    0.7% |          171.00 |           76.03 |  2.249 |          10.00 |    0.0% |      1.66 | `Load_AVX2`
|               18.66 |       53,602,141.29 |    0.3% |           97.00 |           46.34 |  2.093 |           6.00 |    0.0% |      1.66 | `Load_AVX512`
|              135.25 |        7,393,699.75 |    0.5% |          294.00 |          336.43 |  0.874 |           6.00 |    0.0% |      1.66 | `Gather32_AVX512`
|               78.83 |       12,685,063.34 |    0.7% |          295.00 |          196.10 |  1.504 |           6.00 |    0.0% |      1.66 | `Gather64_AVX512`
Process exited with code 0.


- strange avx2 regression?
*/