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
constexpr v16_uint LaneIdx = v16_uint{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

[[gnu::noinline]]
uint32_t Gather_AVX512(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        sum1 += (v16_uint)_mm512_i32gather_epi32(LaneIdx, &src[i + 0], 4);
        sum2 += (v16_uint)_mm512_i32gather_epi32(LaneIdx, &src[i + 16], 4);
    }
    return __builtin_reduce_add(sum1 + sum2);
}

[[gnu::noinline]]
uint32_t Gather64_AVX512(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        v16_uint idx = LaneIdx * 2 + i;
        sum1 += (v16_uint)_mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx, 0), src, 4);
        sum2 += (v16_uint)_mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx, 1), src, 4);
    }
    return __builtin_reduce_add(sum1 + sum2);
}

[[gnu::noinline]]
uint32_t Gather2x32_AVX512_Permx(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        v16_uint idx = i + v16_uint{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        auto lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(idx, 0), src, 4);
        auto hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(idx, 1), src, 4);
        auto x = _mm512_permutex2var_epi32(lo, LaneIdx * 2 + 0, hi);
        auto y = _mm512_permutex2var_epi32(lo, LaneIdx * 2 + 1, hi);

        sum1 += x;
        sum2 += y;
    }
    return __builtin_reduce_add(sum1 + sum2);
}

[[gnu::noinline]]
uint32_t Gather2x32_AVX512_Blend(const uint32_t* src) {
    v16_uint sum1 = 0, sum2 = 0;
    for (uint32_t i = 0; i < BLOCK_SIZE; i += 32) {
        v16_uint idx = i + v16_uint{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        constexpr v16_uint lo_mask = (LaneIdx & 1) - 1;  // -1 0 -1 0 ...
        auto lo = _mm512_i64gather_epi64(idx & lo_mask, src, 4);
        auto hi = _mm512_i64gather_epi64(_mm512_srli_epi64(idx, 32), src, 4);
        auto x = _mm512_ternarylogic_epi32(lo, _mm512_slli_epi64(hi, 32), lo_mask, (_MM_TERNLOG_A & _MM_TERNLOG_C) | _MM_TERNLOG_B);
        auto y = _mm512_ternarylogic_epi32(hi, _mm512_srli_epi64(lo, 32), lo_mask, (_MM_TERNLOG_A & ~_MM_TERNLOG_C) | _MM_TERNLOG_B);

        sum1 += x;
        sum2 += y;
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
    ctx.run("Gather2x32_AVX512_Permx", [&]() {
        auto res = Gather2x32_AVX512_Permx(src);
        ctx.doNotOptimizeAway(res * res);
    });
    ctx.run("Gather2x32_AVX512_Blend", [&]() {
        auto res = Gather2x32_AVX512_Blend(src);
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

kernel mitigations = off
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|              119.10 |        8,396,641.28 |    0.2% |        1,230.00 |          296.90 |  4.143 |          66.00 |    1.5% |      1.65 | `Load_Scalar_32`
|               17.79 |       56,207,606.73 |    0.2% |          171.00 |           44.30 |  3.860 |          10.00 |    0.0% |      1.66 | `Load_AVX2`
|               12.38 |       80,760,665.05 |    0.2% |           97.00 |           30.84 |  3.145 |           6.00 |    0.0% |      1.65 | `Load_AVX512`
|              134.53 |        7,433,123.72 |    0.1% |          294.00 |          335.34 |  0.877 |           6.00 |    0.0% |      1.65 | `Gather32_AVX512`
|               79.82 |       12,528,398.30 |    0.2% |          422.00 |          198.81 |  2.123 |          10.00 |    0.0% |      1.66 | `Gather64_AVX512`
|               83.74 |       11,941,690.89 |    0.1% |          520.00 |          208.63 |  2.492 |          10.00 |    0.0% |      1.65 | `Gather2x32_AVX512_Permx`
|               84.26 |       11,867,744.05 |    0.1% |          601.00 |          209.89 |  2.863 |          18.00 |    0.0% |      1.66 | `Gather2x32_AVX512_Blend`

*/