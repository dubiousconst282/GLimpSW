#include <cstdint>

#include "../SIMD.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

namespace simd = swr::simd;
using swr::v_int, swr::v_mask;

[[gnu::noinline]] v_int Gather32x64_Native(const int data[64], v_int idx) {
    return _mm512_i32gather_epi32(idx, data, 4);
}

[[gnu::noinline]] v_int Gather32x64_Perm2i(const int data[64], v_int idx) {
    v_int v0 = _mm512_load_epi32(&data[0]);
    v_int v1 = _mm512_load_epi32(&data[16]);
    v_int v2 = _mm512_load_epi32(&data[32]);
    v_int v3 = _mm512_load_epi32(&data[48]);
    v_int p01 = _mm512_permutex2var_epi32(v0, idx, v1);
    v_int p23 = _mm512_permutex2var_epi32(v2, idx, v3);
    return idx < 32 ? p01 : p23;
}

int main() {
    alignas(64) int data[64];
    for (int i = 0; i < 64; i++) data[i] = i;

    auto R1=Gather32x64_Native(data, simd::lane_idx*4);
    auto R2=Gather32x64_Perm2i(data, simd::lane_idx*4);

    printf("N: ");
    for (uint32_t i = 0; i < 16; i++) printf("% 5d", R1[i]);
    printf("\n");

    printf("P: ");
    for (uint32_t i = 0; i < 16; i++) printf("% 5d", R2[i]);
    printf("\n");

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.epochs(30);

    ctx.relative(true);
    ctx.batch(64);

    ctx.run("Native", [&]() {
        for (int i = 0; i < 64; i++) {
            auto r = Gather32x64_Native(data, ((simd::lane_idx + i) * 397) & 63);
            ankerl::nanobench::detail::doNotOptimizeAway(r);
        }
    });
    ctx.run("Perm2i", [&]() {
        for (int i = 0; i < 64; i++) {
            auto r = Gather32x64_Perm2i(data, ((simd::lane_idx + i) * 397) & 63);
            ankerl::nanobench::detail::doNotOptimizeAway(r);
        }
    });
}

/*
TigerLake, mitigations=off
| relative |               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|---------:|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|   100.0% |                2.96 |      337,291,629.21 |    0.1% |           14.06 |            7.39 |  1.903 |           3.00 |    0.0% |      1.65 | `Native`
|   224.6% |                1.32 |      757,478,514.44 |    0.2% |           17.06 |            3.29 |  5.188 |           3.00 |    0.0% |      1.66 | `Perm2i`

*/
