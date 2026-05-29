#include <cstdint>

#include "../SIMD.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

constexpr int NumFields = 8;

struct alignas(64) Data {
    float Field[16][1024];
};

[[clang::noinline]]
void GatherMany_Native(const Data* data, v_int idx, v_float* dst) {
    for (uint32_t j = 0; j < NumFields; j++) {
        dst[j] += _mm512_i32gather_ps(idx, data->Field[j], 4);
    }
}

[[clang::noinline, clang::no_stack_protector]]
void GatherMany_Cascade(const Data* data, v_int idx, v_float* dst) {
    v_float tmp[NumFields] = {};

    simd_foreach_unique_lane(i, idx, v_mask(~0u)) {
        for (uint32_t j = 0; j < NumFields; j++) {
            simd::cmov(tmp[j], data->Field[j][i], iterMask);
        }
    }
    for (uint32_t i = 0; i < NumFields; i++) {
        dst[i] += tmp[i];
    }
}

int main() {
    Data data;
    for (uint32_t i = 0; i < 16; i++) {
        for (uint32_t j = 0; j < 1024; j++) {
            data.Field[i][j] = i * 1024 + j;
        }
    }

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.epochs(30);
    ctx.batch(64);

    for (int density = 1; density < 16; density++) {
        v_int baseIdx = 0;
        for (int i = 1; i < density; i++) baseIdx[i] = i;

        std::string tag = "_" + std::to_string(density);

        ctx.run("Gather" + tag, [&]() {
            v_float r[NumFields] = {};
            for (int i = 0; i < 1024; i += 16) GatherMany_Native(&data, baseIdx + i, r);
            ankerl::nanobench::detail::doNotOptimizeAway(r);
        });
        ctx.run("Cascade" + tag, [&]() {
            v_float r[NumFields] = {};
            for (int i = 0; i < 1024; i += 16) GatherMany_Cascade(&data, baseIdx + i, r);
            ankerl::nanobench::detail::doNotOptimizeAway(r);
        });

        printf("\n");
    }
}

/*
TigerLake, 2.5GHz, mitigations=off

NumFields=8
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|               37.12 |       26,942,660.33 |    0.5% |           51.20 |           92.29 |  0.555 |           3.00 |    0.0% |      1.65 | `Gather_1`
|                7.06 |      141,589,518.68 |    0.1% |           61.27 |           17.50 |  3.501 |           4.00 |    0.0% |      1.64 | `Cascade_1`

|               37.11 |       26,946,024.10 |    0.2% |           51.20 |           91.86 |  0.557 |           3.00 |    0.0% |      1.66 | `Gather_2`
|               11.17 |       89,550,412.95 |    0.1% |           77.27 |           27.64 |  2.795 |           5.00 |    0.3% |      1.65 | `Cascade_2`

|               37.15 |       26,920,894.90 |    0.1% |           51.20 |           91.91 |  0.557 |           3.00 |    0.0% |      1.65 | `Gather_3`
|               15.70 |       63,692,853.77 |    0.1% |           93.27 |           38.85 |  2.401 |           6.00 |    0.3% |      1.66 | `Cascade_3`

|               37.15 |       26,918,473.01 |    0.2% |           51.20 |           91.90 |  0.557 |           3.00 |    0.0% |      1.66 | `Gather_4`
|               19.88 |       50,300,534.06 |    0.3% |          109.27 |           49.14 |  2.224 |           7.00 |    0.2% |      1.66 | `Cascade_4`

|               37.08 |       26,969,655.61 |    0.1% |           51.20 |           91.68 |  0.558 |           3.00 |    0.0% |      1.66 | `Gather_5`
|               25.08 |       39,880,045.32 |    0.1% |          125.27 |           61.99 |  2.021 |           8.00 |    0.2% |      1.65 | `Cascade_5`

|               36.95 |       27,062,081.92 |    0.3% |           51.20 |           91.86 |  0.557 |           3.00 |    0.0% |      1.66 | `Gather_6`
|               31.07 |       32,183,256.77 |    0.1% |          141.27 |           77.36 |  1.826 |           9.00 |    0.2% |      1.65 | `Cascade_6`

|               36.90 |       27,099,856.07 |    0.0% |           51.20 |           91.87 |  0.557 |           3.00 |    0.0% |      1.66 | `Gather_7`
|               38.34 |       26,083,417.47 |    0.1% |          157.27 |           95.43 |  1.648 |          10.00 |    0.2% |      1.66 | `Cascade_7`

|               36.98 |       27,044,474.84 |    0.2% |           51.20 |           91.88 |  0.557 |           3.00 |    0.0% |      1.66 | `Gather_8`
|               46.75 |       21,390,576.35 |    0.1% |          173.27 |          115.74 |  1.497 |          11.00 |    0.1% |      1.64 | `Cascade_8`

|               37.03 |       27,004,424.51 |    0.1% |           51.20 |           91.65 |  0.559 |           3.00 |    0.0% |      1.65 | `Gather_9`
|               56.64 |       17,656,180.40 |    0.2% |          189.27 |          140.18 |  1.350 |          12.00 |    0.1% |      1.63 | `Cascade_9`

|               36.91 |       27,089,740.05 |    0.1% |           51.20 |           91.88 |  0.557 |           3.00 |    0.0% |      1.65 | `Gather_10`
|               64.52 |       15,500,193.22 |    0.1% |          205.27 |          160.43 |  1.279 |          13.00 |    0.1% |      1.64 | `Cascade_10`

|               36.94 |       27,073,770.67 |    0.1% |           51.20 |           91.95 |  0.557 |           3.00 |    0.0% |      1.65 | `Gather_11`
|               73.25 |       13,652,425.35 |    0.2% |          221.27 |          182.28 |  1.214 |          14.00 |    0.1% |      1.64 | `Cascade_11`

|               36.93 |       27,077,114.52 |    0.1% |           51.20 |           91.94 |  0.557 |           3.00 |    0.0% |      1.65 | `Gather_12`
|               81.55 |       12,261,888.56 |    0.1% |          237.27 |          203.05 |  1.168 |          15.00 |    0.1% |      1.66 | `Cascade_12`

|               37.00 |       27,030,647.09 |    0.1% |           51.20 |           92.10 |  0.556 |           3.00 |    0.0% |      1.65 | `Gather_13`
|               89.81 |       11,135,081.11 |    0.1% |          253.27 |          223.62 |  1.133 |          16.00 |    0.1% |      1.65 | `Cascade_13`

|               36.95 |       27,067,150.95 |    0.1% |           51.20 |           91.94 |  0.557 |           3.00 |    0.0% |      1.64 | `Gather_14`
|              100.13 |        9,987,438.18 |    0.1% |          269.27 |          249.22 |  1.080 |          17.00 |    0.1% |      1.65 | `Cascade_14`

|               36.93 |       27,078,996.88 |    0.1% |           51.20 |           91.95 |  0.557 |           3.00 |    0.0% |      1.65 | `Gather_15`
|              110.46 |        9,053,390.05 |    0.1% |          285.27 |          275.00 |  1.037 |          18.00 |    0.1% |      1.65 | `Cascade_15`


*/
