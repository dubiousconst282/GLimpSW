#include <cstdint>
#include "../Texture.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

namespace simd = swr::simd;
using swr::v_int, swr::v_mask;

constexpr int LerpFracBits = 8, LerpFracMask = (1 << LerpFracBits) - 1;

[[gnu::noinline]] v_int SampleLinear_G32(swr::Texture2D<swr::pixfmt::RGBA8u>& tex, v_int ixf, v_int iyf, v_int offset, v_int stride, v_int mipLevel) {
    ixf = simd::max(ixf - (LerpFracMask / 2), 0);
    iyf = simd::max(iyf - (LerpFracMask / 2), 0);

    v_int ix = ixf >> LerpFracBits;
    v_int iy = iyf >> LerpFracBits;

    v_int indices00 = offset + ix + (iy << stride);
    v_int indices01 = indices00 + (((iy + 1) << mipLevel) < (int32_t)tex.Height ? 1 << stride : 0);
    v_int inboundX = ((ix + 1) << mipLevel) < (int32_t)tex.Width;

    v_int data00 = _mm512_i32gather_epi32(indices00, tex.Data + 0, 4);
    v_int data10 = _mm512_i32gather_epi32(indices00, tex.Data + 1, 4);
    v_int data01 = _mm512_i32gather_epi32(indices01, tex.Data + 0, 4);
    v_int data11 = _mm512_i32gather_epi32(indices01, tex.Data + 1, 4);

    // 15-bit fraction for mulhrs
    v_int fx = (ixf & LerpFracMask) << (15 - LerpFracBits);
    v_int fy = (iyf & LerpFracMask) << (15 - LerpFracBits);
    fx = (fx << 16) | fx;
    fy = (fy << 16) | fy;
    fx = inboundX ? fx : 0;

    // Lerp 2 channels at the same time (RGBA -> R0B0, 0G0A)
    v_int rbRow1 = simd::lerp16((data00 >> 0) & 0x00FF00FF, (data10 >> 0) & 0x00FF00FF, fx);
    v_int gaRow1 = simd::lerp16((data00 >> 8) & 0x00FF00FF, (data10 >> 8) & 0x00FF00FF, fx);
    v_int rbRow2 = simd::lerp16((data01 >> 0) & 0x00FF00FF, (data11 >> 0) & 0x00FF00FF, fx);
    v_int gaRow2 = simd::lerp16((data01 >> 8) & 0x00FF00FF, (data11 >> 8) & 0x00FF00FF, fx);

    // Columns
    v_int rbCol = simd::lerp16(rbRow1, rbRow2, fy);
    v_int gaCol = simd::lerp16(gaRow1, gaRow2, fy);

    return rbCol | (gaCol << 8);
}

[[gnu::noinline]] v_int SampleLinear_G64(swr::Texture2D<swr::pixfmt::RGBA8u>& tex, v_int ixf, v_int iyf, v_int offset, v_int stride,
                                          v_int mipLevel) {
    ixf = simd::max(ixf - (LerpFracMask / 2), 0);
    iyf = simd::max(iyf - (LerpFracMask / 2), 0);

    v_int ix = ixf >> LerpFracBits;
    v_int iy = iyf >> LerpFracBits;

    v_int indices00 = offset + ix + (iy << stride);
    v_int indices01 = indices00 + (((iy + 1) << mipLevel) < (int32_t)tex.Height ? 1 << stride : 0);
    v_int inboundX = ((ix + 1) << mipLevel) < (int32_t)tex.Width;

    v_int row0_lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices00, 0), tex.Data, 4);
    v_int row0_hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices00, 1), tex.Data, 4);
    v_int row1_lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices01, 0), tex.Data, 4);
    v_int row1_hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices01, 1), tex.Data, 4);

    v_int data00 = _mm512_permutex2var_epi32(row0_lo, simd::lane_idx * 2 + 0, row0_hi);
    v_int data10 = _mm512_permutex2var_epi32(row0_lo, simd::lane_idx * 2 + 1, row0_hi);
    v_int data01 = _mm512_permutex2var_epi32(row1_lo, simd::lane_idx * 2 + 0, row1_hi);
    v_int data11 = _mm512_permutex2var_epi32(row1_lo, simd::lane_idx * 2 + 1, row1_hi);

    // 15-bit fraction for mulhrs
    v_int fx = (ixf & LerpFracMask) << (15 - LerpFracBits);
    v_int fy = (iyf & LerpFracMask) << (15 - LerpFracBits);
    fx = (fx << 16) | fx;
    fy = (fy << 16) | fy;
    fx = inboundX ? fx : 0;

    // Lerp 2 channels at the same time (RGBA -> R0B0, 0G0A)
    v_int rbRow1 = simd::lerp16((data00 >> 0) & 0x00FF00FF, (data10 >> 0) & 0x00FF00FF, fx);
    v_int gaRow1 = simd::lerp16((data00 >> 8) & 0x00FF00FF, (data10 >> 8) & 0x00FF00FF, fx);
    v_int rbRow2 = simd::lerp16((data01 >> 0) & 0x00FF00FF, (data11 >> 0) & 0x00FF00FF, fx);
    v_int gaRow2 = simd::lerp16((data01 >> 8) & 0x00FF00FF, (data11 >> 8) & 0x00FF00FF, fx);

    v_int rbCol = simd::lerp16(rbRow1, rbRow2, fy);
    v_int gaCol = simd::lerp16(gaRow1, gaRow2, fy);

    return rbCol | (gaCol << 8);
}

int main() {
    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.epochs(30);

    auto texture = swr::CreateTexture2D<swr::pixfmt::RGBA8u>(4096, 4096, 1, 1);
    for (uint32_t y = 0; y < texture->Height; y += 4) {
        for (uint32_t x = 0; x < texture->Width; x += 4) {
            texture->WriteTile((simd::lane_idx * 0x505050) | int(0xFF000000), x, y);
        }
    }

    ctx.batch(texture->Width * texture->Height);
    ctx.relative(true);

    ctx.run("Gather32", [&]() {
        for (int y = 0; y < texture->Height; y += 4) {
            for (int x = 0; x < texture->Width; x += 4) {
                v_int tx = ((x + simd::FragPixelOffsetsX) << LerpFracBits) + (LerpFracMask/2);
                v_int ty = ((y + simd::FragPixelOffsetsY) << LerpFracBits) + (LerpFracMask/2);
                auto res = SampleLinear_G32(*texture, tx, ty, 0, (int)texture->RowShift, 0);
                ctx.doNotOptimizeAway(res);
            }
        }
    });
    ctx.run("Gather64", [&]() {
        for (int y = 0; y < texture->Height; y += 4) {
            for (int x = 0; x < texture->Width; x += 4) {
                v_int tx = ((x + simd::FragPixelOffsetsX) << LerpFracBits) + (LerpFracMask / 2);
                v_int ty = ((y + simd::FragPixelOffsetsY) << LerpFracBits) + (LerpFracMask / 2);
                auto res = SampleLinear_G64(*texture, tx, ty, 0, (int)texture->RowShift, 0);
                ctx.doNotOptimizeAway(res);
            }
        }
    });
}

/*
TigerLake, mitigations=off
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|                1.05 |      955,441,466.75 |    0.2% |            5.38 |            2.61 |  2.063 |           0.19 |    0.0% |      1.58 | `Gather32`
|                0.77 |    1,293,362,045.65 |    0.1% |            5.94 |            1.92 |  3.087 |           0.19 |    0.0% |      1.60 | `Gather64`
*/