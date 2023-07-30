#include <bit>
#include <cassert>

#include "SwRast.h"

namespace swr {

static const int kLerpFracBits = 8;

Texture2D::Texture2D(uint32_t width, uint32_t height, uint32_t mipLevels) {
    assert(std::has_single_bit(width) && std::has_single_bit(height));

    Width = width;
    Height = height;
    Stride = width + 1;

    uint32_t nominalMips = std::bit_width(std::min(width, height));
    MipLevels = std::max(std::min(std::min(mipLevels, nominalMips), (uint32_t)VInt::Length), 1u);

    uint32_t offset = 0;

    for (uint32_t i = 0; i < MipLevels; i++) {
        _mipOffsets[i] = (int32_t)offset;
        offset += ((Height >> i) + 1) * ((Width >> i) + 1);
    }
    Data = std::make_unique<uint32_t[]>(offset);

    _scaleU = (float)width;
    _scaleV = (float)height;
    _maskU = width - 1;
    _maskV = height - 1;

    _maskLerpU = width * (1 << kLerpFracBits) - 1;
    _maskLerpV = height * (1 << kLerpFracBits) - 1;
    _scaleLerpU = (float)(_maskLerpU + 1);
    _scaleLerpV = (float)(_maskLerpV + 1);
}

void Texture2D::SetPixels(const uint32_t* pixels, uint32_t stride) {
    for (uint32_t y = 0; y <= Height; y++) {
        uint32_t srcY = y == Height ? y - 1 : y;
        auto srcRow = &pixels[srcY * stride];
        auto dstRow = &Data[y * Stride];

        std::memcpy(dstRow, srcRow, Width * 4);
        Data[Width + y * Stride] = pixels[Width - 1];
    }

    GenerateMips();
}

// 2x2 box downsampling
void Texture2D::GenerateMips() {
    for (uint32_t i = 0; i < MipLevels - 1; i++) {
        uint32_t* src = &Data[(size_t)_mipOffsets[i]];
        uint32_t* dest = &Data[(size_t)_mipOffsets[i + 1]];
        uint32_t stride = (Width >> i) + 1;
        uint32_t dstride = (stride / 2) + 1;
        uint32_t w = Width >> i, h = Height >> i;

        for (uint32_t y = 0; y < h; y += 2) {
            uint32_t* destRow = &dest[(y / 2) * dstride];

            for (uint32_t x = 0; x < w; x += 8) {
                auto rows = _mm256_avg_epu8(_mm256_loadu_si256((__m256i*)&src[x + (y + 0) * stride]),
                                            _mm256_loadu_si256((__m256i*)&src[x + (y + 1) * stride]));

                auto cols = _mm256_avg_epu8(rows, _mm256_bsrli_epi128(rows, 4));
                auto res = _mm256_permutevar8x32_epi32(cols, _mm256_setr_epi32(0, 2, 4, 6, -1, -1, -1, -1));

                _mm_storeu_si128((__m128i*)&destRow[x / 2], _mm256_castsi256_si128(res));
            }
            destRow[w / 2] = destRow[w / 2 - 1];
        }

        uint32_t* lastRow = &dest[(h / 2 - 1) * dstride];
        uint32_t* padRow = &dest[(h / 2) * dstride];
        std::memcpy(padRow, lastRow, dstride * 4);
    }
}

// Calculate coarse partial derivatives for a 4x4 tile.
// https://gamedev.stackexchange.com/a/130933
static VFloat dFdx(VFloat p) {
    auto a = _mm512_shuffle_ps(p, p, 0b10'10'00'00);  //[0 0 2 2]
    auto b = _mm512_shuffle_ps(p, p, 0b11'11'01'01);  //[1 1 3 3]
    return b - a;
}
static VFloat dFdy(VFloat p) {
    // auto a = _mm256_permute2x128_si256(p, p, 0b00'00'00'00);  // dupe lower 128 lanes
    // auto b = _mm256_permute2x128_si256(p, p, 0b01'01'01'01);  // dupe upper 128 lanes
    auto a = _mm512_shuffle_f32x4(p, p, 0b10'10'00'00);
    auto b = _mm512_shuffle_f32x4(p, p, 0b11'11'01'01);
    return b - a;
}

static VFloat CalcMipLevel(VFloat scaledU, VFloat scaledV) {
    VFloat dxu = dFdx(scaledU), dyu = dFdy(scaledU);
    VFloat dxv = dFdx(scaledV), dyv = dFdy(scaledV);

    VFloat maxDeltaSq = simd::max(simd::fma(dxu, dxu, dxv * dxv), simd::fma(dyu, dyu, dyv * dyv));
    return simd::approx_log2(maxDeltaSq) * 0.5f;
}

VFloat4 Texture2D::SampleNearest(VFloat u, VFloat v) const {
    VFloat su = u * _scaleU, sv = v * _scaleV;
    VInt ix = simd::round2i(su) & _maskU;
    VInt iy = simd::round2i(sv) & _maskV;
    VInt stride = VInt(Stride);

    VInt level = simd::trunc2i(CalcMipLevel(su, sv));

    if (_mm512_cmpgt_epi32_mask(level, VInt(0))) {
        level = simd::min(simd::max(level, 0), MipLevels - 1);

        ix = simd::shrl(ix, level) + VInt(_mm512_permutexvar_epi32(level, _mipOffsets.reg));
        iy = simd::shrl(iy, level);
        stride = simd::shrl(Width, level) + 1;
    }
    VInt colors = GatherPixels(ix + iy * stride);
    float scale = 1.0f / 255;

    return {
        simd::conv2f((colors >> 0) & 0xFF) * scale,
        simd::conv2f((colors >> 8) & 0xFF) * scale,
        simd::conv2f((colors >> 16) & 0xFF) * scale,
        simd::conv2f((colors >> 24) & 0xFF) * scale,
    };
}

VFloat4 Texture2D::SampleLinear(VFloat u, VFloat v) const {
    VInt ix = simd::round2i(u * _scaleLerpU) & _maskLerpU;
    VInt iy = simd::round2i(v * _scaleLerpV) & _maskLerpV;

    // 15-bit fraction for mulhrs
    VInt fracMask = (1 << kLerpFracBits) - 1;
    VInt fx = (ix & fracMask) << (15 - kLerpFracBits);
    VInt fy = (iy & fracMask) << (15 - kLerpFracBits);
    VInt fx2 = (fx << 16) | fx;
    VInt fy2 = (fy << 16) | fy;

    ix = ix >> kLerpFracBits;
    iy = iy >> kLerpFracBits;

    // Lerp 2 channels at the same time (RGBA -> R0B0, 0G0A)
    // Row 1
    VInt indices00 = ix + iy * Stride;
    VInt indices10 = indices00 + 1;
    VInt colors00 = GatherPixels(indices00);
    VInt colors10 = GatherPixels(indices10);
    VInt oddByteMask = 0x00FF'00FF;
    VInt rb00 = colors00 & oddByteMask;
    VInt rb10 = colors10 & oddByteMask;
    VInt ga00 = (colors00 >> 8) & oddByteMask;
    VInt ga10 = (colors10 >> 8) & oddByteMask;
    VInt rbRow1 = simd::lerp16(rb00, rb10, fx2);
    VInt gaRow1 = simd::lerp16(ga00, ga10, fx2);

    // Row 2
    VInt colors01 = GatherPixels(indices00 + Stride);
    VInt colors11 = GatherPixels(indices10 + Stride);
    VInt rb01 = colors01 & oddByteMask;
    VInt rb11 = colors11 & oddByteMask;
    VInt ga01 = (colors01 >> 8) & oddByteMask;
    VInt ga11 = (colors11 >> 8) & oddByteMask;
    VInt rbRow2 = simd::lerp16(rb01, rb11, fx2);
    VInt gaRow2 = simd::lerp16(ga01, ga11, fx2);

    // Columns
    VInt rbCol = simd::lerp16(rbRow1, rbRow2, fy2);
    VInt gaCol = simd::lerp16(gaRow1, gaRow2, fy2);

    float scale = 1.0f / 255;

    return {
        simd::conv2f((rbCol >> 0) & 0xFF) * scale,
        simd::conv2f((gaCol >> 0) & 0xFF) * scale,
        simd::conv2f((rbCol >> 16) & 0xFF) * scale,
        simd::conv2f((gaCol >> 16) & 0xFF) * scale,
    };
}

}; // namespace swr