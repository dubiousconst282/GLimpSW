#include <bit>
#include <cassert>

#include "SwRast.h"

namespace swr {

static const int kLerpFracBits = 8;

Texture2D::Texture2D(uint32_t width, uint32_t height, uint32_t mipLevels) {
    assert(std::has_single_bit(width) && std::has_single_bit(height));

    Width = width;
    Height = height;
    StrideLog2 = (uint32_t)std::countr_zero(width);

    uint32_t nominalMips = (uint32_t)std::bit_width(std::min(width, height));
    MipLevels = std::max(std::min(std::min(mipLevels, nominalMips), VInt::Length), 1u);

    uint32_t offset = 0;

    for (uint32_t i = 0; i < MipLevels; i++) {
        _mipOffsets[i] = (int32_t)offset;
        offset += (Height >> i) * (Width >> i);
    }
    Data = std::make_unique<uint32_t[]>(offset + 16); //add a bit of padding because the mipmap downsampler works with 8x2 blocks

    _scaleU = (float)width;
    _scaleV = (float)height;
    _maskU = (int32_t)width - 1;
    _maskV = (int32_t)height - 1;

    _maskLerpU = (int32_t)(width << kLerpFracBits) - 1;
    _maskLerpV = (int32_t)(height << kLerpFracBits) - 1;
    _scaleLerpU = (float)(_maskLerpU + 1);
    _scaleLerpV = (float)(_maskLerpV + 1);
}

void Texture2D::SetPixels(const uint32_t* pixels, uint32_t stride) {
    for (uint32_t y = 0; y < Height; y++) {
        std::memcpy(&Data[y * Width], &pixels[y * stride], Width * 4);
    }
    GenerateMips();
}

// 2x2 box downsampling
void Texture2D::GenerateMips() {
    for (uint32_t i = 0; i < MipLevels - 1; i++) {
        uint32_t* src = &Data[(size_t)_mipOffsets[i]];
        uint32_t* dst = &Data[(size_t)_mipOffsets[i + 1]];
        uint32_t w = Width >> i, h = Height >> i;

        for (uint32_t y = 0; y < h; y += 2) {
            for (uint32_t x = 0; x < w; x += 8) {
                auto rows = _mm256_avg_epu8(_mm256_loadu_si256((__m256i*)&src[x + (y + 0) * w]),
                                            _mm256_loadu_si256((__m256i*)&src[x + (y + 1) * w]));

                auto cols = _mm256_avg_epu8(rows, _mm256_bsrli_epi128(rows, 4));
                auto res = _mm256_permutevar8x32_epi32(cols, _mm256_setr_epi32(0, 2, 4, 6, -1, -1, -1, -1));

                _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(res));
                dst += 4;
            }
        }
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

// TODO: experiment with tiling/swizzling - https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/

VFloat4 Texture2D::SampleNearest(VFloat u, VFloat v) const {
    VFloat su = u * _scaleU, sv = v * _scaleV;
    VInt ix = simd::round2i(su) & _maskU;
    VInt iy = simd::round2i(sv) & _maskV;

    VInt level = simd::trunc2i(CalcMipLevel(su, sv));
    VInt stride = (int32_t)StrideLog2;

    if (_mm512_cmpgt_epi32_mask(level, VInt(0))) {
        level = simd::min(simd::max(level, 0), (int32_t)MipLevels - 1);

        ix = simd::shrl(ix, level) + VInt(_mm512_permutexvar_epi32(level, _mipOffsets.reg));
        iy = simd::shrl(iy, level);
        stride -= level;
    }
    VInt colors = GatherPixels(ix + (iy << stride));
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
    VInt indices00 = ix + (iy << StrideLog2);
    // indices10 = indices00 + (ix < Width - 1)
    //           = indices00 - (ix < Width - 1 ? -1 : 0)
    VInt indices10 = _mm512_sub_epi32(indices00, _mm512_movm_epi32(_mm512_cmplt_epi32_mask(ix, _mm512_set1_epi32((int32_t)Width - 1))));
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
    VInt colors01 = GatherPixels(indices00 + (int32_t)Width);
    VInt colors11 = GatherPixels(indices10 + (int32_t)Width);
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

VFloat4 Texture2D::SampleHybrid(VFloat u, VFloat v) const {
    VFloat level = CalcMipLevel(u * _scaleU, v * _scaleV);

    if (_mm512_cmp_ps_mask(level, VFloat(1.25f), _CMP_GT_OQ)) [[likely]] {
        return SampleNearest(u, v);
    }
    return SampleLinear(u, v);
}

HdrTexture2D::HdrTexture2D(uint32_t width, uint32_t height, uint32_t numLayers) {
    assert(std::has_single_bit(width) && std::has_single_bit(height));

    Width = width;
    Height = height;
    StrideLog2 = (uint32_t)std::countr_zero(width);
    NumLayers = numLayers;

    Data = std::make_unique<uint32_t[]>(numLayers * (width << StrideLog2));

    _scaleU = (float)width;
    _scaleV = (float)height;
    _maskU = (int32_t)width - 1;
    _maskV = (int32_t)height - 1;
}

void HdrTexture2D::SetPixels(const float* pixels, uint32_t stride, uint32_t layer) {
    uint32_t offset = layer * (Height << StrideLog2);

    const auto PackFloat = [](float x, uint32_t fracBits) -> uint32_t {
        uint32_t bits = *(uint32_t*)&x;
        int32_t exp = (bits >> 23 & 255) - 127;
        uint32_t frc = bits & ((1u << 23) - 1);

        exp += 15;
        frc >>= (23 - fracBits);

        if (exp < 0) return 0;

        if (exp > 31) {
            exp = 31;
            frc = 0;
        }
        return (uint32_t)exp << fracBits | frc;
    };

    for (uint32_t y = 0; y < Height; y++) {
        for (uint32_t x = 0; x < Width; x++) {
            uint32_t v = PackFloat(pixels[(x + y * stride) * 3 + 0], 6) << 21 | 
                         PackFloat(pixels[(x + y * stride) * 3 + 1], 6) << 10 | 
                         PackFloat(pixels[(x + y * stride) * 3 + 2], 5) << 0;

            Data[offset + x + (y << StrideLog2)] = v;
        }
    }
}

// 5-bit exp, 6-bit mantissa
inline VFloat UnpackF11(VInt x) {
    // int exp = (x >> 6) & 0x1F;
    // int frc = x & 0x3F;
    // return (exp - 15 + 127) << 23 | frc << (23 - 6);

    x = x << 17;
    return simd::re2f((x & 0x0FFE0000) + 0x38000000);
}
// 5-bit exp, 5-bit mantissa
inline VFloat UnpackF10(VInt x) {
    // int exp = (x >> 5) & 0x1F;
    // int frc = x & 0x1F;
    // return (exp - 15 + 127) << 23 | frc << (23 - 5);

    x = x << 18;
    return simd::re2f((x & 0x0FFC0000) + 0x38000000);
}

VFloat3 HdrTexture2D::SampleNearest(VFloat u, VFloat v, VInt layer) const {
    VFloat su = u * _scaleU, sv = v * _scaleV;
    VInt ix = simd::round2i(su) & _maskU;
    VInt iy = simd::round2i(sv) & _maskV;

    VInt colors = VInt::gather<4>((int32_t*)Data.get(), ix + (iy << StrideLog2));

    return {
        UnpackF11((colors >> 21) & 0x7FF),
        UnpackF11((colors >> 10) & 0x7FF),
        UnpackF10((colors >> 0) & 0x3FF),
    };
}

// https://gamedev.stackexchange.com/questions/183452/how-does-cube-mapping-work
void HdrTexture2D::ProjectCubemap(VFloat3 dir, VFloat& u, VFloat& v, VInt& faceIdx) {
    // Find axis with max magnitude
    auto w = dir.x.reg;

    uint16_t wy = _mm512_cmp_ps_mask(_mm512_abs_ps(dir.y), _mm512_abs_ps(w), _CMP_GT_OQ);
    w = _mm512_mask_mov_ps(w, wy, dir.y);

    uint16_t wz = _mm512_cmp_ps_mask(_mm512_abs_ps(dir.z), _mm512_abs_ps(w), _CMP_GT_OQ);
    w = _mm512_mask_mov_ps(w, wz, dir.z);

    // uv = { x: yz,  y: xz,  z: xy }[w]
    u = _mm512_mask_mov_ps(dir.y, wy, dir.x);  // dir.y == w ? dir.x : dir.y;
    v = _mm512_mask_mov_ps(dir.z, wz, dir.y);  // dir.z == w ? dir.y : dir.z;

    // faceIdx = wz ? 2+2 : (wy ? 2 : 0)
    faceIdx = _mm512_set1_epi32(2);
    faceIdx = _mm512_mask_add_epi32(_mm512_maskz_mov_epi32(wy, faceIdx), wz, faceIdx, faceIdx);

    // faceIdx += w < 0 ? 1 : 0
    faceIdx += _mm512_srli_epi32(_mm512_castps_si512(w), 31);
}

}; // namespace swr