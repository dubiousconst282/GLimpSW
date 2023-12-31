#pragma once

#include <concepts>
#include <functional>

#include "SIMD.h"

namespace swr {

namespace pixfmt {

template<typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

// Represents a SIMD pixel packet in storage form, where the unpacked form consists of floats.
// NOTE: Packed size is currently restricted to 32-bits as that's the only element size texture sampling supports gathering.
template<typename T>
concept Texel = requires(const T& s, const T::UnpackedTy& u) {
    IsAnyOf<typename T::UnpackedTy, VInt, VFloat, VFloat2, VFloat3, VFloat4>;
    IsAnyOf<typename T::LerpedTy, VInt, typename T::UnpackedTy>;
    { s.Unpack() } -> std::same_as<typename T::UnpackedTy>;
    { T::Pack(u) } -> std::same_as<T>;
};

// RGBA x 8-bit unorm
struct RGBA8u {
    using UnpackedTy = VFloat4;
    using LerpedTy = VInt;

    VInt Packed;

    VFloat4 Unpack() const { return simd::UnpackRGBA(Packed); }
    static RGBA8u Pack(const VFloat4& value) { return { simd::PackRGBA(value) }; }
};

// R x 32-bit float
struct R32f {
    using UnpackedTy = VFloat;
    using LerpedTy = UnpackedTy;

    VFloat Packed;

    VFloat Unpack() const { return Packed; }
    static R32f Pack(const VFloat& value) { return { value }; }
};

// RG x 16-bit float
struct RG16f {
    using UnpackedTy = VFloat2;
    using LerpedTy = UnpackedTy;

    VInt Packed;

    UnpackedTy Unpack() const {
        return {
            _mm512_cvtph_ps(_mm512_cvtepi32_epi16(Packed)),
            _mm512_cvtph_ps(_mm512_cvtepi32_epi16(Packed >> 16)),
        };
    }
    static RG16f Pack(const UnpackedTy& value) {
        VInt r = _mm512_cvtepi16_epi32(_mm512_cvtps_ph(value.x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        VInt g = _mm512_cvtepi16_epi32(_mm512_cvtps_ph(value.y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        return { r | (g << 16) };
    }
};

// RG x 11-bit float, B x 10-bit float
struct R11G11B10f {
    using UnpackedTy = VFloat3;
    using LerpedTy = UnpackedTy;

    VInt Packed;

    UnpackedTy Unpack() const {
        return {
            UnpackF11(Packed >> 21),
            UnpackF11(Packed >> 10),
            UnpackF10(Packed >> 0),
        };
    }
    static R11G11B10f Pack(const UnpackedTy& value) {
        return {
            PackF11(value.x) << 21 |
            PackF11(value.y) << 10 |
            PackF10(value.z),
        };
    }

private:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules#11-bit-and-10-bit-floating-point-rules
    // Note that these don't follow denorm/NaN/Inf rules. Only clamping is applied.
    // This code was based on clang's output for the scalar version, as it doesn't seem to optimize the SIMD version well.
    static VInt PackF11(VFloat x) {
        x = simd::max(x, 1.0f / (1 << 15));
        x = simd::min(x, 130048.0f); // UnpackF11(2047)

        return (simd::re2i(x) >> 17 & 0x3FFF) - 0x1C00;
    }
    static VInt PackF10(VFloat x) {
        x = simd::max(x, 1.0f / (1 << 15));
        x = simd::min(x, 129024.0f);  // UnpackF10(1023)

        return (simd::re2i(x) >> 18 & 0x1FFF) - 0x0E00;
    }

    // 5-bit exp, 6-bit mantissa
    static VFloat UnpackF11(VInt x) {
        // int exp = (x >> 6) & 0x1F;
        // int frc = x & 0x3F;
        // return (exp - 15 + 127) << 23 | frc << (23 - 6);

        x = x << 17;
        return simd::re2f((x & 0x0FFE0000) + 0x38000000);
    }
    // 5-bit exp, 5-bit mantissa
    static VFloat UnpackF10(VInt x) {
        x = x << 18;
        return simd::re2f((x & 0x0FFC0000) + 0x38000000);
    }

    // static uint32_t PackFloat(float x, uint32_t fracBits) {
    //     uint32_t bits = *(uint32_t*)&x;
    //     int32_t exp = (bits >> 23 & 255) - 127;
    //     uint32_t frc = bits & ((1u << 23) - 1);

    //     exp += 15;
    //     frc >>= (23 - fracBits);

    //     if (exp < 0) return 0;

    //     if (exp > 31) {
    //         exp = 31;
    //         frc = 0;
    //     }
    //     return (uint32_t)exp << fracBits | frc;
    // }
};

};  // namespace pixfmt

template<pixfmt::Texel Texel>
struct Texture2D;

using RgbaTexture2D = Texture2D<pixfmt::RGBA8u>;
using HdrTexture2D = Texture2D<pixfmt::R11G11B10f>;

struct StbImage {
    enum class PixelType { Empty, RGBA_U8, RGB_F32 };

    uint32_t Width, Height;
    PixelType Type;
    std::unique_ptr<uint8_t[], decltype(&std::free)> Data = { nullptr, &std::free };

    static StbImage Load(std::string_view path, PixelType type = PixelType::RGBA_U8);
};

namespace texutil {

RgbaTexture2D LoadImage(std::string_view path, uint32_t mipLevels = 8);

HdrTexture2D LoadImageHDR(std::string_view path, uint32_t mipLevels = 8);

// Loads a equirectangular panorama into a cubemap.
HdrTexture2D LoadCubemapFromPanoramaHDR(std::string_view path, uint32_t mipLevels = 8);

// Iterates over the given rect in 4x4 tile steps. Visitor takes normalized UVs centered around pixel center.
inline void IterateTiles(uint32_t width, uint32_t height, std::function<void(uint32_t, uint32_t, VFloat, VFloat)> visitor) {
    assert(width % 4 == 0 && height % 4 == 0);

    for (int32_t y = 0; y < height; y += 4) {
        for (int32_t x = 0; x < width; x += 4) {
            VFloat u = simd::conv2f(x + FragPixelOffsetsX) + 0.5f;
            VFloat v = simd::conv2f(y + FragPixelOffsetsY) + 0.5f;
            visitor((uint32_t)x, (uint32_t)y, u * (1.0f / width), v * (1.0f / height));
        }
    }
}

// Calculates mip-level for a 4x4 fragment using the partial derivatives of the given scaled UVs.
inline VInt CalcMipLevel(VFloat scaledU, VFloat scaledV) {
    VFloat dxu = simd::dFdx(scaledU), dyu = simd::dFdy(scaledU);
    VFloat dxv = simd::dFdx(scaledV), dyv = simd::dFdy(scaledV);

    VFloat maxDeltaSq = simd::max(simd::fma(dxu, dxu, dxv * dxv), simd::fma(dyu, dyu, dyv * dyv));
    return simd::ilog2(maxDeltaSq) >> 1;
    // return simd::approx_log2(maxDeltaSq) * 0.5f;
}

// Projects the given direction vector (may be unnormalized) to cubemap face UV and layer.
// This produces slightly different results from other graphics APIs, UVs are not flipped depending on faces.
inline void ProjectCubemap(VFloat3 dir, VFloat& u, VFloat& v, VInt& faceIdx) {
    // https://gamedev.stackexchange.com/a/183463
    // https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing
    // Find axis with max magnitude
    VFloat w = dir.x;

    VMask wy = simd::abs(dir.y) > simd::abs(w);
    w = simd::csel(wy, dir.y, w);

    VMask wz = simd::abs(dir.z) > simd::abs(w);
    w = simd::csel(wz, dir.z, w);

    VMask wx = wy | wz;  // negated
    wy &= ~wz;

    // faceIdx = wz ? 4 : (wy ? 2 : 0)
    faceIdx = simd::csel(wz, 4, simd::csel(wy, VInt(2), 0));
    // faceIdx += w < 0 ? 1 : 0
    faceIdx += simd::shrl(simd::re2i(w), 31);

    // uv = { x: zy,  y: xz,  z: xy }[w]
    w = simd::rcp14(simd::abs(w)) * 0.5f;
    u = simd::csel(wx, dir.x, dir.z) * w + 0.5f;
    v = simd::csel(wy, dir.z, dir.y) * w + 0.5f;
}

// Unprojects the given cubemap face index and UVs to a normalized direction vector.
inline VFloat3 UnprojectCubemap(VFloat u, VFloat v, VInt faceIdx) {
    VFloat w = 1.0f | simd::re2f(faceIdx << 31);  //(faceIdx & 1) ? -1.0 : +1.0
    VInt axis = faceIdx >> 1;

    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // 0:  1,  v,  u
    // 1: -1,  v,  u
    // 2:  u,  1,  v
    // 3:  u, -1,  v
    // 4:  u,  v,  1
    // 5:  u,  v, -1
    VFloat3 unnormDir = {
        simd::csel(axis == 0, w, u),
        simd::csel(axis == 1, w, v),
        simd::csel(axis == 2, w, simd::csel(axis == 0, u, v)),
    };
    return unnormDir * simd::rsqrt14(u * u + v * v + 1.0f);
}

// Lookups the adjacent cube face and UVs to the nearest edge.
inline void GetAdjacentCubeFace(VInt& faceIdx, VInt& u, VInt& v, VInt scaleU, VInt scaleV) {
    static const uint8_t AdjFaceLUT[4][8] = {
        { 0x1b, 0x0b, 0x25, 0x05, 0x23, 0x03 },  //
        { 0x0a, 0x1a, 0x04, 0x24, 0x02, 0x22 },  //
        { 0x15, 0x05, 0x29, 0x09, 0x11, 0x01 },  //
        { 0x04, 0x14, 0x08, 0x28, 0x00, 0x10 },  //
    };
    // int q = abs(u - 0.5) > abs(v - 0.5) ? ((u > 0.5) ? 3 : 2) : ((v > 0.5) ? 1 : 0);
    // uint8_t data = AdjFace[q][face];
    // face = data & 7;
    // if (data & (1 << 3)) std::swap(u, v);
    // if (data & (1 << 4)) u = 1 - u;
    // if (data & (1 << 5)) v = 1 - v;

    VInt cu = (scaleU >> 1) - u, cv = (scaleV >> 1) - v;
    VInt quadIdx = simd::csel(simd::abs(cu) > simd::abs(cv), simd::shrl(cu, 31) + 2, simd::shrl(cv, 31));
    VInt tableIdx = quadIdx * 8 + faceIdx;

    VInt data = _mm512_permutexvar_epi8(tableIdx, _mm512_broadcast_i32x8(_mm256_loadu_epi8(AdjFaceLUT)));

    faceIdx = data & 7;

    VMask swap = (data >> 3 & 1) != 0;
    VMask invU = (data >> 4 & 1) != 0;
    VMask invV = (data >> 5 & 1) != 0;

    VInt su = simd::csel(swap, v, u);
    VInt sv = simd::csel(swap, u, v);
    u = simd::csel(invU, scaleU - su, su);
    v = simd::csel(invV, scaleV - sv, sv);
}

// Texture swizzling doesn't improve performance by much, the functions below are keept for reference.

// 32-bit Z-curve/morton encode. Takes ~8.5 cycles per 16 coord pairs on TigerLake, according to llvm-mca.
// - https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/
// - https://github.com/KWillets/simd_interleave/blob/master/simd.c
inline VInt Interleave(VInt x, VInt y) {
    const __m512i m0 = _mm512_broadcast_i32x4(_mm_set_epi8(85, 84, 81, 80, 69, 68, 65, 64, 21, 20, 17, 16, 5, 4, 1, 0));
    const __m512i m1 = _mm512_slli_epi64(m0, 1);
    const __m512i bm = _mm512_set_epi8(125, 61, 124, 60, 121, 57, 120, 56, 117, 53, 116, 52, 113, 49, 112, 48, 109, 45, 108, 44, 105, 41,
                                       104, 40, 101, 37, 100, 36, 97, 33, 96, 32, 93, 29, 92, 28, 89, 25, 88, 24, 85, 21, 84, 20, 81, 17,
                                       80, 16, 77, 13, 76, 12, 73, 9, 72, 8, 69, 5, 68, 4, 65, 1, 64, 0);

    __m512i xl = _mm512_shuffle_epi8(m0, (x >> 0) & 0x0F'0F'0F'0F);
    __m512i xh = _mm512_shuffle_epi8(m0, (x >> 4) & 0x0F'0F'0F'0F);
    __m512i yl = _mm512_shuffle_epi8(m1, (y >> 0) & 0x0F'0F'0F'0F);
    __m512i yh = _mm512_shuffle_epi8(m1, (y >> 4) & 0x0F'0F'0F'0F);

    __m512i lo = _mm512_or_si512(xl, yl);
    __m512i hi = _mm512_or_si512(xh, yh);

    return _mm512_permutex2var_epi8(lo, bm, hi);
}
inline VInt GetTiledOffset(VInt ix, VInt iy, VInt rowShift) {
    VInt tileId = (ix >> 2) + ((iy >> 2) << (rowShift - 2));
    VInt pixelOffset = (ix & 3) + (iy & 3) * 4;
    return tileId * 16 + pixelOffset;
}

};  // namespace texutil

enum class WrapMode { Repeat, ClampToEdge };
enum class FilterMode { Nearest, Linear };

// Texture sampler parameters
struct SamplerDesc {
    WrapMode Wrap = WrapMode::Repeat;
    FilterMode MagFilter = FilterMode::Linear;
    FilterMode MinFilter = FilterMode::Nearest;
    bool EnableMips = true;
};

template<pixfmt::Texel Texel>
struct Texture2D {
    static_assert(sizeof(Texel) == sizeof(VInt));

    static const int LerpFracBits = 8;  // Number of fractional bits in pixel coords for bilinear interpolation
    static const int LerpFracMask = (1 << LerpFracBits) - 1;

    uint32_t Width, Height, MipLevels, NumLayers;
    uint32_t RowShift, LayerShift;  // Shift amount to get row offset from Y coord / layer offset. Used to avoid expansive i32 vector mul.

    // Indexing: (layer << LayerShift) + _mipOffsets[mipLevel] + (ix >> mipLevel) + (iy >> mipLevel) << RowShift
    AlignedBuffer<uint32_t> Data;

    Texture2D(uint32_t width, uint32_t height, uint32_t mipLevels, uint32_t numLayers) {
        assert(std::has_single_bit(width) && std::has_single_bit(height));

        Width = width;
        Height = height;
        RowShift = (uint32_t)std::countr_zero(width);
        NumLayers = numLayers;

        MipLevels = 0;
        uint32_t layerSize = 0;

        for (; MipLevels < std::min(mipLevels, VInt::Length); MipLevels++) {
            uint32_t i = MipLevels;
            if ((Width >> i) < 4 || (Height >> i) < 4) break;

            _mipOffsets[i] = (int32_t)layerSize;
            layerSize += (Width >> i) * (Height >> i);
            layerSize = (layerSize + 15) & ~15u; // align to 64 bytes
        }

        if (numLayers > 1) {
            LayerShift = (uint32_t)std::bit_width(layerSize);
            layerSize = 1u << LayerShift; // TODO: this wastes quite a bit of memory

            assert(layerSize * (uint64_t)numLayers < UINT_MAX);
        }

        Data = alloc_buffer<uint32_t>(layerSize * numLayers + 16);

        _scaleU = (float)width;
        _scaleV = (float)height;
        _maskU = (int32_t)width - 1;
        _maskV = (int32_t)height - 1;

        _maskLerpU = (int32_t)(width << LerpFracBits) - 1;
        _maskLerpV = (int32_t)(height << LerpFracBits) - 1;
        _scaleLerpU = (float)(_maskLerpU + 1);
        _scaleLerpV = (float)(_maskLerpV + 1);
    }

    // Writes raw packed pixels (with matching formats) to the texture buffer.
    void SetPixels(const void* pixels, uint32_t stride, uint32_t layer) {
        assert(layer < NumLayers);

        for (uint32_t y = 0; y < Height; y++) {
            std::memcpy(&Data[(layer << LayerShift) + (y << RowShift)], (uint32_t*)pixels + y * stride, Width * 4);
        }
    }

    // Writes a 4x4 tile of texels to the texture buffer. Coords are in pixel space.
    void WriteTile(Texel value, uint32_t x, uint32_t y, uint32_t layer = 0, uint32_t mipLevel = 0) {
        assert(x + 3 < Width && y + 3 < Height);
        assert(x % 4 == 0 && y % 4 == 0);
        assert(layer < NumLayers && mipLevel < MipLevels);
        
        uint32_t* dst = &Data[(layer << LayerShift) + (uint32_t)_mipOffsets[mipLevel]];
        uint32_t stride = RowShift - mipLevel;

        __m512i packed = value.Packed;
        _mm_storeu_epi32(&dst[x + ((y + 0) << stride)], _mm512_extracti32x4_epi32(packed, 0));
        _mm_storeu_epi32(&dst[x + ((y + 1) << stride)], _mm512_extracti32x4_epi32(packed, 1));
        _mm_storeu_epi32(&dst[x + ((y + 2) << stride)], _mm512_extracti32x4_epi32(packed, 2));
        _mm_storeu_epi32(&dst[x + ((y + 3) << stride)], _mm512_extracti32x4_epi32(packed, 3));
    }

    void GenerateMips() {
        for (uint32_t layer = 0; layer < NumLayers; layer++) {
            for (uint32_t level = 1; level < MipLevels; level++) {
                GenerateMip(level, layer);
            }
        }
    }

    // Sample() functions are almost always called from very register-pressured places, and so
    // there will be massive spills across the call boundaries if they're not inlined...
    // Not sure if it's a good idea to always_inline everything (instr cache clobbering and such) but let's do it for now...

    template<SamplerDesc SD, bool CalcMipFromUVDerivs_ = true, bool IsCubeSample_ = false>
    [[gnu::pure, gnu::always_inline]] Texel::LerpedTy Sample(VFloat u, VFloat v, VInt layer = 0, VInt mipLevel = 0) const {
        // Scale and round UVs 
        VFloat su = u * _scaleLerpU, sv = v * _scaleLerpV;
        VInt ix = simd::round2i(su), iy = simd::round2i(sv);

        // Wrap
        if constexpr (SD.Wrap == WrapMode::ClampToEdge || IsCubeSample_) {
            ix = simd::min(simd::max(ix, 0), _maskLerpU);
            iy = simd::min(simd::max(iy, 0), _maskLerpV);
        } else {
            static_assert(SD.Wrap == WrapMode::Repeat);
            ix = ix & _maskLerpU;
            iy = iy & _maskLerpV;
        }

        // Select mip and filter mode
        if constexpr (CalcMipFromUVDerivs_) {
            mipLevel = texutil::CalcMipLevel(su, sv) - LerpFracBits;
        }

        FilterMode filter = simd::any(mipLevel > 0) ? SD.MinFilter : SD.MagFilter;
        mipLevel = SD.EnableMips ? simd::min(simd::max(mipLevel, 0), (int32_t)MipLevels - 1) : 0;
        
        // Calculate offsets and mip position
        VInt stride = (int32_t)RowShift;
        VInt offset = layer << LayerShift;

        if (simd::any(mipLevel > 0)) {
            ix = ix >> mipLevel;
            iy = iy >> mipLevel;
            stride -= mipLevel;
            offset += _mm512_permutexvar_epi32(mipLevel, _mipOffsets);
        }

        // Sample
        if (filter == FilterMode::Nearest) [[likely]] {
            ix = ix >> LerpFracBits;
            iy = iy >> LerpFracBits;
            Texel res = GatherTexels<Texel>(offset + ix + (iy << stride));

            if constexpr (std::is_same<typename Texel::LerpedTy, VInt>()) {
                return res.Packed;
            } else {
                return res.Unpack();
            }
        }
        if constexpr (IsCubeSample_) {
            //    x < 1 || x >= N
            // =  (x-1) >= (N-1)     given twos-complement + unsigned cmp
            VMask edgeU = _mm512_cmpge_epu32_mask((ix >> LerpFracBits) - 1, (_maskU >> mipLevel) - 1);
            VMask edgeV = _mm512_cmpge_epu32_mask((iy >> LerpFracBits) - 1, (_maskV >> mipLevel) - 1);

            if (simd::any(edgeU | edgeV)) [[unlikely]] {
                return SampleLinearNearCubeEdge(ix, iy, offset, stride, mipLevel, layer);
            }
        }
        return SampleLinear(ix, iy, offset, stride, mipLevel);
    }

    template<SamplerDesc SD>
    [[gnu::pure, gnu::always_inline]] Texel::LerpedTy SampleCube(const VFloat3& dir, VFloat mipLevel = -1.0f) const {
        VFloat u, v;
        VInt faceIdx;
        texutil::ProjectCubemap(dir, u, v, faceIdx);

        int tmp = mipLevel[0];  // __builtin_constant_p() only works with variables - https://github.com/llvm/llvm-project/issues/65741
        if (__builtin_constant_p(tmp) && tmp < 0) {
            return Sample<SD, true, true>(u, v, faceIdx);
        }
        VInt baseMip = simd::trunc2i(mipLevel);
        auto baseSample = Sample<SD, false, true>(u, v, faceIdx, baseMip);

        VFloat mipFrac = mipLevel - simd::conv2f(baseMip);
        if (simd::any(mipFrac > 0.0f) && simd::any(baseMip < (int32_t)(MipLevels - 1))) {
            auto lowerSample = Sample<SD, false, true>(u, v, faceIdx, baseMip + 1);
            return baseSample + (lowerSample - baseSample) * mipFrac;
        }
        return baseSample;
    }

private:
    VInt _mipOffsets;
    float _scaleU, _scaleV, _scaleLerpU, _scaleLerpV;
    int32_t _maskU, _maskV, _maskLerpU, _maskLerpV;

    // Interpolates texels overlapping the specified pixel coords (in N.LerpFracBits fixed-point). No bounds check.
    [[gnu::pure, gnu::always_inline]] Texel::LerpedTy SampleLinear(VInt ixf, VInt iyf, VInt offset, VInt stride, VInt mipLevel) const {
        ixf = simd::max(ixf - (LerpFracMask / 2), 0);
        iyf = simd::max(iyf - (LerpFracMask / 2), 0);

        VInt ix = ixf >> LerpFracBits;
        VInt iy = iyf >> LerpFracBits;

        VInt indices00 = offset + ix + (iy << stride);
        VInt indices10 = indices00 + simd::csel(((ix + 1) << mipLevel) < (int32_t)Width, VInt(1), 0);
        VInt rowOffset = simd::csel(((iy + 1) << mipLevel) < (int32_t)Height, VInt(1) << stride, 0);

        if constexpr (std::is_same<Texel, pixfmt::RGBA8u>()) {
            // 15-bit fraction for mulhrs
            VInt fx = (ixf & LerpFracMask) << (15 - LerpFracBits);
            VInt fy = (iyf & LerpFracMask) << (15 - LerpFracBits);
            fx = (fx << 16) | fx;
            fy = (fy << 16) | fy;

            // Lerp 2 channels at the same time (RGBA -> R0B0, 0G0A)
            // Row 1
            VInt colors00 = GatherTexels<VInt>(indices00);
            VInt colors10 = GatherTexels<VInt>(indices10);
            VInt rbRow1 = simd::lerp16((colors00 >> 0) & 0x00FF00FF, (colors10 >> 0) & 0x00FF00FF, fx);
            VInt gaRow1 = simd::lerp16((colors00 >> 8) & 0x00FF00FF, (colors10 >> 8) & 0x00FF00FF, fx);

            // Row 2
            VInt colors01 = GatherTexels<VInt>(indices00 + rowOffset);
            VInt colors11 = GatherTexels<VInt>(indices10 + rowOffset);
            VInt rbRow2 = simd::lerp16((colors01 >> 0) & 0x00FF00FF, (colors11 >> 0) & 0x00FF00FF, fx);
            VInt gaRow2 = simd::lerp16((colors01 >> 8) & 0x00FF00FF, (colors11 >> 8) & 0x00FF00FF, fx);

            // Columns
            VInt rbCol = simd::lerp16(rbRow1, rbRow2, fy);
            VInt gaCol = simd::lerp16(gaRow1, gaRow2, fy);

            return rbCol | (gaCol << 8);
        } else {
            using R = Texel::UnpackedTy;

            const float fracScale = 1.0f / (LerpFracMask + 1);
            VFloat fx = simd::conv2f(ixf & LerpFracMask) * fracScale;
            VFloat fy = simd::conv2f(iyf & LerpFracMask) * fracScale;

            R colors00 = GatherTexels<Texel>(indices00).Unpack();
            R colors10 = GatherTexels<Texel>(indices10).Unpack();
            R rowA = colors00 + (colors10 - colors00) * fx;

            R colors01 = GatherTexels<Texel>(indices00 + rowOffset).Unpack();
            R colors11 = GatherTexels<Texel>(indices10 + rowOffset).Unpack();
            R rowB = colors01 + (colors11 - colors01) * fx;

            return rowA + (rowB - rowA) * fy;
        }
    }

    // Interpolates texels overlapping the specified pixel coords near cubemap face edge (in N.LerpFracBits fixed-point). No bounds check.
    [[gnu::pure, gnu::noinline]] Texel::LerpedTy SampleLinearNearCubeEdge(VInt ixf, VInt iyf, VInt offset, VInt stride, VInt mipLevel, VInt faceIdx) const {
        using R = Texel::UnpackedTy;
        
        ixf = ixf - (LerpFracMask / 2);
        iyf = iyf - (LerpFracMask / 2);

        VInt ix = ixf >> LerpFracBits;
        VInt iy = iyf >> LerpFracBits;

        const float fracScale = 1.0f / (LerpFracMask + 1);
        VFloat fx = simd::conv2f(ixf & LerpFracMask) * fracScale;
        VFloat fy = simd::conv2f(iyf & LerpFracMask) * fracScale;

        R colors00 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 0, iy + 0);
        R colors10 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 1, iy + 0);
        R rowA = colors00 + (colors10 - colors00) * fx;

        R colors01 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 0, iy + 1);
        R colors11 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 1, iy + 1);
        R rowB = colors01 + (colors11 - colors01) * fx;

        return rowA + (rowB - rowA) * fy;
    }

    template<typename T>
    T GatherTexels(VInt indices) const {
        VInt v = VInt::gather<4>(Data.get(), indices);
        return std::bit_cast<T>(v);
    }
    Texel::UnpackedTy GatherTexels(int32_t offset, uint32_t stride, VInt ix, VInt iy) const {
        return GatherTexels<Texel>(offset + ix + (iy << stride)).Unpack();
    }
    Texel::UnpackedTy GatherTexelsNearCubeEdge(VInt offset, VInt stride, VInt mipLevel, VInt faceIdx, VInt ix, VInt iy) const {
        VInt scaleU = _maskU >> mipLevel, scaleV = _maskV >> mipLevel;
        VMask fallMask = (ix & scaleU) != ix | (iy & scaleV) != iy;

        if (simd::any(fallMask)) {
            VInt adjFace = faceIdx, adjX = ix, adjY = iy;
            texutil::GetAdjacentCubeFace(adjFace, adjX, adjY, scaleU, scaleV);

            ix = simd::csel(fallMask, adjX, ix);
            iy = simd::csel(fallMask, adjY, iy);
            ix = simd::min(simd::max(ix, 0), scaleU);
            iy = simd::min(simd::max(iy, 0), scaleV);
            offset += simd::csel(fallMask, (adjFace - faceIdx) << LayerShift, 0);
        }
        return GatherTexels<Texel>(offset + ix + (iy << stride)).Unpack();
    }

    void GenerateMip(uint32_t level, uint32_t layer) {
        uint32_t w = Width >> level, h = Height >> level;
        int32_t offset = (int32_t)(layer << LayerShift) + _mipOffsets[level - 1];
        uint32_t stride = RowShift - level + 1;

        for (uint32_t y = 0; y < h; y += 4) {
            for (uint32_t x = 0; x < w; x += 4) {
                VInt ix = ((int32_t)x + FragPixelOffsetsX) << 1;
                VInt iy = ((int32_t)y + FragPixelOffsetsY) << 1;

                // This will never go out of bounds if texture size is POT and >4x4.
                // Storage is padded by +16*4 bytes so nothing bad should happen if we do.
                auto c00 = GatherTexels(offset, stride, ix + 0, iy + 0);
                auto c10 = GatherTexels(offset, stride, ix + 1, iy + 0);
                auto c01 = GatherTexels(offset, stride, ix + 0, iy + 1);
                auto c11 = GatherTexels(offset, stride, ix + 1, iy + 1);
                auto avg = (c00 + c10 + c01 + c11) * 0.25f;

                WriteTile(Texel::Pack(avg), x, y, layer, level);
            }
        }
    }
};

};  // namespace swr
