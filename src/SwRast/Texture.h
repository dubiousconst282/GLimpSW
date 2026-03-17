#pragma once

#include "SIMD.h"

#include <concepts>
#include <functional>
#include <cstring>

namespace swr {

namespace pixfmt {

template<typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

// Represents a SIMD pixel packet in storage form, where the unpacked form consists of floats.
// NOTE: Packed size is currently restricted to 32-bits as that's the only element size texture sampling supports gathering.
template<typename T>
concept Texel = requires(const T& s, const T::UnpackedTy& u, v_int p) {
    IsAnyOf<typename T::UnpackedTy, v_int, v_float, v_float2, v_float3, v_float4>;
    IsAnyOf<typename T::LerpedTy, v_int, typename T::UnpackedTy>;
    { T::Unpack(p) } -> std::same_as<typename T::UnpackedTy>;
    { T::Pack(u) } -> std::same_as<v_int>;
};

// RGBA x 8-bit unorm
struct RGBA8u {
    using UnpackedTy = v_float4;
    using LerpedTy = v_int;

    static v_float4 Unpack(v_int packed) {
        const float scale = 1.0f / 255;
        return {
            simd::conv2f((packed >> 0) & 255) * scale,
            simd::conv2f((packed >> 8) & 255) * scale,
            simd::conv2f((packed >> 16) & 255) * scale,
            simd::conv2f((packed >> 24) & 255) * scale,
        };
    }
    static v_int Pack(const v_float4& value) {
        auto ri = simd::round2i(value.x * 255.0f);
        auto gi = simd::round2i(value.y * 255.0f);
        auto bi = simd::round2i(value.z * 255.0f);
        auto ai = simd::round2i(value.w * 255.0f);

        auto rg = _mm512_packs_epi32(ri, gi);
        auto ba = _mm512_packs_epi32(bi, ai);
        auto cb = _mm512_packus_epi16(rg, ba);

        const auto shuffMask = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        return _mm512_shuffle_epi8(cb, _mm512_broadcast_i32x4(shuffMask));
    }
};

// RGB x 10-bit unorm + unused x 2-bit
struct RGB10u {
    using UnpackedTy = v_float3;
    using LerpedTy = UnpackedTy;

    static v_float3 Unpack(v_int packed) {
        const float scale = 1.0f / 1023;
        return {
            simd::conv2f(packed >> 22 & 1023) * scale,
            simd::conv2f(packed >> 12 & 1023) * scale,
            simd::conv2f(packed >> 2 & 1023) * scale,
        };
    }
    static v_int Pack(const v_float3& value) {
        v_int ri = simd::round2i(value.x * 1023.0f);
        v_int gi = simd::round2i(value.y * 1023.0f);
        v_int bi = simd::round2i(value.z * 1023.0f);

        ri = simd::min(simd::max(ri, 0), 1023);
        gi = simd::min(simd::max(gi, 0), 1023);
        bi = simd::min(simd::max(bi, 0), 1023);

        return ri << 22 | gi << 12 | bi << 2;
    }
};

// R x 32-bit float
struct R32f {
    using UnpackedTy = v_float;
    using LerpedTy = UnpackedTy;

    static v_float Unpack(v_int packed) { return simd::re2f(packed); }
    static v_int Pack(const v_float& value) { return simd::re2i(value); }
};

// RG x 16-bit float
struct RG16f {
    using UnpackedTy = v_float2;
    using LerpedTy = UnpackedTy;

    static v_float2 Unpack(v_int packed) {
        return {
            _mm512_cvtph_ps(_mm512_cvtepi32_epi16(packed)),
            _mm512_cvtph_ps(_mm512_cvtepi32_epi16(packed >> 16)),
        };
    }
    static v_int Pack(const v_float2& value) {
        v_int r = _mm512_cvtepi16_epi32(_mm512_cvtps_ph(value.x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        v_int g = _mm512_cvtepi16_epi32(_mm512_cvtps_ph(value.y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        return r | (g << 16);
    }
};

// RG x 11-bit float, B x 10-bit float
struct R11G11B10f {
    using UnpackedTy = v_float3;
    using LerpedTy = UnpackedTy;

    static v_float3 Unpack(v_int packed) {
        return {
            UnpackF11(packed >> 21),
            UnpackF11(packed >> 10),
            UnpackF10(packed >> 0),
        };
    }
    static v_int Pack(const v_float3& value) {
        return
            PackF11(value.x) << 21 |
            PackF11(value.y) << 10 |
            PackF10(value.z);
    }
private:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules#11-bit-and-10-bit-floating-point-rules
    // Note that these don't follow denorm/NaN/Inf rules. Only clamping is applied.
    // This code was based on clang's output for the scalar version, as it doesn't seem to optimize the SIMD version well.
    static v_int PackF11(v_float x) {
        x = simd::max(x, 1.0f / (1 << 15));
        x = simd::min(x, 130048.0f); // UnpackF11(2047)

        return (simd::re2i(x) >> 17 & 0x3FFF) - 0x1C00;
    }
    static v_int PackF10(v_float x) {
        x = simd::max(x, 1.0f / (1 << 15));
        x = simd::min(x, 129024.0f);  // UnpackF10(1023)

        return (simd::re2i(x) >> 18 & 0x1FFF) - 0x0E00;
    }

    // 5-bit exp, 6-bit mantissa
    static v_float UnpackF11(v_int x) {
        // int exp = (x >> 6) & 0x1F;
        // int frc = x & 0x3F;
        // return (exp - 15 + 127) << 23 | frc << (23 - 6);

        x = x << 17;
        return simd::re2f((x & 0x0FFE0000) + 0x38000000);
    }
    // 5-bit exp, 5-bit mantissa
    static v_float UnpackF10(v_int x) {
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
    std::unique_ptr<uint8_t[], void(*)(void*)> Data = { nullptr, &free };

    static StbImage Load(const char* path, PixelType type = PixelType::RGBA_U8);
};

namespace texutil {

RgbaTexture2D LoadImage(const char* path, uint32_t mipLevels = 8);

HdrTexture2D LoadImageHDR(const char* path, uint32_t mipLevels = 8);

// Loads a equirectangular panorama into a cubemap.
HdrTexture2D LoadCubemapFromPanoramaHDR(const char* path, uint32_t mipLevels = 8);

// Iterates over the given rect in 4x4 tile steps. Visitor takes normalized UVs centered around pixel center.
inline void IterateTiles(uint32_t width, uint32_t height, std::function<void(uint32_t, uint32_t, v_float, v_float)> visitor) {
    assert(width % 4 == 0 && height % 4 == 0);

    for (int32_t y = 0; y < height; y += 4) {
        for (int32_t x = 0; x < width; x += 4) {
            v_float u = simd::conv2f(x + simd::FragPixelOffsetsX) + 0.5f;
            v_float v = simd::conv2f(y + simd::FragPixelOffsetsY) + 0.5f;
            visitor((uint32_t)x, (uint32_t)y, u * (1.0f / width), v * (1.0f / height));
        }
    }
}

// Calculates mip-level for a 4x4 fragment using the partial derivatives of the given scaled UVs.
inline v_int CalcMipLevel(v_float scaledU, v_float scaledV) {
    v_float dxu = simd::dFdx(scaledU), dyu = simd::dFdy(scaledU);
    v_float dxv = simd::dFdx(scaledV), dyv = simd::dFdy(scaledV);

    v_float maxDeltaSq = simd::max(simd::fma(dxu, dxu, dxv * dxv), simd::fma(dyu, dyu, dyv * dyv));
    return simd::ilog2(maxDeltaSq) >> 1;
    // return simd::approx_log2(maxDeltaSq) * 0.5f;
}

// Projects the given direction vector (may be unnormalized) to cubemap face UV and layer.
// This produces slightly different results from other graphics APIs, UVs are not flipped depending on faces.
inline void ProjectCubemap(v_float3 dir, v_float& u, v_float& v, v_int& faceIdx) {
    // https://gamedev.stackexchange.com/a/183463
    // https://en.wikipedia.org/wiki/Cube_mapping#Memory_addressing
    // Find axis with max magnitude
    v_float w = dir.x;

    v_int wy = simd::abs(dir.y) > simd::abs(w);
    w = wy ? dir.y : w;

    v_int wz = simd::abs(dir.z) > simd::abs(w);
    w = wz ? dir.z : w;

    v_int wx = wy | wz;  // negated
    wy &= ~wz;

    faceIdx = wz ? 4 : (wy ? 2 : 0);
    // faceIdx += w < 0 ? 1 : 0
    faceIdx += simd::shrl(simd::re2i(w), 31);

    // uv = { x: zy,  y: xz,  z: xy }[w]
    w = simd::rcp14(simd::abs(w)) * 0.5f;
    u = (wx ? dir.x : dir.z) * w + 0.5f;
    v = (wy ? dir.z : dir.y) * w + 0.5f;
}

// Unprojects the given cubemap face index and UVs to a normalized direction vector.
inline v_float3 UnprojectCubemap(v_float u, v_float v, v_int faceIdx) {
    v_float w = simd::re2f((faceIdx << 31) | 0x3F800000);  //(faceIdx & 1) ? -1.0 : +1.0
    v_int axis = faceIdx >> 1;

    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // 0:  1,  v,  u
    // 1: -1,  v,  u
    // 2:  u,  1,  v
    // 3:  u, -1,  v
    // 4:  u,  v,  1
    // 5:  u,  v, -1
    v_float3 unnormDir = {
        axis == 0 ? w : u,
        axis == 1 ? w : v,
        axis == 2 ? w : (axis == 0 ? u : v),
    };
    return unnormDir * simd::rsqrt14(u * u + v * v + 1.0f);
}

// Lookups the adjacent cube face and UVs to the nearest edge.
inline void GetAdjacentCubeFace(v_int& faceIdx, v_int& u, v_int& v, v_int scaleU, v_int scaleV) {
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

    v_int cu = (scaleU >> 1) - u, cv = (scaleV >> 1) - v;
    v_int quadIdx = simd::abs(cu) > simd::abs(cv) ? simd::shrl(cu, 31) + 2 : simd::shrl(cv, 31);
    v_int tableIdx = quadIdx * 8 + faceIdx;

    v_int data = _mm512_permutexvar_epi8(tableIdx, _mm512_broadcast_i32x8(_mm256_loadu_epi8(AdjFaceLUT)));

    faceIdx = data & 7;

    v_int swap = (data >> 3 & 1) != 0;
    v_int invU = (data >> 4 & 1) != 0;
    v_int invV = (data >> 5 & 1) != 0;

    v_int su = swap ? v : u;
    v_int sv = swap ? u : v;
    u = invU ? scaleU - su : su;
    v = invV ? scaleV - sv : sv;
}

// Texture swizzling doesn't improve performance by much, the functions below are keept for reference.

// 32-bit Z-curve/morton encode. Takes ~8.5 cycles per 16 coord pairs on TigerLake, according to llvm-mca.
// - https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/
// - https://github.com/KWillets/simd_interleave/blob/master/simd.c
inline v_int Interleave(v_int x, v_int y) {
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
inline v_int GetTiledOffset(v_int ix, v_int iy, v_int rowShift) {
    v_int tileId = (ix >> 2) + ((iy >> 2) << (rowShift - 2));
    v_int pixelOffset = (ix & 3) + (iy & 3) * 4;
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

        for (; MipLevels < std::min(mipLevels, simd::vec_width); MipLevels++) {
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
            memcpy(&Data[(layer << LayerShift) + (y << RowShift)], (uint32_t*)pixels + y * stride, Width * 4);
        }
    }

    // Writes a 4x4 tile of texels to the texture buffer. Coords are in pixel space.
    void WriteTile(v_int value, uint32_t x, uint32_t y, uint32_t layer = 0, uint32_t mipLevel = 0) {
        assert(x + 3 < Width && y + 3 < Height);
        assert(x % 4 == 0 && y % 4 == 0);
        assert(layer < NumLayers && mipLevel < MipLevels);
        
        uint32_t* dst = &Data[(layer << LayerShift) + (uint32_t)_mipOffsets[mipLevel]];
        uint32_t stride = RowShift - mipLevel;

        _mm_storeu_epi32(&dst[x + ((y + 0) << stride)], _mm512_extracti32x4_epi32(value, 0));
        _mm_storeu_epi32(&dst[x + ((y + 1) << stride)], _mm512_extracti32x4_epi32(value, 1));
        _mm_storeu_epi32(&dst[x + ((y + 2) << stride)], _mm512_extracti32x4_epi32(value, 2));
        _mm_storeu_epi32(&dst[x + ((y + 3) << stride)], _mm512_extracti32x4_epi32(value, 3));
    }

    void GenerateMips() {
        for (uint32_t layer = 0; layer < NumLayers; layer++) {
            for (uint32_t level = 1; level < MipLevels; level++) {
                GenerateMip(level, layer);
            }
        }
    }

    // We're forcing always_inline everywhere because it helps avoid register spills across call boundaries.

    template<SamplerDesc SD, bool CalcMipFromUVDerivs_ = true, bool IsCubeSample_ = false>
    [[gnu::pure, gnu::always_inline]] Texel::LerpedTy Sample(v_float u, v_float v, v_int layer = 0, v_int mipLevel = 0) const {
        // Scale and round UVs 
        v_float su = u * _scaleLerpU, sv = v * _scaleLerpV;
        v_int ix = simd::round2i(su), iy = simd::round2i(sv);

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
        v_int stride = (int32_t)RowShift;
        v_int offset = layer << LayerShift;

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
            v_int res = GatherRawTexels(offset + ix + (iy << stride));

            if constexpr (std::is_same<typename Texel::LerpedTy, v_int>()) {
                return res;
            } else {
                return Texel::Unpack(res);
            }
        }
        if constexpr (IsCubeSample_) {
            //    x < 1 || x >= N
            // =  (x-1) >= (N-1)     given twos-complement + unsigned cmp
            v_mask edgeU = _mm512_cmpge_epu32_mask((ix >> LerpFracBits) - 1, (_maskU >> mipLevel) - 1);
            v_mask edgeV = _mm512_cmpge_epu32_mask((iy >> LerpFracBits) - 1, (_maskV >> mipLevel) - 1);

            if (simd::any(edgeU | edgeV)) [[unlikely]] {
                return SampleLinearNearCubeEdge(ix, iy, offset, stride, mipLevel, layer);
            }
        }
        return SampleLinear(ix, iy, offset, stride, mipLevel);
    }

    template<SamplerDesc SD>
    [[gnu::pure, gnu::always_inline]] Texel::LerpedTy SampleCube(const v_float3& dir, v_float mipLevel = -1.0f) const {
        v_float u, v;
        v_int faceIdx;
        texutil::ProjectCubemap(dir, u, v, faceIdx);

        int tmp = mipLevel[0];  // __builtin_constant_p() only works with variables - https://github.com/llvm/llvm-project/issues/65741
        if (__builtin_constant_p(tmp) && tmp < 0) {
            return Sample<SD, true, true>(u, v, faceIdx);
        }
        v_int baseMip = simd::trunc2i(mipLevel);
        auto baseSample = Sample<SD, false, true>(u, v, faceIdx, baseMip);

        v_float mipFrac = mipLevel - simd::conv2f(baseMip);
        if (simd::any(mipFrac > 0.0f) && simd::any(baseMip < (int32_t)(MipLevels - 1))) {
            auto lowerSample = Sample<SD, false, true>(u, v, faceIdx, baseMip + 1);
            return baseSample + (lowerSample - baseSample) * mipFrac;
        }
        return baseSample;
    }

private:
    v_int _mipOffsets;
    float _scaleU, _scaleV, _scaleLerpU, _scaleLerpV;
    int32_t _maskU, _maskV, _maskLerpU, _maskLerpV;

    // Interpolates texels overlapping the specified pixel coords (in N.LerpFracBits fixed-point). No bounds check.
    [[gnu::pure, gnu::always_inline]] Texel::LerpedTy SampleLinear(v_int ixf, v_int iyf, v_int offset, v_int stride, v_int mipLevel) const {
        ixf = simd::max(ixf - (LerpFracMask / 2), 0);
        iyf = simd::max(iyf - (LerpFracMask / 2), 0);

        v_int ix = ixf >> LerpFracBits;
        v_int iy = iyf >> LerpFracBits;

        v_int indices00 = offset + ix + (iy << stride);
        v_int indices01 = indices00 + (((iy + 1) << mipLevel) < (int32_t)Height ? 1 << stride : 0);
        v_int inboundX = ((ix + 1) << mipLevel) < (int32_t)Width;

#if 0
        v_int data00 = _mm512_i32gather_epi32(indices00, Data.get() + 0, 4);
        v_int data10 = _mm512_i32gather_epi32(indices00, Data.get() + 1, 4);
        v_int data01 = _mm512_i32gather_epi32(indices01, Data.get() + 0, 4);
        v_int data11 = _mm512_i32gather_epi32(indices01, Data.get() + 1, 4);
#else
        v_int row0_lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices00, 0), Data.get(), 4);
        v_int row0_hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices00, 1), Data.get(), 4);
        v_int row1_lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices01, 0), Data.get(), 4);
        v_int row1_hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices01, 1), Data.get(), 4);

        v_int data00 = _mm512_permutex2var_epi32(row0_lo, simd::lane_idx * 2 + 0, row0_hi);
        v_int data10 = _mm512_permutex2var_epi32(row0_lo, simd::lane_idx * 2 + 1, row0_hi);
        v_int data01 = _mm512_permutex2var_epi32(row1_lo, simd::lane_idx * 2 + 0, row1_hi);
        v_int data11 = _mm512_permutex2var_epi32(row1_lo, simd::lane_idx * 2 + 1, row1_hi);
#endif

        if constexpr (std::is_same<Texel, pixfmt::RGBA8u>()) {
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
        } else {
            using R = Texel::UnpackedTy;

            const float fracScale = 1.0f / (LerpFracMask + 1);
            v_float fx = simd::conv2f(ixf & LerpFracMask) * fracScale;
            v_float fy = simd::conv2f(iyf & LerpFracMask) * fracScale;
            fx = inboundX ? fx : 0;

            R colors00 = Texel::Unpack(data00);
            R colors10 = Texel::Unpack(data10);
            R rowA = colors00 + (colors10 - colors00) * fx;

            R colors01 = Texel::Unpack(data01);
            R colors11 = Texel::Unpack(data11);
            R rowB = colors01 + (colors11 - colors01) * fx;

            return rowA + (rowB - rowA) * fy;
        }
    }

    // Interpolates texels overlapping the specified pixel coords near cubemap face edge (in N.LerpFracBits fixed-point). No bounds check.
    [[gnu::pure, gnu::noinline]] Texel::LerpedTy SampleLinearNearCubeEdge(v_int ixf, v_int iyf, v_int offset, v_int stride, v_int mipLevel, v_int faceIdx) const {
        using R = Texel::UnpackedTy;
        
        ixf = ixf - (LerpFracMask / 2);
        iyf = iyf - (LerpFracMask / 2);

        v_int ix = ixf >> LerpFracBits;
        v_int iy = iyf >> LerpFracBits;

        const float fracScale = 1.0f / (LerpFracMask + 1);
        v_float fx = simd::conv2f(ixf & LerpFracMask) * fracScale;
        v_float fy = simd::conv2f(iyf & LerpFracMask) * fracScale;

        R colors00 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 0, iy + 0);
        R colors10 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 1, iy + 0);
        R rowA = colors00 + (colors10 - colors00) * fx;

        R colors01 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 0, iy + 1);
        R colors11 = GatherTexelsNearCubeEdge(offset, stride, mipLevel, faceIdx, ix + 1, iy + 1);
        R rowB = colors01 + (colors11 - colors01) * fx;

        return rowA + (rowB - rowA) * fy;
    }

    v_int GatherRawTexels(v_int indices) const {
        return simd::gather((const int32_t*)Data.get(), indices);
    }
    Texel::UnpackedTy GatherTexels(int32_t offset, uint32_t stride, v_int ix, v_int iy) const {
        return Texel::Unpack(GatherRawTexels(offset + ix + (iy << stride)));
    }
    Texel::UnpackedTy GatherTexelsNearCubeEdge(v_int offset, v_int stride, v_int mipLevel, v_int faceIdx, v_int ix, v_int iy) const {
        v_int scaleU = _maskU >> mipLevel, scaleV = _maskV >> mipLevel;
        v_int fallMask = (ix & scaleU) != ix || (iy & scaleV) != iy;

        if (simd::any(fallMask)) {
            v_int adjFace = faceIdx, adjX = ix, adjY = iy;
            texutil::GetAdjacentCubeFace(adjFace, adjX, adjY, scaleU, scaleV);

            ix = fallMask ? adjX : ix;
            iy = fallMask ? adjY : iy;
            ix = simd::min(simd::max(ix, 0), scaleU);
            iy = simd::min(simd::max(iy, 0), scaleV);
            offset += fallMask ? ((adjFace - faceIdx) << LayerShift) : 0;
        }
        return Texel::Unpack(GatherRawTexels(offset + ix + (iy << stride)));
    }

    void GenerateMip(uint32_t level, uint32_t layer) {
        uint32_t w = Width >> level, h = Height >> level;
        int32_t offset = (int32_t)(layer << LayerShift) + _mipOffsets[level - 1];
        uint32_t stride = RowShift - level + 1;

        for (uint32_t y = 0; y < h; y += 4) {
            for (uint32_t x = 0; x < w; x += 4) {
                v_int ix = ((int32_t)x + simd::FragPixelOffsetsX) << 1;
                v_int iy = ((int32_t)y + simd::FragPixelOffsetsY) << 1;

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
