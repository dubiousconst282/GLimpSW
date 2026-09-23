#pragma once

#include "SIMD.h"

#include <concepts>
#include <cstring>

namespace swr {

namespace pixfmt {

template<typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);

template<typename T>
concept TexelFormat = requires(const T& s, const T::UnpackedTy& u, v_uint p) {
    IsAnyOf<typename T::UnpackedTy, v_uint, v_float, v_float2, v_float3, v_float4>;
    IsAnyOf<typename T::LerpedTy, v_uint, typename T::UnpackedTy>;
    { T::Unpack(p) } -> std::same_as<typename T::UnpackedTy>;
    { T::Pack(u) } -> std::same_as<v_uint>;
};

// RGBA x 8-bit unorm
struct RGBA8u {
    using UnpackedTy = v_float4;
    using LerpedTy = v_uint;

    static v_float4 Unpack(v_uint packed) {
        const float scale = 1.0f / 255;
        return {
            simd::conv<float>((packed >> 0) & 255) * scale,
            simd::conv<float>((packed >> 8) & 255) * scale,
            simd::conv<float>((packed >> 16) & 255) * scale,
            simd::conv<float>((packed >> 24) & 255) * scale,
        };
    }
    static v_float4 UnpackSrgb(v_uint packed) {
        // TODO: maybe use more accurate approx?
        // https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
        // x * (x * (x * 0.305306011 + 0.682171111) + 0.012522878)

        v_uint rb1 = ((packed << 8) & 0xFF00'FF00) + 0x00FF'00FF;
        v_uint ag1 = ((packed << 0) & 0xFF00'FF00) + 0x00FF'00FF;
        v_uint rb2 = _mm512_mulhi_epu16(rb1, rb1);
        v_uint ag2 = _mm512_mulhi_epu16(ag1, ag1);

        constexpr float scale = 1.0f / 65535;
        return {
            simd::conv<float>(rb2 & 65535) * scale,
            simd::conv<float>(ag2 & 65535) * scale,
            simd::conv<float>(rb2 >> 16) * scale,
            simd::conv<float>(ag1 >> 16) * scale,
        };
    }
    static v_uint Pack(const v_float4& value) {
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

    static v_float3 Unpack(v_uint packed) {
        const float scale = 1.0f / 1023;
        return {
            simd::conv<float>(packed >> 0 & 1023) * scale,
            simd::conv<float>(packed >> 10 & 1023) * scale,
            simd::conv<float>(packed >> 20 & 1023) * scale,
        };
    }
    static v_uint Pack(const v_float3& value) {
        v_int ri = simd::round2i(value.x * 1023.0f);
        v_int gi = simd::round2i(value.y * 1023.0f);
        v_int bi = simd::round2i(value.z * 1023.0f);

        ri = simd::min(simd::max(ri, 0), 1023);
        gi = simd::min(simd::max(gi, 0), 1023);
        bi = simd::min(simd::max(bi, 0), 1023);

        return v_uint(ri << 0 | gi << 10 | bi << 20);
    }
};

// R x 32-bit float
struct R32f {
    using UnpackedTy = v_float;
    using LerpedTy = UnpackedTy;

    static v_float Unpack(v_uint packed) { return simd::as<v_float>(packed); }
    static v_uint Pack(const v_float& value) { return simd::as<v_uint>(value); }
};

// R x 32-bit uint
struct R32ui {
    using UnpackedTy = v_uint;
    using LerpedTy = UnpackedTy;

    static v_uint Unpack(v_uint packed) { return packed; }
    static v_uint Pack(const v_uint& value) { return value; }
};

// RG x 16-bit float
struct RG16f {
    using UnpackedTy = v_float2;
    using LerpedTy = UnpackedTy;

    static v_float2 Unpack(v_uint packed) {
        constexpr simd::vec<uint8_t, 64> shuf = {
            0,  1,  4,  5,  8,  9,  12, 13, 16, 17, 20, 21, 24, 25, 28, 29,  //
            32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61,  //
            2,  3,  6,  7,  10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31,  //
            34, 35, 38, 39, 42, 43, 46, 47, 50, 51, 54, 55, 58, 59, 62, 63,
        };
        // permb is faster than vpmovdw. uncanon to workaround https://github.com/llvm/llvm-project/issues/160348
        v_uint split = _mm512_permutexvar_epi8(simd::uncanon(shuf), packed);

        return {
            _mm512_cvtph_ps(_mm512_extracti32x8_epi32(split, 0)),
            _mm512_cvtph_ps(_mm512_extracti32x8_epi32(split, 1)),
        };
    }
    static v_uint Pack(const v_float2& value) {
        v_uint r = _mm512_cvtepi16_epi32(_mm512_cvtps_ph(value.x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        v_uint g = _mm512_cvtepi16_epi32(_mm512_cvtps_ph(value.y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        return r | (g << 16);
    }
};

// RG x 11-bit float, B x 10-bit float
struct R11G11B10f {
    using UnpackedTy = v_float3;
    using LerpedTy = UnpackedTy;

    static v_float3 Unpack(v_uint packed) {
        return {
            UnpackF11(packed >> 21),
            UnpackF11(packed >> 10),
            UnpackF10(packed >> 0),
        };
    }
    static v_uint Pack(const v_float3& value) {
        return
            PackF11(value.x) << 21 |
            PackF11(value.y) << 10 |
            PackF10(value.z);
    }
private:
    // https://learn.microsoft.com/en-us/windows/win32/direct3d11/floating-point-rules#11-bit-and-10-bit-floating-point-rules
    // Note that these don't follow denorm/NaN/Inf rules. Only clamping is applied.
    // This code was based on clang's output for the scalar version, as it doesn't seem to optimize the SIMD version well.
    static v_uint PackF11(v_float x) {
        // 0x38000000 = (127 - 15) << 23
        v_int q = simd::as<v_int>(x);
        q = simd::clamp(q, 0x38000000, 0x57FC0000);  // 2**-15  ..  130048
        return v_uint(q - 0x38000000) >> 17;
    }
    static v_uint PackF10(v_float x) {
        v_int q = simd::as<v_int>(x);
        q = simd::clamp(q, 0x38000000, 0x47FC0000);  // 2**-15  ..  129024
        return v_uint(q - 0x38000000) >> 18;
    }

    // 5-bit exp, 6-bit mantissa
    static v_float UnpackF11(v_uint x) {
        // int exp = (x >> 6) & 0x1F;
        // int frc = x & 0x3F;
        // return (exp - 15 + 127) << 23 | frc << (23 - 6);

        return simd::as<v_float>(((x << 17) & 0x0FFE0000) + 0x38000000);
    }
    // 5-bit exp, 5-bit mantissa
    static v_float UnpackF10(v_uint x) {
        return simd::as<v_float>(((x << 18) & 0x0FFC0000) + 0x38000000);
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

enum class TextureLayout {
    Linear,   // Row-major                  (optimal for sequential access)
    TiledX4,  // 4x4 tiling, row-major      (optimal for rendering)
    TiledY8,  // 8x8 tiling, POT-only sizes (optimal for sampling)

    Default = TiledY8,
};

template<pixfmt::TexelFormat Format, TextureLayout Layout = TextureLayout::Default>
struct Texture2D;

template<pixfmt::TexelFormat Format, TextureLayout Layout = TextureLayout::Default>
using TexturePtr2D = std::unique_ptr<Texture2D<Format, Layout>, simd::AlignedDeleter>;

template<pixfmt::TexelFormat Format>
using Framebuffer = Texture2D<Format, TextureLayout::TiledX4>;

template<pixfmt::TexelFormat Format>
using FramebufferPtr = TexturePtr2D<Format, TextureLayout::TiledX4>;

using RgbaTexture2D = Texture2D<pixfmt::RGBA8u>;
using HdrTexture2D = Texture2D<pixfmt::R11G11B10f>;

constexpr v_int2 TilePixelOffsets = { (simd::lane_idx<v_int> & 3), (simd::lane_idx<v_int> >> 2) };

// TODO: delete or move this
struct StbImage {
    enum class PixelType { Empty, RGBA_U8, RGB_F32 };

    uint32_t Width, Height;
    PixelType Type;
    std::unique_ptr<uint8_t[], void(*)(void*)> Data = { nullptr, &free };

    static StbImage Load(const char* path, PixelType type = PixelType::RGBA_U8);
};

namespace texutil {

TexturePtr2D<pixfmt::RGBA8u> LoadImage(const char* path, uint32_t mipLevels = 8);

TexturePtr2D<pixfmt::R11G11B10f> LoadImageHDR(const char* path, uint32_t mipLevels = 8);

// Loads a equirectangular panorama into a cubemap.
TexturePtr2D<pixfmt::R11G11B10f> LoadOctahedronFromPanoramaHDR(const char* path, uint32_t mipLevels = 8);

void DownsampleDepth(Framebuffer<pixfmt::R32f>& src, Texture2D<pixfmt::R32f>& dest);

// Calculate partial derivatives for a 4x4 fragment via finite difference.
// https://gamedev.stackexchange.com/a/130933
inline v_float dFdx(v_float p) {
    auto a = _mm512_shuffle_ps(p, p, 0b10'10'00'00);  //[0 0 2 2]
    auto b = _mm512_shuffle_ps(p, p, 0b11'11'01'01);  //[1 1 3 3]
    return _mm512_sub_ps(b, a);
}
inline v_float dFdy(v_float p) {
    auto a = _mm512_shuffle_f32x4(p, p, 0b10'10'00'00);
    auto b = _mm512_shuffle_f32x4(p, p, 0b11'11'01'01);
    return _mm512_sub_ps(b, a);
}
// Calculates mip-level using derivatives from scaled sample UVs (dxu, dxv, dyu, dyv).
inline v_int CalcMipLevel(v_float4 grad) {
    v_float dx = simd::fma(grad.x, grad.x, grad.y * grad.y);
    v_float dy = simd::fma(grad.z, grad.z, grad.w * grad.w);
    return simd::ilog2(simd::max(dx, dy)) >> 1;
}
inline v_int CalcMipLevel(v_float4 grad, v_float2 scale) {
    grad *= v_float4(scale, scale);
    v_float dx = simd::fma(grad.x, grad.x, grad.y * grad.y);
    v_float dy = simd::fma(grad.z, grad.z, grad.w * grad.w);
    return simd::ilog2(simd::max(dx, dy)) >> 1;
}

inline v_float2 MapOctahedron(v_float3 norm) {
    v_float w = simd::approx_rcp(simd::abs(norm.x) + simd::abs(norm.y) + simd::abs(norm.z));
    v_float t = simd::max(-norm.z * w, 0.0);
    v_float u = simd::fma(norm.x, w, simd::mulsign(t, norm.x));
    v_float v = simd::fma(norm.y, w, simd::mulsign(t, norm.y));
    return v_float2(u, v) * 0.5f + 0.5f;
}
inline v_float3 UnmapOctahedron(v_float2 uv) {
    uv = uv * 2.0f - 1.0f;
    v_float3 norm = v_float3(uv, 1.0 - simd::abs(uv.x) - simd::abs(uv.y));
    v_float t = simd::max(-norm.z, 0.0);
    norm.x -= simd::mulsign(t, norm.x);
    norm.y -= simd::mulsign(t, norm.y);
    return simd::normalize(norm);
}
inline v_float2 FixupOctahedronWrap(v_float2 uv) {
    // TODO https://gpuopen.com/learn/fetching-from-cubes-and-octahedrons/
    // https://www.readkong.com/page/octahedron-environment-maps-6054207
    return uv;
}

};  // namespace texutil

enum class WrapMode { Repeat, ClampToEdge, MirroredRepeat };
enum class FilterMode { Nearest, Linear };

// Texture sampler parameters
struct SamplerDesc {
    WrapMode Wrap = WrapMode::Repeat;
    FilterMode MagFilter = FilterMode::Linear;
    FilterMode MinFilter = FilterMode::Nearest;
};

template<pixfmt::TexelFormat Fmt_, TextureLayout Layout_>
struct Texture2D {
    using Ptr = TexturePtr2D<Fmt_, Layout_>;
    using Format = Fmt_;

    static constexpr TextureLayout Layout = Layout_;

    static constexpr int LerpFracBits = 8;  // Number of fractional bits in pixel coords for bilinear interpolation
    static constexpr int LerpFracMask = (1 << LerpFracBits) - 1;

    uint32_t Width, Height, MipLevels, NumLayers;
    uint32_t RowShift, LayerStride;

    float ScaleU, ScaleV, ScaleLerpU, ScaleLerpV;
    int32_t MaskU, MaskV, MaskLerpU, MaskLerpV;
    alignas(64) uint32_t MipOffsets[16];
    alignas(64) uint32_t Data[];

    Texture2D() = delete;
    Texture2D(const Texture2D&) = delete;
    Texture2D& operator=(const Texture2D&) = delete;

    void Clear(uint32_t color, uint32_t layer = 0, uint32_t mipLevel = 0) {
        // Non-temporal fill is ~2-3x faster than memset(), but makes rasterization a bit slower. Still a small win overall.
        // https://en.algorithmica.org/hpc/cpu-cache/bandwidth/
        uint32_t* ptr = &Data[(layer * LayerStride) + MipOffsets[mipLevel]];
        uint32_t* end = &ptr[(Width >> mipLevel) * (Height >> mipLevel)];

        for (; ptr < end; ptr += 16) {
            _mm512_stream_si512(ptr, _mm512_set1_epi32((int32_t)color));
        }
        _mm_sfence();
    }
    void ClearRect(uint2 offset, uint2 size, uint32_t color, uint32_t layer = 0, uint32_t mipLevel = 0) {
        assert(offset.x % 4 == 0 && offset.y % 4 == 0);
        assert(offset.x + size.x <= (Width >> mipLevel));
        assert(offset.y + size.y <= (Height >> mipLevel));

        uint32_t* dstLayer = &Data[(layer * LayerStride) + MipOffsets[mipLevel]];

        for (uint32_t y = 0; y < size.y; y += 4) {
            uint32_t* ptr = &dstLayer[GetTexelOffset(offset.x, offset.y + y, mipLevel)];

            for (uint32_t x = 0; x < size.x; x += 4, ptr += 16) {
                if constexpr (Layout_ == TextureLayout::TiledX4) {
                    _mm512_stream_si512(ptr, _mm512_set1_epi32((int32_t)color));
                } else {
                    WriteTile(v_uint(color), x, y, layer, mipLevel);
                }
            }
        }
        _mm_sfence();
    }

    // Copy pixel array in linear layout to texture.
    void SetPixels(const void* pixels, uint32_t stride, uint32_t layer = 0, uint32_t mipLevel = 0) {
        assert(layer < NumLayers && mipLevel < MipLevels);

        uint32_t* dst = &Data[(layer * LayerStride) + MipOffsets[mipLevel]];
        const uint32_t* src = (const uint32_t*)pixels;
        uint32_t w = Width >> mipLevel, h = Height >> mipLevel;

        if constexpr (Layout_ == TextureLayout::Linear) {
            for (uint32_t y = 0; y < h; y++) {
                memcpy(&dst[y * Width], &src[y * stride], Width * 4);
            }
        } else {
            // TODO: optimize
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    dst[GetTexelOffset(x, y, mipLevel)] = src[x + y * stride];
                }
            }
        }
    }

    // Copy texture data to pixel array in linear layout.
    void GetPixels(void* pixels, uint32_t stride, uint32_t layer = 0, uint32_t mipLevel = 0) const {
        assert(layer < NumLayers && mipLevel < MipLevels);

        const uint32_t* src = &Data[(layer * LayerStride) + MipOffsets[mipLevel]];
        uint32_t* dest = (uint32_t*)pixels;
        uint32_t w = Width >> mipLevel, h = Height >> mipLevel;

        if constexpr (Layout_ == TextureLayout::TiledX4) {
            assert(((uintptr_t)dest % 64) == 0);

            for (uint32_t y = 0; y < h; y += 4) {
                uint32_t x = 0;
                const uint32_t* srcRow = &src[GetTexelOffset(x, y, mipLevel)];

                // 30% faster than copying one tile at a time, limited by mem bw
                for (; x + 15 < w; x += 16, srcRow += 64) {
                    __m512i tileA = _mm512_load_si512(srcRow + 0);
                    __m512i tileB = _mm512_load_si512(srcRow + 16);
                    __m512i tileC = _mm512_load_si512(srcRow + 32);
                    __m512i tileD = _mm512_load_si512(srcRow + 48);

                    __m512i AB01 = _mm512_shuffle_i32x4(tileA, tileB, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i AB23 = _mm512_shuffle_i32x4(tileA, tileB, _MM_SHUFFLE(3, 2, 3, 2));
                    __m512i CD01 = _mm512_shuffle_i32x4(tileC, tileD, _MM_SHUFFLE(1, 0, 1, 0));
                    __m512i CD23 = _mm512_shuffle_i32x4(tileC, tileD, _MM_SHUFFLE(3, 2, 3, 2));

                    __m512i row0 = _mm512_shuffle_i32x4(AB01, CD01, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i row1 = _mm512_shuffle_i32x4(AB01, CD01, _MM_SHUFFLE(3, 1, 3, 1));
                    __m512i row2 = _mm512_shuffle_i32x4(AB23, CD23, _MM_SHUFFLE(2, 0, 2, 0));
                    __m512i row3 = _mm512_shuffle_i32x4(AB23, CD23, _MM_SHUFFLE(3, 1, 3, 1));

                    _mm512_stream_si512(&dest[(y + 0) * stride + x], row0);
                    _mm512_stream_si512(&dest[(y + 1) * stride + x], row1);
                    _mm512_stream_si512(&dest[(y + 2) * stride + x], row2);
                    _mm512_stream_si512(&dest[(y + 3) * stride + x], row3);
                }
                for (; x < w; x += 4, srcRow += 16) {
                    __m512i tile = _mm512_load_si512(srcRow);
                    _mm_stream_si128(&dest[(y + 0) * stride + x], _mm512_extracti32x4_epi32(tile, 0));
                    _mm_stream_si128(&dest[(y + 1) * stride + x], _mm512_extracti32x4_epi32(tile, 1));
                    _mm_stream_si128(&dest[(y + 2) * stride + x], _mm512_extracti32x4_epi32(tile, 2));
                    _mm_stream_si128(&dest[(y + 3) * stride + x], _mm512_extracti32x4_epi32(tile, 3));
                }
            }
            _mm_sfence();
        } else {
            // TODO: optimize
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    dest[x + y * stride] = src[GetTexelOffset(x, y, mipLevel)];
                }
            }
        }
    }

    // Writes a tile of texels to the texture buffer.
    void WriteTile(v_uint value, uint32_t x, uint32_t y, uint32_t layer = 0, uint32_t mipLevel = 0) {
        assert(x + 3 < Width && y + 3 < Height);
        assert(x % 4 == 0 && y % 4 == 0);
        assert(layer < NumLayers && mipLevel < MipLevels);

        uint32_t* dst = &Data[(layer * LayerStride) + MipOffsets[mipLevel] + GetTexelOffset(x, y, mipLevel)];

        if constexpr (Layout_ == TextureLayout::Linear) {
            uint32_t stride = Width >> mipLevel;
            _mm_storeu_epi32(&dst[0 * stride], _mm512_extracti32x4_epi32(value, 0));
            _mm_storeu_epi32(&dst[1 * stride], _mm512_extracti32x4_epi32(value, 1));
            _mm_storeu_epi32(&dst[2 * stride], _mm512_extracti32x4_epi32(value, 2));
            _mm_storeu_epi32(&dst[3 * stride], _mm512_extracti32x4_epi32(value, 3));
        } else if constexpr (Layout_ == TextureLayout::TiledY8) {
            value = _mm512_permutexvar_epi32(v_uint{ 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 }, value);
            _mm_storeu_epi32(&dst[0], _mm512_extracti32x4_epi32(value, 0));
            _mm_storeu_epi32(&dst[8], _mm512_extracti32x4_epi32(value, 1));
            _mm_storeu_epi32(&dst[16], _mm512_extracti32x4_epi32(value, 2));
            _mm_storeu_epi32(&dst[24], _mm512_extracti32x4_epi32(value, 3));
        } else if constexpr (Layout_ == TextureLayout::TiledX4) {
            _mm512_store_epi32(dst, value);
        } else {
            static_assert(!"Bad layout");
        }
    }
    void WriteTile(const Format::UnpackedTy& value, uint32_t x, uint32_t y, uint32_t layer = 0, uint32_t mipLevel = 0) {
        WriteTile(Format::Pack(value), x, y, layer, mipLevel);
    }

    void GenerateMips() {
        for (uint32_t layer = 0; layer < NumLayers; layer++) {
            for (uint32_t level = 1; level < MipLevels; level++) {
                GenerateMip(level, layer);
            }
        }
    }

    // Fetch raw texels from storage. All coordinates must be in-bounds.
    [[gnu::pure, gnu::always_inline]] v_uint Fetch(v_int2 pos, v_int layer = 0, v_int mipLevel = 0, v_int laneMask = -1) const {
        v_int offset = layer * (int)LayerStride;

        if (simd::any(mipLevel > 0)) {
            offset += _mm512_permutexvar_epi32(mipLevel, _mm512_load_epi32(MipOffsets));
        }
        return simd::gather(Data, offset + GetTexelOffset(pos.x, pos.y, mipLevel), laneMask);
    }

    // Sample with implicit derivatives.
    template<SamplerDesc SD>
    [[gnu::pure, gnu::always_inline]] Format::LerpedTy SampleImplicitLod(v_float2 pos, v_int layer = 0) const {
        v_float su = pos.x * ScaleLerpU, sv = pos.y * ScaleLerpV;

        v_float4 grad = { texutil::dFdx(su), texutil::dFdx(sv), texutil::dFdy(su), texutil::dFdy(sv) };
        v_int mipLevel = texutil::CalcMipLevel(grad) - LerpFracBits;
        return SampleLevel<SD>(pos, layer, mipLevel);
    }

    template<SamplerDesc SD>
    [[gnu::pure, gnu::always_inline]] Format::LerpedTy SampleLevel(v_float2 pos, v_int layer, v_int mipLevel) const {
        // Scale and round UVs 
        v_float su = pos.x * ScaleLerpU, sv = pos.y * ScaleLerpV;
        v_int ix = simd::round2i(su), iy = simd::round2i(sv);

        // TODO: support repeat wrap modes for non-POT textures
        static_assert(Layout_ == TextureLayout::TiledY8 || SD.Wrap == WrapMode::ClampToEdge);

        // Wrap
        if constexpr (SD.Wrap == WrapMode::ClampToEdge) {
            ix = simd::min(simd::max(ix, 0), MaskLerpU);
            iy = simd::min(simd::max(iy, 0), MaskLerpV);
        } else if constexpr (SD.Wrap == WrapMode::Repeat) {
            ix &= MaskLerpU;
            iy &= MaskLerpV;
        } else if constexpr (SD.Wrap == WrapMode::MirroredRepeat) {
            ix = (ix & MaskLerpU) ^ ((ix & (MaskLerpU + 1)) != 0 ? MaskLerpU : 0);
            iy = (iy & MaskLerpV) ^ ((iy & (MaskLerpV + 1)) != 0 ? MaskLerpV : 0);
        } else {
            static_assert(!"Bad wrap mode");
        }

        FilterMode filter = simd::any(mipLevel > 0) ? SD.MinFilter : SD.MagFilter;

        int maxLevel = (int)MipLevels - 1;
        [[assume(maxLevel >= 0)]];
        mipLevel = simd::clamp(mipLevel, 0, maxLevel);

        // Calculate offsets and mip position
        v_int offset = layer * (int)LayerStride;

        ix >>= mipLevel, iy >>= mipLevel;
        offset += _mm512_permutexvar_epi32(mipLevel, _mm512_load_epi32(MipOffsets));

        // Sample
        if (filter == FilterMode::Linear) [[unlikely]] {
            return SampleLinear(ix, iy, offset, mipLevel);
        } else {
            v_uint res = simd::gather(Data, offset + GetTexelOffset(ix >> LerpFracBits, iy >> LerpFracBits, mipLevel));

            if constexpr (std::is_same<typename Format::LerpedTy, v_uint>()) {
                return res;
            } else {
                return Format::Unpack(res);
            }
        }
    }

    template<SamplerDesc SD>
    [[gnu::pure, gnu::always_inline]] Format::LerpedTy SampleOctImplicitLod(const v_float3& dir) const {
        return SampleImplicitLod<SD>(texutil::MapOctahedron(dir));
    }

    template<SamplerDesc SD>
    [[gnu::pure, gnu::always_inline]] Format::LerpedTy SampleOctLevel(const v_float3& dir, v_float mipLevel) const {
        v_float2 uv = texutil::MapOctahedron(dir);

        v_int baseMip = simd::conv<int>(mipLevel);
        auto baseSample = SampleLevel<SD>(uv, 0, baseMip);

        v_float mipFrac = mipLevel - simd::conv<float>(baseMip);
        if (simd::any(mipFrac > 0.0f) && simd::any(baseMip < (int32_t)(MipLevels - 1))) {
            auto lowerSample = SampleLevel<SD>(uv, 0, baseMip + 1);
            return baseSample + (lowerSample - baseSample) * mipFrac;
        }
        return baseSample;
    }

    v_int GetTexelOffset(v_int x, v_int y, v_int mipLevel) const {
        if constexpr (Layout_ == TextureLayout::Linear) {
            return _mm512_dpwssds_epi32(x, y, (int)Width >> mipLevel);  // safe for all inputs within [-2**15..2**15)
        }
        if constexpr (Layout_ == TextureLayout::TiledY8) {
            v_int stride = (int)RowShift - mipLevel;
            // ternlog prevents clang from duplicating mem load for const(7)
            v_int t1;
            t1 = _mm512_ternarylogic_epi32(t1, y, v_int(7), _MM_TERNLOG_B & ~_MM_TERNLOG_C);
            return (y & 7) | (x << 3) | (t1 << stride);
        }
        if constexpr (Layout_ == TextureLayout::TiledX4) {
            v_int t1;
            t1 = _mm512_ternarylogic_epi32(t1, y, v_int(3), _MM_TERNLOG_B & ~_MM_TERNLOG_C);
            t1 = _mm512_dpwssds_epi32(x & 3, t1, (int)Width >> mipLevel);

            v_int t2 = _mm512_ternarylogic_epi32(x, y, v_int(3), (_MM_TERNLOG_A & ~_MM_TERNLOG_C) | (_MM_TERNLOG_B & _MM_TERNLOG_C));
            return t1 + (t2 << 2);
        }
        return 0;
    }
    uint32_t GetTexelOffset(uint32_t x, uint32_t y, uint32_t mipLevel) const {
        if constexpr (Layout_ == TextureLayout::Linear) {
            return x + y * (Width >> mipLevel);
        }
        if constexpr (Layout_ == TextureLayout::TiledY8) {
            uint32_t stride = RowShift - mipLevel;
            return (y & 7) | (x << 3) | ((y & ~7u) << stride);
        }
        if constexpr (Layout_ == TextureLayout::TiledX4) {
            uint32_t tileOffset = ((x & ~3u) << 2) + (y & ~3u) * (Width >> mipLevel);
            uint32_t pixelOffset = (x & 3) + ((y & 3) << 2);
            return pixelOffset + tileOffset;
        }
        return 0;
    }

private:

    // Interpolates texels overlapping the specified pixel coords (in N.LerpFracBits fixed-point). No bounds check.
    [[gnu::pure, gnu::always_inline]] Format::LerpedTy SampleLinear(v_int ixf, v_int iyf, v_int offset, v_int mipLevel) const {
        ixf = simd::max(ixf - (LerpFracMask / 2), 0);
        iyf = simd::max(iyf - (LerpFracMask / 2), 0);

        v_int ix = ixf >> LerpFracBits;
        v_int iy = iyf >> LerpFracBits;

        v_int inboundX = ((ix + 1) << mipLevel) < (int32_t)Width;
        v_int inboundY = ((iy + 1) << mipLevel) < (int32_t)Height;

        v_uint data00, data10, data01, data11;

        if constexpr (Layout_ == TextureLayout::Linear) {
            v_int indices00 = offset + GetTexelOffset(ix, iy, mipLevel);
            v_int indices01 = indices00 + (inboundY ? (int)Width : 0);
            v_uint2 lo = simd::gather2(Data, indices00);
            v_uint2 hi = simd::gather2(Data, indices01);
            data00 = lo.x, data10 = lo.y;
            data01 = hi.x, data11 = hi.y;
        } else if constexpr (Layout_ == TextureLayout::TiledY8) {
            v_int indices00 = offset + GetTexelOffset(ix, iy, mipLevel);
            data00 = simd::gather(Data + 0, indices00);
            data10 = simd::gather(Data + 8, indices00);
            v_int indices01 = offset + GetTexelOffset(ix, iy + (inboundY ? 1 : 0), mipLevel);
            data01 = simd::gather(Data + 0, indices01);
            data11 = simd::gather(Data + 8, indices01);
        } else {
            data00 = simd::gather(Data, offset + GetTexelOffset(ix + 0, iy + 0, mipLevel));
            data10 = simd::gather(Data, offset + GetTexelOffset(ix + 1, iy + 0, mipLevel));
            data01 = simd::gather(Data, offset + GetTexelOffset(ix + 0, iy + 1, mipLevel));
            data11 = simd::gather(Data, offset + GetTexelOffset(ix + 1, iy + 1, mipLevel));
        }

        if constexpr (std::is_same<Format, pixfmt::RGBA8u>()) {
            // 15-bit fraction for mulhrs
            v_uint fx = v_uint(ixf & LerpFracMask) << (15 - LerpFracBits);
            v_uint fy = v_uint(iyf & LerpFracMask) << (15 - LerpFracBits);
            fx = (fx << 16) | fx;
            fy = (fy << 16) | fy;
            fx = inboundX ? fx : 0;

            // Lerp 2 channels at the same time (RGBA -> R0B0, 0G0A)
            v_uint rbRow1 = simd::lerp16((data00 >> 0) & 0x00FF00FF, (data10 >> 0) & 0x00FF00FF, fx);
            v_uint gaRow1 = simd::lerp16((data00 >> 8) & 0x00FF00FF, (data10 >> 8) & 0x00FF00FF, fx);
            v_uint rbRow2 = simd::lerp16((data01 >> 0) & 0x00FF00FF, (data11 >> 0) & 0x00FF00FF, fx);
            v_uint gaRow2 = simd::lerp16((data01 >> 8) & 0x00FF00FF, (data11 >> 8) & 0x00FF00FF, fx);

            v_uint rbCol = simd::lerp16(rbRow1, rbRow2, fy);
            v_uint gaCol = simd::lerp16(gaRow1, gaRow2, fy);

            return rbCol | (gaCol << 8);
        } else {
            using R = Format::UnpackedTy;

            const float fracScale = 1.0f / (LerpFracMask + 1);
            v_float fx = simd::conv<float>(ixf & LerpFracMask) * fracScale;
            v_float fy = simd::conv<float>(iyf & LerpFracMask) * fracScale;
            fx = inboundX ? fx : 0;

            R colors00 = Format::Unpack(data00);
            R colors10 = Format::Unpack(data10);
            R rowA = colors00 + (colors10 - colors00) * fx;

            R colors01 = Format::Unpack(data01);
            R colors11 = Format::Unpack(data11);
            R rowB = colors01 + (colors11 - colors01) * fx;

            return rowA + (rowB - rowA) * fy;
        }
    }

    void GenerateMip(uint32_t level, uint32_t layer) {
        uint32_t w = Width >> level, h = Height >> level;
        uint32_t* srcData = &Data[(layer * LayerStride) + MipOffsets[level - 1]];
        int parentLevel = (int)(level - 1);

        for (uint32_t y = 0; y < h; y += 4) {
            for (uint32_t x = 0; x < w; x += 4) {
                v_int ix = ((int)x + TilePixelOffsets.x) << 1;
                v_int iy = ((int)y + TilePixelOffsets.y) << 1;

                auto c00 = Format::Unpack(simd::gather(srcData, GetTexelOffset(ix + 0, iy + 0, parentLevel)));
                auto c10 = Format::Unpack(simd::gather(srcData, GetTexelOffset(ix + 1, iy + 0, parentLevel)));
                auto c01 = Format::Unpack(simd::gather(srcData, GetTexelOffset(ix + 0, iy + 1, parentLevel)));
                auto c11 = Format::Unpack(simd::gather(srcData, GetTexelOffset(ix + 1, iy + 1, parentLevel)));
                auto avg = (c00 + c10 + c01 + c11) * 0.25f;

                WriteTile(Format::Pack(avg), x, y, layer, level);
            }
        }
    }
};

template<typename Tex>
inline Tex::Ptr CreateTexture(uint32_t width, uint32_t height, uint32_t maxLevels = 1, uint32_t numLayers = 1) {
    assert(maxLevels <= simd::vec_width);
    assert(Tex::Layout == TextureLayout::TiledY8 ? (std::has_single_bit(width) && std::has_single_bit(height)) : 
           Tex::Layout == TextureLayout::TiledX4 ? (width % 4 == 0 && height % 4 == 0) : true);

    uint32_t mipOffsets[16] = {};
    uint32_t layerStride = 0;
    uint32_t mip = 0;

    for (; mip < maxLevels; mip++) {
        if ((width >> mip) < 4 || (height >> mip) < 4) break;

        mipOffsets[mip] = layerStride;
        layerStride += ((width >> mip) * (height >> mip) + 63) & ~63u;  // align to 256 bytes
    }
    assert(layerStride * (uint64_t)numLayers < INT32_MAX);

    auto tex = (Tex*)_mm_malloc(sizeof(Tex) + (size_t)layerStride * numLayers * 4 + 256, 64);
    tex->Width = width;
    tex->Height = height;
    tex->RowShift = simd::tzcnt(std::max(width, 8u));
    tex->NumLayers = numLayers;
    tex->LayerStride = layerStride;

    tex->MipLevels = mip;
    memcpy(tex->MipOffsets, mipOffsets, sizeof(mipOffsets));

    tex->ScaleU = (float)width;
    tex->ScaleV = (float)height;
    tex->MaskU = (int32_t)width - 1;
    tex->MaskV = (int32_t)height - 1;

    tex->MaskLerpU = (int32_t)(width << Tex::LerpFracBits) - 1;
    tex->MaskLerpV = (int32_t)(height << Tex::LerpFracBits) - 1;
    tex->ScaleLerpU = (float)(tex->MaskLerpU + 1);
    tex->ScaleLerpV = (float)(tex->MaskLerpV + 1);

    return (typename Tex::Ptr){ tex };
}

template<pixfmt::TexelFormat F, TextureLayout L = TextureLayout::Default>
inline TexturePtr2D<F, L> CreateTexture2D(uint32_t width, uint32_t height, uint32_t maxLevels = 1, uint32_t numLayers = 1) {
    return CreateTexture<Texture2D<F, L>>(width, height, maxLevels, numLayers);
}

template<pixfmt::TexelFormat F>
inline TexturePtr2D<F, TextureLayout::TiledX4> CreateFramebuffer(uint32_t width, uint32_t height, uint32_t maxLevels = 1, uint32_t numLayers = 1) {
    return CreateTexture<Texture2D<F, TextureLayout::TiledX4>>(width, height, maxLevels, numLayers);
}

};  // namespace swr
