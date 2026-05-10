#include "Rasterizer.h"
#include "Texture.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma clang diagnostic pop

namespace swr {

StbImage StbImage::Load(const char* path, PixelType type) {
    int width, height;
    uint8_t* pixels = type == PixelType::RGBA_U8 ? (uint8_t*)stbi_load(path, &width, &height, nullptr, 4) :
                      type == PixelType::RGB_F32 ? (uint8_t*)stbi_loadf(path, &width, &height, nullptr, 4) :
                                                   nullptr;

    if (pixels == nullptr) {
        throw std::runtime_error("Failed to load image");
    }
    return {
        .Width = (uint32_t)width,
        .Height = (uint32_t)height,
        .Type = type,
        .Data = { pixels, stbi_image_free },
    };
}

namespace texutil {

RgbaTexture2D::Ptr LoadImage(const char* path, uint32_t mipLevels) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(path, &width, &height, &channels, 4);

    auto tex = CreateTexture<RgbaTexture2D>((uint32_t)width, (uint32_t)height, mipLevels, 1);
    tex->SetPixels(pixels, tex->Width, 0);
    tex->GenerateMips();

    stbi_image_free(pixels);
    return tex;
}

HdrTexture2D::Ptr LoadImageHDR(const char* path, uint32_t mipLevels) {
    int width, height, channels;
    float* pixels = stbi_loadf(path, &width, &height, &channels, 3);

    auto tex = CreateTexture<HdrTexture2D>((uint32_t)width, (uint32_t)height, mipLevels, 1);

    for (uint32_t y = 0; y < tex->Height; y += 4) {
        for (uint32_t x = 0; x < tex->Width; x += 4) {
            v_float3 tile;

            for (uint32_t sy = 0; sy < 4; sy++) {
                for (uint32_t sx = 0; sx < 4; sx++) {
                    uint32_t idx = (x + sx) + (y + sy) * tex->Width;
                    tile.x[sx + sy * 4] = pixels[idx * 3 + 0];
                    tile.y[sx + sy * 4] = pixels[idx * 3 + 1];
                    tile.z[sx + sy * 4] = pixels[idx * 3 + 2];
                }
            }
            tex->WriteTile(tile, x, y);
        }
    }
    stbi_image_free(pixels);

    tex->GenerateMips();

    return tex;
}
HdrTexture2D::Ptr LoadOctahedronFromPanoramaHDR(const char* path, uint32_t mipLevels) {
    auto panoTex = LoadImageHDR(path, 1);

    uint32_t faceSize = panoTex->Width;
    auto cubeTex = CreateTexture<HdrTexture2D>(faceSize, faceSize, mipLevels);

    constexpr SamplerDesc PanoSampler = {
        .Wrap = WrapMode::Repeat,
        .MagFilter = FilterMode::Linear,
        .MinFilter = FilterMode::Linear,
    };

    float scaleUV = 1.0f / (faceSize - 1);
    float centerUV = 0.5f * scaleUV;
    for (uint32_t y = 0; y < faceSize; y += 4) {
        for (uint32_t x = 0; x < faceSize; x += 4) {
            v_float u = simd::conv<float>((int32_t)x + swr::TilePixelOffsetsX) * scaleUV + centerUV;
            v_float v = simd::conv<float>((int32_t)y + swr::TilePixelOffsetsY) * scaleUV + centerUV;
            v_float3 dir = UnmapOctahedron({ u, v });

            for (uint32_t i = 0; i < simd::vec_width; i++) {
                u[i] = atan2f(dir.z[i], dir.x[i]) / simd::tau + 0.5f;
                v[i] = asinf(-dir.y[i]) / simd::pi + 0.5f;
            }

            v_float3 tile = panoTex->SampleLevel<PanoSampler>(u, v, 0, 0);
            cubeTex->WriteTile(tile, x, y);
        }
    }

    cubeTex->GenerateMips();
    return cubeTex;
}

};  // namespace texutil

void Framebuffer::GetPixels(uint32_t layerIdx, uint32_t* dest, uint32_t stride) const {
    assert(((uintptr_t)dest % 64) == 0);

    for (uint32_t y = 0; y < Height; y += 4) {
        uint32_t x = 0;
        const uint32_t* src = GetLayerData<uint32_t>(layerIdx) + GetPixelOffset(x, y);

        // 30% faster than copying one tile at a time, limited by mem bw
        for (; x + 15 < Width; x += 16, src += 64) {
            __m512i tileA = _mm512_load_si512(src + 0);
            __m512i tileB = _mm512_load_si512(src + 16);
            __m512i tileC = _mm512_load_si512(src + 32);
            __m512i tileD = _mm512_load_si512(src + 48);

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
        for (; x < Width; x += 4, src += 16) {
            __m512i tile = _mm512_load_si512(src);
            _mm_stream_si128(&dest[(y + 0) * stride + x], _mm512_extracti32x4_epi32(tile, 0));
            _mm_stream_si128(&dest[(y + 1) * stride + x], _mm512_extracti32x4_epi32(tile, 1));
            _mm_stream_si128(&dest[(y + 2) * stride + x], _mm512_extracti32x4_epi32(tile, 2));
            _mm_stream_si128(&dest[(y + 3) * stride + x], _mm512_extracti32x4_epi32(tile, 3));
        }
    }
    _mm_sfence();
}

// TODO: could reuse this for mipmap gen
struct Downsampler {
    swr::Framebuffer& Src;
    swr::Texture2D<swr::pixfmt::R32f>& Dst;

    void Load16x16(uint32_t x, uint32_t y, v_float block[16]) {
        const float* layerData = Src.GetLayerData<float>(1);

        if (x + 16 <= Src.Width && y + 16 <= Src.Height) [[likely]] {
            for (uint32_t yo = 0; yo < 4; yo++) {
                const float* src = &layerData[Src.GetPixelOffset(x, y + yo * 4)];

                for (uint32_t xo = 0; xo < 4; xo++, src += 16) {
                    block[xo + yo * 4] = simd::load(src);
                }
            }
        } else {
            for (uint32_t i = 0; i < 16; i++) block[i] = FLT_MAX;

            for (uint32_t yo = 0; yo < 4 && (y + yo * 4 < Src.Height); yo++) {
                const float* src = &layerData[Src.GetPixelOffset(x, y + yo * 4)];

                for (uint32_t xo = 0; xo < 4 && (x + xo * 4 < Src.Width); xo++, src += 16) {
                    block[xo + yo * 4] = simd::load(src);
                }
            }
        }
    }
    void Store8x8(uint32_t x, uint32_t y, uint32_t level, v_float data[4]) {
        if (Dst.Layout == TextureLayout::TiledY8) {
            uint32_t stride = Dst.RowShift - level;
            uint32_t* dst = &Dst.Data[Dst.MipOffsets[level] + Dst.GetTexelOffset(x, y, stride)];

            constexpr v_uint idx0 = { 0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29 };
            constexpr v_uint idx1 = { 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31 };
            _mm512_store_ps(&dst[0], _mm512_permutex2var_ps(data[0], idx0, data[2]));
            _mm512_store_ps(&dst[16], _mm512_permutex2var_ps(data[0], idx1, data[2]));
            _mm512_store_ps(&dst[32], _mm512_permutex2var_ps(data[1], idx0, data[3]));
            _mm512_store_ps(&dst[48], _mm512_permutex2var_ps(data[1], idx1, data[3]));
        } else {
            Dst.WriteTile(data[0], x + 0, y + 0, 0, level);
            Dst.WriteTile(data[1], x + 4, y + 0, 0, level);
            Dst.WriteTile(data[2], x + 0, y + 4, 0, level);
            Dst.WriteTile(data[3], x + 4, y + 4, 0, level);
        }
    }

    v_float Reduce8x8(v_float tile00, v_float tile10, v_float tile01, v_float tile11) {
        constexpr v_uint oddY = { 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27 };
        constexpr v_uint oddX = { 0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30 };
        v_float hy00 = _mm512_permutex2var_ps(tile00, oddY + 0, tile01);  // odd rows
        v_float hy01 = _mm512_permutex2var_ps(tile00, oddY + 4, tile01);  // even rows
        v_float hy10 = _mm512_permutex2var_ps(tile10, oddY + 0, tile11);
        v_float hy11 = _mm512_permutex2var_ps(tile10, oddY + 4, tile11);
        v_float ry0 = _mm512_min_ps(hy00, hy01);  // reduce across Y
        v_float ry1 = _mm512_min_ps(hy10, hy11);
        v_float rx0 = _mm512_permutex2var_ps(ry0, oddX + 0, ry1);
        v_float rx1 = _mm512_permutex2var_ps(ry0, oddX + 1, ry1);
        return _mm512_min_ps(rx0, rx1);
    }
    v_float Reduce16x16(uint32_t x, uint32_t y) {
        v_float A[16], B[4], C;
        [[clang::always_inline]] Load16x16(x, y, A);

        B[0] = Reduce8x8(A[0], A[1], A[4], A[5]);
        B[1] = Reduce8x8(A[2], A[3], A[6], A[7]);
        B[2] = Reduce8x8(A[8], A[9], A[12], A[13]);
        B[3] = Reduce8x8(A[10], A[11], A[14], A[15]);
        C = Reduce8x8(B[0], B[1], B[2], B[3]);

        [[clang::always_inline]] Store8x8(x >> 1, y >> 1, 0, B);

        return C;
    }
    v_float ReduceNxN(uint32_t x, uint32_t y, uint32_t level) {
        if (x >= Src.Width || y >= Src.Height) return FLT_MAX;
        if (level <= 4) return Reduce16x16(x, y);

        level--;
        uint32_t h = 1 << level;

        v_float B[4], C;
        B[0] = ReduceNxN(x + 0, y + 0, level);
        B[1] = ReduceNxN(x + h, y + 0, level);
        B[2] = ReduceNxN(x + 0, y + h, level);
        B[3] = ReduceNxN(x + h, y + h, level);
        C = Reduce8x8(B[0], B[1], B[2], B[3]);

        [[clang::always_inline]] Store8x8(x >> (level - 2), y >> (level - 2), level - 3, B);

        return C;
    }
};

void texutil::DownsampleDepth(swr::Framebuffer& fb, swr::Texture2D<swr::pixfmt::R32f>& dest) {
    uint32_t rootLevel = 32 - simd::lzcnt(std::max(fb.Width, fb.Height));
    v_float r = Downsampler{ fb, dest }.ReduceNxN(0, 0, rootLevel);
    dest.WriteTile(r, 0, 0, 0, rootLevel - 3);
}

};  // namespace swr