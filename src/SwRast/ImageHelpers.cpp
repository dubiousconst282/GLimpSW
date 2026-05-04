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

};  // namespace swr