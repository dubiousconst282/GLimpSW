#include "SwRast.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace swr {

Texture2D Texture2D::LoadImage(std::string_view filename, uint32_t mipLevels) {
    int width, height, channels;
    auto pixels = stbi_load(filename.data(), &width, &height, &channels, 4);

    auto tex = Texture2D((uint32_t)width, (uint32_t)height, mipLevels);
    tex.SetPixels((uint32_t*)pixels, tex.Width);

    stbi_image_free(pixels);

    return tex;
}

void Framebuffer::GetPixels(uint32_t* __restrict dest, uint32_t stride) const {
    for (uint32_t y = 0; y < Height; y += 4) {
        for (uint32_t x = 0; x < Width; x += 4) {
            // Clang is doing some really funky vectorization with this loop. Manual vectorization ftw I guess...
            // for (uint32_t sx = 0; sx < 4; sx++) {
            //     dest[y * stride + x + sx] = src[sx];
            // }
            uint32_t* src = &ColorBuffer[GetPixelOffset(x, y)];

            __m512i tile = _mm512_load_si512(src);
            _mm_storeu_si128((__m128i*)&dest[(y + 0) * stride + x], _mm512_extracti32x4_epi32(tile, 0));
            _mm_storeu_si128((__m128i*)&dest[(y + 1) * stride + x], _mm512_extracti32x4_epi32(tile, 1));
            _mm_storeu_si128((__m128i*)&dest[(y + 2) * stride + x], _mm512_extracti32x4_epi32(tile, 2));
            _mm_storeu_si128((__m128i*)&dest[(y + 3) * stride + x], _mm512_extracti32x4_epi32(tile, 3));
        }
    }
}

void Framebuffer::SaveImage(std::string_view filename) const {
    auto pixels = std::make_unique<uint32_t[]>(Width * Height);
    GetPixels(pixels.get(), Width);
    stbi_write_png(filename.data(), (int)Width, (int)Height, 4, pixels.get(), (int)Width * 4);
}

};  // namespace swr