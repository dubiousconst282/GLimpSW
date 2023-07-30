#include "SwRast.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace swr {

Texture2D Texture2D::LoadImage(std::string_view filename, uint32_t mipLevels) {
    int width, height, channels;
    auto pixels = stbi_load(filename.data(), &width, &height, &channels, 4);

    auto tex = Texture2D(width, height, mipLevels);
    tex.SetPixels((uint32_t*)pixels, width);

    stbi_image_free(pixels);

    return tex;
}

void Framebuffer::GetPixels(uint32_t* __restrict dest, uint32_t stride) const {
    for (uint32_t y = 0; y < Height; y++) {
        for (uint32_t x = 0; x < Width; x += 4) {
            uint32_t* src = &ColorBuffer[GetPixelOffset(x, y)];

            for (uint32_t sx = 0; sx < 4; sx++) {
                dest[y * stride + x + sx] = src[sx];
            }
        }
    }
}

void Framebuffer::SaveImage(std::string_view filename) const {
    auto pixels = std::make_unique<uint32_t[]>(Width * Height);
    GetPixels(pixels.get(), Width);
    stbi_write_png(filename.data(), Width, Height, 4, pixels.get(), Width * 4);
}

};  // namespace swr