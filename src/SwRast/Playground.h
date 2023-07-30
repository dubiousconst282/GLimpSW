#include "SwRast.h"

struct FlatShader {
    static const uint32_t NumCustomAttribs = 5;

    struct Vertex {
        float PosX, PosY, PosZ;
        float TexU, TexV;
        int8_t NormX, NormY, NormZ;
        uint8_t R, G, B, A;
    };

    std::shared_ptr<swr::Texture2D> DiffuseTex;
    glm::mat4 ProjMat;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        auto pos = swr::VFloat4{ .w = 1.0f };
        data.ReadAttribs(&Vertex::PosX, &pos.x, 3);
        vars.Position = swr::simd::TransformVector(ProjMat, pos);

        data.ReadAttribs(&Vertex::TexU, &vars.Attribs[0], 2);
        data.ReadAttribs(&Vertex::NormX, &vars.Attribs[2], 3);
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        auto u = vars.GetSmooth(0);
        auto v = vars.GetSmooth(1);
        auto color = DiffuseTex->SampleLinear(u, v);
        // auto color = swr::VFloat4{vars.W1, vars.W2, 1.0f - vars.W1 - vars.W2, 1.0f};
        fb.WriteTile(vars.TileOffset, vars.TileMask, swr::simd::PackRGBA(color), vars.Depth);
    }
};

static uint64_t SplitMix64(uint64_t state) {
    uint64_t result = (state + 0x9E3779B97F4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

void mesh2d() {
    int w = 64, h = 64;
    float dispFactor = 0.5f;

    auto vertices = std::vector<FlatShader::Vertex>();
    auto indices = std::vector<uint16_t>();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int32_t hash = (int32_t)SplitMix64(x + y * 1234567ull);
            float dx = (hash >> 0 & 255) / 127.0f - 1.0f;
            float dy = (hash >> 8 & 255) / 127.0f - 1.0f;
            float dz = (hash >> 16 & 255) / 127.0f - 1.0f;

            if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
                dx = dy = 0;
            }

            vertices.push_back({ .PosX = (x + dx * dispFactor) / (w - 1) * 2.0f - 1.0f,
                                 .PosY = (y + dy * dispFactor) / (h - 1) * 2.0f - 1.0f,
                                 .TexU = x / (float)(w - 1),
                                 .TexV = 1.0f - y / (float)(h - 1) });

            if (x < w - 1 && y < h - 1) {
                // A----B
                // |    |
                // C----D
                // A,C,B  B,C,D
                indices.push_back((uint16_t)((x + 0) + (y + 0) * w));
                indices.push_back((uint16_t)((x + 0) + (y + 1) * w));
                indices.push_back((uint16_t)((x + 1) + (y + 0) * w));

                indices.push_back((uint16_t)((x + 1) + (y + 0) * w));
                indices.push_back((uint16_t)((x + 0) + (y + 1) * w));
                indices.push_back((uint16_t)((x + 1) + (y + 1) * w));
            }
        }
    }
    indices.reserve(indices.size() + 128);

    auto fb = std::make_shared<swr::Framebuffer>(1920, 1080);
    auto shader = FlatShader();
    shader.ProjMat = glm::ortho(-1.0f, +1.0f, +1.0f, -1.0f, -1.0f, +1.0f);
    shader.DiffuseTex = std::make_shared<swr::Texture2D>(swr::Texture2D::LoadImage("test.png"));

    auto vertexData = swr::VertexReader((uint8_t*)vertices.data(), (uint8_t*)indices.data(), indices.size(), swr::VertexReader::U16);

    swr::Rasterizer rast(fb);

    for (int32_t iter = 0; iter < 4; iter++) {
        fb->Clear(0xFF'FFFFFFu, 1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        rast.DrawIndexed(vertexData, shader);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << elapsed.count() / 1000000.0 << "ms\n";
    }

    fb->SaveImage("out_rast.png");
}

void fillRate() {
    auto tex = swr::Texture2D::LoadImage("test.png");

    auto fb = swr::Framebuffer(1920, 1080);
    float scaleU = 1.0f / fb.Width;
    float scaleV = 1.0f / fb.Height;

    auto offsetU = swr::simd::conv2f(swr::VInt::ramp() & 3) * scaleU;
    auto offsetV = swr::simd::conv2f(swr::VInt::ramp() >> 2) * scaleV;

    for (int32_t iter = 0; iter < 10; iter++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int32_t y = 0; y < fb.Height; y += 4) {
            for (int32_t x = 0; x < fb.Width; x += 4) {
                auto colors = tex.SampleNearest(x * scaleU + offsetU, y * scaleV + offsetV);
                swr::simd::PackRGBA(colors).store((int32_t*)&fb.ColorBuffer[fb.GetPixelOffset(x, y)]);
            }
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        std::cout << elapsed.count() / 1000000.0 << "ms\n";
    }

    fb.SaveImage("out.png");
}