#include <cstdint>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "../SwRast.h"
#include "../Texture.h"
#include "../Scene.h"
#include "../Camera.h"

#include <stb_image_write.h>

namespace simd = swr::simd;
using swr::v_int, swr::v_float, swr::v_float2, swr::v_float3, swr::v_mask;

struct VisShader {
    glm::mat4 ProjMat;
    glm::mat3 ModelMat;
    const scene::Meshlet* Meshlets;
    const scene::Material* Materials;

    void ShadeMeshlet(uint32_t index, swr::ShadedMeshlet& output) const {
        auto& mesh = Meshlets[index];

        for (uint32_t i = 0; i < mesh.NumVertices; i += simd::vec_width) {
            v_float3 worldPos = { simd::load(&mesh.Positions[0][i]), simd::load(&mesh.Positions[1][i]), simd::load(&mesh.Positions[2][i]) };
            output.SetPosition(i, simd::TransformVector(ProjMat, { worldPos, 1.0f }));
        }
        output.PrimCount = mesh.NumTriangles;
        memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));
    }

    void ShadePixels(swr::Framebuffer& fb, swr::FragmentVars& vars) const {
        // Depth test
        v_float oldDepth = simd::load(&fb.DepthBuffer[vars.TileOffset]);
        vars.TileMask &= simd::movemask(vars.Depth > oldDepth);
        if (vars.TileMask == 0) return;

#if 1
        auto& mesh = Meshlets[vars.MeshletId];
        v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.Indices[0]]);
        v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.Indices[1]]);
        v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.Indices[2]]);
        v_float2 uv = vars.Interpolate(uv0, uv1, uv2);

        constexpr swr::SamplerDesc sampler = {
            .Wrap = swr::WrapMode::Repeat,
            .MagFilter = swr::FilterMode::Linear,
            .MinFilter = swr::FilterMode::Linear,
            .EnableMips = true,
        };
        auto tex = Materials[mesh.MaterialId].Texture;
        v_int color = tex->Sample<sampler>(uv.x, uv.y);
#else
        v_int color = swr::pixfmt::RGBA8u::Pack({ vars.Bary, 1 });
#endif
        fb.WriteTile(vars.TileOffset, vars.TileMask, color, vars.Depth);
    }

    static glm::vec2 UnpackHalf2x16(uint32_t x) {
        // GLM emulates this with 100 instructions.
        auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
        return { v[0], v[1] };
    }
};

// [model path] [(camera) <x> <y> <z> <yaw> <pitch>]
int main(int argc, const char** args) {
    auto scene = std::make_unique<scene::Model>(argc > 2 ? args[0] : "assets/models/Sponza/Sponza.gltf");
    auto camera = Camera{ .Position = glm::vec3(-10.41, 1.35f, 0.4), .Euler = glm::vec2(1.0f, -0.05f) };

    if (argc >= 6) {
        camera.Position = { atof(args[2]), atof(args[3]), atof(args[4]) };
        camera.Euler = { atof(args[5]), atof(args[6]) };
    }
    camera.Update({}, 0);

    auto fb = swr::Framebuffer(1920, 1080);
    auto raster = swr::Rasterizer(fb);

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.warmup(10).epochs(100);
    ctx.unit("frame").timeUnit(std::chrono::microseconds(1), "us");

    ctx.run("Render", [&]() {
        auto projMat = camera.GetProjMatrix() * camera.GetViewMatrix(true);
        auto shader = VisShader{
            .Materials = scene->Materials.data(),
        };
        fb.Clear(0, 0);

        scene->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
            shader.ProjMat = projMat * modelMat;
            shader.ModelMat = modelMat;
            shader.Meshlets = &scene->Meshlets[node.MeshletOffset];

            raster.DrawMeshlets(node.MeshletCount, shader);
            return true;
        });
    });

    auto pixels = swr::alloc_buffer<uint32_t>(fb.Width * fb.Height);
    fb.GetPixels(pixels.get(), fb.Width);
    stbi_write_png("logs/bench_view.png", (int)fb.Width, (int)fb.Height, 4, pixels.get(), (int)fb.Width * 4);
}
