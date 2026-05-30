#include <cstdint>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "../Rasterizer.h"
#include "../Texture.h"
#include "../Scene.h"
#include "../Camera.h"

#include <stb_image_write.h>

struct ShadingContext {
    swr::Framebuffer<swr::pixfmt::R32f>* DepthBuffer;
    swr::Framebuffer<swr::pixfmt::RGBA8u>* ColorBuffer;

    glm::mat4 ProjMat;
    glm::mat3 ModelMat;
    const Meshlet* Meshlets;
    const Material* Materials;
};

static glm::vec2 UnpackHalf2x16(uint32_t x) {
    auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
    return { v[0], v[1] };
}

void ShadeMeshlet(const ShadingContext& ctx, uint32_t index, swr::ShadedMeshlet& output) {
    auto& mesh = ctx.Meshlets[index];

    for (uint32_t i = 0; i < mesh.NumVertices; i += simd::vec_width) {
        v_float3 worldPos = { simd::load(&mesh.Positions[0][i]), simd::load(&mesh.Positions[1][i]), simd::load(&mesh.Positions[2][i]) };
        output.SetPosition(i, simd::mul(ctx.ProjMat, { worldPos, 1.0f }));
    }
    output.PrimCount = mesh.NumTriangles;
    memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));
}

void ShadePixels(const ShadingContext& ctx, swr::FragmentVars& vars) {
    // Depth test
    v_float oldDepth = vars.LoadTile(ctx.DepthBuffer);
    vars.TileMask &= simd::movemask(vars.Depth > oldDepth);
    if (vars.TileMask == 0) return;

#if 1
    auto& mesh = ctx.Meshlets[vars.MeshletId];
    v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[0]]);
    v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[1]]);
    v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[2]]);
    v_float2 uv = vars.Interpolate(uv0, uv1, uv2);

    constexpr swr::SamplerDesc sampler = {
        .Wrap = swr::WrapMode::Repeat,
        .MagFilter = swr::FilterMode::Linear,
        .MinFilter = swr::FilterMode::Linear,
    };
    auto tex = ctx.Materials[mesh.MaterialId].Texture;
    v_uint color = tex->SampleImplicitLod<sampler>(uv);
#else
    v_uint color = swr::pixfmt::RGBA8u::Pack({ vars.Bary, 1 });
#endif
    vars.StoreTile(ctx.DepthBuffer, vars.Depth);
    vars.StoreTile(ctx.ColorBuffer, color);
}

// [model path] [(camera) <x> <y> <z> <yaw> <pitch>]
int main(int argc, const char** args) {
    Scene scene;
    scene.ImportGltf(argc > 2 ? args[1] : "assets/models/Sponza/Sponza.gltf");

    auto camera = Camera{ .Position = { -0.46608772755098471, 8.4925445559659511, -1.7022251220187172 }, .Euler = { -3.13643026, -1.4724431 } };

    if (argc >= 6) {
        camera.Position = { atof(args[2]), atof(args[3]), atof(args[4]) };
        camera.Euler = { atof(args[5]), atof(args[6]) };
    }

    uint2 fbSize = uint2(1920, 1080);
    auto depthFb = swr::CreateFramebuffer<swr::pixfmt::R32f>(fbSize.x, fbSize.y);
    auto colorFb = swr::CreateFramebuffer<swr::pixfmt::RGBA8u>(fbSize.x, fbSize.y);

    auto raster = swr::Rasterizer(1);
    raster.EnableBinning = false;
    raster.SetViewport(fbSize);

    camera.Update({ .DeltaTime = 1 / 60.0, .DisplaySize = float2(fbSize) }, 0);

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.warmup(10).epochs(100);
    ctx.unit("frame").timeUnit(std::chrono::microseconds(1), "us");

    ctx.run("Render", [&]() {
        auto projMat = camera.GetProjMatrix() * camera.GetViewMatrix(true);
        auto shaderCtx = ShadingContext{
            .DepthBuffer = depthFb.get(),
            .ColorBuffer = colorFb.get(),
            .Materials = scene.Materials.data(),
        };
        auto shaderDispatch = swr::GetDispatchTable<&ShadeMeshlet, &ShadePixels>();

        colorFb->Clear(0);
        depthFb->Clear(0);

        for (auto& model : scene.Models) {
            for (auto& node : model->Nodes) {
                if (node.MeshCount == 0) continue;

                shaderCtx.ProjMat = projMat * float4x4(node.GlobalTransform);
                shaderCtx.ModelMat = node.GlobalTransform;
                shaderCtx.Meshlets = &scene.Meshlets[node.MeshOffset];

                raster.DrawMeshlets(node.MeshCount, { shaderDispatch, &shaderCtx });
            }
        }
    });

    auto pixels = simd::alloc_buffer<uint32_t>(fbSize.x * fbSize.y);
    colorFb->GetPixels(pixels.get(), fbSize.x);
    stbi_write_png("logs/bench_view.png", (int)fbSize.x, (int)fbSize.y, 4, pixels.get(), (int)fbSize.x * 4);
}
