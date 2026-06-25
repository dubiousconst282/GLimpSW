#include <cstdint>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "../Rasterizer.h"
#include "../Texture.h"
#include "../Scene.h"
#include "../Camera.h"
#include "../Shading.h"

#include <stb_image_write.h>

// [model path] [(camera) <x> <y> <z> <yaw> <pitch>]
int main(int argc, const char** args) {
    Scene scene;
    scene.ImportGltf(argc >= 2 ? args[1] : "assets/models/Sponza/Sponza.gltf");

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

    auto instances = std::vector<InstanceData>();
    auto instanceCounts = std::vector<uint32_t>();

    for (auto& model : scene.Models) {
        for (auto& node : model->Nodes) {
            if (node.MeshCount == 0) continue;

            for (uint32_t i = 0; i < node.MeshCount; i += ShadingContext::MaxMeshletsPerDraw) {
                uint32_t meshCount = std::min(node.MeshCount - i, ShadingContext::MaxMeshletsPerDraw);

                instances.push_back({ .Meshlets = &scene.Meshlets[node.MeshOffset + i], .Transform = node.GlobalTransform });
                instanceCounts.push_back(meshCount);
            }
        }
    }

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.warmup(10).epochs(100);
    ctx.unit("frame").timeUnit(std::chrono::microseconds(1), "us");

    ShadingContext shader = {
        .DepthBuffer = depthFb.get(),
        .ColorBuffer = colorFb.get(),
        .Materials = scene.Materials.data(),
        .Lights = scene.Lights.data(),
        .NumLights = scene.Lights.size(),
        .WorldToClipMat = camera.GetProjMatrix() * camera.GetViewMatrix(true),
        .ViewPos = camera.ViewPosition,
        .Exposure = 1.0f,
    };

    ctx.run("Rasterize", [&]() {
        colorFb->Clear(0);
        depthFb->Clear(0);

        for (uint32_t i = 0; i < instances.size(); i++) {
            shader.InstanceMeshlets = instances[i].Meshlets;
            shader.InstanceId = i;
            shader.ObjectToWorldMat = instances[i].Transform;
            shader.ObjectToClipMat = shader.WorldToClipMat * float4x4(instances[i].Transform);

            raster.DrawMeshlets(instanceCounts[i], { ShadingContext::VisBufferShader, &shader });
        }
    });

    auto resolvedColorFb = swr::CreateFramebuffer<swr::pixfmt::RGBA8u>(fbSize.x, fbSize.y);
    shader.ColorBuffer = resolvedColorFb.get();
    shader.Instances = instances.data();

    ctx.run("VisResolve", [&]() {
        memcpy(resolvedColorFb->Data, colorFb->Data, colorFb->LayerStride * 4);
        shader.Resolve(raster);
    });

    auto pixels = simd::alloc_buffer<uint32_t>(fbSize.x * fbSize.y);
    resolvedColorFb->GetPixels(pixels.get(), fbSize.x);
    stbi_write_png("logs/bench_view.png", (int)fbSize.x, (int)fbSize.y, 4, pixels.get(), (int)fbSize.x * 4);
}
