#include <cstdint>

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <tracy/Tracy.hpp>

#include "../Rasterizer.h"
#include "../Texture.h"
#include "../Scene.h"
#include "../Camera.h"
#include "../Shading.h"

#include <stb_image_write.h>

// [model path] [(camera) <x> <y> <z> <yaw> <pitch>]
//
// ./build/src/SwRast/BenchmarkApp "../GraphicsAssets/Bistro_flat.glb" -8.87356 4.07373 2.37185 1.2531 -0.2071
int main(int argc, const char** args) {
    Scene scene;
    scene.ImportGltf(argc >= 2 ? args[1] : "assets/models/Sponza/Sponza.gltf");

    auto camera = Camera { .Position = { -8.80000, 1.30000, -0.25000 }, .Euler = { 1.5411, -0.1024 } };

    if (argc >= 6) {
        camera.Position = { atof(args[2]), atof(args[3]), atof(args[4]) };
        camera.Euler = { atof(args[5]), atof(args[6]) };
    }

    uint2 fbSize = uint2(1920, 1080);
    auto depthFb = swr::CreateFramebuffer<swr::pixfmt::R32f>(fbSize.x, fbSize.y);
    auto colorFb = swr::CreateFramebuffer<swr::pixfmt::RGBA8u>(fbSize.x, fbSize.y, 1);

    camera.Update({ .DeltaTime = 1 / 60.0, .DisplaySize = float2(fbSize) }, 0);

    auto instances = std::vector<InstanceData>(1);
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
    if (scene.Lights.size() == 0) {
        uint32_t offset = scene.Lights.AllocRange(1);

        scene.Lights[offset] = {
            .Type = Light::kTypeDirectional,
            .Direction = glm::normalize(-float3(0.589494, 0.684509, 0.428906)),
            .Color = float3(1, 1, 1),
            .Intensity = 5.0,
        };
    }

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.warmup(10).epochs(50);
    ctx.unit("frame").timeUnit(std::chrono::microseconds(1), "us");
    ctx.relative(true);

    ShadingContext shader = {
        .DepthBuffer = depthFb.get(),
        .ColorBuffer = colorFb.get(),
        .Materials = scene.Materials.data(),
        .Lights = scene.Lights.data(),
        .NumLights = scene.Lights.size(),
        .ViewPos = camera.ViewPosition,
        .WorldToClipMat = camera.GetProjMatrix() * camera.GetViewMatrix(true),
        .Exposure = 1.0f,
    };

    auto raster = swr::Rasterizer(1);
    raster.SetViewport(fbSize);

    ctx.run("Rasterize", [&]() {
        FrameMark;
        colorFb->Clear(0);
        depthFb->Clear(0);

        for (uint32_t i = 1; i < instances.size(); i++) {
            shader.InstanceMeshlets = instances[i].Meshlets;
            shader.InstanceId = i;
            shader.ObjectToWorldMat = instances[i].Transform;
            shader.ObjectToClipMat = shader.WorldToClipMat * float4x4(instances[i].Transform);

            raster.DrawMeshlets(instanceCounts[i - 1], { ShadingContext::VisBufferShader, &shader });
        }
        // swr::perf::EndFrame();
        // printf("Tris=%ld\n", swr::perf::GetCurrent("RastTrisDrawn"));
    });

    auto resolvedColorFb = swr::CreateFramebuffer<swr::pixfmt::RGBA8u>(fbSize.x, fbSize.y);
    shader.ColorBuffer = resolvedColorFb.get();
    shader.Instances = instances.data();

    ctx.run("VisResolve", [&]() {
        FrameMark;
        memcpy(resolvedColorFb->Data, colorFb->Data, colorFb->LayerStride * 4);
        shader.Resolve(raster);
    });

    auto pixels = simd::alloc_buffer<uint32_t>(fbSize.x * fbSize.y);
    resolvedColorFb->GetPixels(pixels.get(), fbSize.x);
    stbi_write_png("logs/bench_view.png", (int)fbSize.x, (int)fbSize.y, 4, pixels.get(), (int)fbSize.x * 4);
}
