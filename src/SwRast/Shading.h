#pragma once

#include "Rasterizer.h"
#include "Texture.h"

#include "Scene.h"
#include "Voxel.h"

enum class DebugLayer {
    None, BaseColor, Normals, MetallicRoughness,
    MeshletId, TriangleId,
    OverdrawPixel, OverdrawQuad,
    VisResolveHeatmap, RaycastHeatmap,
    VoxelGrid,
};
static const char* kDebugLayerNames[] = {
    "None", "BaseColor", "Normals", "MetallicRoughness",
    "MeshletId", "TriangleId",
    "OverdrawPixel", "OverdrawQuad",
    "VisResolveHeatmap", "RaycastHeatmap",
    "VoxelGrid",
};

struct InstanceData {
    const Meshlet* Meshlets;
    glm::mat4x3 Transform;
};

// https://google.github.io/filament/Filament.html
// https://bruop.github.io/ibl/
struct ShadingContext {
    static constexpr uint32_t  //
        VisbMeshletBits = 11,  //
        VisbDrawShift = VisbMeshletBits + std::countr_zero(swr::ShadedMeshlet::MaxPrims),
        MaxMeshletsPerDraw = 1 << VisbMeshletBits,  //
        MaxDrawCalls = 1 << (32 - VisbDrawShift);

    // Uniform: Per frame
    swr::Framebuffer<swr::pixfmt::R32f>* DepthBuffer;
    swr::Framebuffer<swr::pixfmt::RGBA8u>* ColorBuffer;
    const Material* Materials;
    const Light* Lights;
    uint32_t NumLights;

    // Uniform: Per instance
    const Meshlet* InstanceMeshlets;
    const v_mask* InstanceCullBitmap;
    uint32_t InstanceId;

    glm::mat4 WorldToClipMat, ObjectToClipMat;
    glm::mat3 ObjectToWorldMat;

    // Uniform: Resolve pass
    const InstanceData* Instances;
    swr::HdrTexture2D* SkyboxTex;
    BrickMap* BrickMap = nullptr;
    glm::vec3 ViewPos;

    float Exposure = 1.0f;
    bool ShowPerfHeatmap = false;

    static uint32_t CullMeshlets(v_mask* bitmap, const Meshlet* meshlets, uint32_t count,  //
                                 const float4x4& projMat, const float4x4& viewMat, const float4x4& modelMat,
                                 const float4x4& prevViewMat,  //
                                 float2 frameSize, swr::Texture2D<swr::pixfmt::R32f>* depthMap);

    void Resolve(swr::Rasterizer& raster);
    void ResolveDebug(swr::Rasterizer& raster, DebugLayer layer);

    static const swr::ShaderDispatchTable VisBufferShader, DeferredShader, OverdrawShader;
};