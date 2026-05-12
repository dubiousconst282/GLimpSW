#pragma once

#include "Rasterizer.h"
#include "Texture.h"

#include "Scene.h"

enum class DebugLayer { None, BaseColor, Normals, MetallicRoughness, MeshletId, TriangleId, OverdrawPixel, OverdrawQuad };

// https://google.github.io/filament/Filament.html
// https://bruop.github.io/ibl/
struct ShadingContext {
    static constexpr uint32_t NumFbLayers = 3;

    // Uniform: Per scene
    const Meshlet* Meshlets;
    const Material* Materials;
    const Light* Lights;
    uint32_t NumLights;

    // Uniform: Per instance
    uint32_t MeshletOffset;
    const v_mask* MeshletCullBitmap;

    glm::mat4 WorldToClipMat, ObjectToClipMat;
    glm::mat3 ObjectToWorldMat;

    // Uniform: Resolve pass
    swr::HdrTexture2D* SkyboxTex;
    glm::vec3 ViewPos;

    float Exposure = 1.0f;
    bool ShowPerfHeatmap = false;

    void UpdateProj(const glm::mat4& proj, glm::mat4& view, const glm::mat4& model) {
        WorldToClipMat = proj * view;
        ObjectToClipMat = WorldToClipMat * model;
        ObjectToWorldMat = model;
    }

    static uint32_t CullMeshlets(v_mask* bitmap, const Meshlet* meshlets, uint32_t count,  //
                                 const float4x4& projMat, const float4x4& viewMat, const float4x4& modelMat,
                                 const float4x4& prevViewMat,  //
                                 float2 frameSize, swr::Texture2D<swr::pixfmt::R32f>* depthMap);

    void Resolve(swr::Rasterizer& raster, swr::Framebuffer& fb);
    void ResolveDebug(swr::Rasterizer& raster, swr::Framebuffer& fb, DebugLayer layer);

    static const swr::ShaderDispatchTable& VisBufferShader, DeferredShader, OverdrawShader;
};