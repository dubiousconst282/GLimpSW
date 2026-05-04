#pragma once

#include "Rasterizer.h"
#include "Texture.h"

#include "Scene.h"

enum class DebugLayer { None, BaseColor, Normals, MetallicRoughness, MeshletId, TriangleId, Overdraw };

// https://google.github.io/filament/Filament.html
// https://bruop.github.io/ibl/
struct ShadingContext {
    static constexpr uint32_t NumFbLayers = 3;

    // Uniform: Per instance
    glm::mat4 WorldToClipMat, ObjectToClipMat;
    glm::mat3 ObjectToWorldMat;
    float4 FrustumPlanes[6];
    uint32_t MeshletOffset;

    // Uniform: Per scene
    const Meshlet* Meshlets;
    const Material* Materials;
    const Light* Lights;
    uint32_t NumLights;

    // Uniform: Resolve pass
    swr::HdrTexture2D* SkyboxTex;
    glm::vec3 ViewPos;

    float Exposure = 1.0f;
    bool ShowPerfHeatmap = false;

    void UpdateProj(const glm::mat4& projView, const glm::mat4& model) {
        ObjectToClipMat = projView * model;
        WorldToClipMat = projView;
        ObjectToWorldMat = model;

        // Gribb & Hartmann
        glm::mat4 M = glm::transpose(ObjectToClipMat);
        for (int i = 0; i < 3; i++) {
            float4 planeA = M[3] + M[i];  // Left,  Bottom, Near
            float4 planeB = M[3] - M[i];  // Right, Top,    Far
            planeA /= glm::length(float3(planeA));
            planeB /= glm::length(float3(planeB));
            FrustumPlanes[i * 2 + 0] = planeA;
            FrustumPlanes[i * 2 + 1] = planeB;
        }
    }

    void Resolve(swr::Rasterizer& raster, swr::Framebuffer& fb);
    void ResolveDebug(swr::Rasterizer& raster, swr::Framebuffer& fb, DebugLayer layer);

    static swr::ShaderDispatchTable GetVisBufferShader();
    static swr::ShaderDispatchTable GetDeferredShader();
    static swr::ShaderDispatchTable GetOverdrawShader();
};