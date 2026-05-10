#pragma once

#include "Rasterizer.h"
#include "Texture.h"

#include "Scene.h"

enum class DebugLayer { None, BaseColor, Normals, MetallicRoughness, MeshletId, TriangleId, Overdraw };

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

    glm::mat4 WorldToClipMat, ObjectToClipMat, ObjectToViewMat;
    glm::mat3 ObjectToWorldMat;
    float4 FrustumPlanes[6];

    // Occlusion culling
    float3 SphereProjFactors;  // znear, f/ar, -f
    float ObjectToWorldScale;
    float2 FramebufferSize;
    swr::Texture2D<swr::pixfmt::R32f>* OcclDepthMap = nullptr;

    // Uniform: Resolve pass
    swr::HdrTexture2D* SkyboxTex;
    glm::vec3 ViewPos;

    float Exposure = 1.0f;
    bool ShowPerfHeatmap = false;

    void UpdateProj(const glm::mat4& proj, glm::mat4& view, const glm::mat4& model) {
        WorldToClipMat = proj * view;
        ObjectToClipMat = WorldToClipMat * model;
        ObjectToViewMat = view * model;
        ObjectToWorldMat = model;

        ObjectToWorldScale = glm::length(float3(model[0]));
        SphereProjFactors = float3(proj[3][2], proj[0][0], proj[1][1]);

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

    static const swr::ShaderDispatchTable& VisBufferShader, DeferredShader, OverdrawShader;
};