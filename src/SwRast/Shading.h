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
    VoxelGrid, ProbeIrradiance,
};
static const char* kDebugLayerNames[] = {
    "None", "BaseColor", "Normals", "MetallicRoughness",
    "MeshletId", "TriangleId",
    "OverdrawPixel", "OverdrawQuad",
    "VisResolveHeatmap", "RaycastHeatmap",
    "VoxelGrid", "ProbeIrradiance"
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

    float3 ViewPos;
    float4x4 ProjMat, ViewMat, PrevViewMat;
    float4x4 WorldToClipMat;

    swr::Framebuffer<swr::pixfmt::R32f>* ShadowDepthMap;
    float4x4 ShadowProjMatrices[4];
    float ShadowSampleRadius = 3;

    // Uniform: Per instance
    const Meshlet* InstanceMeshlets;
    const v_mask* InstanceCullBitmap;
    uint32_t InstanceId;

    float4x4 ObjectToClipMat;
    float3x3 ObjectToWorldMat;

    // Uniform: Resolve pass
    const InstanceData* Instances;
    swr::HdrTexture2D* SkyboxTex;
    float SkyboxIntensity = 1.0;

    struct GIProbeField* ProbeField = nullptr;
    BrickMap* BrickMap = nullptr;

    float Exposure = 1.0f;
    struct AutoExposureData* AutoExp = nullptr;

    uint32_t CullMeshlets(v_mask* bitmap, const Meshlet* meshlets, uint32_t count, const float4x4& modelMat,  //
                          float2 frameSize, swr::Texture2D<swr::pixfmt::R32f>* depthMap);

    void Resolve(swr::Rasterizer& raster);
    void ResolveHalfRes(swr::Rasterizer& raster);
    void ResolveDebug(swr::Rasterizer& raster, DebugLayer layer);

    void DrawPointLightsDebug();

    void UpdateProbes(swr::Rasterizer& raster);
    void DrawProbesDebug();

    static void LoadBlueNoise();

    static const swr::ShaderDispatchTable VisBufferShader, ShadowGenShader, OverdrawShader;
};

struct GIProbeField {
    static constexpr uint3 GridSize = { 16, 8, 16 };
    static constexpr uint32_t MaxCascades = 4, ProbesPerCascade = GridSize.x * GridSize.y * GridSize.z;

    float3 ProbeBaseOffset = float3(0);
    float ProbeSpacing = 1.5;

    uint32_t RaysPerProbe = 64;
    float MaxRayDist = 10;
    uint32_t MaxHysteresis = 64;
    bool MultiBounce = false;

    // Sampling
    float NormalBias = 0.12;

    float3 Positions[ProbesPerCascade];
    alignas(64) uint32_t Irradiance[ProbesPerCascade][8 * 8];  // R11_G11_B10F
    alignas(64) uint32_t Depth[ProbesPerCascade][16 * 16];     // 2x float16 (depth, depth²)

    uint8_t SampleCounts[ProbesPerCascade];            // Accumulated updates per probe, used to derive hysteresis
    uint64_t CurrActiveBitmap[ProbesPerCascade / 64];  // Has probe been sampled in the current frame?
    uint64_t PrevActiveBitmap[ProbesPerCascade / 64];

    // Re-position probes and optionally clears irradiance data.
    void Reset(bool clearProbes);
};

// https://bruop.github.io/exposure
struct AutoExposureData {
    static constexpr int NumHistBins = 32;

    float Exposure = 1.0f;
    float AvgLuminance = 0.104166f;

    // Luminance histogram (filled during resolve pass)
    float2 HistRange = float2(-4, 8);
    uint32_t Hist[NumHistBins];

    void UpdateExposure(float deltaTime, float adaptationRate) {
        uint32_t histSum = 0, histWeight = 0;

        for (uint32_t i = 0; i < NumHistBins; i++) {
            histSum += (i + 1) * Hist[i];
            histWeight += Hist[i];
        }
        float meanBin = histSum / (float)histWeight - 1;
        float rangeScale = (HistRange.y - HistRange.x) / NumHistBins;
        float Lavg = glm::exp2(meanBin * rangeScale + HistRange.x);

        float t = 1 - glm::exp(-deltaTime * adaptationRate);
        AvgLuminance = AvgLuminance + (Lavg - AvgLuminance) * t;
        Exposure = GetExposureFromAvgLuma(AvgLuminance);

        // printf("\nLuma: ");
        // for (uint32_t i = 0; i < NumHistBins; i++) printf("% 3d", Hist[i]);

        memset(Hist, 0, sizeof(Hist));
    }
    static float GetExposureFromAvgLuma(float Lavg) {
        // float ev100 = log2(Lavg * 100.0f / 12.5f);
        // float Lmax = exp2(ev100) * 1.2f;
        // return 1.0f / Lmax;
        return 1.0f / (Lavg * 9.6f);
    }
};