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
    VoxelGrid, ProbeIrradiance, ShadowCascades,
};
static const char* kDebugLayerNames[] = {
    "None", "BaseColor", "Normals", "MetallicRoughness",
    "MeshletId", "TriangleId",
    "OverdrawPixel", "OverdrawQuad",
    "VisResolveHeatmap", "RaycastHeatmap",
    "VoxelGrid", "ProbeIrradiance", "ShadowCascades"
};

struct InstanceData {
    const Meshlet* Meshlets;
    glm::mat4x3 Transform;
};

struct DrawContext {
    static constexpr uint32_t  //
        VisbMeshletBits = 12,  //
        VisbInstanceShift = VisbMeshletBits + std::countr_zero(swr::ShadedMeshlet::MaxPrims),
        MaxMeshletsPerDraw = 1 << VisbMeshletBits,  //
        MaxDrawCalls = 1 << (32 - VisbInstanceShift);

    static const swr::ShaderDispatchTable VisBufferShader, ShadowGenShader, OverdrawShader;

    uint32_t NumMeshlets;  // required by rasterizer
    uint32_t InstanceId;

    swr::Framebuffer<swr::pixfmt::R32f>* DepthBuffer;
    swr::Framebuffer<swr::pixfmt::RGBA8u>* ColorBuffer;
    const Meshlet* Meshlets;
    const Material* Materials;
    const uint32_t* CullBitmap;

    // For shadow rendering
    int16_t2 WrapOffset;
    int16_t WrapMask;
    uint16_t CascadeIdx;
    float DepthBiasScale;

    float4x4 ObjectToClipMat;
};

struct ShadingContext {
    swr::Framebuffer<swr::pixfmt::R32f>* DepthBuffer;
    swr::Framebuffer<swr::pixfmt::RGBA8u>* ColorBuffer;
    const Material* Materials;
    const Light* Lights;
    uint32_t NumLights;

    float3 ViewPos;
    float4x4 ProjMat, ViewMat;

    float4x4 WorldToClipMat;
    float3x3 ObjectToWorldMat;

    // Uniform: Resolve pass
    const InstanceData* Instances;
    swr::HdrTexture2D* SkyboxTex;
    float SkyboxIntensity = 1.0;
    float AmbientIntensity = 0.02f;

    struct ShadowClipmap* ShadowMap = nullptr;
    struct GIProbeField* ProbeField = nullptr;
    BrickMap* BrickMap = nullptr;

    float Exposure = 1.0f;
    struct AutoExposureData* AutoExp = nullptr;

    void Resolve(swr::Rasterizer& raster);
    void ResolveHalfRes(swr::Rasterizer& raster);
    void ResolveDebug(swr::Rasterizer& raster, DebugLayer layer);

    void DrawPointLightsDebug();

    void UpdateProbes(swr::Rasterizer& raster);
    void DrawProbesDebug();

    static void LoadBlueNoise();
};

struct MeshletCuller {
    float4x4 WorldToClipMat;
    
    // Optional, for occlusion culling
    float4x4 HizProjMat;
    float4x4 HizViewMat;
    float2 FrameSize;
    swr::Texture2D<swr::pixfmt::R32f>* HizBuffer = nullptr;

    // For shadow page culling
    struct ShadowClipmap* ShadowMap = nullptr;
    uint32_t ShadowCascadeLevel = 0;

    uint32_t CullMeshlets(uint32_t* outBitmap, const Meshlet* meshlets, uint32_t count, const float4x4& modelMat);
};

// https://github.com/JuanDiegoMontoya/Frogfood/wiki/Virtual-Shadow-Maps
// https://ktstephano.github.io/rendering/stratusgfx/svsm
// https://github.com/MatejSakmary/VSM_masters_thesis/blob/main/VSM_Thesis.pdf
struct ShadowClipmap {
    static constexpr uint32_t  //
        MaxCascadeSize = 2048,
        MaxCascades = 8,  // Max depth map resolution
        PageSize = 64,    //
        MaxPageGridSize = MaxCascadeSize / PageSize,
        MaxPagesPerCascade = MaxPageGridSize * MaxPageGridSize,  //
        MaxCascadeBitmapSize = (MaxPagesPerCascade + 31) / 32,
        MaxHierPageMapLevels = simd::tzcnt(PageSize) - 3;

    swr::FramebufferPtr<swr::pixfmt::R32f> DepthBuffer = nullptr;

    float3 Origin;
    float3 LightDir;

    float BaseCascadeRange = 10;
    float SampleRadius = 3;
    uint32_t SampleCount = 8;

    int2 PageWrapOffset;
    int PageWrapMask;

    float ScaleUV;
    alignas(32) float SnapOffsetX[MaxCascades];
    alignas(32) float SnapOffsetY[MaxCascades];
    
    float4x4 ProjMatrix;

    alignas(64) uint32_t DirtyPageMap[MaxCascades][MaxCascadeBitmapSize * 4 / 3 + 1]; // visible & invalid. mipped
    alignas(64) uint32_t VisiblePageMap[MaxCascades][MaxCascadeBitmapSize];
    alignas(64) uint32_t InvalidPageMap[MaxCascades][MaxCascadeBitmapSize];

    void SetResolution(uint32_t cascadeSize, uint32_t numCascades) {
        assert(cascadeSize <= MaxCascadeSize && std::has_single_bit(cascadeSize) && numCascades < MaxCascades);
        DepthBuffer = swr::CreateFramebuffer<swr::pixfmt::R32f>(cascadeSize, cascadeSize, 1, numCascades);
        ScaleUV = (cascadeSize) * 0.5f;

        PageWrapOffset = int2(0);
        PageWrapMask = (int)(cascadeSize) - 1;

        InvalidateAllPages();
    }

    v_float Sample(v_float3 worldPos, uint3 noisePos);
    v_float SampleOne(v_float3 worldPos);

    void SetPosition(float3 viewPos, float3 lightDir);
    float4x4 GetWorldToCascadeMatrix(uint32_t level);

    // Mark visible pages using frame's Hi-Z buffer.
    void MarkVisiblePages(swr::Rasterizer& raster, swr::Texture2D<swr::pixfmt::R32f>& screenHiz, uint2 frameSize,
                          const float4x4& worldToClipMat);

    void InvalidateAllPages();
    void MarkAllPagesVisible();

    void UpdateDirtyBitmap();

    constexpr uint32_t GetHierBitmapOffset(uint32_t level) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < level; i++) offset += MaxCascadeBitmapSize >> (i * 2);
        return offset;
    }

    struct PageLoc {
        v_int2 TexelPos;
        v_int2 VirtualTexelPos;
        v_int Level;
        v_uint PageIdx;
        v_float Depth;
    };
    PageLoc Project(v_float3 worldPos, int fracBits = 0);

    uint32_t GetPhysicalPageIndex(int2 texelPos, int level) {
        int2 offset = PageWrapOffset >> level & ~int(PageSize - 1);
        uint2 pagePos = uint2((texelPos - offset) & PageWrapMask) / PageSize;
        return pagePos.x + pagePos.y * MaxPageGridSize;
    }
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
    bool ForceUpdateAll = false;    // Disable culling and update all probes

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