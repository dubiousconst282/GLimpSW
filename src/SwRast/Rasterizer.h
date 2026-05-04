#pragma once

#include <cstring>
#include <memory>

#include "SIMD.h"

namespace swr {

struct Framebuffer {
    // Data is stored in 4x4 tiles so that rasterizer writes are trivial.
    uint32_t Width, Height, TileStride;
    uint32_t LayerStride, NumLayers;
    
    alignas(64) uint32_t Data[];

    Framebuffer() = delete;
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    uint32_t* GetColorBuffer() { return GetLayerData<uint32_t>(0); }
    float* GetDepthBuffer() { return GetLayerData<float>(1); }

    template<typename T>
    T* GetLayerData(uint32_t idx = 0) {
        assert(idx < NumLayers);
        return (T*)&Data[idx * LayerStride];
    }
    template<typename T>
    const T* GetLayerData(uint32_t idx = 0) const {
        assert(idx < NumLayers);
        return (T*)&Data[idx * LayerStride];
    }

    void Clear(uint32_t color, float depth) {
        ClearLayer(0, color);
        ClearLayer(1, simd::as<uint32_t>(depth));
    }
    void ClearLayer(uint32_t layerIdx, uint32_t value) {
        // Non-temporal fill is ~2-3x faster than memset(), but makes rasterization a bit slower. Still a small win overall.
        // https://en.algorithmica.org/hpc/cpu-cache/bandwidth/
        uint32_t* ptr = GetLayerData<uint32_t>(layerIdx);
        uint32_t* end = ptr + Width * Height;
        for (; ptr < end; ptr += 16) {
            _mm512_stream_si512(ptr, _mm512_set1_epi32((int32_t)value));
        }
        _mm_sfence();
    }

    uint32_t GetPixelOffset(uint32_t x, uint32_t y) const {
        assert(x + 3 < Width && y + 3 < Height);

        uint32_t tileOffset = ((x & ~3u) << 2) + (y & ~3u) * Width;
        uint32_t pixelOffset = (x & 3) + (y & 3) * 4;
        return pixelOffset + tileOffset;
    }
    v_int GetPixelOffset(v_int x, v_int y) const {
        v_int tileOffset = ((x & ~3) << 2) + (y & ~3) * (int)Width;
        v_int pixelOffset = (x & 3) + (y & 3) * 4;
        return pixelOffset + tileOffset;
    }
    void GetPixels(uint32_t layerIdx, uint32_t* dest, uint32_t stride) const;
};
using FramebufferPtr = std::unique_ptr<Framebuffer, simd::AlignedDeleter>;

inline FramebufferPtr CreateFramebuffer(uint32_t width, uint32_t height, uint32_t numLayers = 2) {
    assert(width % 4 == 0 && height % 4 == 0);

    uint32_t layerStride = (width * height + 63) & ~63u;
    auto fb = (Framebuffer*)_mm_malloc(sizeof(Framebuffer) + (size_t)layerStride * numLayers * 4 + 256, 64);

    fb->Width = width;
    fb->Height = height;
    fb->TileStride = width / 4;
    fb->LayerStride = layerStride;
    fb->NumLayers = numLayers;
    return FramebufferPtr(fb);
}

enum class FaceCullMode : uint8_t { None, FrontCCW, FrontCW };

struct ShadedMeshlet {
    static constexpr uint32_t MaxVertices = 64, MaxPrims = 128;

    uint8_t PrimCount = 0;
    FaceCullMode CullMode = FaceCullMode::FrontCCW;

    alignas(64) uint8_t Indices[3][MaxPrims];
    alignas(64) float Position[4][MaxVertices];

    void SetPosition(uint32_t index, v_float4 value) {
        assert(index % simd::vec_width == 0);
        simd::store(&Position[0][index], value.x);
        simd::store(&Position[1][index], value.y);
        simd::store(&Position[2][index], value.z);
        simd::store(&Position[3][index], value.w);
    }
};

SIMD_INLINE v_float BaryLerp(v_float3 bary, v_float v0, v_float v1, v_float v2) {
    return simd::fma(v0, bary.x, simd::fma(v1, bary.y, v2 * bary.z));  // 3 op per attrib + setup
    // return simd::fma(v1 - v0, bary.y, simd::fma(v2 - v0, bary.z, v0));   // 4 op per attrib
}
template<typename T>
SIMD_INLINE T BaryLerp(v_float3 bary, T v0, T v1, T v2) {
    T result;
    result.x = BaryLerp(bary, v0.x, v1.x, v2.x);
    result.y = BaryLerp(bary, v0.y, v1.y, v2.y);
    if constexpr (sizeof(T) == sizeof(v_float) * 3) result.z = BaryLerp(bary, v0.z, v1.z, v2.z);
    if constexpr (sizeof(T) == sizeof(v_float) * 4) result.w = BaryLerp(bary, v0.w, v1.w, v2.w);
    return result;
}

struct FragmentVars {
    uint32_t MeshletId;
    uint8_t PrimId;
    uint8_t VertexId[3];

    size_t TileOffset;
    v_mask TileMask;

    v_float3 Bary;
    v_float Depth;

    // Interpolates vertex attributes using the fragment's barycentric weights.
    template<typename T>
    T Interpolate(T v0, T v1, T v2) const { return BaryLerp(Bary, v0, v1, v2); }

    template<typename T>
    simd::vec<T> LoadTile(const T* ptr) {
        return _mm512_load_si512(&ptr[TileOffset]);
    }
    template<typename T>
    void StoreTile(T* ptr, simd::vec<T> value) {
        _mm512_mask_store_epi32(&ptr[TileOffset], TileMask, value);
    }
};

struct TriangleClipData {
    glm::vec3 ClippedU[simd::vec_width];
    glm::vec3 ClippedV[simd::vec_width];
    uint8_t PrimId[simd::vec_width];
};
struct TrianglePacket {
    v_int Pos0, Pos1, Pos2; // Fixed-point viewport coords (2 x s16)

    v_float Z0, Z1, Z2;
    v_float W0, W1, W2;

    uint8_t VertexId[3][simd::vec_width];
    uint8_t BasePrimId;
    uint32_t MeshletId;
    const TriangleClipData* ClipData = nullptr;

    v_mask Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize, FaceCullMode cullMode);

    v_int2 GetBoundingBox() const;
    v_uint2 GetRenderBoundingBox(glm::ivec2 vpHalfSize) const;
};
struct TriangleEdgeVars {
    uint8_t VertexId[3][simd::vec_width];
    uint8_t BasePrimId;
    uint32_t MeshletId;
    const TriangleClipData* ClipData = nullptr;

    v_int Edge0, Edge1, Edge2;  // Start edge distances
    v_int A01, A12, A20;        // Edge deltas on X axis
    v_int B01, B12, B20;        // Edge deltas on Y axis

    v_float Z0, Z10, Z20;       // Z0, ((Z1, Z2) - Z0) * rcpArea 
    v_float W0, W0S, W1S, W2S;  // W0, (W0, W1, W2) * rcpArea

    void Setup(const TrianglePacket& tris, glm::ivec2 vpHalfSize);
};

template<typename TContext, auto MeshFn, auto FragFn>
concept ShaderEntryPoints = requires(TContext& ctx, ShadedMeshlet& meshlet, Framebuffer& fb, FragmentVars& vars) {
    // { (ctx.*MeshFn)(0u, meshlet) } -> std::same_as<void>;
    // { (ctx.*FragFn)(fb, vars) } -> std::same_as<void>;
    { (*MeshFn)(ctx, 0u, meshlet) } -> std::same_as<void>;
    { (*FragFn)(ctx, fb, vars) } -> std::same_as<void>;
};

struct ShaderDispatchTable {
    void (*MeshFn)(void* ctx, uint32_t idx, ShadedMeshlet& meshlet) = nullptr;
    void (*DrawFn)(void* ctx, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) = nullptr;
    void (*DrawClippedFn)(void* ctx, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) = nullptr;

    ShaderDispatchTable() = default;
};
struct ShaderBinding {
    const ShaderDispatchTable& Dispatch;
    void* Context;
};

// Internal
struct BinQueue;
struct ThreadedRunner;

struct Rasterizer {
    static constexpr uint32_t MaxRenderSize = 2896;  // Determined by integer overflow bound around edge equations. 2**(31/2 - 4)

    // Debug
    bool EnableBinning = true;    // Force use of unbinned rasterizer (may artifact with multi threads)
    bool EnableClipping = true;   // Disable non-trivial clipping
    bool EnableGuardband = true;  // Disable clipping guardband

    Rasterizer(uint32_t numThreads = 0);
    ~Rasterizer();

    void DrawMeshlets(Framebuffer& fb, uint32_t count, ShaderBinding shader);

    // Dispatch a number of invocations to callback in parallel.
    template<typename TCallback>
    void Dispatch(uint32_t countX, uint32_t countY, TCallback&& cb) {
        Dispatch(countX, countY, [](void* pUser, uint32_t x, uint32_t y) { (*(TCallback*)pUser)(x, y); }, &cb);
    }

    void Dispatch(uint32_t countX, uint32_t countY, void (*cb)(void* pUser, uint32_t x, uint32_t y), void* pUser);

    // Iterates over framebuffer tiles in parallel. `cb(uint2 tileBasePos, v_int tilePos, v_float2 tileUV)`
    template<typename TCallback>
    void DispatchPass(Framebuffer& fb, TCallback&& cb) {
        glm::vec2 scaleUV = 2.0f / glm::vec2(fb.Width, fb.Height);
        glm::vec2 centerUV = 0.5f * scaleUV - 1.0f;

        Dispatch((fb.Width + 31) / 32, (fb.Height + 31) / 32, [&](uint32_t x, uint32_t y) {
            for (int sy = 0; sy < 32; sy += 4) {
                for (int sx = 0; sx < 32; sx += 4) {
                    int x0 = (int)x * 32 + sx;
                    int y0 = (int)y * 32 + sy;
                    if (x0 >= fb.Width || y0 >= fb.Height) continue;

                    v_int2 tilePos = v_int2(x0 + FragPixelOffsetsX, y0 + FragPixelOffsetsY);
                    v_float2 tileUV = conv<float>(tilePos) * scaleUV + centerUV;
                    [[clang::always_inline]] cb(glm::uvec2(x0, y0), tilePos, tileUV);
                }
            }
        });
    }

    uint32_t GetThreadCount() const;
    void SetThreadCount(uint32_t count);

    static constexpr v_int FragPixelOffsetsX = simd::lane_idx<v_int> & 3;
    static constexpr v_int FragPixelOffsetsY = simd::lane_idx<v_int> >> 2;

    template<auto FragFn, typename TContext, bool IsClipped>
    static void DrawTriangle(TContext& ctx, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
        [[assume(i < simd::vec_width)]];  // avoids masking on vector indexing

        uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
        uint16_t offsetExtentX = boundRect >> 30, offsetExtentY = boundRect >> 46;  // (max - min) / 4 * 16

        // Using offset for loop control saves 2 registers and a bit of sALU
        size_t offset = (minX * 4) + (minY * 4) * fb.TileStride;
        size_t endOffsetY = (minY * 4 + offsetExtentY) * fb.TileStride;
        size_t offsetIncY = fb.TileStride * 16 - offsetExtentX;

        // Barycentric coordinates at start of row
        v_int3 edgeStepX = v_int3(tri.A12[i], tri.A20[i], tri.A01[i]);
        v_int3 edgeStepY = v_int3(tri.B12[i], tri.B20[i], tri.B01[i]);
        v_int3 edgeOrigin = v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
        edgeOrigin += edgeStepX * (minX + FragPixelOffsetsX);
        edgeOrigin += edgeStepY * (minY + FragPixelOffsetsY);

        edgeStepX *= 4, edgeStepY *= 4;

        uint32_t primId = IsClipped ? tri.ClipData->PrimId[i] : i;

        FragmentVars vars = {
            .MeshletId = tri.MeshletId,
            .PrimId = uint8_t(tri.BasePrimId + primId),
            .VertexId = { tri.VertexId[0][primId], tri.VertexId[1][primId], tri.VertexId[2][primId] },
        };

        [[assume(offset < endOffsetY)]];  // avoids redundant early bounds check

        for (; offset < endOffsetY; offset += offsetIncY) {
            v_int3 edge = edgeOrigin;

            size_t endOffsetX = offset + offsetExtentX;
            [[assume(offset < endOffsetX)]];

            for (; offset < endOffsetX; offset += 16) {
                // v_mask tileMask = simd::movemask((a | b | c) >= 0); // llvm hoists const(-1) into extra zmm register
                v_mask tileMask = _mm512_movepi32_mask(
                    _mm512_ternarylogic_epi32(edge.x, edge.y, edge.z, ~(_MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C)));

                if (tileMask != 0) {
                    v_float u = simd::conv<float>(edge.y);
                    v_float v = simd::conv<float>(edge.z);

                    vars.Depth = simd::fma(u, tri.Z10[i], simd::fma(v, tri.Z20[i], tri.Z0[i]));
                    vars.TileOffset = offset;
                    vars.TileMask = tileMask;

                    // Perspective correction - https://stackoverflow.com/a/24460895
                    // (Relying on code motion to move this after depth-test branch in shader, not optimal.)
                    v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                    v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                    v_float pw2 = tri.W2S[i];
                    v_float w = simd::fma(u, pw1, simd::fma(v, pw2, pw0));
                    v_float rcpW = simd::approx_rcp(w);
                    // Newton-Raphson refinement, prevents some minor wavy/stripe patterns during closeups
                    rcpW *= simd::fma(-w, rcpW, 2.0f);
                    u *= pw1 * rcpW;
                    v *= pw2 * rcpW;

                    if (IsClipped) {
                        glm::vec3 ru = tri.ClipData->ClippedU[i];
                        glm::vec3 rv = tri.ClipData->ClippedV[i];
                        v_float cu = simd::fma(u, ru.y, simd::fma(v, ru.z, ru.x));
                        v_float cv = simd::fma(u, rv.y, simd::fma(v, rv.z, rv.x));
                        u = cu, v = cv;
                    }
                    vars.Bary = { 1 - u - v, u, v };

                    [[clang::always_inline]] FragFn(ctx, fb, vars);
                }
                edge += edgeStepX;
            }
            edgeOrigin += edgeStepY;
        }
    }

private:
    std::unique_ptr<BinQueue> _queue;
    std::unique_ptr<ThreadedRunner> _runner;

    void DrawMeshletsST(Framebuffer& fb, uint32_t count, ShaderBinding shader);

    bool DistributeToBins(uint32_t workerId, uint32_t packetIdx, v_mask acceptMask, v_uint triMinPos, v_uint triMaxPos);
    void RasterizeBin(Framebuffer& fb, ShaderBinding shader, uint32_t binId);

    void WorkerFn(uint32_t workerId);
};

template<auto MeshFn, auto FragFn, typename TContext> requires(ShaderEntryPoints<TContext, MeshFn, FragFn>)
ShaderDispatchTable GetDispatchTable() {
    ShaderDispatchTable d;
    d.MeshFn = [](void* ctx, uint32_t idx, ShadedMeshlet& meshlet) {  //
        MeshFn(*(TContext*)ctx, idx, meshlet);
    };
    d.DrawFn = [](void* ctx, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {  //
        Rasterizer::DrawTriangle<FragFn, TContext, false>(*(TContext*)ctx, fb, tri, i, boundRect);
    };
    d.DrawClippedFn = [](void* ctx, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {  //
        Rasterizer::DrawTriangle<FragFn, TContext, true>(*(TContext*)ctx, fb, tri, i, boundRect);
    };
    return d;
}

enum class PerfCounter {
    TrianglesProcessed,  // Meshlet primitive count
    TrianglesRasterized,
    TrianglesClipped,
    BinQueueFlushes,  // Number of bins rasterized

    DrawTime,
    ResolveTime,

    ShadowTime,
    FrameTime,

    Count_,
};

namespace perf {

extern uint64_t g_GlobalAccum[];
extern uint64_t g_RunningAvg[];
extern thread_local uint64_t g_LocalAccum[];

inline uint64_t GetCurrent(PerfCounter key) { return g_GlobalAccum[(int)key]; }
inline uint64_t GetAverage(PerfCounter key) { return g_RunningAvg[(int)key]; }

void Reset();

uint64_t GetTimestamp();
void FlushThreadCounters();

};  // namespace perf

#define SWR_PERF_INC(key, amount) (swr::perf::g_LocalAccum[(int)swr::PerfCounter::key] += amount)
#define SWR_PERF_BEGIN(key) uint64_t _pc_##key = swr::perf::GetTimestamp()
#define SWR_PERF_END(key) (swr::perf::g_LocalAccum[(int)swr::PerfCounter::key##Time] += swr::perf::GetTimestamp() - _pc_##key)

};  // namespace swr