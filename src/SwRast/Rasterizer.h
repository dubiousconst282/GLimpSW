#pragma once

#include <cstring>
#include <memory>

#include "SIMD.h"

namespace swr {

struct Framebuffer {
    // Data is stored in 4x4 tiles so that rasterizer writes are trivial.
    static constexpr uint32_t TileSize = 4, TileShift = 2, TileMask = TileSize - 1, TileNumPixels = TileSize * TileSize;

    uint32_t Width, Height, TileStride;
    uint32_t AttachmentStride, NumAttachments;

    // TODO: experiment with fast clears for depth
    simd::AlignedBuffer<uint32_t> ColorBuffer;
    simd::AlignedBuffer<float> DepthBuffer;
    simd::AlignedBuffer<uint8_t> AttachmentBuffer;

    Framebuffer(uint32_t width, uint32_t height, uint32_t numAttachments = 0) {
        Width = (width + TileMask) & ~TileMask;
        Height = (height + TileMask) & ~TileMask;
        TileStride = Width / TileSize;
        AttachmentStride = (Width * Height + 63) & ~63u;
        NumAttachments = numAttachments;

        ColorBuffer = simd::alloc_buffer<uint32_t>(Width * Height);
        DepthBuffer = simd::alloc_buffer<float>(Width * Height);
        AttachmentBuffer = simd::alloc_buffer<uint8_t>(AttachmentStride * numAttachments);
    }

    void Clear(uint32_t color, float depth) {
        FillBuffer(ColorBuffer.get(), color);
        ClearDepth(depth);
    }
    void ClearDepth(float depth) { FillBuffer(DepthBuffer.get(), std::bit_cast<uint32_t>(depth)); }

    uint32_t GetPixelOffset(uint32_t x, uint32_t y) const {
        assert(x + 3 < Width && y + 3 < Height);
        uint32_t tileId = (x >> TileShift) + (y >> TileShift) * TileStride;
        uint32_t pixelOffset = (x & TileMask) + (y & TileMask) * TileSize;
        return tileId * TileNumPixels + pixelOffset;
    }
    v_int GetPixelOffset(v_int x, v_int y) const {
        v_int tileId = (x >> TileShift) + (y >> TileShift) * (int32_t)TileStride;
        v_int pixelOffset = (x & TileMask) + (y & TileMask) * TileSize;
        return tileId * TileNumPixels + pixelOffset;
    }

    void WriteTile(uint32_t offset, uint16_t mask, v_uint color, v_float depth) {
        _mm512_mask_store_epi32(&ColorBuffer[offset], mask, color);
        _mm512_mask_store_ps(&DepthBuffer[offset], mask, depth);
    }

    template<typename T>
    T* GetAttachmentBuffer(uint32_t attachmentId, size_t offset = 0) {
        assert(attachmentId + sizeof(T) <= NumAttachments && "Missing attachment storage");
        return (T*)&AttachmentBuffer[attachmentId * AttachmentStride] + offset;
    }

    [[gnu::always_inline]] v_float SampleDepth(v_float x, v_float y) const {
        v_int ix = simd::round2i(x * (int32_t)Width);
        v_int iy = simd::round2i(y * (int32_t)Height);
        return SampleDepth(ix, iy);
    }
    [[gnu::always_inline]] v_float SampleDepth(v_int ix, v_int iy) const {
        // Twos-complement unsigned compare trick to check for both (x >= 0 && x < N) at once.
        v_int indices = GetPixelOffset(ix, iy);
        v_mask boundMask = _mm512_cmplt_epu32_mask(ix, v_int((int32_t)Width)) & _mm512_cmplt_epu32_mask(iy, v_int((int32_t)Height));
        return _mm512_mask_i32gather_ps(_mm512_set1_ps(1.0f), boundMask, indices, DepthBuffer.get(), 4);
    }
    [[gnu::always_inline]] v_uint SampleColor(v_int ix, v_int iy, v_uint defaultColor = 0) const {
        // Twos-complement unsigned compare trick to check for both (x >= 0 && x < N) at once.
        v_int indices = GetPixelOffset(ix, iy);
        v_mask boundMask = _mm512_cmplt_epu32_mask(ix, v_int((int32_t)Width)) & _mm512_cmplt_epu32_mask(iy, v_int((int32_t)Height));
        return _mm512_mask_i32gather_epi32(defaultColor, boundMask, indices, ColorBuffer.get(), 4);
    }

    void GetPixels(uint32_t* dest, uint32_t stride) const;

private:
    void FillBuffer(void* ptr, uint32_t value) {
        // Non-temporal fill is ~2-3x faster than memset(), but makes rasterization a bit slower. Still a small win overall.
        // https://en.algorithmica.org/hpc/cpu-cache/bandwidth/
        uint32_t count = Width * Height;
        for (uint32_t i = 0; i < count; i += 16) {
            _mm512_stream_si512((uint32_t*)ptr + i, _mm512_set1_epi32((int32_t)value));
        }
        _mm_sfence();
    }
};

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

struct FragmentVars {
    uint32_t MeshletId;
    uint8_t PrimId;
    uint8_t VertexId[3];

    uint32_t TileOffset;
    v_mask TileMask;

    v_float3 Bary;
    v_float Depth;

    // Interpolates vertex attributes using the current barycentric weights.
    v_float Interpolate(v_float v0, v_float v1, v_float v2) const {
        return simd::fma(v0, Bary.x, simd::fma(v1, Bary.y, v2 * Bary.z));       // 3 op per attrib + setup
        // return simd::fma(v1 - v0, Bary.y, simd::fma(v2 - v0, Bary.z, v0));   // 4 op per attrib
    }

    // Interpolates a vector of vertex attributes. `T` should be a struct containing only `v_float` fields.
    template<typename T>
    T Interpolate(T v0, T v1, T v2) const {
        constexpr uint32_t count = sizeof(T) / sizeof(v_float);

        T result;
        result.x = Interpolate(v0.x, v1.x, v2.x);
        result.y = Interpolate(v0.y, v1.y, v2.y);
        if constexpr (count >= 3) result.z = Interpolate(v0.z, v1.z, v2.z);
        if constexpr (count >= 4) result.w = Interpolate(v0.w, v1.w, v2.w);
        return result;
    }
};

template<typename T>
concept ShaderProgram =
    requires(const T& sh, ShadedMeshlet& meshlet, Framebuffer& fb, FragmentVars& vars) {
        { sh.ShadeMeshlet(0u, meshlet) } -> std::same_as<void>;
        { sh.ShadePixels(fb, vars) } -> std::same_as<void>;
    };

struct TrianglePacket {
    v_int Pos0, Pos1, Pos2; // Fixed-point viewport coords (2 x s16)

    v_float Z0, Z1, Z2;
    v_float W0, W1, W2;

    uint8_t VertexId[3][simd::vec_width];
    uint8_t BasePrimId;
    uint32_t MeshletId;
    const float* ClippedUV = nullptr;

    v_mask Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize, FaceCullMode cullMode);

    v_int2 GetBoundingBox() const;
    v_uint2 GetRenderBoundingBox(glm::ivec2 vpHalfSize) const;
};
struct TriangleEdgeVars {
    uint8_t VertexId[3][simd::vec_width];
    uint8_t BasePrimId;
    uint32_t MeshletId;
    const float* ClippedUV = nullptr;

    v_int Edge0, Edge1, Edge2;  // Start edge distances
    v_int A01, A12, A20;        // Edge deltas on X axis
    v_int B01, B12, B20;        // Edge deltas on Y axis

    v_float Z0, Z10, Z20;       // Z0, ((Z1, Z2) - Z0) * rcpArea 
    v_float W0, W0S, W1S, W2S;  // W0, (W0, W1, W2) * rcpArea

    void Setup(const TrianglePacket& tris, glm::ivec2 vpHalfSize);
};

// Internal
struct BinQueue;
struct ThreadedRunner;

struct Rasterizer {
    static constexpr uint32_t MaxRenderSize = 2048; // Determined by integer overflow bound in setup of edge equations

    bool ForceDisableBinning = false; // Force use of unbinned rasterizer (may artifact)

    Rasterizer(uint32_t numThreads = 0);
    ~Rasterizer();

    template<ShaderProgram TShader>
    void DrawMeshlets(Framebuffer& fb, uint32_t count, const TShader& shader) {
        DrawMeshlets(fb, count, {
            .Shader = &shader,
            .MeshFn = [](const void* pShader, uint32_t idx, ShadedMeshlet& meshlet) {
                ((const TShader*)pShader)->ShadeMeshlet(idx, meshlet);
            },
            .DrawFn = [](const void* pShader,  Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
                DrawTriangle(*(const TShader*)pShader, fb, tri, i, boundRect);
            },
        });
    }

    // Dispatch a number of invocations to callback in parallel.
    template<typename TCallback>
    void Dispatch(uint32_t countX, uint32_t countY, TCallback&& cb) {
        Dispatch(countX, countY, [](void* pUser, uint32_t x, uint32_t y) { (*(TCallback*)pUser)(x, y); }, &cb);
    }

    void Dispatch(uint32_t countX, uint32_t countY, void (*cb)(void* pUser, uint32_t x, uint32_t y), void* pUser);

    // Iterates over framebuffer tiles in parallel.
    template<typename TCallback>
    void DispatchPass(Framebuffer& fb, TCallback&& cb) {
        Dispatch(fb.Width / 4, fb.Height / 4, [](void* pUser, uint32_t x, uint32_t y) { (*(TCallback*)pUser)(x * 4, y * 4); }, &cb);
    }

    uint32_t GetThreadCount() const;
    void SetThreadCount(uint32_t count);

private:
    std::unique_ptr<BinQueue> _queue;
    std::unique_ptr<ThreadedRunner> _runner;

    struct ShaderDispatcher {
        const void* Shader;

        void (*MeshFn)(const void* shader, uint32_t idx, ShadedMeshlet& meshlet);
        void (*DrawFn)(const void* shader, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect);
    };

    void DrawMeshlets(Framebuffer& fb, uint32_t count, const ShaderDispatcher& shader);
    void DrawMeshletsST(Framebuffer& fb, uint32_t count, const ShaderDispatcher& shader);

    bool DistributeToBins(uint32_t workerId, uint32_t packetIdx, v_mask acceptMask, v_uint triMinPos, v_uint triMaxPos);
    void RasterizeBin(Framebuffer& fb, const ShaderDispatcher& dispatcher, uint32_t binId);

    void WorkerFn(uint32_t workerId);

    static constexpr v_int FragPixelOffsetsX = simd::lane_idx<v_int> & 3;
    static constexpr v_int FragPixelOffsetsY = simd::lane_idx<v_int> >> 2;

    template<ShaderProgram TShader>
    static void DrawTriangle(const TShader& shader, Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
        [[assume(i < simd::vec_width<v_int>)]];  // avoids masking on vector indexing

        uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
        uint16_t offsetExtentX = boundRect >> 32, offsetExtentY = boundRect >> 48;  // (max - min) / 4 * 16

        // Using offset for loop control saves 2 registers and a bit of sALU
        uint32_t offset = ((minX >> 2) + (minY >> 2) * fb.TileStride) * 16;  // fb.GetPixelOffset(minX * 4, minY * 4)
        uint32_t endOffsetY = offset + fb.TileStride * offsetExtentY - offsetExtentX;
        uint32_t offsetIncY = fb.TileStride * 16 - offsetExtentX;

        v_int stepX0 = tri.A12[i], stepX1 = tri.A20[i], stepX2 = tri.A01[i];
        v_int stepY0 = tri.B12[i], stepY1 = tri.B20[i], stepY2 = tri.B01[i];

        // Barycentric coordinates at start of row
        v_int baseX = minX + FragPixelOffsetsX, baseY = minY + FragPixelOffsetsY;
        v_int edgeY0 = tri.Edge0[i] + stepX0 * baseX + stepY0 * baseY;
        v_int edgeY1 = tri.Edge1[i] + stepX1 * baseX + stepY1 * baseY;
        v_int edgeY2 = tri.Edge2[i] + stepX2 * baseX + stepY2 * baseY;

        stepX0 *= 4, stepX1 *= 4, stepX2 *= 4;
        stepY0 *= 4, stepY1 *= 4, stepY2 *= 4;

        FragmentVars vars = {
            .MeshletId = tri.MeshletId,
            .PrimId = (uint8_t)(tri.BasePrimId + i),
            .VertexId = { tri.VertexId[0][i], tri.VertexId[1][i], tri.VertexId[2][i] },
        };

        [[assume(offset < endOffsetY)]];  // avoids redundant early bounds check

        for (; offset < endOffsetY; offset += offsetIncY) {
            v_int edgeX0 = edgeY0, edgeX1 = edgeY1, edgeX2 = edgeY2;
            uint32_t endOffsetX = offset + offsetExtentX;

            for (; offset < endOffsetX; offset += 16) {
                // v_mask tileMask = simd::movemask((edgeX0 | edgeX1 | edgeX2) >= 0);   needs one extra zmm register
                v_mask tileMask = _mm512_movepi32_mask(_mm512_ternarylogic_epi32(edgeX0, edgeX1, edgeX2, ~(_MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C)));

                if (tileMask != 0) [[unlikely]] {
                    v_float u = simd::conv<float>(edgeX1);
                    v_float v = simd::conv<float>(edgeX2);
                    vars.Depth = simd::fma(u, tri.Z10[i], simd::fma(v, tri.Z20[i], tri.Z0[i]));
                    vars.TileOffset = offset;
                    vars.TileMask = tileMask;

                    // Perspective correction - https://stackoverflow.com/a/24460895
                    // (Relying on code motion to move this after depth-test branch in shader)
                    v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                    v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                    v_float pw2 = tri.W2S[i];
                    v_float rcpW = simd::approx_rcp(simd::fma(u, pw1, simd::fma(v, pw2, pw0)));
                    u *= pw1 * rcpW;
                    v *= pw2 * rcpW;
                    vars.Bary = { 1 - u - v, u, v };

                    [[clang::always_inline]] shader.ShadePixels(fb, vars);
                }
                edgeX0 += stepX0, edgeX1 += stepX1, edgeX2 += stepX2;
            }
            edgeY0 += stepY0, edgeY1 += stepY1, edgeY2 += stepY2;
        }
    }
};

enum class PerfCounter {
    TrianglesProcessed,  // Meshlet primitive count
    TrianglesRasterized,
    TrianglesClipped,
    BinQueueFlushes,  // Number of bins rasterized

    DrawTime,
    ComposeTime,

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