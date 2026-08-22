#pragma once

#include <cstring>
#include <memory>

#include "SIMD.h"

namespace swr {

enum class FaceCullMode : uint8_t { None, FrontCCW, FrontCW };

struct ShadedMeshlet {
    static constexpr uint32_t MaxVertices = 64, MaxPrims = 128;

    uint8_t PrimCount = 0;
    uint8_t FragmentShaderId = 0;  // Index of fragment shader entry point in dispatch table
    FaceCullMode CullMode = FaceCullMode::FrontCCW;

    bool ReorientToMaxAxis = false;  // Reorient triangle to dominant normal axis, to maximize scan area (hack used by voxelizer).
                                     // If enabled, index of dom axis will be written to upper 2 bits of `FragVars::VertexId[0]`.

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
    if constexpr (sizeof(T) >= sizeof(v_float) * 3) result.z = BaryLerp(bary, v0.z, v1.z, v2.z);
    if constexpr (sizeof(T) >= sizeof(v_float) * 4) result.w = BaryLerp(bary, v0.w, v1.w, v2.w);
    return result;
}

struct FragmentVars {
    uint32_t MeshletId;
    uint8_t PrimId;
    uint8_t VertexId[3];

    uint2 TilePos;
    size_t TileOffset;
    v_mask TileMask;

    v_float3 Bary;
    v_float Depth;

    // Interpolates vertex attributes using the fragment's barycentric weights.
    template<typename T>
    T Interpolate(T v0, T v1, T v2) const { return BaryLerp(Bary, v0, v1, v2); }

    template<typename T, typename F = T::Format::LerpedTy>
    F LoadTile(const T* texture) {
        return _mm512_load_si512(&texture->Data[TileOffset]);
    }
    template<typename T, typename F>
    void StoreTile(T* texture, F value, uint32_t layer = 0) {
        _mm512_mask_store_epi32(&texture->Data[TileOffset + layer * texture->LayerStride], TileMask, value);
    }
};

struct TriangleClipData {
    float3 ClippedU[simd::vec_width];
    float3 ClippedV[simd::vec_width];
    uint8_t PrimId[simd::vec_width];
};
struct TrianglePacket {
    v_int Pos0, Pos1, Pos2; // Fixed-point viewport coords (2 x s16)

    v_float Z0, Z1, Z2;
    v_float W0, W1, W2;

    uint8_t VertexId[3][simd::vec_width];
    uint8_t BasePrimId;
    uint8_t FragmentShaderId;
    uint16_t BackfacingMask;
    uint32_t MeshletId;
    const TriangleClipData* ClipData = nullptr;

    v_mask Setup(v_float4 v0, v_float4 v1, v_float4 v2, int2 vpHalfSize, FaceCullMode cullMode);

    v_uint2 GetBoundingBox(int2 vpHalfSize, bool alignToTile) const;
};
struct TriangleEdgeVars {
    uint8_t VertexId[3][simd::vec_width];
    uint8_t BasePrimId;
    uint16_t TileStride;
    uint32_t MeshletId;
    const TriangleClipData* ClipData = nullptr;

    v_int3 OriginDist;      // Edge distances at viewport origin
    v_int3 StepX, StepY;    // Edge increments on X and Y axis

    v_float Z0, Z10, Z20;   // Z0, ((Z1, Z2) - Z0) * rcpArea
    v_float W0S, W1S, W2S;  // (W0, W1, W2) * rcpArea

    void Setup(const TrianglePacket& tris, int2 vpHalfSize);
};

struct ShaderDispatchTable {
    static constexpr uint32_t MaxFragmentEntryPoints = 4;

    using MeshFnPtr = void (*)(void* ctx, uint32_t idx, ShadedMeshlet& meshlet);
    using DrawFnPtr = void (*)(void* ctx, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect);

    MeshFnPtr ShadeMeshlet = nullptr;
    DrawFnPtr DrawTriangle[MaxFragmentEntryPoints] = {};
    DrawFnPtr DrawClippedTriangle[MaxFragmentEntryPoints] = {};

    ShaderDispatchTable() = default;

    template<typename T, typename R, typename... Types>
    static std::remove_reference_t<T> GetContextType(R (*fn)(T, Types...));
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
    bool EnableBinning = true;        // Use binned rasterizer (if disabled, may artifact due to thread racing)
    bool EnableClipping = true;       // Enable non-trivial clipping
    bool EnableGuardband = true;      // Enable guardband clipping
    bool EnableBinIntersect = true;   // For large triangles, perform intersections with bins

    Rasterizer(uint32_t numThreads = 0);
    ~Rasterizer();

    uint32_t GetThreadCount() const;
    void SetThreadCount(uint32_t count);

    void SetViewport(uint2 size) {
        _viewSize = size;
        _viewHalfSize = size / 2u;
        _tileStride = size.x / 4;
    }

    void DrawMeshlets(uint32_t count, ShaderBinding shader);

    // Note: nodebug/noinline attributes are for improving readability in callstack/tracy.

    // Dispatch work to the thread pool. `cb(uint32_t workerId)`
    template<typename TCallback>
    [[gnu::nodebug]] void Dispatch(TCallback&& cb) {
        Dispatch([](void* pUser, uint32_t workerId) { [[clang::noinline]] (*(TCallback*)pUser)(workerId); }, &cb);
    }

    void Dispatch(void (*cb)(void* pUser, uint32_t workerId), void* pUser);

    // Dispatches a number of groups `ceil(numInvocs / groupSize)` over worker threads. `cb(uint4 groupRect)`
    template<typename TCallback>
    [[gnu::nodebug]] void DispatchGroups(uint2 numInvocs, uint32_t groupSize, TCallback&& cb) {
        uint32_t numGroupsX = (numInvocs.x + groupSize - 1) / groupSize;
        uint32_t numGroupsY = (numInvocs.y + groupSize - 1) / groupSize;
        uint32_t numGroups = numGroupsX * numGroupsY;
        uint32_t nextGroupId = GetThreadCount();

        Dispatch([&](uint32_t workerId) {
            for (uint32_t i = workerId; i < numGroups; i = __atomic_fetch_add(&nextGroupId, 1, __ATOMIC_RELAXED)) {
                uint32_t x0 = (i % numGroupsX) * groupSize;
                uint32_t y0 = (i / numGroupsX) * groupSize;
                uint32_t x1 = std::min(x0 + groupSize, numInvocs.x);
                uint32_t y1 = std::min(y0 + groupSize, numInvocs.y);
                uint4 rect = uint4(x0, y0, x1, y1);

                [[clang::noinline]] cb(rect);
            }
        });
    }

    // Iterates over texture tiles in parallel. `cb(uint2 tileBasePos, v_int tilePos, v_float2 tileNdc)`
    template<typename TCallback>
    void DispatchPass(uint2 numInvocs, TCallback&& cb) {
        float2 scaleNdc = 2.0f / float2(numInvocs);
        float2 centerNdc = 0.5f * scaleNdc - 1.0f;

        const uint32_t GroupSize = 128;
        uint32_t numGroupsX = (numInvocs.x + GroupSize - 1) / GroupSize;
        uint32_t numGroupsY = (numInvocs.y + GroupSize - 1) / GroupSize;
        uint32_t numGroups = numGroupsX * numGroupsY;
        uint32_t nextGroupId = GetThreadCount();

        Dispatch([&](uint32_t workerId) {
            for (uint32_t i = workerId; i < numGroups; i = __atomic_fetch_add(&nextGroupId, 1, __ATOMIC_RELAXED)) {
                uint32_t x0 = (i % numGroupsX) * GroupSize;
                uint32_t y0 = (i / numGroupsX) * GroupSize;
                uint32_t x1 = std::min(x0 + GroupSize, numInvocs.x);
                uint32_t y1 = std::min(y0 + GroupSize, numInvocs.y);

                for (uint32_t y = y0; y < y1; y += 4) {
                    for (uint32_t x = x0; x < x1; x += 4) {
                        v_int2 tilePos = v_int2((int)x + FragPixelOffsetsX, (int)y + FragPixelOffsetsY);
                        v_float2 tileNdc = conv<float>(tilePos) * scaleNdc + centerNdc;
                        [[clang::noinline]] cb(uint2(x, y), tilePos, tileNdc);
                    }
                }
            }
        });
    }

    static constexpr v_int FragPixelOffsetsX = simd::lane_idx<v_int> & 3;
    static constexpr v_int FragPixelOffsetsY = simd::lane_idx<v_int> >> 2;

    template<auto FragFn, bool IsClipped>
    static void DrawTriangle(void* ctx, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
        [[assume(i < simd::vec_width)]];  // avoids masking on vector indexing

        uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
        uint16_t offsetExtentX = boundRect >> 30, offsetExtentY = boundRect >> 46;  // (max - min) / 4 * 16

        // Using offset for loop control saves 2 registers and a bit of sALU
        // (unless shader uses `FragVars::TilePos`, which should otherwise be DCE'd)
        size_t offset = (minX * 4) + (minY * 4) * tri.TileStride;
        size_t endOffsetY = (minY * 4 + offsetExtentY) * tri.TileStride;
        size_t offsetIncY = tri.TileStride * 16 - offsetExtentX;

        // Barycentric coordinates at start of row
        v_int3 edgeStepX = v_int3(tri.StepX.x[i], tri.StepX.y[i], tri.StepX.z[i]);
        v_int3 edgeStepY = v_int3(tri.StepY.x[i], tri.StepY.y[i], tri.StepY.z[i]);
        v_int3 edgeOrigin = v_int3(tri.OriginDist.x[i], tri.OriginDist.y[i], tri.OriginDist.z[i]);
        edgeOrigin += edgeStepX * (minX + FragPixelOffsetsX);
        edgeOrigin += edgeStepY * (minY + FragPixelOffsetsY);

        edgeStepX *= 4, edgeStepY *= 4;

        uint32_t primId = IsClipped ? tri.ClipData->PrimId[i] : i;

        FragmentVars vars = {
            .MeshletId = tri.MeshletId,
            .PrimId = uint8_t(tri.BasePrimId + primId),
            .VertexId = { tri.VertexId[0][primId], tri.VertexId[1][primId], tri.VertexId[2][primId] },
            .TilePos = uint2(minX, minY),
        };

        [[assume(offset < endOffsetY)]];  // avoids redundant early bounds check

        for (; offset < endOffsetY; offset += offsetIncY, vars.TilePos.y += 4) {
            v_int3 edge = edgeOrigin;

            vars.TilePos.x = minX;
            size_t endOffsetX = offset + offsetExtentX;
            [[assume(offset < endOffsetX)]];

            for (; offset < endOffsetX; offset += 16, vars.TilePos.x += 4) {
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
                    // (Relying on code motion to move this after depth-test branch in shader, flaky.)
                    v_float pw0 = tri.W0S[i] * simd::conv<float>(edge.x);
                    v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                    v_float pw2 = tri.W2S[i];
                    v_float w = simd::fma(u, pw1, simd::fma(v, pw2, pw0));
                    v_float rcpW = simd::approx_rcp(w);
                    // Newton-Raphson refinement, prevents some minor wavy/stripe patterns during closeups
                    rcpW *= simd::fma(-w, rcpW, 2.0f);
                    // Unfortunately we also need to zap divs by zero to deal with tiny triangles that have almost no area.
                    // Masking like this keeps some notion of derivatives, in case shader samples textures.
                    u = w != 0 ? (u * pw1 * rcpW) : u;
                    v = w != 0 ? (v * pw2 * rcpW) : v;

                    if constexpr (IsClipped) {
                        float3 ru = tri.ClipData->ClippedU[i];
                        float3 rv = tri.ClipData->ClippedV[i];
                        v_float cu = simd::fma(u, ru.y, simd::fma(v, ru.z, ru.x));
                        v_float cv = simd::fma(u, rv.y, simd::fma(v, rv.z, rv.x));
                        u = cu, v = cv;
                    }
                    vars.Bary = { 1 - u - v, u, v };

                    using TContext = decltype(ShaderDispatchTable::GetContextType(FragFn));
                    [[clang::always_inline]] FragFn(*(TContext*)ctx, vars);
                }
                edge += edgeStepX;
            }
            edgeOrigin += edgeStepY;
        }
    }

private:
    uint2 _viewSize, _viewHalfSize;
    uint32_t _tileStride;
    std::unique_ptr<BinQueue> _queue;
    std::unique_ptr<ThreadedRunner> _runner;

    void DrawMeshletsST(uint32_t count, ShaderBinding shader);

    bool DistributeToBins(uint32_t workerId, uint32_t packetIdx, v_mask acceptMask, v_uint triMinPos, v_uint triMaxPos);
    void RasterizeBins(ShaderBinding shader, uint32_t workerId);

    void WorkerFn(uint32_t workerId);
};

template<auto MeshFn, auto... FragFn>
ShaderDispatchTable GetDispatchTable() {
    using TContext = decltype(ShaderDispatchTable::GetContextType(MeshFn));
    using DrawFnPtr = ShaderDispatchTable::DrawFnPtr;
    constexpr int numEP = sizeof...(FragFn);

    static_assert(std::is_invocable_v<decltype(MeshFn), TContext&, uint32_t, ShadedMeshlet&>);
    static_assert((std::is_invocable_v<decltype(FragFn), TContext&, FragmentVars&> && ...));
    static_assert(numEP < ShaderDispatchTable::MaxFragmentEntryPoints);

    DrawFnPtr drawFns[] = { &Rasterizer::DrawTriangle<FragFn, false>... };
    DrawFnPtr drawClippedFns[] = { &Rasterizer::DrawTriangle<FragFn, true>... };

    ShaderDispatchTable d;
    d.ShadeMeshlet = (ShaderDispatchTable::MeshFnPtr)MeshFn;

    // Populate all slots, to simplify reuse of mesh shader across different fragment shaders
    for (int i = 0; i < ShaderDispatchTable::MaxFragmentEntryPoints; i++) {
        d.DrawTriangle[i] = drawFns[i % numEP];
        d.DrawClippedTriangle[i] = drawClippedFns[i % numEP];
    }
    return d;
}

namespace perf {

constexpr uint32_t SampleHistSize = 32;

struct Counter {
    const char* Name;
    int64_t* GlobalAccum;
    void (*FlushLocalAccum)();

    int64_t Samples[SampleHistSize];
};

extern std::vector<Counter> CounterRegistry;
extern uint32_t FrameNo;

int64_t RegisterCounter(const char* name, int64_t* globalAccum, void (*flushLocalAccum)());

Counter* GetCounter(const char* name);

inline int64_t GetCurrent(const char* name) {
    Counter* pc = GetCounter(name);
    return pc ? pc->Samples[FrameNo % SampleHistSize] : 0;
}
double GetAverage(const char* name);

void EndFrame();

int64_t GetTimestamp();
void FlushThreadCounters();

};  // namespace perf

#define SWR_PERF_DEF_COUNTER(name)                                                              \
    static thread_local int64_t _swr_lpc_##name;                                                \
    static int64_t _swr_gpc_##name = swr::perf::RegisterCounter(#name, &_swr_gpc_##name, []() { \
        __atomic_fetch_add(&_swr_gpc_##name, _swr_lpc_##name, __ATOMIC_RELAXED);                \
        _swr_lpc_##name = 0;                                                                    \
    });

#define SWR_PERF_INC(name, amount) (_swr_lpc_##name += amount)
#define SWR_PERF_BEGIN(name) int64_t _swr_ts1_##name = swr::perf::GetTimestamp()
#define SWR_PERF_END(name) (_swr_lpc_##name += (swr::perf::GetTimestamp() - _swr_ts1_##name))

};  // namespace swr