#include "Rasterizer.h"

#include <vector>
#include <thread>

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>

#define TracyZoneSpanBegin(ctx, name, color) if (!ctx.active) { TracyCZoneNC(zone, name, color, true); ctx = zone; }
#define TracyZoneSpanEnd(ctx) if (ctx.active) { TracyCZoneEnd(ctx); ctx.active = false; }

// #undef assert
// static void assert_fail(const char* expr) { fprintf(stderr, "Assert failure: %s\n", expr); abort(); }
// #define assert(expr) ((expr) || (__builtin_debugtrap(), assert_fail(#expr), 0))

SWR_PERF_DEF_COUNTER(RastTrisProcessed);    // Number of input triangles given by mesh shader
SWR_PERF_DEF_COUNTER(RastTrisDrawn);        // Number of triangles drawn
SWR_PERF_DEF_COUNTER(RastTrisClipped);      // Number of triangles clipped
SWR_PERF_DEF_COUNTER(RastBinsDispatched);   // Number of bins covered and dispatched to rasterizer
SWR_PERF_DEF_COUNTER(RastTotalElapsed);     // Total time spent by rasterizer

namespace swr {

static constexpr uint32_t                         // mem per batch
    MaxPacketsPerBatch = 200,                     // 125KB
    MaxClipPacketsPerBatch = 32,                  // 12.5KB
    MaxPacketsPerBin = 128, BatchesInFlight = 2,  // 254KB (at 1080p)
    BinShift = 6, BinSize = 1 << BinShift;

static constexpr uint32_t MaxBinGridSize = (Rasterizer::MaxRenderSize + BinSize - 1) / BinSize;
static constexpr uint32_t MaxRasterBinWorkers = 32;

struct alignas(64) TriangleBin {
    uint32_t Ids[MaxPacketsPerBin];
};
struct BinQueue {
    uint2 BinGridSize;
    uint16_t BinsPerFrame = 0, BinCountBatchStride = 0, NumWorkers = 0;
    uint64_t BinsPerRowRcp = 0;
    uint64_t BinWorkPattern = 0;

    std::unique_ptr<uint8_t[]> BinCounts;
    std::unique_ptr<TriangleBin[]> BinPool;
    std::unique_ptr<TrianglePacket[]> PacketPool;
    std::unique_ptr<TriangleClipData[]> ClipDataPool;

    alignas(64) uint64_t ReadyBitmap[MaxRasterBinWorkers * BatchesInFlight][(MaxBinGridSize * MaxBinGridSize + 63) / 64];

    void AllocBins(uint2 fbSize, uint32_t workerCount);
    bool InsertBin(uint32_t workerId, uint2 binPos, uint32_t packetIdx, v_mask mask);
};

enum class ClipPlane {
    Left = 0,    // X-
    Right = 1,   // X+
    Top = 2,     // Y-
    Bottom = 3,  // Y+
    Near = 4,    // Z-
    Far = 5,     // Z+
};
struct ClipCodes {
    v_mask AcceptMask;      // Triangles that are in-bounds and can be immediately rasterized.
    v_mask NonTrivialMask;  // Triangles that need to be clipped.
    uint8_t OutCodes[16];   // Planes that need to be clipped against (per-triangle).
};
struct Clipper {
    static constexpr uint32_t MaxOutPackets = 8;

    float2 GuardBandFactor;
    uint32_t OutIdx = 0;

    v_mask OutMasks[MaxOutPackets];
    TrianglePacket OutTris[MaxOutPackets];
    TriangleClipData OutData[MaxOutPackets];

    alignas(64) float Vertices[6][simd::vec_width * 3 + 16];
    alignas(64) float OutVertices[3][4][simd::vec_width];

    // Compute Cohen-Sutherland clip codes
    static ClipCodes ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2, float2 guardBandFactor);

    void Setup(v_float4 v0, v_float4 v1, v_float4 v2);

    // Clip non-trivial triangles into output buffer.
    void ClipTriangles(ClipCodes cc, int2 vpHalfSize, FaceCullMode cullMode);

    void FlushPacket(int2 vpHalfSize, FaceCullMode cullMode, bool isLast);

    // dot(pos[i].xyz, plane.norm) + pos[i].w * scale
    float GetIntersectDist(uint8_t i, uint32_t planeId, float scale) const {
        float a = Vertices[planeId / 2][i];
        if (planeId % 2) a = -a;

        return a + Vertices[3][i] * scale;
    }
};

struct ThreadedRunner {
    ThreadedRunner(uint32_t numThreads) {
        _threads.resize(numThreads - 1); // One less because Dispatch()'s caller will also take work

        for (uint32_t i = 0; i < _threads.size(); i++) {
            _threads[i] = std::jthread(&ThreadedRunner::WorkerFn, this, i + 1);
        }
    }
    ~ThreadedRunner() {
        assert(_numBusy.load(std::memory_order::acquire) == 0);
        _timestamp.store(UINT64_MAX, std::memory_order::release);
        _timestamp.notify_all();
    }

    void Dispatch(void (*cb)(void* pData, uint32_t workerId), void* pData);

    uint32_t GetThreadCount() const { return _threads.size() + 1; }

private:
    void* _jobData = nullptr;
    void (*_jobCallback)(void* pData, uint32_t workerId) = nullptr;

    std::vector<std::jthread> _threads;

    alignas(std::hardware_constructive_interference_size) std::atomic<uint64_t> _timestamp = 0;
    alignas(std::hardware_constructive_interference_size) std::atomic<uint32_t> _numBusy = 0;

    void WorkerFn(uint32_t workerId);
};

struct Barrier {
    Barrier(uint32_t threadCount) : _threadCount(threadCount), _token(threadCount) {}

    void Wait();
    bool AnyWaiting() { return uint32_t(_token.load(std::memory_order::relaxed)) != _threadCount; }

    alignas(std::hardware_constructive_interference_size) std::atomic<uint64_t> _token;
    uint32_t _threadCount;
};

Rasterizer::Rasterizer(uint32_t numThreads) {
    _queue = std::make_unique<BinQueue>();
    SetThreadCount(numThreads);
}
Rasterizer::~Rasterizer() {}

uint32_t Rasterizer::GetThreadCount() const { return _runner->GetThreadCount(); }
void Rasterizer::SetThreadCount(uint32_t count) {
    if (count == 0) {
        count = std::max(std::thread::hardware_concurrency() / 2, 1u);
    }
    _runner = std::make_unique<ThreadedRunner>(count);
}

static v_float4 GatherPos(const ShadedMeshlet& mesh, const uint8_t* indices) {
    static_assert(ShadedMeshlet::MaxVertices == 64);

    v_uint idx = _mm512_cvtepu8_epi32(_mm_loadu_epi8(indices));
    return {
        simd::gather_preload64(mesh.Position[0], idx),
        simd::gather_preload64(mesh.Position[1], idx),
        simd::gather_preload64(mesh.Position[2], idx),
        simd::gather_preload64(mesh.Position[3], idx),
    };
}

void Rasterizer::DrawMeshletsST(uint32_t count, ShaderBinding shader) {
    float2 guardBandFactor = (float)MaxRenderSize / float2(_viewSize);
    if (!EnableGuardband) guardBandFactor = float2(1.0f);

    Dispatch([&](uint32_t workerId) {
        for (uint32_t meshIdx = workerId; meshIdx < count; meshIdx += _runner->GetThreadCount()) {
            ShadedMeshlet mesh;
            shader.Dispatch.ShadeMeshlet(shader.Context, meshIdx, mesh);
            if (mesh.PrimCount == 0) continue;

            SWR_PERF_INC(RastTrisProcessed, mesh.PrimCount);

            for (uint32_t primOffset = 0; primOffset < mesh.PrimCount; primOffset += simd::vec_width) {
                v_float4 v0 = GatherPos(mesh, &mesh.Indices[0][primOffset]);
                v_float4 v1 = GatherPos(mesh, &mesh.Indices[1][primOffset]);
                v_float4 v2 = GatherPos(mesh, &mesh.Indices[2][primOffset]);

                if (mesh.ReorientToMaxAxis) [[unlikely]] {
                    v_float3 normal = simd::cross(v_float3(v1 - v0), v_float3(v2 - v0));
                    v_int domAxis = simd::abs(normal.x) > simd::max_abs(normal.y, normal.z) ? 0 :  //
                                    simd::abs(normal.y) > simd::abs(normal.z) ? 1 : 2;

                    for (uint32_t i = 0; i < 3; i++) {
                        v_float4& pos = i == 0 ? v0 : i == 1 ? v1 : v2;
                        v_float4 tmp = pos;
                        // 0 yzx
                        // 1 xzy
                        // 2 xyz
                        simd::cmov(pos.x, tmp.y, domAxis == 0);
                        simd::cmov(pos.y, tmp.z, domAxis != 2);
                        simd::cmov(pos.z, tmp.x, domAxis == 0);
                        simd::cmov(pos.z, tmp.y, domAxis == 1);
                        pos.z = pos.z * 0.5 + 0.5;
                    }
                    // Save dom axis so frag shader can unswizzle coords later
                    uint8_t* idx0 = &mesh.Indices[0][primOffset];
                    _mm_storeu_epi8(idx0, _mm_loadu_epi8(idx0) | _mm512_cvtepi32_epi8(domAxis << 6));
                }
                ClipCodes cc = Clipper::ComputeClipCodes(v0, v1, v2, guardBandFactor);

                if (primOffset + simd::vec_width > mesh.PrimCount) {
                    v_mask trailMask = (1u << (mesh.PrimCount % simd::vec_width)) - 1;
                    cc.AcceptMask &= trailMask;
                    cc.NonTrivialMask &= trailMask;
                }

                if (cc.NonTrivialMask != 0 && EnableClipping) [[unlikely]] {
                    Clipper clipper;
                    clipper.GuardBandFactor = guardBandFactor;
                    clipper.Setup(v0, v1, v2);
                    clipper.ClipTriangles(cc, _viewHalfSize, mesh.CullMode);

                    for (uint32_t i = 0; i < clipper.OutIdx; i += simd::vec_width) {
                        TrianglePacket& tris = clipper.OutTris[i / simd::vec_width];
                        v_mask drawMask = clipper.OutMasks[i / simd::vec_width];

                        TriangleEdgeVars vars;
                        vars.Setup(tris, _viewHalfSize);

                        vars.MeshletId = meshIdx;
                        vars.BasePrimId = primOffset;
                        vars.TileStride = _tileStride;
                        vars.ClipData = tris.ClipData;

                        memcpy(vars.VertexId[0], &mesh.Indices[0][primOffset], 16);
                        memcpy(vars.VertexId[1], &mesh.Indices[1][primOffset], 16);
                        memcpy(vars.VertexId[2], &mesh.Indices[2][primOffset], 16);

                        v_uint2 boundBox = tris.GetBoundingBox(_viewHalfSize, true);
                        v_uint extent = boundBox.y - boundBox.x;

                        auto drawFn = shader.Dispatch.DrawClippedTriangle[mesh.FragmentShaderId];

                        for (uint32_t i : simd::BitIter(drawMask)) {
                            uint64_t boundRect = ((uint32_t*)&boundBox.x)[i] | uint64_t(((uint32_t*)&extent)[i]) << 32;
                            drawFn(shader.Context, vars, i, boundRect);
                        }
                        SWR_PERF_INC(RastTrisDrawn, simd::popcnt(drawMask));
                    }
                }
                if (cc.AcceptMask != 0) {
                    TrianglePacket tris;
                    v_mask acceptMask = tris.Setup(v0, v1, v2, _viewHalfSize, mesh.CullMode);
                    acceptMask &= cc.AcceptMask;
                    if (acceptMask == 0) continue;

                    v_uint2 boundBox = tris.GetBoundingBox(_viewHalfSize, true);
                    v_uint extent = boundBox.y - boundBox.x;

                    TriangleEdgeVars vars;
                    vars.Setup(tris, _viewHalfSize);

                    vars.MeshletId = meshIdx;
                    vars.BasePrimId = primOffset;
                    vars.TileStride = _tileStride;

                    memcpy(vars.VertexId[0], &mesh.Indices[0][primOffset], 16);
                    memcpy(vars.VertexId[1], &mesh.Indices[1][primOffset], 16);
                    memcpy(vars.VertexId[2], &mesh.Indices[2][primOffset], 16);

                    auto drawFn = shader.Dispatch.DrawTriangle[mesh.FragmentShaderId];

                    for (uint32_t i : simd::BitIter(acceptMask)) {
                        uint64_t boundRect = ((uint32_t*)&boundBox.x)[i] | uint64_t(((uint32_t*)&extent)[i]) << 32;
                        drawFn(shader.Context, vars, i, boundRect);
                    }
                    SWR_PERF_INC(RastTrisDrawn, simd::popcnt(acceptMask));
                }
            }
        }
        perf::FlushThreadCounters();
    });
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice
v_mask TrianglePacket::Setup(v_float4 v0, v_float4 v1, v_float4 v2, int2 vpHalfSize, FaceCullMode cullMode) {
    v0 = simd::perspective_div(v0);
    v1 = simd::perspective_div(v1);
    v2 = simd::perspective_div(v2);

    BackfacingMask = simd::movemask((v2.x - v0.x) * (v1.y - v0.y) <= (v0.x - v1.x) * (v0.y - v2.y));

    v_mask visibleMask = v_mask(~0u);
    if (cullMode == FaceCullMode::FrontCCW) visibleMask = ~BackfacingMask;
    if (cullMode == FaceCullMode::FrontCW) visibleMask = BackfacingMask;

    // Cull backfacing or degenerate triangles
    if (visibleMask == 0) return 0;

    float2 fixScale = vpHalfSize * 16;
    v_int x0 = simd::round2i(v0.x * fixScale.x), y0 = simd::round2i(v0.y * fixScale.y);
    v_int x1 = simd::round2i(v1.x * fixScale.x), y1 = simd::round2i(v1.y * fixScale.y);
    v_int x2 = simd::round2i(v2.x * fixScale.x), y2 = simd::round2i(v2.y * fixScale.y);

    Pos0 = (x0 & 0xFFFF) | (y0 << 16);
    Pos1 = (x1 & 0xFFFF) | (y1 << 16);
    Pos2 = (x2 & 0xFFFF) | (y2 << 16);

    // Cull thin triangles covering no samples, or outside viewport (some of them will get here due to guard-band)
    v_uint2 boundBox = GetBoundingBox(vpHalfSize, false);
    visibleMask &= _mm512_cmpeq_epi32_mask(_mm512_movm_epi16(_mm512_cmpge_epi16_mask(boundBox.x, boundBox.y)), _mm512_set1_epi32(0));

    Z0 = v0.z, Z1 = v1.z, Z2 = v2.z;
    W0 = v0.w, W1 = v1.w, W2 = v2.w;

    return visibleMask;
}

static v_int2 UnpackS16x2(v_int packed) {
    // Slightly faster sign ext, https://fgiesen.wordpress.com/2024/10/23/zero-or-sign-extend/
    //   return x << 16 >> 16;                  2c thr
    //   return (val ^ sign_bit) - sign_bit;    0.5c thr
    // Clang can actually get rid of the subtract / embed in following ones, which makes this even faster.
    v_int x = packed & simd::uncanon<v_int>(0xFFFF);
    return { (x ^ 0x8000) - 0x8000, packed >> 16 };
}
static v_int ComputeEdge(v_int a, v_int x, v_int b, v_int y) {
    v_int w = a * x + b * y;
    // w += (a > 0 || (a == 0 && b > 0)) ? 0 : -1;  // top-left rule bias
    w += ((a == 0 ? b : a) - 1) >> 31;
    return w >> 4;                               // normalize fixed point
}
void TriangleEdgeVars::Setup(const TrianglePacket& tris, int2 vpHalfSize) {
    v_int2 v0 = UnpackS16x2(tris.Pos0);
    v_int2 v1 = UnpackS16x2(tris.Pos1);
    v_int2 v2 = UnpackS16x2(tris.Pos2);
    v_int3 stepX = v_int3(v2.y, v0.y, v1.y) - v_int3(v1.y, v2.y, v0.y);
    v_int3 stepY = v_int3(v1.x, v2.x, v0.x) - v_int3(v2.x, v0.x, v1.x);

    if (tris.BackfacingMask != 0) {
        v_int flip = simd::unmovemask(tris.BackfacingMask);
        simd::cmov(stepX, -stepX, flip);
        simd::cmov(stepY, -stepY, flip);
    }
    StepX = stepX, StepY = stepY;

    // Evaluate edge coeffs at screen origin (0.5, 0.5)
    v_int sampleX = (-vpHalfSize.x << 4) + 8, sampleY = (-vpHalfSize.y << 4) + 8;
    OriginDist.x = ComputeEdge(StepX.x, sampleX - v1.x, StepY.x, sampleY - v1.y);
    OriginDist.y = ComputeEdge(StepX.y, sampleX - v2.x, StepY.y, sampleY - v2.y);
    OriginDist.z = ComputeEdge(StepX.z, sampleX - v0.x, StepY.z, sampleY - v0.y);

    v_int det = OriginDist.x + OriginDist.y + OriginDist.z;
    v_float rcpArea = det != 0 ? 1.0f / simd::conv<float>(det) : 0.0f;

    Z0 = tris.Z0;
    Z10 = (tris.Z1 - tris.Z0) * rcpArea;
    Z20 = (tris.Z2 - tris.Z0) * rcpArea;

    W0S = tris.W0 * rcpArea;
    W1S = tris.W1 * rcpArea;
    W2S = tris.W2 * rcpArea;
}

v_uint2 TrianglePacket::GetBoundingBox(int2 vpHalfSize, bool alignToTile) const {
    v_int minPos = _mm512_min_epi16(_mm512_min_epi16(Pos0, Pos1), Pos2);
    v_int maxPos = _mm512_max_epi16(_mm512_max_epi16(Pos0, Pos1), Pos2);

    minPos = _mm512_srai_epi16(minPos + 0x0007'0007, 4);  // round to pixel center
    maxPos = _mm512_srai_epi16(maxPos + 0x0007'0007, 4);

    v_int vpSize = vpHalfSize.x | (vpHalfSize.y << 16);
    minPos = _mm512_min_epi16(_mm512_max_epi16(_mm512_add_epi16(minPos, vpSize), v_int(0)), vpSize * 2);
    maxPos = _mm512_min_epi16(_mm512_max_epi16(_mm512_add_epi16(maxPos, vpSize), v_int(0)), vpSize * 2);

    if (alignToTile) {
        minPos = (minPos + 0x0000'0000) & ~0x0003'0003;
        maxPos = (maxPos + 0x0003'0003) & ~0x0003'0003;
    }
    return v_uint2((v_uint)minPos, (v_uint)maxPos);
}

ClipCodes Clipper::ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2, float2 guardBandFactor) {
    using v_byte = uint8_t [[clang::ext_vector_type(16)]];
    using v_bool = bool [[clang::ext_vector_type(16)]];

    // Packets that are fully inside the viewport are most common, do a quick test first
    v_int inside0 = simd::max_abs(simd::max_abs(v0.x, v0.y), v0.z) < v0.w;
    v_int inside1 = simd::max_abs(simd::max_abs(v1.x, v1.y), v1.z) < v1.w;
    v_int inside2 = simd::max_abs(simd::max_abs(v2.x, v2.y), v2.z) < v2.w;
    if (simd::all(inside0 & inside1 & inside2)) [[likely]] {
        return { .AcceptMask = v_mask(~0) };
    }

    v_byte partialOutcodes = 0;       // non-zero if at least one vertex is out
    v_byte combinedOutcodes = 255;    // non-zero if all vertices are out
    v_mask trivialMask = v_mask(~0);  // inside guard band

    float bx = guardBandFactor.x, by = guardBandFactor.y;

    #pragma unroll
    for (uint32_t i = 0; i < 3; i++) {
        v_float4 vi = i == 0 ? v0 : (i == 1 ? v1 : v2);

        v_byte outcode = 0;
        outcode |= v_bool(vi.x < -vi.w) ? v_byte(1 << (int)ClipPlane::Left) : 0;
        outcode |= v_bool(vi.x > +vi.w) ? v_byte(1 << (int)ClipPlane::Right) : 0;
        outcode |= v_bool(vi.y < -vi.w) ? v_byte(1 << (int)ClipPlane::Top) : 0;
        outcode |= v_bool(vi.y > +vi.w) ? v_byte(1 << (int)ClipPlane::Bottom) : 0;
        outcode |= v_bool(vi.z < -vi.w) ? v_byte(1 << (int)ClipPlane::Near) : 0;
        outcode |= v_bool(vi.z > +vi.w) ? v_byte(1 << (int)ClipPlane::Far) : 0;

        partialOutcodes |= outcode;
        combinedOutcodes &= outcode;

        trivialMask &= simd::movemask(simd::abs(vi.x) < vi.w * bx && simd::abs(vi.y) < vi.w * by);
    }
    trivialMask &= _mm_testn_epi8_mask(partialOutcodes, _mm_set1_epi8((1 << (int)ClipPlane::Near) | (1 << (int)ClipPlane::Far)));
    v_mask acceptMask = _mm_cmpeq_epi8_mask(combinedOutcodes, _mm_set1_epi8(0));

    ClipCodes codes = {
        .AcceptMask = v_mask(acceptMask & trivialMask),
        .NonTrivialMask = v_mask(acceptMask & ~trivialMask),
    };
    memcpy(codes.OutCodes, &partialOutcodes, 16);
    return codes;
}

[[clang::always_inline]] // avoids redundant stack spill
void Clipper::Setup(v_float4 v0, v_float4 v1, v_float4 v2) {
    for (uint32_t i = 0; i < 3; i++) {
        v_float4 vi = i == 0 ? v0 : i == 1 ? v1 : v2;
        uint32_t j = i * simd::vec_width;
        simd::store(&Vertices[0][j], vi.x);
        simd::store(&Vertices[1][j], vi.y);
        simd::store(&Vertices[2][j], vi.z);
        simd::store(&Vertices[3][j], vi.w);
        simd::store(&Vertices[4][j], v_float(i == 1 ? 1 : 0));
        simd::store(&Vertices[5][j], v_float(i == 2 ? 1 : 0));
    }
    OutIdx = 0;
}
void Clipper::ClipTriangles(ClipCodes cc, int2 vpHalfSize, FaceCullMode cullMode) {
    SWR_PERF_INC(RastTrisClipped, simd::popcnt(cc.NonTrivialMask));

    for (uint32_t primId : simd::BitIter(cc.NonTrivialMask)) {
        uint8_t indices[32];
        indices[0] = primId + 0 * simd::vec_width;
        indices[1] = primId + 1 * simd::vec_width;
        indices[2] = primId + 2 * simd::vec_width;

        uint32_t vertCount = 3, nextIdx = 3 * simd::vec_width;

        // Clip against intersected planes - https://www.cubic.org/docs/3dclip.htm
        for (uint32_t planeId : simd::BitIter(cc.OutCodes[primId])) {
            uint8_t outIndices[sizeof(indices)];
            uint32_t outCount = 0;
            float planeScale = planeId < 4 ? GuardBandFactor[planeId / 2] : 1.0f;

            for (uint32_t vi = 0; vi < vertCount; vi++) {
                uint8_t ia = indices[vi], ib = indices[(vi + 1) % vertCount];
                float da = GetIntersectDist(ia, planeId, planeScale);
                float db = GetIntersectDist(ib, planeId, planeScale);

                if (da >= 0) outIndices[outCount++] = ia;

                // Calculate intersection if one of the vertices are inside
                if ((da >= 0) != (db >= 0)) {
                    float t = da / (da - db);

                    for (uint32_t k = 0; k < 6; k++) {
                        Vertices[k][nextIdx] = fmaf(Vertices[k][ib], t, fmaf(-t, Vertices[k][ia], Vertices[k][ia]));
                    }
                    outIndices[outCount++] = nextIdx++;
                }
            }
            assert(outCount <= sizeof(indices) && nextIdx <= 64);

            if (vertCount < 3) break;

            vertCount = outCount;
            memcpy(indices, outIndices, sizeof(indices));
        }
        if (vertCount < 2) continue;

        // Triangulate result polygon
        for (uint32_t vi = 0; vi < vertCount - 2; vi++) {
            uint32_t i0 = indices[0], i1 = indices[vi + 1], i2 = indices[vi + 2];
            uint32_t outLane = OutIdx % simd::vec_width;

            for (uint32_t k = 0; k < 4; k++) {
                OutVertices[0][k][outLane] = Vertices[k][i0];
                OutVertices[1][k][outLane] = Vertices[k][i1];
                OutVertices[2][k][outLane] = Vertices[k][i2];
            }

            TriangleClipData& cd = OutData[OutIdx / simd::vec_width];
            float u0 = Vertices[4][i0], v0 = Vertices[5][i0];
            cd.ClippedU[outLane] = { u0, Vertices[4][i1] - u0, Vertices[4][i2] - u0 };
            cd.ClippedV[outLane] = { v0, Vertices[5][i1] - v0, Vertices[5][i2] - v0 };
            cd.PrimId[outLane] = primId;

            if (outLane == simd::vec_width - 1) {
                FlushPacket(vpHalfSize, cullMode, false);
            }
            if (++OutIdx >= MaxOutPackets * simd::vec_width) {
                return;  // fuck it
            }
        }
    }
    if (OutIdx % simd::vec_width != 0) {
        FlushPacket(vpHalfSize, cullMode, true);
    }
}
[[clang::noinline]]
void Clipper::FlushPacket(int2 vpHalfSize, FaceCullMode cullMode, bool isLast) {
    uint32_t i = OutIdx / simd::vec_width;
    TrianglePacket& tri = OutTris[i];
    tri.ClipData = &OutData[i];

    v_float4 v0 = simd::load<v_float4>(OutVertices[0]);
    v_float4 v1 = simd::load<v_float4>(OutVertices[1]);
    v_float4 v2 = simd::load<v_float4>(OutVertices[2]);
    v_mask mask = tri.Setup(v0, v1, v2, vpHalfSize, cullMode);

    mask &= (1u << (OutIdx % simd::vec_width + !isLast)) - 1;
    OutMasks[i] = mask;

    // Rollback to avoid outputting empty packets
    // Ideally we'd skip clipping backfacing triangles, but it's not worth the effort.
    if (mask == 0) OutIdx -= (OutIdx % simd::vec_width);
}

void Rasterizer::DrawMeshlets(uint32_t meshCount, ShaderBinding shader) {
    ZoneScoped;
    SWR_PERF_BEGIN(RastTotalElapsed);

    if (!EnableBinning) {
        DrawMeshletsST(meshCount, shader);
        SWR_PERF_END(RastTotalElapsed);
        return;
    }
    float2 guardBandFactor = (float)MaxRenderSize / float2(_viewSize);
    if (!EnableGuardband) guardBandFactor = float2(1.0f);

    _queue->AllocBins(_viewSize, _runner->GetThreadCount());

    std::atomic<uint32_t> finishedMeshCount = 0;
    std::atomic<uint64_t> activeWorkers = (1ull << _queue->NumWorkers) - 1;

#define DBG(msg, ...) // printf("[W%d] " msg "\n", workerId __VA_OPT__(,) __VA_ARGS__)
    //printf("Draw(%d)\n", meshCount);

    Dispatch([&](uint32_t workerId) {
        if (workerId >= _queue->NumWorkers) return;

        ZoneScopedN("MeshWorker");
        DBG("Begin");

        ShadedMeshlet mesh = { .PrimCount = 0 };
        uint32_t nextMeshIdx = workerId, meshIdx = nextMeshIdx;
        uint32_t primOffset = 0;

        Clipper clipper;
        clipper.GuardBandFactor = guardBandFactor;
        uint32_t clipReadOffset = 0;

        uint32_t batchId = workerId * BatchesInFlight;
        uint32_t batchMeshCount = 0;
        uint32_t nextPacketIdx = 0, nextClipPacketIdx = 0;

        TracyCZoneCtx meshZone = {};

        while (true) {
            bool batchFull = false;
            bool isIdle = true;
            bool clipPending = clipReadOffset < clipper.OutIdx;

            // Fetch another meshlet
            if (nextMeshIdx < meshCount && primOffset >= mesh.PrimCount && !clipPending) {
                TracyZoneSpanBegin(meshZone, "FetchMeshlet", 0x55BB55);
                isIdle = false;

                meshIdx = nextMeshIdx;
                nextMeshIdx += _queue->NumWorkers;
                batchMeshCount++;

                shader.Dispatch.ShadeMeshlet(shader.Context, meshIdx, mesh);
                if (mesh.PrimCount == 0) continue;

                SWR_PERF_INC(RastTrisProcessed, mesh.PrimCount);
                primOffset = 0;
            }

            // Process pending primitives
            for (; primOffset < mesh.PrimCount && !batchFull && !clipPending; primOffset += simd::vec_width) {
                isIdle = false;

                v_float4 v0 = GatherPos(mesh, &mesh.Indices[0][primOffset]);
                v_float4 v1 = GatherPos(mesh, &mesh.Indices[1][primOffset]);
                v_float4 v2 = GatherPos(mesh, &mesh.Indices[2][primOffset]);
                ClipCodes cc = Clipper::ComputeClipCodes(v0, v1, v2, guardBandFactor);

                if (primOffset + simd::vec_width > mesh.PrimCount) {
                    v_mask trailMask = (1u << (mesh.PrimCount % simd::vec_width)) - 1;
                    cc.AcceptMask &= trailMask;
                    cc.NonTrivialMask &= trailMask;
                }

                if (cc.NonTrivialMask != 0 && EnableClipping) [[unlikely]] {
                    clipper.Setup(v0, v1, v2);
                    clipper.ClipTriangles(cc, _viewHalfSize, mesh.CullMode);

                    for (uint32_t i = 0; i < clipper.OutIdx; i += simd::vec_width) {
                        TrianglePacket& tris = clipper.OutTris[i / simd::vec_width];
                        v_mask drawMask = clipper.OutMasks[i / simd::vec_width];

                        tris.MeshletId = meshIdx;
                        tris.BasePrimId = primOffset;
                        tris.FragmentShaderId = mesh.FragmentShaderId;

                        memcpy(tris.VertexId[0], &mesh.Indices[0][primOffset], 16);
                        memcpy(tris.VertexId[1], &mesh.Indices[1][primOffset], 16);
                        memcpy(tris.VertexId[2], &mesh.Indices[2][primOffset], 16);

                        SWR_PERF_INC(RastTrisDrawn, simd::popcnt(drawMask));
                    }
                    clipPending = true;
                    clipReadOffset = 0;
                }

                if (cc.AcceptMask != 0) {
                    assert(nextPacketIdx < MaxPacketsPerWorker);
                    uint32_t packetId = batchId * MaxPacketsPerBatch + nextPacketIdx;
                    TrianglePacket& tris = _queue->PacketPool[packetId];
                    v_mask acceptMask = tris.Setup(v0, v1, v2, _viewHalfSize, mesh.CullMode);

                    acceptMask &= cc.AcceptMask;
                    if (acceptMask == 0) continue;

                    SWR_PERF_INC(RastTrisDrawn, simd::popcnt(acceptMask));

                    tris.MeshletId = meshIdx;
                    tris.BasePrimId = primOffset;
                    tris.FragmentShaderId = mesh.FragmentShaderId;
                    tris.ClipData = nullptr;

                    memcpy(tris.VertexId[0], &mesh.Indices[0][primOffset], 16);
                    memcpy(tris.VertexId[1], &mesh.Indices[1][primOffset], 16);
                    memcpy(tris.VertexId[2], &mesh.Indices[2][primOffset], 16);

                    v_uint2 boundBox = tris.GetBoundingBox(_viewHalfSize, false);
                    batchFull |= !DistributeToBins(batchId, packetId, acceptMask, boundBox.x, boundBox.y);
                    batchFull |= ++nextPacketIdx >= MaxPacketsPerBatch;
                }
            }

            // Process pending clipper output
            for (; !batchFull && clipReadOffset < clipper.OutIdx; clipReadOffset += simd::vec_width) {
                isIdle = false;

                uint32_t i = clipReadOffset / simd::vec_width;
                uint32_t triPacketId = batchId * MaxPacketsPerBatch + nextPacketIdx;
                uint32_t clipPacketId = batchId * MaxClipPacketsPerBatch + nextClipPacketIdx;

                _queue->PacketPool[triPacketId] = clipper.OutTris[i];
                _queue->PacketPool[triPacketId].ClipData = &_queue->ClipDataPool[clipPacketId];
                _queue->ClipDataPool[clipPacketId] = clipper.OutData[i];

                v_uint2 boundBox = clipper.OutTris[i].GetBoundingBox(_viewHalfSize, false);
                batchFull |= !DistributeToBins(batchId, triPacketId, clipper.OutMasks[i], boundBox.x, boundBox.y);
                batchFull |= ++nextPacketIdx >= MaxPacketsPerBatch;
                batchFull |= ++nextClipPacketIdx >= MaxClipPacketsPerBatch;
            }

            // Rasterize
            if (batchFull || isIdle) {
                TracyZoneSpanEnd(meshZone);
                DBG("Commit m=%d p=%d b=%d", batchMeshCount, nextPacketIdx, batchId % BatchesInFlight);

                // Commit bins
                if (nextPacketIdx > 0) {
                    for (uint32_t i = 0; i < _queue->BinsPerFrame; i += 64) {
                        assert(_queue->ReadyBitmap[batchId][i / 64u] == 0);
                        auto counts = simd::load<uint8_t, 64>(&_queue->BinCounts[batchId * _queue->BinCountBatchStride + i]);
                        uint64_t mask = simd::movemask(counts != 0);
                        _queue->ReadyBitmap[batchId][i / 64u] = mask;

                        SWR_PERF_INC(RastBinsDispatched, simd::popcnt(mask));
                    }
                    _mm_mfence();
                }

                while (true) {
                    RasterizeBins(shader, workerId);

                    // Switch to next batch
                    batchId = (batchId / BatchesInFlight * BatchesInFlight) + ((batchId + 1) % BatchesInFlight);
                    nextPacketIdx = 0;
                    nextClipPacketIdx = 0;

                    bool isNextBatchBusy = false;
                    for (uint32_t i = 0; i < _queue->BinsPerFrame; i += 64) {
                        isNextBatchBusy |= _queue->ReadyBitmap[batchId][i / 64u] != 0;
                    }
                    if (!isNextBatchBusy) {
                        memset(&_queue->BinCounts[batchId * _queue->BinCountBatchStride], 0, _queue->BinsPerFrame);
                        break;
                    }
                    _mm_pause();
                }

                // Exit if no more meshes
                if (isIdle) {
                    uint32_t prevCount = finishedMeshCount.fetch_add(batchMeshCount, std::memory_order::relaxed);
                    if (prevCount >= meshCount - batchMeshCount) {
                        if (activeWorkers.fetch_and(~(1ull << workerId), std::memory_order::acq_rel) == 0) break;
                    }
                    batchMeshCount = 0;
                }
            }
        }
        DBG("Exit");
        perf::FlushThreadCounters();
    });
    SWR_PERF_END(RastTotalElapsed);

    memset(_queue->BinCounts.get(), 0, _queue->BinCountBatchStride * _queue->NumWorkers * BatchesInFlight);
}

void Rasterizer::RasterizeBins(ShaderBinding shader, uint32_t workerId) {
    TracyCZoneCtx rasterZone = {};

    uint32_t numBatches = _queue->NumWorkers * BatchesInFlight;
    uint64_t binPattern = _queue->BinWorkPattern << workerId;
    
    for (uint32_t binChunkOffset = 0; binChunkOffset < _queue->BinsPerFrame; binChunkOffset += 64) {
        // Create local copy of commit bitmap to minimize MESI transitions
        static_assert(MaxRasterBinWorkers * BatchesInFlight <= 64);
        uint64_t binMasks[MaxRasterBinWorkers * BatchesInFlight];
        uint64_t batchMask = 0;

        #pragma nounroll
        for (uint32_t batchId = 0; batchId < numBatches; batchId++) {
            binMasks[batchId] = _queue->ReadyBitmap[batchId][binChunkOffset / 64u] & binPattern;
            batchMask |= binMasks[batchId] ? (1ull << batchId) : 0;
        }
        if (batchMask == 0) continue;

        TracyZoneSpanBegin(rasterZone, "RasterBin", 0xEE5555);

        for (uint32_t batchId : simd::BitIter(batchMask)) {
            for (uint32_t binMaskOffset : simd::BitIter(binMasks[batchId])) {
                uint32_t binId = binChunkOffset + binMaskOffset;

                // volatile prevents clang from hoisting broadcastd pointlessly - https://github.com/llvm/llvm-project/issues/120015
                uint32_t binY = (binId * _queue->BinsPerRowRcp) >> 32;
                uint32_t binX = binId - binY * _queue->BinGridSize.x;
                volatile uint32_t binPosMin = (binX | binY << 16) * BinSize;
                volatile uint32_t binPosMax = binPosMin + (BinSize * 0x0001'0001);

                uint8_t count = _queue->BinCounts[batchId * _queue->BinCountBatchStride + binId];
                uint32_t* ids = _queue->BinPool[batchId * _queue->BinsPerFrame + binId].Ids;

                for (uint32_t i = 0; i < count; i++) {
                    TrianglePacket& tris = _queue->PacketPool[ids[i] >> 16];
                    v_mask acceptMask = v_mask(ids[i] & 0xFFFF);

                    v_uint2 boundBox = tris.GetBoundingBox(_viewHalfSize, true);
                    v_uint bbMin = _mm512_max_epi16(boundBox.x, _mm512_set1_epi32((int)binPosMin));
                    v_uint bbMax = _mm512_min_epi16(boundBox.y, _mm512_set1_epi32((int)binPosMax));
                    v_uint bbExtent = bbMax - bbMin;

                    TriangleEdgeVars edges;
                    edges.Setup(tris, _viewHalfSize);

                    memcpy(edges.VertexId, tris.VertexId, sizeof(edges.VertexId));
                    edges.BasePrimId = tris.BasePrimId;
                    edges.TileStride = _tileStride;
                    edges.MeshletId = tris.MeshletId;
                    edges.ClipData = tris.ClipData;

                    {
                        v_uint x0 = bbMin & 0xFFFF, y0 = bbMin >> 16;
                        v_uint x1 = bbMax & 0xFFFF, y1 = bbMax >> 16;
                        assert((simd::movemask(x0 > x1 || y0 > y1) & acceptMask) == 0);
                    }

                    auto drawFn = shader.Dispatch.DrawTriangle[tris.FragmentShaderId];
                    if (tris.ClipData != nullptr) {
                        drawFn = shader.Dispatch.DrawClippedTriangle[tris.FragmentShaderId];
                    }

                    for (uint32_t j : simd::BitIter(acceptMask)) {
                        // clang sometimes emits full vector reload when indexing (only for local vars?)
                        uint64_t boundRect = ((uint32_t*)&bbMin)[j] | (uint64_t)((uint32_t*)&bbExtent)[j] << 32;
                        drawFn(shader.Context, edges, j, boundRect);
                    }
                }
            }
            __atomic_fetch_and(&_queue->ReadyBitmap[batchId][binChunkOffset / 64u], ~binMasks[batchId], __ATOMIC_RELAXED);
        }
    }

    TracyZoneSpanEnd(rasterZone);
}

static uint32_t ReduceMin_U16x2(v_uint value, v_mask mask) {
    value = _mm512_mask_mov_epi32(_mm512_set1_epi32(-1), mask, value);
    __m256i v256 = _mm256_min_epu16(_mm512_extracti32x8_epi32(value, 0), _mm512_extracti32x8_epi32(value, 1));
    __m128i v128 = _mm_min_epu16(_mm256_extracti128_si256(v256, 0), _mm256_extracti128_si256(v256, 1));
    __m128i v64 = _mm_min_epu16(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(3, 2, 3, 2)));
    __m128i v32 = _mm_min_epu16(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 1, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(v32);
}
static uint32_t ReduceMax_U16x2(v_uint value, v_mask mask) {
    value = _mm512_mask_mov_epi32(_mm512_set1_epi32(0), mask, value);
    __m256i v256 = _mm256_max_epu16(_mm512_extracti32x8_epi32(value, 0), _mm512_extracti32x8_epi32(value, 1));
    __m128i v128 = _mm_max_epu16(_mm256_extracti128_si256(v256, 0), _mm256_extracti128_si256(v256, 1));
    __m128i v64 = _mm_max_epu16(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(3, 2, 3, 2)));
    __m128i v32 = _mm_max_epu16(v64, _mm_shuffle_epi32(v64, _MM_SHUFFLE(0, 1, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(v32);
}

// Returns false when a bin gets full.
bool Rasterizer::DistributeToBins(uint32_t batchId, uint32_t packetIdx, v_mask acceptMask, v_uint triMinPos, v_uint triMaxPos) {
    v_uint binMinPos = _mm512_srli_epi16(triMinPos, BinShift);
    v_uint binMaxPos = _mm512_srli_epi16(triMaxPos - 0x0001'0001, BinShift);
    v_uint firstLaneBinPos = _mm512_permutexvar_epi32(_mm512_set1_epi32(_mm_tzcnt_32(acceptMask)), binMinPos);
    v_mask sameBinMask = simd::movemask(binMinPos == binMaxPos && binMinPos == firstLaneBinPos);

    // Fast path for packets that span one bin
    if ((acceptMask & ~sameBinMask) == 0) [[likely]] {
        uint32_t binPos = firstLaneBinPos[0];
        return _queue->InsertBin(batchId, uint2(binPos & 0xFFFF, binPos >> 16), packetIdx, acceptMask);
    }

    v_uint binExtent = binMaxPos - binMinPos;
    uint32_t startBB = ReduceMin_U16x2(binMinPos, acceptMask);
    uint32_t endBB = ReduceMax_U16x2(binMaxPos, acceptMask);
    uint32_t startX = startBB & 0xFFFF, startY = startBB >> 16;
    uint32_t endX = endBB & 0xFFFF, endY = endBB >> 16;

    // Set `binMinPos > binMaxPos` for inactive tris, so loop cmp is never true.
    v_int rejectMask = !simd::unmovemask(acceptMask);
    simd::cmov(binMinPos, 0, rejectMask);
    simd::cmov(binMaxPos, 0xFFFF'FFFF, rejectMask);

    bool underCap = true;

    // Fast path for medium triangles (spanning less than 3x3 bins)
    if (simd::all(simd::min(binExtent, binExtent << 16) < 0x0003'0000) || !EnableBinIntersect) [[likely]] {
        for (uint32_t y = startY; y <= endY; y++) {
            v_uint packedPos = v_uint(startX | (y << 16));

            for (uint32_t x = startX; x <= endX; x++, packedPos += 1) {
                v_uint clampedPos = _mm512_min_epi16(_mm512_max_epi16(packedPos, binMinPos), binMaxPos);
                v_mask mask = simd::movemask(clampedPos == packedPos);

                if (mask != 0) {
                    underCap &= _queue->InsertBin(batchId, uint2(x, y), packetIdx, mask);
                }
            }
        }
        return underCap;
    }
    TriangleEdgeVars edges;
    [[clang::always_inline]] edges.Setup(_queue->PacketPool[packetIdx], _viewHalfSize);

    v_int3 edgeStepX = edges.StepX;
    v_int3 edgeStepY = edges.StepY;

    // Edge offsets at innermost tile corner
    v_int3 edgeOrigin = {
        (simd::max(edgeStepX.x, 0) + simd::max(edgeStepY.x, 0)) * (BinSize - 1),
        (simd::max(edgeStepX.y, 0) + simd::max(edgeStepY.y, 0)) * (BinSize - 1),
        (simd::max(edgeStepX.z, 0) + simd::max(edgeStepY.z, 0)) * (BinSize - 1),
    };
    edgeStepX *= BinSize, edgeStepY *= BinSize;

    edgeOrigin += edges.OriginDist;
    edgeOrigin += edgeStepX * int(startX);
    edgeOrigin += edgeStepY * int(startY);

    for (uint32_t y = startY; y <= endY; y++) {
        v_uint packedPos = v_uint(startX | (y << 16));
        v_int3 edge = edgeOrigin;

        for (uint32_t x = startX; x <= endX; x++, packedPos += 1) {
            v_int insideTri = (edge.x | edge.y | edge.z) >= 0;
            v_uint clampedPos = _mm512_min_epi16(_mm512_max_epi16(packedPos, binMinPos), binMaxPos);
            v_mask mask = simd::movemask(insideTri && clampedPos == packedPos);

            if (mask != 0) {
                underCap &= _queue->InsertBin(batchId, uint2(x, y), packetIdx, mask);
            }
            edge += edgeStepX;
        }
        edgeOrigin += edgeStepY;
    }
    return underCap;
}

[[clang::always_inline]]
bool BinQueue::InsertBin(uint32_t batchId, uint2 binPos, uint32_t packetIdx, v_mask mask) {
    uint32_t binId = binPos.x + binPos.y * BinGridSize.x;
    uint32_t count = BinCounts[batchId * BinCountBatchStride + binId]++;
    uint32_t* ids = BinPool[batchId * BinsPerFrame + binId].Ids;

    assert(count < MaxPacketsPerBin);
    static_assert(simd::vec_width <= 16);
    static_assert(MaxPacketsPerBatch * MaxRasterBinWorkers < UINT16_MAX);

    ids[count] = (packetIdx << 16) | mask;
    return count < MaxPacketsPerBin;
}

void BinQueue::AllocBins(uint2 fbSize, uint32_t workerCount) {
    workerCount = std::min(workerCount, MaxRasterBinWorkers);
    uint2 newGridSize = (fbSize + BinSize - 1u) >> BinShift;

    if (BinGridSize.x == newGridSize.x && BinGridSize.y == newGridSize.y && workerCount == NumWorkers) return;

    BinGridSize = newGridSize;
    BinsPerFrame = uint16_t(newGridSize.x * newGridSize.y);

    NumWorkers = uint16_t(workerCount);
    BinCountBatchStride = uint16_t((BinsPerFrame + 63) & ~63u);  // align per-worker `BinCounts` to cacheline

    BinsPerRowRcp = ((1ull << 32) / BinGridSize.x) + 1;  // magic div coeff, good for up to 16 bits

    BinCounts = std::make_unique<uint8_t[]>(BinCountBatchStride * NumWorkers * BatchesInFlight);
    BinPool = std::make_unique<TriangleBin[]>(BinsPerFrame * NumWorkers * BatchesInFlight);
    PacketPool = std::make_unique<TrianglePacket[]>(MaxPacketsPerBatch * NumWorkers * BatchesInFlight);
    ClipDataPool = std::make_unique<TriangleClipData[]>(MaxClipPacketsPerBatch * NumWorkers * BatchesInFlight);
    
    BinWorkPattern = 0;
    for (uint32_t i = 0; i < 64; i += NumWorkers) {
        BinWorkPattern |= 1ull << i;
    }

    // TODO: investigate alternative raster patterns https://schmalstieg.github.io/pdf/Schmalstieg_338.pdf
    // for (uint32_t y = 0; y < gridSize.y; y++) {
    //     // X-shift + offset
    //     float k = (numWorkers + 1) / ceilf(sqrtf(numWorkers));
    //     uint32_t offset = uint32_t(ceilf(y * k));

    //     // Van der Corput (broken for N != POT, must skip if offset >= N)
    //     uint32_t m = simd::lzcnt(y | 1);
    //     uint32_t k = __builtin_elementwise_bitreverse(y) >> m;
    //     uint32_t offset = (uint32_t)ceilf(k / float(1 << (32 - m)) * N);
    //     // uint32_t offset = (k * N) >> (32 - m);
    // }
}

// Toggle between futex and mm_pause() spin loops for syncing.
// Spin loops have lower latency but can degrade perf past core count,
// see https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html
#define SYNC_SPIN_ONLY 1

void Rasterizer::Dispatch(void (*cb)(void* pUser, uint32_t workerId), void* pUser) { _runner->Dispatch(cb, pUser); }

void ThreadedRunner::Dispatch(void (*cb)(void* pData, uint32_t workerId), void* pData) {
    if (_threads.empty()) { cb(pData, 0); return; }

    assert(_numBusy.load(std::memory_order::acquire) == 0);

    _jobData = pData;
    _jobCallback = cb;

    // Notify workers of a new job
    _numBusy.store(_threads.size(), std::memory_order::release);
    _timestamp.fetch_add(1, std::memory_order::release);
    _timestamp.notify_all();

    cb(pData, 0);

    ZoneScopedNC("JoinDispatch", 0xEECC55);

    // Wait idle
#if SYNC_SPIN_ONLY
    while (_numBusy.load(std::memory_order::acquire) > 0) _mm_pause();
#else
    for (uint32_t n; (n = _numBusy.load(std::memory_order::acquire)) > 0;) {
        _numBusy.wait(n, std::memory_order::acquire);
    }
#endif

    _jobData = nullptr;
    _jobCallback = nullptr;
}

void ThreadedRunner::WorkerFn(uint32_t workerId) {
    tracy::SetThreadName("RasterWorker");
    uint64_t prevTimestamp = 0;

    while (true) {
        uint64_t currTimestamp = _timestamp.load(std::memory_order::acquire);

        if (currTimestamp == prevTimestamp) {
#if SYNC_SPIN_ONLY
            _mm_pause();
#else
            _timestamp.wait(currTimestamp, std::memory_order::acquire);
#endif
            continue;
        }
        if (currTimestamp == UINT64_MAX) break;  // shutdown signal

        prevTimestamp = currTimestamp;

        _jobCallback(_jobData, workerId);
        if (_numBusy.fetch_sub(1, std::memory_order::acq_rel) == 1) {
            _numBusy.notify_all();
        }
    }
}
void Barrier::Wait() {
    ZoneScoped;

    uint64_t currToken = _token.fetch_sub(1, std::memory_order::acq_rel);
    uint32_t prevGen = uint32_t(currToken >> 32);
    uint32_t prevCount = uint32_t(currToken);

    if (prevCount == 1) {
        _token.fetch_add((1ull << 32) + _threadCount, std::memory_order::release);
#if !SYNC_SPIN_ONLY
        _token.notify_all();
#endif
    } else {
        while ((currToken >> 32) == prevGen) {
#if SYNC_SPIN_ONLY
            _mm_pause();
#else
            _token.wait(currToken, std::memory_order::acquire);
#endif
            currToken = _token.load(std::memory_order::acquire);
        }
        assert((currToken >> 32) == (prevGen + 1));
    }
}

std::vector<perf::Counter> perf::CounterRegistry = {};
uint32_t perf::FrameNo = 0;

static std::vector<void(*)()> CounterLocalFlusherRegistry;

int64_t perf::RegisterCounter(const char* name, int64_t* globalAccum, void (*flushLocalAccum)()) {
    CounterRegistry.push_back({
        .Name = name,
        .GlobalAccum = globalAccum,
        .FlushLocalAccum = flushLocalAccum,
    });
    CounterLocalFlusherRegistry.push_back(flushLocalAccum);
    return 0;
}

perf::Counter* perf::GetCounter(const char* name) {
    for (Counter& counter : CounterRegistry) {
        if (strcmp(counter.Name, name) == 0) return &counter;
    }
    return nullptr;
}
double perf::GetAverage(const char* name) {
    Counter* pc = GetCounter(name);
    if (!pc) return 0;

    double sum = 0;
    for (uint32_t i = 0; i < SampleHistSize; i++) {
        sum += pc->Samples[i];
    }
    return sum / SampleHistSize;
}

void perf::EndFrame() {
    FrameNo++;

    for (Counter& counter : CounterRegistry) {
        counter.Samples[FrameNo % SampleHistSize] = *counter.GlobalAccum;
        *counter.GlobalAccum = 0;
    }
}

int64_t perf::GetTimestamp() {
#ifdef CLOCK_MONOTONIC
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec * 1'000'000'000LL + ts.tv_nsec);
#else
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

[[clang::noinline]]
void perf::FlushThreadCounters() {
    for (auto& flusher : CounterLocalFlusherRegistry) {
        flusher();
    }
}

};  // namespace swr
