// #undef NDEBUG

#include "Rasterizer.h"

#include <tracy/Tracy.hpp>

#include <vector>
#include <thread>

#define SYNC_SPIN_ONLY 1

namespace swr {

static constexpr uint32_t MaxWorkerBatch = 256, MaxBinSize = 63;
static constexpr uint32_t BinShift = 7, BinSize = 1 << BinShift;
static constexpr uint32_t MaxRenderBins = (Rasterizer::MaxRenderSize + BinSize - 1) / BinSize;
static constexpr uint32_t FrameBitmapWords = (MaxRenderBins * MaxRenderBins) / 64;

struct alignas(64) TriangleBin {
    uint32_t Count;
    uint32_t Ids[MaxBinSize];
};
struct BinQueue {
    uint16_t BinsPerRow = 0, BinsPerFrame = 0, NumWorkers = 0;
    uint64_t BinsPerRowRcp = 0;

    // Binner working set
    std::unique_ptr<TriangleBin[]> Bins;
    std::unique_ptr<TrianglePacket[]> Batch;
    std::unique_ptr<uint64_t[]> WorkerBitmap;  // Per worker bitmaps

    // Commited to rasterizer
    std::unique_ptr<TriangleBin[]> CommitedBins;
    std::unique_ptr<TrianglePacket[]> CommitedBatch;
    uint64_t FrameBitmap[FrameBitmapWords];    // Merged from all workers

    void AllocBins(uint32_t fbWidth, uint32_t fbHeight, uint32_t workerCount);
    bool InsertBin(uint32_t workerId, uint32_t binId, uint32_t packetIdx, v_mask mask);

    void Commit();
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
    glm::vec2 GuardBandFactor;
    uint32_t OutIdx, OutMaxIdx;
    
    TrianglePacket* OutTris;
    TriangleClipData* OutData;
    v_mask* OutMasks;

    alignas(64) float Vertices[6][simd::vec_width * 3 + 16];
    alignas(64) float OutVertices[3][4][simd::vec_width];

    // Compute Cohen-Sutherland clip codes
    static ClipCodes ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2, glm::vec2 guardBandFactor);

    // Clip non-trivial triangles into output buffer.
    void ClipTriangles(v_float4 v0, v_float4 v1, v_float4 v2, ClipCodes cc, glm::ivec2 vpHalfSize, FaceCullMode cullMode);

    void FlushPacket(glm::ivec2 vpHalfSize, FaceCullMode cullMode, bool isLast);

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

    template<typename TCallback>
    void Dispatch(TCallback&& cb);

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
    // The bin rasterizer cannot handle more than 64 threads.
    // This architecture probably won't scale well past too many threads anyway, so better to just cap it.
    count = std::min(count, 64u);

    _runner = std::make_unique<ThreadedRunner>(count);
}

static v_float4 GatherPos(const ShadedMeshlet& mesh, v_uint idx) {
    static_assert(ShadedMeshlet::MaxVertices == 64);
    return {
        simd::gather_preload64(mesh.Position[0], idx),
        simd::gather_preload64(mesh.Position[1], idx),
        simd::gather_preload64(mesh.Position[2], idx),
        simd::gather_preload64(mesh.Position[3], idx),
    };
}

void Rasterizer::DrawMeshletsST(Framebuffer& fb, uint32_t count, ShaderBinding shader) {
    glm::ivec2 vpHalfSize = glm::ivec2(fb.Width, fb.Height) / 2;
    glm::vec2 guardBandFactor = { MaxRenderSize / (float)fb.Width, MaxRenderSize / (float)fb.Height };
    if (!EnableGuardband) guardBandFactor = glm::vec2(1.0f);

    _runner->Dispatch([&](uint32_t workerId) {
        for (uint32_t meshIdx = workerId; meshIdx < count; meshIdx += _runner->GetThreadCount()) {
            ShadedMeshlet mesh;
            shader.Dispatch.MeshFn(shader.Context, meshIdx, mesh);
            if (mesh.PrimCount == 0) continue;

            SWR_PERF_INC(TrianglesProcessed, mesh.PrimCount);

            for (uint32_t primOffset = 0; primOffset < mesh.PrimCount; primOffset += simd::vec_width) {
                v_uint i0 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[0][primOffset]));
                v_uint i1 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[1][primOffset]));
                v_uint i2 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[2][primOffset]));
                v_float4 v0 = GatherPos(mesh, i0);
                v_float4 v1 = GatherPos(mesh, i1);
                v_float4 v2 = GatherPos(mesh, i2);
                ClipCodes cc = Clipper::ComputeClipCodes(v0, v1, v2, guardBandFactor);

                if (primOffset + simd::vec_width > mesh.PrimCount) {
                    v_mask trailMask = (1u << (mesh.PrimCount % simd::vec_width)) - 1;
                    cc.AcceptMask &= trailMask;
                    cc.NonTrivialMask &= trailMask;
                }

                if (cc.AcceptMask != 0) {
                    TrianglePacket tris;
                    v_mask acceptMask = tris.Setup(v0, v1, v2, vpHalfSize, mesh.CullMode);
                    acceptMask &= cc.AcceptMask;

                    if (acceptMask == 0) continue;

                    v_uint2 boundBox = tris.GetRenderBoundingBox(vpHalfSize);
                    v_uint extent = boundBox.y - boundBox.x;

                    TriangleEdgeVars vars;
                    vars.Setup(tris, vpHalfSize);

                    vars.MeshletId = meshIdx;
                    vars.BasePrimId = primOffset;
                    memcpy(vars.VertexId[0], &mesh.Indices[0][primOffset], 16);
                    memcpy(vars.VertexId[1], &mesh.Indices[1][primOffset], 16);
                    memcpy(vars.VertexId[2], &mesh.Indices[2][primOffset], 16);

                    for (uint32_t i : simd::BitIter(acceptMask)) {
                        uint64_t boundRect = ((uint32_t*)&boundBox.x)[i] | uint64_t(((uint32_t*)&extent)[i]) << 32;
                        shader.Dispatch.DrawFn(shader.Context, fb, vars, i, boundRect);
                    }
                    SWR_PERF_INC(TrianglesRasterized, simd::popcnt(acceptMask));
                }

                if (cc.NonTrivialMask != 0 && EnableClipping) [[unlikely]] {
                    SWR_PERF_INC(TrianglesClipped, simd::popcnt(cc.NonTrivialMask));

                    TrianglePacket clippedTris[8];
                    TriangleClipData clipData[8];
                    v_mask clipValidMask[8];

                    Clipper clipper;
                    clipper.GuardBandFactor = guardBandFactor;
                    clipper.OutIdx = 0;
                    clipper.OutMaxIdx = 8 * simd::vec_width;
                    clipper.OutTris = clippedTris;
                    clipper.OutData = clipData;
                    clipper.OutMasks = clipValidMask;
                    clipper.ClipTriangles(v0, v1, v2, cc, vpHalfSize, mesh.CullMode);

                    for (uint32_t i = 0; i < clipper.OutIdx; i += simd::vec_width) {
                        TrianglePacket& tris = clippedTris[i / simd::vec_width];
                        v_uint2 boundBox = tris.GetRenderBoundingBox(vpHalfSize);
                        v_uint extent = boundBox.y - boundBox.x;

                        TriangleEdgeVars vars;
                        vars.Setup(tris, vpHalfSize);

                        vars.MeshletId = meshIdx;
                        vars.BasePrimId = primOffset;
                        vars.ClipData = tris.ClipData;
                        memcpy(vars.VertexId[0], &mesh.Indices[0][primOffset], 16);
                        memcpy(vars.VertexId[1], &mesh.Indices[1][primOffset], 16);
                        memcpy(vars.VertexId[2], &mesh.Indices[2][primOffset], 16);

                        v_mask drawMask = clipValidMask[i / simd::vec_width];

                        for (uint32_t i : simd::BitIter(drawMask)) {
                            uint64_t boundRect = ((uint32_t*)&boundBox.x)[i] | uint64_t(((uint32_t*)&extent)[i]) << 32;
                            shader.Dispatch.DrawClippedFn(shader.Context, fb, vars, i, boundRect);
                        }
                        SWR_PERF_INC(TrianglesRasterized, simd::popcnt(drawMask));
                    }
                }
            }
        }
        perf::FlushThreadCounters();
    });
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice
v_mask TrianglePacket::Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize, FaceCullMode cullMode) {
    v0 = simd::perspective_div(v0);
    v1 = simd::perspective_div(v1);
    v2 = simd::perspective_div(v2);

    v_float det = (v2.x - v0.x) * (v1.y - v0.y) - (v0.x - v1.x) * (v0.y - v2.y);

    if (cullMode != FaceCullMode::FrontCCW) {
        v_int flip = (cullMode == FaceCullMode::FrontCW) ? -1 : (det < 0);
        det = flip ? -det : det;
    }
    // Cull backfacing or degenerate triangles
    v_mask mask = simd::movemask(det > 0);
    if (mask == 0) return 0;

    glm::vec2 fixScale = vpHalfSize * 16;
    v_int x0 = simd::round2i(v0.x * fixScale.x), y0 = simd::round2i(v0.y * fixScale.y);
    v_int x1 = simd::round2i(v1.x * fixScale.x), y1 = simd::round2i(v1.y * fixScale.y);
    v_int x2 = simd::round2i(v2.x * fixScale.x), y2 = simd::round2i(v2.y * fixScale.y);

    Pos0 = (x0 & 0xFFFF) | (y0 << 16);
    Pos1 = (x1 & 0xFFFF) | (y1 << 16);
    Pos2 = (x2 & 0xFFFF) | (y2 << 16);

    // Cull thin triangles covering no samples, or outside viewport (some of them will get here due to guard-band)
    v_uint2 boundBox = GetRenderBoundingBox(vpHalfSize);
    mask &= _mm512_cmpeq_epi32_mask(_mm512_movm_epi16(_mm512_cmpge_epi16_mask(boundBox.x, boundBox.y)), _mm512_set1_epi32(0));

    Z0 = v0.z, Z1 = v1.z, Z2 = v2.z;
    W0 = v0.w, W1 = v1.w, W2 = v2.w;

    return mask;
}

static v_int ComputeEdge(v_int a, v_int x, v_int b, v_int y) {
    v_int w = a * x + b * y;
    w += (a > 0 || (a == 0 && b > 0)) ? 0 : -1;  // top-left rule bias
    return w >> 4;                               // normalize fixed point
}
void TriangleEdgeVars::Setup(const TrianglePacket& tris, glm::ivec2 vpHalfSize) {
    v_int x0 = tris.Pos0 << 16 >> 16, y0 = tris.Pos0 >> 16;
    v_int x1 = tris.Pos1 << 16 >> 16, y1 = tris.Pos1 >> 16;
    v_int x2 = tris.Pos2 << 16 >> 16, y2 = tris.Pos2 >> 16;

    A01 = y1 - y0, B01 = x0 - x1;
    A12 = y2 - y1, B12 = x1 - x2;
    A20 = y0 - y2, B20 = x2 - x0;
    v_int det = B20 * A01 - B01 * A20;

    if (simd::any(det < 0)) {
        v_int flip = (det < 0);
        A01 = flip ? -A01 : A01, B01 = flip ? -B01 : B01;
        A12 = flip ? -A12 : A12, B12 = flip ? -B12 : B12;
        A20 = flip ? -A20 : A20, B20 = flip ? -B20 : B20;
        det = flip ? -det : det;
    }

    // Evaluate edge coeffs at screen origin (0.5, 0.5)
    v_int sampleX = (-vpHalfSize.x << 4) + 8, sampleY = (-vpHalfSize.y << 4) + 8;
    Edge0 = ComputeEdge(A12, sampleX - x1, B12, sampleY - y1);
    Edge1 = ComputeEdge(A20, sampleX - x2, B20, sampleY - y2);
    Edge2 = ComputeEdge(A01, sampleX - x0, B01, sampleY - y0);

    v_float rcpArea = 16.0f / simd::conv<float>(det);
    Z0 = tris.Z0;
    Z10 = (tris.Z1 - tris.Z0) * rcpArea;
    Z20 = (tris.Z2 - tris.Z0) * rcpArea;

    W0 = tris.W0;
    W0S = tris.W0 * rcpArea;
    W1S = tris.W1 * rcpArea;
    W2S = tris.W2 * rcpArea;
}

v_int2 TrianglePacket::GetBoundingBox() const {
    v_int minPos = _mm512_min_epi16(_mm512_min_epi16(Pos0, Pos1), Pos2);
    v_int maxPos = _mm512_max_epi16(_mm512_max_epi16(Pos0, Pos1), Pos2);

    minPos = _mm512_srai_epi16(minPos + 0x0007'0007, 4);  // round to pixel center
    maxPos = _mm512_srai_epi16(maxPos + 0x0007'0007, 4);

    return { minPos, maxPos };
}
v_uint2 TrianglePacket::GetRenderBoundingBox(glm::ivec2 vpHalfSize) const {
    v_int2 packed = GetBoundingBox();

    v_int vpSize = vpHalfSize.x | (vpHalfSize.y << 16);
    packed.x = _mm512_min_epi16(_mm512_max_epi16(_mm512_add_epi16(packed.x, vpSize), v_int(0)), vpSize * 2);
    packed.y = _mm512_min_epi16(_mm512_max_epi16(_mm512_add_epi16(packed.y, vpSize), v_int(0)), vpSize * 2);

    packed.x = (packed.x + 0x0000'0000) & ~0x0003'0003;  // align to tile boundary
    packed.y = (packed.y + 0x0003'0003) & ~0x0003'0003;

    return v_uint2((v_uint)packed.x, (v_uint)packed.y);
}

ClipCodes Clipper::ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2, glm::vec2 guardBandFactor) {
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
void Clipper::ClipTriangles(v_float4 v0, v_float4 v1, v_float4 v2, ClipCodes cc, glm::ivec2 vpHalfSize, FaceCullMode cullMode) {
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
            if (++OutIdx >= OutMaxIdx) {
                return;  // fuck it
            }
        }
    }
    if (OutIdx % simd::vec_width != 0) {
        FlushPacket(vpHalfSize, cullMode, true);
    }
}
void Clipper::FlushPacket(glm::ivec2 vpHalfSize, FaceCullMode cullMode, bool isLast) {
    uint32_t i = OutIdx / simd::vec_width;
    TrianglePacket& tri = OutTris[i];
    tri.ClipData = &OutData[i];

    v_float4 v0 = simd::load<v_float4>(OutVertices[0]);
    v_float4 v1 = simd::load<v_float4>(OutVertices[1]);
    v_float4 v2 = simd::load<v_float4>(OutVertices[2]);
    v_mask mask = tri.Setup(v0, v1, v2, vpHalfSize, cullMode);

    mask &= (1u << (OutIdx % simd::vec_width + !isLast)) - 1;
    OutMasks[i] = mask;
}

void Rasterizer::DrawMeshlets(Framebuffer& fb, uint32_t meshCount, ShaderBinding shader) {
    ZoneScoped;
    SWR_PERF_BEGIN(Draw);

    if (!EnableBinning) {
        DrawMeshletsST(fb, meshCount, shader);
        SWR_PERF_END(Draw);
        return;
    }
    _queue->AllocBins(fb.Width, fb.Height, _runner->GetThreadCount());

    uint64_t workerBinElectMask = 0;
    for (uint32_t i = 0; i < 64; i += _queue->NumWorkers) {
        workerBinElectMask |= 1ull << i;
    }
    glm::ivec2 vpHalfSize = glm::ivec2(fb.Width, fb.Height) / 2;
    glm::vec2 guardBandFactor = { MaxRenderSize / (float)fb.Width, MaxRenderSize / (float)fb.Height };

    auto readyBarrier = Barrier(_queue->NumWorkers);
    std::atomic<uint32_t> doneMeshCount = 0;

    // #define DBG(msg, ...) printf("[RasterJob %d] " msg "\n", workerId __VA_OPT__(,) __VA_ARGS__)
    #define DBG(...)

    _runner->Dispatch([&](uint32_t workerId) {
        ZoneScopedN("MeshWorker");

        DBG("Begin");
        ShadedMeshlet mesh = { .PrimCount = 0 };

        uint32_t meshJobs = 0;
        uint32_t nextMeshIdx = workerId, meshIdx = nextMeshIdx;
        uint32_t primOffset = 0;

        uint32_t nextPacketIdx = workerId * MaxWorkerBatch;
        uint32_t maxPacketIdx = nextPacketIdx + MaxWorkerBatch;

        while (true) {
            bool batchFull = false;
            bool isIdle = true;

            // Fetch another meshlet
            if (nextMeshIdx < meshCount && primOffset >= mesh.PrimCount) {
                isIdle = false;

                meshIdx = nextMeshIdx;
                nextMeshIdx += _queue->NumWorkers;
                meshJobs++;

                shader.Dispatch.MeshFn(shader.Context, meshIdx, mesh);
                if (mesh.PrimCount == 0) continue;

                SWR_PERF_INC(TrianglesProcessed, mesh.PrimCount);
                primOffset = 0;
            }

            // Process pending primitives
            for (; primOffset < mesh.PrimCount && !batchFull; primOffset += simd::vec_width) {
                isIdle = false;

                v_uint i0 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[0][primOffset]));
                v_uint i1 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[1][primOffset]));
                v_uint i2 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[2][primOffset]));
                v_float4 v0 = GatherPos(mesh, i0);
                v_float4 v1 = GatherPos(mesh, i1);
                v_float4 v2 = GatherPos(mesh, i2);
                ClipCodes cc = Clipper::ComputeClipCodes(v0, v1, v2, guardBandFactor);

                if (primOffset + simd::vec_width > mesh.PrimCount) {
                    v_mask trailMask = (1u << (mesh.PrimCount % simd::vec_width)) - 1;
                    cc.AcceptMask &= trailMask;
                    cc.NonTrivialMask &= trailMask;
                }

                if (cc.NonTrivialMask != 0) [[unlikely]] {
                    SWR_PERF_INC(TrianglesClipped, simd::popcnt(cc.NonTrivialMask));
                }

                if (cc.AcceptMask != 0) {
                    assert(nextPacketIdx < maxPacketIdx);
                    TrianglePacket& tris = _queue->Batch[nextPacketIdx];
                    v_mask acceptMask = tris.Setup(v0, v1, v2, vpHalfSize, mesh.CullMode);

                    acceptMask &= cc.AcceptMask;
                    if (acceptMask == 0) continue;

                    SWR_PERF_INC(TrianglesRasterized, simd::popcnt(acceptMask));

                    tris.MeshletId = meshIdx;
                    tris.BasePrimId = primOffset;
                    memcpy(tris.VertexId[0], &mesh.Indices[0][primOffset], 16);
                    memcpy(tris.VertexId[1], &mesh.Indices[1][primOffset], 16);
                    memcpy(tris.VertexId[2], &mesh.Indices[2][primOffset], 16);

                    v_uint2 boundBox = tris.GetRenderBoundingBox(vpHalfSize);
                    batchFull |= !DistributeToBins(workerId, nextPacketIdx, acceptMask, boundBox.x, boundBox.y);
                    batchFull |= nextPacketIdx + 1 >= maxPacketIdx;

                    nextPacketIdx++;
                }
            }

            // Rasterize!
            if (batchFull || isIdle) {
                {
                    ZoneScopedN("Sync"); DBG("Batch Ready m=%d pkt=%d/%d", meshJobs, nextPacketIdx, maxPacketIdx);

                    readyBarrier.Wait();
                    if (doneMeshCount.load(std::memory_order::acquire) >= meshCount) break;

                    if (workerId == 0) {
                        _queue->Commit();
                    }
                    readyBarrier.Wait();
                    DBG("Batch Accept");
                }

                {
                    ZoneScopedN("Rasterize");

                    for (uint32_t i = 0; i < FrameBitmapWords; i++) {
                        uint32_t shift = (workerId + i * 64) % 64;
                        uint64_t mask = _queue->FrameBitmap[i] & (workerBinElectMask << shift);

                        for (uint32_t j : simd::BitIter(mask)) {
                            uint32_t binId = i * 64 + j;
                            RasterizeBin(fb, shader, binId);
                        }
                        SWR_PERF_INC(BinQueueFlushes, simd::popcnt(mask));
                    }
                }
                DBG("Batch Done");

                if (nextMeshIdx >= meshCount && primOffset >= mesh.PrimCount) {
                    doneMeshCount.fetch_add(meshJobs, std::memory_order::acq_rel);
                    meshJobs = 0;
                }

                // Reset batch counters
                nextPacketIdx = workerId * MaxWorkerBatch;
            }
        }
        DBG("Exit");
        perf::FlushThreadCounters();
    });

    for (uint32_t i = 0; i < _queue->BinsPerFrame * _queue->NumWorkers; i++) {
        assert(_queue->Bins[i].Count == 0);
    }
    SWR_PERF_END(Draw);
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
bool Rasterizer::DistributeToBins(uint32_t workerId, uint32_t packetIdx, v_mask acceptMask, v_uint triMinPos, v_uint triMaxPos) {
    v_uint binMinPos = _mm512_srli_epi16(triMinPos, BinShift);
    v_uint binMaxPos = _mm512_srli_epi16(triMaxPos - 0x0001'0001, BinShift);
    v_uint firstLaneBinPos = _mm512_permutexvar_epi32(_mm512_set1_epi32(_mm_tzcnt_32(acceptMask)), binMinPos);
    v_mask sameBinMask = simd::movemask(binMinPos == binMaxPos && binMinPos == firstLaneBinPos);

    // Fast path for packets that span one bin
    if ((acceptMask & ~sameBinMask) == 0) [[likely]] {
        uint32_t binPos = firstLaneBinPos[0];
        uint32_t binId = (binPos & 0xFFFF) + (binPos >> 16) * _queue->BinsPerRow;
        return _queue->InsertBin(workerId, binId, packetIdx, acceptMask);
    }

    uint32_t startBB = ReduceMin_U16x2(binMinPos, acceptMask);
    uint32_t endBB = ReduceMax_U16x2(binMaxPos, acceptMask);
    uint32_t startX = startBB & 0xFFFF, startY = startBB >> 16;
    uint32_t endX = endBB & 0xFFFF, endY = endBB >> 16;

    bool underCap = true;

    for (uint32_t y = startY; y <= endY; y++) {
        for (uint32_t x = startX; x <= endX; x++) {
            v_mask mask = acceptMask & simd::movemask(x >= (binMinPos & 0xFFFF) & x <= (binMaxPos & 0xFFFF) &  //
                                                      y >= (binMinPos >> 16) & y <= (binMaxPos >> 16));
            if (mask == 0) continue;

            uint32_t binId = x + y * _queue->BinsPerRow;
            underCap &= _queue->InsertBin(workerId, binId, packetIdx, mask);
        }
    }
    return underCap;
}
void Rasterizer::RasterizeBin(Framebuffer& fb, ShaderBinding shader, uint32_t binId) {
    uint32_t binY = (binId * _queue->BinsPerRowRcp) >> 32;
    uint32_t binX = binId - binY * _queue->BinsPerRow;
    binX *= BinSize, binY *= BinSize;

    glm::ivec2 vpHalfSize = glm::ivec2(fb.Width, fb.Height) / 2;

    // volatile prevents clang from hoisting broadcastd pointlessly - https://github.com/llvm/llvm-project/issues/120015
    volatile uint32_t binPosMin = binX | binY << 16;
    volatile uint32_t binPosMax = binPosMin + (BinSize * 0x0001'0001);

    for (uint32_t workerId = 0; workerId < _queue->NumWorkers; workerId++) {
        TriangleBin& bin = _queue->CommitedBins[binId * _queue->NumWorkers + workerId];

        for (uint32_t i = 0; i < bin.Count; i++) {
            TrianglePacket& tris = _queue->CommitedBatch[bin.Ids[i] >> 16];
            v_mask acceptMask = bin.Ids[i];

            v_uint2 boundBox = tris.GetRenderBoundingBox(vpHalfSize);
            v_uint bbMin = _mm512_max_epi16(boundBox.x, _mm512_set1_epi32((int)binPosMin));
            v_uint bbMax = _mm512_min_epi16(boundBox.y, _mm512_set1_epi32((int)binPosMax));
            v_uint bbExtent = bbMax - bbMin;

            TriangleEdgeVars edges;
            memcpy(edges.VertexId, tris.VertexId, 64);
            edges.Setup(tris, vpHalfSize);

            {
                v_uint x0 = bbMin & 0xFFFF, y0 = bbMin >> 16;
                v_uint x1 = bbMax & 0xFFFF, y1 = bbMax >> 16;
                assert((simd::movemask(x0 > x1 || y0 > y1) & acceptMask) == 0);
            }

            for (uint32_t j : simd::BitIter(acceptMask)) {
                // clang sometimes emits full vector reload when indexing (only for local vars?)
                uint64_t boundRect = ((uint32_t*)&bbMin)[j] | (uint64_t)((uint32_t*)&bbExtent)[j] << 32;
                shader.Dispatch.DrawFn(shader.Context, fb, edges, j, boundRect);
            }
        }
        bin.Count = 0;
    }
}

void BinQueue::AllocBins(uint32_t fbWidth, uint32_t fbHeight, uint32_t workerCount) {
    uint32_t newBinsPerRow = (fbWidth + BinSize - 1) >> BinShift;
    uint32_t newBinCount = ((fbHeight + BinSize - 1) >> BinShift) * newBinsPerRow;

    if (newBinsPerRow == BinsPerRow && newBinCount == BinsPerFrame && workerCount == NumWorkers) return;

    assert(MaxWorkerBatch * NumWorkers < UINT16_MAX);

    BinsPerRow = newBinsPerRow;
    BinsPerFrame = newBinCount;
    NumWorkers = workerCount;

    BinsPerRowRcp = ((1ull << 32) / BinsPerRow) + 1;  // magic div coeff, good for up to 16 bits

    Bins = std::make_unique<TriangleBin[]>(BinsPerFrame * NumWorkers);
    CommitedBins = std::make_unique<TriangleBin[]>(BinsPerFrame * NumWorkers);

    Batch = std::make_unique<TrianglePacket[]>(MaxWorkerBatch * NumWorkers);
    CommitedBatch = std::make_unique<TrianglePacket[]>(MaxWorkerBatch * NumWorkers);

    WorkerBitmap = std::make_unique<uint64_t[]>(FrameBitmapWords * NumWorkers);

    for (uint32_t i = 0; i < BinsPerFrame * NumWorkers; i++) {
        Bins[i].Count = 0;
        CommitedBins[i].Count = 0;
    }
}
bool BinQueue::InsertBin(uint32_t workerId, uint32_t binId, uint32_t packetIdx, v_mask mask) {
    TriangleBin& bin = Bins[binId * NumWorkers + workerId];
    assert(bin.Count < MaxBinSize);

    bin.Ids[bin.Count++] = (packetIdx << 16) | mask;
    WorkerBitmap[workerId * FrameBitmapWords + (binId / 64u)] |= (1ull << (binId & 63));

    return bin.Count < MaxBinSize;
}

void BinQueue::Commit() {
    std::swap(CommitedBatch, Batch);
    std::swap(CommitedBins, Bins);

    memset(FrameBitmap, 0, sizeof(FrameBitmap));

    for (uint32_t i = 0; i < NumWorkers; i++) {
        for (uint32_t j = 0; j < FrameBitmapWords; j++) {
            FrameBitmap[j] |= WorkerBitmap[i * FrameBitmapWords + j];
        }
        for (uint32_t j = 0; j < FrameBitmapWords; j++) {
            WorkerBitmap[i * FrameBitmapWords + j] = 0;
        }
    }

    // Worth transpose for bit-iter over non-empty worker bins?
    // dest.bit[binId * NumWorkers + workerId] = src.bit[workerId * WorkerBitmapSize + binId]
    //
    // uint32_t binStride = (NumWorkers + 7) / 8;
    // for (uint32_t j = 0; j < BinsPerFrame; j += 64) {
    //     for (uint32_t i = 0; i < NumWorkers; i += 8) {
    //         uint8_t boundMask = i + 8 <= NumWorkers ? 255 : (1u << (NumWorkers % 8)) - 1;
    //         auto x = _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), boundMask,                          //
    //                                              _mm512_castsi512_si256(simd::lane_idx * FrameBitmapWords),  //
    //                                              &WorkerBitmaps[(i * FrameBitmapWords) + (j / 64u)], 8);

    //         // Transpose 64x8 to 8x64
    //         // https://haroldbot.nl/avx512bpc.html?q=b%206%2C7%2C8%2C0%2C1%2C2%2C3%2C4%2C5
    //         __m512i s1 = _mm512_set_epi8(7, 15, 23, 31, 39, 47, 55, 63,  //
    //                                      6, 14, 22, 30, 38, 46, 54, 62,  //
    //                                      5, 13, 21, 29, 37, 45, 53, 61,  //
    //                                      4, 12, 20, 28, 36, 44, 52, 60,  //
    //                                      3, 11, 19, 27, 35, 43, 51, 59,  //
    //                                      2, 10, 18, 26, 34, 42, 50, 58,  //
    //                                      1, 9, 17, 25, 33, 41, 49, 57,   //
    //                                      0, 8, 16, 24, 32, 40, 48, 56);
    //         x = _mm512_permutexvar_epi8(s1, x);
    //         x = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64((int64_t)0x80'40'20'10'08'04'02'01), x, 0);

    //         //_mm512_storeu_epi64(&dest[(j / 64) * binStride + i], x);
    //     }
    // }
}

void Rasterizer::Dispatch(uint32_t countX, uint32_t countY, void (*cb)(void* pUser, uint32_t x, uint32_t y), void* pUser) {
    const uint32_t BlockSize = 4;
    uint32_t numBlocksX = (countX + BlockSize - 1) / BlockSize;
    uint32_t numBlocksY = (countY + BlockSize - 1) / BlockSize;
    uint32_t numBlocks = numBlocksX * numBlocksY;
    std::atomic<uint32_t> nextBlockId = 0;

    _runner->Dispatch([&](uint32_t workerId) {
        while (true) {
            uint32_t i = nextBlockId.fetch_add(1, std::memory_order::relaxed);
            if (i >= numBlocks) break;

            uint32_t x0 = (i % numBlocksX) * BlockSize;
            uint32_t y0 = (i / numBlocksX) * BlockSize;
            uint32_t x1 = std::min(x0 + BlockSize, countX);
            uint32_t y1 = std::min(y0 + BlockSize, countY);

            for (uint32_t y = y0; y < y1; y++) {
                for (uint32_t x = x0; x < x1; x++) {
                    cb(pUser, x, y);
                }
            }
        }
    });
}

template<typename TCallback>
void ThreadedRunner::Dispatch(TCallback&& cb) {
    if (_threads.empty()) { cb(0); return; }

    assert(_numBusy.load(std::memory_order::acquire) == 0);

    _jobData = &cb;
    _jobCallback = [](void* pData, uint32_t workerId) { (*(TCallback*)pData)(workerId); };

    // Notify workers of a new job
    _numBusy.store(_threads.size(), std::memory_order::release);
    _timestamp.fetch_add(1, std::memory_order::release);
    _timestamp.notify_all();

    cb(0);

    // Wait idle
    while (_numBusy.load(std::memory_order::acquire) > 0) _mm_pause();

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
        _numBusy.fetch_sub(1, std::memory_order::acq_rel);
    }
}
void Barrier::Wait() {
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
            _token.wait(prevToken, std::memory_order::acquire);
#endif
            currToken = _token.load(std::memory_order::acquire);
        }
        assert((currToken >> 32) == (prevGen + 1));
    }
}

uint64_t perf::g_GlobalAccum[(int)PerfCounter::Count_];
uint64_t perf::g_RunningAvg[(int)PerfCounter::Count_];
thread_local uint64_t perf::g_LocalAccum[(int)PerfCounter::Count_];

uint64_t perf::GetTimestamp() {
    auto time = std::chrono::high_resolution_clock::now();
    return (uint64_t)time.time_since_epoch().count();
}

[[clang::noinline]]
void perf::FlushThreadCounters() {
    for (uint32_t i = 0; i < (int)PerfCounter::Count_; i++) {
        __atomic_fetch_add(&g_GlobalAccum[i], g_LocalAccum[i], __ATOMIC_RELAXED);
        g_LocalAccum[i] = 0;
    }
}

void perf::Reset() {
    for (uint32_t i = 0; i < (int)PerfCounter::Count_; i++) {
        const uint64_t a = 95;
        g_RunningAvg[i] = (g_RunningAvg[i] * a + g_GlobalAccum[i] * (100 - a)) / 100;
        g_GlobalAccum[i] = 0;
    }
}

};  // namespace swr
