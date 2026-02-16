#include <chrono>
#include <execution>
#include <ranges>

#include "SwRast.h"
#include "ProfilerStats.h"

namespace swr {

static void ParallelDispatch(uint32_t numInvocs, auto fn) {
    auto range = std::ranges::iota_view(0u, numInvocs);
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), fn);
}
static v_float4 GatherPos(const ShadedMeshlet& mesh, v_int idx) {
    return {
        simd::gather(mesh.Attribs[0], idx),
        simd::gather(mesh.Attribs[1], idx),
        simd::gather(mesh.Attribs[2], idx),
        simd::gather(mesh.Attribs[3], idx),
    };
}

void Rasterizer::DrawMeshletsImpl(uint32_t count, const ShaderDispatcher& shader) {
    STAT_TIME_BEGIN(Rasterize);

    for (uint32_t meshIdx = 0; meshIdx < count; meshIdx++) {
        ShadedMeshlet mesh;
        shader.MeshFn(meshIdx, mesh);
        if (mesh.PrimCount == 0) continue;

        for (uint32_t primIdx = 0; primIdx < mesh.PrimCount; primIdx += simd::vec_width) {
            v_int i0 = _mm512_cvtepu8_epi32(_mm_loadu_epi32(&mesh.Indices[0][primIdx]));
            v_int i1 = _mm512_cvtepu8_epi32(_mm_loadu_epi32(&mesh.Indices[1][primIdx]));
            v_int i2 = _mm512_cvtepu8_epi32(_mm_loadu_epi32(&mesh.Indices[2][primIdx]));
            v_float4 v0 = GatherPos(mesh, i0);
            v_float4 v1 = GatherPos(mesh, i1);
            v_float4 v2 = GatherPos(mesh, i2);
            Clipper::ClipCodes cc = _clipper.ComputeClipCodes(v0, v1, v2);

            if (primIdx + simd::vec_width >= mesh.PrimCount) {
                v_mask trailMask = (1u << (mesh.PrimCount % simd::vec_width)) - 1;
                cc.AcceptMask &= trailMask;
            }

            if (cc.AcceptMask != 0) {
                v0 = simd::PerspectiveDiv(v0);
                v1 = simd::PerspectiveDiv(v1);
                v2 = simd::PerspectiveDiv(v2);

                TrianglePacket tris;
                tris.Setup(v0, v1, v2, glm::ivec2(_fb.Width, _fb.Height) / 2);

                uint16_t mask = cc.AcceptMask;

                // Cull backfacing triangles or with zero area
                mask &= simd::movemask(tris.RcpArea > 0.0f && tris.RcpArea < 1.0f);

                for (uint32_t i : BitIter(mask)) {
                    shader.DrawFn(tris, mesh, i, primIdx + i, meshIdx);
                }
                STAT_INCREMENT(TrianglesDrawn, (uint32_t)std::popcount(mask));
            }
            if (cc.NonTrivialMask != 0) {
                STAT_INCREMENT(TrianglesClipped, (uint32_t)std::popcount(cc.NonTrivialMask));
            }
        }
    }
    STAT_TIME_END(Rasterize);
}

static v_int ComputeMinBB(v_int a, v_int b, v_int c, int32_t vpSize) {
    v_int r = simd::min(simd::min(a, b), c);
    r = (r + 15) >> 4;                                        // round up to int
    r = r & ~(int32_t)Framebuffer::TileMask;                  // align min bb coords to tile boundary
    r = simd::min(simd::max(r + vpSize, 0), vpSize * 2 - 4);  // translate to 0,0 origin and clamp to vp size
    return r;
}
static v_int ComputeMaxBB(v_int a, v_int b, v_int c, int32_t vpSize) {
    v_int r = simd::max(simd::max(a, b), c);
    r = (r + 15) >> 4;                                        // round up to int
    r = simd::min(simd::max(r + vpSize, 0), vpSize * 2 - 4);  // translate to 0,0 origin and clamp to vp size
    return r;
}

static v_int ComputeEdge(v_int a, v_int x, v_int b, v_int y) {
    v_int w = a * x + b * y;
    w += (a > 0 || (a == 0 && b > 0)) ? 0 : -1; // Top-left rule bias
    return w >> 4;
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
// This is missing handling on a few subtleties listed in the article:
//  - Overflow: work-able region is only 2048x2048, but could be extended to 8192x8192
//  - Top-left bias: vertex attributes will be interpolated with some slight shift
void TrianglePacket::Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize) {
    glm::vec2 fixScale = vpHalfSize * 16;

    Z0 = { v0.z, v0.w };
    Z1 = { v1.z, v1.w };
    Z2 = { v2.z, v2.w };

    v_int x0 = simd::round2i(v0.x * fixScale.x);
    v_int x1 = simd::round2i(v1.x * fixScale.x);
    v_int x2 = simd::round2i(v2.x * fixScale.x);
    MinX = ComputeMinBB(x0, x1, x2, vpHalfSize.x);
    MaxX = ComputeMaxBB(x0, x1, x2, vpHalfSize.x);

    v_int y0 = simd::round2i(v0.y * fixScale.y);
    v_int y1 = simd::round2i(v1.y * fixScale.y);
    v_int y2 = simd::round2i(v2.y * fixScale.y);
    MinY = ComputeMinBB(y0, y1, y2, vpHalfSize.y);
    MaxY = ComputeMaxBB(y0, y1, y2, vpHalfSize.y);

    A01 = y0 - y1, B01 = x1 - x0;
    A12 = y1 - y2, B12 = x2 - x1;
    A20 = y2 - y0, B20 = x0 - x2;

    v_int minX = (MinX - vpHalfSize.x) << 4, minY = (MinY - vpHalfSize.y) << 4;
    W0 = ComputeEdge(A12, minX - x1, B12, minY - y1);
    W1 = ComputeEdge(A20, minX - x2, B20, minY - y2);
    W2 = ComputeEdge(A01, minX - x0, B01, minY - y0);

    RcpArea = 16.0f / simd::conv2f(B01 * A20 - B20 * A01);
}

Clipper::ClipCodes Clipper::ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2) {
    using v_byte = uint8_t [[clang::ext_vector_type(16)]];
    using v_bool = bool [[clang::ext_vector_type(16)]];

    v_byte partialOut = 0;     // non-zero if at least one vertex is out
    v_byte combinedOut = 255;  // non-zero if all vertices are out
    float bx = GuardBandPlaneDistXY[0], by = GuardBandPlaneDistXY[1];

    #pragma unroll
    for (uint32_t i = 0; i < 3; i++) {
        v_float4 pos = i == 0 ? v0 : (i == 1 ? v1 : v2);
        v_byte outcode = 0;

        outcode |= __builtin_convertvector(pos.x < -pos.w * bx, v_byte) ? v_byte(1 << (int)Plane::Left) : 0;
        outcode |= __builtin_convertvector(pos.x > +pos.w * bx, v_byte) ? v_byte(1 << (int)Plane::Right) : 0;
        outcode |= __builtin_convertvector(pos.y < -pos.w * by, v_byte) ? v_byte(1 << (int)Plane::Bottom) : 0;
        outcode |= __builtin_convertvector(pos.y > +pos.w * by, v_byte) ? v_byte(1 << (int)Plane::Top) : 0;
        outcode |= __builtin_convertvector(pos.z < -pos.w, v_byte) ? v_byte(1 << (int)Plane::Near) : 0;
        outcode |= __builtin_convertvector(pos.z > +pos.w, v_byte) ? v_byte(1 << (int)Plane::Far) : 0;

        partialOut |= outcode;
        combinedOut &= outcode;
    }
    v_mask acceptMask = _mm_cmpeq_epi8_mask(partialOut, _mm_set1_epi8(0));
    v_mask rejectMask = _mm_cmpneq_epi8_mask(combinedOut, _mm_set1_epi8(0));

    ClipCodes codes = {
        .AcceptMask = (v_mask)(acceptMask & ~rejectMask),
        .NonTrivialMask = (v_mask)(~acceptMask & ~rejectMask),
    };
    _mm_storeu_si128((__m128i*)codes.OutCodes, partialOut);
    return codes;
}

// dot(vtx.XYZ, plane.Norm) + vtx.W * dist
static float GetIntersectDist(const Clipper::Vertex& vtx, Clipper::Plane plane, float dist) {
    float a = vtx.Attribs[(uint32_t)plane / 2];
    if ((uint32_t)plane % 2) a = -a;

    return a + vtx.Attribs[3] * dist;
}

// https://www.cubic.org/docs/3dclip.htm
void Clipper::ClipAgainstPlane(Plane plane, uint32_t numAttribs) {
    if (Count < 3) return;  // triangle has been fully clipped away

    float planeDist = plane < Plane::Near ? GuardBandPlaneDistXY[(uint32_t)plane / 2] : 1.0f;
    uint8_t tempIndices[24];
    uint32_t j = 0;

    for (uint32_t i = 0; i < Count; i++) {
        Vertex& va = Vertices[Indices[i]];
        Vertex& vb = Vertices[Indices[(i + 1) % Count]];

        float da = GetIntersectDist(va, plane, planeDist);
        float db = GetIntersectDist(vb, plane, planeDist);

        if (da >= 0) {
            tempIndices[j++] = Indices[i];
        }

        // Calculate intersection if distance signs differ
        if ((da >= 0) != (db >= 0)) {
            tempIndices[j++] = FreeVtx;
            Vertex& dest = Vertices[FreeVtx++];

            float t = da / (da - db);

            assert(numAttribs <= sizeof(va.Attribs) / 4);

            for (uint32_t ai = 0; ai < numAttribs; ai += simd::vec_width) {
                v_float a = simd::load(&va.Attribs[ai]);
                v_float b = simd::load(&vb.Attribs[ai]);
                v_float r = simd::lerp(a, b, t);
                simd::store(r, &dest.Attribs[ai]);
            }
        }
    }
    memcpy(Indices, tempIndices, 24);  // small constant size copy can be inlined
    Count = j;
    assert(Count < 24);
}

void Framebuffer::IterateTiles(std::function<void(uint32_t, uint32_t)> visitor, uint32_t downscaleFactor) {
    downscaleFactor *= 4;

    ParallelDispatch(Height / downscaleFactor, [&](uint32_t y) {
        for (uint32_t x = 0; x < Width; x += downscaleFactor) {
            visitor(x, y * downscaleFactor);
        }
    });
}

ProfilerStats g_Stats = {};

uint64_t ProfilerStats::CurrentTime() {
    auto time = std::chrono::high_resolution_clock::now();
    return (uint64_t)time.time_since_epoch().count();
}

};  // namespace swr
