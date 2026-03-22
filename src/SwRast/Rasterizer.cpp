#include <chrono>
#include <execution>
#include <ranges>

#include "SwRast.h"
#include "ProfilerStats.h"
#include "tracy/Tracy.hpp"

namespace swr {

static void ParallelDispatch(uint32_t numInvocs, auto fn) {
    auto range = std::ranges::iota_view(0u, numInvocs);
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), fn);
}

static v_float GatherPreload64(const float data[64], v_int idx) {
    v_float v0 = _mm512_load_ps(&data[0]);
    v_float v1 = _mm512_load_ps(&data[16]);
    v_float v2 = _mm512_load_ps(&data[32]);
    v_float v3 = _mm512_load_ps(&data[48]);
    v_float p01 = _mm512_permutex2var_ps(v0, idx, v1);
    v_float p23 = _mm512_permutex2var_ps(v2, idx, v3);
    return idx < 32 ? p01 : p23;
}
static v_float4 GatherPos(const ShadedMeshlet& mesh, v_int idx) {
    static_assert(ShadedMeshlet::MaxVertices == 64);
    return {
        GatherPreload64(mesh.Position[0], idx),
        GatherPreload64(mesh.Position[1], idx),
        GatherPreload64(mesh.Position[2], idx),
        GatherPreload64(mesh.Position[3], idx),
    };
}

void Rasterizer::DrawMeshletsImpl(uint32_t count, const ShaderDispatcher& shader) {
    ZoneScoped;
    STAT_TIME_BEGIN(Rasterize);

    for (uint32_t meshIdx = 0; meshIdx < count; meshIdx++) {
        ShadedMeshlet mesh;
        shader.MeshFn(meshIdx, mesh);
        if (mesh.PrimCount == 0) continue;

        STAT_INCREMENT(VerticesShaded, mesh.PrimCount);

        for (uint32_t primIdx = 0; primIdx < mesh.PrimCount; primIdx += simd::vec_width) {
            v_int i0 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[0][primIdx]));
            v_int i1 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[1][primIdx]));
            v_int i2 = _mm512_cvtepu8_epi32(_mm_load_epi32(&mesh.Indices[2][primIdx]));
            v_float4 v0 = GatherPos(mesh, i0);
            v_float4 v1 = GatherPos(mesh, i1);
            v_float4 v2 = GatherPos(mesh, i2);
            Clipper::ClipCodes cc = _clipper.ComputeClipCodes(v0, v1, v2);

            if (mesh.PrimCount - primIdx < simd::vec_width) {
                v_mask trailMask = (1u << (mesh.PrimCount - primIdx)) - 1;
                cc.AcceptMask &= trailMask;
            }

            if (cc.AcceptMask != 0) {
                v0 = simd::PerspectiveDiv(v0);
                v1 = simd::PerspectiveDiv(v1);
                v2 = simd::PerspectiveDiv(v2);

                TrianglePacket tris;
                v_mask mask = tris.Setup(v0, v1, v2, glm::ivec2(_fb.Width, _fb.Height) / 2, CullMode);

                mask &= cc.AcceptMask;

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
    r = (r + 7) >> 4;                                         // round to pixel center
    r = r & ~(int32_t)Framebuffer::TileMask;                  // align min bb coords to tile boundary
    r = simd::min(simd::max(r + vpSize, 0), vpSize * 2 - 4);  // translate to 0,0 origin and clamp to vp size
    return r;
}
static v_int ComputeMaxBB(v_int a, v_int b, v_int c, int32_t vpSize) {
    v_int r = simd::max(simd::max(a, b), c);
    r = (r + 7) >> 4;                                         // round to pixel center
    r = simd::min(simd::max(r + vpSize, 0), vpSize * 2 - 4);  // translate to 0,0 origin and clamp to vp size
    return r;
}

static v_int ComputeEdge(v_int a, v_int x, v_int b, v_int y) {
    v_int w = a * x + b * y;
    w += (a > 0 || (a == 0 && b > 0)) ? 0 : -1; // Top-left rule bias
    return w >> 4;
}

// https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice
v_mask TrianglePacket::Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize, FaceCullMode cullMode) {
    glm::vec2 fixScale = vpHalfSize * 16;
    v_int x0 = simd::round2i(v0.x * fixScale.x);
    v_int x1 = simd::round2i(v1.x * fixScale.x);
    v_int x2 = simd::round2i(v2.x * fixScale.x);
    v_int y0 = simd::round2i(v0.y * fixScale.y);
    v_int y1 = simd::round2i(v1.y * fixScale.y);
    v_int y2 = simd::round2i(v2.y * fixScale.y);

    A01 = y1 - y0, B01 = x0 - x1;
    A12 = y2 - y1, B12 = x1 - x2;
    A20 = y0 - y2, B20 = x2 - x0;
    v_int det = B20 * A01 - B01 * A20;

    if (cullMode != FaceCullMode::FrontCCW) {
        v_int flip = (cullMode == FaceCullMode::FrontCW) ? -1 : (det < 0);
        A01 = flip ? -A01 : A01, B01 = flip ? -B01 : B01;
        A12 = flip ? -A12 : A12, B12 = flip ? -B12 : B12;
        A20 = flip ? -A20 : A20, B20 = flip ? -B20 : B20;
        det = flip ? -det : det;
    }
    // Cull backfacing triangles or with zero area
    v_mask mask = simd::movemask(det > 0);
    if (mask == 0) return 0;

    v_int minX = ComputeMinBB(x0, x1, x2, vpHalfSize.x);
    v_int minY = ComputeMinBB(y0, y1, y2, vpHalfSize.y);
    v_int maxX = ComputeMaxBB(x0, x1, x2, vpHalfSize.x);
    v_int maxY = ComputeMaxBB(y0, y1, y2, vpHalfSize.y);
    MinXY = (v_uint)(minX | minY << 16);
    MaxXY = (v_uint)(maxX | maxY << 16);

    v_int minCX = ((minX - vpHalfSize.x) << 4) + 8, minCY = ((minY - vpHalfSize.y) << 4) + 8;
    Edge0 = ComputeEdge(A12, minCX - x1, B12, minCY - y1);
    Edge1 = ComputeEdge(A20, minCX - x2, B20, minCY - y2);
    Edge2 = ComputeEdge(A01, minCX - x0, B01, minCY - y0);

    v_float rcpArea = 16.0f / simd::conv2f(det);
    Z0 = v0.z;
    Z10 = (v1.z - v0.z) * rcpArea;
    Z20 = (v2.z - v0.z) * rcpArea;

    W0 = v0.w;
    W0S = v0.w * rcpArea;
    W1S = v1.w * rcpArea;
    W2S = v2.w * rcpArea;
    
    return mask;
}

Clipper::ClipCodes Clipper::ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2) {
    using v_byte = uint8_t [[clang::ext_vector_type(16)]];
    using v_bool = bool [[clang::ext_vector_type(16)]];

    float bx = GuardBandPlaneDist.x, by = GuardBandPlaneDist.y;

    // Only a very small fraction of triangles intersect with the frustum planes (< 2%),
    // it's worth doing a cheaper test first since most packets are trivial to clip.
    v_mask partialOutMask = 0;
    v_mask combinedOutMask = v_mask(~0u);

    #pragma unroll
    for (uint32_t i = 0; i < 3; i++) {
        v_float4 pos = i == 0 ? v0 : (i == 1 ? v1 : v2);
        v_mask outmask = simd::movemask(
            (simd::abs(pos.x) > pos.w * bx) |
            (simd::abs(pos.y) > pos.w * by) |
            (simd::abs(pos.z) > pos.w));

        partialOutMask |= outmask;
        combinedOutMask &= outmask;
    }
    if ((partialOutMask & ~combinedOutMask) == 0) [[likely]] {
        return { .AcceptMask = (v_mask)~combinedOutMask };
    }

    v_byte partialOutCodes = 0;     // non-zero if at least one vertex is out
    v_byte combinedOutCodes = 255;  // non-zero if all vertices are out

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

        partialOutCodes |= outcode;
        combinedOutCodes &= outcode;
    }
    v_mask acceptMask = _mm_cmpeq_epi8_mask(partialOutCodes, _mm_set1_epi8(0));
    v_mask rejectMask = _mm_cmpneq_epi8_mask(combinedOutCodes, _mm_set1_epi8(0));

    ClipCodes codes = {
        .AcceptMask = (v_mask)(acceptMask & ~rejectMask),
        .NonTrivialMask = (v_mask)(~acceptMask & ~rejectMask),
    };
    _mm_storeu_si128((__m128i*)codes.OutCodes, partialOutCodes);
    return codes;
}

// dot(pos.xyz, plane.norm) + pos.w * dist
static float GetIntersectDist(const glm::vec4& pos, Clipper::Plane plane, float dist) {
    float a = pos[(uint32_t)plane / 2];
    if ((uint32_t)plane % 2) a = -a;

    return a + pos.w * dist;
}

// https://www.cubic.org/docs/3dclip.htm
void Clipper::ClipAgainstPlane(Plane plane) {
    if (Count < 3) return;  // triangle has been fully clipped away

    float planeDist = plane < Plane::Near ? GuardBandPlaneDist[(uint32_t)plane / 2] : 1.0f;
    uint8_t tempIndices[24];
    uint32_t j = 0;

    for (uint32_t i = 0; i < Count; i++) {
        glm::vec4& va = Vertices[Indices[i]];
        glm::vec4& vb = Vertices[Indices[(i + 1) % Count]];

        float da = GetIntersectDist(va, plane, planeDist);
        float db = GetIntersectDist(vb, plane, planeDist);

        if (da >= 0) tempIndices[j++] = Indices[i];

        // Calculate intersection if distance signs differ
        if ((da >= 0) != (db >= 0)) {
            float t = da / (da - db);

            tempIndices[j++] = FreeVtx;
            Vertices[FreeVtx++] = glm::mix(va, vb, t);
        }
    }
    assert(j < 24);

    memcpy(Indices, tempIndices, 24);  // small constant size copy can be inlined
    Count = j;
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
