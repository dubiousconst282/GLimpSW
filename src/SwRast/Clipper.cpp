#include "SwRast.h"

namespace swr {

// dot(vtx.XYZ, plane.Norm) + vtx.W * dist
static float GetIntersectDist(const Clipper::Vertex& vtx, Clipper::Plane plane, float dist) {
    float a = vtx.Attribs[(uint32_t)plane / 2];
    if ((uint32_t)plane % 2) a = -a;

    return a + vtx.Attribs[3] * dist;
}

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

            for (uint32_t ai = 0; ai < numAttribs; ai += VFloat::Length) {
                VFloat a = VFloat::load(&va.Attribs[ai]);
                VFloat b = VFloat::load(&vb.Attribs[ai]);
                VFloat r = simd::fma(b - a, t, a);  // a + (b - a) * t
                r.store(&dest.Attribs[ai]);
            }
        }
    }
    std::memcpy(Indices, tempIndices, 24); //small constant size copy likely to perform better than var
    Count = j;
    assert(Count < 24);
}

Clipper::ClipCodes Clipper::ComputeClipCodes(const TrianglePacket& tri) {
    auto partialOut = _mm_set1_epi8(0);    // non-zero if at least one vertex is out
    auto combinedOut = _mm_set1_epi8(~0);  // non-zero if all vertices are out

    for (uint32_t i = 0; i < 3; i++) {
        const VFloat4& pos = tri.Vertices[i].Position;
        auto outcode = _mm_set1_epi8(0);

        const auto MaskN = [&](VFloat x, Clipper::Plane p) {
            auto m = _mm512_cmp_ps_mask(x, -pos.w, _CMP_LT_OQ);
            auto c = _mm_maskz_mov_epi8(m, _mm_set1_epi8(1 << (int)p));
            outcode = _mm_or_si128(outcode, c);
        };
        const auto MaskP = [&](VFloat x, Clipper::Plane p) {
            auto m = _mm512_cmp_ps_mask(x, pos.w, _CMP_GT_OQ);
            auto c = _mm_maskz_mov_epi8(m, _mm_set1_epi8(1 << (int)p));
            outcode = _mm_or_si128(outcode, c);
        };
        MaskN(pos.x, Clipper::Plane::Left);
        MaskP(pos.x, Clipper::Plane::Right);
        MaskN(pos.y, Clipper::Plane::Bottom);
        MaskP(pos.y, Clipper::Plane::Top);
        MaskN(pos.z, Clipper::Plane::Near);
        MaskP(pos.z, Clipper::Plane::Far);

        partialOut = _mm_or_si128(partialOut, outcode);
        combinedOut = _mm_and_si128(combinedOut, outcode);
    }

    uint16_t acceptMask = _mm_cmpeq_epi8_mask(partialOut, _mm_set1_epi8(0));
    uint16_t rejectMask = _mm_cmpneq_epi8_mask(combinedOut, _mm_set1_epi8(0));

    ClipCodes codes = {
        .AcceptMask = (uint16_t)(acceptMask & ~rejectMask),
        .NonTrivialMask = (uint16_t)(~acceptMask & ~rejectMask),
    };
    _mm_storeu_si128((__m128i*)codes.OutCodes, partialOut);
    return codes;
}

void Clipper::LoadTriangle(TrianglePacket& srcTri, uint32_t srcTriIdx, uint32_t numAttribs) {
    for (uint32_t vi = 0; vi < 3; vi++) {
        VFloat* src = &srcTri.Vertices[vi].Position.x;
        float* dest = Vertices[vi].Attribs;

        for (uint32_t ai = 0; ai < numAttribs; ai++) {
            dest[ai] = src[ai][srcTriIdx];
        }
        Indices[vi] = vi;
    }
    FreeVtx = Count = 3;
}
void Clipper::StoreTriangle(TrianglePacket& destTri, uint32_t destTriIdx, uint32_t srcTriFanIdx, uint32_t numAttribs) {
    uint32_t srcIdx[3]{ 0, srcTriFanIdx + 1, srcTriFanIdx + 2 };

    for (uint32_t vi = 0; vi < 3; vi++) {
        VFloat* dest = &destTri.Vertices[vi].Position.x;
        float* src = Vertices[Indices[srcIdx[vi]]].Attribs;

        for (uint32_t ai = 0; ai < numAttribs; ai++) {
            dest[ai][destTriIdx] = src[ai];
        }
    }
}

}; // namespace swr