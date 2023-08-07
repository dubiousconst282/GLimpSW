
#include <array>
#include <chrono>
#include <execution>
#include <ranges>

#include "Misc.h"
#include "SwRast.h"

namespace swr {

Rasterizer::Rasterizer(std::shared_ptr<Framebuffer> fb) {
    const float maxViewSize = 2048.0f;
    _clipper.GuardBandPlaneDistXY[0] = maxViewSize / fb->Width;
    _clipper.GuardBandPlaneDistXY[1] = maxViewSize / fb->Height;

    _batch = std::make_unique<TriangleBatch>(fb->Width, fb->Height);

    _fb = std::move(fb);
}

void Rasterizer::SetupTriangles(TriangleBatch& batch, uint32_t numCustomAttribs) {
    STAT_TIME_BEGIN(Clipping);

    TrianglePacket& tri = batch.PeekLast();
    Clipper::ClipCodes cc = _clipper.ComputeClipCodes(tri);
    uint32_t addedTriangles = 0;
    uint32_t numAttribs = numCustomAttribs + 4;

    // Clip non-trivial triangles
    for (uint32_t i : BitIter(cc.NonTrivialMask)) {
        _clipper.LoadTriangle(tri, i, numAttribs);

        for (uint32_t j : BitIter(cc.OutCodes[i])) {
            _clipper.ClipAgainstPlane((Clipper::Plane)j, numAttribs);
        }

        if (_clipper.Count < 2) continue;

        // Triangulate result polygon
        for (uint32_t j = 0; j < _clipper.Count - 2; j++) {
            uint32_t usedMask = cc.AcceptMask | (cc.NonTrivialMask & (~1u << i));
            uint32_t freeIdx = (uint32_t)std::countr_one(usedMask);

            if (freeIdx < VFloat::Length) {
                _clipper.StoreTriangle(tri, freeIdx, j, numAttribs);
                cc.AcceptMask |= 1u << freeIdx;
            } else {
                if (addedTriangles % VFloat::Length == 0) {
                    TrianglePacket& newTri = batch.Alloc();
                    assert(&newTri == (&tri + i / VFloat::Length + 1));
                }
                _clipper.StoreTriangle(batch.PeekLast(), addedTriangles % VFloat::Length, j, numAttribs);
                addedTriangles++;
            }
        }
        STAT_INCREMENT(TrianglesClipped, 1);
    }
    STAT_TIME_END(Clipping);

    if (cc.AcceptMask == 0 && addedTriangles == 0) {
        batch.Count--;  // free unused triangle
        return;
    }

    STAT_TIME_BEGIN(Binning);

    if (cc.AcceptMask != 0) {
        BinTriangles(batch, tri, cc.AcceptMask, numCustomAttribs);
    }

    for (uint32_t i = 0; i < addedTriangles; i += VFloat::Length) {
        uint16_t mask = (1u << std::min(VFloat::Length, addedTriangles - i)) - 1;
        BinTriangles(batch, *(&tri + i / VFloat::Length + 1), mask, numCustomAttribs);
    }

    STAT_TIME_END(Binning);
}

void Rasterizer::BinTriangles(TriangleBatch& batch, TrianglePacket& tris, uint16_t mask, uint32_t numAttribs) {
    int32_t width = (int32_t)_fb->Width, height = (int32_t)_fb->Height;
    tris.Setup(width, height, numAttribs);

    mask &= _mm512_cmp_ps_mask(tris.RcpArea, _mm512_set1_ps(0.0f), _CMP_GT_OQ);  // backface culling (skip triangles with negative area)
    mask &= _mm512_cmp_ps_mask(tris.RcpArea, _mm512_set1_ps(1.0f), _CMP_LT_OQ);  // skip triangles with zero area

    for (uint32_t i : BitIter(mask)) {
        const uint32_t binShift = TriangleBatch::BinSizeLog2;

        uint32_t minX = (uint32_t)((tris.MinX[i] + width / 2) >> binShift);
        uint32_t minY = (uint32_t)((tris.MinY[i] + height / 2) >> binShift);
        uint32_t maxX = (uint32_t)((tris.MaxX[i] + width / 2) >> binShift);
        uint32_t maxY = (uint32_t)((tris.MaxY[i] + height / 2) >> binShift);

        for (uint32_t y = minY; y <= maxY; y++) {
            for (uint32_t x = minX; x <= maxX; x++) {
                batch.AddBin(x, y, tris, i);
                STAT_INCREMENT(BinsFilled, 1);
            }
        }
    }
    STAT_INCREMENT(TrianglesDrawn, (uint32_t)std::popcount(mask));
}

void Rasterizer::Draw(VertexReader& vertexData, const ShaderInterface& shader) {
    uint32_t pos = 0;
    uint32_t count = vertexData.Count / 3;

    while (pos < count) {
        TriangleBatch& batch = *_batch;

        STAT_TIME_BEGIN(Setup);

        for (; pos < count && !batch.IsFull(); pos += VFloat::Length) {
            TrianglePacket& tri = batch.Alloc();

            // Read vertices and assemble triangles
            shader.ReadVtxFn(pos * 3, tri.Vertices);

            // Clip, setup, and bin
            SetupTriangles(batch, shader.NumCustomAttribs);
        }

        STAT_TIME_END(Setup);

        if (batch.Count == 0) continue;
        
        STAT_TIME_BEGIN(Rasterize);

        auto binRange = std::ranges::iota_view(0u, batch.NumBins);
        std::atomic_int64_t elapsed = 0;

        std::for_each(std::execution::par_unseq, binRange.begin(), binRange.end(), [&](const uint32_t& bid) {
            auto startTime = std::chrono::high_resolution_clock::now();

            std::vector<uint16_t>& bin = batch.Bins[bid];
            if (bin.size() == 0) return;

            BinnedTriangle bt = {
                .X = (bid % batch.BinsPerRow) * TriangleBatch::BinSize,
                .Y = (bid / batch.BinsPerRow) * TriangleBatch::BinSize,
            };
            for (uint16_t triangleId : bin) {
                bt.Triangle = &batch.Triangles[triangleId / VFloat::Length];
                bt.TriangleIndex = triangleId % VFloat::Length;
                shader.DrawFn(bt);
            }
            bin.clear();

            elapsed += (std::chrono::high_resolution_clock::now() - startTime).count();
        });
        g_Stats.RasterizeCpuTime += elapsed.load();
        batch.Count = 0;

        STAT_TIME_END(Rasterize);
    }
}

ProfilerStats g_Stats = {};

void ProfilerStats::MeasureTime(int64_t key[2], bool begin) {
    int64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    if (begin) {
        assert(key[1] == 0 && "STAT_TIME_X() must be called in pairs");
        key[1] = now;
    } else {
        key[0] += now - key[1];
        assert(key[1] != 0 && !(key[1] = 0) && "STAT_TIME_X() must be called in pairs");
    }
}


static VInt EdgeWeight(VInt a, VInt x, VInt b, VInt y) {
    VInt w = a * x + b * y;
    // Add top-left rule bias
    //  w += (a > 0 || (a == 0 && b > 0)) ? 0 : -1
    VInt zero = _mm512_set1_epi32(0);
    uint16_t tlA = _mm512_kand(_mm512_cmpeq_epi32_mask(a, zero), _mm512_cmpgt_epi32_mask(b, zero));
    uint16_t tlB = _mm512_kor(_mm512_cmpgt_epi32_mask(a, zero), tlA);
    w = _mm512_add_epi32(w, _mm512_movm_epi32(_mm512_knot(tlB)));

    return w >> 4;
}

//return min(r, bb...)
static VInt ComputeMinBB(VInt a, VInt b, VInt c, int32_t vpSize) {
    VInt r = simd::min(simd::max(simd::min(simd::min(a, b), c), (vpSize << 4) / -2), (vpSize << 4) / +2);
    r = (r + 15) >> 4;                // round up to int
    r = r & ~(int32_t)Framebuffer::TileMask;  // align min bb coords to tile boundary
    return r;
}
static VInt ComputeMaxBB(VInt a, VInt b, VInt c, int32_t vpSize) {
    VInt r = simd::min(simd::max(simd::max(simd::max(a, b), c), (vpSize << 4) / -2), (vpSize << 4) / +2);
    r = (r + 15) >> 4;  // round up to int
    return r;
}

void TrianglePacket::Setup(int32_t vpWidth, int32_t vpHeight, uint32_t numAttribs) {
    // Perspective division
    for (uint32_t i = 0; i < 3; i++) {
        VFloat4& pos = Vertices[i].Position;
        VFloat rw = 1.0f / pos.w;
        pos = { pos.x * rw, pos.y * rw, pos.z * rw, rw };
    }

    const auto LoadFixedPos = [&](uint32_t axis, float scale) -> std::array<VInt, 3> {
        return {
            simd::round2i(*(&Vertices[0].Position.x + axis) * scale),
            simd::round2i(*(&Vertices[1].Position.x + axis) * scale),
            simd::round2i(*(&Vertices[2].Position.x + axis) * scale),
        };
    };

    auto [x0, x1, x2] = LoadFixedPos(0, (vpWidth / 2) * 16.0f);
    MinX = ComputeMinBB(x0, x1, x2, vpWidth);
    MaxX = ComputeMaxBB(x0, x1, x2, vpWidth - 4);

    auto [y0, y1, y2] = LoadFixedPos(1, (vpHeight / 2) * 16.0f);
    MinY = ComputeMinBB(y0, y1, y2, vpHeight);
    MaxY = ComputeMaxBB(y0, y1, y2, vpHeight - 4);

    A01 = y0 - y1, B01 = x1 - x0;
    A12 = y1 - y2, B12 = x2 - x1;
    A20 = y2 - y0, B20 = x0 - x2;

    auto minX = MinX << 4, minY = MinY << 4;
    Weight0 = EdgeWeight(A12, minX - x1, B12, minY - y1);
    Weight1 = EdgeWeight(A20, minX - x2, B20, minY - y2);
    Weight2 = EdgeWeight(A01, minX - x0, B01, minY - y0);

    RcpArea = 16.0f / simd::conv2f(B01 * A20 - B20 * A01);

    // Prepare attributes for interpolation
    for (uint32_t i = 0; i <= numAttribs; i++) {
        int32_t j = i < numAttribs ? (int32_t)i : VaryingBuffer::AttribZ;

        VFloat v0 = Vertices[0].Attribs[j];
        Vertices[1].Attribs[j] -= v0;
        Vertices[2].Attribs[j] -= v0;
    }
}

};  // namespace swr
