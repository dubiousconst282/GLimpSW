#include <array>
#include <chrono>
#include <execution>
#include <ranges>

#include "SwRast.h"

namespace swr {

static void ParallelDispatch(uint32_t numInvocs, auto fn) {
    auto range = std::ranges::iota_view(0u, numInvocs);
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), fn);
}

Rasterizer::Rasterizer(std::shared_ptr<Framebuffer> fb) {
    const float maxViewSize = 2048.0f;
    _clipper.GuardBandPlaneDistXY[0] = maxViewSize / fb->Width;
    _clipper.GuardBandPlaneDistXY[1] = maxViewSize / fb->Height;

    _batch = std::make_unique<TriangleBatch>(fb->Width, fb->Height);

    _fb = std::move(fb);
}

void Rasterizer::Draw(VertexReader& vertexData, const ShaderInterface& shader) {
    uint32_t pos = 0;
    uint32_t count = vertexData.Count / 3;

    while (pos < count) {
        TriangleBatch& batch = *_batch;

        STAT_TIME_BEGIN(Setup);

        for (; pos < count && !batch.IsFull(); pos += simd::vec_width) {
            TrianglePacket& tri = batch.Alloc();

            // Read vertices and assemble triangles
            shader.ReadVtxFn(pos * 3, tri.Vertices);

            // Clip, setup, and bin
            SetupTriangles(batch, shader.NumCustomAttribs);
        }

        STAT_TIME_END(Setup);

        if (batch.Count == 0) continue;

        STAT_TIME_BEGIN(Rasterize);

        ParallelDispatch(batch.NumBins, [&](uint32_t bid) {
            std::vector<uint16_t>& bin = batch.Bins[bid];
            if (bin.size() == 0) return;

            BinnedTriangle bt = {
                .X = (bid % batch.BinsPerRow) * TriangleBatch::BinSize,
                .Y = (bid / batch.BinsPerRow) * TriangleBatch::BinSize,
            };
            for (uint16_t triangleId : bin) {
                bt.Triangle = &batch.Triangles[triangleId / simd::vec_width];
                bt.TriangleIndex = triangleId % simd::vec_width;
                shader.DrawFn(bt);
            }
            bin.clear();
        });
        batch.Count = 0;

        STAT_TIME_END(Rasterize);
    }
}

void Rasterizer::SetupTriangles(TriangleBatch& batch, uint32_t numCustomAttribs) {
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

            if (freeIdx < simd::vec_width) {
                _clipper.StoreTriangle(tri, freeIdx, j, numAttribs);
                cc.AcceptMask |= 1u << freeIdx;
            } else {
                if (addedTriangles % simd::vec_width == 0) {
                    TrianglePacket& newTri = batch.Alloc();
                    assert(&newTri == (&tri + i / simd::vec_width + 1));
                }
                _clipper.StoreTriangle(batch.PeekLast(), addedTriangles % simd::vec_width, j, numAttribs);
                addedTriangles++;
            }
        }
        STAT_INCREMENT(TrianglesClipped, 1);
    }

    if (cc.AcceptMask == 0 && addedTriangles == 0) {
        batch.Count--;  // free unused triangle
        return;
    }

    if (cc.AcceptMask != 0) {
        BinTriangles(batch, tri, cc.AcceptMask, numCustomAttribs);
    }

    for (uint32_t i = 0; i < addedTriangles; i += simd::vec_width) {
        uint16_t mask = (1u << std::min(simd::vec_width, addedTriangles - i)) - 1;
        BinTriangles(batch, *(&tri + i / simd::vec_width + 1), mask, numCustomAttribs);
    }
}

void Rasterizer::BinTriangles(TriangleBatch& batch, TrianglePacket& tris, v_mask mask, uint32_t numAttribs) {
    int32_t width = (int32_t)_fb->Width, height = (int32_t)_fb->Height;
    tris.Setup(width, height, numAttribs);

    // Cull backfacing triangles or with zero area
    mask &= simd::movemask(tris.RcpArea > 0.0f && tris.RcpArea < 1.0f);

    for (uint32_t i : BitIter(mask)) {
        const uint32_t binShift = TriangleBatch::BinSizeLog2;

        uint32_t minX = (uint32_t)tris.MinX[i] >> binShift;
        uint32_t minY = (uint32_t)tris.MinY[i] >> binShift;
        uint32_t maxX = (uint32_t)tris.MaxX[i] >> binShift;
        uint32_t maxY = (uint32_t)tris.MaxY[i] >> binShift;

        for (uint32_t y = minY; y <= maxY; y++) {
            for (uint32_t x = minX; x <= maxX; x++) {
                batch.AddBin(x, y, tris, i);
                STAT_INCREMENT(BinsFilled, 1);
            }
        }
    }
    STAT_INCREMENT(TrianglesDrawn, (uint32_t)std::popcount(mask));
}

static std::array<v_int, 3> LoadFixedPos(const TrianglePacket& tri, uint32_t axis, float scale) {
    return {
        simd::round2i(*(&tri.Vertices[0].Position.x + axis) * scale),
        simd::round2i(*(&tri.Vertices[1].Position.x + axis) * scale),
        simd::round2i(*(&tri.Vertices[2].Position.x + axis) * scale),
    };
};

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
void TrianglePacket::Setup(int32_t vpWidth, int32_t vpHeight, uint32_t numAttribs) {
    // Perspective division
    for (uint32_t i = 0; i < 3; i++) {
        v_float4& pos = Vertices[i].Position;
        pos = simd::PerspectiveDiv(pos);
    }

    vpWidth /= 2, vpHeight /= 2;

    auto [x0, x1, x2] = LoadFixedPos(*this, 0, vpWidth * 16.0f);
    MinX = ComputeMinBB(x0, x1, x2, vpWidth);
    MaxX = ComputeMaxBB(x0, x1, x2, vpWidth);

    auto [y0, y1, y2] = LoadFixedPos(*this, 1, vpHeight * 16.0f);
    MinY = ComputeMinBB(y0, y1, y2, vpHeight);
    MaxY = ComputeMaxBB(y0, y1, y2, vpHeight);

    A01 = y0 - y1, B01 = x1 - x0;
    A12 = y1 - y2, B12 = x2 - x1;
    A20 = y2 - y0, B20 = x0 - x2;

    auto minX = (MinX - vpWidth) << 4, minY = (MinY - vpHeight) << 4;
    Weight0 = ComputeEdge(A12, minX - x1, B12, minY - y1);
    Weight1 = ComputeEdge(A20, minX - x2, B20, minY - y2);
    Weight2 = ComputeEdge(A01, minX - x0, B01, minY - y0);

    RcpArea = 16.0f / simd::conv2f(B01 * A20 - B20 * A01);

    // Prepare attributes for interpolation
    for (uint32_t i = 0; i <= numAttribs; i++) {
        int32_t j = i < numAttribs ? (int32_t)i : VaryingBuffer::AttribZ;

        v_float v0 = Vertices[0].Attribs[j];
        Vertices[1].Attribs[j] -= v0;
        Vertices[2].Attribs[j] -= v0;
    }
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
