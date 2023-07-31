#include <algorithm>
#include <array>
#include <execution>
#include <ranges>
#include <chrono>

#include "Misc.h"
#include "SwRast.h"

namespace swr {

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

static VInt ComputeMinBB(VInt a, VInt b, VInt c, int32_t vpSize) {
    VInt r = simd::min(simd::max(simd::min(simd::min(a, b), c), (vpSize << 4) / -2), (vpSize << 4) / +2);
    r = (r + 15) >> 4;                // round up to int
    r = r & ~(int32_t)Framebuffer::kTileMask;  // align min bb coords to tile boundary
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

void Rasterizer::SetupTriangles(TrianglePacket& tris, uint32_t mask, uint32_t numAttribs)
{
    int32_t width = (int32_t)_fb->Width, height = (int32_t)_fb->Height;
    tris.Setup(width, height, numAttribs);

    //TODO: backface culling (need to flip vertices or smt)
    mask &= _mm512_cmp_ps_mask(tris.RcpArea, _mm512_set1_ps(1.0f), _CMP_LT_OQ); //skip zero area triangles

    for (uint32_t i : BitIter(mask)) {
        uint32_t minX = (uint32_t)((tris.MinX[i] + width / 2) >> kBinSizeLog2);
        uint32_t minY = (uint32_t)((tris.MinY[i] + height / 2) >> kBinSizeLog2);
        uint32_t maxX = (uint32_t)((tris.MaxX[i] + width / 2) >> kBinSizeLog2);
        uint32_t maxY = (uint32_t)((tris.MaxY[i] + height / 2) >> kBinSizeLog2);

        uint16_t triangleId = (uint16_t)((&tris - _triangles.get()) * VFloat::Length + i);
        uint32_t binStride = (uint32_t)width >> kBinSizeLog2;

        for (uint32_t y = minY; y <= maxY; y++) {
            for (uint32_t x = minX; x <= maxX; x++) {
                _bins[x + y * _binsW].push_back(triangleId);
                STAT_INCREMENT(BinsFilled, 1);
            }
        }
    }
    STAT_INCREMENT(TrianglesDrawn, (uint32_t)std::popcount(mask));
}

Rasterizer::Rasterizer(std::shared_ptr<Framebuffer> fb) {
    _triangles = std::make_unique<TrianglePacket[]>(kTriangleBatchSize);

    _binsW = (fb->Width + kBinSize - 1) >> kBinSizeLog2;
    _binsH = (fb->Height + kBinSize - 1) >> kBinSizeLog2;
    _numBins = _binsW * _binsH;
    _bins = std::make_unique<std::vector<uint16_t>[]>(_numBins);

    const float maxViewSize = 2048.0f;
    _clipper.GuardBandPlaneDistXY[0] = maxViewSize / fb->Width;
    _clipper.GuardBandPlaneDistXY[1] = maxViewSize / fb->Height;

    _fb = std::move(fb);
}

// This impl performs no clipping and is intended for debugging only.
static void RenderWireframe(Framebuffer& fb, const TrianglePacket& tri, uint32_t i, uint32_t color = 0xFF'0000FF) {
    float sx = fb.Width * 0.5f, sy = fb.Height * 0.5f;

    for (uint32_t vi = 0; vi < 3; vi++) {
        const VFloat4& p1 = tri.Vertices[vi].Position;
        const VFloat4& p2 = tri.Vertices[(vi + 1) % 3].Position;

        // Draw line using fixed-point DDA algorithm
        int32_t x1 = (int32_t)((p1.x[i] * sx + sx) * 16384.0f);
        int32_t y1 = (int32_t)((p1.y[i] * sy + sy) * 16384.0f);
        int32_t x2 = (int32_t)((p2.x[i] * sx + sx) * 16384.0f);
        int32_t y2 = (int32_t)((p2.y[i] * sy + sy) * 16384.0f);

        int32_t dx = x2 - x1, dy = y2 - y1;
        int32_t length = std::max(std::abs(dx), std::abs(dy));

        if (length == 0 || length == INT_MIN) continue;

        dx = (int32_t)(((int64_t)dx << 14) / length);
        dy = (int32_t)(((int64_t)dy << 14) / length);

        int32_t steps = length >> 14;

        while (steps-- >= 0) {
            uint32_t x = (uint32_t)(x1 >> 14), y = (uint32_t)(y1 >> 14);

            if (x < fb.Width && y < fb.Height) {
                fb.ColorBuffer[fb.GetPixelOffset(x, y)] = color;
            }
            x1 += dx;
            y1 += dy;
        }
    }
}

void Rasterizer::RenderBins(std::function<void(const BinnedTriangle&)> drawFn) {
    STAT_TIME_BEGIN(Rasterize);

    auto binRange = std::ranges::iota_view(0u, _binsW * _binsH);

    std::vector<uint32_t> wireframeTids;

    if (EnableWireframe) {
        wireframeTids.resize(65536 / 32);

        for (uint32_t bid : binRange) {
            for (uint16_t tid : _bins[bid]) {
                wireframeTids[tid / 32] |= 1u << (tid % 32);
            }
        }
    }

    std::for_each(std::execution::par_unseq, binRange.begin(), binRange.end(), [&](const uint32_t& bid) {
        std::vector<uint16_t>& bin = _bins[bid];

        BinnedTriangle bt = {
            .X = (bid % _binsW) * kBinSize,
            .Y = (bid / _binsW) * kBinSize,
        };
        for (uint16_t triangleId : bin) {
            bt.TriangleId = triangleId;
            drawFn(bt);
        }
        bin.clear();
    });

    // TODO: This is slow and shitty, maybe use some math magic to draw wireframe in the rasterizer?
    //       https://math.stackexchange.com/questions/3748903/closest-point-to-triangle-edge-with-barycentric-coordinates
    for (uint32_t i = 0; i < wireframeTids.size(); i++) {
        if (wireframeTids[i] == 0) continue;

        for (uint32_t j : BitIter(wireframeTids[i])) {
            uint32_t tid = i * 32 + j;
            RenderWireframe(*_fb, _triangles[tid / VFloat::Length], tid % VFloat::Length);
        }
    }

    STAT_TIME_END(Rasterize);
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

};  // namespace swr
