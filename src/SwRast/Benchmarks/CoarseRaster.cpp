#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "../Rasterizer.h"
#include "../Texture.h"
#include <stb_image_write.h>

using namespace swr;

static constexpr v_int FragPixelOffsetsX = simd::lane_idx<v_int> & 3;
static constexpr v_int FragPixelOffsetsY = simd::lane_idx<v_int> >> 2;

SIMD_INLINE static v_mask ReduceEdgeMask(v_int3 e) {
    // return simd::movemask((a | b | c) >= 0); // llvm hoists const(-1) into extra zmm register
    return _mm512_movepi32_mask(_mm512_ternarylogic_epi32(e.x, e.y, e.z, ~(_MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C)));
}

static void DrawTriangle(Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
    [[assume(i < simd::vec_width)]];  // avoids masking on vector indexing

    uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
    uint16_t offsetExtentX = boundRect >> 32, offsetExtentY = boundRect >> 48;  // (max - min) / 4 * 16

    // Using offset for loop control saves 2 registers and a bit of sALU
    uint32_t offset = ((minX >> 2) + (minY >> 2) * fb.TileStride) * 16;  // fb.GetPixelOffset(minX * 4, minY * 4)
    uint32_t endOffsetY = offset + fb.TileStride * offsetExtentY - offsetExtentX;
    uint32_t offsetIncY = fb.TileStride * 16 - offsetExtentX;

    // Barycentric coordinates at start of row
    v_int3 edgeStepX = v_int3(tri.A12[i], tri.A20[i], tri.A01[i]);
    v_int3 edgeStepY = v_int3(tri.B12[i], tri.B20[i], tri.B01[i]);
    v_int3 edgeOrigin = v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
    edgeOrigin += edgeStepX * (minX + FragPixelOffsetsX);
    edgeOrigin += edgeStepY * (minY + FragPixelOffsetsY);

    edgeStepX *= 4, edgeStepY *= 4;

    [[assume(offset < endOffsetY)]];  // avoids redundant early bounds check

    for (; offset < endOffsetY; offset += offsetIncY) {
        v_int3 edge = edgeOrigin;

        uint32_t endOffsetX = offset + offsetExtentX;
        [[assume(offset < endOffsetX)]];

        for (; offset < endOffsetX; offset += 16) {
            v_mask tileMask = ReduceEdgeMask(edge);

            if (tileMask != 0) {
                v_float u = simd::conv<float>(edge.y);
                v_float v = simd::conv<float>(edge.z);

                // Perspective correction - https://stackoverflow.com/a/24460895
                // (Relying on code motion to move this after depth-test branch in shader)
                v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                v_float pw2 = tri.W2S[i];
                v_float rcpW = simd::approx_rcp(simd::fma(u, pw1, simd::fma(v, pw2, pw0)));
                u *= pw1 * rcpW;
                v *= pw2 * rcpW;

                v_uint color = swr::pixfmt::RGBA8u::Pack({ 1 - u - v, u, v, 1 });
                _mm512_mask_store_epi32(&fb.GetColorBuffer()[offset], tileMask, color);
            }
            edge += edgeStepX;
        }
        edgeOrigin += edgeStepY;
    }
}

static void DrawTriangle_Hier_TileLoop(Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
    [[assume(i < simd::vec_width)]];  // avoids masking on vector indexing

    uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
    uint16_t extentX = boundRect >> 34, extentY = boundRect >> 50;  // (max - min)

    // Barycentric coordinates at start of row
    v_int3 edgeStepX = v_int3(tri.A12[i], tri.A20[i], tri.A01[i]);
    v_int3 edgeStepY = v_int3(tri.B12[i], tri.B20[i], tri.B01[i]);
    v_int3 edgeOrigin = v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
    edgeOrigin += edgeStepX * (minX + FragPixelOffsetsX);
    edgeOrigin += edgeStepY * (minY + FragPixelOffsetsY);

    uint16_t blocks[(1024 * 1024) / (16 * 16) * 3];
    uint16_t* lastBlock = blocks;

    // Edge accept/reject thresholds at outer/innermost tile corner
    v_int3 edgeTileRej = {
        (simd::max(edgeStepX.x, 0) + simd::max(edgeStepY.x, 0)) * 3,
        (simd::max(edgeStepX.y, 0) + simd::max(edgeStepY.y, 0)) * 3,
        (simd::max(edgeStepX.z, 0) + simd::max(edgeStepY.z, 0)) * 3,
    };
    v_int3 edgeTileAcc = {
        (simd::min(edgeStepX.x, 0) + simd::min(edgeStepY.x, 0)) * 3,
        (simd::min(edgeStepX.y, 0) + simd::min(edgeStepY.y, 0)) * 3,
        (simd::min(edgeStepX.z, 0) + simd::min(edgeStepY.z, 0)) * 3,
    };
    edgeTileAcc -= edgeTileRej;

    edgeTileRej += v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
    edgeTileRej += edgeStepX * (minX + FragPixelOffsetsX * 4);
    edgeTileRej += edgeStepY * (minY + FragPixelOffsetsY * 4);

    edgeStepX <<= 4, edgeStepY <<= 4;

    for (uint32_t y = 0; y < extentY; y += 16) {
        v_int3 edgeRow = edgeTileRej;

        for (uint32_t x = 0; x < extentX; x += 16) {
            v_mask partialMask = ReduceEdgeMask(edgeRow);

            if (partialMask != 0) {
                v_mask acceptMask = ReduceEdgeMask(edgeRow + edgeTileAcc);

                // Partial mask may overestimate, mask to avoid going off bin rect
                if (x + 16 > extentX) partialMask &= (0x1111 << ((extentX & 15) / 4)) - 0x1111;
                if (y + 16 > extentY) partialMask &= 0xFFFF >> (-extentY & 15);

                lastBlock[0] = (x >> 4) | (y << 4);
                lastBlock[1] = partialMask;
                lastBlock[2] = acceptMask;
                lastBlock += 3;
            }
            edgeRow += edgeStepX;
        }
        edgeTileRej += edgeStepY;
    }

    edgeStepX >>= 4, edgeStepY >>= 4;

    for (uint16_t* b = blocks; b < lastBlock; b += 3) {
        uint32_t blockMask = b[1];
        uint32_t trivialMask = b[2];

        uint32_t blockX = (b[0] << 4 & 0xFF0), blockY = (b[0] >> 4 & 0xFF0);
        uint32_t tileOffset = fb.GetPixelOffset(minX + blockX, minY + blockY);

        v_int3 tileEdge = edgeOrigin + edgeStepX * (int)blockX + edgeStepY * (int)blockY;

        for (uint32_t y = 0; y < 4; y++) {
            v_int3 edge = tileEdge;

            for (uint32_t x = 0; x < 4; x++) {
                uint32_t offset = tileOffset + x * 16;

                // simd::store(&fb.GetColorBuffer()[offset], (trivialMask & (1u << tileId)) ? v_int(0xFF'40E040) : v_int(0xFF'E0E040));

                if (blockMask & 1) {
                    v_mask tileMask = ReduceEdgeMask(edge);

                    v_float u = simd::conv<float>(edge.y);
                    v_float v = simd::conv<float>(edge.z);

                    // Perspective correction - https://stackoverflow.com/a/24460895
                    // (Relying on code motion to move this after depth-test branch in shader)
                    v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                    v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                    v_float pw2 = tri.W2S[i];
                    v_float rcpW = simd::approx_rcp(simd::fma(u, pw1, simd::fma(v, pw2, pw0)));
                    u *= pw1 * rcpW;
                    v *= pw2 * rcpW;

                    v_uint color = swr::pixfmt::RGBA8u::Pack({ 1 - u - v, u, v, 1 });
                    _mm512_mask_store_epi32(&fb.GetColorBuffer()[offset], tileMask, color);
                }
                edge += edgeStepX << 2;
                blockMask >>= 1;
            }
            tileOffset += fb.TileStride * 16;
            tileEdge += edgeStepY << 2;
        }
    }
}

static void DrawTriangle_Hier_BitIter(Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
    [[assume(i < simd::vec_width)]];  // avoids masking on vector indexing

    uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
    uint16_t extentX = boundRect >> 34, extentY = boundRect >> 50;  // (max - min)

    // Barycentric coordinates at start of row
    v_int3 edgeStepX = v_int3(tri.A12[i], tri.A20[i], tri.A01[i]);
    v_int3 edgeStepY = v_int3(tri.B12[i], tri.B20[i], tri.B01[i]);
    v_int3 edgeOrigin = v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
    edgeOrigin += edgeStepX * (minX + FragPixelOffsetsX);
    edgeOrigin += edgeStepY * (minY + FragPixelOffsetsY);

    uint16_t blocks[(1024 * 1024) / (16 * 16) * 3];
    uint16_t* lastBlock = blocks;

    // Edge accept/reject thresholds at outer/innermost tile corner
    v_int3 edgeTileRej = {
        (simd::max(edgeStepX.x, 0) + simd::max(edgeStepY.x, 0)) * 3,
        (simd::max(edgeStepX.y, 0) + simd::max(edgeStepY.y, 0)) * 3,
        (simd::max(edgeStepX.z, 0) + simd::max(edgeStepY.z, 0)) * 3,
    };
    v_int3 edgeTileAcc = {
        (simd::min(edgeStepX.x, 0) + simd::min(edgeStepY.x, 0)) * 3,
        (simd::min(edgeStepX.y, 0) + simd::min(edgeStepY.y, 0)) * 3,
        (simd::min(edgeStepX.z, 0) + simd::min(edgeStepY.z, 0)) * 3,
    };
    edgeTileAcc -= edgeTileRej;

    edgeTileRej += v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
    edgeTileRej += edgeStepX * (minX + FragPixelOffsetsX * 4);
    edgeTileRej += edgeStepY * (minY + FragPixelOffsetsY * 4);

    edgeStepX <<= 4, edgeStepY <<= 4;

    for (uint32_t y = 0; y < extentY; y += 16) {
        v_int3 edgeRow = edgeTileRej;

        for (uint32_t x = 0; x < extentX; x += 16) {
            v_mask partialMask = ReduceEdgeMask(edgeRow);

            if (partialMask != 0) {
                v_mask trivialMask = ReduceEdgeMask(edgeRow + edgeTileAcc);

                // Partial mask may overestimate, mask to avoid going off bin rect
                if (x + 16 > extentX) partialMask &= (0x1111 << ((extentX & 15) / 4)) - 0x1111;
                if (y + 16 > extentY) partialMask &= 0xFFFF >> (-extentY & 15);

                lastBlock[0] = (x >> 4) | (y << 4);
                lastBlock[1] = partialMask;
                lastBlock[2] = trivialMask;
                lastBlock += 3;
            }
            edgeRow += edgeStepX;
        }
        edgeTileRej += edgeStepY;
    }

    edgeStepX >>= 2, edgeStepY >>= 2;

    v_int edgeSum = edgeOrigin.x + edgeOrigin.y + edgeOrigin.z;

    for (uint16_t* b = blocks; b < lastBlock; b += 3) {
        uint32_t blockMask = b[1];
        uint32_t trivialMask = b[2];

        uint32_t blockX = (b[0] << 2 & 0x3FC), blockY = (b[0] >> 6 & 0x3FC);
        uint32_t tileOffset = fb.GetPixelOffset(minX + blockX * 4, minY + blockY * 4);

        v_int3 tileEdge = edgeOrigin + edgeStepX * (int)blockX + edgeStepY * (int)blockY;

        if (trivialMask == 0xFFFF) {
            for (uint32_t y = 0; y < 4; y++) {
                v_int3 edge = tileEdge;

                for (uint32_t x = 0; x < 4; x++) {
                    uint32_t offset = tileOffset + x * 16;
                    v_mask tileMask = 0xFFFF;

                    v_float u = simd::conv<float>(edge.y);
                    v_float v = simd::conv<float>(edge.z);

                    // Perspective correction - https://stackoverflow.com/a/24460895
                    // (Relying on code motion to move this after depth-test branch in shader)
                    v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                    v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                    v_float pw2 = tri.W2S[i];
                    v_float rcpW = simd::approx_rcp(simd::fma(u, pw1, simd::fma(v, pw2, pw0)));
                    u *= pw1 * rcpW;
                    v *= pw2 * rcpW;

                    v_uint color = swr::pixfmt::RGBA8u::Pack({ 1 - u - v, u, v, 1 });
                    _mm512_mask_store_epi32(&fb.GetColorBuffer()[offset], tileMask, color);

                    edge += edgeStepX;
                }
                tileOffset += fb.TileStride * 16;
                tileEdge += edgeStepY;
            }
        } else {
            uint32_t tj = 3;
            for (uint32_t ti : simd::BitIter(blockMask)) {
                #pragma nounroll
                while (tj < ti) tileEdge += edgeStepY, tj += 4;

                uint32_t offset = tileOffset + ((ti & 3) + (ti >> 2) * fb.TileStride) * 16;

                v_int3 edge = tileEdge + edgeStepX * (ti & 3);// + edgeStepY * (i >> 2);
                edge.x = edgeSum - edge.y - edge.z; // saves 2 muls

                v_mask tileMask = ReduceEdgeMask(edge);

                v_float u = simd::conv<float>(edge.y);
                v_float v = simd::conv<float>(edge.z);

                // Perspective correction - https://stackoverflow.com/a/24460895
                // (Relying on code motion to move this after depth-test branch in shader)
                v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                v_float pw2 = tri.W2S[i];
                v_float rcpW = simd::approx_rcp(simd::fma(u, pw1, simd::fma(v, pw2, pw0)));
                u *= pw1 * rcpW;
                v *= pw2 * rcpW;

                v_uint color = swr::pixfmt::RGBA8u::Pack({ 1 - u - v, u, v, 1 });
                _mm512_mask_store_epi32(&fb.GetColorBuffer()[offset], tileMask, color);
            }
        }
    }
}

static void DrawTriangle_RowSkip(Framebuffer& fb, const TriangleEdgeVars& tri, uint32_t i, uint64_t boundRect) {
    [[assume(i < simd::vec_width)]];  // avoids masking on vector indexing

    uint16_t minX = boundRect >> 0, minY = boundRect >> 16;
    uint16_t extentX = boundRect >> 36, extentY = boundRect >> 52;

    // Using offset for loop control saves 2 registers and a bit of sALU
    uint32_t offset = ((minX >> 2) + (minY >> 2) * fb.TileStride) * 16;  // fb.GetPixelOffset(minX * 4, minY * 4)
    uint32_t offsetIncY = fb.TileStride * 16;
    uint32_t endOffsetY = offset + extentY * offsetIncY;

    // Barycentric coordinates at start of row
    v_int3 edgeStepX = v_int3(tri.A12[i], tri.A20[i], tri.A01[i]);
    v_int3 edgeStepY = v_int3(tri.B12[i], tri.B20[i], tri.B01[i]);
    v_int3 edgeOrigin = v_int3(tri.Edge0[i], tri.Edge1[i], tri.Edge2[i]);
    edgeOrigin += edgeStepX * (minX + FragPixelOffsetsX);
    edgeOrigin += edgeStepY * (minY + FragPixelOffsetsY);

    // Edge rejection thresholds at top-left tile corner
    v_int3 distRejCorner = {
        (simd::max(edgeStepX.x, 0) + simd::max(edgeStepY.x, 0)) * 3,
        (simd::max(edgeStepX.y, 0) + simd::max(edgeStepY.y, 0)) * 3,
        (simd::max(edgeStepX.z, 0) + simd::max(edgeStepY.z, 0)) * 3,
    };
    distRejCorner += edgeStepX * (simd::lane_idx<v_int> * 4 - swr::TilePixelOffsetsX);
    distRejCorner -= edgeStepY * swr::TilePixelOffsetsY;

    edgeStepX *= 4, edgeStepY *= 4;

    [[assume(offset < endOffsetY)]];  // avoids redundant early bounds check

    for (; offset < endOffsetY; offset += offsetIncY) {
        v_int3 edge = edgeOrigin;
        uint32_t endX = extentX;

        for (uint32_t x = 0; x < endX;) {
            if ((x & 15) == 0) [[unlikely]] {
                uint32_t rowMask = ReduceEdgeMask(edge + distRejCorner);

                if (rowMask != 0 && (rowMask & 0x8000) == 0) {
                    endX = std::min((uint32_t)extentX, x + (32 - simd::lzcnt(rowMask)));
                }
                uint32_t x0 = simd::tzcnt(rowMask | 0x1'0000);
                edge += edgeStepX * (int)x0;
                x += x0;
                if (rowMask == 0) continue;
            }
            v_mask tileMask = ReduceEdgeMask(edge);

            if (tileMask != 0) [[likely]] {
                v_float u = simd::conv<float>(edge.y);
                v_float v = simd::conv<float>(edge.z);

                // Perspective correction - https://stackoverflow.com/a/24460895
                // (Relying on code motion to move this after depth-test branch in shader)
                v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                v_float pw2 = tri.W2S[i];
                v_float rcpW = simd::approx_rcp(simd::fma(u, pw1, simd::fma(v, pw2, pw0)));
                u *= pw1 * rcpW;
                v *= pw2 * rcpW;

                v_uint color = swr::pixfmt::RGBA8u::Pack({ 1 - u - v, u, v, 1 });
                _mm512_mask_store_epi32(&fb.GetColorBuffer()[offset + x * 16], tileMask, color);
            }
            edge += edgeStepX;
            x++;
        }
        edgeOrigin += edgeStepY;
    }
}

int main(int argc, const char** args) {
    const int TriN = 3;
    glm::vec2 vertices[TriN][3] = {
        // A ____ C
        //   \  /
        //    \/
        //     B
        { { -1, -1 }, { 0, +1 }, { +1, -1 } },

        // A ___ C
        //   | /
        //   |/
        //   B
        { { -1, -1 }, { -1, +1 }, { +1, -1 } },

        // Stretched thin
        { { -1, -1 }, { +1, +1 }, { 0.2, -0.2 } },
    };

    v_float4 v[3];

    for (uint32_t vi = 0; vi < 3; vi++) {
        for (uint32_t ti = 0; ti < TriN; ti++) {
            v[vi].x[ti] = vertices[ti][vi].x;
            v[vi].y[ti] = vertices[ti][vi].y;
        }
        v[vi].z = 0;
        v[vi].w = 1;
    }

    for (uint32_t sz : { 1024u, 64u, 32u }) {
        auto fb = swr::CreateFramebuffer(sz, sz);
        fb->Clear(0, 0.0f);

        swr::TrianglePacket tris;
        glm::ivec2 vpHalfSize = glm::ivec2(fb->Width, fb->Height) / 2;

        tris.Setup(v[2], v[1], v[0], vpHalfSize, swr::FaceCullMode::None);
        v_uint2 boundBox = tris.GetRenderBoundingBox(vpHalfSize);

        swr::TriangleEdgeVars edges;
        edges.Setup(tris, vpHalfSize);

        if (sz == 1024) {
            uint32_t i = 2;
            uint64_t boundRect = boundBox.x[i] | uint64_t(boundBox.y[i] - boundBox.x[i]) << 34;

            fb->Clear(0, 0.0f);
            DrawTriangle_Hier_BitIter(*fb, edges, i, boundRect);

            auto pixels = simd::alloc_buffer<uint32_t>(fb->Width * fb->Height);
            fb->GetPixels(0, pixels.get(), fb->Width);
            stbi_write_png("logs/bench_view.png", (int)fb->Width, (int)fb->Height, 4, pixels.get(), (int)fb->Width * 4);
        }

        for (uint32_t i = 0; i < TriN; i++) {
            uint64_t boundRect = boundBox.x[i] | uint64_t(boundBox.y[i] - boundBox.x[i]) << 34;
            std::string tag = "_" + std::to_string(sz) + "_T" + std::to_string(i);

            fb->Clear(0, 0.0f);

            ankerl::nanobench::Bench ctx;
            ctx.minEpochTime(std::chrono::milliseconds(30));
            ctx.warmup(10).epochs(50);
            ctx.relative(true);

            ctx.run("Standard" + tag, [&]() { DrawTriangle(*fb, edges, i, boundRect); });
           // ctx.run("Hier_TileLoop" + tag, [&]() { DrawTriangle_Hier_TileLoop(*fb, edges, i, boundRect); });
            ctx.run("Hier_BitIter" + tag, [&]() { DrawTriangle_Hier_BitIter(*fb, edges, i, boundRect); });
            ctx.run("RowSkip" + tag, [&]() { DrawTriangle_RowSkip(*fb, edges, i, boundRect); });
            printf("\n");
        }
    }

    return 0;
}


/*

| relative |      ns/op |         op/s | err% |     ins/op |     cyc/op |    IPC |    bra/op |   miss% | benchmark
|---------:|-----------:|-------------:|-----:|-----------:|-----------:|-------:|----------:|--------:|:----------
|   100.0% | 361,548.48 |     2,765.88 | 0.4% | 720,925.29 | 370,038.87 |  1.948 | 70,577.69 |    0.4% | `Standard_1024_T0`
|   123.2% | 293,520.18 |     3,406.92 | 0.2% | 529,002.63 | 300,191.03 |  1.762 | 12,205.24 |    0.5% | `Hier_BitIter_1024_T0`
|    99.8% | 362,304.43 |     2,760.11 | 0.2% | 659,294.67 | 368,692.22 |  1.788 | 47,024.26 |    1.7% | `RowSkip_1024_T0`

|   100.0% | 356,038.03 |     2,808.69 | 0.1% | 719,862.20 | 363,980.77 |  1.978 | 70,472.91 |    0.3% | `Standard_1024_T1`
|   123.3% | 288,824.08 |     3,462.32 | 0.1% | 518,651.06 | 294,857.24 |  1.759 | 11,393.04 |    0.5% | `Hier_BitIter_1024_T1`
|   102.3% | 347,942.17 |     2,874.04 | 0.2% | 644,065.51 | 354,023.73 |  1.819 | 44,785.24 |    1.7% | `RowSkip_1024_T1`

|   100.0% | 150,895.25 |     6,627.11 | 0.2% | 395,450.82 | 155,156.27 |  2.549 | 59,271.24 |    0.5% | `Standard_1024_T2`
|   221.3% |  68,186.71 |    14,665.62 | 0.1% | 138,585.50 |  69,856.27 |  1.984 |  6,844.02 |    0.2% | `Hier_BitIter_1024_T2`
|   151.9% |  99,360.22 |    10,064.39 | 0.3% | 161,111.46 | 102,083.13 |  1.578 | 14,123.17 |    1.3% | `RowSkip_1024_T2`

|   100.0% |   1,408.40 |   710,025.13 | 0.2% |   3,005.53 |   1,460.13 |  2.058 |    280.66 |    0.0% | `Standard_64_T0`
|    95.3% |   1,478.43 |   676,393.82 | 0.2% |   3,048.95 |   1,528.25 |  1.995 |    141.48 |    0.0% | `Hier_BitIter_64_T0`
|    91.0% |   1,547.44 |   646,228.80 | 0.2% |   3,025.61 |   1,588.81 |  1.904 |    218.76 |    0.0% | `RowSkip_64_T0`

|   100.0% |   1,354.49 |   738,286.08 | 0.2% |   2,889.59 |   1,387.56 |  2.082 |    274.92 |    0.0% | `Standard_64_T1`
|   104.2% |   1,299.65 |   769,435.18 | 0.2% |   2,555.24 |   1,336.46 |  1.912 |     94.48 |    0.0% | `Hier_BitIter_64_T1`
|    91.9% |   1,474.22 |   678,322.80 | 0.3% |   2,896.79 |   1,522.40 |  1.903 |    209.89 |    0.0% | `RowSkip_64_T1`

|   100.0% |     630.60 | 1,585,787.84 | 0.2% |   1,779.39 |     648.15 |  2.745 |    236.91 |    0.0% | `Standard_64_T2`
|   108.8% |     579.42 | 1,725,853.63 | 0.2% |   1,703.55 |     747.41 |  2.279 |    116.88 |    0.0% | `Hier_BitIter_64_T2`
|   102.1% |     617.55 | 1,619,301.54 | 0.2% |   1,583.85 |     850.47 |  1.862 |    125.03 |    0.0% | `RowSkip_64_T2`

|   100.0% |     373.53 | 2,677,166.85 | 0.2% |   1,058.94 |     512.72 |  2.065 |     93.97 |    0.0% | `Standard_32_T0`
|    77.4% |     482.31 | 2,073,376.48 | 0.2% |   1,425.86 |     666.45 |  2.139 |     83.25 |    0.0% | `Hier_BitIter_32_T0`
|    78.6% |     475.01 | 2,105,225.39 | 0.1% |   1,246.76 |     654.46 |  1.905 |     91.78 |    0.0% | `RowSkip_32_T0`

|   100.0% |     347.78 | 2,875,384.91 | 0.1% |   1,008.84 |     478.51 |  2.108 |     92.22 |    0.0% | `Standard_32_T1`
|    88.6% |     392.33 | 2,548,888.89 | 0.1% |   1,094.65 |     539.55 |  2.029 |     51.57 |    0.0% | `Hier_BitIter_32_T1`
|    79.4% |     438.12 | 2,282,502.81 | 0.1% |   1,145.33 |     603.55 |  1.898 |     84.86 |    0.0% | `RowSkip_32_T1`

|   100.0% |     193.65 | 5,163,884.47 | 0.2% |     692.41 |     265.96 |  2.603 |     81.20 |    0.0% | `Standard_32_T2`
|    83.2% |     232.74 | 4,296,664.92 | 0.2% |     737.55 |     321.67 |  2.293 |     49.35 |    0.0% | `Hier_BitIter_32_T2`
|    75.1% |     257.96 | 3,876,622.06 | 0.1% |     667.50 |     355.12 |  1.880 |     52.29 |    0.0% | `RowSkip_32_T2`

*/