#pragma once

#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "SIMD.h"

namespace swr {

struct Framebuffer {
    // Data is stored in tiles of 4x4 so that rasterizer writes are cheap.
    static constexpr uint32_t TileSize = 4, TileShift = 2, TileMask = TileSize - 1, TileNumPixels = TileSize * TileSize;

    uint32_t Width, Height, TileStride;
    uint32_t AttachmentStride, NumAttachments;

    // TODO: experiment with fast clears for depth
    AlignedBuffer<uint32_t> ColorBuffer;
    AlignedBuffer<float> DepthBuffer;
    AlignedBuffer<uint8_t> AttachmentBuffer;

    Framebuffer(uint32_t width, uint32_t height, uint32_t numAttachments = 0) {
        Width = (width + TileMask) & ~TileMask;
        Height = (height + TileMask) & ~TileMask;
        TileStride = Width / TileSize;
        AttachmentStride = (Width * Height + 63) & ~63u;
        NumAttachments = numAttachments;

        ColorBuffer = alloc_buffer<uint32_t>(Width * Height);
        DepthBuffer = alloc_buffer<float>(Width * Height);
        AttachmentBuffer = alloc_buffer<uint8_t>(AttachmentStride * numAttachments);
    }

    void Clear(uint32_t color, float depth) {
        FillBuffer(ColorBuffer.get(), color);
        ClearDepth(depth);
    }
    void ClearDepth(float depth) { FillBuffer(DepthBuffer.get(), std::bit_cast<uint32_t>(depth)); }

    // Iterate through framebuffer tiles, potentially in parallel. `visitor` takes base tile X and Y coords.
    void IterateTiles(std::function<void(uint32_t, uint32_t)> visitor, uint32_t downscaleFactor = 1);

    uint32_t GetPixelOffset(uint32_t x, uint32_t y) const {
        assert(x + 3 < Width && y + 3 < Height);
        uint32_t tileId = (x >> TileShift) + (y >> TileShift) * TileStride;
        uint32_t pixelOffset = (x & TileMask) + (y & TileMask) * TileSize;
        return tileId * TileNumPixels + pixelOffset;
    }
    v_int GetPixelOffset(v_int x, v_int y) const {
        v_int tileId = (x >> TileShift) + (y >> TileShift) * (int32_t)TileStride;
        v_int pixelOffset = (x & TileMask) + (y & TileMask) * TileSize;
        return tileId * TileNumPixels + pixelOffset;
    }

    void WriteTile(uint32_t offset, uint16_t mask, v_int color, v_float depth) {
        _mm512_mask_store_epi32(&ColorBuffer[offset], mask, color);
        _mm512_mask_store_ps(&DepthBuffer[offset], mask, depth);
    }

    template<typename T>
    T* GetAttachmentBuffer(uint32_t attachmentId, size_t offset = 0) {
        assert(attachmentId + sizeof(T) <= NumAttachments && "Missing attachment storage");
        return (T*)&AttachmentBuffer[attachmentId * AttachmentStride] + offset;
    }

    [[gnu::always_inline]] v_float SampleDepth(v_float x, v_float y) const {
        v_int ix = simd::round2i(x * (int32_t)Width);
        v_int iy = simd::round2i(y * (int32_t)Height);
        return SampleDepth(ix, iy);
    }
    [[gnu::always_inline]] v_float SampleDepth(v_int ix, v_int iy) const {
        // Twos-complement unsigned compare trick to check for both (x >= 0 && x < N) at once.
        v_int indices = GetPixelOffset(ix, iy);
        v_mask boundMask = _mm512_cmplt_epu32_mask(ix, v_int((int32_t)Width)) & _mm512_cmplt_epu32_mask(iy, v_int((int32_t)Height));
        return _mm512_mask_i32gather_ps(_mm512_set1_ps(1.0f), boundMask, indices, DepthBuffer.get(), 4);
    }
    [[gnu::always_inline]] v_int SampleColor(v_int ix, v_int iy, v_int defaultColor = 0) const {
        // Twos-complement unsigned compare trick to check for both (x >= 0 && x < N) at once.
        v_int indices = GetPixelOffset(ix, iy);
        v_mask boundMask = _mm512_cmplt_epu32_mask(ix, v_int((int32_t)Width)) & _mm512_cmplt_epu32_mask(iy, v_int((int32_t)Height));
        return _mm512_mask_i32gather_epi32(defaultColor, boundMask, indices, ColorBuffer.get(), 4);
    }

    void GetPixels(uint32_t* dest, uint32_t stride) const;

private:
    void FillBuffer(void* ptr, uint32_t value) {
        // Non-temporal fill is ~2-3x faster than memset(), but makes rasterization a bit slower. Still a small win overall.
        // https://en.algorithmica.org/hpc/cpu-cache/bandwidth/
        uint32_t count = Width * Height;
        for (uint32_t i = 0; i < count; i += 16) {
            _mm512_stream_si512((uint32_t*)ptr + i, _mm512_set1_epi32((int32_t)value));
        }
        _mm_sfence();
    }
};

struct ShadedMeshlet {
    static constexpr uint32_t MaxVertices = 64, MaxPrims = 128;

    uint8_t PrimCount = 0;
    alignas(64) uint8_t Indices[3][MaxPrims];
    alignas(64) float Position[4][MaxVertices];

    void SetPosition(uint32_t index, v_float4 value) {
        assert(index % simd::vec_width == 0);
        simd::store(value.x, &Position[0][index]);
        simd::store(value.y, &Position[1][index]);
        simd::store(value.z, &Position[2][index]);
        simd::store(value.w, &Position[3][index]);
    }
};

enum class FaceCullMode { None, FrontCCW, FrontCW };

struct TrianglePacket {
    v_uint MinXY, MaxXY;
    v_int Edge0, Edge1, Edge2;
    v_int A01, A12, A20;
    v_int B01, B12, B20;

    v_float Z0, Z10, Z20;
    v_float W0, W0S, W1S, W2S;

    // Computes edge variables based on vertices in NDC space.
    v_mask Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize, FaceCullMode cullMode);
};

struct FragmentVars {
    uint32_t MeshletId, PrimitiveId;
    uint8_t Indices[3];

    uint32_t TileOffset;
    v_mask TileMask;

    v_float3 Bary;
    v_float Depth;

    // Interpolates vertex attributes using the current barycentric weights.
    v_float Interpolate(v_float v0, v_float v1, v_float v2) const {
        return simd::fma(v0, Bary.x, simd::fma(v1, Bary.y, v2 * Bary.z));       // 3 op per attrib + setup
        // return simd::fma(v1 - v0, Bary.y, simd::fma(v2 - v0, Bary.z, v0));   // 4 op per attrib
    }

    // Interpolates a vector of vertex attributes. `T` should be a struct containing only `v_float` fields.
    template<typename T>
    T Interpolate(T v0, T v1, T v2) const {
        constexpr uint32_t count = sizeof(T) / sizeof(v_float);

        T result;
        result.x = Interpolate(v0.x, v1.x, v2.x);
        result.y = Interpolate(v0.y, v1.y, v2.y);
        if constexpr (count >= 3) result.z = Interpolate(v0.z, v1.z, v2.z);
        if constexpr (count >= 4) result.w = Interpolate(v0.w, v1.w, v2.w);
        return result;
    }
};

struct Clipper {
    enum class Plane {
        Left = 0,    // X-
        Right = 1,   // X+
        Bottom = 2,  // Y-
        Top = 3,     // Y+
        Near = 4,    // Z-
        Far = 5,     // Z+
    };
    struct ClipCodes {
        v_mask AcceptMask;      // Triangles that are in-bounds and can be immediately rasterized.
        v_mask NonTrivialMask;  // Triangles that need to be clipped.
        uint8_t OutCodes[16];   // Planes that need to be clipped against (per-triangle).
    };

    uint32_t Count, FreeVtx;
    glm::vec2 GuardBandPlaneDist = { 1.0f, 1.0f };
    uint8_t Indices[24];
    glm::vec4 Vertices[24];

    // Compute Cohen-Sutherland clip codes
    ClipCodes ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2);

    void ClipAgainstPlane(Plane plane);
};

template<typename T>
concept ShaderProgram =
    requires(T s, ShadedMeshlet& meshlet, Framebuffer& fb, FragmentVars& vars) {
        { s.ShadeMeshlet(0u, meshlet) } -> std::same_as<void>;
        { s.ShadePixels(fb, vars) } -> std::same_as<void>;
    };

struct TriangleBatch {
    static const uint32_t MaxSize = 4096 / simd::vec_width;
    static const uint32_t BinSizeLog2 = 7, BinSize = 1 << BinSizeLog2;

    std::unique_ptr<std::vector<uint16_t>[]> Bins;
    uint32_t BinsPerRow, NumBins;

    uint32_t Count = 0;
    TrianglePacket Triangles[MaxSize];

    TriangleBatch(uint32_t fbWidth, uint32_t fbHeight) {
        BinsPerRow = (fbWidth + BinSize - 1) >> BinSizeLog2;
        NumBins = ((fbHeight + BinSize - 1) >> BinSizeLog2) * BinsPerRow;
        Bins = std::make_unique<std::vector<uint16_t>[]>(NumBins);
    }

    TrianglePacket& Alloc() {
        assert(Count < MaxSize);
        return Triangles[Count++];
    }
    TrianglePacket& PeekLast(uint32_t offset = 0) {
        assert(Count - 1 + offset < MaxSize);
        return Triangles[Count - 1 + offset];
    }
    void AddBin(uint32_t x, uint32_t y, TrianglePacket& tri, uint32_t index) {
        assert((&tri - Triangles) < MaxSize);

        uint32_t id = (&tri - Triangles) * simd::vec_width + index;
        Bins[x + y * BinsPerRow].push_back(id);
    }
    bool IsFull() { return Count >= MaxSize - 24; } //Reserve 24*vec triangles for clipping
};

class Rasterizer {
    Framebuffer& _fb;
    Clipper _clipper;
    TriangleBatch _batch;

    struct BinnedTriangle {
        uint32_t X, Y;
        uint16_t TriangleIndex;
        const TrianglePacket* Triangle;
    };
    struct ShaderDispatcher {
        std::function<void(uint32_t, ShadedMeshlet&)> MeshFn;
        std::function<void(const TrianglePacket&, const ShadedMeshlet&, uint8_t, uint8_t, uint32_t)> DrawFn;
        uint32_t NumCustomAttribs;
    };

    void DrawMeshletsImpl(uint32_t count, const ShaderDispatcher& shader);

    template<ShaderProgram TShader>
    void DrawTriangleImmediate(const TShader& shader, const TrianglePacket& tri, const ShadedMeshlet& mesh,
                               uint8_t i, uint8_t primIdx, uint32_t meshletId) {
        uint32_t minX = tri.MinXY[i] & 0xFFFF, minY = tri.MinXY[i] >> 16;
        uint32_t maxX = tri.MaxXY[i] & 0xFFFF, maxY = tri.MaxXY[i] >> 16;
        [[assume(minX <= maxX), assume(minY <= maxY)]];         // avoids redundant initial bounds check
        [[assume((minX & 3) == 0), assume((minY & 3) == 0)]];   // avoids redundant masking in GetTileOffset()

        v_int stepX0 = tri.A12[i], stepX1 = tri.A20[i], stepX2 = tri.A01[i];
        v_int stepY0 = tri.B12[i], stepY1 = tri.B20[i], stepY2 = tri.B01[i];

        // Barycentric coordinates at start of row
        constexpr v_int tileOffsX = simd::FragPixelOffsetsX, tileOffsY = simd::FragPixelOffsetsY;
        v_int edge0 = tri.Edge0[i] + stepX0 * tileOffsX + stepY0 * tileOffsY;
        v_int edge1 = tri.Edge1[i] + stepX1 * tileOffsX + stepY1 * tileOffsY;
        v_int edge2 = tri.Edge2[i] + stepX2 * tileOffsX + stepY2 * tileOffsY;

        stepX0 *= 4, stepX1 *= 4, stepX2 *= 4;
        stepY0 *= 4, stepY1 *= 4, stepY2 *= 4;

        FragmentVars vars = {
            .MeshletId = meshletId,
            .PrimitiveId = primIdx,
            .Indices = { mesh.Indices[0][primIdx], mesh.Indices[1][primIdx], mesh.Indices[2][primIdx] },
        };

        for (uint32_t y = minY; y <= maxY; y += 4) {
            v_int edgeX0 = edge0, edgeX1 = edge1, edgeX2 = edge2;
            vars.TileOffset = _fb.GetPixelOffset(minX, y);

            for (uint32_t x = minX; x <= maxX; x += 4, vars.TileOffset += 16) {
                v_mask tileMask = simd::movemask((edgeX0 | edgeX1 | edgeX2) >= 0);

                if (tileMask != 0) [[unlikely]] {
                    v_float u = simd::conv2f(edgeX1);
                    v_float v = simd::conv2f(edgeX2);

                    vars.TileMask = tileMask;
                    vars.Depth = simd::fma(tri.Z10[i], u, simd::fma(tri.Z20[i], v, tri.Z0[i]));

                    // Perspective correction - https://stackoverflow.com/a/24460895
                    // (Relying on code motion to move this after depth-test branch in shader)
                    v_float pw0 = simd::fma(u + v, -tri.W0S[i], tri.W0[i]);
                    v_float pw1 = tri.W1S[i];  // (1/area) is baked in!
                    v_float pw2 = tri.W2S[i];
                    v_float rcpW = simd::rcp14(simd::fma(pw1, u, simd::fma(pw2, v, pw0)));
                    u *= pw1 * rcpW;
                    v *= pw2 * rcpW;
                    vars.Bary = { 1 - u - v, u, v };

                    [[clang::always_inline]] shader.ShadePixels(_fb, vars);
                }
                edgeX0 += stepX0, edgeX1 += stepX1, edgeX2 += stepX2;
            }
            edge0 += stepY0, edge1 += stepY1, edge2 += stepY2;
        }
    }

public:
    FaceCullMode CullMode = FaceCullMode::FrontCCW;

    Rasterizer(Framebuffer& fb) : _fb(fb), _batch(fb.Width, fb.Height) {
        const float maxViewSize = 2048.0f;
        _clipper.GuardBandPlaneDist = maxViewSize / glm::vec2(fb.Width, fb.Height);
    }

    template<ShaderProgram TShader>
    void DrawMeshlets(uint32_t count, const TShader& shader) {
        DrawMeshletsImpl(count, {
            .MeshFn = [&](uint32_t index, ShadedMeshlet& meshlet) { shader.ShadeMeshlet(index, meshlet); },
            .DrawFn = [&](auto&&... args) { DrawTriangleImmediate(shader, args...); }
        });
    }
};

};  // namespace swr