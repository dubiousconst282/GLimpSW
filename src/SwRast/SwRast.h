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
    static constexpr uint32_t MaxVertices = 64, MaxPrims = 128, MaxAttribs = 16;

    uint8_t PrimCount = 0;
    alignas(64) uint8_t Indices[3][MaxPrims];
    alignas(64) float Attribs[16][MaxVertices];  // 0..3: SV_Position, 4..: Custom

    template<typename T>
    void SetAttrib(uint32_t attrId, uint32_t offset, const T& value) {
        assert(offset % 16 == 0 && sizeof(T) % sizeof(v_float) == 0);
        for (uint32_t i = 0; i < sizeof(T) / sizeof(v_float); i++) {
            memcpy(&Attribs[attrId + i][offset], (v_float*)&value + i, sizeof(v_float));
        }
    }
};

struct TrianglePacket {
    v_float2 Z0, Z1, Z2; // Z,W
    v_int MinX, MinY, MaxX, MaxY;
    v_int W0, W1, W2;
    v_int A01, A12, A20;
    v_int B01, B12, B20;
    v_float RcpArea;

    // Computes edge variables based on shaded vertices.
    void Setup(v_float4 v0, v_float4 v1, v_float4 v2, glm::ivec2 vpHalfSize);
};

struct VaryingBuffer {
    const ShadedMeshlet* Meshlet;
    uint32_t MeshletId, PrimitiveId;
    uint8_t Indices[3];

    uint32_t TileOffset;
    v_mask TileMask;

    v_float W0, W1, W2;
    v_float Depth;

    // Interpolates vertex attributes using the current barycentric weights.
    v_float GetSmooth(uint32_t attrId) const {
        float v0 = GetFlat(attrId, 0);
        float v1 = GetFlat(attrId, 1);
        float v2 = GetFlat(attrId, 2);
        return simd::fma(v0, W0, simd::fma(v1, W1, v2 * W2));           // 3 op per attrib + 2 setup
        // return simd::fma(v1 - v0, W1, simd::fma(v2 - v0, W2, v0));   // 4 op per attrib
    }
    // Returns the value of the specified vertex attribute, without interpolation.
    // If `vertexId != 0`, returns `attr[vertexId] - attr[0]`.
    float GetFlat(uint32_t attrId, uint32_t vertexId = 0) const {
        assert(attrId < 16 && vertexId < 3);
        return Meshlet->Attribs[attrId][Indices[vertexId]];
    }

    // Interpolates a vector of vertex attributes. `T` should be a struct containing only `v_float` fields.
    template<typename T>
    T GetSmooth(uint32_t attrId) const {
        static_assert(sizeof(T) % sizeof(v_float) == 0);

        constexpr uint32_t count = sizeof(T) / sizeof(v_float);
        v_float dest[count];

        for (uint32_t i = 0; i < count; i++) {
            dest[i] = GetSmooth(attrId + i);
        }
        return *(T*)&dest;
    }

    void ApplyPerspectiveCorrection() {
        // https://stackoverflow.com/a/24460895
        const uint32_t AttribW = 3;
        v_float v0 = GetFlat(AttribW, 0);
        v_float v1 = GetFlat(AttribW, 1);
        v_float v2 = GetFlat(AttribW, 2);
        v_float vp = simd::rcp14(simd::fma(v0, W0, simd::fma(v1, W1, v2 * W2)));

        W1 *= vp * v1;
        W2 *= vp * v2;
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
    struct Vertex {
        float Attribs[ShadedMeshlet::MaxAttribs];
    };
    struct ClipCodes {
        v_mask AcceptMask;       // Triangles that are in-bounds and can be immediately rasterized.
        v_mask NonTrivialMask;   // Triangles that need to be clipped.
        uint8_t OutCodes[16];   // Planes that need to be clipped against (per-triangle).
    };

    uint32_t Count, FreeVtx;
    float GuardBandPlaneDistXY[2]{ 1.0f, 1.0f };
    uint8_t Indices[24];
    Vertex Vertices[24];

    // Compute Cohen-Sutherland clip codes
    ClipCodes ComputeClipCodes(v_float4 v0, v_float4 v1, v_float4 v2);

    void ClipAgainstPlane(Plane plane, uint32_t numAttribs);

    void LoadTriangle(TrianglePacket& srcTri, uint32_t srcTriIdx, uint32_t numAttribs);
    void StoreTriangle(TrianglePacket& destTri, uint32_t destTriIdx, uint32_t srcTriFanIdx, uint32_t numAttribs);
};

template<typename T>
concept ShaderProgram =
    requires(T s, ShadedMeshlet& meshlet, Framebuffer& fb, VaryingBuffer& vars) {
        { T::NumCustomAttribs } -> std::convertible_to<uint32_t>;
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
        uint32_t minX = (uint32_t)tri.MinX[i];
        uint32_t minY = (uint32_t)tri.MinY[i];
        uint32_t maxX = (uint32_t)tri.MaxX[i];
        uint32_t maxY = (uint32_t)tri.MaxY[i];

        v_int tileOffsX = simd::FragPixelOffsetsX, tileOffsY = simd::FragPixelOffsetsY;

        v_int stepX0 = tri.A12[i], stepX1 = tri.A20[i], stepX2 = tri.A01[i];
        v_int stepY0 = tri.B12[i], stepY1 = tri.B20[i], stepY2 = tri.B01[i];

        // Barycentric coordinates at start of row
        v_int rowW0 = tri.W0[i] + stepX0 * tileOffsX + stepY0 * tileOffsY;
        v_int rowW1 = tri.W1[i] + stepX1 * tileOffsX + stepY1 * tileOffsY;
        v_int rowW2 = tri.W2[i] + stepX2 * tileOffsX + stepY2 * tileOffsY;

        stepX0 *= 4, stepX1 *= 4, stepX2 *= 4;
        stepY0 *= 4, stepY1 *= 4, stepY2 *= 4;

        float area = tri.RcpArea[i];
        VaryingBuffer vars = {
            .Meshlet = &mesh,
            .MeshletId = meshletId,
            .PrimitiveId = primIdx,
            .Indices = { mesh.Indices[0][primIdx], mesh.Indices[1][primIdx], mesh.Indices[2][primIdx] },
        };

        for (uint32_t y = minY; y <= maxY; y += 4) {
            v_int w0 = rowW0, w1 = rowW1, w2 = rowW2;

            for (uint32_t x = minX; x <= maxX; x += 4) {
                v_mask tileMask = simd::movemask((w0 | w1 | w2) >= 0);

                if (tileMask != 0) [[unlikely]] {
                    vars.TileOffset = _fb.GetPixelOffset(x, y);
                    vars.W1 = simd::conv2f(w1) * area;
                    vars.W2 = simd::conv2f(w2) * area;
                    vars.W0 = 1.0f - vars.W1 - vars.W2;

                    v_float oldDepth = simd::load(&_fb.DepthBuffer[vars.TileOffset]);
                    v_float newDepth = simd::fma(tri.Z0.x, vars.W0, simd::fma(tri.Z1.x, vars.W1, tri.Z2.x * vars.W2));

                    //tileMask &= simd::movemask(newDepth < oldDepth);

                    if (tileMask != 0) {
                        vars.Depth = newDepth;
                        vars.TileMask = tileMask;

                        [[clang::always_inline]] shader.ShadePixels(_fb, vars);
                    }
                }
                w0 += stepX0, w1 += stepX1, w2 += stepX2;
            }
            rowW0 += stepY0, rowW1 += stepY1, rowW2 += stepY2;
        }
    }

public:
    Rasterizer(Framebuffer& fb) : _fb(fb), _batch(fb.Width, fb.Height) {
        const float maxViewSize = 2048.0f;
        _clipper.GuardBandPlaneDistXY[0] = maxViewSize / fb.Width;
        _clipper.GuardBandPlaneDistXY[1] = maxViewSize / fb.Height;
    }

    template<ShaderProgram TShader>
    void DrawMeshlets(uint32_t count, const TShader& shader) {
        DrawMeshletsImpl(count, {
            .MeshFn = [&](uint32_t index, ShadedMeshlet& meshlet) { shader.ShadeMeshlet(index, meshlet); },
            .DrawFn = [&](auto&&... args) { DrawTriangleImmediate(shader, args...); }
          //  .NumCustomAttribs = shader.NumCustomAttribs,
        });
    }
};

};  // namespace swr