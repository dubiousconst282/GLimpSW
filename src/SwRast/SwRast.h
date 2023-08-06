#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "SIMD.h"

#define STAT_INCREMENT(key, amount) (swr::g_Stats.key += (amount))
#define STAT_TIME_BEGIN(key) swr::ProfilerStats::MeasureTime(swr::g_Stats.key##Time, 1)
#define STAT_TIME_END(key) swr::ProfilerStats::MeasureTime(swr::g_Stats.key##Time, 0)

namespace swr {

struct ProfilerStats {
    uint32_t TrianglesDrawn;
    uint32_t TrianglesClipped;
    uint32_t BinsFilled;
    int64_t SetupTime[2];
    int64_t ClippingTime[2];
    int64_t BinningTime[2];
    int64_t RasterizeTime[2];
    int64_t RasterizeCpuTime;

    void Reset() { *this = {}; }
    static void MeasureTime(int64_t key[2], bool begin);
};
extern ProfilerStats g_Stats;

struct Texture2D {
    uint32_t Width, Height, MipLevels;
    uint32_t StrideLog2; //Shift amount to get row offset from Y coord. Used to avoid expansive i32 vector mul.
    std::unique_ptr<uint32_t[]> Data;  // RGBA8 pixel data.

    Texture2D(uint32_t width, uint32_t height, uint32_t mipLevels);

    static Texture2D LoadImage(std::string_view filename, uint32_t mipLevels = 16);

    void SetPixels(const uint32_t* pixels, uint32_t stride);

    // NearestMipmapNearest
    VFloat4 __vectorcall SampleNearest(VFloat u, VFloat v) const;

    // Linear (no mipmap)
    VFloat4 __vectorcall SampleLinear(VFloat u, VFloat v) const;

    // Mag: Linear, Min: NearestMipmapNearest
    VFloat4 __vectorcall SampleHybrid(VFloat u, VFloat v) const;

private:
    float _scaleU, _scaleV, _scaleLerpU, _scaleLerpV;
    int32_t _maskU, _maskV, _maskLerpU, _maskLerpV;
    VInt _mipOffsets;

    VInt GatherPixels(VInt indices) const {
        return VInt::gather<4>((int32_t*)Data.get(), indices);
    }
    void GenerateMips();
};

struct Framebuffer {
    //Data is stored in tiles of 4x4 so that rasterizer writes are cheap.
    static const uint32_t TileSize = 4, TileShift = 2, TileMask = TileSize - 1, TileNumPixels = TileSize * TileSize;

    uint32_t Width, Height, TileStride;
    uint32_t* ColorBuffer;
    float* DepthBuffer;

    Framebuffer(uint32_t width, uint32_t height) {
        Width = (width + TileMask) & ~TileMask;
        Height = (height + TileMask) & ~TileMask;
        TileStride = Width / TileSize;

        ColorBuffer = (uint32_t*)_mm_malloc(Width * Height * 4, 64);
        DepthBuffer = (float*)_mm_malloc(Width * Height * 4, 64);
    }
    ~Framebuffer() {
        _mm_free(ColorBuffer);
        _mm_free(DepthBuffer);
    }

    void Clear(uint32_t color, float depth) {
        std::fill(&ColorBuffer[0], &ColorBuffer[Width * Height], color);
        ClearDepth(depth);
    }
    void ClearDepth(float depth) { std::fill(&DepthBuffer[0], &DepthBuffer[Width * Height], depth); }

    size_t GetPixelOffset(uint32_t x, uint32_t y) const {
        size_t tileId = (x >> TileShift) + (y >> TileShift) * TileStride;
        size_t pixelOffset = (x & TileMask) + (y & TileMask) * TileSize;
        return tileId * TileNumPixels + pixelOffset;
    }

    void WriteTile(uint32_t offset, uint16_t mask, VInt color, VFloat depth) const {
        _mm512_mask_storeu_epi32(&ColorBuffer[offset], mask, color);
        _mm512_mask_storeu_ps(&DepthBuffer[offset], mask, depth);
    }

    VFloat SampleDepth(VFloat x, VFloat y) const {
        VInt ix = simd::round2i(x * (int32_t)Width);
        VInt iy = simd::round2i(y * (int32_t)Height);

        VInt tileId = (ix >> TileShift) + (iy >> TileShift) * (int32_t)TileStride;
        VInt pixelOffset = (ix & TileMask) + (iy & TileMask) * TileSize;
        VInt indices = tileId * TileNumPixels + pixelOffset;
        uint16_t boundMask = _mm512_cmplt_epu32_mask(ix, VInt((int32_t)Width)) & _mm512_cmplt_epu32_mask(iy, VInt((int32_t)Height));

        return _mm512_mask_i32gather_ps(_mm512_set1_ps(1.0f), boundMask, indices, DepthBuffer, 4);
    }

    void GetPixels(uint32_t* dest, uint32_t stride) const;
    void SaveImage(std::string_view filename) const;
};

struct VertexReader {
    enum IndexFormat { U8, U16, U32 };

    const uint8_t* VertexBuffer;
    const uint8_t* IndexBuffer;
    uint32_t Count;

    IndexFormat IndexFormat;
    VInt _Indices = 0;    // Vertex indices to be read next

    // Note that the index buffer should be over-allocated by at least 256 extra bytes, as the rasterizer 
    // may read beyond `count` due to vector alignment.
    VertexReader(const uint8_t* vertexBuffer, const uint8_t* indexBuffer, uint32_t count, enum IndexFormat ixf) {
        VertexBuffer = vertexBuffer;
        IndexBuffer = indexBuffer;
        Count = count;
        IndexFormat = ixf;
    }

    uint32_t ReadIndex(size_t offset) {
        if (offset >= Count) return 0;
        if (IndexFormat == U32) return *((uint32_t*)IndexBuffer + offset);
        if (IndexFormat == U16) return *((uint16_t*)IndexBuffer + offset);
        if (IndexFormat == U8) return *((uint8_t*)IndexBuffer + offset);

        assert(!"Unknown index buffer format");
        return 0;
    }

    VInt ReadIndices(size_t offset);
    
    // Reads and de-interleaves indices for 3x16 vertices.
    void ReadTriangleIndices(size_t offset, VInt indices[3]);

    VFloat ReadAttribF(int offset, int stride) const {
        return VFloat::gather((float*)&VertexBuffer[offset], _Indices * stride);
    }
    VInt ReadAttribS32(int offset, int stride) const {
        return VInt::gather((int32_t*)&VertexBuffer[offset], _Indices * stride);
    }

    /// Reads float or normalized integer attributes
    template<typename V, typename A>
    void ReadAttribs(A V::* vertexMember, VFloat* dest, uint32_t count) const {
        size_t offset = (size_t)&(((V*)0)->*vertexMember);

        for (uint32_t i = 0; i < count;) {
            static_assert(sizeof(A) <= 4);

            if (std::is_same<A, float>()) {
                dest[i] = ReadAttribF(offset + i * 4, sizeof(V));
                i++;
            } else {
                // Normalized integer.
                // Since this is generally used with small types, do a single 32-bit gather and unpack bits manually
                VInt data = ReadAttribS32(offset + i * 4, sizeof(V));
                uint32_t elemSize = sizeof(A) * 8;
                bool sign = elemSize != 32 && std::is_signed<A>();

                for (uint32_t pos = 0; pos < 32 && i < count; pos += elemSize, i++) {
                    dest[i] = sign ? UnpackSNorm(data, pos, elemSize) : UnpackUNorm(data, pos, elemSize);
                }
            }
        }
    }

    static VFloat UnpackUNorm(VInt data, uint32_t bitPos, uint32_t bitCount) {
        assert(bitCount != 32);

        int32_t mask = (1 << bitCount) - 1;
        VInt attr = (data >> bitPos) & mask;
        return simd::conv2f(attr) * (1.0f / mask);
    }
    static VFloat UnpackSNorm(VInt data, uint32_t bitPos, uint32_t bitCount) {
        assert(bitCount != 32);

        int32_t scale = (1 << bitCount) / 2 - 1;
        VInt attr = (data << (32 - bitCount - bitPos)) >> (32 - bitCount);
        return simd::conv2f(attr) * (1.0f / scale);
    }
};

struct ShadedVertexPacket {
    static const uint32_t MaxAttribs = 12;

    VFloat4 Position;
    VFloat Attribs[MaxAttribs];
};

struct TrianglePacket {
    VInt MinX, MinY, MaxX, MaxY;
    VInt Weight0, Weight1, Weight2;
    VInt A01, A12, A20;
    VInt B01, B12, B20;
    VFloat RcpArea;

    ShadedVertexPacket Vertices[3];

    // Computes edge variables based on shaded vertices.
    void Setup(int32_t vpWidth, int32_t vpHeight, uint32_t numAttribs);
};

struct VaryingBuffer {
    static const int32_t AttribX = -4, AttribY = -3, AttribZ = -2, AttribW = -1;  //&Position.z == &Attribs[-2];

    const float* Attribs;
    uint32_t TileOffset;
    uint16_t TileMask;

    VFloat W1, W2;
    VFloat Depth;

    // Interpolates vertex attributes using the current barycentric weights.
    VFloat GetSmooth(int32_t attrId) const {
        VFloat v0 = GetFlat(attrId, 0);
        VFloat v1 = GetFlat(attrId, 1);
        VFloat v2 = GetFlat(attrId, 2);
        //return v0 * W0 + v1 * W1 + v2 * W2;
        //return v0 + (v1 - v0) * W1 + (v2 - v0) * W2;
        return simd::fma(v1, W1, simd::fma(v2, W2, v0));
    }
    // Returns the value of the specified vertex attribute, without interpolation.
    // If `vertexId != 0`, returns `attr[vertexId] - attr[0]`.
    VFloat GetFlat(int32_t attrId, uint32_t vertexId = 0) const {
        assert(attrId >= -4 && attrId < (int32_t)ShadedVertexPacket::MaxAttribs);
        assert(vertexId >= 0 && vertexId < 3);

        int32_t idx = attrId * (int32_t)VFloat::Length + (int32_t)(vertexId * (sizeof(ShadedVertexPacket) / 4));
        return Attribs[idx];
    }

    void ApplyPerspectiveCorrection() {
        // https://stackoverflow.com/a/24460895
        VFloat v0 = GetFlat(AttribW, 0);
        VFloat v1 = GetFlat(AttribW, 1);
        VFloat v2 = GetFlat(AttribW, 2);
        VFloat vp = _mm512_rcp14_ps(simd::fma(v1 - v0, W1, simd::fma(v2 - v0, W2, v0)));

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
        float Attribs[(ShadedVertexPacket::MaxAttribs + 4 + VFloat::Length - 1) & ~(VFloat::Length - 1)];
    };
    struct ClipCodes {
        uint16_t AcceptMask;        // Triangles that don't need to be clipped.
        uint16_t NonTrivialMask;    // Triangles that need to be clipped.
        uint8_t OutCodes[16];       // Planes that need to be clipped against (per-triangle).
    };

    uint32_t Count, FreeVtx;
    float GuardBandPlaneDistXY[2]{ 1.0f, 1.0f };
    uint8_t Indices[24];
    Vertex Vertices[24];

    // Compute Cohen-Sutherland clip codes
    ClipCodes ComputeClipCodes(const TrianglePacket& tri);

    void ClipAgainstPlane(Plane plane, uint32_t numAttribs);

    void LoadTriangle(TrianglePacket& srcTri, uint32_t srcTriIdx, uint32_t numAttribs);
    void StoreTriangle(TrianglePacket& destTri, uint32_t destTriIdx, uint32_t srcTriFanIdx, uint32_t numAttribs);
};

template<typename T>
concept ShaderProgram =
    requires(T s, const VertexReader& vertexData, ShadedVertexPacket& vertexPacket, const Framebuffer& fb, VaryingBuffer& vars) {
        { T::NumCustomAttribs } -> std::convertible_to<uint32_t>;
        { s.ShadeVertices(vertexData, vertexPacket) } -> std::same_as<void>;
        { s.ShadePixels(fb, vars) } -> std::same_as<void>;
    };

struct TriangleBatch {
    static const uint32_t MaxSize = 2048 / VFloat::Length;
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

        uint32_t id = (&tri - Triangles) * VFloat::Length + index;
        Bins[x + y * BinsPerRow].push_back(id);
    }
    bool IsFull() { return Count >= MaxSize - 24; } //Reserve 24*vec triangles for clipping
};

class Rasterizer {
    std::shared_ptr<Framebuffer> _fb;
    std::unique_ptr<TriangleBatch> _batch;
    Clipper _clipper;

    struct BinnedTriangle {
        uint32_t X, Y;
        uint16_t TriangleIndex;
        const TrianglePacket* Triangle;
    };
    struct ShaderInterface {
        std::function<void(size_t, ShadedVertexPacket[3])> ReadVtxFn;
        std::function<void(const BinnedTriangle&)> DrawFn;
        uint32_t NumCustomAttribs;
    };

    void Draw(VertexReader& vertexData, const ShaderInterface& shader);

    void SetupTriangles(TriangleBatch& batch, uint32_t numAttribs);
    void BinTriangles(TriangleBatch& batch, TrianglePacket& tris, uint16_t mask, uint32_t numAttribs);

    template<ShaderProgram TShader>
    void DrawBinnedTriangle(const TShader& shader, const BinnedTriangle& bin) {
        Framebuffer& fb = *_fb.get();
        const TrianglePacket& tri = *bin.Triangle;
        uint32_t i = bin.TriangleIndex;

        int32_t centerX = (int32_t)(fb.Width / 2), centerY = (int32_t)(fb.Height / 2);

        uint32_t minX = (uint32_t)(tri.MinX[i] + centerX);
        uint32_t minY = (uint32_t)(tri.MinY[i] + centerY);
        uint32_t maxX = std::min((uint32_t)(tri.MaxX[i] + centerX), bin.X + TriangleBatch::BinSize - 4);
        uint32_t maxY = std::min((uint32_t)(tri.MaxY[i] + centerY), bin.Y + TriangleBatch::BinSize - 4);

        VInt tileOffsX = VInt::ramp() & 3;
        VInt tileOffsY = VInt::ramp() >> 2;

        if (minX < bin.X) {
            tileOffsX += (int32_t)(bin.X - minX);
            minX = bin.X;
        }
        if (minY < bin.Y) {
            tileOffsY += (int32_t)(bin.Y - minY);
            minY = bin.Y;
        }

        // Barycentric coordinates at start of row
        VInt stepX0 = tri.A12[i], stepX1 = tri.A20[i], stepX2 = tri.A01[i];
        VInt stepY0 = tri.B12[i], stepY1 = tri.B20[i], stepY2 = tri.B01[i];

        VInt rowW0 = tri.Weight0[i] + stepX0 * tileOffsX + stepY0 * tileOffsY;
        VInt rowW1 = tri.Weight1[i] + stepX1 * tileOffsX + stepY1 * tileOffsY;
        VInt rowW2 = tri.Weight2[i] + stepX2 * tileOffsX + stepY2 * tileOffsY;

        stepX0 *= 4, stepX1 *= 4, stepX2 *= 4;
        stepY0 *= 4, stepY1 *= 4, stepY2 *= 4;

        float area = tri.RcpArea[i];

        for (uint32_t y = minY; y <= maxY; y += 4) {
            VInt w0 = rowW0, w1 = rowW1, w2 = rowW2;

            for (uint32_t x = minX; x <= maxX; x += 4) {
                uint16_t tileMask = _mm512_cmpge_epi32_mask(w0 | w1 | w2, _mm512_set1_epi32(0));

                if (tileMask != 0) {
                    VaryingBuffer vars = {
                        .Attribs = (float*)&tri.Vertices->Attribs + i,
                        .W1 = simd::conv2f(w1) * area,
                        .W2 = simd::conv2f(w2) * area,
                    };
                    uint32_t tileOffset = fb.GetPixelOffset(x, y);
                    VFloat oldDepth = VFloat::load(&fb.DepthBuffer[tileOffset]);
                    VFloat newDepth = vars.GetSmooth(VaryingBuffer::AttribZ);

                    tileMask &= _mm512_cmp_ps_mask(newDepth, oldDepth, _CMP_LT_OQ);

                    if (tileMask != 0) {
                        vars.Depth = newDepth;
                        vars.TileMask = tileMask;
                        vars.TileOffset = tileOffset;

                        shader.ShadePixels(fb, vars);
                    }
                }
                w0 += stepX0, w1 += stepX1, w2 += stepX2;
            }
            rowW0 += stepY0, rowW1 += stepY1, rowW2 += stepY2;
        }
    }

public:
    Rasterizer(std::shared_ptr<Framebuffer> fb);

    template<ShaderProgram TShader>
    void Draw(VertexReader& vertexData, const TShader& shader) {
        ShaderInterface shifc = {
            .ReadVtxFn =
                [&](size_t offset, ShadedVertexPacket vertices[3]) {
                    VInt indices[3];
                    vertexData.ReadTriangleIndices(offset, indices);

                    for (uint32_t vi = 0; vi < 3; vi++) {
                        vertexData._Indices = indices[vi];
                        shader.ShadeVertices(vertexData, vertices[vi]);
                    }
                },
            .DrawFn = [&](const BinnedTriangle& bt) { DrawBinnedTriangle(shader, bt); },
            .NumCustomAttribs = shader.NumCustomAttribs,
        };
        Draw(vertexData, shifc);
    }
};

};  // namespace swr