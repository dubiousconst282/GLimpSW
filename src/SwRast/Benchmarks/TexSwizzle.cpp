#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include "../Rasterizer.h"
#include "../Texture.h"
#include "../Scene.h"
#include "../Camera.h"

#include <stb_image_write.h>

inline v_int GetOffset_TiledY4(v_int x, v_int y, v_int rowShift) {
    // ternlog prevents clang from duplicating mem load for const(3)
    v_int t1 = _mm512_ternarylogic_epi32(y, y, v_int(3), _MM_TERNLOG_A & ~_MM_TERNLOG_C);
    return (y & 3) | (x << 2) | (t1 << rowShift);
}
inline v_int GetOffset_TiledY8(v_int x, v_int y, v_int rowShift) {
    // ternlog prevents clang from duplicating mem load for const(7)
    v_int t1 = _mm512_ternarylogic_epi32(y, y, v_int(7), _MM_TERNLOG_A & ~_MM_TERNLOG_C);
    return (y & 7) | (x << 3) | (t1 << rowShift);
}
inline v_int GetOffset_Tiled16X4(v_int x, v_int y, v_int rowShift) {
    v_int tileY = _mm512_ternarylogic_epi32(y, y, v_int(15), _MM_TERNLOG_A & ~_MM_TERNLOG_C);
    v_int idx0 = (y & 15) | (x << 4) | (tileY << rowShift);
    // shuffle bits to form 4x4 tiles
    v_int idx1 = _mm512_gf2p8affine_epi64_epi8(idx0, _mm512_set1_epi64((int64_t)0x10'20'01'02'40'80'04'08), 0);
    return _mm512_ternarylogic_epi32(idx0, idx1, v_int(255), (_MM_TERNLOG_A & ~_MM_TERNLOG_C) | (_MM_TERNLOG_B & _MM_TERNLOG_C));
}
inline v_int BitInterleave(v_int x, v_int y) {
    const __m512i bm = _mm512_set_epi8(125, 61, 124, 60, 121, 57, 120, 56, 117, 53, 116, 52, 113, 49, 112, 48, 109, 45, 108, 44, 105, 41,
                                       104, 40, 101, 37, 100, 36, 97, 33, 96, 32, 93, 29, 92, 28, 89, 25, 88, 24, 85, 21, 84, 20, 81, 17,
                                       80, 16, 77, 13, 76, 12, 73, 9, 72, 8, 69, 5, 68, 4, 65, 1, 64, 0);
    const __m512i m0 = _mm512_set1_epi64((int64_t)0x01'10'02'20'04'40'08'80);
    __m512i lo = _mm512_ternarylogic_epi32(x, y << 4, v_int(0x0F'0F'0F'0F), (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
    __m512i hi = _mm512_ternarylogic_epi32(x >> 4, y, v_int(0x0F'0F'0F'0F), (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
    lo = _mm512_gf2p8affine_epi64_epi8(lo, m0, 0);
    hi = _mm512_gf2p8affine_epi64_epi8(hi, m0, 0);
    return _mm512_permutex2var_epi8(lo, bm, hi);
}

template<int SwizzleMode>
v_uint SampleNearestAsIfSwizzled(const swr::Texture2D<swr::pixfmt::RGBA8u>& tex, v_float u, v_float v, v_int layer = 0) {
    // Scale and round UVs
    v_float su = u * tex.ScaleLerpU, sv = v * tex.ScaleLerpV;
    v_int ix = simd::round2i(su), iy = simd::round2i(sv);

    ix &= tex.MaskLerpU;
    iy &= tex.MaskLerpV;

    // Calculate offsets and mip position
    v_int stride = (int)tex.RowShift;
    v_int offset = layer * (int)tex.LayerStride;

    v_float4 grad = { swr::texutil::dFdx(su), swr::texutil::dFdx(sv), swr::texutil::dFdy(su), swr::texutil::dFdy(sv) };
    v_int mipLevel = swr::texutil::CalcMipLevel(grad) - tex.LerpFracBits;
    if (simd::any(mipLevel > 0)) {
        ix >>= mipLevel;
        iy >>= mipLevel;
        stride -= mipLevel;
        offset += _mm512_permutexvar_epi32(mipLevel, _mm512_load_epi32(tex.MipOffsets));
    }

    ix >>= tex.LerpFracBits;
    iy >>= tex.LerpFracBits;

    if constexpr (SwizzleMode == 0) {
        offset += ix + (iy << stride);
    } else if constexpr (SwizzleMode == 1) {
        offset += GetOffset_TiledY4(ix, iy, stride);
    } else if constexpr (SwizzleMode == 2) {
        offset += GetOffset_TiledY8(ix, iy, stride);
    } else if constexpr (SwizzleMode == 3) {
        offset += GetOffset_Tiled16X4(ix, iy, stride);
    } else if constexpr (SwizzleMode == 4) {
        offset += BitInterleave(ix, iy);
    }
    return simd::gather(tex.Data, offset);
}
struct FragmentInfo {
    uint32_t MaterialId;
    uint32_t TileOffset;
    uint32_t UV[16];
};

struct ShadingContext {
    glm::mat4 ProjMat;
    glm::mat3 ModelMat;
    const Meshlet* Meshlets;
    const Material* Materials;

    std::vector<FragmentInfo>* Frags;
};

static glm::vec2 UnpackHalf2x16(uint32_t x) {
    auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
    return { v[0], v[1] };
}

void ShadeMeshlet(const ShadingContext& ctx, uint32_t index, swr::ShadedMeshlet& output) {
    auto& mesh = ctx.Meshlets[index];

    for (uint32_t i = 0; i < mesh.NumVertices; i += simd::vec_width) {
        v_float3 worldPos = { simd::load(&mesh.Positions[0][i]), simd::load(&mesh.Positions[1][i]), simd::load(&mesh.Positions[2][i]) };
        output.SetPosition(i, simd::mul(ctx.ProjMat, { worldPos, 1.0f }));
    }
    output.PrimCount = mesh.NumTriangles;
    memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));
}

void ShadePixels(ShadingContext& ctx, swr::Framebuffer& fb, swr::FragmentVars& vars) {
    // Depth test
    v_float oldDepth = vars.LoadTile(fb.GetDepthBuffer());
    vars.TileMask &= simd::movemask(vars.Depth > oldDepth);
    if (vars.TileMask == 0) return;

    auto& mesh = ctx.Meshlets[vars.MeshletId];
    auto& frag = ctx.Frags->emplace_back(FragmentInfo{ .MaterialId = mesh.MaterialId, .TileOffset = (uint32_t)vars.TileOffset });

    v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[0]]);
    v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[1]]);
    v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[2]]);
    v_float2 uv = vars.Interpolate(uv0, uv1, uv2);
    simd::store(frag.UV, swr::pixfmt::RG16f::Pack({ simd::fract(uv.x), simd::fract(uv.y) }));

    vars.StoreTile(fb.GetDepthBuffer(), vars.Depth);
}

#if 1
// [model path] [(camera) <x> <y> <z> <yaw> <pitch>]
int main(int argc, const char** args) {
    Scene scene;
    scene.ImportGltf(argc > 2 ? args[1] : "assets/models/Sponza/Sponza.gltf");

    auto camera = Camera{ .Position = { -0.46608772755098471, 8.4925445559659511, -1.7022251220187172 }, .Euler = { -3.13643026, -1.4724431 } };

    if (argc >= 6) {
        camera.Position = { atof(args[2]), atof(args[3]), atof(args[4]) };
        camera.Euler = { atof(args[5]), atof(args[6]) };
    }

    auto fb = swr::CreateFramebuffer(1920, 1080);
    auto raster = swr::Rasterizer(1);
    raster.EnableBinning = false;

    camera.Update({ .DeltaTime = 1 / 60.0, .DisplaySize = { fb->Width, fb->Height } }, 0);

    auto projMat = camera.GetProjMatrix() * camera.GetViewMatrix(true);
    auto shaderDispatch = swr::GetDispatchTable<&ShadeMeshlet, &ShadePixels>();

    fb->Clear(0, 0);

    std::vector<FragmentInfo> fragments;

    for (auto& model : scene.Models) {
        for (auto& node : model->Nodes) {
            if (node.MeshCount == 0) continue;

            auto shaderCtx = ShadingContext{
                .ProjMat = projMat * float4x4(node.GlobalTransform),
                .Meshlets = &scene.Meshlets[node.MeshOffset],
                .Frags = &fragments,
            };
            raster.DrawMeshlets(*fb, node.MeshCount, { shaderDispatch, &shaderCtx });
        }
    }
    printf("Fragments: %.3fK\n", fragments.size() / 1e3);

    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.warmup(10).epochs(50);
    ctx.unit("frame").timeUnit(std::chrono::microseconds(1), "us");
    ctx.relative(true);

    auto bench = [&]<int swizzle>() {
        for (auto& frag : fragments) {
            auto tex = scene.Materials[frag.MaterialId].Texture;
            v_float2 uv = swr::pixfmt::RG16f::Unpack(simd::load(frag.UV));

            v_uint color = SampleNearestAsIfSwizzled<swizzle>(*tex, uv.x, uv.y);
            v_uint normalMR = SampleNearestAsIfSwizzled<swizzle>(*tex, uv.x, uv.y, 1);
            v_uint finalColor = color ^ (normalMR & 0x1F1F1F1F);

            _mm512_store_epi32(&fb->GetColorBuffer()[frag.TileOffset], finalColor);
        }
    };

    ctx.run("Linear", [&]() { bench.template operator()<0>(); });
    ctx.run("TiledY4", [&]() { bench.template operator()<1>(); });
    ctx.run("TiledY8", [&]() { bench.template operator()<2>(); });
    ctx.run("Tiled16X4", [&]() { bench.template operator()<3>(); });
    ctx.run("ZCurve", [&]() { bench.template operator()<4>(); });

    auto pixels = simd::alloc_buffer<uint32_t>(fb->Width * fb->Height);
    fb->GetPixels(0, pixels.get(), fb->Width);
    stbi_write_png("logs/bench_view.png", (int)fb->Width, (int)fb->Height, 4, pixels.get(), (int)fb->Width * 4);

    return 0;
}

#else

v_float2 Rotate(v_float x, v_float y, float a) {
    x -= 0.5f, y -= 0.5f;
    float s = sinf(a), c = cosf(a);
    return {
        (x*c + y*s) + 0.5f,
        (y*c - x*s) + 0.5f,
    };
}

int main() {
    ankerl::nanobench::Bench ctx;
    ctx.minEpochTime(std::chrono::milliseconds(50));
    ctx.epochs(50);

    auto texture = swr::CreateTexture2D<swr::pixfmt::RGBA8u>(4096, 4096, 1, 1);
    for (uint32_t y = 0; y < texture->Height; y += 4) {
        for (uint32_t x = 0; x < texture->Width; x += 4) {
            texture->WriteTile((simd::lane_idx<v_uint> * 0x505050) | 0xFF000000, x, y);
        }
    }

    ctx.batch(texture->Width * texture->Height * 1e-6).unit("Msample").timeUnit(std::chrono::milliseconds(1), "ms");
    ctx.relative(true);
    
    #pragma nounroll
    for (int i = 0; i < 3; i++) {
        float angle = (i * 0.78539);
        std::string strAngle = std::to_string((int)(angle * 180 / 3.14159 + 0.5));

        auto bench = [&]<int swizzle>() {
            v_float scaleU = 2.0f / texture->Width, scaleV = 2.0f / texture->Height;
            scaleU *= 1.618, scaleV *= 1.618;

            for (int y = 0; y < texture->Height; y += 4) {
                for (int x = 0; x < texture->Width; x += 4) {
                    v_float u = (simd::conv<float>(x + swr::TilePixelOffsetsX) + 0.5f) * scaleU;
                    v_float v = (simd::conv<float>(y + swr::TilePixelOffsetsY) + 0.5f) * scaleV;
                    v_float2 r = Rotate(u, v, angle);
                    auto res = SampleNearestAsIfSwizzled<swizzle, false>(*texture, r.x, r.y);
                    ctx.doNotOptimizeAway(res);
                }
            }
        };
        ctx.run("Linear_" + strAngle, [&]() { bench.template operator()<0>(); });
        ctx.run("TiledY4_" + strAngle, [&]() { bench.template operator()<1>(); });
        ctx.run("TiledY8_" + strAngle, [&]() { bench.template operator()<2>(); });
        ctx.run("Tiled16X4_" + strAngle, [&]() { bench.template operator()<3>(); });
        ctx.run("ZCurve_" + strAngle, [&]() { bench.template operator()<4>(); });

        printf("\n");
    }
}
#endif
/* clang-format off
TigerLake, 2.5GHz, mitigations=off

https://stackoverflow.com/questions/77052435/why-do-mem-load-retired-l1-hit-and-mem-load-retired-l1-miss-not-add-to-the-total

./build/src/SwRast/BenchmarkApp
| relative |            us/frame |             frame/s |    err% |       ins/frame |       cyc/frame |    IPC |  L1 cache refs |L1 miss% |  L3 cache refs |L3 miss% |      bra/frame |   miss% |     total | benchmark
|---------:|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|---------------:|--------:|---------------:|--------:|----------:|:----------
|   100.0% |            9,525.02 |              104.99 |    0.3% |   11,514,595.42 |    9,802,563.33 |  1.175 |  2,629,824.917 | 26.417% |    422,647.083 | 42.022% |     668,999.00 |    0.0% |      2.72 | `Linear`
|   100.7% |            9,462.74 |              105.68 |    0.2% |   15,721,226.35 |   12,726,216.75 |  1.235 |  3,423,157.867 | 22.631% |    526,661.333 | 38.592% |     868,770.20 |    0.0% |      2.73 | `TiledY4`
|   103.4% |            9,213.15 |              108.54 |    0.2% |   16,104,491.92 |   12,628,460.00 |  1.275 |  3,508,137.167 | 26.585% |    531,774.417 | 37.901% |     896,292.58 |    0.0% |      2.77 | `TiledY8`
|    93.6% |           10,181.04 |               98.22 |    0.2% |   16,865,362.25 |   13,933,377.25 |  1.210 |  3,510,857.583 | 24.269% |    534,963.733 | 39.296% |     894,314.40 |    0.0% |      2.76 | `Tiled16X4`
|    90.0% |           10,580.51 |               94.51 |    0.2% |   16,765,339.28 |   14,489,908.33 |  1.157 |  3,335,276.617 | 26.785% |    537,005.067 | 38.048% |     894,412.07 |    0.0% |      2.76 | `ZCurve`

./build/src/SwRast/BenchmarkApp "../GraphicsAssets/Bistro.glb" 3.5428452789783478 19.325049156538867 15.537590757012367 -0.448623389 -0.809218227
| relative |            us/frame |             frame/s |    err% |       ins/frame |       cyc/frame |    IPC |  L1 cache refs |L1 miss% |  L3 cache refs |L3 miss% |      bra/frame |   miss% |     total | benchmark
|---------:|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|---------------:|--------:|---------------:|--------:|----------:|:----------
|   100.0% |           15,256.90 |               65.54 |    0.2% |   19,290,994.17 |   15,601,170.00 |  1.237 |  4,414,547.667 | 22.909% |    530,945.000 | 40.302% |   1,156,888.00 |    0.0% |      2.78 | `Linear`
|   100.5% |           15,173.93 |               65.90 |    0.2% |   20,028,655.46 |   15,643,338.75 |  1.280 |  4,371,897.375 | 18.251% |    527,029.000 | 37.864% |   1,145,584.71 |    0.0% |      2.76 | `TiledY4`
|   102.9% |           14,830.70 |               67.43 |    0.1% |   20,169,827.08 |   15,241,037.50 |  1.323 |  4,404,698.458 | 18.259% |    517,293.250 | 36.852% |   1,155,576.92 |    0.0% |      2.80 | `TiledY8`
|    96.9% |           15,752.35 |               63.48 |    0.3% |   20,996,523.25 |   16,194,809.58 |  1.296 |  4,381,743.750 | 17.720% |    518,704.292 | 36.667% |   1,147,436.25 |    0.0% |      2.74 | `Tiled16X4`
|    90.4% |           16,878.74 |               59.25 |    0.3% |   20,609,301.67 |   17,281,474.17 |  1.193 |  4,118,125.000 | 18.723% |    521,632.500 | 38.376% |   1,137,100.83 |    0.0% |      2.65 | `ZCurve`

./build/src/SwRast/BenchmarkApp "../GraphicsAssets/Bistro.glb" -4.6415226863009593 3.4518891315899394 -0.28760416875593364 -2.35103106 -1.57079637
| relative |            us/frame |             frame/s |    err% |       ins/frame |       cyc/frame |    IPC |  L1 cache refs |L1 miss% |  L3 cache refs |L3 miss% |      bra/frame |   miss% |     total | benchmark
|---------:|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|---------------:|--------:|---------------:|--------:|----------:|:----------
|   100.0% |            7,663.88 |              130.48 |    0.4% |    6,950,779.37 |    7,896,070.00 |  0.880 |  1,561,448.982 | 45.517% |    510,678.795 | 39.582% |     308,445.21 |    0.7% |      2.79 | `Linear`
|   109.6% |            6,994.87 |              142.96 |    0.2% |    7,108,171.15 |    7,130,771.74 |  0.997 |  1,525,567.214 | 30.913% |    412,985.321 | 44.510% |     311,907.38 |    0.7% |      2.72 | `TiledY4`
|   119.5% |            6,413.49 |              155.92 |    0.2% |    7,200,482.11 |    6,486,365.35 |  1.110 |  1,539,443.778 | 36.232% |    461,749.667 | 38.448% |     304,987.78 |    0.7% |      2.75 | `TiledY8`
|   110.4% |            6,942.70 |              144.04 |    0.2% |    7,651,839.90 |    7,107,611.88 |  1.077 |  1,557,348.420 | 35.117% |    448,449.236 | 38.333% |     304,441.44 |    0.7% |      2.74 | `Tiled16X4`
|   104.6% |            7,330.19 |              136.42 |    0.2% |    7,534,736.43 |    7,485,451.07 |  1.007 |  1,450,118.143 | 39.071% |    469,501.688 | 37.637% |     305,428.04 |    0.7% |      2.75 | `ZCurve`

Full scan 4096x4096
| relative | Msample/s |    err% |     ins/Msample |     cyc/Msample |    IPC |  L1 cache refs |L1 miss% |  L3 cache refs |L3 miss% |    bra/Msample |   miss% |     total | benchmark
|---------:|----------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|---------------:|--------:|---------------:|--------:|----------:|:----------
|   100.0% |  1,364.59 |    0.4% |    1,748,501.02 |      747,417.25 |  2.339 |    537,971.377 | 13.642% |     25,957.957 | 35.302% |     215,228.50 |    0.0% |      2.15 | `Linear_0`
|    58.8% |    802.55 |    0.1% |    1,859,308.73 |    1,276,523.07 |  1.457 |    592,791.041 | 50.428% |     78,536.943 | 46.212% |     215,606.54 |    0.0% |      2.85 | `TiledY4_0`
|    49.0% |    668.19 |    0.5% |    1,840,250.03 |    1,518,972.37 |  1.212 |    586,715.877 | 46.439% |     82,984.924 | 73.009% |     213,397.83 |    0.0% |      2.51 | `TiledY8_0`
|    48.5% |    662.45 |    0.5% |    1,913,722.32 |    1,547,492.07 |  1.237 |    646,852.374 | 49.275% |     88,494.182 | 70.775% |     215,667.77 |    0.0% |      2.53 | `Tiled16X4_0`
|    48.0% |    654.92 |    0.2% |    1,935,993.39 |    1,566,101.31 |  1.236 |    591,529.191 | 52.089% |    101,678.655 | 54.890% |     215,148.82 |    0.0% |      2.56 | `ZCurve_0`

|    25.9% |    353.89 |    0.6% |    2,493,615.06 |    2,933,057.70 |  0.850 |    750,749.379 | 36.095% |     69,677.442 | 78.677% |     160,946.10 |    0.0% |      2.37 | `Linear_45`
|    31.1% |    424.05 |    0.2% |    2,609,971.11 |    2,426,808.48 |  1.075 |    807,184.607 | 37.806% |    150,760.114 | 38.758% |     161,507.28 |    0.0% |      2.18 | `TiledY4_45`
|    34.5% |    470.65 |    0.2% |    2,589,752.03 |    2,134,874.98 |  1.213 |    800,930.470 | 37.772% |    148,589.849 | 37.720% |     160,254.21 |    0.0% |      2.85 | `TiledY8_45`
|    32.8% |    448.10 |    0.3% |    2,664,828.84 |    2,279,368.04 |  1.169 |    861,328.691 | 35.643% |    144,102.409 | 47.297% |     161,569.43 |    0.0% |      2.55 | `Tiled16X4_45`
|    29.9% |    407.36 |    0.6% |    3,515,571.33 |    3,268,979.04 |  1.075 |  1,054,646.909 | 37.409% |    184,596.539 | 57.631% |     211,021.27 |    0.0% |      2.06 | `ZCurve_45`

|    17.7% |    241.10 |    0.6% |    3,345,175.53 |    5,650,185.05 |  0.592 |  1,007,133.365 | 36.902% |    331,653.595 | 62.763% |     215,914.91 |    0.0% |      3.48 | `Linear_90`
|    24.4% |    332.89 |    0.3% |    3,460,994.42 |    4,073,249.25 |  0.850 |  1,070,382.655 | 38.012% |    554,638.118 | 23.542% |     214,172.45 |    0.0% |      2.53 | `TiledY4_90`
|    29.2% |    398.59 |    0.4% |    3,482,436.48 |    3,399,358.99 |  1.024 |  1,077,011.794 | 25.706% |    412,368.655 | 25.250% |     215,495.97 |    0.0% |      2.11 | `TiledY8_90`
|    33.1% |    451.42 |    0.3% |    3,531,282.98 |    3,000,046.31 |  1.177 |  1,141,384.006 | 36.743% |    215,374.410 | 36.240% |     214,102.42 |    0.0% |      2.64 | `Tiled16X4_90`
|    34.4% |    469.11 |    0.6% |    3,607,801.35 |    2,917,919.83 |  1.236 |  1,082,313.687 | 39.085% |    137,659.475 | 60.801% |     216,554.40 |    0.0% |      2.83 | `ZCurve_90`

*/