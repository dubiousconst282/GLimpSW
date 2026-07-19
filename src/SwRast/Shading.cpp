#include "Shading.h"

namespace {

constexpr swr::SamplerDesc SurfaceSampler = {
    .Wrap = swr::WrapMode::Repeat,
    .MagFilter = swr::FilterMode::Linear,
    .MinFilter = swr::FilterMode::Nearest,  // Downsample with nearest for better perf, quality loss is quite subtle.
};
constexpr swr::SamplerDesc EnvSampler = {
    .Wrap = swr::WrapMode::ClampToEdge,
    .MagFilter = swr::FilterMode::Linear,
    .MinFilter = swr::FilterMode::Linear,
};

v_float pow5(v_float x) { return (x * x) * (x * x) * x; }

v_float D_GGX(v_float NoH, v_float roughness) {
    v_float a = NoH * roughness;
    v_float k = roughness * simd::approx_rcp(1.0f - NoH * NoH + a * a);
    return k * k * simd::inv_pi;
}
v_float V_SmithGGXCorrelatedFast(v_float NoV, v_float NoL, v_float roughness) {
    v_float a = 2.0f * NoL * NoV;
    v_float b = NoL + NoV;
    return 0.5f / simd::lerp(a, b, roughness);
}
v_float3 F_Schlick(v_float u, v_float3 f0) {
    v_float f = pow5(1.0 - u);
    return f + f0 * (1.0f - f);
}
v_float Fd_Lambert() { return 1.0f / simd::pi; }

v_float3 Tonemap_Unreal(v_float3 x) {
    // Unreal 3, Documentation: "Color Grading"
    // Adapted to be close to Tonemap_ACES, with similar range
    // Gamma 2.2 correction is baked in, don't use with sRGB conversion!
    return x / (x + 0.155) * 1.019;
}

// https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
v_float DitherRoberts(v_float2 pos) {
    v_float r = simd::fract(pos.x * 0.7548776662 + pos.y * 0.56984029);
    return simd::abs(r * 2.0 - 1.0);
}

float2 UnpackHalf2x16(uint32_t x) {
    auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
    return { v[0], v[1] };
}
v_float3 UnpackNormal(v_uint packed[3], const v_float3& bary) {
    // RGB10: 9141us  OctUnmap: 10568us  (1.15x)

    // Folding snorm scaling after bary lerp saves 6 FMAs
    // v_float3 v0 = swr::pixfmt::RGB10u::Unpack(packed[0]) * 2 - 1;

    v_float3 v0 = simd::conv<float>(packed[0] >> v_uint3(0, 10, 20) & 1023);
    v_float3 v1 = simd::conv<float>(packed[1] >> v_uint3(0, 10, 20) & 1023);
    v_float3 v2 = simd::conv<float>(packed[2] >> v_uint3(0, 10, 20) & 1023);
    return swr::BaryLerp(bary, v0, v1, v2) * (2.0f / 1023) - 1.0f;
}

// Standard "source over" alpha blending. Not gamma correct.
v_uint AlphaBlendU8(v_uint bg, v_uint fg) {
    v_uint t = (fg >> 1) & 0x7F80'0000;
    t |= (t >> 16) | 0x0040'0040;

    v_uint rb = simd::lerp16((bg >> 0 & 0x00FF'00FF), (fg >> 0 & 0x00FF'00FF), t);
    v_uint ag = simd::lerp16((bg >> 8 & 0x00FF'00FF), (fg >> 8 & 0x00FF'00FF), t);
    return rb | (ag << 8);
}

// https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization
v_float3 ColormapTurbo(v_float x) {
    static constexpr float c[] = {
        0.13572138, 4.61539260,  -42.66032258, 132.13108234, -152.94239396, 59.28637943,  //
        0.09140261, 2.19418839,  4.84296658,   -14.18503333, 4.27729857,    2.82956604,   //
        0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912,  27.34824973,  //
    };
    return {
        x * (x * (x * (x * (x * c[5] + c[4]) + c[3]) + c[2]) + c[1]) + c[0],
        x * (x * (x * (x * (x * c[11] + c[10]) + c[9]) + c[8]) + c[7]) + c[6],
        x * (x * (x * (x * (x * c[17] + c[16]) + c[15]) + c[14]) + c[13]) + c[12],
    };
}

// 2D Polyhedral Bounds of a Clipped, Perspective Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013.
// proj = P00, P11, znear
v_int ProjectSphere(v_float3 c, v_float r, float3 proj, v_float4& bb) {
    c.z *= -1;
    v_int mask = c.z >= r + proj.z;
    if (!simd::any(mask)) return 0;

    v_float3 cr = c * r;
    v_float vx = simd::approx_sqrt(c.x * c.x + c.z * c.z - r * r);
    v_float vy = simd::approx_sqrt(c.y * c.y + c.z * c.z - r * r);

    bb.x = (vx * c.x - cr.z) * simd::approx_rcp(vx * c.z + cr.x) * (proj.x * 0.5f) + 0.5f;
    bb.y = (vy * c.y + cr.z) * simd::approx_rcp(vy * c.z - cr.y) * (proj.y * 0.5f) + 0.5f;
    bb.z = (vx * c.x + cr.z) * simd::approx_rcp(vx * c.z - cr.x) * (proj.x * 0.5f) + 0.5f;
    bb.w = (vy * c.y - cr.z) * simd::approx_rcp(vy * c.z + cr.y) * (proj.y * 0.5f) + 0.5f;

    return mask;
}
bool ProjectSphere(float3 c, float r, float3 proj, float4& bb) {
    c.z *= -1;
    if (c.z < r + proj.z) return false;

    float3 cr = c * r;
    float vx = glm::sqrt(c.x * c.x + c.z * c.z - r * r);
    float vy = glm::sqrt(c.y * c.y + c.z * c.z - r * r);

    bb.x = (vx * c.x - cr.z) / (vx * c.z + cr.x) * (proj.x * 0.5f) + 0.5f;
    bb.y = (vy * c.y + cr.z) / (vy * c.z - cr.y) * (proj.y * 0.5f) + 0.5f;
    bb.z = (vx * c.x + cr.z) / (vx * c.z - cr.x) * (proj.x * 0.5f) + 0.5f;
    bb.w = (vy * c.y - cr.z) / (vy * c.z + cr.y) * (proj.y * 0.5f) + 0.5f;

    return true;
}

// https://momentsingraphics.de/CameraRays.html
float3x3 GetCameraRayMatrix(const float4x4& worldToClip) {
    glm::mat4x3 M = glm::transpose(worldToClip);
    return { glm::cross(M[1], M[3]), glm::cross(M[3], M[0]), glm::cross(M[0], M[1]) };
}
v_float3 GetCameraRayDir(v_float2 screenNdc, const float3x3& m) {
    // Must use precise sqrt to avoid noticeable distortions.
    return simd::normalize_precise(v_float3(screenNdc.x) * m[0] + v_float3(screenNdc.y) * m[1] + m[2]);
}

// Point in sphere for given UV in range [0..1)
v_float3 SampleUniformSphere(v_float2 u) {
    v_float cosTheta = u.x * 2 - 1;
    v_float sinTheta = simd::approx_sqrt(1.000001f - cosTheta * cosTheta);
    return v_float3(simd::sincos_2pi(u.y) * sinTheta, cosTheta);
}
// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
float3x3 BuildOrthonormalBasis(float3 n) {
    float sign = n.z < 0.0f ? -1.0f : 1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    float3 b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    float3 b2 = float3(b, sign + n.y * n.y * a, -n.y);
    return float3x3(b1, b2, n);
}

void ShadeMeshlet(const ShadingContext& ctx, uint32_t index, swr::ShadedMeshlet& output) {
    if (ctx.InstanceCullBitmap != nullptr) {
        v_mask mask = ctx.InstanceCullBitmap[index / simd::vec_width];

        if ((mask >> (index % simd::vec_width) & 1) == 0) {
            output.PrimCount = 0;
            return;
        }
    }
    const Meshlet& mesh = ctx.InstanceMeshlets[index];

    // TODO: meshlet compression, the mem loads take longest here (>65%)
    // https://liamtyler.github.io/posts/meshlet_compression/
    // https://gpuopen.com/learn/mesh_shaders/mesh_shaders-meshlet_compression/
    for (uint32_t i = 0; i < mesh.NumVertices; i += simd::vec_width) {
        v_float3 pos = { simd::load(&mesh.Positions[0][i]), simd::load(&mesh.Positions[1][i]), simd::load(&mesh.Positions[2][i]) };
        output.SetPosition(i, simd::mul(ctx.ObjectToClipMat, v_float4(pos, 1.0f)));
    }
    output.PrimCount = mesh.NumTriangles;
    memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));

    if (mesh.MaterialId != UINT_MAX) {
        const Material& material = ctx.Materials[mesh.MaterialId];
        output.CullMode = material.IsDoubleSided ? swr::FaceCullMode::None : swr::FaceCullMode::FrontCCW;
        output.FragmentShaderId = material.AlphaCutoff < 255 ? 1 : 0;
    }
}

template<bool EnableAlphaTest, bool IsShadowRender>
void FS_EncodeSurfaceId(const ShadingContext& ctx, swr::FragmentVars& vars) {
    // Depth test
    v_float oldDepth = vars.LoadTile(ctx.DepthBuffer);
    vars.TileMask &= simd::movemask(vars.Depth > oldDepth);
    if (vars.TileMask == 0) return;

    if constexpr (EnableAlphaTest) {
        const Meshlet& mesh = ctx.InstanceMeshlets[vars.MeshletId];
        const Material& material = ctx.Materials[mesh.MaterialId];

        v_float2 uv0 = UnpackHalf2x16(mesh.Attribs[vars.VertexId[0]].x);
        v_float2 uv1 = UnpackHalf2x16(mesh.Attribs[vars.VertexId[1]].x);
        v_float2 uv2 = UnpackHalf2x16(mesh.Attribs[vars.VertexId[2]].x);
        v_float2 uv = vars.Interpolate(uv0, uv1, uv2);
        v_uint color = material.Texture->SampleImplicitLod<SurfaceSampler>(uv);

        vars.TileMask &= simd::movemask(color >= uint32_t(material.AlphaCutoff << 24));
    }
    if constexpr (IsShadowRender) {
        // Planar bias method, as described by Avelina9X
        // https://www.reddit.com/r/GraphicsProgramming/comments/1oyeer3/bias_free_shadow_mapping_removing_shadow
        v_float dx = swr::texutil::dFdx(vars.Depth);
        v_float dy = swr::texutil::dFdy(vars.Depth);
        vars.Depth -= simd::max(simd::abs(dx) + simd::abs(dy), 5e-6) * ctx.ShadowSampleRadius;
    }
    vars.StoreTile(ctx.DepthBuffer, vars.Depth);

    if constexpr (!IsShadowRender) {
        v_uint surfaceId = vars.MeshletId * swr::ShadedMeshlet::MaxPrims + vars.PrimId;
        surfaceId |= ctx.InstanceId << ShadingContext::VisbDrawShift;

        vars.StoreTile(ctx.ColorBuffer, surfaceId);
    }
}

void FS_Overdraw(const ShadingContext& ctx, swr::FragmentVars& vars) {
    v_uint counter = vars.LoadTile(ctx.ColorBuffer);

    v_int isActive = _mm512_movm_epi32(vars.TileMask);
    counter = _mm512_adds_epu16(counter, isActive ? v_uint(0x0001'0000) : v_uint(0x0000'0001));

    vars.TileMask = 0xFFFF;
    vars.StoreTile(ctx.ColorBuffer, counter);
    vars.StoreTile(ctx.DepthBuffer, simd::max(vars.Depth, vars.LoadTile(ctx.DepthBuffer)));
}

// http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
void IntersectTriangle(v_float4 v0, v_float4 v1, v_float4 v2, v_float2 screenNdc, float2 screenSize,  //
                       v_float3& bary, v_float3& ddx, v_float3& ddy) {
    v_float3 invW = 1.0f / v_float3(v0.w, v1.w, v2.w);

    v_float2 p0 = v_float2(v0) * invW.x;
    v_float2 p1 = v_float2(v1) * invW.y;
    v_float2 p2 = v_float2(v2) * invW.z;

    v_float2 m0 = (p2 - p1), m1 = (p0 - p1);
    v_float invDet = 1.0f / (m0.x * m1.y - m1.x * m0.y);

    ddx = v_float3(p1.y - p2.y, p2.y - p0.y, p0.y - p1.y) * (invDet * invW);
    ddy = v_float3(p2.x - p1.x, p0.x - p2.x, p1.x - p0.x) * (invDet * invW);
    v_float ddxSum = ddx.x + ddx.y + ddx.z;
    v_float ddySum = ddy.x + ddy.y + ddy.z;

    v_float2 rel0 = screenNdc - p0;
    v_float interpInvW = invW.x + rel0.x * ddxSum + rel0.y * ddySum;
    v_float interpW = 1.0f / interpInvW;

    bary.y = interpW * (rel0.x * ddx.y + rel0.y * ddy.y);
    bary.z = interpW * (rel0.x * ddx.z + rel0.y * ddy.z);
    bary.x = 1 - bary.y - bary.z;

    // TODO: seems like this could be simplified a bit, see DAIS paper
#if 0
    ddx = v_float3(p1.y - p2.y, p2.y - p0.y, p0.y - p1.y) * invDet;
    ddy = v_float3(p2.x - p1.x, p0.x - p2.x, p1.x - p0.x) * invDet;

    v_float u = ddx.x * (screenUV.x - p2.x) + ddy.x * (screenUV.y - p2.y);
    v_float v = ddx.y * (screenUV.x - p2.x) + ddy.y * (screenUV.y - p2.y);

    v_float invW = 1.0f / simd::fma(u, p0.w, simd::fma(v, p1.w, p2.w * (1 - u - v)));
    u *= p0.w * invW;
    v *= p1.w * invW;
#endif

    ddx *= (2.0f / screenSize.x);
    ddy *= -(2.0f / screenSize.y);
    ddxSum *= (2.0f / screenSize.x);
    ddySum *= -(2.0f / screenSize.y);

    v_float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    v_float interpW_ddy = 1.0f / (interpInvW + ddySum);

    ddx = interpW_ddx * (bary * interpInvW + ddx) - bary;
    ddy = interpW_ddy * (bary * interpInvW + ddy) - bary;
}
struct SurfaceAttribs {
    v_uint Albedo;
    v_float3 Normal;
    v_float Metallic, Roughness;
    v_uint PackedNMR;
};
[[clang::always_inline]]
SurfaceAttribs ResolveSurface(const ShadingContext& ctx, v_uint surfaceId, v_float2 screenNdc, float2 screenSize) {
    v_float3 pos[3];
    v_uint packedTexCoord[3];
    v_uint packedNormal[3];
    v_uint packedTangent[3];
    v_uint materialId = UINT_MAX;

    v_int v_exec = surfaceId != 0;
    if (!simd::any(v_exec)) return {};

    simd_foreach_unique_lane(i, surfaceId, true) {
        const InstanceData& instance = ctx.Instances[i >> ShadingContext::VisbDrawShift];
        const Meshlet& mesh = instance.Meshlets[i / swr::ShadedMeshlet::MaxPrims % ShadingContext::MaxMeshletsPerDraw];
        uint32_t triangleId = i % swr::ShadedMeshlet::MaxPrims;

        simd::cmov(materialId, mesh.MaterialId, v_exec);

        #pragma unroll
        for (uint32_t vi = 0; vi < 3; vi++) {
            uint32_t idx = mesh.Indices[vi][triangleId];

            simd::cmov(pos[vi].x, mesh.Positions[0][idx], v_exec);
            simd::cmov(pos[vi].y, mesh.Positions[1][idx], v_exec);
            simd::cmov(pos[vi].z, mesh.Positions[2][idx], v_exec);
            simd::cmov(packedTexCoord[vi], mesh.Attribs[idx].x, v_exec);
            simd::cmov(packedNormal[vi], mesh.Attribs[idx].y, v_exec);
            simd::cmov(packedTangent[vi], mesh.Attribs[idx].z, v_exec);
        }
    }

    v_float transform[4][3];
    v_uint instanceId = surfaceId >> ShadingContext::VisbDrawShift;

    simd_foreach_unique_lane(i, instanceId, true) {
        const InstanceData& instance = ctx.Instances[i];

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            simd::cmov(transform[j][0], instance.Transform[j].x, v_exec);
            simd::cmov(transform[j][1], instance.Transform[j].y, v_exec);
            simd::cmov(transform[j][2], instance.Transform[j].z, v_exec);
        }
    }

    // For 3 vectors, (M1 * (M2 * V)) is slightly cheaper than ((M1 * M2) * V)  (64 vs 54 FLOPs)
    pos[0] = simd::mul(transform, v_float4(pos[0], 1));
    pos[1] = simd::mul(transform, v_float4(pos[1], 1));
    pos[2] = simd::mul(transform, v_float4(pos[2], 1));

    v_float4 ndc0 = simd::mul(ctx.WorldToClipMat, v_float4(pos[0], 1));
    v_float4 ndc1 = simd::mul(ctx.WorldToClipMat, v_float4(pos[1], 1));
    v_float4 ndc2 = simd::mul(ctx.WorldToClipMat, v_float4(pos[2], 1));

    v_float3 bary, ddx, ddy;
    IntersectTriangle(ndc0, ndc1, ndc2, screenNdc, screenSize, bary, ddx, ddy);

    v_float2 texCoord0 = swr::pixfmt::RG16f::Unpack(packedTexCoord[0]);
    v_float2 texCoord10 = swr::pixfmt::RG16f::Unpack(packedTexCoord[1]) - texCoord0;
    v_float2 texCoord20 = swr::pixfmt::RG16f::Unpack(packedTexCoord[2]) - texCoord0;

    v_float2 texCoord = texCoord0 + texCoord10 * bary.y + texCoord20 * bary.z;
    v_float4 texGrad = {
        texCoord10.x * ddx.y + texCoord20.x * ddx.z,
        texCoord10.y * ddx.y + texCoord20.y * ddx.z,
        texCoord10.x * ddy.y + texCoord20.x * ddy.z,
        texCoord10.y * ddy.y + texCoord20.y * ddy.z,
    };

    v_uint packedAlbedo = 0;
    v_uint packedNMR = 0xFF00'8080; // 100% roughness, norm(0,0,+1)

    simd_foreach_unique_lane(i, materialId, materialId != UINT_MAX) {
        // TODO: should probably add access mask to texture sampling, since UVs might vary wildly across different tris/lanes
        const swr::RgbaTexture2D& tex = *ctx.Materials[i].Texture;
        v_int mipLevel = swr::texutil::CalcMipLevel(texGrad, v_float2(tex.ScaleU, tex.ScaleV));
        simd::cmov(packedAlbedo, tex.SampleLevel<SurfaceSampler>(texCoord, 0, mipLevel), v_exec);

        if (tex.NumLayers >= 2) {
            simd::cmov(packedNMR, tex.SampleLevel<SurfaceSampler>(texCoord, 1, mipLevel), v_exec);
        }
    }

    v_float3 normalWS = simd::normalize(simd::mul(transform, v_float4(UnpackNormal(packedNormal, bary), 0)));
    v_float3 tangentWS = simd::normalize(simd::mul(transform, v_float4(UnpackNormal(packedTangent, bary), 0)));
    v_float3 bitangentWS = simd::cross(normalWS, tangentWS);
    simd::cmov(bitangentWS, -bitangentWS, v_int(packedTangent[0]) < 0);

    // Reconstruct Z from 2-channel normap map https://aras-p.info/texts/CompactNormalStorage.html
    v_float3 normalTS = v_float3(simd::conv<float>(packedNMR >> v_uint2(0, 8) & 255) * (1.0f / 127.5f) - 1.0f, 0);
    normalTS.z = simd::approx_sqrt(1.0f - (normalTS.x * normalTS.x + normalTS.y * normalTS.y));

    // simd::normalize(simd::mul(normalTS, float3x3(tangentWS, bitangentWS, normalWS)));
    v_float3 normalRemapped = simd::normalize({
        normalTS.x * tangentWS.x + normalTS.y * bitangentWS.x + normalTS.z * normalWS.x,
        normalTS.x * tangentWS.y + normalTS.y * bitangentWS.y + normalTS.z * normalWS.y,
        normalTS.x * tangentWS.z + normalTS.y * bitangentWS.z + normalTS.z * normalWS.z,
    });

    return {
        .Albedo = packedAlbedo,
        .Normal = normalRemapped,
        .Metallic = simd::conv<float>(packedNMR >> 16 & 255) * (1.0f / 255),
        .Roughness = simd::conv<float>(packedNMR >> 24 & 255) * (1.0f / 255),
        .PackedNMR = packedNMR,
    };
}

simd::vec<uint16_t> s_BlueNoise[64][32][32];

v_float2 GetTemporalBlueNoise(uint3 pos, uint32_t sampleIdx) {
    // R2 sequence
    pos.x += (sampleIdx * 3242174888 + 2147483647) >> 25;
    pos.y += (sampleIdx * 2447445413 + 2147483647) >> 25;
    v_int q = simd::conv<int>(s_BlueNoise[pos.z & 63][pos.x >> 2 & 31][pos.y >> 2 & 31]);
    return simd::conv<float>(q >> v_int2(0, 8) & 255) / 255.0f;
}

float2 Hammersley(uint32_t i, uint32_t numSamples) {
    return { i / float(numSamples), __builtin_elementwise_bitreverse(i) * float(1.0 / UINT_MAX) };
}

[[clang::always_inline]]
v_float SampleShadowMap(const ShadingContext& ctx, v_float3 worldPos, uint3 noisePos, uint32_t numSamples) {
    auto* shadowMap = ctx.ShadowDepthMap;
    v_float4 mappedPos = simd::mul(ctx.ShadowProjMatrices[0], v_float4(worldPos, 1));
    mappedPos /= mappedPos.w;

    v_int visibleCount = 0;

    float2 uvScale = float2(shadowMap->ScaleU, shadowMap->ScaleV) * 0.5f;
    v_float2 uv = v_float2(mappedPos) * uvScale + uvScale;

    float jitterRadius = ctx.ShadowSampleRadius - 1;
    float jitterBias = jitterRadius * 0.5f;
    v_float2 noise = GetTemporalBlueNoise(noisePos, 0);

    for (uint32_t i = 0; i < numSamples; i++) {
        v_float2 jitter = Hammersley(i, numSamples);
        jitter.x = simd::fract(jitter.x + noise.x);
        jitter.y = simd::fract(jitter.y + noise.y);

        v_int2 pos = simd::conv<int>(uv + jitter * jitterRadius - jitterBias);
        v_int inbounds = v_uint(pos.x) < shadowMap->Width && v_uint(pos.y) < shadowMap->Height;

        v_float depth = simd::as<v_float>(shadowMap->Fetch(pos, 0, 0, inbounds));
        visibleCount += depth < mappedPos.z || !inbounds ? 1 : 0;
    }
    return simd::conv<float>(visibleCount) / numSamples;
}

v_float GetLightAttenuation(const Light& light, v_float3 surfacePosWS) {
    if (light.Type == Light::kTypeDirectional) return 1.0;

    // from Filament, equivalent to GLTF spec: `attenuation = max( min( 1.0 - ( dist / range )^4, 1 ), 0 ) / dist^2`
    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md
    v_float3 posToLight = light.Position - surfacePosWS;
    v_float distanceSquare = simd::dot(posToLight, posToLight);
    v_float factor = distanceSquare * light.InvRadiusSq;
    v_float smoothFactor = simd::max(1.0 - factor * factor, 0.0);
    v_float attenuation = (smoothFactor * smoothFactor) * simd::approx_rcp(simd::max(distanceSquare, 1e-4));

    if (light.Type == Light::kTypeSpot) {
        // float spotScale = 1.0 / max(cos(SpotInnerAngle) - cos(SpotOuterAngle), 1e-4);
        // float spotOffset = -cos(SpotOuterAngle) * spotScale;
        v_float cd = simd::dot(-light.Direction, simd::normalize(posToLight));
        v_float spotAttenuation = simd::clamp(cd * light.SpotScale + light.SpotOffset, 0.0f, 1.0f);
        attenuation *= spotAttenuation * spotAttenuation;
    }
    return attenuation;
}

[[clang::always_inline]]
v_float3 EvalLighting(const ShadingContext& ctx, const SurfaceAttribs& surf, v_float3 worldPos, v_float3 irradiance, uint3 noisePos,
                      uint32_t numShadowSamples) {
    v_float3 finalColor = 0;
    v_float reflectance = 0.5;  // parametized IOR

    v_float3 baseColor = v_float3(swr::pixfmt::RGBA8u::UnpackSrgb(surf.Albedo));
    v_float alphaRoughness = simd::max(surf.Roughness * surf.Roughness, 1e-4f);
    v_float3 f0 = {
        simd::lerp(0.16f * reflectance * reflectance, baseColor.x, surf.Metallic),
        simd::lerp(0.16f * reflectance * reflectance, baseColor.y, surf.Metallic),
        simd::lerp(0.16f * reflectance * reflectance, baseColor.z, surf.Metallic),
    };
    v_float3 diffuseColor = baseColor * (1.0 - surf.Metallic);

    v_float3 viewDir = simd::normalize(ctx.ViewPos - worldPos);
    v_float NoV = simd::abs(simd::dot(surf.Normal, viewDir)) + 1e-5f;

    // TODO: clustered lights?
    // - https://advances.realtimerendering.com/s2025/content/s2025_stb_lighting_v1.1_notes.pdf

    for (uint32_t i = 0; i < ctx.NumLights; i++) {
        const Light& light = ctx.Lights[i];
        v_float3 lightDir = light.Type == Light::kTypeDirectional ? -light.Direction : simd::normalize(light.Position - worldPos);

        v_float NoL = simd::dot(surf.Normal, lightDir);
        if (simd::all(NoL < 1e-4)) continue;

        if (i == 0 && ctx.ShadowDepthMap != nullptr) {
            NoL *= SampleShadowMap(ctx, worldPos, noisePos, numShadowSamples);
        }

        v_float attenuation = GetLightAttenuation(light, worldPos) * light.Intensity;
        if (simd::all(NoL * attenuation < 1e-4)) continue;

        v_float3 halfwayDir = simd::normalize(viewDir + lightDir);
        v_float NoH = simd::clamp(simd::dot(surf.Normal, halfwayDir), 0, 1);
        v_float LoH = simd::clamp(simd::dot(lightDir, halfwayDir), 0, 1);

        // BRDF
        v_float D = D_GGX(NoH, alphaRoughness);
        v_float V = V_SmithGGXCorrelatedFast(NoV, NoL, alphaRoughness);
        v_float3 F = F_Schlick(LoH, f0);
        v_float3 Fr = (D * V) * F;
        v_float3 Fd = diffuseColor * simd::inv_pi;  // Lambert

        finalColor += (Fd + Fr) * light.Color * simd::max(NoL * attenuation, 0.0f);
    }
    finalColor += diffuseColor * irradiance;

    return finalColor;
}

// TODO: Store border data for proper octahedron filtering
static v_float2 SampleProbeDepth(GIProbeField& field, v_int probeIdx, v_float3 dir) {
    v_float2 oct = swr::texutil::MapOctahedron(dir) * 15;
    // TODO: blue noise works, but obviously noisy. Maybe with half-res + blurring.
    // oct += GetTemporalBlueNoise(noisePos, 1);

    v_int2 q = simd::conv<int>(oct);
    v_int idx00 = ((q.x + 0) & 15) | ((q.y + 0) & 15) << 4;
    v_int idx01 = ((q.x + 0) & 15) | ((q.y + 1) & 15) << 4;
    v_int idx10 = ((q.x + 1) & 15) | ((q.y + 0) & 15) << 4;
    v_int idx11 = ((q.x + 1) & 15) | ((q.y + 1) & 15) << 4;

    v_float2 d00 = swr::pixfmt::RG16f::Unpack(simd::gather(field.Depth[0], probeIdx * 256 + idx00));
    v_float2 d10 = swr::pixfmt::RG16f::Unpack(simd::gather(field.Depth[0], probeIdx * 256 + idx10));
    v_float2 d01 = swr::pixfmt::RG16f::Unpack(simd::gather(field.Depth[0], probeIdx * 256 + idx01));
    v_float2 d11 = swr::pixfmt::RG16f::Unpack(simd::gather(field.Depth[0], probeIdx * 256 + idx11));

    v_float2 A = d00 + (d10 - d00) * simd::fract(oct.x);
    v_float2 B = d01 + (d11 - d01) * simd::fract(oct.x);
    return A + (B - A) * simd::fract(oct.y);
}
v_float3 SampleProbeIrradiance(GIProbeField& field, v_int probeIdx, v_float3 dir) {
    v_float2 oct = swr::texutil::MapOctahedron(dir) * 7;

    v_int2 q = simd::conv<int>(oct);
    v_int idx00 = ((q.x + 0) & 7) | ((q.y + 0) & 7) << 3;
    v_int idx01 = ((q.x + 0) & 7) | ((q.y + 1) & 7) << 3;
    v_int idx10 = ((q.x + 1) & 7) | ((q.y + 0) & 7) << 3;
    v_int idx11 = ((q.x + 1) & 7) | ((q.y + 1) & 7) << 3;

    v_float3 d00 = swr::pixfmt::R11G11B10f::Unpack(simd::gather(field.Irradiance[0], probeIdx * 64 + idx00));
    v_float3 d10 = swr::pixfmt::R11G11B10f::Unpack(simd::gather(field.Irradiance[0], probeIdx * 64 + idx10));
    v_float3 d01 = swr::pixfmt::R11G11B10f::Unpack(simd::gather(field.Irradiance[0], probeIdx * 64 + idx01));
    v_float3 d11 = swr::pixfmt::R11G11B10f::Unpack(simd::gather(field.Irradiance[0], probeIdx * 64 + idx11));

    v_float3 A = d00 + (d10 - d00) * simd::fract(oct.x);
    v_float3 B = d01 + (d11 - d01) * simd::fract(oct.x);
    return A + (B - A) * simd::fract(oct.y);
}

[[clang::always_inline]]
v_float3 SampleDiffuse(GIProbeField& field, v_float3 surfacePos, v_float3 surfaceNormal, v_float3 viewDir) {
    v_float3 biasedSurfacePos = surfacePos + (surfaceNormal + viewDir * 3) * field.NormalBias;

    v_float3 baseSamplePos = (biasedSurfacePos - field.ProbeBaseOffset) / field.ProbeSpacing;
    v_int3 baseGridPos = { simd::floor2i(baseSamplePos.x), simd::floor2i(baseSamplePos.y), simd::floor2i(baseSamplePos.z) };
    v_float3 cageOffset = { simd::fract(baseSamplePos.x), simd::fract(baseSamplePos.y), simd::fract(baseSamplePos.z) };

    v_float3 sumIrradiance = 0;
    v_float sumWeight = 0;

    int3 gridSize = field.GridSize;

    for (int i = 0; i < 8; i++) {
        v_int3 gridPos = baseGridPos + (i >> v_int3(0, 1, 2) & 1);
        gridPos.x = simd::clamp(gridPos.x, 0, gridSize.x - 1);
        gridPos.y = simd::clamp(gridPos.y, 0, gridSize.y - 1);
        gridPos.z = simd::clamp(gridPos.z, 0, gridSize.z - 1);
        v_int probeIdx = gridPos.x + (gridPos.z * gridSize.x) + (gridPos.y * (gridSize.x * gridSize.z));

        v_float3 probePos;
        simd_foreach_unique_lane(i, probeIdx, true) {
            simd::cmov(probePos, field.Positions[i], v_exec);

            uint64_t& activeMask = field.CurrActiveBitmap[(uint32_t)i / 64u];
            uint64_t maskBit = 1ull << (i & 63);
            if ((activeMask & maskBit) == 0) __atomic_fetch_or(&activeMask, maskBit, __ATOMIC_RELAXED);
        }

        v_float weight = 1.0f;

        // Backface weight
        {
            v_float3 unbiasedDir = simd::normalize(probePos - surfacePos);
            weight *= simd::max(simd::dot(surfaceNormal, unbiasedDir), 0.001);

            // Original softening appears to cause pinch artifacts in some cases.
            // v_float NdotP = simd::max(0.0001, simd::dot(surfaceNormal, unbiasedDir) * 0.5 + 0.5);
            // weight *= (NdotP * NdotP);
        }

        // Chebyshev visibility
        {
            v_float2 moments = SampleProbeDepth(field, probeIdx, biasedSurfacePos - probePos);
            v_float distToProbe = simd::length(biasedSurfacePos - probePos);

            v_float variance = simd::abs(moments.x * moments.x - moments.y);
            v_float diff = simd::max(distToProbe - moments.x, 0);
            v_float visibility = variance * simd::approx_rcp(variance + diff * diff);

            visibility = simd::max(visibility * visibility * visibility, 0.0);  // increase contrast

            weight *= visibility;
        }

        // Trilinear weights
        {
            weight *= (i & 1) ? cageOffset.x : 1 - cageOffset.x;
            weight *= (i & 2) ? cageOffset.y : 1 - cageOffset.y;
            weight *= (i & 4) ? cageOffset.z : 1 - cageOffset.z;
        }

        sumIrradiance += SampleProbeIrradiance(field, probeIdx, surfaceNormal) * weight;
        sumWeight += weight;
    }
    return sumIrradiance / simd::max(sumWeight, 1e-5);
}

SurfaceAttribs UnpackVoxelAttribs(v_uint packed) {
    v_uint3 color = packed >> v_uint3(0, 5, 10) & v_uint3(31, 31, 15);
    v_float2 octNormal = simd::conv<float>(packed >> v_uint2(14, 19) & 31) / 31.0f;
    v_float2 mr = simd::conv<float>(packed >> v_uint2(24, 28) & 15) / 15.0f;

    return {
        .Albedo = color.x << 3 | color.y << 11 | color.z << 20,
        .Normal = swr::texutil::UnmapOctahedron(octNormal),
        .Metallic = mr.x,
        .Roughness = mr.y,
    };
}

};  // namespace

// clang-format off
const swr::ShaderDispatchTable
    ShadingContext::VisBufferShader = swr::GetDispatchTable<&ShadeMeshlet,
                                                            &FS_EncodeSurfaceId<0, 0>,
                                                            &FS_EncodeSurfaceId<1, 0>
                                                        >(),
    ShadingContext::ShadowGenShader = swr::GetDispatchTable<&ShadeMeshlet,
                                                            &FS_EncodeSurfaceId<0, 1>,
                                                            &FS_EncodeSurfaceId<1, 1>
                                                        >(),
    ShadingContext::OverdrawShader = swr::GetDispatchTable<&ShadeMeshlet, &FS_Overdraw>();       //

// clang-format on

void ShadingContext::Resolve(swr::Rasterizer& raster) {
    assert(Instances[0].Meshlets == nullptr);  // 0th instance id must be reserved for "null"

    uint2 fbSize = uint2(ColorBuffer->Width, ColorBuffer->Height);
    float4x4 invProj = glm::inverse(WorldToClipMat);

    float2 scaleNdc = 2.0f / float2(fbSize);
    float2 centerNdc = 0.5f * scaleNdc - 1.0f;

    constexpr uint32_t GroupSize = 64;

    raster.DispatchGroups(fbSize, GroupSize, [&](uint4 groupRect) {
        v_float lumaSum = 1e-7;

        v_uint3 surfaceData[(GroupSize * GroupSize) / simd::vec_width];
        uint32_t surfaceDataOffset = 0;

        for (uint32_t y = groupRect.y; y < groupRect.w; y += 4) {
            uint32_t tileOffset = ColorBuffer->GetTexelOffset(groupRect.x, y, 0);

            for (uint32_t x = groupRect.x; x < groupRect.z; x += 4, tileOffset += 16) {
                v_int2 screenPos = int2(x, y) + swr::TilePixelOffsets;
                v_float2 screenNdc = simd::conv<float>(screenPos) * scaleNdc + centerNdc;

                // TODO: https://filmicworlds.com/blog/decoupled-visibility-multisampling/
                // TODO: could we also do some sort of VRS here?
                v_uint surfaceId = simd::load(&ColorBuffer->Data[tileOffset]);
                SurfaceAttribs surf = ResolveSurface(*this, surfaceId, screenNdc, fbSize);

                v_int3 qnorm = simd::conv<int>(surf.Normal * 511.5 + 511.5);
                surfaceData[surfaceDataOffset++] = {
                    surf.Albedo, surf.PackedNMR,
                    v_uint(qnorm.x | qnorm.y << 10 | qnorm.z << 20),
                };
            }
        }

        // TODO: half-res shadow + probe sample (+ SSAO/GTAO?)

        surfaceDataOffset = 0;
        for (uint32_t y = groupRect.y; y < groupRect.w; y += 4) {
            uint32_t tileOffset = ColorBuffer->GetTexelOffset(groupRect.x, y, 0);

            for (uint32_t x = groupRect.x; x < groupRect.z; x += 4, tileOffset += 16) {
                v_float depth = simd::load<v_float>(&DepthBuffer->Data[tileOffset]);
                v_int skyMask = depth <= 0.0f;

                v_int2 screenPos = int2(x, y) + swr::TilePixelOffsets;
                v_float2 screenNdc = simd::conv<float>(screenPos) * scaleNdc + centerNdc;
                v_float3 worldPos = v_float3(simd::perspective_div(simd::mul(invProj, v_float4(screenNdc, skyMask ? 1.0f : depth, 1.0f))));
                v_float3 finalColor = 0;

                v_uint3 packedSurf = surfaceData[surfaceDataOffset++];

                if (simd::any(!skyMask)) {
                    SurfaceAttribs surf = {
                        .Albedo = packedSurf.x,
                        .Normal = simd::normalize(swr::pixfmt::RGB10u::Unpack(packedSurf.z) * 2.0f - 1.0f),
                        .Metallic = simd::conv<float>(packedSurf.y >> 16 & 255) / 255.0f,
                        .Roughness = simd::conv<float>(packedSurf.y >> 24 & 255) / 255.0f,
                    };
                    v_float3 irradiance = Lights[0].Intensity * 0.01f;
                    if (ProbeField != nullptr) {
                        irradiance = SampleDiffuse(*ProbeField, worldPos, surf.Normal, simd::normalize(ViewPos - worldPos));
                    }
                    finalColor = EvalLighting(*this, surf, worldPos, irradiance, int3(x, y, 0), 8);
                }
                if (simd::any(skyMask) && SkyboxTex != nullptr) {
                    v_float3 skyColor = SkyboxTex->SampleOctLevel<EnvSampler>(worldPos - ViewPos, 1);
                    simd::cmov(finalColor, skyColor, skyMask);
                }

                lumaSum += simd::dot(finalColor, v_float3(0.2127, 0.7152, 0.0722));
                finalColor = Tonemap_Unreal(finalColor * Exposure);

                v_uint packedColor = swr::pixfmt::RGBA8u::Pack({ finalColor, 1.0f });
                simd::store(&ColorBuffer->Data[tileOffset], packedColor);
            }
        }
        if (AutoExp != nullptr) {
            uint32_t sampleCount = ((groupRect.z - groupRect.x) * (groupRect.w - groupRect.y)) / simd::vec_width;
            float lumaAvg = _mm512_reduce_add_ps(simd::approx_log2(lumaSum / sampleCount)) / simd::vec_width;

            float normBin = (lumaAvg - AutoExp->HistRange.x) / (AutoExp->HistRange.y - AutoExp->HistRange.x);
            int bin = glm::clamp((int)(normBin * AutoExposureData::NumHistBins), 0, AutoExposureData::NumHistBins - 1);
            __atomic_fetch_add(&AutoExp->Hist[bin], 1, __ATOMIC_RELAXED);
        }
    });

    // Draw lights in a kind of shitty way
    for (uint32_t i = 0; i < NumLights; i++) {
        const Light& light = Lights[i];
        if (light.Type == Light::kTypeDirectional) continue;

        float4 clipPos = WorldToClipMat * float4(light.Position, 1);
        if (glm::max(glm::max(glm::abs(clipPos.x), glm::abs(clipPos.y)), glm::abs(clipPos.z)) > clipPos.w) continue;

        float depth = clipPos.z / clipPos.w;
        float2 screenPos = float2(clipPos) / clipPos.w;

        screenPos = screenPos * 0.5f + 0.5f;
        float radius = (glm::max(fbSize.x, fbSize.y) / 30.0f) / clipPos.w;

        float2 centerPos = screenPos * float2(fbSize);
        int2 startTilePos = glm::max(int2(centerPos - radius), 0) & ~3;
        int2 endTilePos = glm::min(int2(centerPos + radius), int2(fbSize));

        for (int y = startTilePos.y; y < endTilePos.y; y += 4) {
            uint32_t tileOffset = ColorBuffer->GetTexelOffset(uint32_t(startTilePos.x), uint32_t(y), 0);

            for (int x = startTilePos.x; x < endTilePos.x; x += 4, tileOffset += 16) {
                v_int depthMask = depth > simd::load<v_float>(&DepthBuffer->Data[tileOffset]);

                v_int2 tilePos = int2(x, y) + swr::TilePixelOffsets;
                v_float2 relPos = simd::conv<float>(tilePos) + 0.5f - centerPos;
                v_float distSq = simd::dot(relPos, relPos) - radius * radius;

                if (!simd::any(distSq < 0 && depthMask)) continue;

                uint32_t* colorData = &ColorBuffer->Data[tileOffset];
                v_uint bgColor = simd::load(colorData);

                v_float a = 1 - (-distSq / (radius * radius));
                v_uint fgColor = swr::pixfmt::RGBA8u::Pack({ light.Color, 1 - a * a });
                v_uint finalColor = AlphaBlendU8(bgColor, fgColor);

                simd::store(colorData, depthMask ? finalColor : bgColor);
            }
        }
    }
}

void ShadingContext::ResolveDebug(swr::Rasterizer& raster, DebugLayer layer) {
    uint2 fbSize = uint2(ColorBuffer->Width, ColorBuffer->Height);
    float4x4 invProj = glm::inverse(WorldToClipMat);

    raster.DispatchPass(fbSize, [&](uint2 tileBasePos, v_int2 tilePos, v_float2 tileNdc) {
        uint32_t tileOffset = ColorBuffer->GetTexelOffset(tileBasePos.x, tileBasePos.y, 0);
        v_float depth = simd::load<v_float>(&DepthBuffer->Data[tileOffset]);
        v_int skyMask = depth <= 0.0f;

        v_uint tileData = simd::load(&ColorBuffer->Data[tileOffset]);
        SurfaceAttribs surface;

        uint64_t ts1 = __rdtsc();

        if (layer == DebugLayer::BaseColor || layer == DebugLayer::Normals ||                    //
            layer == DebugLayer::MetallicRoughness || layer == DebugLayer::VisResolveHeatmap ||  //
            layer == DebugLayer::ProbeIrradiance                                                 //
        ) {
            surface = ResolveSurface(*this, tileData, tileNdc, fbSize);
        }

        v_float3 finalColor = 0.0f;
        if (layer == DebugLayer::BaseColor) {
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack(surface.Albedo));
        } else if (layer == DebugLayer::Normals) {
            finalColor = surface.Normal * 0.5 + 0.5;
        } else if (layer == DebugLayer::MetallicRoughness) {
            finalColor = v_float3(surface.Metallic, surface.Roughness, 0);
        } else if (layer == DebugLayer::MeshletId) {
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack((tileData / swr::ShadedMeshlet::MaxPrims) * 123456789));
        } else if (layer == DebugLayer::TriangleId) {
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack(tileData * 123456789));
        } else if (layer == DebugLayer::OverdrawPixel) {
            v_float countPix = simd::conv<float>(tileData >> 16);
            finalColor = ColormapTurbo(countPix / 30.0f);
        } else if (layer == DebugLayer::OverdrawQuad) {
            v_float countPix = simd::conv<float>(tileData >> 16);
            v_float countFrag = simd::conv<float>(tileData & 0xFFFF);
            finalColor = ColormapTurbo((countPix + countFrag * 0.5) / 30.0f);
        } else if ((layer == DebugLayer::VoxelGrid || layer == DebugLayer::RaycastHeatmap) && BrickMap != nullptr) {
            v_float3 dir = GetCameraRayDir(tileNdc, GetCameraRayMatrix(WorldToClipMat));

            VoxelHitResult hit = BrickMap->RayCast(ViewPos, dir, v_float2(0, FLT_MAX), 0xFFFF, true);
            SurfaceAttribs attribs = UnpackVoxelAttribs(hit.Attribs);

            // finalColor = EvalLighting(*this, attribs, ViewPos + dir * hit.Dist, 0.03f, int3(tileBasePos, 0), 0);
            // finalColor = Tonemap_Unreal(finalColor);
            skyMask = 0;
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack(attribs.Albedo));
        } else if (layer == DebugLayer::ProbeIrradiance && ProbeField != nullptr) {
            v_float3 worldPos = v_float3(simd::perspective_div(simd::mul(invProj, v_float4(tileNdc, skyMask ? 1.0f : depth, 1.0f))));
            finalColor = SampleDiffuse(*ProbeField, worldPos, surface.Normal, simd::normalize(ViewPos - worldPos));
            finalColor = Tonemap_Unreal(finalColor);
        }

        if (layer == DebugLayer::VisResolveHeatmap || layer == DebugLayer::RaycastHeatmap) {
            uint64_t ts2 = __rdtsc();
            double q = layer == DebugLayer::RaycastHeatmap ? 8000.0 : 4000.0;
            finalColor = ColormapTurbo((ts2 - ts1) / q);
            skyMask = 0;
        }

        uint32_t backgroundRGB = ((tileBasePos.x ^ tileBasePos.y) & 4) ? 0xFF'A0A0A0 : 0xFF'FFFFFF;
        v_uint finalRGB = skyMask ? backgroundRGB : swr::pixfmt::RGBA8u::Pack({ finalColor, 1.0f });

        simd::store(&ColorBuffer->Data[tileOffset], finalRGB);
    });
}

uint32_t ShadingContext::CullMeshlets(v_mask* bitmap, const Meshlet* meshlets, uint32_t count, const float4x4& modelMat,  //
                                      float2 frameSize, swr::Texture2D<swr::pixfmt::R32f>* depthMap) {
    float4 frustumPlanes[6];
    float4x4 objectToPrevViewMat = PrevViewMat * modelMat;
    float4x4 objectToClipMatT = glm::transpose(WorldToClipMat * modelMat);

    // Gribb & Hartmann
    for (int i = 0; i < 3; i++) {
        float4 planeA = objectToClipMatT[3] + objectToClipMatT[i];  // Left,  Bottom, Near
        float4 planeB = objectToClipMatT[3] - objectToClipMatT[i];  // Right, Top,    Far
        planeA /= glm::length(float3(planeA));
        planeB /= glm::length(float3(planeB));
        frustumPlanes[i * 2 + 0] = planeA;
        frustumPlanes[i * 2 + 1] = planeB;
    }

    float objectToWorldScale = glm::length(float3(modelMat[0])); // assuming uniform
    float3 projFactors = float3(ProjMat[0][0], ProjMat[1][1], ProjMat[3][2]);  // f/ar, -f, znear
    uint32_t visibleCount = 0;

    for (uint32_t offset = 0; offset < count; offset += simd::vec_width) {
        v_int visible = (simd::lane_idx<v_uint> + offset) < count;

        // TODO-PERF: split meshlet bound spheres into SoA? 65% of time is spent here, we're wasting 3/4 of cachelines
        v_float4 boundSphere = simd::gather4(&meshlets[offset].BoundCenter.x, simd::lane_idx<v_int> * sizeof(Meshlet) / 4, visible);
        v_float3 boundCenter = v_float3(boundSphere);
        v_float boundRadius = boundSphere.w;

        // Frustum culling
        for (uint32_t i = 0; i < 5; i++) {
            v_float dist = simd::dot(boundCenter, float3(frustumPlanes[i])) + frustumPlanes[i].w;
            visible &= dist > -boundRadius;
        }

        // Occlusion culling
        if (depthMap != nullptr && simd::any(visible)) {
            v_float3 boundCenterVS = v_float3(simd::mul(objectToPrevViewMat, v_float4(boundCenter, 1)));
            v_float boundRadiusVS = boundRadius * objectToWorldScale;

            v_float4 screenBB;
            v_int projMask = visible & ProjectSphere(boundCenterVS, boundRadiusVS, projFactors, screenBB);

            if (simd::any(projMask)) {
                v_float2 bbSize = v_float2(screenBB.z - screenBB.x, screenBB.w - screenBB.y) * frameSize;
                v_int mipLevel = simd::ilog2(simd::max(bbSize.x, bbSize.y));
                mipLevel = simd::clamp(mipLevel, 1, (int)depthMap->MipLevels - 1);

                v_int x0 = simd::max(simd::conv<int>(screenBB.x * frameSize.x), 0) >> mipLevel;
                v_int y0 = simd::max(simd::conv<int>(screenBB.y * frameSize.y), 0) >> mipLevel;
                v_int x1 = simd::min(simd::conv<int>(screenBB.z * frameSize.x), int(frameSize.x) - 1) >> mipLevel;
                v_int y1 = simd::min(simd::conv<int>(screenBB.w * frameSize.y), int(frameSize.y) - 1) >> mipLevel;

                v_float depthSphere = projFactors.z / (-boundCenterVS.z - boundRadiusVS);
                v_float depthVisible = FLT_MAX;

                for (v_int y = y0; simd::any(y <= y1); y += 1) {
                    for (v_int x = x0; simd::any(x <= x1); x += 1) {
                        v_int activeMask = projMask && x <= x1 && y <= y1;
                        v_uint value = depthMap->Fetch(v_int2(x, y), 0, mipLevel - 1, activeMask);
                        simd::cmov(depthVisible, simd::min(depthVisible, simd::as<v_float>(value)), activeMask);
                    }
                }
                visible &= !projMask || depthSphere > depthVisible;

#if 0
                for (uint32_t i = 0; i < 16; i++) {
                    if (!projMask[i]) continue;

                    ImDrawList* dl = ImGui::GetForegroundDrawList();
                    ImVec2 dp = ImGui::GetIO().DisplaySize;
                    dl->AddRect(ImVec2(screenBB.x[i], screenBB.y[i]) * dp, ImVec2(screenBB.z[i], screenBB.w[i]) * dp, 0xFF0000FF);

                    ImVec2 scale = dp / ImVec2(frameSize.x, frameSize.y);
                    int texelSize = (1 << mipLevel[i]);

                    for (int y = y0[i]; y <= y1[i]; y++) {
                        for (int x = x0[i]; x <= x1[i]; x++) {
                            ImVec2 p0 = ImVec2(x << mipLevel[i], y << mipLevel[i]);
                            ImVec2 p1 = p0 + ImVec2(texelSize, texelSize);
                            dl->AddRect(p0 * scale, p1 * scale, 0xFF00FF00);
                        }
                    }
                    char text[128];
                    snprintf(text, sizeof(text), "bb=%.0fx%.0f mip=%d", bbSize.x[i], bbSize.y[i], mipLevel[i]);
                    dl->AddText(ImVec2(screenBB.x[i], screenBB.y[i]) * dp, 0xFFFFFFFF, text);
                }
#endif
            }
        }
        bitmap[offset / simd::vec_width] = simd::movemask(visible);
        visibleCount += simd::popcnt(simd::movemask(visible));
    }
    return visibleCount;
}

void ShadingContext::LoadBlueNoise() {
    FILE* fs = fopen("assets/bluenoise/stbn_unorm8x2_128x128x64.bin", "rb");
    if (!fs) {
        fprintf(stderr, "Missing blue noise texture");
        return;
    }

    for (uint32_t i = 0; i < 64; i++) {
        uint16_t buffer[128 * 128];
        fread(buffer, sizeof(buffer), 1, fs);

        for (uint32_t y = 0; y < 128; y++) {
            for (uint32_t x = 0; x < 128; x++) {
                s_BlueNoise[i][x / 4][y / 4][(x & 3) + (y & 3) * 4] = buffer[x + y * 128];
            }
        }
    }
    fclose(fs);
}

constexpr int ProbeMaxRays = 128;

static v_float GetProbeIrradianceWeight(v_float3 texelDir, float rayDirs[3][ProbeMaxRays], uint32_t j) {
    float3 rayDir = { rayDirs[0][j], rayDirs[1][j], rayDirs[2][j] };
    return simd::max(simd::dot(texelDir, rayDir), 0);
}
static v_float GetProbeDepthWeight(v_float3 texelDir, float rayDirs[3][ProbeMaxRays], uint32_t j) {
    float3 rayDir = { rayDirs[0][j], rayDirs[1][j], rayDirs[2][j] };
    v_float weight = simd::max(simd::dot(texelDir, rayDir), 0);
    return simd::approx_pow(weight, 50);
}

void ShadingContext::UpdateProbes(swr::Rasterizer& raster) {
    uint32_t numRays = ProbeField->RaysPerProbe;
    float maxDist = ProbeField->MaxRayDist;

    assert(numRays < MaxRays);

    // Generate primary rays
    auto rand = simd::XoRandom();
    v_float3 randomDir = SampleUniformSphere(rand.V_NextUFloat2());
    float3x3 randomBasis = BuildOrthonormalBasis({ randomDir.x[0], randomDir.y[0], randomDir.z[0] });

    float rayDirs[3][ProbeMaxRays];
    uint16_t irradWeights[ProbeMaxRays * 64];
    uint16_t depthWeights[ProbeMaxRays * 256];

    for (uint32_t i = 0; i < numRays; i += simd::vec_width) {
        v_float t = simd::conv<float>(i + simd::lane_idx<v_uint>);
        v_float2 uv = v_float2(t / (numRays - 1), t * 0.6180339887498949f);
        v_float3 dir = simd::mul(randomBasis, SampleUniformSphere(uv));
        simd::store(&rayDirs[0][i], dir.x);
        simd::store(&rayDirs[1][i], dir.y);
        simd::store(&rayDirs[2][i], dir.z);
    }

    // Pre-compute convolution weights
    for (uint32_t i = 0; i < 64; i += simd::vec_width) {
        v_float2 uv = simd::conv<float>((i + simd::lane_idx<v_uint>) >> v_uint2(0, 3) & 7) / 7.0f;
        v_float3 texelDir = swr::texutil::UnmapOctahedron(uv);

        v_float sumWeight = 0;

        for (uint32_t j = 0; j < numRays; j++) {
            sumWeight += GetProbeIrradianceWeight(texelDir, rayDirs, j);
        }
        v_float normScale = 1.0 / simd::max(sumWeight * (2 * simd::inv_pi), 1e-4);

        for (uint32_t j = 0; j < numRays; j++) {
            v_float weight = GetProbeIrradianceWeight(texelDir, rayDirs, j) * normScale;

            uint32_t offset = j * simd::vec_width + (i * ProbeMaxRays);
            _mm256_storeu_epi16(&irradWeights[offset], _mm512_cvtps_ph(weight, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }
    for (uint32_t i = 0; i < 256; i += simd::vec_width) {
        v_float2 uv = simd::conv<float>((i + simd::lane_idx<v_uint>) >> v_uint2(0, 4) & 15) / 15.0f;
        v_float3 texelDir = swr::texutil::UnmapOctahedron(uv);

        v_float sumWeight = 0;

        for (uint32_t j = 0; j < numRays; j++) {
            sumWeight += GetProbeDepthWeight(texelDir, rayDirs, j);
        }
        v_float normScale = 1.0 / simd::max(sumWeight, 1e-4);

        for (uint32_t j = 0; j < numRays; j++) {
            v_float weight = GetProbeDepthWeight(texelDir, rayDirs, j) * normScale;

            uint32_t offset = j * simd::vec_width + (i * ProbeMaxRays);
            _mm256_storeu_epi16(&depthWeights[offset], _mm512_cvtps_ph(weight, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
    }

    // Trace probes
    raster.Dispatch([&](uint32_t workerId) {
        uint32_t probesPerThread = GIProbeField::ProbesPerCascade / raster.GetThreadCount();
        uint32_t startOffset = workerId * probesPerThread;
        uint32_t endOffset = glm::min(GIProbeField::ProbesPerCascade, startOffset + probesPerThread);

        // Force denormal flushing to avoid slowing convolution with near zero weights
        // (shouldn't be an issue now with pre-computed FP16 weights, but just in case)
        _mm_setcsr(_mm_getcsr() | _MM_FLUSH_ZERO_ON);

        for (uint32_t probeIdx = startOffset; probeIdx < endOffset; probeIdx++) {
            bool isActive = ProbeField->CurrActiveBitmap[probeIdx / 64u] & (1ull << (probeIdx & 63));
            bool wasActive = ProbeField->PrevActiveBitmap[probeIdx / 64u] & (1ull << (probeIdx & 63));
            if (!isActive) continue;

            uint8_t& sampleCount = ProbeField->SampleCounts[probeIdx];
            uint32_t maxSampleCount = wasActive ? ProbeField->MaxHysteresis : ProbeField->MaxHysteresis / 3 + 2;
            sampleCount = (uint8_t)glm::min(sampleCount + 1u, maxSampleCount);

            float hysteresis = (1.0f / sampleCount);

            float3 center = ProbeField->Positions[probeIdx];
            float rayRadiance[3][ProbeMaxRays];
            float rayDist[ProbeMaxRays];

            for (uint32_t i = 0; i < numRays; i += simd::vec_width) {
                v_float3 rayDir = { simd::load(&rayDirs[0][i]), simd::load(&rayDirs[1][i]), simd::load(&rayDirs[2][i]) };

                VoxelHitResult hit = BrickMap->RayCast(center, rayDir, v_float2(0, maxDist), v_mask(~0u), true);
                v_int isValidHit = hit.Dist > sqrtf(3) / BrickMap->CascadeBaseScale;
                v_int isMiss = hit.Dist < 0;

                v_float3 radiance = 0;
                
                simd_if(isValidHit) {
                    SurfaceAttribs surf = UnpackVoxelAttribs(hit.Attribs);
                    v_float3 worldPos = center + rayDir * hit.Dist;

                    v_float3 irradiance = 0;
                    if (ProbeField->MultiBounce) {
                        irradiance = SampleDiffuse(*ProbeField, worldPos, surf.Normal, rayDir);
                    }
                    simd::cmov(radiance, EvalLighting(*this, surf, worldPos, irradiance, int3(0), 1), v_exec);
                }
                simd_if(isMiss) {
                    constexpr swr::SamplerDesc sampler = { .MagFilter = swr::FilterMode::Nearest, .MinFilter = swr::FilterMode::Nearest };
                    simd::cmov(radiance, SkyboxTex->SampleOctLevel<sampler>(rayDir, 4) * SkyboxIntensity, v_exec);
                }

                // Clamp fireflies (mostly coming from skybox)
                v_float luminance = simd::dot(radiance, 1);
                const float maxLuminance = 20;
                radiance *= simd::min(1.0f, (maxLuminance * 2) * simd::approx_rcp(maxLuminance + luminance));

                simd::store(&rayRadiance[0][i], radiance.x);
                simd::store(&rayRadiance[1][i], radiance.y);
                simd::store(&rayRadiance[2][i], radiance.z);
                simd::store(&rayDist[i], isValidHit ? hit.Dist : maxDist * 2);
            }

            // Convolution
            uint32_t* irradianceData = ProbeField->Irradiance[probeIdx];
            uint32_t* depthData = ProbeField->Depth[probeIdx];

            for (uint32_t i = 0; i < 64; i += simd::vec_width) {
                v_float3 sumIrradiance = 0;

                for (uint32_t j = 0; j < numRays; j++) {
                    v_float weight = _mm512_cvtph_ps(_mm256_loadu_epi16(&irradWeights[j * simd::vec_width + (i * ProbeMaxRays)]));
                    sumIrradiance += v_float3(rayRadiance[0][j], rayRadiance[1][j], rayRadiance[2][j]) * weight;
                }
                v_float3 oldIrradiance = swr::pixfmt::R11G11B10f::Unpack(simd::load(&irradianceData[i]));
                v_float3 newIrradiance = oldIrradiance + (sumIrradiance - oldIrradiance) * hysteresis;
                simd::store(&irradianceData[i], swr::pixfmt::R11G11B10f::Pack(newIrradiance));
            }
            for (uint32_t i = 0; i < 256; i += simd::vec_width) {
                v_float2 sumDepth = 0;

                for (uint32_t j = 0; j < numRays; j++) {
                    v_float weight = _mm512_cvtph_ps(_mm256_loadu_epi16(&depthWeights[j * simd::vec_width + (i * ProbeMaxRays)]));
                    sumDepth += v_float2(rayDist[j], rayDist[j] * rayDist[j]) * weight;
                }
                v_float2 oldDepth = swr::pixfmt::RG16f::Unpack(simd::load(&depthData[i]));
                v_float2 newDepth = oldDepth + (sumDepth - oldDepth) * hysteresis;
                simd::store(&depthData[i], swr::pixfmt::RG16f::Pack(newDepth));
            }
        }
    });

    memcpy(ProbeField->PrevActiveBitmap, ProbeField->CurrActiveBitmap, sizeof(ProbeField->CurrActiveBitmap));
    memset(ProbeField->CurrActiveBitmap, 0, sizeof(ProbeField->CurrActiveBitmap));
}
void ShadingContext::DrawProbesDebug() {
    float3 projFactors = float3(ProjMat[0][0], ProjMat[1][1], ProjMat[3][2]);  // f/ar, -f, znear
    uint2 fbSize = uint2(ColorBuffer->Width, ColorBuffer->Height);

    float2 scaleNdc = 2.0f / float2(fbSize);
    float2 centerNdc = 0.5f * scaleNdc - 1.0f;
    float3x3 rayMatrix = GetCameraRayMatrix(WorldToClipMat);

    for (uint32_t i = 0; i < GIProbeField::ProbesPerCascade; i++) {
        float3 center = ProbeField->Positions[i];

        float radius = 0.15f;

        float4 clipBounds;
        if (!ProjectSphere(ViewMat * float4(center, 1), radius, projFactors, clipBounds)) continue;

        uint4 screenBB = uint4(glm::clamp(clipBounds, 0.0f, 1.0f) * float4(fbSize, fbSize));
        screenBB.x &= ~3u;
        screenBB.y &= ~3u;

        // https://iquilezles.org/articles/intersectors
        float3 origin = ViewPos;
        float3 oc = origin - center;
        float c = glm::dot(oc, oc) - radius * radius;
        if (c <= 0) continue;

        for (uint32_t y = screenBB.y; y < screenBB.w; y += 4) {
            uint32_t tileOffset = ColorBuffer->GetTexelOffset(screenBB.x, y, 0);

            for (uint32_t x = screenBB.x; x < screenBB.z; x += 4, tileOffset += 16) {
                v_int2 screenPos = int2(x, y) + swr::TilePixelOffsets;
                v_float2 screenNdc = simd::conv<float>(screenPos) * scaleNdc + centerNdc;
                v_float3 dir = GetCameraRayDir(screenNdc, rayMatrix);

                v_float b = simd::dot(oc, dir);
                v_float h = b * b - c;
                v_int mask = b <= 0 && h >= 0;
                if (!simd::any(mask)) continue;

                v_float t = -b - simd::sqrt(h);
                v_float3 surfacePos = origin + dir * t;

                v_float oldDepth = simd::load<v_float>(&DepthBuffer->Data[tileOffset]);
                v_float newDepth = projFactors.z / -simd::mul(ViewMat, v_float4(surfacePos, 1)).z;
                mask &= newDepth > oldDepth;
                if (!simd::any(mask)) continue;

#if 1
                v_float3 color = SampleProbeIrradiance(*ProbeField, (int)i, surfacePos - center);
                color = Tonemap_Unreal(color);
#else
                v_float2 moments = SampleProbeDepth(*ProbeField, (int)i, surfacePos - center);
                v_float variance = simd::abs(moments.x * moments.x - moments.y);
                v_float3 color = v_float3(moments.x, simd::approx_sqrt(variance), 0) * 0.1;
#endif

                v_uint packedColor = swr::pixfmt::RGBA8u::Pack({ color, 1 });
                simd::store(&ColorBuffer->Data[tileOffset], packedColor, mask);
                simd::store(&DepthBuffer->Data[tileOffset], newDepth, mask);
            }
        }
    }
}

void GIProbeField::Reset(bool clearProbes) {
    for (uint32_t y = 0; y < GridSize.y; y++) {
        for (uint32_t z = 0; z < GridSize.z; z++) {
            for (uint32_t x = 0; x < GridSize.x; x++) {
                uint32_t i = x + z * GridSize.x + y * (GridSize.x * GridSize.z);
                Positions[i] = ProbeBaseOffset + float3(x, y, z) * ProbeSpacing;
            }
        }
    }
    if (clearProbes) {
        memset(CurrActiveBitmap, 255, sizeof(CurrActiveBitmap));
        memset(PrevActiveBitmap, 0, sizeof(PrevActiveBitmap));
        memset(Irradiance, 0, sizeof(Irradiance));
        memset(Depth, 0, sizeof(Depth));
    }
}
