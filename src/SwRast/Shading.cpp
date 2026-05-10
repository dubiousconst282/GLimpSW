#include "Shading.h"
#include "Camera.h"

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"

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

// Tonemap importance samples to prevent fireflies - http://graphicrants.blogspot.com/2013/12/tone-mapping.html
v_float3 TonemapSample(v_float3 color) {
    v_float luma = simd::dot(color, { 0.299f, 0.587f, 0.114f });
    return color * simd::approx_rcp(1.0f + luma);
}
v_float3 TonemapSampleInv(v_float3 color) {
    v_float luma = simd::dot(color, { 0.299f, 0.587f, 0.114f });
    return color * simd::approx_rcp(1.0f - luma);
}

glm::vec2 Hammersley(uint32_t i, uint32_t numSamples) {
    uint32_t bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    return { i / (float)numSamples, bits * (1.0 / UINT_MAX) };
}
v_float3 ImportanceSampleGGX(glm::vec2 Xi, v_float roughness, v_float3 N) {
    v_float a = roughness * roughness;

    v_float phi = 2.0 * simd::pi * Xi.x;
    v_float cosTheta = simd::sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    v_float sinTheta = simd::sqrt(1.0 - cosTheta * cosTheta);

    v_float3 H = { simd::cos(phi) * sinTheta, simd::sin(phi) * sinTheta, cosTheta };

    // vec3 upVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    v_int nzMask = simd::abs(N.z) < 0.999f;
    v_float3 upVector = { nzMask ? 0.0f : 1.0f, 0.0, nzMask ? 1.0f : 0.0f };
    v_float3 tangentX = simd::normalize(simd::cross(upVector, N));
    v_float3 tangentY = simd::cross(N, tangentX);
    return tangentX * H.x + tangentY * H.y + N * H.z;
}

v_float3 PrefilterDiffuseIrradiance(const swr::HdrTexture2D& envTex, v_float3 N) {
    v_int nzMask = simd::abs(N.z) < 0.999f;
    v_float3 up = { nzMask ? 0.0f : 1.0f, 0.0, nzMask ? 1.0f : 0.0f };
    v_float3 right = simd::normalize(simd::cross(up, N));
    up = simd::cross(N, right);

    v_float3 color = 0.0f;
    uint32_t sampleCount = 0;

    constexpr float halfPi = simd::pi * 0.5;
    float deltaPhi = simd::tau / 360.0;
    float deltaTheta = halfPi / 90.0;

#ifndef NDEBUG
    deltaPhi = simd::tau / 16.0;
    deltaTheta = halfPi / 4.0;
#endif

    for (float phi = 0.0; phi < simd::tau; phi += deltaPhi) {
        for (float theta = 0.0; theta < halfPi; theta += deltaTheta) {
            // Spherical to World Space in two steps...
            v_float3 tempVec = cosf(phi) * right + sinf(phi) * up;
            v_float3 sampleVector = cosf(theta) * N + sinf(theta) * tempVec;
            v_float3 envColor = envTex.SampleOctLevel<EnvSampler>(sampleVector, 2);

            color = color + TonemapSample(envColor) * cosf(theta) * sinf(theta);
            sampleCount++;
        }
    }
    return TonemapSampleInv(color * (simd::pi * 1.0f / sampleCount));
}

// From Karis, 2014
v_float3 PrefilterEnvMap(const swr::HdrTexture2D& envTex, float roughness, v_float3 R) {
#ifdef NDEBUG
    const uint32_t numSamples = 128;
#else
    const uint32_t numSamples = 16;
#endif
    v_float3 N = R;
    v_float3 V = R;

    roughness = std::max(roughness, 0.01f);

    v_float3 prefilteredColor = 0.0f;
    v_float totalWeight = 0.0f;

    for (uint32_t i = 0; i < numSamples; i++) {
        glm::vec2 Xi = Hammersley(i, numSamples);
        v_float3 H = ImportanceSampleGGX(Xi, roughness, N);
        v_float VoH = simd::dot(V, H);
        v_float NoH = simd::max(VoH, 0.0f);  // Since N = V in our approximation
        // Use microfacet normal H to find L
        v_float3 L = 2.0f * VoH * H - V;
        v_float NoL = simd::max(simd::dot(N, L), 0.0f);

        if (simd::any(NoL > 0.0f)) {
            // Based off https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
            // Typically you'd have the following:
            // float pdf = D_GGX(NoH, roughness) * NoH / (4.0 * VoH);
            // but since V = N => VoH == NoH
            v_float pdf = D_GGX(NoH, roughness) / 4.0f + 0.001f;
            // Solid angle of current sample -- bigger for less likely samples
            v_float omegaS = 1.0f / (numSamples * pdf);
            // Solid angle of texel
            float omegaP = 4.0f * simd::pi / (6 * envTex.Width * envTex.Width);
            // Mip level is determined by the ratio of our sample's solid angle to a texel's solid angle
            v_float mipLevel = simd::max(0.5f * simd::approx_log2(omegaS / omegaP), 0.0f);

            v_float3 envColor = envTex.SampleOctLevel<EnvSampler>(L, mipLevel);

            prefilteredColor = prefilteredColor + TonemapSample(envColor) * NoL;
            totalWeight += NoL;
        }
    }
    return TonemapSampleInv(prefilteredColor / totalWeight);
}

v_float2 IntegrateBRDF(v_float NoV, v_float roughness) {
    v_float3 N = { 0, 0, 1 };
    v_float3 V = { simd::sqrt(1.0f - NoV * NoV), 0.0f, NoV };
    v_float2 r = { 0.0f, 0.0f };
    const uint32_t sampleCount = 1024;

    for (uint32_t i = 0; i < sampleCount; i++) {
        glm::vec2 Xi = Hammersley(i, sampleCount);
        v_float3 H = ImportanceSampleGGX(Xi, roughness, N);
        v_float3 L = 2.0f * simd::dot(V, H) * H - V;

        v_float VoH = simd::max(0.0f, simd::dot(V, H));
        v_float NoL = simd::max(0.0f, L.z);
        v_float NoH = simd::max(0.0f, H.z);

        if (simd::any(NoL > 0.0f)) {
            v_float G = V_SmithGGXCorrelatedFast(NoV, NoL, roughness * roughness);
            v_float Gv = G * 4.0f * VoH * NoL / NoH;
            Gv = NoL > 0.0f ? Gv : 0.0f;

            v_float Fc = pow5(1 - VoH);
            r.x += Gv * (1 - Fc);
            r.y += Gv * Fc;
        }
    }
    return { r.x * (1.0f / sampleCount), r.y * (1.0f / sampleCount) };
}

swr::HdrTexture2D::Ptr GenerateIrradianceMap(const swr::HdrTexture2D& envTex) {
    auto tex = swr::CreateTexture<swr::HdrTexture2D>(64, 64, 1, 1);

    swr::texutil::IterateTiles(tex->Width, tex->Height, [&](uint32_t x, uint32_t y, v_float u, v_float v) {
        v_float3 dir = swr::texutil::UnmapOctahedron({ u, v });
        v_float3 color = PrefilterDiffuseIrradiance(envTex, dir);
        tex->WriteTile(color, x, y);
    });
    return tex;
}
swr::HdrTexture2D::Ptr GenerateRadianceMap(const swr::HdrTexture2D& envTex) {
    auto tex = swr::CreateTexture<swr::HdrTexture2D>(256, 256, 8);

    for (uint32_t level = 0; level < tex->MipLevels; level++) {
        uint32_t w = tex->Width >> level;
        uint32_t h = tex->Height >> level;

        swr::texutil::IterateTiles(w, h, [&](uint32_t x, uint32_t y, v_float u, v_float v) {
            v_float3 dir = swr::texutil::UnmapOctahedron({ u, v });
            v_float3 color = PrefilterEnvMap(envTex, level / (tex->MipLevels - 1.0f), dir);
            tex->WriteTile(color, x, y, 0, level);
        });
    }
    return tex;
}
swr::TexturePtr2D<swr::pixfmt::RG16f> GenerateEnvBRDFLut() {
    auto tex = swr::CreateTexture2D<swr::pixfmt::RG16f>(128, 128, 1, 1);

    swr::texutil::IterateTiles(tex->Width, tex->Height, [&](uint32_t x, uint32_t y, v_float u, v_float v) {
        v_float2 f_ab = IntegrateBRDF(u, v);
        tex->WriteTile(f_ab, x, y);
    });
    return tex;
}

// https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
v_float2 ApproxEnvBRDF(v_float NoV, v_float roughness) {
    const v_float4 c0 = { -1, -0.0275, -0.572, 0.022 };
    const v_float4 c1 = { 1, 0.0425, 1.04, -0.04 };
    v_float4 r = roughness * c0 + c1;
    v_float a004 = simd::min(r.x * r.x, simd::approx_exp2(-9.28 * NoV)) * r.x + r.y;
    return { a004 * -1.04 + r.z, a004 * 1.04 + r.w };
}

v_float3 Tonemap_Unreal(v_float3 x) {
    // Unreal 3, Documentation: "Color Grading"
    // Adapted to be close to Tonemap_ACES, with similar range
    // Gamma 2.2 correction is baked in, don't use with sRGB conversion!
    return x / (x + 0.155) * 1.019;
}

glm::vec2 UnpackHalf2x16(uint32_t x) {
    auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
    return { v[0], v[1] };
}
void UnpackNormalTangent(v_uint packed, v_float3& norm, v_float3& tang) {
    v_float4 oct = swr::pixfmt::RGBA8u::Unpack(packed);
    norm = swr::texutil::UnmapOctahedron({ oct.x, oct.y });
    tang = swr::texutil::UnmapOctahedron({ oct.z, oct.w });
}

// Standard "source over" alpha blending. Not gamma correct.
v_uint AlphaBlendU8(v_uint bg, v_uint fg) {
    v_uint t = (fg >> 1) & 0x7F80'0000;
    t |= (t >> 16) | 0x0040'0040;

    v_uint rb = simd::lerp16((bg >> 0 & 0x00FF'00FF), (fg >> 0 & 0x00FF'00FF), t);
    v_uint ag = simd::lerp16((bg >> 8 & 0x00FF'00FF), (fg >> 8 & 0x00FF'00FF), t);
    return rb | (ag << 8);
}

// 2D Polyhedral Bounds of a Clipped, Perspective - Projected 3D Sphere. Michael Mara, Morgan McGuire. 2013.
// proj = znear, P00, P11
bool ProjectSphere(float3 c, float r, float3 proj, float2& minPos, float2& maxPos) {
    c.z *= -1;
    if (c.z < r + proj.x) return false;

    float3 cr = c * r;
    float vx = glm::sqrt(c.x * c.x + c.z * c.z - r * r);
    float vy = glm::sqrt(c.y * c.y + c.z * c.z - r * r);

    minPos.x = (vx * c.x - cr.z) / (vx * c.z + cr.x) * (proj.y * 0.5f) + 0.5f;
    minPos.y = (vy * c.y + cr.z) / (vy * c.z - cr.y) * (proj.z * 0.5f) + 0.5f;
    maxPos.x = (vx * c.x + cr.z) / (vx * c.z - cr.x) * (proj.y * 0.5f) + 0.5f;
    maxPos.y = (vy * c.y - cr.z) / (vy * c.z + cr.y) * (proj.z * 0.5f) + 0.5f;

    return true;
}

void ShadeMeshlet(const ShadingContext& ctx, uint32_t index, swr::ShadedMeshlet& output) {
    const Meshlet& mesh = ctx.Meshlets[ctx.MeshletOffset + index];

    output.PrimCount = 0;

    // Frustum culling
    for (uint32_t i = 0; i < 5; i++) {
        float dist = glm::dot(mesh.BoundCenter, float3(ctx.FrustumPlanes[i])) + ctx.FrustumPlanes[i].w;
        if (dist < -mesh.BoundRadius) return;
    }

    if (ctx.OcclDepthMap != nullptr) {
        float3 boundCenterVS = float3(ctx.ObjectToViewMat * float4(mesh.BoundCenter, 1));
        float boundRadiusVS = mesh.BoundRadius * ctx.ObjectToWorldScale;

        float2 screenMin, screenMax;
        if (ProjectSphere(boundCenterVS, boundRadiusVS, ctx.SphereProjFactors, screenMin, screenMax)) {
            // It is very worth picking the finer mip and testing more samples here,
            // since our raster is still slow as fuck compared to hardware (though that
            // may be true in general, since culling is so ridiculously cheap).
            float2 bbSize = (screenMax - screenMin) * ctx.FramebufferSize;
            int mipLevel = (simd::as<int>(glm::max(bbSize.x, bbSize.y)) >> 23) - 127;
            mipLevel = glm::clamp(mipLevel, 1, (int)ctx.OcclDepthMap->MipLevels - 1);

            int2 texMin = glm::max(int2(screenMin * ctx.FramebufferSize), 0) >> mipLevel;
            int2 texMax = glm::min(int2(screenMax * ctx.FramebufferSize), int2(ctx.FramebufferSize) - 1) >> mipLevel;

            uint32_t mipOffset = ctx.OcclDepthMap->MipOffsets[mipLevel - 1];
            uint32_t stride = ctx.OcclDepthMap->RowShift - (mipLevel - 1);

            float depthSphere = ctx.SphereProjFactors.x / (-boundCenterVS.z - boundRadiusVS);
            float depthVisible = FLT_MAX;

            for (int y = texMin.y; y <= texMax.y; y++) {
                for (int x = texMin.x; x <= texMax.x; x++) {
                    uint32_t texelOffset = ctx.OcclDepthMap->GetTexelOffset((uint32_t)x, (uint32_t)y, stride);
                    float value = simd::as<float>(ctx.OcclDepthMap->Data[mipOffset + texelOffset]);
                    depthVisible = glm::min(depthVisible, value);
                }
            }

#if 0
            ImDrawList* dl = ImGui::GetForegroundDrawList();
            ImVec2 dp = ImGui::GetIO().DisplaySize;
            dl->AddRect(ImVec2(screenMin.x, screenMin.y) * dp, ImVec2(screenMax.x, screenMax.y) * dp, 0xFF0000FF);

            ImVec2 scale = dp / ImVec2(ctx.FramebufferSize.x, ctx.FramebufferSize.y);
            int texelSize = (1 << mipLevel);

            for (int y = texMin.y; y <= texMax.y; y++) {
                for (int x = texMin.x; x <= texMax.x; x++) {
                    ImVec2 p0 = ImVec2(x << mipLevel, y << mipLevel);
                    ImVec2 p1 = p0 + ImVec2(texelSize, texelSize);
                    dl->AddRect(p0 * scale, p1 * scale, 0xFF00FF00);
                }
            }
            char text[128];
            snprintf(text, sizeof(text), "bb=%.0fx%.0f mip=%d", bbSize.x, bbSize.y, mipLevel);
            dl->AddText(ImVec2(screenMin.x, screenMin.y) * dp, 0xFFFFFFFF, text);
#endif

            if (depthSphere < depthVisible) return;
        }
    }

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

template<bool EnableAlphaTest>
void FS_EncodeSurfaceId(const ShadingContext& ctx, swr::Framebuffer& fb, swr::FragmentVars& vars) {
    // Depth test
    v_float oldDepth = vars.LoadTile(fb.GetDepthBuffer());
    vars.TileMask &= simd::movemask(vars.Depth > oldDepth);
    if (vars.TileMask == 0) return;

    if constexpr (EnableAlphaTest) {
        const Meshlet& mesh = ctx.Meshlets[ctx.MeshletOffset + vars.MeshletId];
        const Material& material = ctx.Materials[mesh.MaterialId];

        v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[0]]);
        v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[1]]);
        v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[2]]);
        v_float2 uv = vars.Interpolate(uv0, uv1, uv2);
        v_uint color = material.Texture->SampleImplicitLod<SurfaceSampler>(uv.x, uv.y);

        vars.TileMask &= simd::movemask(color >= uint32_t(material.AlphaCutoff << 24));
    }
    v_uint surfaceId = (ctx.MeshletOffset + vars.MeshletId) * swr::ShadedMeshlet::MaxPrims + vars.PrimId;
    vars.StoreTile(fb.GetDepthBuffer(), vars.Depth);
    vars.StoreTile(fb.GetColorBuffer(), surfaceId);
}

void FS_Overdraw(const ShadingContext& ctx, swr::Framebuffer& fb, swr::FragmentVars& vars) {
    uint32_t* buffer = fb.GetColorBuffer();
    v_uint color = vars.LoadTile(buffer);
    v_int isActive = _mm512_movm_epi32(vars.TileMask);
    color = _mm512_adds_epu8(color, isActive ? v_uint(0xFF'301010) : v_uint(0xFF'101030));

    vars.TileMask = 0xFFFF;
    vars.StoreTile(buffer, color);
    vars.StoreTile(fb.GetDepthBuffer(), simd::max(vars.Depth, vars.LoadTile(fb.GetDepthBuffer())));
}

void FS_EncodeGBuffer(const ShadingContext& ctx, swr::Framebuffer& fb, swr::FragmentVars& vars) {
    // Depth test
    v_float oldDepth = vars.LoadTile(fb.GetDepthBuffer());
    vars.TileMask &= simd::movemask(vars.Depth > oldDepth);
    if (vars.TileMask == 0) return;

    const Meshlet& mesh = ctx.Meshlets[ctx.MeshletOffset + vars.MeshletId];

    if (mesh.MaterialId == UINT_MAX) {
        vars.StoreTile(fb.GetDepthBuffer(), vars.Depth);
        return;
    }

    const Material& material = ctx.Materials[mesh.MaterialId];

    v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[0]]);
    v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[1]]);
    v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[2]]);
    v_float2 uv = vars.Interpolate(uv0, uv1, uv2);

    v_uint baseColor = material.Texture->SampleImplicitLod<SurfaceSampler>(uv.x, uv.y);
    vars.TileMask &= simd::movemask(baseColor >= uint32_t(material.AlphaCutoff << 24));
    if (vars.TileMask == 0) return;

    v_uint packedCh2 = 0;

    if (material.Texture->NumLayers >= 2) {
        // Nx Ny M R
        v_uint packedNMR = material.Texture->SampleImplicitLod<SurfaceSampler>(uv.x, uv.y, 1);

        // Reconstruct Z from 2-channel normap map https://aras-p.info/texts/CompactNormalStorage.html
        v_float3 normalTS = v_float3(simd::conv<float>(packedNMR >> v_uint2(0, 8) & 255) * (1.0f / 127.5f) - 1.0f, 0);
        normalTS.z = simd::approx_sqrt(1.0f - (normalTS.x * normalTS.x + normalTS.y * normalTS.y));

        v_float3 normal0, normal1, normal2;
        v_float3 tangent0, tangent1, tangent2;
        UnpackNormalTangent(mesh.NormalTangents[vars.VertexId[0]], normal0, tangent0);
        UnpackNormalTangent(mesh.NormalTangents[vars.VertexId[1]], normal1, tangent1);
        UnpackNormalTangent(mesh.NormalTangents[vars.VertexId[2]], normal2, tangent2);

        v_float3 normalWS = simd::normalize(simd::mul(ctx.ObjectToWorldMat, vars.Interpolate(normal0, normal1, normal2)));
        v_float3 tangentWS = simd::normalize(simd::mul(ctx.ObjectToWorldMat, vars.Interpolate(tangent0, tangent1, tangent2)));
        uint32_t handedness = (mesh.TangentHandedness >> vars.VertexId[0] & 1) << 31;

        v_float3 bitangentWS = simd::cross(normalWS, tangentWS);
        bitangentWS = simd::as<v_float3>(simd::as<v_uint3>(bitangentWS) ^ handedness);

        // normalize(mul(normalTS, float3x3(tangentWS, bitangentWS, normalWS)));
        v_float3 normalRemapped = simd::normalize({
            normalTS.x * tangentWS.x + normalTS.y * bitangentWS.x + normalTS.z * normalWS.x,
            normalTS.x * tangentWS.y + normalTS.y * bitangentWS.y + normalTS.z * normalWS.y,
            normalTS.x * tangentWS.z + normalTS.y * bitangentWS.z + normalTS.z * normalWS.z,
        });

        v_float2 oct = swr::texutil::MapOctahedron(normalRemapped);
        packedCh2 = simd::conv<uint32_t>(oct.x * 1023.0f + 0.5f);
        packedCh2 |= simd::conv<uint32_t>(oct.y * 1023.0f + 0.5f) << 10;
        packedCh2 |= (packedNMR >> 18 & 0x3F) << 20;
        packedCh2 |= (packedNMR >> 26 & 0x3F) << 26;
    }
    vars.StoreTile(fb.GetDepthBuffer(), vars.Depth);
    vars.StoreTile(fb.GetColorBuffer(), baseColor);
    vars.StoreTile(fb.GetLayerData<uint32_t>(2), packedCh2);

    // bool hasEmissive = MaterialTex->NumLayers >= 3 && any(alpha == 255);

    // if (hasEmissive) [[unlikely]] {
    //     VInt emissiveColor = MaterialTex->Sample<SurfaceSampler>(u, v, 2);
    //     _mm512_mask_store_epi32(fb.GetAttachmentBuffer<uint32_t>(4, vars.TileOffset), vars.TileMask, emissiveColor);
    // }
}

// http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
void IntersectTriangle(v_float4 v0, v_float4 v1, v_float4 v2, v_float2 screenUV, float2 screenSize,  //
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

    v_float2 rel0 = screenUV - p0;
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

struct SurfaceData {
    v_uint Albedo;
    v_float3 Normal;
    v_float Metallic, Roughness;
};
[[clang::always_inline]]
SurfaceData ResolveSurface(const ShadingContext& ctx, const uint32_t* tileData, v_mask activeMask, v_float2 screenUV, float2 screenSize) {
    v_float3 pos[3];
    v_uint packedTexCoord[3];
    v_uint packedNormalTangent[3];
    v_uint tangentHandedness = 0;
    v_uint materialId = 0;

    v_uint surfaceId = simd::load(tileData);

    // Waterfall loop appears to be faster than gathers, almost always
    for (v_mask pendingMask = activeMask; pendingMask != 0;) {
        uint32_t surfaceId_i = tileData[simd::tzcnt(pendingMask)];
        uint32_t meshletId_i = surfaceId_i / swr::ShadedMeshlet::MaxPrims;
        uint32_t triangle_i = surfaceId_i % swr::ShadedMeshlet::MaxPrims;

        v_int iterMask = (surfaceId == surfaceId_i);
        pendingMask &= ~simd::movemask(iterMask);

        const Meshlet& mesh = ctx.Meshlets[meshletId_i];

        #pragma unroll
        for (uint32_t vi = 0; vi < 3; vi++) {
            uint32_t idx = mesh.Indices[vi][triangle_i];

            simd::cmov(pos[vi].x, v_float(mesh.Positions[0][idx]), iterMask);
            simd::cmov(pos[vi].y, v_float(mesh.Positions[1][idx]), iterMask);
            simd::cmov(pos[vi].z, v_float(mesh.Positions[2][idx]), iterMask);
            simd::cmov(packedTexCoord[vi], v_uint(mesh.TexCoords[idx]), iterMask);
            simd::cmov(packedNormalTangent[vi], v_uint(mesh.NormalTangents[idx]), iterMask);

            if (vi == 0 && (mesh.TangentHandedness >> idx & 1)) {
                simd::cmov(tangentHandedness, v_uint(1u << 31), iterMask);
            }
        }
        simd::cmov(materialId, v_uint(mesh.MaterialId), iterMask);
    }

    v_float4 ndc0 = simd::mul(ctx.ObjectToClipMat, v_float4(pos[0], 1.0f));
    v_float4 ndc1 = simd::mul(ctx.ObjectToClipMat, v_float4(pos[1], 1.0f));
    v_float4 ndc2 = simd::mul(ctx.ObjectToClipMat, v_float4(pos[2], 1.0f));

    v_float3 bary, ddx, ddy;
    IntersectTriangle(ndc0, ndc1, ndc2, screenUV, screenSize, bary, ddx, ddy);

    v_float2 texCoord0 = swr::pixfmt::RG16f::Unpack(packedTexCoord[0]);
    v_float2 texCoord10 = swr::pixfmt::RG16f::Unpack(packedTexCoord[1]) - texCoord0;
    v_float2 texCoord20 = swr::pixfmt::RG16f::Unpack(packedTexCoord[2]) - texCoord0;

    v_float texU = texCoord0.x + texCoord10.x * bary.y + texCoord20.x * bary.z;
    v_float texV = texCoord0.y + texCoord10.y * bary.y + texCoord20.y * bary.z;
    v_float4 texGrad = {
        texCoord10.x * ddx.y + texCoord20.x * ddx.z,
        texCoord10.y * ddx.y + texCoord20.y * ddx.z,
        texCoord10.x * ddy.y + texCoord20.x * ddy.z,
        texCoord10.y * ddy.y + texCoord20.y * ddy.z,
    };

    v_uint packedAlbedo = 0;
    v_uint packedNMR = 0;

    for (v_mask pendingMask = activeMask & simd::movemask(materialId != UINT_MAX); pendingMask != 0;) {
        uint32_t id = ((uint32_t*)&materialId)[simd::tzcnt(pendingMask)];

        v_int iterMask = materialId == id;
        pendingMask &= ~simd::movemask(iterMask);

        const swr::RgbaTexture2D& tex = *ctx.Materials[id].Texture;
        v_int mipLevel = swr::texutil::CalcMipLevel(texGrad, v_float2(tex.ScaleU, tex.ScaleV));
        simd::cmov(packedAlbedo, tex.SampleLevel<SurfaceSampler>(texU, texV, 0, mipLevel), iterMask);

        if (tex.NumLayers >= 2) {
            simd::cmov(packedNMR, tex.SampleLevel<SurfaceSampler>(texU, texV, 1, mipLevel), iterMask);
        }
    }
    v_float3 normal0, normal1, normal2;
    v_float3 tangent0, tangent1, tangent2;
    UnpackNormalTangent(packedNormalTangent[0], normal0, tangent0);
    UnpackNormalTangent(packedNormalTangent[1], normal1, tangent1);
    UnpackNormalTangent(packedNormalTangent[2], normal2, tangent2);

    v_float3 normalWS = simd::normalize(simd::mul(ctx.ObjectToWorldMat, swr::BaryLerp(bary, normal0, normal1, normal2)));
    v_float3 normalRemapped = normalWS;

    if (simd::any((packedNMR & 0xFFFF) != 0)) {
        v_float3 tangentWS = simd::normalize(simd::mul(ctx.ObjectToWorldMat, swr::BaryLerp(bary, tangent0, tangent1, tangent2)));

        v_float3 bitangentWS = simd::cross(normalWS, tangentWS);
        bitangentWS = simd::as<v_float3>(simd::as<v_uint3>(bitangentWS) ^ tangentHandedness);

        // Reconstruct Z from 2-channel normap map https://aras-p.info/texts/CompactNormalStorage.html
        v_float3 normalTS = v_float3(simd::conv<float>(packedNMR >> v_uint2(0, 8) & 255) * (1.0f / 127.5f) - 1.0f, 0);
        normalTS.z = simd::approx_sqrt(1.0f - (normalTS.x * normalTS.x + normalTS.y * normalTS.y));

        // simd::normalize(simd::mul(normalTS, float3x3(tangentWS, bitangentWS, normalWS)));
        normalRemapped = simd::normalize({
            normalTS.x * tangentWS.x + normalTS.y * bitangentWS.x + normalTS.z * normalWS.x,
            normalTS.x * tangentWS.y + normalTS.y * bitangentWS.y + normalTS.z * normalWS.y,
            normalTS.x * tangentWS.z + normalTS.y * bitangentWS.z + normalTS.z * normalWS.z,
        });
    }

    return {
        .Albedo = packedAlbedo,
        .Normal = normalRemapped,
        .Metallic = simd::conv<float>(packedNMR >> 16 & 255) * (1.0f / 255),
        .Roughness = simd::conv<float>(packedNMR >> 24 & 255) * (1.0f / 255),
    };
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
v_float3 EvalLighting(const SurfaceData& surf, const Light* lights, uint32_t numLights, float exposure, v_float3 worldPos, v_float3 viewPos) {
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

    v_float3 viewDir = simd::normalize(viewPos - worldPos);
    v_float NoV = simd::abs(simd::dot(surf.Normal, viewDir)) + 1e-5f;

    for (uint32_t i = 0; i < numLights; i++) {
        const Light& light = lights[i];
        v_float3 lightDir = light.Type == Light::kTypeDirectional ? -light.Direction : simd::normalize(light.Position - worldPos);

        v_float NoL = simd::dot(surf.Normal, lightDir);
        if (simd::all(NoL < 1e-4)) continue;

        v_float attenuation = GetLightAttenuation(light, worldPos) * light.Intensity * exposure;
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
    finalColor += baseColor * 0.05;  // ambient

    return finalColor;
}

};  // namespace

void ShadingContext::Resolve(swr::Rasterizer& raster, swr::Framebuffer& fb) {
    glm::mat4 invProj = GetInverseScreenProjMatrix(WorldToClipMat, glm::ivec2(fb.Width, fb.Height));

    raster.DispatchPass(fb, [&](uint2 tileBasePos, v_int2 tilePos, v_float2 tileUV) {
        uint32_t tileOffset = fb.GetPixelOffset(tileBasePos.x, tileBasePos.y);
        v_float tileDepth = simd::load(&fb.GetDepthBuffer()[tileOffset]);
        v_int skyMask = tileDepth <= 0.0f;

        v_float4 ndcPos = v_float4(conv<float>(tilePos), skyMask ? 1.0f : tileDepth, 1.0f);
        v_float3 worldPos = v_float3(simd::perspective_div(simd::mul(invProj, ndcPos)));
        v_float3 finalColor = 0;

        uint64_t t1 = ShowPerfHeatmap ? __rdtsc() : 0;

        if (simd::any(~skyMask)) {
            SurfaceData surf = ResolveSurface(*this, &fb.GetColorBuffer()[tileOffset], simd::movemask(~skyMask), tileUV, float2(fb.Width, fb.Height));
            finalColor = EvalLighting(surf, Lights, NumLights, Exposure * 0.001f, worldPos, ViewPos);
        }
        if (simd::any(skyMask) && SkyboxTex != nullptr) {
            v_float3 skyColor = SkyboxTex->SampleOctLevel<EnvSampler>(worldPos - ViewPos, 1);
            cmov(finalColor, skyColor, skyMask);
        }
        finalColor = Tonemap_Unreal(finalColor * Exposure);

        if (ShowPerfHeatmap) {
            uint64_t t2 = __rdtsc();
            float d = (t2 - t1) / 1000.0;
            finalColor = finalColor * 0.3 + v_float3(d * 0.7, d * 0.8, d * 0.9);
        }
        v_uint packedColor = swr::pixfmt::RGBA8u::Pack({ finalColor, 1.0f });
        simd::store(&fb.GetColorBuffer()[tileOffset], packedColor);
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
        float radius = (glm::max(fb.Width, fb.Height) / 30.0f) / clipPos.w;

        float2 centerPos = screenPos * float2(fb.Width, fb.Height);
        int2 startTilePos = glm::max(int2(centerPos - radius), 0) & ~3;
        int2 endTilePos = glm::min(int2(centerPos + radius), int2(fb.Width, fb.Height));

        for (int y = startTilePos.y; y < endTilePos.y; y += 4) {
            uint32_t tileOffset = fb.GetPixelOffset(uint32_t(startTilePos.x), uint32_t(y));

            for (int x = startTilePos.x; x < endTilePos.x; x += 4, tileOffset += 16) {
                v_int depthMask = depth > simd::load(fb.GetDepthBuffer() + tileOffset);

                v_int2 tilePos = v_int2(x + swr::TilePixelOffsetsX, y + swr::TilePixelOffsetsY);
                v_float2 relPos = simd::conv<float>(tilePos) + 0.5f - centerPos;
                v_float distSq = simd::dot(relPos, relPos) - radius * radius;

                if (!simd::any(distSq < 0 && depthMask)) continue;

                uint32_t* colorData = fb.GetColorBuffer() + tileOffset;
                v_uint bgColor = simd::load(colorData);

                v_float a = 1 - (-distSq / (radius * radius));
                v_uint fgColor = swr::pixfmt::RGBA8u::Pack({ light.Color, 1 - a * a });
                v_uint finalColor = AlphaBlendU8(bgColor, fgColor);

                simd::store(colorData, depthMask ? finalColor : bgColor);
            }
        }
    }
}

void ShadingContext::ResolveDebug(swr::Rasterizer& raster, swr::Framebuffer& fb, DebugLayer layer) {
    raster.DispatchPass(fb, [&](uint2 tileBasePos, v_int2 tilePos, v_float2 tileUV) {
        uint32_t tileOffset = fb.GetPixelOffset(tileBasePos.x, tileBasePos.y);
        v_float tileDepth = simd::load(&fb.GetDepthBuffer()[tileOffset]);
        v_int skyMask = tileDepth <= 0.0f;

        uint32_t* tileData = &fb.GetColorBuffer()[tileOffset];
        SurfaceData data = ResolveSurface(*this, tileData, simd::movemask(~skyMask), tileUV, float2(fb.Width, fb.Height));

        v_float3 finalColor = 0.0f;
        if (layer == DebugLayer::BaseColor) {
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack(data.Albedo));
        } else if (layer == DebugLayer::Normals) {
            finalColor = data.Normal * 0.5 + 0.5;
        } else if (layer == DebugLayer::MetallicRoughness) {
            finalColor = v_float3(data.Metallic, data.Roughness, 0);
        } else if (layer == DebugLayer::MeshletId) {
            v_uint meshletId = simd::load(tileData) / swr::ShadedMeshlet::MaxPrims;
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack(meshletId * 123456789));
        } else if (layer == DebugLayer::TriangleId) {
            v_uint meshletId = simd::load(tileData);
            finalColor = v_float3(swr::pixfmt::RGBA8u::Unpack(meshletId * 123456789));
        }

        uint32_t backgroundRGB = ((tileBasePos.x ^ tileBasePos.y) & 4) ? 0xFF'A0A0A0 : 0xFF'FFFFFF;
        v_uint finalRGB = skyMask ? backgroundRGB : swr::pixfmt::RGBA8u::Pack({ finalColor, 1.0f });

        simd::store(&fb.GetColorBuffer()[tileOffset], finalRGB);
    });
}

const swr::ShaderDispatchTable &       //
    ShadingContext::VisBufferShader =  //
    swr::GetDispatchTable<&ShadeMeshlet,
                          &FS_EncodeSurfaceId<false>,  // 0
                          &FS_EncodeSurfaceId<true>    // 1
                          >(),
    ShadingContext::DeferredShader = swr::GetDispatchTable<&ShadeMeshlet, &FS_EncodeGBuffer>(),  //
    ShadingContext::OverdrawShader = swr::GetDispatchTable<&ShadeMeshlet, &FS_Overdraw>();       //
