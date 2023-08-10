#pragma once

#include "SwRast.h"

namespace renderer {

using swr::VInt, swr::VFloat, swr::VFloat3, swr::VFloat4;
using namespace swr::simd;

struct PhongShader {
    static const uint32_t NumCustomAttribs = 12;

    glm::mat4 ProjMat, ModelMat;
    glm::vec3 LightPos;
    glm::mat4 ShadowProjMat;
    const swr::Framebuffer* ShadowBuffer;
    const swr::Texture2D* DiffuseTex;
    const swr::Texture2D* NormalTex;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        VFloat4 pos = { .w = 1.0f };
        data.ReadAttribs(&scene::Vertex::x, &pos.x, 3);
        vars.Position = TransformVector(ProjMat, pos);

        data.ReadAttribs(&scene::Vertex::u, &vars.Attribs[0], 2);
        data.ReadAttribs(&scene::Vertex::nx, &vars.Attribs[2], 3);
        data.ReadAttribs(&scene::Vertex::tx, &vars.Attribs[5], 3);

        pos = TransformVector(ModelMat, pos);
        vars.Attribs[6] = pos.x;
        vars.Attribs[7] = pos.y;
        vars.Attribs[8] = pos.z;

        if (ShadowBuffer != nullptr) {
            VFloat4 shadowPos = TransformVector(ShadowProjMat, pos);
            VFloat shadowRcpW = _mm512_rcp14_ps(shadowPos.w);

            vars.Attribs[9] = shadowPos.x * shadowRcpW * 0.5f + 0.5f;
            vars.Attribs[10] = shadowPos.y * shadowRcpW * 0.5f + 0.5f;
            vars.Attribs[11] = shadowPos.z * shadowRcpW;
        }
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        vars.ApplyPerspectiveCorrection();

        VFloat u = vars.GetSmooth(0);
        VFloat v = vars.GetSmooth(1);
        VFloat4 diffuseColor = DiffuseTex->SampleHybrid(u, v);

        VFloat3 N = normalize({ vars.GetSmooth(2), vars.GetSmooth(3), vars.GetSmooth(4) });

        if (NormalTex != nullptr) {
            // T = normalize(T - dot(T, N) * N);
            // mat3 TBN = mat3(T, cross(N, T), N);
            // vec3 n = texture(u_NormalTex, UV).rgb * 2.0 - 1.0;
            // norm = normalize(TBN * n);
            VFloat3 T = { vars.GetSmooth(5), vars.GetSmooth(6), vars.GetSmooth(7) };

            // Gram-schmidt process (produces higher-quality normal mapping on large meshes)
            // Re-orthogonalize T with respect to N
            VFloat TN = dot(T, N);
            T = normalize({ T.x - TN * N.x, T.y - TN * N.y, T.z - TN * N.z });
            VFloat3 B = cross(N, T);

            VFloat4 SN = NormalTex->SampleHybrid(u, v);

            VFloat Sx = SN.x * 2.0f - 1.0f;
            VFloat Sy = SN.y * 2.0f - 1.0f;
            VFloat Sz = SN.z * 2.0f - 1.0f;
            // TODO: use GA channels for PBR roughness/metalness
            // https://aras-p.info/texts/CompactNormalStorage.html#q-comparison
            // VFloat Sz = sqrt14(1.0f - Sx * Sx + Sy * Sy);  // sqrt(1.0f - dot(n.xy, n.xy));

            N.x = T.x * Sx + B.x * Sy + N.x * Sz;
            N.y = T.y * Sx + B.y * Sy + N.y * Sz;
            N.z = T.z * Sx + B.z * Sy + N.z * Sz;
        }

        VFloat3 lightDir = normalize({
            LightPos.x - vars.GetSmooth(6),
            LightPos.y - vars.GetSmooth(7),
            LightPos.z - vars.GetSmooth(8),
        });
        VFloat NdotL = dot(N, lightDir);
        VFloat diffuseLight = max(NdotL, 0.0f);
        VFloat shadowLight = 1.0f;

        if (ShadowBuffer != nullptr) [[unlikely]] {
            VFloat sx = vars.GetSmooth(9);
            VFloat sy = vars.GetSmooth(10);
            VFloat bias = max((1.0f - NdotL) * 0.015f, 0.003f);
            VFloat currentDepth = vars.GetSmooth(11) - bias;
            VFloat closestDepth = ShadowBuffer->SampleDepth(sx, sy);

            // closestDepth > pos.z ? 1.0 : 0.0
            shadowLight = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(closestDepth, currentDepth, _CMP_GE_OQ), shadowLight);
        }
        VFloat combinedLight = diffuseLight * shadowLight + 0.3f;

        VFloat4 color = {
            diffuseColor.x * combinedLight,
            diffuseColor.y * combinedLight,
            diffuseColor.z * combinedLight,
            diffuseColor.w,
        };

        auto rgba = PackRGBA(color);
        uint16_t alphaMask = _mm512_cmpgt_epu32_mask(rgba, _mm512_set1_epi32(200 << 24));
        fb.WriteTile(vars.TileOffset, vars.TileMask & alphaMask, rgba, vars.Depth);
    }

    static VFloat DistributionGGX(VFloat3 N, VFloat3 H, VFloat roughness) {
        VFloat a = roughness * roughness;
        VFloat a2 = a * a;
        VFloat NdotH = max(dot(N, H), 0.0);
        VFloat NdotH2 = NdotH * NdotH;

        VFloat num = a2;
        VFloat denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = pi * denom * denom;

        return num / denom;
    }
    static VFloat GeometrySchlickGGX(VFloat NdotV, VFloat roughness) {
        VFloat r = (roughness + 1.0f);
        VFloat k = (r * r) / 8.0f;

        VFloat num = NdotV;
        VFloat denom = NdotV * (1.0f - k) + k;

        return num / denom;
    }
    static VFloat GeometrySmith(VFloat3 N, VFloat3 V, VFloat3 L, VFloat roughness) {
        VFloat NdotV = max(dot(N, V), 0.0f);
        VFloat NdotL = max(dot(N, L), 0.0f);
        return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
    }
};
struct DepthOnlyShader {
    static const uint32_t NumCustomAttribs = 0;

    glm::mat4 ProjMat;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        VFloat4 pos = { .w = 1.0f };
        data.ReadAttribs(&scene::Vertex::x, &pos.x, 3);
        vars.Position = TransformVector(ProjMat, pos);
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        _mm512_mask_storeu_ps(&fb.DepthBuffer[vars.TileOffset], vars.TileMask, vars.Depth);
    }
};

inline void DrawSkybox(swr::Framebuffer& fb, const swr::HdrTexture2D& texture, const glm::mat4& projMat, const glm::mat4& viewMat) {
    // Screen-space cubemap rendering, see https://gamedev.stackexchange.com/a/60377
    glm::mat4 invProj = glm::inverse(projMat * glm::mat4(viewMat[0], viewMat[1], viewMat[2], glm::vec4(0, 0, 0, 1)));

    float minDepth = 1.0f;

    VFloat du = 2.0f / fb.Width;
    VFloat dv = 2.0f / fb.Height;

    for (uint32_t y = 0; y < fb.Height; y += 4) {
        for (uint32_t x = 0; x < fb.Width; x += 4) {
            uint32_t tileOffset = fb.GetPixelOffset(x, y);
            VFloat tileDepth = VFloat::load(&fb.DepthBuffer[tileOffset]);
            uint16_t tileMask = _mm512_cmp_ps_mask(tileDepth, _mm512_set1_ps(minDepth), _CMP_GE_OQ);

            if (!tileMask) continue;

            VFloat u = conv2f((int32_t)x + (VInt::ramp() & 3)) * du - 1.0f;
            VFloat v = conv2f((int32_t)y + (VInt::ramp() >> 2)) * dv - 1.0f;

            VFloat4 eyeDir = TransformVector(invProj, { u, v, 0.0f, 1.0f });
            VFloat rcpW = _mm512_rcp14_ps(eyeDir.w);

            VFloat3 eyeDirN = normalize({ eyeDir.x * rcpW, eyeDir.y * rcpW, eyeDir.z * rcpW });

            VFloat kU, kV;

            for (size_t i = 0; i < 16; i++) {
                kU[i] = atan2f(eyeDirN.z[i], eyeDirN.x[i]) / (pi * 2) + 0.5f;
                kV[i] = asinf(-eyeDirN.y[i]) / pi + 0.5f;
            }

            VFloat3 color = texture.SampleNearest(kU, kV, 0);
            _mm512_mask_store_epi32(&fb.ColorBuffer[tileOffset], tileMask, PackRGBA({ color.x, color.y, color.z, 1.0f }));
        }
    }
}

}; // namespace renderer