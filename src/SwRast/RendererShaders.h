#pragma once

#include "SwRast.h"
#include <random>

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

        if (ShadowBuffer != nullptr && _mm512_cmp_ps_mask(NdotL, _mm512_set1_ps(0.0f), _CMP_GT_OQ)) [[unlikely]] {
            VFloat sx = vars.GetSmooth(9);
            VFloat sy = vars.GetSmooth(10);
            VFloat bias = max((1.0f - NdotL) * 0.015f, 0.003f);
            VFloat currentDepth = vars.GetSmooth(11) - bias;

            static const int8_t PoissonDisk[2][16] = {
                { -9, 85, -101, 2, -21, 36, 40, 86, -53, -69, 17, -87, 43, -8, -51, 70 },
                { -3, 17, -9, -43, 67, 88, -85, -34, 22, -64, 29, 48, -10, -99, -25, 63 },
            };
            auto samples = _mm_set1_epi8(0);

            VInt x = round2i(sx * (float)(ShadowBuffer->Width << 6) + 31);
            VInt y = round2i(sy * (float)(ShadowBuffer->Height << 6) + 31);

            for (uint32_t i = 0; i < 16; i++) {
                // If at the 4th iter samples are all 0 or 4, assume this area is not in a shadow edge
                if (i == 4 && (!_mm_cmpgt_epu8_mask(samples, _mm_set1_epi8(0)) || !_mm_cmplt_epu8_mask(samples, _mm_set1_epi8(4)))) {
                    diffuseLight *= conv2f(_mm512_cvtepi8_epi32(samples)) * (1.0 / 4);
                    goto EarlyOutExit;
                }
                VFloat occlusionDepth = ShadowBuffer->SampleDepth((x + PoissonDisk[0][i]) >> 6, (y + PoissonDisk[1][i]) >> 6);
                samples = _mm_mask_add_epi8(samples, _mm512_cmp_ps_mask(occlusionDepth, currentDepth, _CMP_GE_OQ), samples, _mm_set1_epi8(1));
            }
            diffuseLight *= conv2f(_mm512_cvtepi8_epi32(samples)) * (1.0 / 16);

        EarlyOutExit:;
        }
        diffuseLight += 0.3f;

        VFloat4 color = {
            diffuseColor.x * diffuseLight,
            diffuseColor.y * diffuseLight,
            diffuseColor.z * diffuseLight,
            diffuseColor.w,
        };

        auto rgba = PackRGBA(color);
        uint16_t alphaMask = _mm512_cmpgt_epu32_mask(rgba, _mm512_set1_epi32(200 << 24));
        fb.WriteTile(vars.TileOffset, vars.TileMask & alphaMask, rgba, vars.Depth);
    }

    // TODO: Move this to PBR shader if/when
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

struct EffectSSAO {
    static const uint32_t KernelSize = 32;
    float Radius = 0.5f;

    float Kernel[3][KernelSize];
    float Noise[2][16];

    EffectSSAO() {
        std::mt19937 prng(123456);
        std::uniform_real_distribution<float> unid;

        for (uint32_t i = 0; i < KernelSize; i++) {
            glm::vec3 s = {
                unid(prng) * 2.0f - 1.0f,
                unid(prng) * 2.0f - 1.0f,
                unid(prng),
            };
            s = glm::normalize(s) * unid(prng);  // Re-distrubute to hemisphere

            // Cluster samples closer to origin
            float scale = (float)i / KernelSize;
            s *= glm::lerp(0.1f, 1.0f, scale * scale);

            for (int32_t j = 0; j < 3; j++) {
                Kernel[j][i] = s[j];
            }
        }
        for (uint32_t i = 0; i < 16; i++) {
            glm::vec2 s = {
                unid(prng) * 2.0f - 1.0f,
                unid(prng) * 2.0f - 1.0f,
            };
            s = glm::normalize(s);

            Noise[0][i] = s.x;
            Noise[1][i] = s.y;
        }
    }

    void Generate(swr::Framebuffer& fb, const glm::mat4& projViewMat) {
        glm::mat4 invProj = glm::inverse(projViewMat);
        // Bias matrix so that input UVs can be in range [0..1] rather than [-1..1]
        invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
        invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

        bool altView = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000000000 % 2;

        for (uint32_t y = 0; y < fb.Height; y += 4) {
            for (uint32_t x = 0; x < fb.Width; x += 4) {
                uint32_t tileOffset = fb.GetPixelOffset(x, y);

                VFloat u = conv2f((int32_t)x + (VInt::ramp() & 3));
                VFloat v = conv2f((int32_t)y + (VInt::ramp() >> 2));
                VFloat z = VFloat::load(&fb.DepthBuffer[tileOffset]);

                VFloat4 pos = PerspectiveDiv(TransformVector(invProj, { u, v, z, 1.0f }));

                VFloat3 posDx = { dFdx(pos.x), dFdx(pos.y), dFdx(pos.z) };
                VFloat3 posDy = { dFdy(pos.x), dFdy(pos.y), dFdy(pos.z) };
                VFloat3 N = normalize(cross(posDx, posDy));

                VFloat3 rotation = { VFloat::load(Noise[0]), VFloat::load(Noise[1]), 0.0f };
                VFloat NdotR = dot(rotation, N);
                VFloat3 T = normalize({
                    rotation.x - N.x * NdotR,
                    rotation.y - N.y * NdotR,
                    rotation.z - N.z * NdotR,
                });
                VFloat3 B = cross(N, T);

                auto occlusion = _mm_set1_epi8(0);

                for (uint32_t i = 0; i < KernelSize; i++) {
                    VFloat sx = Kernel[0][i], sy = Kernel[1][i], sz = Kernel[2][i];
                    sx = T.x * sx + B.x * sy + N.x * sz;
                    sy = T.y * sx + B.y * sy + N.y * sz;
                    sz = T.z * sx + B.z * sy + N.z * sz;

                    sx = pos.x + sx * Radius;
                    sy = pos.y + sy * Radius;
                    sz = pos.z + sz * Radius;

                    VFloat4 samplePos = PerspectiveDiv(TransformVector(projViewMat, { sx, sy, sz, 1.0f }));
                    VFloat sampleDepth = fb.SampleDepth(samplePos.x * 0.5f + 0.5f, samplePos.y * 0.5f + 0.5f);
                    VFloat sampleDist = _mm512_abs_ps(LinearizeDepth(z) - LinearizeDepth(sampleDepth));

                    uint16_t rangeMask = _mm512_cmp_ps_mask(sampleDist, _mm512_set1_ps(Radius), _CMP_LT_OQ);
                    uint16_t occluMask = _mm512_cmp_ps_mask(sampleDepth, samplePos.z - 0.00005f, _CMP_LE_OQ);
                    occlusion = _mm_sub_epi8(occlusion, _mm_movm_epi8(occluMask & rangeMask));

                    //float rangeCheck = abs(origin.z - sampleDepth) < uRadius ? 1.0 : 0.0;
                    //occlusion += (sampleDepth <= sample.z ? 1.0 : 0.0) * rangeCheck;
                }
                VFloat occlusionFac = 1.0f - conv2f(_mm512_cvtepi8_epi32(occlusion)) * (1.0f / KernelSize);

                VFloat4 color = { occlusionFac, occlusionFac, occlusionFac, 1.0f };

                _mm512_store_epi32(&fb.ColorBuffer[tileOffset], PackRGBA(color));
            }
        }
    }
    __m128i ComputeOcclusion(VFloat u, VFloat v, VFloat z) {
        
    }
    static VFloat LinearizeDepth(VFloat d) {
        const float zNear = 0.05f, zFar = 1000.0f;
        VFloat z_n = d * 2.0f - 1.0f;
        return (2.0f * zNear * zFar) / ((zFar + zNear) - z_n * (zFar - zNear));
    }
};

inline void DrawSkybox(swr::Framebuffer& fb, const swr::HdrTexture2D& cubeMapTex, const glm::mat4& projMat, const glm::mat4& viewMat) {
    // Screen-space cubemap rendering, see https://gamedev.stackexchange.com/a/60377
    glm::mat4 invProj = glm::inverse(projMat * glm::mat4(viewMat[0], viewMat[1], viewMat[2], glm::vec4(0, 0, 0, 1)));
    // Bias matrix so that input UVs can be in range [0..1] rather than [-1..1]
    invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
    invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

    float minDepth = 1.0f;

    for (uint32_t y = 0; y < fb.Height; y += 4) {
        for (uint32_t x = 0; x < fb.Width; x += 4) {
            uint32_t tileOffset = fb.GetPixelOffset(x, y);
            VFloat tileDepth = VFloat::load(&fb.DepthBuffer[tileOffset]);
            uint16_t tileMask = _mm512_cmp_ps_mask(tileDepth, _mm512_set1_ps(minDepth), _CMP_GE_OQ);

            if (!tileMask) continue;

            VFloat u = conv2f((int32_t)x + (VInt::ramp() & 3));
            VFloat v = conv2f((int32_t)y + (VInt::ramp() >> 2));

            VFloat4 eyeDir = TransformVector(invProj, { u, v, 0.0f, 1.0f });

            VFloat faceU, faceV;
            VInt faceIdx;
            swr::HdrTexture2D::ProjectCubemap({ eyeDir.x, eyeDir.y, eyeDir.z }, faceU, faceV, faceIdx);
            VFloat3 color = cubeMapTex.SampleNearest(faceU, faceV, faceIdx);

            _mm512_mask_store_epi32(&fb.ColorBuffer[tileOffset], tileMask, PackRGBA({ color.x, color.y, color.z, 1.0f }));
        }
    }
}

}; // namespace renderer