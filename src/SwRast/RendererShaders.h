#pragma once

#include "SwRast.h"
#include <random>

#include <execution>
#include <ranges>

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

        VFloat4 color = {
            diffuseColor.x,
            diffuseColor.y,
            diffuseColor.z,
            diffuseLight,
        };

        auto rgba = PackRGBA(color);
        uint16_t alphaMask = _mm512_cmp_ps_mask(diffuseColor.w, _mm512_set1_ps(0.9f), _CMP_GT_OQ);
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

struct SSAO {
    static const uint32_t KernelSize = 16;
    float Radius = 0.75f, MaxRange = 0.5f;

    float Kernel[3][KernelSize];
    VInt _randSeed;

    SSAO() {
        std::mt19937 prng(123456);
        std::uniform_real_distribution<float> unid;

        // TODO: could probably make better use of samples with poisson disk instead 
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

        for (uint32_t i = 0; i < 16; i++)
        {
            _randSeed[i] = (int32_t)prng();
        }
    }

    void Generate(swr::Framebuffer& fb, const scene::DepthPyramid& depthMap, glm::mat4& projViewMat) {
        glm::mat4 invProj = glm::inverse(projViewMat);
        // Bias matrix so that input UVs can be in range [0..1] rather than [-1..1]
        invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
        invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

        uint32_t stride = fb.Width / 2;
        uint8_t* aoBuffer = fb.GetAttachmentBuffer(0, 1);

        const int BlurRadius = 3, BlurSamples = BlurRadius * 2 + 1;

        auto range = std::ranges::iota_view(0u, (fb.Height + 7) / 8);

        std::for_each(std::execution::par_unseq, range.begin(), range.end(), [&](uint32_t y_) {
            VInt rng = _randSeed ^ (int32_t)(y_ * 12345);
            uint32_t y = y_ * 8;

            for (uint32_t x = 0; x < fb.Width; x += 8) {
                VInt iu = (int32_t)x + (VInt::ramp() & 3) * 2;
                VInt iv = (int32_t)y + (VInt::ramp() >> 2) * 2;
                VFloat z = depthMap.SampleDepth(iu, iv, 0);

                if (!_mm512_cmp_ps_mask(z, _mm512_set1_ps(1.0f), _CMP_LT_OQ)) continue;

                VFloat4 pos = PerspectiveDiv(TransformVector(invProj, { conv2f(iu), conv2f(iv), z, 1.0f }));

                // TODO: better normal reconstruction - https://atyuwen.github.io/posts/normal-reconstruction/
                VFloat3 posDx = { dFdx(pos.x), dFdx(pos.y), dFdx(pos.z) };
                VFloat3 posDy = { dFdy(pos.x), dFdy(pos.y), dFdy(pos.z) };
                VFloat3 N = normalize(cross(posDx, posDy));

                // Tiled random texture causes some quite ugly artifacts, xorshift is cheap enough
                XorShiftStep(rng);
                VFloat3 rotation = normalize({ conv2f(rng & 255) * (1.0f / 127) - 1.0f, conv2f(rng >> 8 & 255) * (1.0f / 127) - 1.0f,0.0f });
               
                VFloat NdotR = dot(rotation, N);
                VFloat3 T = normalize({
                    rotation.x - N.x * NdotR,
                    rotation.y - N.y * NdotR,
                    rotation.z - N.z * NdotR,
                });
                VFloat3 B = cross(N, T);

                auto occlusion = _mm_set1_epi8(0);

                for (uint32_t i = 0; i < KernelSize; i++) {
                    VFloat kx = Kernel[0][i], ky = Kernel[1][i], kz = Kernel[2][i];

                    VFloat sx = (T.x * kx + B.x * ky + N.x * kz) * Radius + pos.x;
                    VFloat sy = (T.y * kx + B.y * ky + N.y * kz) * Radius + pos.y;
                    VFloat sz = (T.z * kx + B.z * ky + N.z * kz) * Radius + pos.z;

                    VFloat4 samplePos = PerspectiveDiv(TransformVector(projViewMat, { sx, sy, sz, 1.0f }));
                    VFloat sampleDepth = depthMap.SampleDepth(samplePos.x * 0.5f + 0.5f, samplePos.y * 0.5f + 0.5f, 0);
                    // FIXME: range check kinda breaks when the camera gets close to geom
                    VFloat sampleDist = _mm512_abs_ps(LinearizeDepth(z) - LinearizeDepth(sampleDepth));

                    uint16_t rangeMask = _mm512_cmp_ps_mask(sampleDist, _mm512_set1_ps(MaxRange), _CMP_LT_OQ);
                    uint16_t occluMask = _mm512_cmp_ps_mask(sampleDepth, samplePos.z - 0.0005f, _CMP_LE_OQ);
                    occlusion = _mm_sub_epi8(occlusion, _mm_movm_epi8(occluMask & rangeMask));

                    //float rangeCheck = abs(origin.z - sampleDepth) < uRadius ? 1.0 : 0.0;
                    //occlusion += (sampleDepth <= sample.z ? 1.0 : 0.0) * rangeCheck;
                }
                occlusion = _mm_slli_epi16(occlusion, 9 - std::bit_width(KernelSize));
                occlusion = _mm_sub_epi8(_mm_set1_epi8((char)255), occlusion);
                __m256i tmp = _mm256_cvtepu8_epi16(occlusion);
                occlusion = _mm256_cvtepi16_epi8(_mm256_srai_epi16(_mm256_mullo_epi16(tmp, tmp), 8));

                for (uint32_t sy = 0; sy < 4; sy++) {
                    std::memcpy(&aoBuffer[(x / 2) + (y / 2 + sy) * stride], (uint32_t*)&occlusion + sy, 4);
                }
            }
        });
        ApplyBlur(fb);
    }

    static VFloat SampleTileAO(swr::Framebuffer& fb, uint32_t x, uint32_t y) {
        VFloat res = 0;
        uint8_t* basePtr = fb.GetAttachmentBuffer(0, 1);
        uint32_t stride = fb.Width / 2;

        for (uint32_t ty = 0; ty < 4; ty++) {
            for (uint32_t tx = 0; tx < 4; tx++) {
                uint8_t ao = basePtr[((x + tx) / 2) + ((y + ty) / 2) * stride];
                res[tx + ty * 4] = ao * (1.0f / 255);
            }
        }
        return res;

        //__m128i rowOffs = _mm_add_epi32(_mm_set1_epi32(y), _mm_setr_epi32(0, 1, 2, 3));
        //__m128i indices = _mm_mullo_epi32(_mm_srli_epi32(rowOffs, 1), _mm_set1_epi32(stride));
        //__m128i values = _mm_i32gather_epi32(basePtr + (x / 2), indices, 1);
        //return 1.0f - conv2f(_mm512_cvtepu8_epi32(values)) / 255.0f;
    }

private:
    static void ApplyBlur(swr::Framebuffer& fb) {
        uint8_t* aoBuffer = fb.GetAttachmentBuffer(0, 1);
        uint32_t stride = fb.Width / 2;
        // Box blur
        // No clamping of course, as long as it doesn't crash it's fine :)
        // Buffer size is actually W*H, but we use half of that - it's really okay.
        uint32_t altBufferOffset = (fb.Height / 2) * stride;

        for (uint32_t y = 0; y < fb.Height / 2; y++) {
            for (uint32_t x = 0; x < fb.Width / 2; x += 32) {
                uint8_t* src = &aoBuffer[x + y * stride];
                BlurX32(src + altBufferOffset, src, 1);
            }
        }

        for (uint32_t y = 0; y < fb.Height / 2; y++) {
            for (uint32_t x = 0; x < fb.Width / 2; x += 32) {
                uint8_t* src = &aoBuffer[x + y * stride];
                BlurX32(src, src + altBufferOffset, (int32_t)stride);
            }
        }
    }
    static void BlurX32(uint8_t* dst, uint8_t* src, int32_t lineStride) {
        const int BlurRadius = 3, BlurSamples = BlurRadius * 2 + 1;
        __m512i accum = _mm512_set1_epi16(0);

        for (int32_t so = -BlurRadius; so <= BlurRadius; so++) {
            accum = _mm512_add_epi16(accum, _mm512_cvtepu8_epi16(_mm256_loadu_epi8(&src[so * lineStride])));
        }
        __m256i c = _mm512_cvtepi16_epi8(_mm512_mulhrs_epi16(accum, _mm512_set1_epi16(32768 / BlurSamples)));
        _mm256_storeu_epi8(dst, c);
    }

    static VFloat LinearizeDepth(VFloat d) {
        const float zNear = 0.05f, zFar = 1000.0f;
        VFloat z_n = d * 2.0f - 1.0f;
        return (2.0f * zNear * zFar) / ((zFar + zNear) - z_n * (zFar - zNear));
    }

    void XorShiftStep(VInt& x) {
        x = x ^ x << 13;
        x = x ^ x >> 17;
        x = x ^ x << 5;
    }
};

inline void ComposeDeferred(swr::Framebuffer& fb, const swr::HdrTexture2D& cubeMapTex, const glm::mat4& projMat, const glm::mat4& viewMat, bool hasSSAO) {
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

            VInt color;

            if (tileMask != 0xFFFF) {
                VFloat ao =  hasSSAO ? SSAO::SampleTileAO(fb, x, y) : 1.0f;
                color = _mm512_load_epi32(&fb.ColorBuffer[tileOffset]);

                VFloat diffuseLight = ao * (conv2f(shrl(color, 24))*(1.0f/255) + 0.3f);

                VInt lightFac = round2i(diffuseLight * 255.0f);

                // Rescale to 1.15 and replicate 16-bits to 32
                lightFac = min(max(lightFac, 0), 255) << (15 - 8);
                lightFac = lightFac | lightFac << 16;

                VInt rb = (color >> 0) & 0x00FF'00FF;
                VInt ga = (color >> 8) & 0x00FF'00FF;

                // mulhrs does round(a*b) in 0.15-bit fixed point
                // TODO: this limits us to a pretty bad range, maybe just give up and use floats like a sane person
                rb = _mm512_mulhrs_epi16(rb, lightFac);
                ga = _mm512_mulhrs_epi16(ga, lightFac);

                color = rb << 0 | ga << 8 | (int32_t)0xFF000000u;
            }

            if (tileMask) {
                VFloat u = conv2f((int32_t)x + (VInt::ramp() & 3));
                VFloat v = conv2f((int32_t)y + (VInt::ramp() >> 2));

                VFloat4 eyeDir = TransformVector(invProj, { u, v, 0.0f, 1.0f });

                VFloat faceU, faceV;
                VInt faceIdx;
                swr::HdrTexture2D::ProjectCubemap({ eyeDir.x, eyeDir.y, eyeDir.z }, faceU, faceV, faceIdx);
                VFloat3 skyColor = cubeMapTex.SampleNearest(faceU, faceV, faceIdx);

                color = _mm512_mask_mov_epi32(color, tileMask, PackRGBA({ skyColor.x, skyColor.y, skyColor.z, 1.0f }));
            }

            _mm512_store_epi32(&fb.ColorBuffer[tileOffset], color);
        }
    }
}

}; // namespace renderer