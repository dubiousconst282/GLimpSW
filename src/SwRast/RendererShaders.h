#pragma once

#include "SwRast.h"
#include <random>

#include <execution>
#include <ranges>

namespace renderer {

using swr::VInt, swr::VFloat, swr::VFloat2, swr::VFloat3, swr::VFloat4, swr::VMask;
using namespace swr::simd;

// https://google.github.io/filament/Filament.html
struct DefaultShader {
    static const uint32_t NumCustomAttribs = 8;

    // Forward
    glm::mat4 ProjMat, ModelMat;
    const swr::Texture2D* BaseColorTex;
    const swr::Texture2D* NormalMetallicRoughnessTex;

    // Compose pass
    const swr::Framebuffer* ShadowBuffer;
    const swr::HdrTexture2D* SkyboxTex;
    glm::mat4 ShadowProjMat, ViewMat;
    glm::vec3 LightPos, ViewPos;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        VFloat3 pos = data.ReadAttribs<VFloat3>(&scene::Vertex::x);
        vars.Position = TransformVector(ProjMat, { pos, 1.0f });

        vars.SetAttribs(0, data.ReadAttribs<VFloat2>(&scene::Vertex::u));

        VFloat3 norm = data.ReadAttribs<VFloat3>(&scene::Vertex::nx);
        vars.SetAttribs(2, TransformNormal(ModelMat, norm));

        VFloat3 tang = data.ReadAttribs<VFloat3>(&scene::Vertex::tx);
        vars.SetAttribs(5, TransformNormal(ModelMat, tang));
    }

    void ShadePixels(swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        vars.ApplyPerspectiveCorrection();

        VFloat u = vars.GetSmooth(0);
        VFloat v = vars.GetSmooth(1);
        VFloat4 baseColor = BaseColorTex->SampleHybrid(u, v);

        VFloat3 N = normalize(vars.GetSmooth<VFloat3>(2));

        VFloat metalness = 0.0f;
        VFloat roughness = 0.5f;

        if (NormalMetallicRoughnessTex != nullptr) {
            // T = normalize(T - dot(T, N) * N);
            // mat3 TBN = mat3(T, cross(N, T), N);
            // vec3 n = texture(u_NormalMetallicRoughnessTex, UV).rgb * 2.0 - 1.0;
            // norm = normalize(TBN * n);
            VFloat3 T = vars.GetSmooth<VFloat3>(5);

            // Gram-schmidt process (produces higher-quality normal mapping on large meshes)
            // Re-orthogonalize T with respect to N
            T = normalize(T - dot(T, N) * N);
            VFloat3 B = cross(N, T);

            VFloat4 SN = NormalMetallicRoughnessTex->SampleHybrid(u, v);

            // Reconstruct Z from 2-channel normap map
            // https://aras-p.info/texts/CompactNormalStorage.html
            // https://www.researchgate.net/publication/259000109_Real-Time_Normal_Map_DXT_Compression
            // This isn't exact due to interpolation and mipmapping, but it's quite subtle after texturing anyway.
            VFloat Sx = SN.x * 2.0f - 1.0f;
            VFloat Sy = SN.y * 2.0f - 1.0f;
            VFloat Sz = sqrt14(1.0f - (Sx * Sx + Sy * Sy));

            N.x = T.x * Sx + B.x * Sy + N.x * Sz;
            N.y = T.y * Sx + B.y * Sy + N.y * Sz;
            N.z = T.z * Sx + B.z * Sy + N.z * Sz;

            metalness = SN.z;
            roughness = SN.w;
        }
        VMask mask = vars.TileMask & (baseColor.w > 0.9f);

        // GBuffer
        //   #1 [888: RGB]    [8: Metalness]
        //   #2 [888: Normal] [8: Roughness]
        baseColor.w = metalness;
        fb.WriteTile(vars.TileOffset, mask, PackRGBA(baseColor), vars.Depth);
        _mm512_mask_store_epi32(fb.GetAttachmentBuffer<uint32_t>(0, vars.TileOffset), mask, PackRGBA({ N * 0.5f + 0.5f, roughness }));
    }

    void Compose(swr::Framebuffer& fb, bool hasSSAO) {
        glm::mat4 invProj = glm::inverse(ProjMat);
        // Bias matrix to take UVs in range [0..screen] rather than [-1..1]
        invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
        invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

        float minDepth = 1.0f;

        for (uint32_t y = 0; y < fb.Height; y += 4) {
            for (uint32_t x = 0; x < fb.Width; x += 4) {
                uint32_t tileOffset = fb.GetPixelOffset(x, y);
                VFloat tileDepth = VFloat::load(&fb.DepthBuffer[tileOffset]);
                VMask skyMask = tileDepth >= minDepth;

                VFloat u = conv2f((int32_t)x + (VInt::ramp() & 3));
                VFloat v = conv2f((int32_t)y + (VInt::ramp() >> 2));
                VFloat3 worldPos = VFloat3(PerspectiveDiv(TransformVector(invProj, { u, v, tileDepth, 1.0f })));

                VFloat3 finalColor;

                if (!all(skyMask)) {
                    VFloat4 G1 = UnpackRGBA(VInt::load(&fb.ColorBuffer[tileOffset]));
                    VFloat4 G2 = UnpackRGBA(VInt::load(fb.GetAttachmentBuffer<uint32_t>(0, tileOffset)));

                    VFloat3 baseColor = VFloat3(G1);
                    VFloat3 N = VFloat3(G2) * 2.0f - 1.0f;
                    VFloat metalness = G1.w;
                    VFloat roughness = max(G2.w * G2.w, 0.045f);  // square to remap from perceptual roughness,  bias to prevent div by zero

                    // Renormalization helps reduce quantization artifacts (8-bits are not enough, maybe compress with Signed Octahedron?)
                    N = normalize(N);
                    baseColor = baseColor * baseColor;

                    VFloat3 VP = normalize(ViewPos - worldPos);
                    VFloat3 L = normalize(LightPos - worldPos);
                    VFloat3 H = normalize(VP + L);

                    VFloat NoV = max(dot(N, VP), 0.05f);
                    VFloat NoL = max(dot(N, L), 0.0f);
                    VFloat NoH = max(dot(N, H), 0.0f);
                    VFloat LoH = max(dot(L, H), 0.0f);

                    float reflectance = 0.5f;

                    VFloat D = D_GGX(NoH, roughness); // microfacet distribution
                    VFloat V = V_SmithGGXCorrelatedFast(NoV, NoL, roughness);  // microfacet visibility
                    VFloat3 F0 = (0.16f * reflectance * reflectance) * (1.0f - metalness) + baseColor * metalness;
                    VFloat3 F = F_Schlick(LoH, F0);
                    
                    // specular BRDF
                    VFloat3 Fr = D * V * F;

                    // diffuse BRDF
                    VFloat3 diffuseColor = (1.0f - metalness) * VFloat3(baseColor);
                    VFloat3 Fd = diffuseColor * Fd_Lambert();

                    finalColor = Fd + Fr;
                }

                if (any(skyMask)) {
                    // Render skybox using cubemap texture. https://gamedev.stackexchange.com/a/60377
                    VFloat faceU, faceV;
                    VInt faceIdx;
                    swr::HdrTexture2D::ProjectCubemap(worldPos - ViewPos, faceU, faceV, faceIdx);
                    VFloat3 skyColor = SkyboxTex->SampleNearest(faceU, faceV, faceIdx);

                    finalColor.x = csel(skyMask, skyColor.x, finalColor.x);
                    finalColor.y = csel(skyMask, skyColor.y, finalColor.y);
                    finalColor.z = csel(skyMask, skyColor.z, finalColor.z);
                }
                _mm512_store_epi32(&fb.ColorBuffer[tileOffset], PackRGBA({ finalColor, 1.0f }));
            }
        }
    }

    VFloat GetShadow(VFloat3 worldPos, VFloat NoL) {
        static const int8_t PoissonDisk[2][16] = {
            { -9, 85, -101, 2, -21, 36, 40, 86, -53, -69, 17, -87, 43, -8, -51, 70 },
            { -3, 17, -9, -43, 67, 88, -85, -34, 22, -64, 29, 48, -10, -99, -25, 63 },
        };

        VFloat4 shadowPos = TransformVector(ShadowProjMat, { worldPos, 1.0f });
        shadowPos.w = rcp14(shadowPos.w);

        VFloat bias = max((1.0f - NoL) * 0.015f, 0.003f);
        VFloat currentDepth = shadowPos.z * shadowPos.w - bias;

        VFloat sx = shadowPos.x * shadowPos.w * 0.5f + 0.5f;
        VFloat sy = shadowPos.y * shadowPos.w * 0.5f + 0.5f;
        VInt x = round2i(sx * (float)(ShadowBuffer->Width << 6)) + 31;
        VInt y = round2i(sy * (float)(ShadowBuffer->Height << 6)) + 31;

        auto samples = _mm_set1_epi8(0);
        float weight = 1.0f / 16;

        for (uint32_t i = 0; i < 16; i++) {
            // If at the 4th iter samples are all 0 or 4, assume this area is not in a shadow edge
            if (i == 4 && (!_mm_cmpgt_epu8_mask(samples, _mm_set1_epi8(0)) || !_mm_cmplt_epu8_mask(samples, _mm_set1_epi8(4)))) {
                weight = 1.0f / 4;
                break;
            }
            VFloat occlusionDepth = ShadowBuffer->SampleDepth((x + PoissonDisk[0][i]) >> 6, (y + PoissonDisk[1][i]) >> 6);
            samples = _mm_mask_add_epi8(samples, _mm512_cmp_ps_mask(occlusionDepth, currentDepth, _CMP_GE_OQ), samples, _mm_set1_epi8(1));
        }
        return conv2f(_mm512_cvtepi8_epi32(samples)) * weight;
    }

    static VFloat D_GGX(VFloat NoH, VFloat roughness) {
        VFloat a = NoH * roughness;
        VFloat k = roughness / (1.0f - NoH * NoH + a * a);
        return k * k * (1.0f / pi);
    }
    static VFloat V_SmithGGXCorrelatedFast(VFloat NoV, VFloat NoL, VFloat roughness) {
        // Hammon 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"
        //float v = 0.5 / mix(2.0 * NoL * NoV, NoL + NoV, roughness);
        VFloat a = 2.0f * NoL * NoV;
        VFloat b = NoL + NoV;
        return 0.5f / (a + (b - a) * roughness);
    }
    static VFloat3 F_Schlick(VFloat u, VFloat3 f0) {
        //VFloat f = pow(1.0 - u, 5.0);
        //return f + f0 * (1.0 - f);
        VFloat f = 1.0 - u;
        VFloat f2 = f * f;
        f = f2 * f2 * f;
        return f + f0 * (1.0f - f);
    }
    static VFloat Fd_Lambert() { return 1.0f / pi; }
};
struct DepthOnlyShader {
    static const uint32_t NumCustomAttribs = 0;

    glm::mat4 ProjMat;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        VFloat4 pos = { data.ReadAttribs<VFloat3>(&scene::Vertex::x), 1.0f };
        vars.Position = TransformVector(ProjMat, pos);
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        _mm512_mask_storeu_ps(&fb.DepthBuffer[vars.TileOffset], vars.TileMask, vars.Depth);
    }
};

struct SSAO {
    static const uint32_t KernelSize = 16, FbAttachId = 4;

    float Radius = 0.75f, MaxRange = 0.25f;

    float Kernel[3][KernelSize];
    VInt _randSeed;

    SSAO() {
        std::mt19937 prng(123453);
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

        for (uint32_t i = 0; i < 16; i++) {
            _randSeed[i] = (int32_t)prng();
        }
    }

    void Generate(swr::Framebuffer& fb, const scene::DepthPyramid& depthMap, glm::mat4& projViewMat) {
        glm::mat4 invProj = glm::inverse(projViewMat);
        // Bias matrix so that input UVs can be in range [0..1] rather than [-1..1]
        invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
        invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

        uint32_t stride = fb.Width / 2;
        uint8_t* aoBuffer = fb.GetAttachmentBuffer<uint8_t>(FbAttachId);

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
                    VFloat sampleDepth = LinearizeDepth(depthMap.SampleDepth(samplePos.x * 0.5f + 0.5f, samplePos.y * 0.5f + 0.5f, 0));
                    // FIXME: range check kinda breaks when the camera gets close to geom
                    //        depth linearization might also be wrong (cam close ups)
                    VFloat sampleDist = abs(LinearizeDepth(z) - sampleDepth);

                    uint16_t rangeMask = _mm512_cmp_ps_mask(sampleDist, _mm512_set1_ps(MaxRange), _CMP_LT_OQ);
                    uint16_t occluMask = _mm512_cmp_ps_mask(sampleDepth, LinearizeDepth(samplePos.z) - 0.02f, _CMP_LE_OQ);
                    occlusion = _mm_sub_epi8(occlusion, _mm_movm_epi8(occluMask & rangeMask));

                    // float rangeCheck = abs(origin.z - sampleDepth) < uRadius ? 1.0 : 0.0;
                    // occlusion += (sampleDepth <= sample.z ? 1.0 : 0.0) * rangeCheck;
                }
                occlusion = _mm_slli_epi16(occlusion, 9 - std::bit_width(KernelSize));
                occlusion = _mm_sub_epi8(_mm_set1_epi8((char)255), occlusion);

                // pow(occlusion, 3)
                __m256i o1 = _mm256_cvtepu8_epi16(occlusion);
                __m256i o2 = _mm256_srli_epi16(_mm256_mullo_epi16(o1, o1), 8);
                __m256i o3 = _mm256_srli_epi16(_mm256_mullo_epi16(o1, o2), 8);
                occlusion = _mm256_cvtepi16_epi8(o3);

                for (uint32_t sy = 0; sy < 4; sy++) {
                    std::memcpy(&aoBuffer[(x / 2) + (y / 2 + sy) * stride], (uint32_t*)&occlusion + sy, 4);
                }
            }
        });
        ApplyBlur(fb);
    }

    static VFloat SampleTileAO(swr::Framebuffer& fb, uint32_t x, uint32_t y) {
        VFloat res = 0;
        uint8_t* basePtr = fb.GetAttachmentBuffer<uint8_t>(FbAttachId);
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
        uint8_t* aoBuffer = fb.GetAttachmentBuffer<uint8_t>(FbAttachId);
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
        __m256i c = _mm512_cvtepi16_epi8(_mm512_mulhrs_epi16(accum, _mm512_set1_epi16(32767 / BlurSamples)));
        _mm256_storeu_epi8(dst, c);
    }

    static VFloat LinearizeDepth(VFloat d) {
        const float zNear = 0.05f, zFar = 1000.0f;
        return (zNear * zFar) / (zFar + d * (zNear - zFar));
    }

    void XorShiftStep(VInt& x) {
        x = x ^ x << 13;
        x = x ^ x >> 17;
        x = x ^ x << 5;
    }
};

}; // namespace renderer