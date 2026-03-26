#pragma once

#include "SwRast.h"
#include "Texture.h"
#include "Scene.h"
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace renderer {

using swr::v_int, swr::v_float, swr::v_float2, swr::v_float3, swr::v_float4, swr::v_mask;
using namespace swr::simd;

enum class DebugLayer { None, BaseColor, Normals, MetallicRoughness, Occlusion, EmissiveMask, Overdraw, TexCacheHeatmap };

// Deferred PBR shader
// https://google.github.io/filament/Filament.html
// https://bruop.github.io/ibl/
struct DefaultShader {
    static constexpr uint32_t NumCustomAttribs = 8, NumFbAttachments = 9;

    static constexpr swr::SamplerDesc SurfaceSampler = {
        .Wrap = swr::WrapMode::Repeat,
        .MagFilter = swr::FilterMode::Linear,
        .MinFilter = swr::FilterMode::Nearest,  // Downsample with nearest for better perf, quality loss is quite subtle.
        .EnableMips = true,
    };
    static constexpr swr::SamplerDesc EnvSampler = {
        .Wrap = swr::WrapMode::ClampToEdge,
        .MagFilter = swr::FilterMode::Linear,
        .MinFilter = swr::FilterMode::Linear,
        .EnableMips = true,
    };

    // Owned
    swr::HdrTexture2D::Ptr SkyboxTex;
    swr::HdrTexture2D::Ptr IrradianceMap, FilteredEnvMap;
    swr::TexturePtr2D<swr::pixfmt::RG16f> BRDFEnvLut = GenerateBRDFEnvLut();

    // TAA
    uint32_t FrameNo;
    glm::vec2 Jitter, PrevJitter;
    glm::mat4 PrevProjMat;
    std::unique_ptr<swr::Framebuffer> PrevFrame;
    swr::AlignedBuffer<uint32_t> ResolvedTAABuffer;

    // Uniform: Forward
    glm::mat4 ProjMat, ModelMat;
    const scene::Meshlet* Meshlets;
    const scene::Material* Materials;

    // Uniform: Compose pass
    const swr::Framebuffer* ShadowBuffer;
    glm::mat4 ShadowProjMat, ViewMat;
    glm::vec3 LightPos, ViewPos;

    float IntensityIBL = 0.3f;
    float Exposure = 1.0f;
    bool EnableTAA = false;
    bool BlurSkybox = false;

    void ShadeMeshlet(uint32_t index, swr::ShadedMeshlet& output) const {
        auto& mesh = Meshlets[index];

        // output.PrimCount = 0;
        // if (glm::dot(glm::normalize(mesh.ConeApex - ViewPos), mesh.ConeAxis) >= mesh.ConeCutoff) return;

        for (uint32_t i = 0; i < mesh.NumVertices; i += vec_width) {
            v_float3 worldPos = { load(&mesh.Positions[0][i]), load(&mesh.Positions[1][i]), load(&mesh.Positions[2][i]) };
            output.SetPosition(i, TransformVector(ProjMat, { worldPos, 1.0f }));
        }
        output.PrimCount = mesh.NumTriangles;
        memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));
    }

    static glm::vec2 UnpackHalf2x16(uint32_t x) {
        // GLM emulates this with 100 instructions.
        auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
        return { v[0], v[1] };
    }

    void ShadePixels(swr::Framebuffer& fb, swr::FragmentVars& vars) const {
        // Depth test
        v_float oldDepth = load(&fb.DepthBuffer[vars.TileOffset]);
        vars.TileMask &= movemask(vars.Depth > oldDepth);

        if (vars.TileMask == 0) return;

        const scene::Meshlet& mesh = Meshlets[vars.MeshletId];
        v_int color;

        if (mesh.MaterialId != UINT_MAX) {
            const scene::Material& material = Materials[mesh.MaterialId];

            v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[0]]);
            v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[1]]);
            v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[2]]);
            v_float2 uv = vars.Interpolate(uv0, uv1, uv2);

            color = material.Texture->Sample<SurfaceSampler>(uv.x, uv.y);
        } else {
            color = swr::pixfmt::RGBA8u::Pack({ vars.Bary, 1 });
        }
        //color = 255<<24|(vars.MeshletId*1234567);

        fb.WriteTile(vars.TileOffset, vars.TileMask, color, vars.Depth);
    }

        glm::mat4 invProj = glm::inverse(ProjMat);
        // Bias matrix to take UVs in range [0..screen] rather than [-1..1]
        invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
        invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

        fb.IterateTiles([&](uint32_t x, uint32_t y) {
            uint32_t tileOffset = fb.GetPixelOffset(x, y);
            v_float tileDepth = load(&fb.DepthBuffer[tileOffset]);
            v_int skyMask = tileDepth <= 0.0f;

            v_int tileX = (int32_t)x + FragPixelOffsetsX;
            v_int tileY = (int32_t)y + FragPixelOffsetsY;
            v_float3 worldPos = v_float3(PerspectiveDiv(TransformVector(invProj, { conv2f(tileX), conv2f(tileY), skyMask ? 0.5f : tileDepth, 1.0f })));

            v_float3 finalColor;

            if (!all(skyMask)) {
                // Unpack G-Buffer
                v_float4 G1 = swr::pixfmt::RGBA8u::Unpack(load(&fb.ColorBuffer[tileOffset]));
                v_int G2r = load(fb.GetAttachmentBuffer<uint32_t>(0, tileOffset));
                v_int emissiveMask = (G2r & 1) != 0;
                v_float3 N = SignedOctDecode(G2r);

                v_float3 baseColor = SrgbToLinear(v_float3(G1));
                v_float metalness = G1.w;

                v_float perceptualRoughness = max(conv2f(G2r >> 1 & 63) / 63.0f, 0.045f);  // clamp to prevent div by zero
                v_float roughness = perceptualRoughness * perceptualRoughness;

                // Microfacet BRDF
                v_float3 V = normalize(0 - worldPos);
                v_float3 L = normalize(LightPos);
                v_float3 H = normalize(V + L);

                v_float NoV = max(dot(N, V), 1e-4f);
                v_float NoL = max(dot(N, L), 0.0f);
                v_float NoH = max(dot(N, H), 0.0f);
                v_float LoH = max(dot(L, H), 0.0f);

                float reflectance = 0.5f;

                v_float D = D_GGX(NoH, roughness);
                v_float G = V_SmithGGXCorrelatedFast(NoV, NoL, roughness);
                v_float3 f0 = (0.16f * reflectance * reflectance) * (1.0f - metalness) + baseColor * metalness;
                v_float3 F = F_Schlick(LoH, f0);

                // specular
                v_float3 Fr = D * G * F;

                // diffuse
                v_float3 diffuseColor = (1.0f - metalness) * v_float3(baseColor);
                v_float3 Fd = (1.0f - F) * diffuseColor * Fd_Lambert();

                if (ShadowBuffer != nullptr && any(NoL > 0.0f)) {
                    [[clang::always_inline]] NoL *= GetShadow(worldPos, NoL);
                }
                finalColor = (Fd + Fr) * NoL;

                if (IntensityIBL > 0.0f && SkyboxTex != nullptr) {
                    v_float3 R = reflect(0 - V, N);
                    v_float3 irradiance = IrradianceMap->SampleCube<EnvSampler>(R);
                    v_float3 radiance = FilteredEnvMap->SampleCube<EnvSampler>(R, perceptualRoughness * (int32_t)FilteredEnvMap->MipLevels);
                    v_float2 dfg = BRDFEnvLut->Sample<EnvSampler>(NoV, perceptualRoughness);
                    v_float3 specularColor = f0 * dfg.x + dfg.y;
                    v_float3 ibl = irradiance * diffuseColor + radiance * specularColor;

                    // TODO: multi-scatter IBL

                    //if (hasSSAO) {
                    //    ibl = ibl * GetAO(fb, x, y);
                    //}
                    finalColor = finalColor + ibl * IntensityIBL;
                }

                // This is technically not PBR but if we add AO to IBL it will never be intense enough.
                // (though it's not like anything here is strictly correct anyway...)
                if (hasSSAO) {
                    finalColor = finalColor * GetAO(fb, x, y);
                }

                // Emissive color
                if (any(emissiveMask)) {
                    v_int G3r = load(fb.GetAttachmentBuffer<uint32_t>(4, tileOffset));
                    G3r = emissiveMask ? G3r : 0;

                    v_float3 emissiveColor = v_float3(swr::pixfmt::RGBA8u::Unpack(G3r));
                    finalColor = finalColor + SrgbToLinear(emissiveColor);
                }
            }

            if (any(skyMask) && SkyboxTex != nullptr) {
                auto& envTex = BlurSkybox ? FilteredEnvMap : SkyboxTex;
                v_float3 skyColor = envTex->SampleCube<EnvSampler>(worldPos, 1);

                finalColor.x = skyMask ? skyColor.x : finalColor.x;
                finalColor.y = skyMask ? skyColor.y : finalColor.y;
                finalColor.z = skyMask ? skyColor.z : finalColor.z;
            }
            finalColor = Tonemap_Unreal(finalColor * Exposure);

            v_int packedColor = swr::pixfmt::RGBA8u::Pack({ finalColor, 1.0f });
            store(packedColor, & fb.ColorBuffer[tileOffset]);

            if (EnableTAA && FrameNo != 0) {
                v_float4 prevNDC = PerspectiveDiv(TransformVector(PrevProjMat, { v_float3(worldPos), 1.0f }));
                prevNDC.x -= Jitter.x;
                prevNDC.y -= Jitter.y;
                prevNDC = prevNDC * 0.5 + 0.5;

                v_int prevX = round2i(prevNDC.x * fb.Width);
                v_int prevY = round2i(prevNDC.y * fb.Height);

                // Save prevColor to avoid having to recompute worldDepth on the TAA pass
                v_int prevColor = PrevFrame->SampleColor(prevX, prevY, packedColor);
                store(prevColor, fb.GetAttachmentBuffer<uint32_t>(0, tileOffset));
            }
        });

        // Temporal Anti-Alias
        // - https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/
        // - https://alextardif.com/TAA.html
        // - https://sugulee.wordpress.com/2021/06/21/temporal-anti-aliasingtaa-tutorial/
        // 
        // TODO: it might be possible to work in tiles rather than on the entire fb to improve cache-ability
        if (EnableTAA && FrameNo > 0) {
            fb.IterateTiles([&](uint32_t x, uint32_t y) {
                uint32_t tileOffset = fb.GetPixelOffset(x, y);
                v_int currColor = load(&fb.ColorBuffer[tileOffset]);
                v_int prevColor = load(fb.GetAttachmentBuffer<uint32_t>(0, tileOffset));

                v_int minColor = (int32_t)0xFFFF'FFFF, maxColor = 0;
                v_int tileX = (int32_t)x + FragPixelOffsetsX;
                v_int tileY = (int32_t)y + FragPixelOffsetsY;

                // Sample a 3x3 neighborhood to create a box in color space
                for (int32_t xo = -1; xo <= 1; xo++) {
                    for (int32_t yo = -1; yo <= 1; yo++) {
                        v_int color = fb.SampleColor(tileX + xo, tileY + yo, currColor);
                        minColor = _mm512_min_epu8(minColor, color);
                        maxColor = _mm512_max_epu8(maxColor, color);
                    }
                }
                prevColor = _mm512_min_epu8(_mm512_max_epu8(prevColor, minColor), maxColor);

                //v_float3 currColorF = v_float3(swr::pixfmt::RGBA8u::Unpack(currColor));
                //v_float3 prevColorF = v_float3(swr::pixfmt::RGBA8u::Unpack(prevColor));
                //v_float3 resolvedColor = prevColorF + (currColorF - prevColorF) * 0.1f;
                //v_int finalColor = swr::pixfmt::RGBA8u::Pack({ resolvedColor, 1.0f });

                // Fixed-point is a smidge faster
                v_int alpha = _mm512_set1_epi16(0.1f * ((1 << 15) - 1));
                v_int finalColor = lerp16((prevColor >> 0) & 0x00FF'00FF, (currColor >> 0) & 0x00FF'00FF, alpha) |
                                  lerp16((prevColor >> 8) & 0x00FF'00FF, (currColor >> 8) & 0x00FF'00FF, alpha) << 8;
                store(finalColor, &ResolvedTAABuffer[tileOffset]);
            });
            std::swap(fb.ColorBuffer, ResolvedTAABuffer);
        }
        FrameNo++;
        PrevProjMat = ProjMat;
    }
    
    void UpdateJitter(glm::mat4& projMat, swr::Framebuffer& fb) {
        if (!EnableTAA) return;
        
        PrevJitter = Jitter;
        Jitter = (Halton23[FrameNo % 8] * 2.0f - 1.0f) / glm::vec2(fb.Width, fb.Height);
        projMat[2][0] += Jitter.x;
        projMat[2][1] += Jitter.y;

        if (PrevFrame == nullptr || fb.Width != PrevFrame->Width || fb.Height != PrevFrame->Height) {
            PrevFrame = std::make_unique<swr::Framebuffer>(fb.Width, fb.Height);
            PrevFrame->DepthBuffer = nullptr; // save memory we shouldn't have allocated in the first place.
            ResolvedTAABuffer = swr::alloc_buffer<uint32_t>(fb.Width * fb.Height);
            FrameNo = 0;
        }
        std::swap(fb.ColorBuffer, PrevFrame->ColorBuffer);
    }

    void SetSkybox(swr::HdrTexture2D::Ptr&& envCube) {
        IrradianceMap = GenerateIrradianceMap(*envCube);
        FilteredEnvMap = GenerateRadianceMap(*envCube);
        SkyboxTex = std::move(envCube);
    }

    void ComposeDebug(swr::Framebuffer& fb, DebugLayer layer) {
        fb.IterateTiles([&](uint32_t x, uint32_t y) {
            uint32_t tileOffset = fb.GetPixelOffset(x, y);
            v_float tileDepth = load(&fb.DepthBuffer[tileOffset]);
            v_int skyMask = tileDepth <= 0.0f;

            v_float3 finalColor = 0.0f;

            v_float4 G1 = swr::pixfmt::RGBA8u::Unpack(load(&fb.ColorBuffer[tileOffset]));
            v_int G2r = load(fb.GetAttachmentBuffer<uint32_t>(0, tileOffset));
            v_float3 G2 = SignedOctDecode(G2r);

            if (layer == DebugLayer::BaseColor) {
                finalColor = v_float3(G1);
            } else if (layer == DebugLayer::Normals) {
                finalColor = v_float3(G2) * 0.5f + 0.5f;
            } else if (layer == DebugLayer::MetallicRoughness) {
                finalColor = { G1.w, conv2f(G2r >> 1 & 63) / 63.0f, 0.0f };
            } else if (layer == DebugLayer::Occlusion) {
                finalColor = GetAO(fb, x, y);
            } else if (layer == DebugLayer::EmissiveMask) {
                finalColor.x = (G2r & 1) != 0 ? 1.0f : 0;
            }

            uint32_t backgroundRGB = (x / 4 + y / 4) % 2 ? 0xFF'A0A0A0 : 0xFF'FFFFFF;
            v_int finalRGB = skyMask ? (int32_t)backgroundRGB : swr::pixfmt::RGBA8u::Pack({ finalColor, 1.0f });

            store(finalRGB, &fb.ColorBuffer[tileOffset]);
        });
    }

private:
    v_float GetShadow(v_float3 worldPos, v_float NoL) {
        // I don't even know how I got these values, but the order has some impact on the final result.
        // Filament's disk is quite noisy even with TAA.
        static const float PoissonDisk[2][16] = {
            { -0.03, 0.17, -0.09, -0.43, 0.67, 0.88, -0.85, -0.34, 0.22, -0.64, 0.29, 0.48, -0.1, -0.99, -0.25, 0.63 },
            { -0.09, 0.85, -1.01, 0.02, -0.21, 0.36, 0.4, 0.86, -0.53, -0.69, 0.17, -0.87, 0.43, -0.08, -0.51, 0.7 },
        };

        v_float4 shadowPos = TransformVector(ShadowProjMat, { worldPos, 1.0f });
        // ShadowProj is orthographic, so W is always 1.0
        // shadowPos.w = rcp14(shadowPos.w);

        v_float bias = max((1.0f - NoL) * 0.015f, 0.003f);
        v_float currentDepth = shadowPos.z * shadowPos.w - bias;

        v_float sx = shadowPos.x * shadowPos.w * 0.5f + 0.5f;
        v_float sy = shadowPos.y * shadowPos.w * 0.5f + 0.5f;
        sx *= ShadowBuffer->Width;
        sy *= ShadowBuffer->Height;

        // aesenc costs 5c/1t, this hash probably takes around ~8-10c on TGL.
        v_int rng = _mm512_aesenc_epi128(re2i(sx) + (int32_t)FrameNo, _mm512_set1_epi32(0)) ^ 
                   _mm512_aesenc_epi128(re2i(sy), _mm512_set1_epi32(0));

        v_float rc, rs;
        v_float randomAngle = conv2f(shrl(rng, 8)) * (tau / (1 << 24));
        sincos(randomAngle, rc, rs);

        auto samples = _mm_set1_epi8(0);

        for (uint32_t i = 0; i < 8; i++) {
            v_float jx = PoissonDisk[0][i], jy = PoissonDisk[1][i];
            v_float rx = jx * rc - jy * rs + sx; // 2x fma
            v_float ry = jx * rs + jy * rc + sy;

            v_float occlusionDepth = ShadowBuffer->SampleDepth(round2i(rx), round2i(ry));
            samples = _mm_mask_add_epi8(samples, movemask(occlusionDepth >= currentDepth), samples, _mm_set1_epi8(1));
        }
        return conv2f(_mm512_cvtepi8_epi32(samples)) * (1.0f / 8);
    }
    static v_float GetAO(swr::Framebuffer& fb, uint32_t x, uint32_t y) {
        uint8_t* basePtr = fb.GetAttachmentBuffer<uint8_t>(8);
        uint32_t stride = fb.Width / 2;
        uint8_t temp[4 * 4];

        for (uint32_t ty = 0; ty < 4; ty++) {
            for (uint32_t tx = 0; tx < 4; tx++) {
                temp[tx + ty * 4] = basePtr[((x + tx) / 2) + ((y + ty) / 2) * stride];
            }
        }
        return conv2f(_mm512_cvtepu8_epi32(_mm_loadu_epi8(temp))) * (1.0f / 255);
    }

    static swr::HdrTexture2D::Ptr GenerateIrradianceMap(const swr::HdrTexture2D& envTex) {
        auto tex = swr::CreateTexture<swr::HdrTexture2D>(32, 32, 1, 6);

        for (uint32_t layer = 0; layer < 6; layer++) {
            swr::texutil::IterateTiles(tex->Width, tex->Height, [&](uint32_t x, uint32_t y, v_float u, v_float v) {
                v_float3 dir = swr::texutil::UnprojectCubemap(u, v, (int32_t)layer);
                v_float3 color = PrefilterDiffuseIrradiance(envTex, dir);
                tex->WriteTile(color, x, y, layer);
            });
        }
        return tex;
    }
    static swr::HdrTexture2D::Ptr GenerateRadianceMap(const swr::HdrTexture2D& envTex) {
        auto tex = swr::CreateTexture<swr::HdrTexture2D>(128, 128, 8, 6);

        for (uint32_t layer = 0; layer < 6; layer++) {
            for (uint32_t level = 0; level < tex->MipLevels; level++) {
                uint32_t w = tex->Width >> level;
                uint32_t h = tex->Height >> level;

                swr::texutil::IterateTiles(w, h, [&](uint32_t x, uint32_t y, v_float u, v_float v) {
                    v_float3 dir = swr::texutil::UnprojectCubemap(u, v, (int32_t)layer);
                    v_float3 color = PrefilterEnvMap(envTex, level / (tex->MipLevels - 1.0f), dir);
                    tex->WriteTile(color, x, y, layer, level);
                });
            }
        }
        return tex;
    }
    static swr::TexturePtr2D<swr::pixfmt::RG16f> GenerateBRDFEnvLut() {
        auto tex = swr::CreateTexture2D<swr::pixfmt::RG16f>(128, 128, 1, 1);

        swr::texutil::IterateTiles(tex->Width, tex->Height, [&](uint32_t x, uint32_t y, v_float u, v_float v) {
            v_float2 f_ab = IntegrateBRDF(u, v);
            tex->WriteTile(f_ab, x, y);
        });
        return tex;
    }

    static v_float3 PrefilterDiffuseIrradiance(const swr::HdrTexture2D& envTex, v_float3 N) {
        v_int nzMask = abs(N.z) < 0.999f;
        v_float3 up = { nzMask ? 0.0f : 1.0f, 0.0, nzMask ? 1.0f : 0.0f };
        v_float3 right = normalize(cross(up, N));
        up = cross(N, right);

        v_float3 color = 0.0f;
        uint32_t sampleCount = 0;

        const float twoPi = tau, halfPi = pi * 0.5;
        float deltaPhi = twoPi / 360.0;
        float deltaTheta = halfPi / 90.0;

#ifndef NDEBUG
        deltaPhi = twoPi / 16.0;
        deltaTheta = halfPi / 4.0;
#endif

        for (float phi = 0.0; phi < twoPi; phi += deltaPhi) {
            for (float theta = 0.0; theta < halfPi; theta += deltaTheta) {
                // Spherical to World Space in two steps...
                v_float3 tempVec = cosf(phi) * right + sinf(phi) * up;
                v_float3 sampleVector = cosf(theta) * N + sinf(theta) * tempVec;
                v_float3 envColor = envTex.SampleCube<EnvSampler>(sampleVector, 2);

                color = color + TonemapSample(envColor) * cosf(theta) * sinf(theta);
                sampleCount++;
            }
        }
        return TonemapSampleInv(color * (pi * 1.0f / sampleCount));
    }

    // From Karis, 2014
    static v_float3 PrefilterEnvMap(const swr::HdrTexture2D& envTex, float roughness, v_float3 R) {
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
            v_float VoH = dot(V, H);
            v_float NoH = max(VoH, 0.0f);  // Since N = V in our approximation
            // Use microfacet normal H to find L
            v_float3 L = 2.0f * VoH * H - V;
            v_float NoL = max(dot(N, L), 0.0f);

            if (any(NoL > 0.0f)) {
                // Based off https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
                // Typically you'd have the following:
                // float pdf = D_GGX(NoH, roughness) * NoH / (4.0 * VoH);
                // but since V = N => VoH == NoH
                v_float pdf = D_GGX(NoH, roughness) / 4.0f + 0.001f;
                // Solid angle of current sample -- bigger for less likely samples
                v_float omegaS = 1.0f / (numSamples * pdf);
                // Solid angle of texel
                float omegaP = 4.0f * pi / (6 * envTex.Width * envTex.Width);
                // Mip level is determined by the ratio of our sample's solid angle to a texel's solid angle
                v_float mipLevel = max(0.5f * approx_log2(omegaS / omegaP), 0.0f);

                v_float3 envColor = envTex.SampleCube<EnvSampler>(L, mipLevel);

                prefilteredColor = prefilteredColor + TonemapSample(envColor) * NoL;
                totalWeight += NoL;
            }
        }
        return TonemapSampleInv(prefilteredColor / totalWeight);
    }

    static v_float2 IntegrateBRDF(v_float NoV, v_float roughness) {
        v_float3 N = { 0, 0, 1 };
        v_float3 V = { sqrt(1.0f - NoV * NoV), 0.0f, NoV };
        v_float2 r = { 0.0f, 0.0f };
        const uint32_t sampleCount = 1024;

        for (uint32_t i = 0; i < sampleCount; i++) {
            glm::vec2 Xi = Hammersley(i, sampleCount);
            v_float3 H = ImportanceSampleGGX(Xi, roughness, N);
            v_float3 L = 2.0f * dot(V, H) * H - V;

            v_float VoH = max(0.0f, dot(V, H));
            v_float NoL = max(0.0f, L.z);
            v_float NoH = max(0.0f, H.z);

            if (any(NoL > 0.0f)) {
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

    // Tonemap importance samples to prevent fireflies - http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    static v_float3 TonemapSample(v_float3 color) {
        v_float luma = dot(color, { 0.299f, 0.587f, 0.114f });
        return color * rcp14(1.0f + luma);
    }
    static v_float3 TonemapSampleInv(v_float3 color) {
        v_float luma = dot(color, { 0.299f, 0.587f, 0.114f });
        return color * rcp14(1.0f - luma);
    }

    static v_float D_GGX(v_float NoH, v_float roughness) {
        v_float a = NoH * roughness;
        v_float k = roughness / (1.0f - NoH * NoH + a * a);
        return k * k * (1.0f / pi);
    }
    static v_float V_SmithGGXCorrelatedFast(v_float NoV, v_float NoL, v_float roughness) {
        v_float a = 2.0f * NoL * NoV;
        v_float b = NoL + NoV;
        return 0.5f / lerp(a, b, roughness);
    }
    static v_float3 F_Schlick(v_float u, v_float3 f0) {
        v_float f = pow5(1.0 - u);
        return f + f0 * (1.0f - f);
    }
    static v_float Fd_Lambert() { return 1.0f / pi; }

    static v_float pow5(v_float x) { return (x * x) * (x * x) * x; }

    static glm::vec2 Hammersley(uint32_t i, uint32_t numSamples) {
        uint32_t bits = i;
        bits = (bits << 16) | (bits >> 16);
        bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
        bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
        bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
        bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
        return { i / (float)numSamples, bits * (1.0 / UINT_MAX) };
    }
    static v_float3 ImportanceSampleGGX(glm::vec2 Xi, v_float roughness, v_float3 N) {
        v_float a = roughness * roughness;

        v_float phi = 2.0 * pi * Xi.x;
        v_float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
        v_float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        v_float3 H = { cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta };

        // vec3 upVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        v_int nzMask = abs(N.z) < 0.999f;
        v_float3 upVector = { nzMask ? 0.0f : 1.0f, 0.0, nzMask ? 1.0f : 0.0f };
        v_float3 tangentX = normalize(cross(upVector, N));
        v_float3 tangentY = cross(N, tangentX);
        return tangentX * H.x + tangentY * H.y + N * H.z;
    }

    // https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
    static v_float2 EnvBRDFApprox(v_float NoV, v_float roughness) {
        const v_float4 c0 = { -1, -0.0275, -0.572, 0.022 };
        const v_float4 c1 = { 1, 0.0425, 1.04, -0.04 };
        v_float4 r = roughness * c0 + c1;
        v_float a004 = min(r.x * r.x, approx_exp2(-9.28 * NoV)) * r.x + r.y;
        return { a004 * -1.04 + r.z, a004 * 1.04 + r.w };
    }

    // https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
    static v_float3 SrgbToLinear(v_float3 x) {
        return x * (x * (x * 0.305306011 + 0.682171111) + 0.012522878);
    }
    static v_float3 Tonemap_Unreal(v_float3 x) {
        // Unreal 3, Documentation: "Color Grading"
        // Adapted to be close to Tonemap_ACES, with similar range
        // Gamma 2.2 correction is baked in, don't use with sRGB conversion!
        return x / (x + 0.155) * 1.019;
    }
    //static v_float3 LinearToSrgb(v_float3 x) {
    //    v_float3 S1 = sqrt(x);
    //    v_float3 S2 = sqrt(S1);
    //    v_float3 S3 = sqrt(S2);
    //    return 0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.0225411470 * x;
    //}

    // Encode normal to signed octahedron + arbitrary extra into 12:12:1. Lower 7 bits are unused.
    // https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
    static v_int SignedOctEncode(v_float3 n) {
        v_float m = rcp14(abs(n.x) + abs(n.y) + abs(n.z)) * 0.5f;
        n = n * m;

        v_float ny = n.y + 0.5;
        v_float nx = n.x + ny;
        ny = ny - n.x;

        return round2i(nx * 4095.0f) << 19 | round2i(ny * 4095.0f) << 7 | (re2i(n.z) & (1 << 31));
    }
    static v_float3 SignedOctDecode(v_int p) {
        v_float px = conv2f((p >> 19) & 4095) / 4095.0f;
        v_float py = conv2f((p >> 7) & 4095) / 4095.0f;

        v_float nx = px - py;
        v_float ny = px + py - 1.0f;
        v_float nz = re2f(re2i(1.0 - abs(nx) - abs(ny)) ^ (p & (1 << 31)));

        return normalize({ nx, ny, nz });

        // OutN.z = n.z * 2.0 - 1.0;  // n.z ? 1 : -1
        // OutN.z = OutN.z * (1.0 - abs(OutN.x) - abs(OutN.y));
    }

    // Temporal anti-alias sub-pixel jitter offsets - Halton(2, 3)
    static inline const glm::vec2 Halton23[16] = {
        { 0.50000, 0.33333 }, { 0.25000, 0.66667 }, { 0.75000, 0.11111 }, { 0.12500, 0.44444 },  //
        { 0.62500, 0.77778 }, { 0.37500, 0.22222 }, { 0.87500, 0.55556 }, { 0.06250, 0.88889 },  //
        { 0.56250, 0.03704 }, { 0.31250, 0.37037 }, { 0.81250, 0.70370 }, { 0.18750, 0.14815 },  //
        { 0.68750, 0.48148 }, { 0.43750, 0.81481 }, { 0.93750, 0.25926 }, { 0.03125, 0.59259 },
    };

    friend struct SSAO;
};
struct DepthOnlyShader {
    static constexpr uint32_t NumCustomAttribs = 0;

    glm::mat4 ProjMat;

    // void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
    //     v_float4 pos = { data.ReadAttribs<v_float3>(&scene::Vertex::x), 1.0f };
    //     vars.Position = TransformVector(ProjMat, pos);
    // }

    void ShadePixels(swr::Framebuffer& fb, swr::FragmentVars& vars) const {
        _mm512_mask_storeu_ps(&fb.DepthBuffer[vars.TileOffset], vars.TileMask, vars.Depth);
    }
};
struct OverdrawShader {
    static constexpr uint32_t NumCustomAttribs = 0;

    glm::mat4 ProjMat;

    // void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
    //     v_float4 pos = { data.ReadAttribs<v_float3>(&scene::Vertex::x), 1.0f };
    //     vars.Position = TransformVector(ProjMat, pos);
    // }

    void ShadePixels(swr::Framebuffer& fb, swr::FragmentVars& vars) const {
        v_int color = load(&fb.ColorBuffer[vars.TileOffset]);
        color = _mm512_adds_epu8(color, _mm512_set1_epi32((int32_t)0xFF'000020));
        fb.WriteTile(vars.TileOffset, 0xFFFF, color, vars.Depth);
    }
};

// TODO: Implement possibly better and faster approach from "Scalable Ambient Obscurance" +/or maybe copy a few tricks from XeGTAO or something?
// https://www.shadertoy.com/view/3dK3zR
struct SSAO {
    static const uint32_t KernelSize = 16, FbAttachId = 8;

    float Radius = 1.3f, MaxRange = 0.35f;

    float Kernel[3][KernelSize];
    v_int _randSeed;

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
            s *= glm::mix(0.1f, 1.0f, scale * scale);

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

        XorShiftStep(_randSeed); // update RNG on each frame for TAA

        fb.IterateTiles([&](uint32_t x, uint32_t y) {
            v_int rng = _randSeed * (int32_t)(x * 12345 + y * 9875);

            v_int iu = (int32_t)x + FragPixelOffsetsX * 2;
            v_int iv = (int32_t)y + FragPixelOffsetsY * 2;
            v_float z = depthMap.SampleDepth(iu, iv, 0);

            if (!any(z < 1.0f)) return; // skip over tiles that don't have geometry

            v_float4 pos = PerspectiveDiv(TransformVector(invProj, { conv2f(iu), conv2f(iv), z, 1.0f }));

            // TODO: better normal reconstruction - https://atyuwen.github.io/posts/normal-reconstruction/
            // v_float3 posDx = { dFdx(pos.x), dFdx(pos.y), dFdx(pos.z) };
            // v_float3 posDy = { dFdy(pos.x), dFdy(pos.y), dFdy(pos.z) };
            // v_float3 N = normalize(cross(posDx, posDy));

            // Using textured normals is better than reconstructing from blocky derivatives, particularly around edges.
            v_int G2r = gather(fb.GetAttachmentBuffer<int32_t>(0), fb.GetPixelOffset(iu, iv));
            v_float3 N = v_float3(DefaultShader::SignedOctDecode(G2r));

            XorShiftStep(rng);
            v_float3 rotation = normalize({ conv2f(rng & 255) * (1.0f / 127) - 1.0f, conv2f(rng >> 8 & 255) * (1.0f / 127) - 1.0f,0.0f });
            
            v_float NdotR = dot(rotation, N);
            v_float3 T = normalize({
                rotation.x - N.x * NdotR,
                rotation.y - N.y * NdotR,
                rotation.z - N.z * NdotR,
            });
            v_float3 B = cross(N, T);

            auto occlusion = _mm_set1_epi8(0);

            for (uint32_t i = 0; i < KernelSize; i++) {
                v_float kx = Kernel[0][i], ky = Kernel[1][i], kz = Kernel[2][i];

                v_float sx = (T.x * kx + B.x * ky + N.x * kz) * Radius + pos.x;
                v_float sy = (T.y * kx + B.y * ky + N.y * kz) * Radius + pos.y;
                v_float sz = (T.z * kx + B.z * ky + N.z * kz) * Radius + pos.z;

                v_float4 samplePos = PerspectiveDiv(TransformVector(projViewMat, { sx, sy, sz, 1.0f }));
                v_float sampleDepth = LinearizeDepth(depthMap.SampleDepth(samplePos.x * 0.5f + 0.5f, samplePos.y * 0.5f + 0.5f, 0));
                // FIXME: range check kinda breaks when the camera gets close to geom
                //        depth linearization might be wrong (cam close ups)
                v_float sampleDist = abs(LinearizeDepth(z) - sampleDepth);

                v_mask rangeMask = movemask(sampleDist < MaxRange);
                v_mask occluMask = movemask(sampleDepth <= LinearizeDepth(samplePos.z) - 0.03f);
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
        }, 2);

        //ApplyBlur(fb);
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
        const int BlurRadius = 1, BlurSamples = BlurRadius * 2 + 1;
        __m512i accum = _mm512_set1_epi16(0);

        for (int32_t so = -BlurRadius; so <= BlurRadius; so++) {
            accum = _mm512_add_epi16(accum, _mm512_cvtepu8_epi16(_mm256_loadu_epi8(&src[so * lineStride])));
        }
        __m256i c = _mm512_cvtepi16_epi8(_mm512_mulhrs_epi16(accum, _mm512_set1_epi16(32767 / BlurSamples)));
        _mm256_storeu_epi8(dst, c);
    }

    static v_float LinearizeDepth(v_float d) {
        // TODO: avoid hardcoding this, get from Camera&
        const float zNear = 0.01f, zFar = 1000.0f;
        return (zNear * zFar) / (zFar + d * (zNear - zFar));
    }

    void XorShiftStep(v_int& x) {
        x = x ^ x << 13;
        x = x ^ x >> 17;
        x = x ^ x << 5;
    }
};

}; // namespace renderer