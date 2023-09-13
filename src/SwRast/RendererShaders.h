#pragma once

#include "SwRast.h"
#include "Texture.h"
#include <random>

namespace renderer {

using swr::VInt, swr::VFloat, swr::VFloat2, swr::VFloat3, swr::VFloat4, swr::VMask;
using namespace swr::simd;

// Deferred PBR shader
// https://google.github.io/filament/Filament.html
// https://bruop.github.io/ibl/
struct DefaultShader {
    static const uint32_t NumCustomAttribs = 8;

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
    swr::HdrTexture2D IrradianceMap, FilteredEnvMap;
    swr::Texture2D<swr::pixfmt::RG16f> BRDFEnvLut;
    swr::HdrTexture2D SkyboxTex;

    // Uniform: Forward
    glm::mat4 ProjMat, ModelMat;
    // TODO: combine these two into a single layered texture (may help get mip/scaled UVs and stuff CSE'ed)
    const swr::RgbaTexture2D* BaseColorTex;
    const swr::RgbaTexture2D* NormalMetallicRoughnessTex;

    // Uniform: Compose pass
    const swr::Framebuffer* ShadowBuffer;
    glm::mat4 ShadowProjMat, ViewMat;
    glm::vec3 LightPos, ViewPos;

    float IntensityIBL = 0.3f;
    float Exposure = 1.0f;

    DefaultShader(swr::HdrTexture2D&& envTex)
        : IrradianceMap(GenerateIrradianceMap(envTex)),
          FilteredEnvMap(GenerateRadianceMap(envTex)),
          BRDFEnvLut(GenerateBRDFEnvLut()),
          SkyboxTex(std::move(envTex)) {} // note member decl order

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
        VFloat4 baseColor;

        if (BaseColorTex != nullptr) {
            baseColor = UnpackRGBA(BaseColorTex->Sample<SurfaceSampler>(u, v));
        }

        VFloat3 N = normalize(vars.GetSmooth<VFloat3>(2));

        VFloat metalness = 0.0f;
        VFloat roughness = 0.5f;

        if (NormalMetallicRoughnessTex != nullptr) {
            VFloat4 SN = UnpackRGBA(NormalMetallicRoughnessTex->Sample<SurfaceSampler>(u, v));
            VFloat3 T = vars.GetSmooth<VFloat3>(5);

            // Gram-schmidt process (produces higher-quality normal mapping on large meshes)
            // Re-orthogonalize T with respect to N
            T = normalize(T - dot(T, N) * N);
            VFloat3 B = cross(N, T);

            // Flip bitangent depending on whether UVs are mirrored - https://stackoverflow.com/a/44901073
            // TODO: maybe cheaper to compute tangents using screen derivatives all the way. (maybe too blocky?)
            VFloat det = (dFdy(u) * dFdx(v) - dFdx(u) * dFdy(v)) & -0.0f;
            B = { B.x ^ det, B.y ^ det, B.z ^ det };  // det < 0 ? -B : B

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
        VMask mask = vars.TileMask & (baseColor.w > 0.5f);

        // GBuffer
        //   #1 [888: RGB]    [8: Metalness]
        //   #2 [10.10.1: Normal] [10: Roughness] [1: Unused]
        baseColor.w = metalness;
        fb.WriteTile(vars.TileOffset, mask, PackRGBA(baseColor), vars.Depth);
        _mm512_mask_store_epi32(fb.GetAttachmentBuffer<uint32_t>(0, vars.TileOffset), mask, SignedOctEncode(N, roughness));
    }

    void Compose(swr::Framebuffer& fb, bool hasSSAO) {
        glm::mat4 invProj = glm::inverse(ProjMat);
        // Bias matrix to take UVs in range [0..screen] rather than [-1..1]
        invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
        invProj = glm::scale(invProj, glm::vec3(2.0f / fb.Width, 2.0f / fb.Height, 1.0f));

        fb.IterateTiles([&](uint32_t x, uint32_t y) {
            uint32_t tileOffset = fb.GetPixelOffset(x, y);
            VFloat tileDepth = VFloat::load(&fb.DepthBuffer[tileOffset]);
            VMask skyMask = tileDepth >= 1.0f;

            VFloat u = conv2f((int32_t)x + (VInt::ramp() & 3));
            VFloat v = conv2f((int32_t)y + (VInt::ramp() >> 2));
            VFloat3 worldPos = VFloat3(PerspectiveDiv(TransformVector(invProj, { u, v, tileDepth, 1.0f })));

            VFloat3 finalColor;

            if (!all(skyMask)) {
                VFloat4 G1 = UnpackRGBA(VInt::load(&fb.ColorBuffer[tileOffset]));
                VFloat4 G2 = SignedOctDecode(VInt::load(fb.GetAttachmentBuffer<uint32_t>(0, tileOffset)));

                VFloat3 baseColor = SrgbToLinear(VFloat3(G1));
                VFloat3 N = VFloat3(G2);
                VFloat metalness = G1.w;

                VFloat perceptualRoughness = max(G2.w, 0.045f);  // clamp to prevent div by zero
                VFloat roughness = perceptualRoughness * perceptualRoughness;

                VFloat3 V = normalize(ViewPos - worldPos);
                VFloat3 L = normalize(LightPos);
                VFloat3 H = normalize(V + L);

                VFloat NoV = max(dot(N, V), 1e-4f);
                VFloat NoL = max(dot(N, L), 0.0f);
                VFloat NoH = max(dot(N, H), 0.0f);
                VFloat LoH = max(dot(L, H), 0.0f);

                float reflectance = 0.5f;

                VFloat D = D_GGX(NoH, roughness);
                VFloat G = V_SmithGGXCorrelatedFast(NoV, NoL, roughness);
                VFloat3 f0 = (0.16f * reflectance * reflectance) * (1.0f - metalness) + baseColor * metalness;
                VFloat3 F = F_Schlick(LoH, f0);

                // specular BRDF
                VFloat3 Fr = D * G * F;

                // diffuse BRDF
                VFloat3 diffuseColor = (1.0f - metalness) * VFloat3(baseColor);
                VFloat3 Fd = (1.0f - F) * diffuseColor * Fd_Lambert();

                if (ShadowBuffer != nullptr && any(NoL > 0.0f)) {
                    NoL *= GetShadow(worldPos, NoL);
                }
                finalColor = (Fd + Fr) * NoL;

                if (IntensityIBL > 0.0f) {
                    // TODO: cubemap seams are _very_ noticeable for things like spheres with high roughness
                    //       http://www.ludicon.com/castano/blog/articles/seamless-cube-map-filtering/
                    VFloat3 R = reflect(0 - V, N);
                    VFloat3 irradiance = IrradianceMap.SampleCube<EnvSampler>(R);
                    VFloat3 radiance = FilteredEnvMap.SampleCube<EnvSampler>(R, perceptualRoughness * (int32_t)FilteredEnvMap.MipLevels);
                    VFloat2 dfg = BRDFEnvLut.Sample<EnvSampler>(NoV, perceptualRoughness);
                    VFloat3 specularColor = f0 * dfg.x + dfg.y;
                    VFloat3 ibl = irradiance * diffuseColor + radiance * specularColor;

                    // TODO: multi-scatter IBL

                    if (hasSSAO) {
                        ibl = ibl * GetAO(fb, x, y);
                    }
                    finalColor = finalColor + ibl * IntensityIBL;
                }
            }

            if (any(skyMask)) {
                VFloat3 skyColor = SkyboxTex.SampleCube<SurfaceSampler>(worldPos - ViewPos);
                //VFloat3 skyColor = FilteredEnvMap.SampleCube<EnvSampler>(worldPos - ViewPos, 1.2f);

                finalColor.x = csel(skyMask, skyColor.x, finalColor.x);
                finalColor.y = csel(skyMask, skyColor.y, finalColor.y);
                finalColor.z = csel(skyMask, skyColor.z, finalColor.z);
            }

            finalColor = Tonemap_Unreal(finalColor * Exposure);
            PackRGBA({ finalColor, 1.0f }).store(&fb.ColorBuffer[tileOffset]);
        });
    }

private:
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
            if (i == 4 && (_mm_cmpeq_epu8_mask(samples, _mm_set1_epi8(0)) || _mm_cmpeq_epu8_mask(samples, _mm_set1_epi8(4)))) {
                weight = 1.0f / 4;
                break;
            }
            VFloat occlusionDepth = ShadowBuffer->SampleDepth((x + PoissonDisk[0][i]) >> 6, (y + PoissonDisk[1][i]) >> 6);
            samples = _mm_mask_add_epi8(samples, occlusionDepth >= currentDepth, samples, _mm_set1_epi8(1));
        }
        return conv2f(_mm512_cvtepi8_epi32(samples)) * weight;
    }
    static VFloat GetAO(swr::Framebuffer& fb, uint32_t x, uint32_t y) {
        uint8_t* basePtr = fb.GetAttachmentBuffer<uint8_t>(4);
        uint32_t stride = fb.Width / 2;
        uint8_t temp[4 * 4];

        for (uint32_t ty = 0; ty < 4; ty++) {
            for (uint32_t tx = 0; tx < 4; tx++) {
                temp[tx + ty * 4] = basePtr[((x + tx) / 2) + ((y + ty) / 2) * stride];
            }
        }
        return conv2f(_mm512_cvtepu8_epi32(_mm_loadu_epi8(temp))) * (1.0f / 255);
    }

    static swr::HdrTexture2D GenerateIrradianceMap(const swr::HdrTexture2D& envTex) {
        swr::HdrTexture2D tex(32, 32, 1, 6);

        for (uint32_t layer = 0; layer < 6; layer++) {
            swr::texutil::IterateTiles(tex.Width, tex.Height, [&](uint32_t x, uint32_t y, VFloat u, VFloat v) {
                VFloat3 dir = swr::texutil::UnprojectCubemap(u, v, (int32_t)layer);
                VFloat3 color = PrefilterDiffuseIrradiance(envTex, dir);
                tex.WriteTile(swr::pixfmt::R11G11B10f::Pack(color), x, y, layer);
            });
        }
        return tex;
    }
    static swr::HdrTexture2D GenerateRadianceMap(const swr::HdrTexture2D& envTex) {
        swr::HdrTexture2D tex(128, 128, 8, 6);

        for (uint32_t layer = 0; layer < 6; layer++) {
            for (uint32_t level = 0; level < tex.MipLevels; level++) {
                uint32_t w = tex.Width >> level;
                uint32_t h = tex.Height >> level;

                swr::texutil::IterateTiles(w, h, [&](uint32_t x, uint32_t y, VFloat u, VFloat v) {
                    VFloat3 dir = swr::texutil::UnprojectCubemap(u, v, (int32_t)layer);
                    VFloat3 color = PrefilterEnvMap(envTex, level / (tex.MipLevels - 1.0f), dir);
                    tex.WriteTile(swr::pixfmt::R11G11B10f::Pack(color), x, y, layer, level);
                });
            }
        }
        return tex;
    }
    static swr::Texture2D<swr::pixfmt::RG16f> GenerateBRDFEnvLut() {
        swr::Texture2D<swr::pixfmt::RG16f> tex(128, 128, 1, 1);

        swr::texutil::IterateTiles(tex.Width, tex.Height, [&](uint32_t x, uint32_t y, VFloat u, VFloat v) {
            VFloat2 f_ab = IntegrateBRDF(u, v);
            tex.WriteTile(swr::pixfmt::RG16f::Pack(f_ab), x, y);
        });
        return tex;
    }

    static VFloat3 PrefilterDiffuseIrradiance(const swr::HdrTexture2D& envTex, VFloat3 N) {
        VMask nzMask = abs(N.z) < 0.999f;
        VFloat3 up = { csel(nzMask, VFloat(0.0f), 1.0f), 0.0, csel(nzMask, VFloat(1.0f), 0.0f) };
        VFloat3 right = normalize(cross(up, N));
        up = cross(N, right);

        VFloat3 color = 0.0f;
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
                VFloat3 tempVec = cosf(phi) * right + sinf(phi) * up;
                VFloat3 sampleVector = cosf(theta) * N + sinf(theta) * tempVec;
                VFloat3 envColor = envTex.SampleCube<SurfaceSampler>(sampleVector, 2);

                // Shitty workaround to reduce artifacts around overly bright spots
                envColor.x = min(envColor.x, 1000.0f);
                envColor.y = min(envColor.y, 1000.0f);
                envColor.z = min(envColor.z, 1000.0f);

                color = color + envColor * cosf(theta) * sinf(theta);
                sampleCount++;
            }
        }
        return color * (pi * 1.0f / sampleCount);
    }

    // From Karis, 2014
    static VFloat3 PrefilterEnvMap(const swr::HdrTexture2D& envTex, float roughness, VFloat3 R) {
#ifdef NDEBUG
        const uint32_t numSamples = 1024;
#else
        const uint32_t numSamples = 16;
#endif
        VFloat3 N = R;
        VFloat3 V = R;

        roughness = std::max(roughness, 0.01f);

        VFloat3 prefilteredColor = 0.0f;
        VFloat totalWeight = 0.0f;

        for (uint32_t i = 0; i < numSamples; i++) {
            glm::vec2 Xi = Hammersley(i, numSamples);
            VFloat3 H = ImportanceSampleGGX(Xi, roughness, N);
            VFloat VoH = dot(V, H);
            VFloat NoH = max(VoH, 0.0f);  // Since N = V in our approximation
            // Use microfacet normal H to find L
            VFloat3 L = 2.0f * VoH * H - V;
            VFloat NoL = max(dot(N, L), 0.0f);

            if (any(NoL > 0.0f)) {
                // Based off https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
                // Typically you'd have the following:
                // float pdf = D_GGX(NoH, roughness) * NoH / (4.0 * VoH);
                // but since V = N => VoH == NoH
                VFloat pdf = D_GGX(NoH, roughness) / 4.0f + 0.001f;
                // Solid angle of current sample -- bigger for less likely samples
                VFloat omegaS = 1.0f / (numSamples * pdf);
                // Solid angle of texel
                float omegaP = 4.0f * pi / (6 * envTex.Width * envTex.Width);
                // Mip level is determined by the ratio of our sample's solid angle to a texel's solid angle
                VFloat mipLevel = max(0.5f * approx_log2(omegaS / omegaP), 0.0);

                VFloat3 envColor = envTex.SampleCube<SurfaceSampler>(L, mipLevel);

                // Shitty workaround to reduce artifacts around overly bright spots
                envColor.x = min(envColor.x, 50.0f);
                envColor.y = min(envColor.y, 50.0f);
                envColor.z = min(envColor.z, 50.0f);

                prefilteredColor = prefilteredColor + envColor * NoL;
                totalWeight += NoL;
            }
        }
        return prefilteredColor / totalWeight;
    }

    static VFloat2 IntegrateBRDF(VFloat NoV, VFloat roughness) {
        VFloat3 N = { 0, 0, 1 };
        VFloat3 V = { sqrt(1.0f - NoV * NoV), 0.0f, NoV };
        VFloat2 r = { 0.0f, 0.0f };
        const uint32_t sampleCount = 1024;

        for (uint32_t i = 0; i < sampleCount; i++) {
            glm::vec2 Xi = Hammersley(i, sampleCount);
            VFloat3 H = ImportanceSampleGGX(Xi, roughness, N);
            VFloat3 L = 2.0f * dot(V, H) * H - V;

            VFloat VoH = max(0.0f, dot(V, H));
            VFloat NoL = max(0.0f, L.z);
            VFloat NoH = max(0.0f, H.z);

            if (any(NoL > 0.0f)) {
                VFloat G = V_SmithGGXCorrelatedFast(NoV, NoL, roughness * roughness);
                VFloat Gv = G * 4.0f * VoH * NoL / NoH;
                Gv = csel(NoL > 0.0f, Gv, 0.0f);

                // VFloat Fc = pow(1 - VoH, 5.0f);
                VFloat Fc = 1 - VoH;
                VFloat Fc2 = Fc * Fc;
                Fc = Fc * Fc2 * Fc2;

                r.x += Gv * (1 - Fc);
                r.y += Gv * Fc;
            }
        }
        return { r.x * (1.0f / sampleCount), r.y * (1.0f / sampleCount) };
    }

    static VFloat D_GGX(VFloat NoH, VFloat roughness) {
        VFloat a = NoH * roughness;
        VFloat k = roughness / (1.0f - NoH * NoH + a * a);
        return k * k * (1.0f / pi);
    }
    static VFloat V_SmithGGXCorrelatedFast(VFloat NoV, VFloat NoL, VFloat roughness) {
        VFloat a = 2.0f * NoL * NoV;
        VFloat b = NoL + NoV;
        return 0.5f / lerp(a, b, roughness);
    }
    static VFloat3 F_Schlick(VFloat u, VFloat3 f0) {
        VFloat f = pow5(1.0 - u);
        return f + f0 * (1.0f - f);
    }
    static VFloat Fd_Lambert() { return 1.0f / pi; }

    static VFloat pow5(VFloat x) { return x * x * x * x * x; }
    static VFloat3 max3(VFloat3 a, VFloat3 b) { return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) }; }

    static glm::vec2 Hammersley(uint32_t i, uint32_t numSamples) {
        uint32_t bits = i;
        bits = (bits << 16) | (bits >> 16);
        bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
        bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
        bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
        bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
        return { i / (float)numSamples, bits * (1.0 / UINT_MAX) };
    }
    static VFloat3 ImportanceSampleGGX(glm::vec2 Xi, VFloat roughness, VFloat3 N) {
        VFloat a = roughness * roughness;

        VFloat phi = 2.0 * pi * Xi.x;
        VFloat cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
        VFloat sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        VFloat3 H = { cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta };

        // vec3 upVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        VMask nzMask = abs(N.z) < 0.999f;
        VFloat3 upVector = { csel(nzMask, VFloat(0.0f), 1.0f), 0.0, csel(nzMask, VFloat(1.0f), 0.0f) };
        VFloat3 tangentX = normalize(cross(upVector, N));
        VFloat3 tangentY = cross(N, tangentX);
        return tangentX * H.x + tangentY * H.y + N * H.z;
    }

    // https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
    static VFloat2 EnvBRDFApprox(VFloat NoV, VFloat roughness) {
        const VFloat4 c0 = { -1, -0.0275, -0.572, 0.022 };
        const VFloat4 c1 = { 1, 0.0425, 1.04, -0.04 };
        VFloat4 r = roughness * c0 + c1;
        VFloat a004 = min(r.x * r.x, approx_exp2(-9.28 * NoV)) * r.x + r.y;
        return { a004 * -1.04 + r.z, a004 * 1.04 + r.w };
    }

    // https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
    static VFloat3 SrgbToLinear(VFloat3 x) {
        return x * (x * (x * 0.305306011 + 0.682171111) + 0.012522878);
    }
    static VFloat3 Tonemap_Unreal(VFloat3 x) {
        // Unreal 3, Documentation: "Color Grading"
        // Adapted to be close to Tonemap_ACES, with similar range
        // Gamma 2.2 correction is baked in, don't use with sRGB conversion!
        return x / (x + 0.155) * 1.019;
    }
    //static VFloat3 LinearToSrgb(VFloat3 x) {
    //    VFloat3 S1 = sqrt(x);
    //    VFloat3 S2 = sqrt(S1);
    //    VFloat3 S3 = sqrt(S2);
    //    return 0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.0225411470 * x;
    //}

    // Encode normal using signed octahedron + arbitrary extra into 10:10:1 + 10 bits (1-bit is unused)
    // https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
    static VInt SignedOctEncode(VFloat3 n, VFloat w) {
        VFloat m = rcp14(abs(n.x) + abs(n.y) + abs(n.z)) * 0.5f;
        n = n * m;

        VFloat ny = n.y + 0.5;
        VFloat nx = n.x + ny;
        ny = ny - n.x;

        return round2i(nx * 1023.0f) << 22 | round2i(ny * 1023.0f) << 12 | round2i(w * 1023.0f) << 2 | shrl(re2i(n.z), 31);
    }
    static VFloat4 SignedOctDecode(VInt p) {
        float scale = 1.0f / 1023.0f;
        VFloat px = conv2f((p >> 22) & 1023) * scale;
        VFloat py = conv2f((p >> 12) & 1023) * scale;
        VFloat pw = conv2f((p >> 2) & 1023) * scale;

        VFloat nx = px - py;
        VFloat ny = px + py - 1.0f;
        VFloat nz = (1.0 - abs(nx) - abs(ny)) ^ re2f(p << 31);

        return { normalize({ nx, ny, nz }), pw };

        // OutN.z = n.z * 2.0 - 1.0;  // n.z ? 1 : -1
        // OutN.z = OutN.z * (1.0 - abs(OutN.x) - abs(OutN.y));
    }
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

// TODO: Implement possibly better and faster approach from "Scalable Ambient Obscurance" +/or maybe copy a few tricks from XeGTAO or something?
// https://www.shadertoy.com/view/3dK3zR
struct SSAO {
    static const uint32_t KernelSize = 16, FbAttachId = 4;

    float Radius = 1.3f, MaxRange = 0.35f;

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

        fb.IterateTiles([&](uint32_t x, uint32_t y) {
            VInt rng = _randSeed * (int32_t)(x * 12345 + y * 9875);

            VInt iu = (int32_t)x + (VInt::ramp() & 3) * 2;
            VInt iv = (int32_t)y + (VInt::ramp() >> 2) * 2;
            VFloat z = depthMap.SampleDepth(iu, iv, 0);

            if (!any(z < 1.0f)) return; // skip over tiles that don't have geometry

            VFloat4 pos = PerspectiveDiv(TransformVector(invProj, { conv2f(iu), conv2f(iv), z, 1.0f }));

            // TODO: better normal reconstruction - https://atyuwen.github.io/posts/normal-reconstruction/
            VFloat3 posDx = { dFdx(pos.x), dFdx(pos.y), dFdx(pos.z) };
            VFloat3 posDy = { dFdy(pos.x), dFdy(pos.y), dFdy(pos.z) };
            VFloat3 N = normalize(cross(posDx, posDy));

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
                //        depth linearization might be wrong (cam close ups)
                VFloat sampleDist = abs(LinearizeDepth(z) - sampleDepth);

                VMask rangeMask = sampleDist < MaxRange;
                VMask occluMask = sampleDepth <= LinearizeDepth(samplePos.z) - 0.03f;
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
        ApplyBlur(fb);
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
        const int BlurRadius = 2, BlurSamples = BlurRadius * 2 + 1;
        __m512i accum = _mm512_set1_epi16(0);

        for (int32_t so = -BlurRadius; so <= BlurRadius; so++) {
            accum = _mm512_add_epi16(accum, _mm512_cvtepu8_epi16(_mm256_loadu_epi8(&src[so * lineStride])));
        }
        __m256i c = _mm512_cvtepi16_epi8(_mm512_mulhrs_epi16(accum, _mm512_set1_epi16(32767 / BlurSamples)));
        _mm256_storeu_epi8(dst, c);
    }

    static VFloat LinearizeDepth(VFloat d) {
        // TODO: avoid hardcoding this, get from Camera&
        const float zNear = 0.01f, zFar = 1000.0f;
        return (zNear * zFar) / (zFar + d * (zNear - zFar));
    }

    void XorShiftStep(VInt& x) {
        x = x ^ x << 13;
        x = x ^ x >> 17;
        x = x ^ x << 5;
    }
};

}; // namespace renderer