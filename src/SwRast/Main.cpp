#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <tracy/Tracy.hpp>

#include <nfd.h>
#include <nfd_glfw3.h>

#include "Rasterizer.h"
#include "Texture.h"

#include "Camera.h"
#include "Scene.h"
#include "Shading.h"
#include "Voxel.h"

#include <vector>

#define YSON_ENABLE_EXTRA_GLM
#define YSON_ENABLE_EXTRA_STD_UNORDERED_MAP
#include "Yson.h"

#if __linux__
    #include <sys/stat.h>
#endif

struct ModelDesc {
    std::string Path;
    glm::mat4x3 Transform;
};
struct RendererSettings {
    bool OcclusionCulling = true;
    bool EnableDDGI = false;
    bool EnableShadows = false;
    bool DrawProbes = false;

    DebugLayer DebugView = DebugLayer::None;
    float ExposureValue = -15.1f;
    float2 ExposureRange = float2(-4.0f, +8.0f);

    std::unordered_map<ImGuiID, bool> OpenCollapsingHeaders;
};
YSON_SERIALIZER_STRUCT_INLINE(ModelDesc, Path, Transform);
YSON_SERIALIZER_STRUCT_INLINE(Camera, Position, Euler, MoveSpeed);
YSON_SERIALIZER_STRUCT_INLINE(RendererSettings, OcclusionCulling, EnableShadows, EnableDDGI, DrawProbes, DebugView, ExposureValue,
                              ExposureRange, OpenCollapsingHeaders);
YSON_SERIALIZER_STRUCT_INLINE(GIProbeField, ProbeBaseOffset, ProbeSpacing, RaysPerProbe, MaxRayDist, MaxHysteresis, MultiBounce,
                              ForceUpdateAll, NormalBias);

namespace {

GLFWwindow* _window;

swr::FramebufferPtr<swr::pixfmt::RGBA8u> _colorFb;
swr::FramebufferPtr<swr::pixfmt::R32f> _depthFb;
swr::FramebufferPtr<swr::pixfmt::R32f> _shadowDepthFb;

std::unique_ptr<swr::Rasterizer> _rast;
swr::TexturePtr2D<swr::pixfmt::R32f> _prevDepthMap;
GLuint _fbTexture = 0;

Camera _cam;
RendererSettings _settings;
std::unique_ptr<Scene> _scene;
swr::HdrTexture2D::Ptr _skyboxTex;
std::unique_ptr<BrickMap> _brickmap;
std::unique_ptr<GIProbeField> _probes;

SWR_PERF_DEF_COUNTER(MeshCull);
SWR_PERF_DEF_COUNTER(FrameElapsed);

GLuint CreateTexture(uint32_t width, uint32_t height, uint32_t mipLevels, GLuint fmt) {
    GLuint handle;
    glCreateTextures(GL_TEXTURE_2D, 1, &handle);

    glTextureStorage2D(handle, (GLsizei)mipLevels, fmt, (GLsizei)width, (GLsizei)height);

    glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTextureParameteri(handle, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(handle, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return handle;
}

void CreateFramebuffer(uint32_t width, uint32_t height) {
    _depthFb = swr::CreateFramebuffer<swr::pixfmt::R32f>(width, height);
    _colorFb = swr::CreateFramebuffer<swr::pixfmt::RGBA8u>(width, height);

    uint32_t halfW = 1 << (32 - simd::lzcnt((width - 1) / 2));
    uint32_t halfH = 1 << (32 - simd::lzcnt((height - 1) / 2));
    _prevDepthMap = swr::CreateTexture2D<swr::pixfmt::R32f>(halfW, halfH, 16);

    if (_fbTexture != 0) glDeleteTextures(1, &_fbTexture);
    _fbTexture = CreateTexture(_colorFb->Width, _colorFb->Height, 1, GL_RGBA8);
}
void LoadScene(const std::vector<ModelDesc>& models) {
    ZoneScoped;
    _scene = std::make_unique<Scene>();
    for (auto& model : models) {
        _scene->ImportGltf(model.Path);
    }

    if (_scene->Lights.size() == 0) {
        uint32_t offset = _scene->Lights.AllocRange(1);

        _scene->Lights[offset] = {
            .Type = Light::kTypeDirectional,
            .Direction = glm::normalize(-float3(0.589494, 0.684509, 0.428906)),
            .Color = float3(1, 1, 1),
            .Intensity = 5.0,
        };
    }

    _brickmap = std::make_unique<BrickMap>(4, 16.0f);
    _brickmap->VoxelizeScene(*_scene, float3(0));
}
void LoadSkybox(const std::string& path) {
    ZoneScoped;
    _skyboxTex = swr::texutil::LoadOctahedronFromPanoramaHDR(path.c_str()); //
}

void InitRenderer() {
    LoadSkybox("assets/skyboxes/sunflowers_puresky_4k.hdr");
    ShadingContext::LoadBlueNoise();

    _rast = std::make_unique<swr::Rasterizer>(1);

    _shadowDepthFb = swr::CreateFramebuffer<swr::pixfmt::R32f>(1024, 1024, 1, 1);
    _probes = std::make_unique<GIProbeField>();
    _probes->ProbeBaseOffset = -float3(GIProbeField::GridSize) * _probes->ProbeSpacing / 2.0f;

    std::string rawSettings = R"({
        Models: [{ Path: "assets/models/Sponza/Sponza.gltf" }],
        Camera: {  Position: [-8.8, 1.3, -0.25],  Euler: [1.54, -0.10],  MoveSpeed: 5  },
        Rasterizer: { NumThreads: 4,   Resolution: [1920, 1080] },
        ProbeGI: { ProbeBaseOffset: [-12, -0.5, -12.25],  ProbeSpacing: 1.5, }
    })";

    if (FILE* fs = fopen("priv/settings.yml", "rb")) {
        fseek(fs, 0, SEEK_END);
        rawSettings.resize((size_t)ftell(fs));
        fseek(fs, 0, SEEK_SET);

        fread(rawSettings.data(), rawSettings.size(), 1, fs);
        fclose(fs);
    }

    auto rd = yson::Reader(rawSettings);
    rd.ReadExpect(yson::kTypeObject);

    while (rd.ReadNext()) {
        if (rd.MatchObject("Camera")) {
            rd.Parse(_cam);
        }  //
        else if (rd.MatchObject("Renderer")) {
            rd.Parse(_settings);
        }  //
        else if (rd.MatchArray("Models")) {
            LoadScene(rd.Parse<std::vector<ModelDesc>>());
        }  //
        else if (rd.MatchObject("ProbeGI")) {
            rd.Parse(*_probes);
        }  //
        else if (rd.MatchObject("Rasterizer")) {
            while (rd.ReadNext()) {
                if (rd.Match("NumThreads", yson::kTypeNumber)) _rast->SetThreadCount(rd.GetU32());
                if (rd.Match("Resolution", yson::kTypeArray)) {
                    uint2 res = rd.Parse<uint2>();
                    CreateFramebuffer(res.x, res.y);
                }
            }
        }
    }

    _probes->Reset(true);
}
void SaveSettings() {
    auto wr = yson::Writer();
    wr.BeginObject();

    wr.Write("Camera", _cam);
    wr.Write("Renderer", _settings);
    wr.Write("ProbeGI", *_probes);

    wr.BeginObject("Rasterizer");
    wr.WriteInt("NumThreads", _rast->GetThreadCount());
    wr.Write("Resolution", uint2(_colorFb->Width, _colorFb->Height));
    wr.EndObject();

    wr.BeginArray("Models");
    for (auto& model : _scene->Models) {
        wr.Write(ModelDesc {
            .Path = model->Path,
            .Transform = model->RootNode.GlobalTransform,
        });
    }
    wr.EndArray();

    wr.EndObject();

#if __linux__
    mkdir("./priv", 0777);
#endif

    FILE* fs = fopen("priv/settings.yml", "wb");
    if (!fs) return;
    fwrite(wr.Buffer.data(), wr.Buffer.size(), 1, fs);
    fclose(fs);
}

void DrawModel(Model& model, ShadingContext& shader, const swr::ShaderDispatchTable& dispatcher,
               std::vector<InstanceData>* instances,bool enableOcclusionCull) {
    ZoneScoped;

    for (auto& node : model.Nodes) {
        if (node.MeshCount == 0) continue;

        constexpr uint32_t batchSize = std::min(16384u, ShadingContext::MaxMeshletsPerDraw);
        v_mask cullBitmap[batchSize / simd::vec_width];

        for (uint32_t i = 0; i < node.MeshCount; i += batchSize) {
            uint32_t meshCount = std::min(node.MeshCount - i, batchSize);

            SWR_PERF_BEGIN(MeshCull);
            uint32_t visibleCount =
                shader.CullMeshlets(cullBitmap, &_scene->Meshlets[node.MeshOffset + i], meshCount, node.GlobalTransform,
                                    float2(_colorFb->Width, _colorFb->Height), enableOcclusionCull ? _prevDepthMap.get() : nullptr);
            SWR_PERF_END(MeshCull);

            if (visibleCount > 0) {
                shader.InstanceMeshlets = &_scene->Meshlets[node.MeshOffset + i];
                shader.InstanceCullBitmap = cullBitmap;
                shader.ObjectToWorldMat = node.GlobalTransform;
                shader.ObjectToClipMat = shader.WorldToClipMat * float4x4(node.GlobalTransform);

                if (instances != nullptr) {
                    shader.InstanceId = instances->size();
                    instances->push_back({ .Meshlets = shader.InstanceMeshlets, .Transform = node.GlobalTransform });
                }
                _rast->DrawMeshlets(meshCount, { dispatcher, &shader });
            }
        }
    }
}

constexpr uint32_t MaxShadowCascades = 4;

void ComputeShadowMatrices(const Camera& cam, float3 lightDir, uint32_t renderSize, float4x4 worldToShadowMat[MaxShadowCascades]) {
    float splitDist[MaxShadowCascades] = { 2.0f, 5.0f, 12.0f, 20.0f };
    float maxDist = 30.0f;

    float3 frustumCorners[8];
    float4x4 clipToWorldMat = glm::inverse(cam.GetProjMatrix() * cam.GetViewMatrix(true));

    for (int i = 0; i < 8; i++) {
        int3 corner = i >> int3(0, 1, 2) & 1;
        float4 pos = clipToWorldMat * float4(int2(corner) * 2 - 1, corner.z ? cam.NearZ / maxDist : 1, 1);
        frustumCorners[i] = float3(pos) / pos.w;
    }

    for (uint32_t cascadeIdx = 0; cascadeIdx < MaxShadowCascades; cascadeIdx++) {
        float tSplit0 = cascadeIdx == 0 ? 0 : splitDist[cascadeIdx - 1] / maxDist;
        float tSplit1 = splitDist[cascadeIdx] / maxDist;

        float3 corners[8];
        float3 center = float3(0);

        for (uint32_t i = 0; i < 4; i++) {
            corners[i + 0] = glm::mix(frustumCorners[i], frustumCorners[i + 4], tSplit0);
            corners[i + 4] = glm::mix(frustumCorners[i], frustumCorners[i + 4], tSplit1);
            center += (corners[i + 0] + corners[i + 4]);
        }
        center /= 8;

        // Using radius as projection bounds prevents shimmering on camera rotation
        float radius = 0;
        for (uint32_t i = 0; i < 8; i++) {
            radius = glm::max(radius, glm::length(corners[i] - center));
        }
        radius /= 2;

        float4x4 proj = glm::ortho(-radius, +radius, -radius, +radius, 0.0f, radius * 12);
        float4x4 view = glm::lookAt(center + lightDir * radius, center, float3(0, 1, 0));
        float4x4 viewProj = proj * view;

        // Texel-space stabilization
        float halfRenderSize = renderSize * 0.5f;
        float2 clipOrigin = viewProj * float4(0, 0, 0, halfRenderSize);
        float2 snapOffset = (clipOrigin - glm::round(clipOrigin)) / halfRenderSize;

        viewProj[3][0] -= snapOffset.x;
        viewProj[3][1] -= snapOffset.y;

        worldToShadowMat[cascadeIdx] = viewProj;
    }
}

// https://github.com/ocornut/imgui/issues/6215
bool CollapsingHeaderWithSavedState(const char* label, bool defaultState) {
    ImGuiID id = ImGui::GetID(label);
    auto [savedState, firstTime] = _settings.OpenCollapsingHeaders.emplace(ImGui::GetID(label), false);

    ImGui::SetNextItemOpen(firstTime ? defaultState : savedState->second, ImGuiCond_Once);
    return savedState->second = ImGui::CollapsingHeader(label);
}

void RenderFrame() {
    FrameMark;

    ImGui::Begin("Settings");
    
    ImGui::SeparatorText("Scene");

    if (ImGui::Button("Import Model")) {
        nfdu8filteritem_t filters[1] = { { "GLTF models", "gltf,glb" } };
        nfdopendialogu8args_t args = {
            .filterList = filters,
            .filterCount = 1,
        };
        NFD_GetNativeWindowFromGLFWWindow(_window, &args.parentWindow);
        nfdu8char_t* outPath;
        nfdresult_t result = NFD_OpenDialogU8_With(&outPath, &args);

        if (result == NFD_OKAY) {
            LoadScene({ { .Path = std::string(outPath) } });
            NFD_FreePathU8(outPath);
        }
    }

    if (CollapsingHeaderWithSavedState("Rasterizer", true)) {
        char currRes[64];
        snprintf(currRes, sizeof(currRes), "%dx%d", _colorFb->Width, _colorFb->Height);
        if (ImGui::BeginCombo("Resolution", currRes)) {
            if (ImGui::Selectable("Window Size")) {
                int2 size;
                glfwGetFramebufferSize(_window, &size.x, &size.y);
                // Min alignment is 4x4 but GetPixels() will explode if the row stride isn't cache aligned, so ...16x4 it is.
                size = glm::clamp((size + int2(15, 3)) & ~int2(15, 3), 32, (int)swr::Rasterizer::MaxRenderSize);
                CreateFramebuffer((uint32_t)size.x, (uint32_t)size.y);

                float2 scale;
                glfwGetWindowContentScale(_window, &scale.x, &scale.y);
                glfwSetWindowSize(_window, (int)(size.x / scale.x + 0.5f), (int)(size.y / scale.y + 0.5f));
            }
            if (ImGui::Selectable("2880x1620")) CreateFramebuffer(2880, 1620);
            if (ImGui::Selectable("2560x1440")) CreateFramebuffer(2560, 1440);
            if (ImGui::Selectable("1920x1080")) CreateFramebuffer(1920, 1080);
            if (ImGui::Selectable("1280x720")) CreateFramebuffer(1280, 720);
            if (ImGui::Selectable("848x480")) CreateFramebuffer(848, 480);
            if (ImGui::Selectable("320x240")) CreateFramebuffer(320, 240);
            ImGui::EndCombo();
        }

        int numThreads = (int)_rast->GetThreadCount();
        if (ImGui::SliderInt("Threads", &numThreads, 1, (int)std::thread::hardware_concurrency())) {
            _rast->SetThreadCount((uint32_t)numThreads);
        }
        ImGui::Checkbox("Binning", &_rast->EnableBinning);
        ImGui::Checkbox("Clipping", &_rast->EnableClipping);
        ImGui::Checkbox("Guardband", &_rast->EnableGuardband);
        ImGui::Checkbox("Intersect Bins", &_rast->EnableBinIntersect);
    }

    static AutoExposureData s_ExposureData = {};
    bool autoExposure = _settings.ExposureValue < -15.0f;

    if (CollapsingHeaderWithSavedState("Renderer", true)) {
        ImGui::Checkbox("Occlusion Culling", &_settings.OcclusionCulling);

        {
            const int count = std::size(kDebugLayerNames);
            ImGui::Combo("Debug View", (int*)&_settings.DebugView, kDebugLayerNames, count);

            int idx = (int)_settings.DebugView;
            if (ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_1)) _settings.DebugView = (DebugLayer)((idx == 0 ? count : idx) - 1);
            if (ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_2)) _settings.DebugView = (DebugLayer)((idx + 1 < count ? idx + 1 : 0));
        }

        ImGui::SliderFloat("Exposure", &_settings.ExposureValue, -15.1f, 15.0f, autoExposure ? "Auto" : "%+.1f");
        ImGui::DragFloatRange2("Exposure Range", &_settings.ExposureRange.x, &_settings.ExposureRange.y, 0.1f, -12.0f, 12.0f, "%.3f");

        if (!autoExposure) {
            s_ExposureData.Exposure = 1.0f / exp2f(-_settings.ExposureValue) * 1.2f;  // EV100 -> Lmax
        }
        s_ExposureData.HistRange = _settings.ExposureRange;

        ImGui::SeparatorText("Camera");
        ImGui::SliderFloat("Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

        ImGui::DragScalarN("Pos", ImGuiDataType_Double, &_cam.Position, 3, 0.01f * _cam.MoveSpeed, nullptr, nullptr, "%.4f");

        if (ImGui::IsItemActive() && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_C)) {
            char text[128];
            snprintf(text, sizeof(text), ".Position = { %.5f, %.5f, %.5f }, .Euler = { %.4f, %.4f }",  //
                     _cam.Position.x, _cam.Position.y, _cam.Position.z, _cam.Euler.x, _cam.Euler.y);
            ImGui::SetClipboardText(text);
        }
        ImGui::DragScalarN("Rot", ImGuiDataType_Float, &_cam.Euler, 2, 0.01f);
    }

    float4x4 prevCamViewMat = _cam.GetViewMatrix(true);

    _cam.Update(Camera::GetInputsFromImGui());

    float4x4 camProjMat = _cam.GetProjMatrix();
    float4x4 camViewMat = _cam.GetViewMatrix(true);
    float4x4 camProjViewMat = camProjMat * camViewMat;

    SWR_PERF_BEGIN(FrameElapsed);

    ShadingContext shader = {
        .ColorBuffer = _colorFb.get(),
        .Materials = _scene->Materials.data(),
        .Lights = _scene->Lights.data(),
        .NumLights = _scene->Lights.size(),

        .ViewPos = _cam.ViewPosition,
        .PrevViewMat = prevCamViewMat,

        .SkyboxTex = _skyboxTex.get(),
        .BrickMap = _brickmap.get(),
        .Exposure = s_ExposureData.Exposure,
        .AutoExp = autoExposure ? &s_ExposureData : nullptr,
    };

    Light* sunLight = _scene->Lights[0].Type == Light::kTypeDirectional ? &_scene->Lights[0] : nullptr;
    static float shadowSampleRadius = 3;
    static int shadowCascadeIdx = 2;
    static bool shadowDebug = false;

    if (CollapsingHeaderWithSavedState("Shadows", _settings.EnableShadows)) {
        if (sunLight != nullptr) {
            ImGui::Checkbox("Enabled##Shadows", &_settings.EnableShadows);

            float3 dir = sunLight->Direction;
            static float2 angle = float2(atan2f(dir.z, dir.x), asinf(dir.y));
            if (ImGui::DragFloat2("Direction", &angle.x, 0.01f)) {
                float r = cosf(angle.y);
                sunLight->Direction = float3(cosf(angle.x) * r, sinf(angle.y), sinf(angle.x) * r);
            }
        }
        ImGui::DragFloat("Sample Radius", &shadowSampleRadius, 0.01, 1, 10, "%.1f");
        ImGui::SliderInt("Active Cascade", &shadowCascadeIdx, 0, 3);
        ImGui::Checkbox("Debug", &shadowDebug);

        ImGui::DragFloat("Intensity", &sunLight->Intensity, 0.1f, 0, 1000);
    }

    static bool s_HalfRes = false;

    if (CollapsingHeaderWithSavedState("ProbeGI", _settings.EnableDDGI)) {
        ImGui::Checkbox("Enabled##DDGI", &_settings.EnableDDGI);
        ImGui::Checkbox("Draw Probes", &_settings.DrawProbes);
        ImGui::Checkbox("HalfRes Resolve", &s_HalfRes);

        bool reinitProbes = false;
        reinitProbes |= ImGui::DragFloat3("Probe Offset", &_probes->ProbeBaseOffset.x, 0.1f);
        reinitProbes |= ImGui::DragFloat("Probe Spacing", &_probes->ProbeSpacing, 0.1f, 0.1f, 5.0f);
        if (reinitProbes) _probes->Reset(false);

        ImGui::SliderInt("Rays/Probe", (int*)&_probes->RaysPerProbe, 16, 128);
        ImGui::SliderInt("Max Hysteresis", (int*)&_probes->MaxHysteresis, 1, 128);
        ImGui::DragFloat("Ray Max Dist", &_probes->MaxRayDist, 0.1f, 1.0f, 100.0f);
        ImGui::DragFloat("Normal Bias", &_probes->NormalBias, 0.001, 0.01f, 0.5f);

        static float skyboxIntensity = 1;
        ImGui::DragFloat("Skybox Intensity", &skyboxIntensity, 0.01, 0.001f, 100.0f);
        shader.SkyboxIntensity = skyboxIntensity;

        ImGui::Checkbox("Multi-Bounce", &_probes->MultiBounce);
        ImGui::Checkbox("Force Update All", &_probes->ForceUpdateAll);
    }

    if (sunLight != nullptr && _settings.EnableShadows) {
        ZoneScopedN("ShadowMap");
        SWR_PERF_DEF_COUNTER(ShadowElapsed);
        SWR_PERF_BEGIN(ShadowElapsed);

        ComputeShadowMatrices(_cam, sunLight->Direction, _shadowDepthFb->Width, shader.ShadowProjMatrices);
        shader.ShadowDepthMap = _shadowDepthFb.get();
        shader.ShadowSampleRadius = shadowSampleRadius;
        shader.ShadowProjMatrices[0] = shader.ShadowProjMatrices[shadowCascadeIdx];  // TODO: sample cascades

        _shadowDepthFb->Clear(0);
        shader.DepthBuffer = _shadowDepthFb.get();
        _rast->SetViewport(uint2(_shadowDepthFb->Width, _shadowDepthFb->Height));

        shader.ProjMat = float4x4(1);
        shader.ViewMat = float4x4(1);
        shader.WorldToClipMat = shader.ShadowProjMatrices[0];

        for (auto& model : _scene->Models) {
            DrawModel(*model, shader, ShadingContext::ShadowGenShader, nullptr, false);
        }
        SWR_PERF_END(ShadowElapsed);
    }

    bool debugOverdraw = _settings.DebugView == DebugLayer::OverdrawPixel || _settings.DebugView == DebugLayer::OverdrawQuad;
    auto& dispatchTable = debugOverdraw ? ShadingContext::OverdrawShader : ShadingContext::VisBufferShader;

    {
        ZoneScopedN("ClearFb");
        _colorFb->Clear(0);
        _depthFb->Clear(0);

        shader.DepthBuffer = _depthFb.get();
        _rast->SetViewport(uint2(_colorFb->Width, _colorFb->Height));
    }

    shader.ProjMat = camProjMat;
    shader.ViewMat = camViewMat;
    shader.WorldToClipMat = camProjViewMat;
    shader.ProbeField = _settings.EnableDDGI ? _probes.get() : nullptr;

    auto instances = std::vector<InstanceData>();
    instances.resize(1); // reserve 0th instance id for "null"
    instances.reserve(1024);

    for (auto& model : _scene->Models) {
        DrawModel(*model, shader, dispatchTable, &instances, _settings.OcclusionCulling);
    }

    SWR_PERF_DEF_COUNTER(Resolve);
    SWR_PERF_BEGIN(Resolve);

    shader.DepthBuffer = _depthFb.get();
    shader.Instances = instances.data();

    if (_settings.DebugView == DebugLayer::None) {
        ZoneScopedN("Resolve");

        if (s_HalfRes && _settings.EnableDDGI) {
            shader.ResolveHalfRes(*_rast);
        } else {
            shader.Resolve(*_rast);
        }
        shader.DrawPointLightsDebug();

        if (autoExposure) {
            s_ExposureData.UpdateExposure(ImGui::GetIO().DeltaTime, 1.1f);
        }
    } else if (_settings.DebugView != DebugLayer::None) {
        shader.ResolveDebug(*_rast, _settings.DebugView);
    }
    SWR_PERF_END(Resolve);

    if (_settings.EnableDDGI) {
        ZoneScopedN("ProbeUpdate");

        SWR_PERF_DEF_COUNTER(ProbeUpdateElapsed);
        SWR_PERF_BEGIN(ProbeUpdateElapsed);
        shader.UpdateProbes(*_rast);
        SWR_PERF_END(ProbeUpdateElapsed);

        if (_settings.DrawProbes) {
            shader.DrawProbesDebug();
        }
    }

    ImGui::End();

    if (shadowDebug) {
        uint2 size = uint2(_colorFb->Width / 3, _colorFb->Height / 3);
        uint2 offset = uint2(_colorFb->Width - size.x, _colorFb->Height - size.y) & ~3u;

        for (uint32_t y = 0; y < size.y; y += 4) {
            for (uint32_t x = 0; x < size.x; x += 4) {
                v_int2 pos = int2(x, y) + swr::TilePixelOffsets;
                pos = simd::conv<int>(simd::conv<float>(pos) * float2(_shadowDepthFb->Width / (float)size.x, _shadowDepthFb->Height / (float)size.y));
                v_float depth = simd::as<v_float>(_shadowDepthFb->Fetch(pos));

                v_float3 color = depth;
                _colorFb->WriteTile(swr::pixfmt::RGBA8u::Pack({ color, 1 }), offset.x + x, offset.y + y);
            }
        }
    }

    if (_settings.OcclusionCulling) {
        ZoneScopedN("BuildHiZ");

        SWR_PERF_BEGIN(MeshCull);
        swr::texutil::DownsampleDepth(*_depthFb, *_prevDepthMap);
        SWR_PERF_END(MeshCull);
    }
    {
        ZoneScopedN("UploadFrameToGPU");
        
        GLuint pbo;
        glCreateBuffers(1, &pbo);
        glNamedBufferStorage(pbo, _colorFb->Width * _colorFb->Height * 4, nullptr, GL_MAP_WRITE_BIT);

        uint32_t* pixels = (uint32_t*)glMapNamedBuffer(pbo, GL_WRITE_ONLY); // spec guarantees align >= 64
        _colorFb->GetPixels(pixels, _colorFb->Width);
        glUnmapNamedBuffer(pbo);

        // This copy could be avoided by reading buffer data directly from shader, but this is good enough.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTextureSubImage2D(_fbTexture, 0, 0, 0, (GLsizei)_colorFb->Width, (GLsizei)_colorFb->Height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glDeleteBuffers(1, &pbo);
    }

    SWR_PERF_END(FrameElapsed);

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddImage((ImTextureID)_fbTexture, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 0), ImVec2(1, 1));

    swr::perf::FlushThreadCounters();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::PushStyleVarY(ImGuiStyleVar_ItemSpacing, 0);
    ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize);

    swr::perf::EndFrame();

    double frameElapsedAvg = swr::perf::GetAverage("FrameElapsed");
    ImGui::Text("Frame: %.1fms (%.0f FPS), Cull: %.2fms, Resolve: %.2fms",   //
                frameElapsedAvg / 1e6,                                       //
                1e9 / frameElapsedAvg,                                       //
                swr::perf::GetAverage("MeshCull") / 1e6,  //
                swr::perf::GetAverage("Resolve") / 1e6);

    ImGui::Text("Raster: %.2fms (%.1fK/%.1fK tris, %.1fK clip, %.1fK bins)",  //
                swr::perf::GetAverage("RastTotalElapsed") / 1e6,              //
                swr::perf::GetCurrent("RastTrisDrawn") / 1e3,                 //
                swr::perf::GetCurrent("RastTrisProcessed") / 1e3,             //
                swr::perf::GetCurrent("RastTrisClipped") / 1e3,               //
                swr::perf::GetCurrent("RastBinsDispatched") / 1e3);

    ImGui::Text("Shadows: %.2fms, Probes: %.2fms",             //
                swr::perf::GetAverage("ShadowElapsed") / 1e6,  //
                swr::perf::GetAverage("ProbeUpdateElapsed") / 1e6);
    ImGui::End();
    ImGui::PopStyleVar();
}

};

int main(int argc, char** args) {
    if (!glfwInit()) return -1;
    NFD_Init();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    _window = glfwCreateWindow(1280, 720, "Glimpsw", NULL, NULL);
    if (!_window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1); // Enable vsync

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "priv/imgui.ini";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    io.Fonts->AddFontFromFileTTF("assets/Roboto-Medium.ttf", 16.0f);

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 2;
    style.GrabRounding = 3;
    style.ScrollbarSize = 8;
    style.ScrollbarPadding = 1;
    style.FramePadding.y = 2;
    style.SeparatorTextBorderSize = 1;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]               = ImVec4(0.12f, 0.12f, 0.14f, 0.93f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.19f, 0.19f, 0.22f, 0.93f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.30f, 0.33f, 0.37f, 0.54f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.53f, 0.58f, 0.64f, 0.40f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.53f, 0.58f, 0.64f, 0.54f);
    colors[ImGuiCol_CheckboxSelectedBg]     = ImVec4(0.30f, 0.33f, 0.37f, 0.54f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 0.31f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.02f, 0.02f, 0.02f, 0.20f);

    float scale;
    glfwGetWindowContentScale(_window, &scale, nullptr);

    if (scale != (int)scale) {
        scale = 1.0f + (scale - (int)scale);
        style.FontScaleDpi = scale;
        style.ScaleAllSizes(scale);
    }

    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init();

    InitRenderer();

    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        ImGuiIO& io = ImGui::GetIO();

        // On platforms like Wayland, fractional FB scaling makes text and lines blurry and shimmery.
        // We'll do a little hacking to force ImGui render everything at int scale, and adjust style at startup.
        ImVec2 fbScale = io.DisplayFramebufferScale;
        ImVec2 fbScaleI = ImVec2((int)fbScale.x, (int)fbScale.y);
        if (fbScale.x != fbScaleI.x || fbScale.y != fbScaleI.y) {
            fbScale.x = 1.0f + (fbScale.x - fbScaleI.x);
            fbScale.y = 1.0f + (fbScale.y - fbScaleI.y);

            io.DisplaySize.x *= fbScale.x;
            io.DisplaySize.y *= fbScale.y;
            io.DisplayFramebufferScale = fbScaleI;

            for (auto& event : ImGui::GetCurrentContext()->InputEventsQueue) {
                static ImU32 _lastRescaledMouseEventId = 0;
                if (event.Type == ImGuiInputEventType_MousePos && event.EventId > _lastRescaledMouseEventId) {
                    event.MousePos.PosX *= fbScale.x;
                    event.MousePos.PosY *= fbScale.y;
                    // ImGui will occasionally defer events due to the trickling rule (see ImGui::UpdateInputEvents()).
                    // This ensures we only rewrite the same event once.
                    _lastRescaledMouseEventId = event.EventId;
                }
            }
        }

        ImGui::NewFrame();

        if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
            static int winRect[4];
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();

            if (!glfwGetWindowMonitor(_window)) {
                glfwGetWindowPos(_window, &winRect[0], &winRect[1]);
                glfwGetWindowSize(_window, &winRect[2], &winRect[3]);

                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                glfwSetWindowMonitor(_window, monitor, 0, 0, mode->width, mode->height, 0);
            } else {
                glfwSetWindowMonitor(_window, nullptr, winRect[0], winRect[1], winRect[2], winRect[3], 0);
            }
            glfwSwapInterval(1);
        }

        int display_w, display_h;
        glfwGetFramebufferSize(_window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        RenderFrame();

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(_window);
    }

    SaveSettings();
    glDeleteTextures(1, &_fbTexture);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    NFD_Quit();
    glfwTerminate();
    return 0;
}
