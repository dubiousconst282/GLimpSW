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
    bool DrawProbes = false;

    bool EnableShadows = true;
    bool DisableShadowCache = false;
    bool DebugShadows = false;
    uint32_t ShadowCascadeResolution = 2048;
    uint32_t ShadowCascadeCount = 4;
    float3 SunDirection = glm::normalize(float3(-0.413, -0.859, -0.300));

    DebugLayer DebugView = DebugLayer::None;
    float ExposureValue = -15.1f;
    float2 ExposureRange = float2(-4.0f, +8.0f);

    float SunIntensity = 5.0f;
    float AmbientFactor = 0.02f;
    float SkyboxIntensity = 1.0f;

    std::unordered_map<ImGuiID, bool> OpenCollapsingHeaders;
};
YSON_SERIALIZER_STRUCT_INLINE(ModelDesc, Path, Transform);
YSON_SERIALIZER_STRUCT_INLINE(Camera, Position, Euler, MoveSpeed);
YSON_SERIALIZER_STRUCT_INLINE(RendererSettings, OcclusionCulling,               //
                              EnableDDGI, DrawProbes,                           //
                              EnableShadows, DebugShadows, DisableShadowCache,  //
                              ShadowCascadeResolution, ShadowCascadeCount, SunDirection,
                              DebugView,                                                                 //
                              ExposureValue, ExposureRange, SunIntensity, AmbientFactor, SkyboxIntensity,  //
                              OpenCollapsingHeaders);

YSON_SERIALIZER_STRUCT_INLINE(ShadowClipmap, BaseCascadeRange, SampleRadius, SampleCount);
YSON_SERIALIZER_STRUCT_INLINE(GIProbeField, ProbeBaseOffset, ProbeSpacing, RaysPerProbe, MaxRayDist, MaxHysteresis, MultiBounce,
                              ForceUpdateAll, NormalBias);

namespace {

GLFWwindow* _window;

swr::FramebufferPtr<swr::pixfmt::RGBA8u> _colorFb;
swr::FramebufferPtr<swr::pixfmt::R32f> _depthFb;

std::unique_ptr<swr::Rasterizer> _rast;
swr::TexturePtr2D<swr::pixfmt::R32f> _hierDepth;
GLuint _fbTexture = 0;

Camera _cam;
RendererSettings _settings;
std::unique_ptr<Scene> _scene;
swr::HdrTexture2D::Ptr _skyboxTex;

std::unique_ptr<ShadowClipmap> _shadowMap;
std::unique_ptr<BrickMap> _brickmap;
std::unique_ptr<GIProbeField> _probes;

SWR_PERF_DEF_COUNTER(Culling);
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
    _hierDepth = swr::CreateTexture2D<swr::pixfmt::R32f>(halfW, halfH, 16);

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
            .Direction = _settings.SunDirection,
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

    _shadowMap = std::make_unique<ShadowClipmap>();
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
        else if (rd.MatchObject("ShadowMap")) {
            rd.Parse(*_shadowMap);
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

    _shadowMap->SetResolution(_settings.ShadowCascadeResolution, _settings.ShadowCascadeCount);
    _probes->Reset(true);
}
void SaveSettings() {
    auto wr = yson::Writer();
    wr.BeginObject();

    wr.Write("Camera", _cam);
    wr.Write("Renderer", _settings);
    wr.Write("ShadowMap", *_shadowMap);
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

struct DrawCallBuilder {
    MeshletCuller Culler;
    bool GenerateInstanceData = true;

    std::vector<DrawContext> DrawCalls;
    std::vector<InstanceData> Instances = { InstanceData() };  // 0th instance id must be reserved for "null"
    std::vector<std::unique_ptr<uint32_t[]>> CullBitmaps;

    void AddModel(Model& model, DrawContext& templateDraw);
};

void DrawCallBuilder::AddModel(Model& model, DrawContext& templateDraw) {
    ZoneScoped;

    uint32_t cullBitmapPos = 0, cullBitmapSize = 0;
    for (ModelNode& node : model.Nodes) {
        cullBitmapSize += (node.MeshCount + 31) / 32;
    }
    auto cullBitmap = std::make_unique<uint32_t[]>(cullBitmapSize);

    for (ModelNode& node : model.Nodes) {
        if (node.MeshCount == 0) continue;

        for (uint32_t i = 0; i < node.MeshCount; i += DrawContext::MaxMeshletsPerDraw) {
            uint32_t meshCount = std::min(node.MeshCount - i, DrawContext::MaxMeshletsPerDraw);
            Meshlet* meshes = &_scene->Meshlets[node.MeshOffset + i];

            SWR_PERF_BEGIN(Culling);
            assert(cullBitmapPos < cullBitmapSize);
            uint32_t visibleCount = Culler.CullMeshlets(&cullBitmap[cullBitmapPos], meshes, meshCount, node.GlobalTransform);
            SWR_PERF_END(Culling);

            if (visibleCount == 0) continue;
                
            DrawContext& draw = DrawCalls.emplace_back(templateDraw);
            draw.NumMeshlets = meshCount;
            draw.InstanceId = Instances.size();
            draw.Meshlets = meshes;
            draw.ObjectToClipMat = draw.ObjectToClipMat * float4x4(node.GlobalTransform);

            draw.CullBitmap = &cullBitmap[cullBitmapPos];
            cullBitmapPos += (meshCount + 31) / 32u;

            if (GenerateInstanceData) {
                Instances.push_back({ .Meshlets = draw.Meshlets, .Transform = node.GlobalTransform });
            }
        }
    }
    CullBitmaps.emplace_back(std::move(cullBitmap));
}

// https://github.com/ocornut/imgui/issues/6215
bool CollapsingHeaderWithSavedState(const char* label, bool defaultState) {
    auto [savedState, firstTime] = _settings.OpenCollapsingHeaders.emplace(ImGui::GetID(label), false);

    ImGui::SetNextItemOpen(firstTime ? defaultState : savedState->second, ImGuiCond_Once);
    return savedState->second = ImGui::CollapsingHeader(label);
}

void RenderShadowMap();
void DrawShadowDebug();

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
    }

    if (!autoExposure) {
        s_ExposureData.Exposure = 1.0f / exp2f(-_settings.ExposureValue) * 1.2f;  // EV100 -> Lmax
    }
    s_ExposureData.HistRange = _settings.ExposureRange;

    if (CollapsingHeaderWithSavedState("Camera", false)) {
        ImGui::SliderFloat("Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::DragScalarN("Pos", ImGuiDataType_Double, &_cam.Position, 3, 0.01f * _cam.MoveSpeed, nullptr, nullptr, "%.4f");

        if (ImGui::IsItemActive() && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_C)) {
            char text[128];
            snprintf(text, sizeof(text), "Position: [%.5f, %.5f, %.5f], Euler: [%.4f, %.4f]",  //
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

    Light* sunLight = _scene->Lights[0].Type == Light::kTypeDirectional ? &_scene->Lights[0] : nullptr;

    if (CollapsingHeaderWithSavedState("Shadows", _settings.EnableShadows)) {
        if (sunLight != nullptr) {
            ImGui::Checkbox("Enabled##Shadows", &_settings.EnableShadows);

            float3 dir = _settings.SunDirection;
            static float2 angle = float2(atan2f(dir.z, dir.x), asinf(dir.y));

            if (ImGui::DragFloat2("Direction", &angle.x, 0.01f)) {
                float r = cosf(angle.y);
                _settings.SunDirection = float3(cosf(angle.x) * r, sinf(angle.y), sinf(angle.x) * r);
                sunLight->Direction = _settings.SunDirection;
            }
            ImGui::DragFloat("Intensity", &_settings.SunIntensity, 0.1f, 0, 1000);
            ImGui::DragFloat("Ambient Factor", &_settings.AmbientFactor, 0.001f, 0, 100);

            sunLight->Intensity = _settings.SunIntensity;
        }

        int resIdx = 31 - 9 - (int)simd::lzcnt(_settings.ShadowCascadeResolution);
        bool wantChangeRes = false;
        wantChangeRes |= ImGui::Combo("Resolution##Shadows", &resIdx, "512x512\0" "1024x1024\0" "2048x2048\0");
        wantChangeRes |= ImGui::SliderInt("Cascades", (int*)&_settings.ShadowCascadeCount, 1, ShadowClipmap::MaxCascades);

        if (wantChangeRes) {
            _settings.ShadowCascadeResolution = 1 << (resIdx + 9);
            _shadowMap->SetResolution(_settings.ShadowCascadeResolution, _settings.ShadowCascadeCount);
        }

        bool inval = false;
        inval |= ImGui::DragFloat("Base Range", &_shadowMap->BaseCascadeRange, 0.01, 1, 32, "%.1f");

        ImGui::DragFloat("Sample Radius", &_shadowMap->SampleRadius, 0.01, 0, 10, "%.1f");
        ImGui::SliderInt("Sample Count", (int*)&_shadowMap->SampleCount, 1, 16);
        ImGui::Checkbox("Disable Cache", &_settings.DisableShadowCache);
        ImGui::Checkbox("Debug", &_settings.DebugShadows);

        if (inval) _shadowMap->InvalidateAllPages();

        if (_settings.DebugShadows) {
            if (ImGui::Begin("Debug: Shadow Map")) DrawShadowDebug();
            ImGui::End();
        }
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
        ImGui::DragFloat("Skybox Intensity", &_settings.SkyboxIntensity, 0.01, 0.001f, 100.0f);

        ImGui::Checkbox("Multi-Bounce", &_probes->MultiBounce);
        ImGui::Checkbox("Force Update All", &_probes->ForceUpdateAll);
    }

    {
        ZoneScopedN("ClearFb");
        _colorFb->Clear(0);
        _depthFb->Clear(0);

        _rast->SetViewport(uint2(_colorFb->Width, _colorFb->Height));
    }

    ShadingContext shader = {
        .DepthBuffer = _depthFb.get(),
        .ColorBuffer = _colorFb.get(),
        .Materials = _scene->Materials.data(),
        .Lights = _scene->Lights.data(),
        .NumLights = _scene->Lights.size(),

        .ViewPos = _cam.ViewPosition,
        .ProjMat = camProjMat,
        .ViewMat = camViewMat,
        .WorldToClipMat = camProjViewMat,

        .SkyboxTex = _skyboxTex.get(),
        .SkyboxIntensity = _settings.SkyboxIntensity,
        .AmbientIntensity = _settings.SunIntensity * _settings.AmbientFactor,
        .ShadowMap = _settings.EnableShadows ? _shadowMap.get() : nullptr,
        .ProbeField = _settings.EnableDDGI ? _probes.get() : nullptr,
        .BrickMap = _brickmap.get(),
        .Exposure = s_ExposureData.Exposure,
        .AutoExp = autoExposure ? &s_ExposureData : nullptr,
    };
    DrawCallBuilder drawBuilder = {
        .Culler = {
            .WorldToClipMat = shader.WorldToClipMat,
            .HizProjMat = shader.ProjMat,
            .HizViewMat = prevCamViewMat,
            .FrameSize = uint2(_colorFb->Width, _colorFb->Height),
            .HizBuffer = _settings.OcclusionCulling ? _hierDepth.get() : nullptr,
        },
        .GenerateInstanceData = true,
    };
    DrawContext templateDraw = {
        .DepthBuffer = _depthFb.get(),
        .ColorBuffer = _colorFb.get(),
        .Materials = _scene->Materials.data(),
        .ObjectToClipMat = shader.WorldToClipMat,
    };

    for (auto& model : _scene->Models) {
        drawBuilder.AddModel(*model, templateDraw);
    }

    bool debugOverdraw = _settings.DebugView == DebugLayer::OverdrawPixel || _settings.DebugView == DebugLayer::OverdrawQuad;
    auto& dispatchTable = debugOverdraw ? DrawContext::OverdrawShader : DrawContext::VisBufferShader;
    _rast->MultiDrawMeshlets(dispatchTable, drawBuilder.DrawCalls.size(), drawBuilder.DrawCalls.data(), sizeof(DrawContext));

    shader.Instances = drawBuilder.Instances.data();

    if (_settings.OcclusionCulling || _settings.EnableShadows) {
        ZoneScopedN("BuildHiZ");

        SWR_PERF_BEGIN(Culling);
        swr::texutil::DownsampleDepth(*_depthFb, *_hierDepth);
        SWR_PERF_END(Culling);
    }
    if (_settings.EnableShadows && sunLight != nullptr) {
        SWR_PERF_DEF_COUNTER(ShadowElapsed);
        SWR_PERF_BEGIN(ShadowElapsed);

        static bool pinShadowPos = false;
        static float3 pinnedShadowPos;

        if (ImGui::IsKeyPressed(ImGuiKey_O)) {
            pinShadowPos ^= true;
            pinnedShadowPos = _cam.ViewPosition;
        }

        _shadowMap->SetPosition(pinShadowPos ? pinnedShadowPos : float3(_cam.ViewPosition), sunLight->Direction);
        if (_settings.DisableShadowCache) _shadowMap->InvalidateAllPages();

        _shadowMap->MarkVisiblePages(*_rast, *_hierDepth, uint2(_colorFb->Width, _colorFb->Height), camProjViewMat);
        _shadowMap->UpdateHierPageBitmap();

        RenderShadowMap();

        SWR_PERF_END(ShadowElapsed);
    }

    SWR_PERF_DEF_COUNTER(Resolve);
    SWR_PERF_BEGIN(Resolve);

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
    ImGui::Text("Frame: %.1fms (%.0f FPS), Cull: %.2fms, Resolve: %.2fms",  //
                frameElapsedAvg / 1e6,                                      //
                1e9 / frameElapsedAvg,                                      //
                swr::perf::GetAverage("Culling") / 1e6,                     //
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

void RenderShadowMap() {
    ZoneScopedN("Shadows");

    auto depthBuffer = _shadowMap->DepthBuffer.get();
    _rast->SetViewport(uint2(depthBuffer->Width, depthBuffer->Height));

    for (uint32_t level = 0; level < depthBuffer->NumLayers; level++) {
        const uint32_t* bitmap = _shadowMap->DirtyPageMap[level];
        uint32_t numDirtyPages = 0;

        for (uint32_t i = 0; i < ShadowClipmap::MaxCascadeBitmapSize; i++) {
            for (uint32_t j : simd::BitIter(bitmap[i])) {
                uint2 offset = uint2(i * 32 + j);
                offset.x = offset.x % ShadowClipmap::MaxPageGridSize;
                offset.y = offset.y / ShadowClipmap::MaxPageGridSize;
                depthBuffer->ClearRect(offset * ShadowClipmap::PageSize, uint2(ShadowClipmap::PageSize), 0, level);
            }
            numDirtyPages += simd::popcnt(bitmap[i]);
        }
        if (numDirtyPages == 0) continue;

        float4x4 worldToCascadeMat = _shadowMap->GetWorldToCascadeMatrix(level);

        DrawCallBuilder drawBuilder = {
            .Culler = {
                .WorldToClipMat = worldToCascadeMat,
                .HizViewMat = worldToCascadeMat,
                .ShadowMap = _shadowMap.get(),
                .ShadowCascadeLevel = level,
            },
            .GenerateInstanceData = false,
        };
        DrawContext templateDraw = {
            .DepthBuffer = depthBuffer,
            .Materials = _scene->Materials.data(),  // needed for alpha tests
            .WrapOffset = int16_t2(_shadowMap->PageWrapOffset >> int(level) & ~int(ShadowClipmap::PageSize - 1)),
            .WrapMask = int16_t(_shadowMap->PageWrapMask),
            .CascadeIdx = uint16_t(level),
            .DepthBiasScale = 1.0f + (_shadowMap->SampleRadius / float(1 << level)),
            .ObjectToClipMat = worldToCascadeMat,
        };

        for (auto& model : _scene->Models) {
            drawBuilder.AddModel(*model, templateDraw);
        }

        _rast->EnableBinStencil = true;
        memset(_rast->BinStencilMask, 0, sizeof(_rast->BinStencilMask));

        constexpr uint32_t binSize = swr::Rasterizer::BinSize;
        uint2 binGridSize = (uint2(depthBuffer->Width, depthBuffer->Height) + binSize - 1u) / binSize;

        for (uint32_t y = 0; y < binGridSize.y; y++) {
            for (uint32_t x = 0; x < binGridSize.x; x++) {
                uint32_t pageIdx = _shadowMap->GetPhysicalPageIndex(uint2(x, y) * binSize, (int)level);

                if (bitmap[pageIdx / 32u] >> (pageIdx & 31u) & 1) {
                    uint32_t binIdx = x + y * binGridSize.x;
                    _rast->BinStencilMask[binIdx / 64u] |= 1ull << (binIdx & 63);
                }
            }
        }
        _rast->MultiDrawMeshlets(DrawContext::ShadowGenShader, drawBuilder.DrawCalls.size(), drawBuilder.DrawCalls.data(),
                                 sizeof(DrawContext));
    }
    _rast->EnableBinStencil = false;
}

void DrawShadowDebug() {
    auto depthBuffer = _shadowMap->DepthBuffer.get();

    static int cascadeIdx = 0;
    ImGui::SliderInt("Cascade", &cascadeIdx, 0, (int)depthBuffer->NumLayers - 1);

    static GLuint debugTexId = 0;
    uint32_t debugTexSize = 512;

    if (debugTexId == 0) {
        debugTexId = CreateTexture(debugTexSize, debugTexSize, 1, GL_R8);
        GLint swizzle[] = { GL_RED, GL_RED, GL_RED, GL_ONE };
        glTextureParameteriv(debugTexId, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
    }
    GLuint pbo;
    glCreateBuffers(1, &pbo);
    glNamedBufferStorage(pbo, debugTexSize * debugTexSize, nullptr, GL_MAP_WRITE_BIT);
    uint8_t* pixels = (uint8_t*)glMapNamedBuffer(pbo, GL_WRITE_ONLY);

    static float2 depthRange = float2(0.51, 0.52);
    v_float2 srcRange = v_float2(FLT_MAX, -FLT_MAX);

    for (uint32_t y = 0; y < debugTexSize; y++) {
        for (uint32_t x = 0; x < debugTexSize; x += simd::vec_width) {
            v_int2 pos = v_int2((int)x + simd::lane_idx<v_int>, (int)y);
            pos = simd::conv<int>(simd::conv<float>(pos) / debugTexSize * float2(depthBuffer->Width, depthBuffer->Height));
            v_float depth = simd::as<v_float>(depthBuffer->Fetch(pos, cascadeIdx));
            v_float color = simd::approx_sqrt((depth - depthRange.x) / (depthRange.y - depthRange.x));

            simd::store(&pixels[x + y * debugTexSize], simd::conv_sat<uint8_t>(color * 255));
            simd::cmov(srcRange.x, simd::min(srcRange.x, depth), depth != 0);
            simd::cmov(srcRange.y, simd::max(srcRange.y, depth), depth != 0);
        }
    }
    float2 newRange = float2(__builtin_reduce_min(srcRange.x), __builtin_reduce_max(srcRange.y));
    if (newRange.x != FLT_MAX) depthRange = glm::mix(depthRange, newRange, 0.1f);

    glUnmapNamedBuffer(pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTextureSubImage2D(debugTexId, 0, 0, 0, (GLsizei)debugTexSize, (GLsizei)debugTexSize, GL_RED, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo);

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();
    ImGui::Dummy(size);

    uint32_t pagesPerAxis = _shadowMap->DepthBuffer->Width / ShadowClipmap::PageSize;
    float cellSize = size.x / pagesPerAxis;

    for (uint32_t y = 0; y < pagesPerAxis; y++) {
        for (uint32_t x = 0; x < pagesPerAxis; x++) {
            uint32_t pageIdx = _shadowMap->GetPhysicalPageIndex(uint2(x, y) * ShadowClipmap::PageSize, cascadeIdx);
            uint32_t pageX = pageIdx % ShadowClipmap::MaxPageGridSize;
            uint32_t pageY = pageIdx / ShadowClipmap::MaxPageGridSize;

            bool visible = _shadowMap->VisiblePageMap[cascadeIdx][pageIdx / 32u] >> (pageIdx & 31) & 1;
            bool invalid = _shadowMap->InvalidPageMap[cascadeIdx][pageIdx / 32u] >> (pageIdx & 31) & 1;

            ImVec2 minPos = pos + ImVec2(round(x * cellSize), round(y * cellSize));
            ImVec2 maxPos = minPos + ImVec2(ceil(cellSize), ceil(cellSize));

            ImVec2 uv0 = ImVec2(pageX, pageY) / pagesPerAxis;
            ImVec2 uv1 = uv0 + ImVec2(1.0f / pagesPerAxis, 1.0f / pagesPerAxis);
            dl->AddImage(ImTextureRef(debugTexId), minPos, maxPos, uv0, uv1);

            if (!visible || invalid) dl->AddRectFilled(minPos, maxPos, invalid ? IM_COL32(255, 0, 255, 64) : IM_COL32(255, 0, 0, 96));
        }
    }
    ImU32 gridLineColor = ImGui::ColorConvertFloat4ToU32(ImGui::GetStyleColorVec4(ImGuiCol_Separator));
    for (uint32_t i = 0; i <= pagesPerAxis; i++) {
        float offset = round(i * cellSize);
        float w = i % 4 == 0 ? 2.0f : 1.0f;
        dl->AddLineH(pos.x, pos.x + size.x, pos.y + offset, gridLineColor, w);
        dl->AddLineV(pos.x + offset, pos.y, pos.y + size.x, gridLineColor, w);
    }
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
