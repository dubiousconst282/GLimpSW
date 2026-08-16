#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
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

namespace {

GLFWwindow* _window;

swr::FramebufferPtr2D<swr::pixfmt::RGBA8u> _colorFb;
swr::FramebufferPtr2D<swr::pixfmt::R32f> _depthFb;
swr::FramebufferPtr2D<swr::pixfmt::R32f> _shadowDepthFb;

std::unique_ptr<swr::Rasterizer> _rast;
swr::TexturePtr2D<swr::pixfmt::R32f> _prevDepthMap;
GLuint _fbTexture = 0;

Camera _cam;
std::unique_ptr<Scene> _scene;
swr::HdrTexture2D::Ptr _skyboxTex;
std::unique_ptr<BrickMap> _brickmap;
std::unique_ptr<GIProbeField> _probes;

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
void LoadScene(const std::string& path) {
    _scene = std::make_unique<Scene>();
    _scene->ImportGltf(path);

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
    _skyboxTex = swr::texutil::LoadOctahedronFromPanoramaHDR(path.c_str()); //
}

void InitRenderer() {
    LoadScene("assets/models/Sponza/Sponza.gltf");
    LoadSkybox("assets/skyboxes/sunflowers_puresky_4k.hdr");
    ShadingContext::LoadBlueNoise();

    _cam = Camera{ .Position = { -9.38414, 1.63823, -0.00313 }, .Euler = { 0.7993, -0.0849 }, .MoveSpeed = 5.0f };

    _rast = std::make_unique<swr::Rasterizer>(4);
    CreateFramebuffer(1920, 1080);

    _shadowDepthFb = swr::CreateFramebuffer<swr::pixfmt::R32f>(1024, 1024, 1, 1);
    _probes = std::make_unique<GIProbeField>();
    _probes->ProbeBaseOffset = -float3(GIProbeField::GridSize) * _probes->ProbeSpacing / 2.0f;

    // Adjustments for Sponza
    _probes->ProbeBaseOffset.y = -0.5f;
    _probes->ProbeBaseOffset.z -= 0.25f;

    _probes->Reset(true);
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

            SWR_PERF_BEGIN(Culling);
            uint32_t visibleCount =
                shader.CullMeshlets(cullBitmap, &_scene->Meshlets[node.MeshOffset + i], meshCount, node.GlobalTransform,
                                    float2(_colorFb->Width, _colorFb->Height), enableOcclusionCull ? _prevDepthMap.get() : nullptr);
            SWR_PERF_END(Culling);

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

void DrawTranslationGizmo(glm::vec3& pos) {
    ImGuizmo::BeginFrame();

    ImGuiIO& io = ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

    glm::mat4 projMat = _cam.GetProjMatrix();
    glm::mat4 viewMat = _cam.GetViewMatrix(true);
    float matrix[16];
    float temp[3]{ 1.0f, 1.0f, 1.0f };
    ImGuizmo::RecomposeMatrixFromComponents(&pos.x, temp, temp, matrix);

    if (ImGuizmo::Manipulate(&viewMat[0].x, &projMat[0].x, ImGuizmo::TRANSLATE, ImGuizmo::WORLD, matrix)) {
        ImGuizmo::DecomposeMatrixToComponents(matrix, &pos.x, temp, temp);
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
            LoadScene(std::string(outPath));
            NFD_FreePathU8(outPath);
        }
    }

    ImGui::SeparatorText("Rasterizer");

    char currRes[64];
    snprintf(currRes, sizeof(currRes), "%dx%d", _colorFb->Width, _colorFb->Height);
    if (ImGui::BeginCombo("Resolution", currRes)) {
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

    ImGui::SeparatorText("Renderer");

    static bool s_OcclusionCulling = true;
    static bool s_EnableDDGI = true;
    static bool s_DrawProbes = true;
    ImGui::Checkbox("Occlusion Culling", &s_OcclusionCulling);
    ImGui::Checkbox("DDGI", &s_EnableDDGI);
    ImGui::Checkbox("Draw Probes", &s_DrawProbes);

    static DebugLayer s_DebugView = DebugLayer::None;
    {
        const int count = std::size(kDebugLayerNames);
        ImGui::Combo("Debug View", (int*)&s_DebugView, kDebugLayerNames, count);

        int idx = (int)s_DebugView;
        if (ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_1)) s_DebugView = (DebugLayer)((idx == 0 ? count : idx) - 1);
        if (ImGui::IsKeyChordPressed(ImGuiMod_Alt | ImGuiKey_2)) s_DebugView = (DebugLayer)((idx + 1 < count ? idx + 1 : 0));
    }
    static int debugShadowLayerIdx = -1;
    ImGui::SliderInt("Debug Shadows", &debugShadowLayerIdx, -1, MaxShadowCascades);

    static AutoExposureData s_ExposureData = {};
    static float s_ExposureValue = -15.1f;
    bool autoExposure = s_ExposureValue < -15.0f;

    if (ImGui::SliderFloat("Exposure", &s_ExposureValue, -15.1f, 15.0f, autoExposure ? "Auto" : "%+.1f")) {
        s_ExposureData.Exposure = 1.0f / exp2(-s_ExposureValue) * 1.2f; // EV100 -> Lmax
    }

    ImGui::DragFloatRange2("LogRange", &s_ExposureData.HistRange.x, &s_ExposureData.HistRange.y, 0.1f, -12.0f, 12.0f, "%.3f");

    ImGui::SeparatorText("Misc");
    ImGui::SliderFloat("Cam Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

    ImGui::InputScalarN("Pos", ImGuiDataType_Double, &_cam.Position, 3);
    if (ImGui::IsItemActive() && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_C)) {
        char text[128];
        snprintf(text, sizeof(text), ".Position = { %.5f, %.5f, %.5f }, .Euler = { %.4f, %.4f }",  //
                 _cam.Position.x, _cam.Position.y, _cam.Position.z, _cam.Euler.x, _cam.Euler.y);
        ImGui::SetClipboardText(text);
    }
    ImGui::End();

    if (ImGui::IsKeyPressed(ImGuiKey_C)) {
        _cam.Mode = _cam.Mode == Camera::InputMode::FirstPerson ? Camera::InputMode::Arcball : Camera::InputMode::FirstPerson;
    }

    float4x4 prevCamViewMat = _cam.GetViewMatrix(true);

    _cam.Update(Camera::GetInputsFromImGui());

    float4x4 camProjMat = _cam.GetProjMatrix();
    float4x4 camViewMat = _cam.GetViewMatrix(true);
    float4x4 camProjViewMat = camProjMat * camViewMat;

    SWR_PERF_BEGIN(Frame);

    ShadingContext shader = {
        .ColorBuffer = _colorFb.get(),
        .Materials = _scene->Materials.data(),
        .Lights = _scene->Lights.data(),
        .NumLights = _scene->Lights.size(),

        .ViewPos = _cam.ViewPosition,
        .PrevViewMat = prevCamViewMat,

        .SkyboxTex = _skyboxTex.get(),
        .ProbeField = s_EnableDDGI ? _probes.get() : nullptr,
        .BrickMap = _brickmap.get(),
        .Exposure = s_ExposureData.Exposure,
        .AutoExp = autoExposure ? &s_ExposureData : nullptr,
    };

    Light* sunLight = _scene->Lights[0].Type == Light::kTypeDirectional ? &_scene->Lights[0] : nullptr;

    if (sunLight != nullptr) {
        ZoneScopedN("ShadowMap");
        SWR_PERF_BEGIN(ShadowRender);

        float3 dir = sunLight->Direction;
        float2 angle = float2(atan2f(dir.z, dir.x), asinf(dir.y));
        if (ImGui::DragFloat2("Sun angle", &angle.x, 0.01f)) {
            float r = cosf(angle.y);
            sunLight->Direction = float3(cosf(angle.x) * r, sinf(angle.y), sinf(angle.x) * r);
        }

        static float sampleRadius = 3;
        ImGui::DragFloat("Sample Radius", &sampleRadius, 0.01, 1, 10, "%.1f");

        static int cascadeIdx = 2;
        ImGui::SliderInt("Active Cascade", &cascadeIdx, 0, 3);

        ComputeShadowMatrices(_cam, sunLight->Direction, _shadowDepthFb->Width, shader.ShadowProjMatrices);
        shader.ShadowDepthMap = _shadowDepthFb.get();
        shader.ShadowSampleRadius = sampleRadius;
        shader.ShadowProjMatrices[0] = shader.ShadowProjMatrices[cascadeIdx];  // TODO: sample cascades

        _shadowDepthFb->Clear(0);
        shader.DepthBuffer = _shadowDepthFb.get();
        _rast->SetViewport(uint2(_shadowDepthFb->Width, _shadowDepthFb->Height));

        shader.ProjMat = float4x4(1);
        shader.ViewMat = float4x4(1);
        shader.WorldToClipMat = shader.ShadowProjMatrices[0];

        for (auto& model : _scene->Models) {
            DrawModel(*model, shader, ShadingContext::ShadowGenShader, nullptr, false);
        }
        SWR_PERF_END(ShadowRender);
    }

    bool debugOverdraw = s_DebugView == DebugLayer::OverdrawPixel || s_DebugView == DebugLayer::OverdrawQuad;
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

    auto instances = std::vector<InstanceData>();
    instances.resize(1); // reserve 0th instance id for "null"
    instances.reserve(1024);

    for (auto& model : _scene->Models) {
        DrawModel(*model, shader, dispatchTable, &instances, s_OcclusionCulling);
    }

    SWR_PERF_BEGIN(Resolve);

    shader.DepthBuffer = _depthFb.get();
    shader.Instances = instances.data();

    if (s_DebugView == DebugLayer::None) {
        ZoneScopedN("Resolve");

        shader.Resolve(*_rast);

        if (autoExposure) {
            s_ExposureData.UpdateExposure(ImGui::GetIO().DeltaTime, 1.1f);
        }
    } else if (s_DebugView != DebugLayer::None) {
        shader.ResolveDebug(*_rast, s_DebugView);
    }
    SWR_PERF_END(Resolve);

    bool reinitProbes = false;
    reinitProbes |= ImGui::DragFloat3("Probe Offset", &_probes->ProbeBaseOffset.x, 0.1f);
    reinitProbes |= ImGui::DragFloat("Probe Spacing", &_probes->ProbeSpacing, 0.1f, 0.1f, 5.0f);
    if (reinitProbes) _probes->Reset(false);

    static bool forceUpdateAll = false;
    ImGui::SliderInt("Rays/Probe", (int*)&_probes->RaysPerProbe, 16, 128);
    ImGui::SliderInt("Max Hysteresis", (int*)&_probes->MaxHysteresis, 1, 128);
    ImGui::DragFloat("Ray Max Dist", &_probes->MaxRayDist, 0.1f, 1.0f, 100.0f);
    ImGui::DragFloat("Normal Bias", &_probes->NormalBias, 0.001, 0.01f, 0.5f);

    static float skyboxIntensity = 1;
    ImGui::DragFloat("Skybox Intensity", &skyboxIntensity, 0.01, 0.001f, 100.0f);
    shader.SkyboxIntensity = skyboxIntensity;

    ImGui::Checkbox("Multi-Bounce", &_probes->MultiBounce);
    ImGui::Checkbox("Force Update All", &forceUpdateAll);

    if (forceUpdateAll) memset(_probes->CurrActiveBitmap, 255, sizeof(_probes->CurrActiveBitmap));

    if (s_EnableDDGI) {
        ZoneScopedN("ProbeUpdate");

        SWR_PERF_BEGIN(ProbeUpdate);
        shader.UpdateProbes(*_rast);
        SWR_PERF_END(ProbeUpdate);

        if (s_DrawProbes) {
            shader.DrawProbesDebug();
        }
    }

    if (debugShadowLayerIdx >= 0) {
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

    if (s_OcclusionCulling) {
        ZoneScopedN("BuildHiZ");
        SWR_PERF_BEGIN(Culling);
        swr::texutil::DownsampleDepth(*_depthFb, *_prevDepthMap);
        SWR_PERF_END(Culling);
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

    SWR_PERF_END(Frame);

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddImage((ImTextureID)_fbTexture, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 0), ImVec2(1, 1));

    swr::perf::FlushThreadCounters();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Frame: %.1fms (%.0f FPS), Cull: %.2fms, Resolve: %.2fms",   //
                swr::perf::GetAverage(swr::PerfCounter::FrameTime) / 1e6,    //
                1e9 / swr::perf::GetAverage(swr::PerfCounter::FrameTime),    //
                swr::perf::GetAverage(swr::PerfCounter::CullingTime) / 1e6,  //
                swr::perf::GetAverage(swr::PerfCounter::ResolveTime) / 1e6);

    ImGui::Text("Raster: %.2fms (%.1fK/%.1fK tris, %.1fK clip, %.1fK bins)",         //
                swr::perf::GetAverage(swr::PerfCounter::DrawTime) / 1e6,             //
                swr::perf::GetCurrent(swr::PerfCounter::TrianglesRasterized) / 1e3,  //
                swr::perf::GetCurrent(swr::PerfCounter::TrianglesProcessed) / 1e3,   //
                swr::perf::GetCurrent(swr::PerfCounter::TrianglesClipped) / 1e3,     //
                swr::perf::GetCurrent(swr::PerfCounter::BinsFlushed) / 1e3);

    ImGui::Text("Shadows: %.2fms, Probes: %.2fms",                                //
                swr::perf::GetAverage(swr::PerfCounter::ShadowRenderTime) / 1e6,  //
                swr::perf::GetAverage(swr::PerfCounter::ProbeUpdateTime) / 1e6);
    ImGui::End();

    swr::perf::Reset();
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
    io.IniFilename = "logs/imgui.ini";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    io.Fonts->AddFontFromFileTTF("assets/Roboto-Medium.ttf", 16.0f);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init();

    InitRenderer();

    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
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
    glDeleteTextures(1, &_fbTexture);

    NFD_Quit();
    glfwTerminate();
    return 0;
}
