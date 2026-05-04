#include <vector>

#include "Rasterizer.h"
#include "Texture.h"

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

#include "Camera.h"

#include "Scene.h"
#include "Shading.h"

namespace {

GLFWwindow* _window;

swr::FramebufferPtr _fb;
std::unique_ptr<swr::Rasterizer> _rast;
GLuint _fbTexture = 0;

Camera _cam;
std::unique_ptr<Scene> _scene;
swr::HdrTexture2D::Ptr _skyboxTex;

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
    _fb = swr::CreateFramebuffer(width, height, 3);

    if (_fbTexture != 0) glDeleteTextures(1, &_fbTexture);
    _fbTexture = CreateTexture(_fb->Width, _fb->Height, 1, GL_RGBA8);
}
void LoadScene(const std::string& path) {
    _scene = std::make_unique<Scene>();
    _scene->ImportGltf(path);
}
void LoadSkybox(const std::string& path) {
    _skyboxTex = swr::texutil::LoadOctahedronFromPanoramaHDR(path.c_str()); //
}

void InitRenderer() {
    //LoadScene("assets/models/Sponza/Sponza.gltf");
    LoadScene("../GraphicsAssets/Sponza_Lights.glb");
    LoadSkybox("assets/skyboxes/sunflowers_puresky_4k.hdr");

    _cam = Camera{ .Position = {9.6753334456680022, 1.1288966350423235, -0.48801679478492588}, .Euler = {-1.58308733, -0.608505249}, .MoveSpeed = 5.0f };
    
    _rast = std::make_unique<swr::Rasterizer>(4);
    _rast->EnableBinning = false;
    CreateFramebuffer(1920, 1080);
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

void RenderFrame() {
    FrameMark;

    static bool s_VSync = false;

    static DebugLayer s_Layer = DebugLayer::None;
    static bool s_EnableVisBuffer = true;
    static bool s_ShowPerfHeatmap = false;
    static float s_Exposure = 1.0f;

    ImGui::Begin("Settings");
    
    ImGui::SeparatorText("Environment");

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
    snprintf(currRes, sizeof(currRes), "%dx%d", _fb->Width, _fb->Height);
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

    ImGui::SeparatorText("Renderer");

    ImGui::Checkbox("Vis-buffer", &s_EnableVisBuffer);
    ImGui::Checkbox("Perf Heatmap", &s_ShowPerfHeatmap);

    ImGui::Combo("Debug Channel", (int*)&s_Layer, "None\0BaseColor\0Normals\0MetallicRoughness\0MeshletId\0TriangleId\0Overdraw\0");
    ImGui::SliderFloat("Exposure", &s_Exposure, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);

    ImGui::SeparatorText("Misc");
    ImGui::SliderFloat("Cam Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

    ImGui::End();

    if (ImGui::IsKeyPressed(ImGuiKey_C)) {
        _cam.Mode = _cam.Mode == Camera::InputMode::FirstPerson ? Camera::InputMode::Arcball : Camera::InputMode::FirstPerson;
    }
    _cam.Update(Camera::GetInputsFromImGui());

    SWR_PERF_BEGIN(Frame);

    glm::mat4 projMat = _cam.GetProjMatrix();
    glm::mat4 viewMat = _cam.GetViewMatrix(true);
    glm::mat4 projViewMat = projMat * viewMat;

    ShadingContext shader = {
        .Meshlets = _scene->Meshlets.data(),
        .Materials = _scene->Materials.data(),
        .Lights = _scene->Lights.data(),
        .NumLights = _scene->Lights.size(),
        .SkyboxTex = _skyboxTex.get(),
        .ViewPos = _cam.ViewPosition,
        .Exposure = s_Exposure,
        .ShowPerfHeatmap = s_ShowPerfHeatmap,
    };

    if (_scene->Lights.Storage.size() == 0) {
        static Light light = {
            .Type = Light::kTypeDirectional,
            .Direction = glm::normalize(-float3(0.589494, 0.684509, 0.428906)),
            .Color = float3(1, 1, 1),
            .Intensity = 1500,
        };
        shader.Lights = &light;
        shader.NumLights = 1;
    }

    {
        ZoneScopedN("ClearFb");
        _fb->Clear(0xFF000000, 0.0f);
    }

    uint32_t drawCalls = 0;

    swr::ShaderDispatchTable dispatchTable;
    if (s_EnableVisBuffer) {
        dispatchTable = ShadingContext::GetVisBufferShader();
    } else if (s_Layer == DebugLayer::Overdraw) {
        dispatchTable = ShadingContext::GetOverdrawShader();
    } else {
        dispatchTable = ShadingContext::GetDeferredShader();
    }

    for (auto& model : _scene->Models) {
        for (auto& node : model->Nodes) {
            if (node.MeshCount == 0) continue;

            ZoneScopedN("Draw Node");
            shader.UpdateProj(projViewMat, node.GlobalTransform);
            shader.MeshletOffset = node.MeshOffset;

            _rast->DrawMeshlets(*_fb, node.MeshCount, { dispatchTable, &shader });
            drawCalls++;
        }
    }
    SWR_PERF_BEGIN(Resolve);

    if (s_EnableVisBuffer && s_Layer != DebugLayer::Overdraw) {
        if (s_Layer == DebugLayer::None) {
            shader.Resolve(*_rast, *_fb);
        } else {
            shader.ResolveDebug(*_rast, *_fb, s_Layer);
        }
    }

    SWR_PERF_END(Resolve);

    {
        ZoneScopedN("UploadFrameToGPU");
        
        GLuint pbo;
        glCreateBuffers(1, &pbo);
        glNamedBufferStorage(pbo, _fb->Width * _fb->Height * 4, nullptr, GL_MAP_WRITE_BIT);

        uint32_t* pixels = (uint32_t*)glMapNamedBuffer(pbo, GL_WRITE_ONLY); // spec guarantees align >= 64
        _fb->GetPixels(0, pixels, _fb->Width);
        glUnmapNamedBuffer(pbo);

        // This copy could be avoided by reading buffer data directly from shader, but this is good enough.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTextureSubImage2D(_fbTexture, 0, 0, 0, (GLsizei)_fb->Width, (GLsizei)_fb->Height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glDeleteBuffers(1, &pbo);
    }

    SWR_PERF_END(Frame);

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    drawList->AddImage((ImTextureID)_fbTexture, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 0), ImVec2(1, 1));

    swr::perf::FlushThreadCounters();

    ImGui::Begin("Rasterizer Stats");
    ImGui::Text("Frame: %.1fms (%.0f FPS), Resolve: %.2fms",   //
                swr::perf::GetAverage(swr::PerfCounter::FrameTime) / 1e6,   //
                1e9 / swr::perf::GetAverage(swr::PerfCounter::FrameTime),   //
                // swr::perf::GetAverage(swr::PerfCounter::ShadowTime) / 1e6,  //
                swr::perf::GetAverage(swr::PerfCounter::ResolveTime) / 1e6);

    ImGui::Text("Raster: %.2fms (%.1fK/%.1fK tris, %.1fK clip, %.1fK bins)",         //
                swr::perf::GetAverage(swr::PerfCounter::DrawTime) / 1e6,             //
                swr::perf::GetCurrent(swr::PerfCounter::TrianglesRasterized) / 1e3,  //
                swr::perf::GetCurrent(swr::PerfCounter::TrianglesProcessed) / 1e3,   //
                swr::perf::GetCurrent(swr::PerfCounter::TrianglesClipped) / 1e3,     //
                swr::perf::GetCurrent(swr::PerfCounter::BinQueueFlushes) / 1e3);
    ImGui::End();

    swr::perf::Reset();

    // DrawTranslationGizmo(_lightPos);
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
