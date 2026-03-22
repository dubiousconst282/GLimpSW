#include <vector>
#include <format>
#include <filesystem>

#include "SwRast.h"
#include "Texture.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>

#include "Camera.h"

#include "Scene.h"
#include "RendererShaders.h"
#include "ProfilerStats.h"

#include "tracy/Tracy.hpp"

class SwRenderer {
    std::unique_ptr<swr::Framebuffer> _fb;
    std::unique_ptr<swr::Framebuffer> _prevFb;
    std::unique_ptr<swr::Rasterizer> _rast;
    std::unique_ptr<scene::Model> _scene, _shadowScene;
    std::unique_ptr<renderer::DefaultShader> _shader;

    Camera _cam;
    scene::DepthPyramid _depthPyramid;

    GLuint _frontTex = 0;
    GLuint _shadowDebugTex = 0;

    glm::vec3 _lightPos;
    glm::mat4 _shadowProjMat;

    std::unique_ptr<swr::Framebuffer> _shadowFb;
    std::unique_ptr<swr::Rasterizer> _shadowRast;

    std::vector<std::filesystem::path> _scenePaths;
    std::vector<std::filesystem::path> _skyboxPaths;
    std::string _currSceneName, _currSkyboxName;

    renderer::SSAO _ssao;

public:
    SwRenderer() {
        std::vector<std::string_view> modelExts = { ".gltf", ".glb", ".fbx", ".obj" };
        SearchFiles("assets/", _scenePaths, modelExts);
        SearchFiles("logs/assets/", _scenePaths, modelExts);  // git ignored

        SearchFiles("assets/skyboxes/", _skyboxPaths, { ".hdr", ".jpg", ".png" });
        SearchFiles("logs/assets/", _skyboxPaths, { ".hdr" });

        _shader = std::make_unique<renderer::DefaultShader>();

        LoadScene("assets/models/Sponza/Sponza.gltf");
        LoadSkybox("assets/skyboxes/sunflowers_puresky_4k.hdr");

        _cam = Camera{ .Position = glm::vec3(-10.41, 1.35f, 0.4), .Euler = glm::vec2(1.0f, -0.05f), .MoveSpeed = 5.0f };

        InitRasterizer(1920, 1080);

        _lightPos = glm::vec3(0.589494, 0.684509, 0.428906) * 18.0f;
    }
    ~SwRenderer() {
        if (_frontTex != 0) glDeleteTextures(1, &_frontTex);
    }

    void InitRasterizer(uint32_t width, uint32_t height) {
        _fb = std::make_unique<swr::Framebuffer>(width, height, renderer::DefaultShader::NumFbAttachments);
        _rast = std::make_unique<swr::Rasterizer>(*_fb);

        if (_frontTex != 0) glDeleteTextures(1, &_frontTex);
        _frontTex = CreateTexture(_fb->Width, _fb->Height, 1, GL_RGBA8);
    }
    void LoadScene(const std::filesystem::path& path) {
        _scene = std::make_unique<scene::Model>(path.string());

        if (path.filename().compare("Sponza.gltf") == 0) {
            auto shadowModelPath = path;
            _shadowScene = std::make_unique<scene::Model>(shadowModelPath.replace_filename("Sponza_LowPoly.gltf").string());
        } else {
            _shadowScene = nullptr;
        }
        _currSceneName = path.filename().string();
    }
    void LoadSkybox(const std::filesystem::path& path) {
        auto tex = swr::texutil::LoadCubemapFromPanoramaHDR(path.string().c_str());
        _shader->SetSkybox(std::move(tex));

        _currSkyboxName = path.filename().string();
    }

    void Render() {
        FrameMark;

        static bool s_EnableShadows = false;
        static float s_ShadowRange = 15.0f;
        static int s_ShadowRes = 1024;
        static bool s_ShadowFollowCam = false;

        static bool s_EnableSSAO = false;
        static bool s_HzbOcclusion = false;
        static bool s_AnimateLight = false;
        static bool s_VSync = false;

        static renderer::DebugLayer s_Layer = renderer::DebugLayer::BaseColor;

        ImGui::Begin("Settings");
        
        ImGui::SeparatorText("Environment");

        if (ImGui::BeginCombo("Scene", _currSceneName.c_str())) {
            for (auto& path : _scenePaths) {
                if (ImGui::Selectable(path.filename().string().c_str())) {
                    LoadScene(path);
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::BeginCombo("Skybox", _currSkyboxName.c_str())) {
            for (auto& path : _skyboxPaths) {
                if (ImGui::Selectable(path.filename().string().c_str())) {
                    LoadSkybox(path);
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SeparatorText("Rendering");

        std::string currRes = std::format("{}x{}", _fb->Width, _fb->Height);
        if (ImGui::BeginCombo("Resolution", currRes.c_str())) {
            if (ImGui::Selectable("1920x1080")) InitRasterizer(1920, 1080);
            if (ImGui::Selectable("1280x720")) InitRasterizer(1280, 720);
            if (ImGui::Selectable("848x480")) InitRasterizer(848, 480);
            if (ImGui::Selectable("320x240")) InitRasterizer(320, 240);
            ImGui::EndCombo();
        }

        ImGui::Combo("Debug Channel", (int*)&s_Layer, "None\0BaseColor\0Normals\0Metallic-Roughness\0Ambient Occlusion\0Emissive Mask\0Overdraw\0");
        ImGui::SliderFloat("Exposure", &_shader->Exposure, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("IBL Intensity", &_shader->IntensityIBL, 0.0f, 1.0f, "%.2f");
        ImGui::Checkbox("Hier-Z Occlusion", &s_HzbOcclusion);
        if (ImGui::Checkbox("VSync", &s_VSync)) {
            glfwSwapInterval(s_VSync ? 1 : 0);
        }

        ImGui::SeparatorText("Effects");

        ImGui::Checkbox("Shadow Mapping", &s_EnableShadows);
        if (s_EnableShadows) {
            ImGui::Indent();
            ImGui::InputFloat("Range##Shadow", &s_ShadowRange, 0.5f);
            if (ImGui::SliderInt("Resolution##Shadow", &s_ShadowRes, 128, 2048)) {
                s_ShadowRes = (s_ShadowRes + 64) & ~127;
            }
            ImGui::Checkbox("Follow Camera", &s_ShadowFollowCam);
            ImGui::Checkbox("Animate Sun", &s_AnimateLight);
            ImGui::Unindent();

            if (s_AnimateLight) {
                static float time;
                _lightPos = { 7.5, 16.5, sinf(time * 0.3f) * 10 };
                time += ImGui::GetIO().DeltaTime;
            }
        }

        ImGui::Checkbox("SSAO", &s_EnableSSAO);
        if (s_EnableSSAO) {
            ImGui::Indent();
            ImGui::InputFloat("Radius", &_ssao.Radius, 0.1f);
            ImGui::InputFloat("Range##SSAO", &_ssao.MaxRange, 0.1f);
            ImGui::Unindent();
        }
        ImGui::Checkbox("Temporal AA", &_shader->EnableTAA);
        ImGui::Checkbox("Blur Skybox", &_shader->BlurSkybox);

        ImGui::SeparatorText("Misc");
        ImGui::SliderFloat("Cam Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

        ImGui::End();

        if (ImGui::IsKeyPressed(ImGuiKey_C)) {
            _cam.Mode = _cam.Mode == Camera::InputMode::FirstPerson ? Camera::InputMode::Arcball : Camera::InputMode::FirstPerson;
        }
        _cam.Update(Camera::GetInputsFromImGui());

        STAT_TIME_BEGIN(Frame);

        if (s_EnableShadows) {
            STAT_TIME_BEGIN(Shadow);
            RenderShadow(s_ShadowRes, s_ShadowRange, s_ShadowFollowCam);
            STAT_TIME_END(Shadow);
        }

        glm::mat4 projMat = _cam.GetProjMatrix();
        glm::mat4 viewMat = _cam.GetViewMatrix(true);

        _shader->UpdateJitter(projMat, *_fb);  // add jitter to `projMat`

        glm::mat4 projViewMat = projMat * viewMat;

        _shader->ShadowBuffer = s_EnableShadows ? _shadowFb.get() : nullptr;
        _shader->ShadowProjMat = _shadowProjMat;
        _shader->ViewMat = viewMat;
        _shader->LightPos = _lightPos;
        _shader->ViewPos = _cam.ViewPosition;
        _shader->Materials = _scene->Materials.data();

        {
            ZoneScopedN("ClearFb");

            if (s_Layer == renderer::DebugLayer::Overdraw) {
                _fb->Clear(0xFF000000, 0.0f);
            } else {
                _fb->ClearDepth(0.0f);
            }
        }

        uint32_t drawCalls = 0;

        _scene->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
            ZoneScopedN("Draw Node");
            _shader->ProjMat = projViewMat * modelMat;
            _shader->ModelMat = modelMat;
            _shader->Meshlets = &_scene->Meshlets[node.MeshletOffset];

            if (s_Layer == renderer::DebugLayer::Overdraw) {
                // _rast->DrawMeshlets(renderer::OverdrawShader{ .ProjMat = _shader->ProjMat });
            } else {
                _rast->DrawMeshlets(node.MeshletCount, *_shader);
            }
            drawCalls++;
            return true;
        });

        STAT_TIME_BEGIN(Compose);

        if (s_HzbOcclusion) {
            _depthPyramid.Update(*_fb, projViewMat);
            //RenderDebugHzb();
        }
        if (s_EnableSSAO) {
            _ssao.Generate(*_fb, _depthPyramid, projViewMat);
        }

        _shader->ProjMat = projMat * _cam.GetViewMatrix(false);

        if (s_Layer == renderer::DebugLayer::None) {
            _shader->Compose(*_fb, s_EnableSSAO, *_prevFb);
        } else if (s_Layer != renderer::DebugLayer::Overdraw) {
            _shader->ComposeDebug(*_fb, s_Layer);
        }

        STAT_TIME_END(Compose);

        {
            ZoneScopedN("UploadFrameToGPU");
            
            GLuint pbo;
            glCreateBuffers(1, &pbo);
            glNamedBufferStorage(pbo, _fb->Width * _fb->Height * 4, nullptr, GL_MAP_WRITE_BIT);

            uint32_t* pixels = (uint32_t*)glMapNamedBuffer(pbo, GL_WRITE_ONLY); // spec guarantees align >= 64
            _fb->GetPixels(pixels, _fb->Width);
            glUnmapNamedBuffer(pbo);

            // This copy could be avoided by reading buffer data directly from shader, but this is good enough.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTextureSubImage2D(_frontTex, 0, 0, 0, (GLsizei)_fb->Width, (GLsizei)_fb->Height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glDeleteBuffers(1, &pbo);
        }

        STAT_TIME_END(Frame);

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        drawList->AddImage((ImTextureID)_frontTex, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 0), ImVec2(1, 1));

        // clang-format off
        ImGui::Begin("Rasterizer Stats");
        ImGui::Text("Frame: %.1fms (%.0f FPS), Shadow: %.1fms, Post: %.2fms", STAT_GET_TIME(Frame), 1000.0 / STAT_GET_TIME(Frame), STAT_GET_TIME(Shadow), STAT_GET_TIME(Compose));
        ImGui::Text("Setup: %.1fms (%.1fK vertices), Rasterize: %.2fms", STAT_GET_TIME(Setup), STAT_GET_COUNT(VerticesShaded), STAT_GET_TIME(Rasterize));
        ImGui::Text("Triangles: %.1fK (%.1fK clipped, %.1fK bins, %d calls)", STAT_GET_COUNT(TrianglesDrawn), STAT_GET_COUNT(TrianglesClipped), STAT_GET_COUNT(BinsFilled), drawCalls);
        ImGui::End();
        // clang-format on

        swr::g_Stats.Reset();
        // DrawTranslationGizmo(_lightPos);
    }

    void RenderShadow(int size, float range, bool followCam) {
        if (_shadowFb == nullptr || _shadowFb->Width != size) {
            _shadowFb = std::make_unique<swr::Framebuffer>(size, size);
            _shadowRast = std::make_unique<swr::Rasterizer>(*_shadowFb);
        }

        glm::vec3 centerPos = followCam ? glm::vec3(_cam.Position.x, 0, _cam.Position.z) : glm::vec3(0.0f);

        _shadowProjMat = glm::ortho(-range, +range, -range, +range, 0.05f, 40.0f) * 
                         glm::lookAt(centerPos + _lightPos, centerPos, glm::vec3(0, 1, 0));

        _shadowFb->ClearDepth(1.0f);

        auto activeScene = _shadowScene ? _shadowScene.get() : _scene.get();
        // activeScene->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
        //     for (uint32_t meshId : node.Meshes) {
        //         scene::Mesh& mesh = activeScene->Meshes[meshId];

        //         _shadowRast->DrawMeshlets(100, renderer::DepthOnlyShader{ .ProjMat = _shadowProjMat * modelMat });
        //     }
        //     return true;
        // });

        ImGui::SetNextWindowCollapsed(true, ImGuiCond_Appearing);
        if (ImGui::Begin("Shadow Debug")) {
            if (_shadowDebugTex == 0) _shadowDebugTex = CreateTexture(_shadowFb->Width, _shadowFb->Height, 1, GL_RGBA8);
            auto buf = std::make_unique<uint32_t[]>(_shadowFb->Width * _shadowFb->Height);

            for (uint32_t y = 0; y < _shadowFb->Height; y++) {
                for (uint32_t x = 0; x < _shadowFb->Width; x++) {
                    float d = _shadowFb->DepthBuffer[_shadowFb->GetPixelOffset(x, y)];
                    uint32_t i = x + y * _shadowFb->Width;

                    uint8_t c = (uint8_t)(glm::sqrt(1.0f - d * d) * 255.0f);
                    buf[i] = c * 0x01'01'01 | 0xFF'000000;
                }
            }
            glTextureSubImage2D(_shadowDebugTex, 0, 0, 0, (GLsizei)_shadowFb->Width, (GLsizei)_shadowFb->Height, GL_RGBA, GL_UNSIGNED_BYTE, buf.get());

            ImGui::Image((ImTextureID)_shadowDebugTex, ImGui::GetContentRegionAvail(), ImVec2(0, 1), ImVec2(1, 0));
        }
        ImGui::End();
    }

    void RenderDebugHzb() {
        if (ImGui::Begin("Depth Pyramid")) {
            static int level = 0;
            ImGui::SliderInt("Level", &level, 0, 12);

            uint32_t w = _fb->Width >> level;
            uint32_t h = _fb->Height >> level;
            if (_shadowDebugTex == 0) _shadowDebugTex = CreateTexture(w, h, 1, GL_RGBA8);
            auto buf = std::make_unique<uint32_t[]>(w * h);

            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    float d = _depthPyramid.GetDepth(x / (float)(w - 1), y / (float)(h - 1), level);

                    uint8_t c = (uint8_t)(glm::sqrt(1.0f - d * d) * 255.0f);
                    buf[x + y * w] = c * 0x01'01'01 | 0xFF'000000;
                }
            }
            glTextureSubImage2D(_shadowDebugTex, 0, 0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, buf.get());

            float s = 1.0f / (1 << level);
            ImGui::Image((ImTextureID)_shadowDebugTex, ImGui::GetContentRegionAvail(), ImVec2(0, s), ImVec2(s, 0));
        }
        ImGui::End();
    }

    static GLuint CreateTexture(uint32_t width, uint32_t height, uint32_t mipLevels, GLuint fmt) {
        GLuint handle;
        glCreateTextures(GL_TEXTURE_2D, 1, &handle);

        glTextureStorage2D(handle, (GLsizei)mipLevels, fmt, (GLsizei)width, (GLsizei)height);

        glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glTextureParameteri(handle, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(handle, GL_TEXTURE_WRAP_T, GL_REPEAT);
        return handle;
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

    static void SearchFiles(std::string_view basePath, std::vector<std::filesystem::path>& dest, const std::vector<std::string_view>& extensions) {
        for (auto& entry : std::filesystem::recursive_directory_iterator(basePath)) {
            if (!entry.is_regular_file()) continue;

            for (auto& ext : extensions) {
                if (entry.path().extension().compare(ext) == 0) {
                    dest.push_back(entry.path());
                    break;
                }
            }
        }
    }
};

int main(int argc, char** args) {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Glimpsw", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "logs/imgui.ini";
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    io.Fonts->AddFontFromFileTTF("assets/Roboto-Medium.ttf", 18.0f);

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();

    SwRenderer renderer;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
            static int winRect[4];
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();

            if (!glfwGetWindowMonitor(window)) {
                glfwGetWindowPos(window, &winRect[0], &winRect[1]);
                glfwGetWindowSize(window, &winRect[2], &winRect[3]);

                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, 0);
            } else {
                glfwSetWindowMonitor(window, nullptr, winRect[0], winRect[1], winRect[2], winRect[3], 0);
            }
            glfwSwapInterval(1);
        }

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        renderer.Render();

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
    glfwTerminate();
    return 0;
}
