#include <iostream>
#include <chrono>
#include <vector>
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
#include "QuickGL.h"
#include "RendererShaders.h"

class SwRenderer {
    std::shared_ptr<swr::Framebuffer> _fb;
    std::unique_ptr<swr::Rasterizer> _rast;
    std::unique_ptr<scene::Model> _scene, _shadowScene;
    std::unique_ptr<renderer::DefaultShader> _shader;

    Camera _cam;
    scene::DepthPyramid _depthPyramid;

    std::unique_ptr<ogl::Texture2D> _frontTex;
    std::unique_ptr<ogl::Texture2D> _shadowDebugTex;
    std::unique_ptr<uint32_t[]> _tempPixels;

    glm::vec3 _lightPos, _lightRot;
    glm::mat4 _shadowProjMat;

    std::shared_ptr<swr::Framebuffer> _shadowFb;
    std::unique_ptr<swr::Rasterizer> _shadowRast;

    std::vector<std::filesystem::path> _scenePaths;
    std::string _currSceneName;

    renderer::SSAO _ssao;

public:
    SwRenderer() {
        SearchScenes("assets/", _scenePaths);
        SearchScenes("logs/assets/", _scenePaths); // git ignored

        LoadScene("assets/models/Sponza/Sponza.gltf");

        auto skyboxTex = swr::texutil::LoadCubemapFromPanoramaHDR("assets/skyboxes/footprint_court.hdr");
        //auto skyboxTex = swr::texutil::LoadCubemapFromPanoramaHDR("assets/skyboxes/sunflowers_puresky_4k.hdr");
        _shader = std::make_unique<renderer::DefaultShader>(std::move(skyboxTex));

        _cam = Camera{ .Position = glm::vec3(-7, 5.5f, 0), .Euler = glm::vec2(-0.88f, -0.32f), .MoveSpeed = 5.0f };

        InitRasterizer(1280, 720);

        _shadowFb = std::make_shared<swr::Framebuffer>(1024, 1024);
        _shadowRast = std::make_unique<swr::Rasterizer>(_shadowFb);

        _lightRot = glm::vec3(0.0f, -90.0f, 0.0f);
        _lightPos = glm::vec3(3.5f, 12.0f, 2.9f);
    }

    void InitRasterizer(uint32_t width, uint32_t height) {
        _fb = std::make_shared<swr::Framebuffer>(width, height, 5); // 4 attachments for deferred normals/roughness + 1 for AO
        _rast = std::make_unique<swr::Rasterizer>(_fb);

        _frontTex = std::make_unique<ogl::Texture2D>(_fb->Width, _fb->Height, 1, GL_RGBA8);
        _tempPixels = std::make_unique<uint32_t[]>(_fb->Width * _fb->Height);
    }
    void LoadScene(const std::filesystem::path& path) {
        _scene = std::make_unique<scene::Model>(path.string());
        _shadowScene = nullptr;

        _currSceneName = path.filename().string();
    }

    void Render() {
        static bool s_NormalMapping = true;
        static bool s_EnableShadows = false;
        static bool s_EnableSSAO = false;
        static bool s_HzbOcclusion = true;

        ImGui::Begin("Settings");
        if (ImGui::BeginCombo("Scene", _currSceneName.c_str())) {
            for (auto& path : _scenePaths) {
                if (ImGui::Selectable(path.filename().string().c_str())) {
                    LoadScene(path);
                }
            }
            ImGui::EndCombo();
        }
        std::string currRes = std::format("{}x{}", _fb->Width, _fb->Height);
        if (ImGui::BeginCombo("Res", currRes.c_str())) {
            if (ImGui::Selectable("1920x1080")) InitRasterizer(1920, 1088);
            if (ImGui::Selectable("1280x720")) InitRasterizer(1280, 720);
            if (ImGui::Selectable("854x480")) InitRasterizer(854, 480);
            if (ImGui::Selectable("320x240")) InitRasterizer(320, 240);
            ImGui::EndCombo();
        }
        ImGui::Checkbox("Normal Mapping", &s_NormalMapping);
        ImGui::Checkbox("Shadow Mapping", &s_EnableShadows);
        ImGui::Checkbox("Hier-Z Occlusion", &s_HzbOcclusion);
        ImGui::Checkbox("SSAO", &s_EnableSSAO);

        if (s_EnableSSAO) {
            ImGui::SeparatorText("SSAO");
            ImGui::InputFloat("Radius", &_ssao.Radius, 0.1f);
            ImGui::InputFloat("Range", &_ssao.MaxRange, 0.1f);
        }

        ImGui::Separator();

        ImGui::SliderFloat("Cam Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::End();

        if (ImGui::IsKeyPressed(ImGuiKey_C)) {
            _cam.Mode = _cam.Mode == Camera::InputMode::FirstPerson ? Camera::InputMode::Arcball : Camera::InputMode::FirstPerson;
        }
        _cam.Update();

        auto renderStart = std::chrono::high_resolution_clock::now();

        if (s_EnableShadows && _shadowScene != nullptr) {
            RenderShadow();
        }

        double shadowElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;

        glm::mat4 projMat = _cam.GetProjMatrix();
        glm::mat4 viewMat = _cam.GetViewMatrix();
        glm::mat4 projViewMat = projMat * viewMat;

        _shader->ShadowBuffer = s_EnableShadows ? _shadowFb.get() : nullptr;
        _shader->ShadowProjMat = _shadowProjMat;
        _shader->ViewMat = viewMat;
        _shader->LightPos = _lightPos;
        _shader->ViewPos = _cam._ViewPosition;

        _fb->ClearDepth(1.0f);

        uint32_t drawCalls = 0;

        _scene->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
            for (uint32_t meshId : node.Meshes) {
                scene::Mesh& mesh = _scene->Meshes[meshId];

                if (s_HzbOcclusion && !_depthPyramid.IsVisible(mesh, modelMat)) continue;

                _shader->ProjMat = projViewMat * modelMat;
                _shader->ModelMat = modelMat;
                _shader->BaseColorTex = mesh.Material->DiffuseTex;
                _shader->NormalMetallicRoughnessTex = s_NormalMapping ? mesh.Material->NormalTex : nullptr;

                swr::VertexReader data(
                    (uint8_t*)&_scene->VertexBuffer[mesh.VertexOffset], 
                    (uint8_t*)&_scene->IndexBuffer[mesh.IndexOffset],
                    mesh.IndexCount, swr::VertexReader::U16);

                _rast->Draw(data, *_shader);
                drawCalls++;
            }
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

        _shader->ProjMat = projViewMat;
        _shader->Compose(*_fb, s_EnableSSAO);

        STAT_TIME_END(Compose);

        _fb->GetPixels(_tempPixels.get(), _fb->Width);
        _frontTex->SetPixels(_tempPixels.get(), _fb->Width);

        double totalElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;
        swr::ProfilerStats& stats = swr::g_Stats;

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto texId = (ImTextureID)(uintptr_t)_frontTex->Handle;
        drawList->AddImage(texId, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 1), ImVec2(1, 0));

        // clang-format off
        ImGui::Begin("Rasterizer Stats");
        ImGui::Text("Frame: %.1fms (%.0f FPS), Shadow: %.1fms, Post: %.1fms", totalElapsed, 1000.0 / totalElapsed, shadowElapsed, STAT_GET_TIME(Compose));
        ImGui::Text("Setup: %.1fms (%.1fK vertices)", STAT_GET_TIME(Setup), STAT_GET_COUNT(VerticesShaded));
        ImGui::Text("Rasterize: %.1fms", STAT_GET_TIME(Rasterize));
        ImGui::Text("Triangles: %.1fK (%.1fK clipped, %.1fK bins, %d calls)", STAT_GET_COUNT(TrianglesDrawn), STAT_GET_COUNT(TrianglesClipped), STAT_GET_COUNT(BinsFilled), drawCalls);
        ImGui::End();
        // clang-format on

        stats.Reset();
        DrawTranslationGizmo(_lightPos, _lightRot);
    }

    void RenderShadow() {
        //TODO: fix shadow matrix
        _shadowProjMat = glm::ortho(-12.0f, +12.0f, -12.0f, +12.0f, 0.05f, 40.0f) * glm::lookAt(_lightPos, glm::vec3(0, -1, 0), glm::vec3(0, 1, 0));

        _shadowFb->ClearDepth(1.0f);

        _shadowScene->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
            for (uint32_t meshId : node.Meshes) {
                scene::Mesh& mesh = _shadowScene->Meshes[meshId];

                swr::VertexReader data(
                    (uint8_t*)&_shadowScene->VertexBuffer[mesh.VertexOffset],
                    (uint8_t*)&_shadowScene->IndexBuffer[mesh.IndexOffset],
                    mesh.IndexCount, swr::VertexReader::U16);

                _shadowRast->Draw(data, renderer::DepthOnlyShader{ .ProjMat = _shadowProjMat * modelMat });
            }
            return true;
        });

        if (ImGui::Begin("Shadow Debug")) {
            _shadowDebugTex = std::make_unique<ogl::Texture2D>(_shadowFb->Width, _shadowFb->Height, 1, GL_RGBA8);
            auto buf = std::make_unique<uint32_t[]>(_shadowFb->Width * _shadowFb->Height);

            for (uint32_t y = 0; y < _shadowFb->Height; y++) {
                for (uint32_t x = 0; x < _shadowFb->Width; x++) {
                    float d = _shadowFb->DepthBuffer[_shadowFb->GetPixelOffset(x, y)];
                    uint32_t i = x + y * _shadowFb->Width;

                    uint8_t c = (uint8_t)(glm::sqrt(1.0f - d * d) * 255.0f);
                    buf[i] = c * 0x01'01'01 | 0xFF'000000;
                }
            }
            _shadowDebugTex->SetPixels(buf.get(), _shadowFb->Width);

            ImGui::Image((ImTextureID)(uintptr_t)_shadowDebugTex->Handle, ImGui::GetContentRegionAvail(), ImVec2(0, 1), ImVec2(1, 0));
        }
        ImGui::End();
    }

    void RenderDebugHzb() {
        if (ImGui::Begin("Depth Pyramid")) {
            static int level = 0;
            ImGui::SliderInt("Level", &level, 0, 12);

            uint32_t w = _fb->Width >> level;
            uint32_t h = _fb->Height >> level;
            _shadowDebugTex = std::make_unique<ogl::Texture2D>(w, h, 1, GL_RGBA8);
            auto buf = std::make_unique<uint32_t[]>(w * h);

            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    float d = _depthPyramid.GetDepth(x / (float)(w - 1), y / (float)(h - 1), level);

                    uint8_t c = (uint8_t)(glm::sqrt(1.0f - d * d) * 255.0f);
                    buf[x + y * w] = c * 0x01'01'01 | 0xFF'000000;
                }
            }
            _shadowDebugTex->SetPixels(buf.get(), w);

            ImGui::Image((ImTextureID)(uintptr_t)_shadowDebugTex->Handle, ImGui::GetContentRegionAvail(), ImVec2(0, 1), ImVec2(1, 0));
        }
        ImGui::End();
    }

    void DrawTranslationGizmo(glm::vec3& pos, glm::vec3& rot) {
        ImGuizmo::BeginFrame();

        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

        glm::mat4 viewMat = _cam.GetViewMatrix();
        glm::mat4 projMat = _cam.GetProjMatrix();
        float matrix[16];
        float scale[3]{ 1.0f, 1.0f, 1.0f };
        ImGuizmo::RecomposeMatrixFromComponents(&pos.x, &rot.x, scale, matrix);

        if (ImGuizmo::Manipulate(&viewMat[0].x, &projMat[0].x, ImGuizmo::TRANSLATE, ImGuizmo::WORLD, matrix)) {
            ImGuizmo::DecomposeMatrixToComponents(matrix, &pos.x, &rot.x, scale);
        }
    }

    static void SearchScenes(std::string_view basePath, std::vector<std::filesystem::path>& dest) {
        const std::string_view kExtensions[] = { ".gltf", ".glb", ".fbx", ".obj" };

        for (auto& entry : std::filesystem::recursive_directory_iterator(basePath)) {
            if (!entry.is_regular_file()) continue;

            bool matchesExt = false;

            for (uint32_t i = 0; i < std::size(kExtensions) && !matchesExt; i++) {
                matchesExt |= entry.path().extension().compare(kExtensions[i]) == 0;
            }
            if (!matchesExt) continue;

            dest.push_back(entry.path());
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