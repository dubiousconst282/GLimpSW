#include <iostream>
#include <chrono>
#include <vector>
#include "SwRast.h"

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
    std::unique_ptr<scene::Model> _model, _shadowModel;
    std::unique_ptr<swr::HdrTexture2D> _skyboxTex;

    Camera _cam;
    scene::DepthPyramid _depthPyramid;

    std::unique_ptr<ogl::Texture2D> _frontTex;
    std::unique_ptr<ogl::Texture2D> _shadowDebugTex;
    std::unique_ptr<uint32_t[]> _tempPixels;

    glm::vec3 _lightPos, _lightRot;
    glm::mat4 _shadowProjMat;

    std::shared_ptr<swr::Framebuffer> _shadowFb;
    std::unique_ptr<swr::Rasterizer> _shadowRast;

public:
    SwRenderer() {
        _model = std::make_unique<scene::Model>("assets/models/Sponza/Sponza.gltf");
        _shadowModel = std::make_unique<scene::Model>("assets/models/Sponza/Sponza_LowPoly.gltf");

        _skyboxTex = std::make_unique<swr::HdrTexture2D>(swr::HdrTexture2D::LoadCubemapFromPanorama("assets/skyboxes/sunflowers_puresky_4k.hdr"));

        //_model = std::make_unique<scene::Model>("Models/sea_keep_lonely_watcher/scene.gltf");
        //_model = std::make_unique<scene::Model>("Models/SunTemple_v4/SunTemple.fbx");

        _cam = Camera{ .Position = glm::vec3(-7, 5.5f, 0), .Euler = glm::vec2(-0.88f, -0.32f), .MoveSpeed = 10.0f };

        InitRasterizer(1280, 720);

        _shadowFb = std::make_shared<swr::Framebuffer>(1024, 1024);
        _shadowRast = std::make_unique<swr::Rasterizer>(_shadowFb);

        _lightRot = glm::vec3(0.0f, -90.0f, 0.0f);
        _lightPos = glm::vec3(3.5f, 12.0f, 2.9f);
    }

    void InitRasterizer(uint32_t width, uint32_t height) {
        _fb = std::make_shared<swr::Framebuffer>(width, height);
        _rast = std::make_unique<swr::Rasterizer>(_fb);

        _frontTex = std::make_unique<ogl::Texture2D>(_fb->Width, _fb->Height, 1, GL_RGBA8);
        _tempPixels = std::make_unique<uint32_t[]>(_fb->Width * _fb->Height);
    }

    void Render() {
        static bool s_NormalMapping = true;
        static bool s_EnableShadows = false;
        static bool s_HzbOcclusion = true;

        ImGui::Begin("Settings");

        std::string currRes = std::format("{}x{}", _fb->Width, _fb->Height);
        if (ImGui::BeginCombo("Res", currRes.c_str())) {
            if (ImGui::Selectable("1920x1080")) InitRasterizer(1920, 1080);
            if (ImGui::Selectable("1280x720")) InitRasterizer(1280, 720);
            if (ImGui::Selectable("854x480")) InitRasterizer(854, 480);
            if (ImGui::Selectable("320x240")) InitRasterizer(320, 240);
            ImGui::EndCombo();
        }
        ImGui::Checkbox("Normal Mapping", &s_NormalMapping);
        ImGui::Checkbox("Shadow Mapping", &s_EnableShadows);
        ImGui::Checkbox("Hier-Z Occlusion", &s_HzbOcclusion);
        ImGui::SliderFloat("Cam Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::End();

        _cam.Update();

        auto renderStart = std::chrono::high_resolution_clock::now();

        if (s_EnableShadows) {
            RenderShadow();
        }

        double shadowElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;

        glm::mat4 projMat = _cam.GetViewProjMatrix();

        renderer::PhongShader shader = {
            .LightPos = _lightPos,
            .ShadowProjMat = _shadowProjMat,
            .ShadowBuffer = s_EnableShadows ? _shadowFb.get() : nullptr,
        };

        _fb->Clear(0xFF'FFEED0, 1.0f);

        uint32_t drawCalls = 0;

        _model->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
            for (uint32_t meshId : node.Meshes) {
                scene::Mesh& mesh = _model->Meshes[meshId];

                shader.ProjMat = projMat * modelMat;
                shader.ModelMat = modelMat;

                if (s_HzbOcclusion && !_depthPyramid.IsVisibleAABB(mesh.BoundMin, mesh.BoundMax)) continue;

                shader.DiffuseTex = mesh.Material->DiffuseTex;
                shader.NormalTex = s_NormalMapping ? mesh.Material->NormalTex : nullptr;

                swr::VertexReader data(
                    (uint8_t*)&_model->VertexBuffer[mesh.VertexOffset], 
                    (uint8_t*)&_model->IndexBuffer[mesh.IndexOffset],
                    mesh.IndexCount, swr::VertexReader::U16);

                _rast->Draw(data, shader);
                drawCalls++;
            }
            return true;
        });

        renderer::DrawSkybox(*_fb, *_skyboxTex, _cam.GetProjectionMatrix(), _cam.GetViewMatrix());

        if (ImGui::IsKeyPressed(ImGuiKey_M)) {
            _lightPos = _cam.Position;
        }

        if (s_HzbOcclusion) {
            _depthPyramid.Update(*_fb, shader.ProjMat);
            RenderDebugHzb();
        }

        _fb->GetPixels(_tempPixels.get(), _fb->Width);
        _frontTex->SetPixels(_tempPixels.get(), _fb->Width);

        double totalElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;
        swr::ProfilerStats& stats = swr::g_Stats;

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto texId = (ImTextureID)(uintptr_t)_frontTex->Handle;
        drawList->AddImage(texId, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::Begin("Rasterizer Stats");
        ImGui::Text("Frame: %.1fms (%.0f FPS), Shadow: %.1fms", totalElapsed, 1000.0 / totalElapsed, shadowElapsed);
        ImGui::Text("Setup: %.1fms (clip: %.1fms, bin: %.1fms)", stats.SetupTime[0] / 1000000.0, stats.ClippingTime[0] / 1000000.0, stats.BinningTime[0] / 1000000.0);
        ImGui::Text("Rasterize: %.1fms (CPU: %.0f%%)", stats.RasterizeTime[0] / 1000000.0, stats.RasterizeCpuTime / (double)(stats.RasterizeTime[0] * std::thread::hardware_concurrency()) * 100);
        ImGui::Text("Triangles: %.1fK (%.1fK clipped, %.1fK bins, %d calls)", stats.TrianglesDrawn / 1000.0, stats.TrianglesClipped / 1000.0, stats.BinsFilled / 1000.0, drawCalls);
        stats.Reset();
        ImGui::End();

        DrawTranslationGizmo(_lightPos, _lightRot);
    }

    void RenderShadow() {
        //TODO: fix shadow matrix
        _shadowProjMat = glm::ortho(-12.0f, +12.0f, -12.0f, +12.0f, 0.05f, 40.0f) * glm::lookAt(_lightPos, glm::vec3(0, -1, 0), glm::vec3(0, 1, 0));

        _shadowFb->ClearDepth(1.0f);

        _shadowModel->Traverse([&](const scene::Node& node, const glm::mat4& modelMat) {
            for (uint32_t meshId : node.Meshes) {
                scene::Mesh& mesh = _shadowModel->Meshes[meshId];

                swr::VertexReader data(
                    (uint8_t*)&_shadowModel->VertexBuffer[mesh.VertexOffset],
                    (uint8_t*)&_shadowModel->IndexBuffer[mesh.IndexOffset],
                    mesh.IndexCount, swr::VertexReader::U16);

                _shadowRast->Draw(data, renderer::DepthOnlyShader{ .ProjMat = _shadowProjMat * modelMat });
            }
            return true;
        });

        if (ImGui::Begin("Shadow Debug")) {
            uint32_t n = _shadowFb->Width * _shadowFb->Height;

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
        glm::mat4 projMat = _cam.GetProjectionMatrix();
        float matrix[16];
        float scale[3]{ 1.0f, 1.0f, 1.0f };
        ImGuizmo::RecomposeMatrixFromComponents(&pos.x, &rot.x, scale, matrix);

        if (ImGuizmo::Manipulate(&viewMat[0].x, &projMat[0].x, ImGuizmo::TRANSLATE, ImGuizmo::WORLD, matrix)) {
            ImGuizmo::DecomposeMatrixToComponents(matrix, &pos.x, &rot.x, scale);
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