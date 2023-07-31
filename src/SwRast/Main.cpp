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

#include "Model.h"
#include "QuickGL.h"

struct PhongShader {
    static const uint32_t NumCustomAttribs = 9;

    glm::mat4 ProjMat;
    glm::vec3 LightPos;
    const swr::Texture2D* DiffuseTex;
    const swr::Texture2D* NormalTex;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        auto pos = swr::VFloat4{ .w = 1.0f };
        data.ReadAttribs(&Vertex::x, &pos.x, 3);
        vars.Position = swr::simd::TransformVector(ProjMat, pos);

        data.ReadAttribs(&Vertex::u, &vars.Attribs[0], 2);
        data.ReadAttribs(&Vertex::nx, &vars.Attribs[2], 3);
        data.ReadAttribs(&Vertex::tx, &vars.Attribs[5], 3);

        vars.Attribs[6] = pos.x;
        vars.Attribs[7] = pos.y;
        vars.Attribs[8] = pos.z;
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        vars.ApplyPerspectiveCorrection();

        swr::VFloat u = vars.GetSmooth(0);
        swr::VFloat v = vars.GetSmooth(1);
        swr::VFloat4 diffuseColor = DiffuseTex->SampleHybrid(u, v);

        swr::VFloat3 N = swr::simd::normalize({ vars.GetSmooth(2), vars.GetSmooth(3), vars.GetSmooth(4) });

        if (NormalTex != nullptr) {
            // T = normalize(T - dot(T, N) * N);
            // mat3 TBN = mat3(T, cross(N, T), N);
            // vec3 n = texture(u_NormalTex, UV).rgb * 2.0 - 1.0;
            // norm = normalize(TBN * n);
            swr::VFloat3 T = swr::simd::normalize({ vars.GetSmooth(5), vars.GetSmooth(6), vars.GetSmooth(7) });
            
            // Gram-schmidt process (produces higher-quality normal mapping on large meshes)
            // Re-orthogonalize T with respect to N
            swr::VFloat TN = swr::simd::dot(T, N);
            T = swr::simd::normalize({ T.x - TN * N.x, T.y - TN * N.y, T.z - TN * N.z });
            swr::VFloat3 B = swr::simd::cross(N, T);

            swr::VFloat4 SN = NormalTex->SampleHybrid(u, v);

            // https://aras-p.info/texts/CompactNormalStorage.html#q-comparison
            swr::VFloat Sx = SN.x * 2.0f - 1.0f;
            swr::VFloat Sy = SN.y * 2.0f - 1.0f;
            swr::VFloat Sz = swr::simd::sqrt14(1.0f - Sx * Sx + Sy * Sy);  // sqrt(1.0f - dot(n.xy, n.xy));

            N.x = T.x * Sx + B.x * Sy + N.x * Sz;
            N.y = T.y * Sx + B.y * Sy + N.y * Sz;
            N.z = T.z * Sx + B.z * Sy + N.z * Sz;
        }

        swr::VFloat3 lightDir = swr::simd::normalize({
            LightPos.x - vars.GetSmooth(6),
            LightPos.y - vars.GetSmooth(7),
            LightPos.z - vars.GetSmooth(8),
        });
        swr::VFloat NdotL = swr::simd::max(swr::simd::dot(N, lightDir), 0.0f);

        swr::VFloat4 color = {
            diffuseColor.x * NdotL,
            diffuseColor.y * NdotL,
            diffuseColor.z * NdotL,
            diffuseColor.w,
        };

        auto rgba = swr::simd::PackRGBA(color);
        uint16_t alphaMask = _mm512_cmpgt_epu32_mask(rgba, _mm512_set1_epi32(200 << 24));
        fb.WriteTile(vars.TileOffset, vars.TileMask & alphaMask, rgba, vars.Depth);
    }
};

class SwRenderer {
    std::shared_ptr<swr::Framebuffer> _fb;
    std::unique_ptr<swr::Rasterizer> _rast;
    std::unique_ptr<Model> _model;
    Camera _cam;
    std::unique_ptr<ogl::Texture2D> _glTex;
    std::unique_ptr<uint32_t[]> _tempPixels;
    glm::vec3 _lightPos;

public:
    SwRenderer() {
        _model = std::make_unique<Model>("Models/Sponza/Sponza.gltf");
        //_model = std::make_unique<Model>("Models/sea_keep_lonely_watcher/scene.gltf");
        //_model = std::make_unique<Model>("Models/San_Miguel/san-miguel-low-poly.obj");
        _cam = Camera();
        _cam.MoveSpeed = 10.0f;

        InitRasterizer(1920, 1080);
    }

    void InitRasterizer(uint32_t width, uint32_t height) {
        _fb = std::make_shared<swr::Framebuffer>(width, height);
        _rast = std::make_unique<swr::Rasterizer>(_fb);

        _glTex = std::make_unique<ogl::Texture2D>(_fb->Width, _fb->Height, 1, GL_RGBA8);
        _tempPixels = std::make_unique<uint32_t[]>(_fb->Width * _fb->Height);
    }

    void Render() {
        auto renderStart = std::chrono::high_resolution_clock::now();

        static bool s_NormalMapping = true;

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
        ImGui::Checkbox("Wireframe", &_rast->EnableWireframe);

        static int s_MaxDrawProgress = _model->Meshes.size();
        ImGui::SliderInt("Draw Progress", &s_MaxDrawProgress, 1, _model->Meshes.size());
        ImGui::SliderFloat("CamSpeed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::End();

        _cam.Update();

        PhongShader shader = {
            .ProjMat = _cam.GetViewProjMatrix(),
            .LightPos = _lightPos
        };

        _fb->Clear(0xD0EEFF, 1.0f);

        for (Mesh& mesh : _model->Meshes) {
            shader.DiffuseTex = mesh.Material->DiffuseTex;
            shader.NormalTex = s_NormalMapping ? mesh.Material->NormalTex : nullptr;

            swr::VertexReader data(
                (uint8_t*)&_model->VertexBuffer[mesh.VertexOffset], 
                (uint8_t*)&_model->IndexBuffer[mesh.IndexOffset],
                mesh.IndexCount, swr::VertexReader::U16);

            _rast->DrawIndexed(data, shader);
        }

        if (ImGui::IsKeyPressed(ImGuiKey_M)) {
            _lightPos = _cam.Position;
        }

        _fb->GetPixels(_tempPixels.get(), _fb->Width);
        _glTex->SetPixels(_tempPixels.get(), _fb->Width);

        double totalElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;
        swr::ProfilerStats& stats = swr::g_Stats;

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto texId = (ImTextureID)(uintptr_t)_glTex->Handle;
        drawList->AddImage(texId, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::Begin("Rasterizer Stats");
        ImGui::Text("Frame: %.2fms (%.0f FPS)", totalElapsed, 1000.0 / totalElapsed);
        ImGui::Text("Vertex setup: %.2fms", stats.VertexSetup[0] / 1000000.0);
        ImGui::Text("Rasterize: %.2fms", stats.Rasterize[0] / 1000000.0);
        ImGui::Text("Triangles: %.1fK (%.1fK clipped)", stats.TrianglesDrawn / 1000.0, stats.TrianglesClipped / 1000.0);
        ImGui::Text("Bins filled: %.1fK", stats.BinsFilled / 1000.0);
        stats.Reset();
        ImGui::End();

        DrawTranslationGizmo(_lightPos);
    }

    void DrawTranslationGizmo(glm::vec3& pos) {
        ImGuizmo::BeginFrame();

        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

        glm::mat4 viewMat = _cam.GetViewMatrix();
        glm::mat4 projMat = _cam.GetProjectionMatrix();
        glm::mat4 lightMat = glm::translate(glm::identity<glm::mat4>(), pos);

        if (ImGuizmo::Manipulate(&viewMat[0].x, &projMat[0].x, ImGuizmo::TRANSLATE, ImGuizmo::WORLD, &lightMat[0].x)) {
            pos = lightMat[3];
        }
    }
};



int main(int argc, char** args) {
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "SwRast", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    io.Fonts->AddFontFromFileTTF("Roboto-Medium.ttf", 18.0f);

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