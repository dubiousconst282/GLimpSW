#include <iostream>
#include <chrono>
#include <vector>
#include "SwRast.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "Camera.h"

#include "Model.h"
#include "QuickGL.h"

struct PhongShader {
    static const uint32_t NumCustomAttribs = 9;

    glm::mat4 ProjMat;
    glm::vec3 LightPos;
    const swr::Texture2D* DiffuseTex;

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
        swr::VFloat4 diffuseColor = DiffuseTex->SampleNearest(u, v);

        // Gram-schmidt process (produces higher-quality normal mapping on large meshes)
        // Re-orthogonalize T with respect to N

        swr::VFloat3 norm = swr::simd::normalize({ vars.GetSmooth(2), vars.GetSmooth(3), vars.GetSmooth(4) });
        swr::VFloat3 lightDir = swr::simd::normalize({
            LightPos.x - vars.GetSmooth(6),
            LightPos.y - vars.GetSmooth(7),
            LightPos.z - vars.GetSmooth(8),
        });
        swr::VFloat NdotL = swr::simd::max(swr::simd::dot(norm, lightDir), 0.0f);

        swr::VFloat4 color = {
            diffuseColor.x * NdotL,
            diffuseColor.y * NdotL,
            diffuseColor.z * NdotL,
            diffuseColor.w,
        };
        // auto color = swr::VFloat4{vars.W1, vars.W2, 1.0f - vars.W1 - vars.W2, 1.0f};

        auto rgba = swr::simd::PackRGBA(color);
        uint16_t alphaMask = _mm512_cmpgt_epu32_mask(rgba, _mm512_set1_epi32(128 << 24));
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
        _fb = std::make_shared<swr::Framebuffer>(1920, 1080);
        _rast = std::make_unique<swr::Rasterizer>(_fb);

        _model = std::make_unique<Model>("Models/Sponza/Sponza.gltf");
        //_model = std::make_unique<Model>("Models/San_Miguel/san-miguel-low-poly.obj");
        _cam = Camera();
        _cam.MoveSpeed = 20.0f;

        _glTex = std::make_unique<ogl::Texture2D>(_fb->Width, _fb->Height, 1, GL_RGBA8);
        _tempPixels = std::make_unique<uint32_t[]>(_fb->Width * _fb->Height);
    }

    void Render() {
        _cam.Update();

        PhongShader shader = {
            .ProjMat = _cam.GetViewProjMatrix(),
            .LightPos = _lightPos
        };

        _fb->Clear(0xD0EEFF, 1.0f);
        
        for (Mesh& mesh : _model->Meshes) {
            shader.DiffuseTex = mesh.Material->DiffuseTex;

            swr::VertexReader data(
                (uint8_t*)&_model->VertexBuffer[mesh.VertexOffset], 
                (uint8_t*)&_model->IndexBuffer[mesh.IndexOffset],
                mesh.IndexCount, swr::VertexReader::U32);

            _rast->DrawIndexed(data, shader);
        }

        if (ImGui::IsKeyPressed(ImGuiKey_M)) {
            _lightPos = _cam.Position;
        }

        // Vertex vertices[16];
        // uint16_t indices[16];
        // vertices[0] = { .x = -1.0f, .y = -1.0f, .z = -2.0f, .nx = 127 };
        // vertices[1] = { .x = +1.0f, .y = -1.0f, .z = -2.0f, .ny = 127 };
        // vertices[2] = { .x = +1.0f, .y = +1.0f, .z = -2.0f, .nz = 127 };

        // indices[0] = 0;
        // indices[1] = 1;
        // indices[2] = 2;
        // auto rd = swr::VertexReader((uint8_t*)vertices, (uint8_t*)indices, 3, swr::VertexReader::U16);
        // _rast->DrawIndexed(rd, shader);

        _fb->GetPixels(_tempPixels.get(), _fb->Width);
        _glTex->SetPixels(_tempPixels.get(), _fb->Width);

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto texId = (ImTextureID)(uintptr_t)_glTex->Handle;
        drawList->AddImage(texId, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::Begin("Rasterizer Stats");

        swr::ProfilerStats& stats = swr::g_Stats;
        double totalTime = stats.TotalElapsed[0] / 1000000.0;
        ImGui::Text("Frame: %.2fms (%.0f FPS) @ %dx%d", totalTime, 1000.0 / totalTime, _fb->Width, _fb->Height);
        ImGui::Text("Vertex setup: %.2fms", stats.VertexSetup[0] / 1000000.0);
        ImGui::Text("Rasterize: %.2fms", stats.Rasterize[0] / 1000000.0);
        ImGui::Text("Triangles: %.1fK (%.1fK clipped)", stats.TrianglesDrawn / 1000.0, stats.TrianglesClipped / 1000.0);
        ImGui::Text("Bins filled: %.1fK", stats.BinsFilled / 1000.0);
        ImGui::Checkbox("Wireframe", &_rast->EnableWireframe);
        stats.Reset();
        ImGui::End();

        ImGui::ShowMetricsWindow();
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