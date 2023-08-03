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

using swr::VFloat, swr::VFloat3, swr::VFloat4;

struct PhongShader {
    static const uint32_t NumCustomAttribs = 12;

    glm::mat4 ProjMat;
    glm::vec3 LightPos;
    glm::mat4 ShadowProjMat;
    const swr::Framebuffer* ShadowBuffer;
    const swr::Texture2D* DiffuseTex;
    const swr::Texture2D* NormalTex;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        VFloat4 pos = { .w = 1.0f };
        data.ReadAttribs(&Vertex::x, &pos.x, 3);
        vars.Position = swr::simd::TransformVector(ProjMat, pos);

        data.ReadAttribs(&Vertex::u, &vars.Attribs[0], 2);
        data.ReadAttribs(&Vertex::nx, &vars.Attribs[2], 3);
        data.ReadAttribs(&Vertex::tx, &vars.Attribs[5], 3);

        vars.Attribs[6] = pos.x;
        vars.Attribs[7] = pos.y;
        vars.Attribs[8] = pos.z;

        if (ShadowBuffer != nullptr) {
            VFloat4 shadowPos = swr::simd::TransformVector(ShadowProjMat, pos);
            VFloat shadowRcpW = _mm512_rcp14_ps(shadowPos.w);

            vars.Attribs[9] = shadowPos.x * shadowRcpW * 0.5f + 0.5f;
            vars.Attribs[10] = shadowPos.y * shadowRcpW * 0.5f + 0.5f;
            vars.Attribs[11] = shadowPos.z * shadowRcpW ;
        }
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        vars.ApplyPerspectiveCorrection();

        VFloat u = vars.GetSmooth(0);
        VFloat v = vars.GetSmooth(1);
        VFloat4 diffuseColor = DiffuseTex->SampleHybrid(u, v);

        VFloat3 N = swr::simd::normalize({ vars.GetSmooth(2), vars.GetSmooth(3), vars.GetSmooth(4) });

        if (NormalTex != nullptr) {
            // T = normalize(T - dot(T, N) * N);
            // mat3 TBN = mat3(T, cross(N, T), N);
            // vec3 n = texture(u_NormalTex, UV).rgb * 2.0 - 1.0;
            // norm = normalize(TBN * n);
            VFloat3 T = swr::simd::normalize({ vars.GetSmooth(5), vars.GetSmooth(6), vars.GetSmooth(7) });
            
            // Gram-schmidt process (produces higher-quality normal mapping on large meshes)
            // Re-orthogonalize T with respect to N
            VFloat TN = swr::simd::dot(T, N);
            T = swr::simd::normalize({ T.x - TN * N.x, T.y - TN * N.y, T.z - TN * N.z });
            VFloat3 B = swr::simd::cross(N, T);

            VFloat4 SN = NormalTex->SampleHybrid(u, v);

            // https://aras-p.info/texts/CompactNormalStorage.html#q-comparison
            VFloat Sx = SN.x * 2.0f - 1.0f;
            VFloat Sy = SN.y * 2.0f - 1.0f;
            VFloat Sz = swr::simd::sqrt14(1.0f - Sx * Sx + Sy * Sy);  // sqrt(1.0f - dot(n.xy, n.xy));

            N.x = T.x * Sx + B.x * Sy + N.x * Sz;
            N.y = T.y * Sx + B.y * Sy + N.y * Sz;
            N.z = T.z * Sx + B.z * Sy + N.z * Sz;
        }

        VFloat3 lightDir = swr::simd::normalize({
            LightPos.x - vars.GetSmooth(6),
            LightPos.y - vars.GetSmooth(7),
            LightPos.z - vars.GetSmooth(8),
        });
        VFloat NdotL = swr::simd::dot(N, lightDir);
        VFloat diffuseLight = swr::simd::max(NdotL, 0.0f);
        VFloat shadowLight = 1.0f;

        if (ShadowBuffer != nullptr) [[unlikely]] {
            VFloat sx = vars.GetSmooth(9);
            VFloat sy = vars.GetSmooth(10);
            VFloat bias = swr::simd::max(0.08f * (1.0f - NdotL), 0.001f);
            VFloat currentDepth = vars.GetSmooth(11) - bias;
            VFloat closestDepth = ShadowBuffer->SampleDepth(sx, sy);

            // closestDepth > pos.z ? 1.0 : 0.0
            shadowLight = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(closestDepth, currentDepth, _CMP_GE_OQ), shadowLight);
        }
        VFloat combinedLight = diffuseLight * shadowLight + 0.3;

        VFloat4 color = {
            diffuseColor.x * combinedLight,
            diffuseColor.y * combinedLight,
            diffuseColor.z * combinedLight,
            diffuseColor.w,
        };

        auto rgba = swr::simd::PackRGBA(color);
        uint16_t alphaMask = _mm512_cmpgt_epu32_mask(rgba, _mm512_set1_epi32(200 << 24));
        fb.WriteTile(vars.TileOffset, vars.TileMask & alphaMask, rgba, vars.Depth);
    }
};
struct DepthOnlyShader {
    static const uint32_t NumCustomAttribs = 0;

    glm::mat4 ProjMat;

    void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
        VFloat4 pos = { .w = 1.0f };
        data.ReadAttribs(&Vertex::x, &pos.x, 3);
        vars.Position = swr::simd::TransformVector(ProjMat, pos);
    }

    void ShadePixels(const swr::Framebuffer& fb, swr::VaryingBuffer& vars) const {
        _mm512_mask_storeu_ps(&fb.DepthBuffer[vars.TileOffset], vars.TileMask, vars.Depth);
    }
};

class SwRenderer {
    std::shared_ptr<swr::Framebuffer> _fb;
    std::unique_ptr<swr::Rasterizer> _rast;
    std::unique_ptr<Model> _model, _shadowModel;
    Camera _cam;
    std::unique_ptr<ogl::Texture2D> _frontTex;
    std::unique_ptr<ogl::Texture2D> _shadowDebugTex;
    std::unique_ptr<uint32_t[]> _tempPixels;
    glm::vec3 _lightPos, _lightDir;
    glm::mat4 _shadowProjMat;

    std::shared_ptr<swr::Framebuffer> _shadowFb;
    std::unique_ptr<swr::Rasterizer> _shadowRast;

public:
    SwRenderer() {
        _model = std::make_unique<Model>("Models/Sponza/Sponza.gltf");
        _shadowModel = std::make_unique<Model>("Models/Sponza/Sponza_LowPoly.gltf");

        //_model = std::make_unique<Model>("Models/sea_keep_lonely_watcher/scene.gltf");
        //_model = std::make_unique<Model>("Models/San_Miguel/san-miguel-low-poly.obj");
        _cam = Camera{ .Position = glm::vec3(0, 4, 0), .MoveSpeed = 10.0f };

        InitRasterizer(1280, 720);

        _shadowFb = std::make_shared<swr::Framebuffer>(1024, 1024);
        _shadowRast = std::make_unique<swr::Rasterizer>(_shadowFb);

        _lightDir = glm::vec3(0.0f, -1.0f, 0.0f);
        _lightPos = glm::vec3(-0.5f, 12.0f, 0.5f);
    }

    void InitRasterizer(uint32_t width, uint32_t height) {
        _fb = std::make_shared<swr::Framebuffer>(width, height);
        _rast = std::make_unique<swr::Rasterizer>(_fb);

        _frontTex = std::make_unique<ogl::Texture2D>(_fb->Width, _fb->Height, 1, GL_RGBA8);
        _tempPixels = std::make_unique<uint32_t[]>(_fb->Width * _fb->Height);
    }

    void Render() {
        static bool s_NormalMapping = true;
        static bool s_EnableShadows = true;

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
        ImGui::Checkbox("Wireframe", &_rast->EnableWireframe);
        ImGui::SliderFloat("Cam Speed", &_cam.MoveSpeed, 0.5f, 500.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::End();

        _cam.Update();

        auto renderStart = std::chrono::high_resolution_clock::now();

        if (s_EnableShadows) {
            RenderShadow();
        }

        double shadowElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;

        PhongShader shader = {
            .ProjMat = _cam.GetViewProjMatrix(),
            .LightPos = _lightPos,
            .ShadowProjMat = _shadowProjMat,
            .ShadowBuffer = s_EnableShadows ? _shadowFb.get() : nullptr,
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
        _frontTex->SetPixels(_tempPixels.get(), _fb->Width);

        double totalElapsed = (std::chrono::high_resolution_clock::now() - renderStart).count() / 1000000.0;
        swr::ProfilerStats& stats = swr::g_Stats;

        ImDrawList* drawList = ImGui::GetBackgroundDrawList();

        auto texId = (ImTextureID)(uintptr_t)_frontTex->Handle;
        drawList->AddImage(texId, drawList->GetClipRectMin(), drawList->GetClipRectMax(), ImVec2(0, 1), ImVec2(1, 0));

        ImGui::Begin("Rasterizer Stats");
        ImGui::Text("Frame: %.1fms (%.0f FPS), Shadow: %.1fms", totalElapsed, 1000.0 / totalElapsed, shadowElapsed);
        ImGui::Text("Vertex setup: %.1fms", stats.VertexSetup[0] / 1000000.0);
        ImGui::Text("Rasterize: %.1fms", stats.Rasterize[0] / 1000000.0);
        ImGui::Text("Triangles: %.1fK (%.1fK clipped)", stats.TrianglesDrawn / 1000.0, stats.TrianglesClipped / 1000.0);
        ImGui::Text("Bins filled: %.1fK", stats.BinsFilled / 1000.0);
        stats.Reset();
        ImGui::End();

        DrawTranslationGizmo(_lightPos, _lightDir);
    }

    void RenderShadow() {
        _shadowProjMat = glm::ortho(-12.0f, +12.0f, -12.0f, +12.0f, 0.1f, 40.0f) * glm::lookAt(_lightPos, _lightDir, glm::vec3(0, 1, 0));

        DepthOnlyShader shader = { .ProjMat = _shadowProjMat };

        _shadowFb->ClearDepth(1.0f);

        for (Mesh& mesh : _shadowModel->Meshes) {
            swr::VertexReader data((uint8_t*)&_shadowModel->VertexBuffer[mesh.VertexOffset],
                                   (uint8_t*)&_shadowModel->IndexBuffer[mesh.IndexOffset], mesh.IndexCount, swr::VertexReader::U16);

            _shadowRast->DrawIndexed(data, shader);
        }

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
            ImGui::End();
        }
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

        if (ImGuizmo::Manipulate(&viewMat[0].x, &projMat[0].x, ImGuizmo::TRANSLATE | ImGuizmo::ROTATE, ImGuizmo::WORLD, matrix)) {
            ImGuizmo::DecomposeMatrixToComponents(matrix, &pos.x, &rot.x, scale);
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