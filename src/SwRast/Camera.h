#pragma once

#include "SwRast.h"
#include <glm/gtx/euler_angles.hpp>

//https://github.com/syoyo/tinygltf
//https://learnopengl.com/Getting-started/Camera
struct Camera {
    glm::vec3 Position, _ViewPosition;
    glm::quat Rotation;
    glm::vec2 Euler; //yaw, pitch

    float FieldOfView = 90.0f;
    float AspectRatio = 1.0f;
    float MoveSpeed = 30.0f;

    glm::mat4 GetViewMatrix() { return glm::translate(glm::mat4_cast(Rotation), -_ViewPosition); }
    glm::mat4 GetProjectionMatrix() { return glm::perspective(FieldOfView, AspectRatio, 0.05f, 1000.0f); }
    glm::mat4 GetViewProjMatrix() { return GetProjectionMatrix() * GetViewMatrix(); }

    void Update() {
        ImGuiIO& io = ImGui::GetIO();
        float sensitivity = io.DeltaTime * 0.25f;
        float speed = io.DeltaTime * MoveSpeed;
        float pitchRange = glm::pi<float>() / 2.0f;
        float damping = 0.3f; //TODO: calculate this based on dt
        glm::vec3 mv(0);

        if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow)) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                float rx = io.MouseDelta.x * sensitivity;
                float ry = io.MouseDelta.y * sensitivity;

                Euler.x = NormalizeRadians(Euler.x - rx);
                Euler.y = std::clamp(Euler.y - ry, -pitchRange, +pitchRange);
            }

            if (ImGui::IsKeyDown(ImGuiKey_W)) mv.z--;
            if (ImGui::IsKeyDown(ImGuiKey_S)) mv.z++;
            if (ImGui::IsKeyDown(ImGuiKey_A)) mv.x--;
            if (ImGui::IsKeyDown(ImGuiKey_D)) mv.x++;
            if (ImGui::IsKeyDown(ImGuiKey_Space)) mv.y++;
            if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) mv.y--;
        }

        glm::quat destRotation = glm::eulerAngleX(-Euler.y) * glm::eulerAngleY(-Euler.x);

        Position += glm::vec3(mv * destRotation) * speed;

        _ViewPosition = glm::lerp(_ViewPosition, Position, damping);
        Rotation = glm::slerp(Rotation, destRotation, damping);

        AspectRatio = io.DisplaySize.x / io.DisplaySize.y;

        ImGui::Begin("Camera");
        ImGui::Text("D %.3f  %f", damping, MoveSpeed);
        ImGui::InputFloat3("Pos", &Position.x);
        ImGui::InputFloat2("Rot", &Euler.x);
        ImGui::End();
    }

    static float NormalizeRadians(float ang) {
        const float tau = 6.28318530717958f;
        float r = glm::round(ang * (1.0f / tau));
        return ang - (r * tau);
    }
};