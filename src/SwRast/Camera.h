#pragma once

#include "SwRast.h"
#include <glm/gtx/euler_angles.hpp>

struct Camera {
    enum class InputMode { FirstPerson, Arcball };

    glm::vec3 Position;
    glm::vec2 Euler; //yaw, pitch
    float ArcDistance = 5.0f;

    InputMode Mode = InputMode::FirstPerson;

    float FieldOfView = 90.0f;
    float AspectRatio = 1.0f;
    float MoveSpeed = 10.0f;
    float NearZ = 0.01f, FarZ = 1000.0f;

    // Smoothed values
    glm::vec3 _ViewPosition;
    glm::quat _ViewRotation;
    float _ViewArcDistance = 5.0f;

    glm::mat4 GetViewMatrix() {
        if (Mode == InputMode::Arcball) {
            return glm::lookAt(_ViewPosition, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        }
        return glm::translate(glm::mat4_cast(_ViewRotation), -_ViewPosition);
    }
    glm::mat4 GetProjMatrix() { return glm::perspective(glm::radians(FieldOfView), AspectRatio, NearZ, FarZ); }

    void Update() {
        ImGuiIO& io = ImGui::GetIO();
        float sensitivity = io.DeltaTime * 0.25f;
        float speed = io.DeltaTime * MoveSpeed;
        float pitchRange = glm::pi<float>() / 2.01f; // a bit less than 90deg to prevent issues with LookAt()
        float damping = 1.0f - powf(1e-8f, io.DeltaTime);  // https://www.construct.net/en/blogs/ashleys-blog-2/using-lerp-delta-time-924
        glm::quat destRotation = glm::eulerAngleXY(-Euler.y, -Euler.x);

        if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) && !ImGuizmo::IsUsing()) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                float rx = io.MouseDelta.x * sensitivity;
                float ry = io.MouseDelta.y * sensitivity;

                Euler.x = NormalizeRadians(Euler.x - rx);
                Euler.y = std::clamp(Euler.y - ry, -pitchRange, +pitchRange);

                destRotation = glm::eulerAngleXY(-Euler.y, -Euler.x);
            }

            if (Mode == InputMode::FirstPerson) {
                glm::vec3 mv(0.0f);
                
                if (ImGui::IsKeyDown(ImGuiKey_W)) mv.z--;
                if (ImGui::IsKeyDown(ImGuiKey_S)) mv.z++;
                if (ImGui::IsKeyDown(ImGuiKey_A)) mv.x--;
                if (ImGui::IsKeyDown(ImGuiKey_D)) mv.x++;
                if (ImGui::IsKeyDown(ImGuiKey_Space)) mv.y++;
                if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) mv.y--;

                Position += mv * destRotation * speed;
            } else if (Mode == InputMode::Arcball) {
                if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
                    ArcDistance = std::clamp(ArcDistance - io.MouseWheel * 0.5f, NearZ, FarZ * 0.8f);
                }
                _ViewArcDistance = glm::lerp(_ViewArcDistance, ArcDistance, damping);

                Position = glm::vec3(0, 0, ArcDistance) * destRotation;
                // TODO: implement panning for arcball camera
            }
        }
        _ViewRotation = glm::slerp(_ViewRotation, destRotation, damping);
        _ViewPosition = glm::lerp(_ViewPosition, Position, damping);

        AspectRatio = io.DisplaySize.x / io.DisplaySize.y;
    }

    static float NormalizeRadians(float ang) {
        const float tau = 6.28318530717958f;
        float r = glm::round(ang * (1.0f / tau));
        return ang - (r * tau);
    }
};