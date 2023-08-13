#pragma once

#include "SwRast.h"
#include <glm/gtx/euler_angles.hpp>

struct Camera {
    glm::vec3 Position, _ViewPosition;
    glm::quat Rotation;
    glm::vec2 Euler; //yaw, pitch

    float FieldOfView = 90.0f;
    float AspectRatio = 1.0f;
    float MoveSpeed = 30.0f;

    glm::mat4 GetViewMatrix() { return glm::translate(glm::mat4_cast(Rotation), -_ViewPosition); }
    glm::mat4 GetProjMatrix() { return glm::perspective(FieldOfView, AspectRatio, 0.05f, 1000.0f); }

    void Update() {
        ImGuiIO& io = ImGui::GetIO();
        float sensitivity = io.DeltaTime * 0.25f;
        float speed = io.DeltaTime * MoveSpeed;
        float pitchRange = glm::pi<float>() / 2.0f;
        float damping = 1.0f - powf(0.00001f, io.DeltaTime);  // https://www.construct.net/en/blogs/ashleys-blog-2/using-lerp-delta-time-924

        glm::vec3 mv(0);

        if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) && !ImGuizmo::IsUsing()) {
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
    }

    static float NormalizeRadians(float ang) {
        const float tau = 6.28318530717958f;
        float r = glm::round(ang * (1.0f / tau));
        return ang - (r * tau);
    }
};