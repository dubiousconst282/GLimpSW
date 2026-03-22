#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/matrix_transform.hpp>

struct Camera {
    enum InputMode { FirstPerson, Arcball };
    struct InputState {
        glm::vec3 DeltaMovement = {};   // X: -left +right, Z: -forward +backward, Y: +up -down
        glm::vec2 DeltaRotation = {};   // Change in yaw and pitch.
        bool IsPanning = false;         // In arcball mode, use `DeltaRotation` for panning instead of rotation.
        float DeltaTime = 0;
        float MouseWheel = 0;           // For zooming in/out in arcball mode.
        glm::vec2 DisplaySize = { 1, 1 };
    };

    glm::dvec3 Position;
    glm::vec2 Euler;  // yaw, pitch
    float ArcDistance = 5.0f;

    InputMode Mode = FirstPerson;

    float FieldOfView = 90.0f;
    float AspectRatio = 1.0f;
    float MoveSpeed = 10.0f;
    float NearZ = 0.01f;

    // Smoothed values available after Update().
    glm::dvec3 ViewPosition;
    glm::quat ViewRotation;

    glm::mat4 GetViewMatrix(bool translateToOrigin) const {
        glm::mat4 mat = glm::mat4_cast(ViewRotation);
        if (translateToOrigin) mat = glm::translate(mat, glm::vec3(-ViewPosition));
        return mat;
    }
    glm::mat4 GetProjMatrix() const {
        // Reverse Z perspective matrix: https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
        float f = 1.0f / glm::tan(glm::radians(FieldOfView) / 2.0f);
        float f_ar = f / AspectRatio;

        return glm::mat4(
            f_ar, 0.0f,  0.0f,  0.0f,
            0.0f,   -f,  0.0f,  0.0f,
            0.0f, 0.0f,  0.0f, -1.0f,
            0.0f, 0.0f, NearZ,  0.0f);
    }

    // Update camera position and rotation from given inputs.
    void Update(const InputState& inputs, float smoothingDurationMs = 150) {
        if (inputs.DeltaRotation != glm::vec2(0) && !inputs.IsPanning) {
            const float pitchRange = glm::pi<float>() / 2.0f;
            Euler.x = NormalizeRadians(Euler.x + inputs.DeltaRotation.x);
            Euler.y = glm::clamp(Euler.y + inputs.DeltaRotation.y, -pitchRange, +pitchRange);
        }
        // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        // rotateX(-Euler.y) * rotateY(Euler.x);
        float sx = glm::sin(Euler.y * -0.5f);
        float cx = glm::cos(Euler.y * -0.5f);
        float sy = glm::sin(Euler.x * 0.5f);
        float cy = glm::cos(Euler.x * 0.5f);
        glm::quat destRotation = { cx * cy, sx * cy, cx * sy, sx * sy }; // wxyz

        if (Mode == InputMode::FirstPerson) {
            float speed = inputs.DeltaTime * MoveSpeed;
            glm::vec3 mv = inputs.DeltaMovement * speed;
            Position.y += mv.y; // up/down axis locked
            Position += glm::vec3(mv.x, 0, mv.z) * destRotation;
        }
        // double smoothingFactor = glm::pow(0.01, 1000.0 / smoothingDurationMs);  // ~1% max dist to target after given duration
        // double blend = 1.0 - glm::pow(smoothingFactor, inputs.DeltaTime);       // https://gamedev.stackexchange.com/a/149106
        double blend = 1.0 - glm::pow(0.01, inputs.DeltaTime * (1000.0 / smoothingDurationMs));

        // Avoid interpolating when close to final position to prevent jittering.
        if (smoothingDurationMs > 0 && glm::abs(glm::dot(ViewRotation, destRotation)) < 0.99999905f) {
            ViewRotation = glm::slerp(ViewRotation, destRotation, (float)blend);
        } else {
            ViewRotation = destRotation;
        }

        if (Mode == InputMode::Arcball) {
            if (inputs.IsPanning) {
                float speed = glm::max(ArcDistance, 1.0f) * 0.05f * MoveSpeed;
                Position += glm::vec3(inputs.DeltaRotation * speed, 0) * ViewRotation;
            }
            ArcDistance = glm::max(ArcDistance + inputs.MouseWheel, NearZ);
            ViewPosition = Position + glm::dvec3(glm::vec3(0, 0, ArcDistance) * ViewRotation);
        } else if (smoothingDurationMs > 0 && glm::distance(ViewPosition, Position) > MoveSpeed * 0.0005) {
            ViewPosition = glm::mix(ViewPosition, Position, blend);
        } else {
            ViewPosition = Position;
        }

        AspectRatio = inputs.DisplaySize.x / inputs.DisplaySize.y;
    }
#ifdef IMGUI_API
    static InputState GetInputsFromImGui(bool mouseLocked = false, float mouseSensitivity = 0.5f) {
        ImGuiIO& io = ImGui::GetIO();
        InputState inputs;

        inputs.DeltaTime = io.DeltaTime;
        inputs.DisplaySize = glm::vec2(io.DisplaySize.x, io.DisplaySize.y);

        if (!io.WantCaptureKeyboard) {
            glm::vec3 mv(0);
            if (ImGui::IsKeyDown(ImGuiKey_W)) mv.z--;
            if (ImGui::IsKeyDown(ImGuiKey_A)) mv.x--;
            if (ImGui::IsKeyDown(ImGuiKey_S)) mv.z++;
            if (ImGui::IsKeyDown(ImGuiKey_D)) mv.x++;
            if (ImGui::IsKeyDown(ImGuiKey_Space)) mv.y++;
            if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) mv.y--;

            if (ImGui::IsKeyDown(ImGuiKey_Q)) mv *= 5.0f;
            if (ImGui::IsKeyDown(ImGuiKey_E)) mv /= 5.0f;

            inputs.DeltaMovement = mv;
        }
        if (mouseLocked || (!io.WantCaptureMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Left))) {
            float scale = glm::radians(mouseSensitivity);
            inputs.DeltaRotation.x = io.MouseDelta.x * scale;
            inputs.DeltaRotation.y = -io.MouseDelta.y * scale;
            inputs.IsPanning = !io.WantCaptureKeyboard && ImGui::IsKeyDown(ImGuiMod_Alt);
        }
        if (!io.WantCaptureMouse) {
            inputs.MouseWheel = io.MouseWheel * mouseSensitivity;
        }
        return inputs;
    }
#endif

    static float NormalizeRadians(float ang) {
        const float tau = glm::two_pi<float>();
        float r = glm::round(ang * (1.0f / tau));
        return ang - (r * tau);
    }
};

// Computes inverse projection matrix, scaled to take coordinates in range [0..viewSize] rather than [-1..1].
static glm::mat4 GetInverseScreenProjMatrix(const glm::mat4& mat, glm::ivec2 viewSize, glm::vec2 subpixelOffset = glm::vec2(0.5f)) {
    glm::mat4 invProj = glm::inverse(mat);
    invProj = glm::translate(invProj, glm::vec3(-1.0f, -1.0f, 0.0f));
    invProj = glm::scale(invProj, glm::vec3(2.0f / glm::vec2(viewSize), 1.0f));
    invProj = glm::translate(invProj, glm::vec3(subpixelOffset, 0.0f));
    return invProj;
}
