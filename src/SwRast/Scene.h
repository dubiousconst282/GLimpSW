#pragma once

#include <cfloat>
#include <vector>

#include "Texture.h"

namespace scene {

struct Material {
    // Layer 0: BaseColor
    // Layer 1?: Normal (XY), Metallic (Z), Roughness (W)
    // Layer 2?: Emissive, BaseColor.A==255 is a mask for non-zero emission. Normal values range between [0..254]
    const swr::RgbaTexture2D* Texture;
    bool IsDoubleSided;
    uint8_t AlphaCutoff;
};

struct Node {
    std::vector<Node> Children;
    uint32_t MeshletOffset = 0, MeshletCount = 0;
    glm::mat4 LocalTransform = glm::mat4(1);
    glm::mat4 GlobalTransform = glm::mat4(1);
    glm::vec3 BoundMin = glm::vec3(FLT_MAX);
    glm::vec3 BoundMax = glm::vec3(-FLT_MAX);
};
struct Meshlet {
    glm::vec4 BoundSphere;
    glm::vec3 ConeApex, ConeAxis;
    float ConeCutoff;

    uint16_t NumVertices, NumTriangles;
    uint32_t MaterialId;
    uint8_t AlphaCutoff;

    alignas(64) float Positions[3][64];
    uint32_t TexCoords[64];                // float16
    uint32_t Normals[64];                  // unorm10
    uint32_t Tangents[64];                 // unorm10, handedness
    uint8_t Indices[3][128];
};

struct Model {
    std::string BasePath;

    std::vector<Material> Materials;
    std::vector<swr::RgbaTexture2D::Ptr> Textures;

    std::vector<Meshlet> Meshlets;

    Node RootNode;

    Model(const std::string& path);

    void Traverse(const auto& visitor, const glm::mat4& _parentMat = glm::mat4(1.0f), Node* _node = nullptr) {
        if (_node == nullptr) {
            _node = &RootNode;
        }

        glm::mat4 localMat = _node->LocalTransform * _parentMat;

        if (_node->MeshletCount > 0 && !visitor(*_node, localMat)) return;

        for (Node& child : _node->Children) {
            Traverse(visitor, localMat, &child);
        }
    }
};

};  // namespace scene