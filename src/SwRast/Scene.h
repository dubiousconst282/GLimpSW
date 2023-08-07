#pragma once

#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>

#include "SwRast.h"

namespace scene {

struct Material {
    const swr::Texture2D* DiffuseTex;
    const swr::Texture2D* NormalTex;
};

struct Mesh {
    uint32_t VertexOffset, IndexOffset, IndexCount;
    Material* Material;
    glm::vec3 BoundMin, BoundMax;
};


struct Node {
    std::vector<Node> Children;
    std::vector<uint32_t> Meshes;
    glm::mat4 Transform;
    glm::vec3 BoundMin, BoundMax;
};

struct Vertex {
    float x, y, z;
    float u, v;
    int8_t nx, ny, nz;
    int8_t tx, ty, tz;
};
using Index = uint16_t;

class Model {
public:

    std::string BasePath;

    std::vector<Mesh> Meshes;
    std::vector<Material> Materials;
    std::unordered_map<std::string, swr::Texture2D> Textures;

    std::unique_ptr<Vertex[]> VertexBuffer;
    std::unique_ptr<Index[]> IndexBuffer;

    Node RootNode;

    Model(std::string_view path);

    void Traverse(std::function<bool(Node&, const glm::mat4&)> visitor, const glm::mat4& _parentMat = glm::mat4(1.0f), Node* _node = nullptr) {
        if (_node == nullptr) {
            _node = &RootNode;
        }

        glm::mat4 localMat = _node->Transform * _parentMat;

        if (_node->Meshes.size() > 0 && !visitor(*_node, localMat)) {
            return;
        }
        for (Node& child : _node->Children) {
            Traverse(visitor, localMat, &child);
        }
    }

    const swr::Texture2D* LoadTexture(const std::string& name);
};

// Hierarchical depth buffer for occlusion culling
// - https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
// - https://vkguide.dev/docs/gpudriven/compute_culling/
class DepthPyramid {
    float* _storage;
    uint32_t _width, _height, _levels;
    uint32_t _offsets[16];

    void Realloc(uint32_t width, uint32_t height);

public:
    ~DepthPyramid() { _mm_free(_storage); }

    void Update(const swr::Framebuffer& fb);
};

};  // namespace scene