#pragma once

#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>

#include "SwRast.h"
#include "Texture.h"

namespace scene {

struct Material {
    const swr::RgbaTexture2D* DiffuseTex;
    const swr::RgbaTexture2D* NormalTex;
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
    std::unordered_map<std::string, swr::RgbaTexture2D> Textures;

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
};

// Hierarchical depth buffer for occlusion culling
// - https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
// - https://vkguide.dev/docs/gpudriven/compute_culling/
class DepthPyramid {
    float* _storage = nullptr;
    uint32_t _width, _height, _levels;
    uint32_t _offsets[16]{};
    glm::mat4 _viewProj;

    void EnsureStorage(uint32_t width, uint32_t height);

public:
    ~DepthPyramid() { _mm_free(_storage); }

    void Update(const swr::Framebuffer& fb, const glm::mat4& viewProj);

    float GetDepth(float u, float v, float lod) const;

    bool IsVisible(const Mesh& mesh, const glm::mat4& transform) const;

    float* GetMipBuffer(uint32_t level, uint32_t& width, uint32_t& height) {
        width = _width >> level;
        height = _height >> level;
        return &_storage[_offsets[level]];
    }

    swr::VFloat __vectorcall SampleDepth(swr::VFloat x, swr::VFloat y, uint32_t level) const {
        swr::VInt ix = swr::simd::round2i(x * (int32_t)_width);
        swr::VInt iy = swr::simd::round2i(y * (int32_t)_height);

        return SampleDepth(ix << 1, iy << 1, level);
    }
    swr::VFloat __vectorcall SampleDepth(swr::VInt ix, swr::VInt iy, uint32_t level) const {
        ix = ix >> 1, iy = iy >> 1;
        uint16_t boundMask = _mm512_cmplt_epu32_mask(ix, swr::VInt((int32_t)_width)) & _mm512_cmplt_epu32_mask(iy, swr::VInt((int32_t)_height));
        swr::VInt indices = (ix >> level) + (iy >> level) * (int32_t)(_width >> level);

        return _mm512_mask_i32gather_ps(_mm512_set1_ps(1.0f), boundMask, indices, &_storage[_offsets[level]], 4);
    }
};

};  // namespace scene