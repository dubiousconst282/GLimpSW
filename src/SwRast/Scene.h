#pragma once

#include <cfloat>
#include <vector>
#include <memory>

#include "Rasterizer.h"
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

// Hierarchical depth buffer for occlusion culling
// - https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/
// - https://vkguide.dev/docs/gpudriven/compute_culling/
class DepthPyramid {
    swr::AlignedBuffer<float> _storage = nullptr;
    uint32_t _width, _height, _levels;
    uint32_t _offsets[16]{};
    glm::mat4 _viewProj;

    void EnsureStorage(uint32_t width, uint32_t height);

public:
    void Update(const swr::Framebuffer& fb, const glm::mat4& viewProj);

    float GetDepth(float u, float v, float lod) const;

    float* GetMipBuffer(uint32_t level, uint32_t& width, uint32_t& height) {
        width = _width >> level;
        height = _height >> level;
        return &_storage[_offsets[level]];
    }

    swr::v_float __vectorcall SampleDepth(swr::v_float x, swr::v_float y, uint32_t level) const {
        swr::v_int ix = swr::simd::round2i(x * (int32_t)_width);
        swr::v_int iy = swr::simd::round2i(y * (int32_t)_height);

        return SampleDepth(ix << 1, iy << 1, level);
    }
    swr::v_float __vectorcall SampleDepth(swr::v_int ix, swr::v_int iy, uint32_t level) const {
        ix = ix >> 1, iy = iy >> 1;
        uint16_t boundMask = _mm512_cmplt_epu32_mask(ix, swr::v_int((int32_t)_width)) & _mm512_cmplt_epu32_mask(iy, swr::v_int((int32_t)_height));
        swr::v_int indices = (ix >> level) + (iy >> level) * (int32_t)(_width >> level);

        return _mm512_mask_i32gather_ps(_mm512_set1_ps(1.0f), boundMask, indices, &_storage[_offsets[level]], 4);
    }
};

};  // namespace scene