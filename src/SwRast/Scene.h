#pragma once

#include <vector>

#include "Texture.h"

struct Material {
    // Layer 0: BaseColor
    // Layer 1?: Normal (XY), Metallic (Z), Roughness (W)
    // Layer 2?: Emissive, BaseColor.A==255 is a mask for non-zero emission. Normal values range between [0..254]
    const swr::RgbaTexture2D* Texture;
    bool IsDoubleSided;
    uint8_t AlphaCutoff;
};
struct Meshlet {
    glm::vec3 BoundCenter;
    float BoundRadius;
    glm::vec3 ConeApex, ConeAxis;
    float ConeCutoff;

    uint8_t NumVertices, NumTriangles;
    uint8_t AlphaCutoff;
    uint32_t MaterialId;
    uint64_t TangentHandedness;

    alignas(64) float Positions[3][64];
    uint32_t TexCoords[64];                // float16
    uint32_t NormalTangents[64];           // oct16 x2
    uint8_t Indices[3][128];
};

struct TransformTRS {
    float3 Translation = {};
    float3 Scale = {};
    float4 Rotation = {};

    bool HasValue() const { return Scale.x != 0; }
    float4x4 ToMatrix() const;
};

struct ModelNode {
    uint32_t MeshOffset, MeshCount;
    uint32_t ParentIdx;

    std::vector<uint32_t> Joints;
    std::vector<glm::mat4x3> InverseBindMatrices;
    TransformTRS LocalTRS;
    glm::mat4x3 LocalTransform;
    glm::mat4x3 GlobalTransform;
};

struct Light {
    enum LightType : uint32_t { kTypeDirectional, kTypePoint, kTypeSpot };

    LightType Type;
    glm::vec3 Position;
    glm::vec3 Direction;
    glm::vec3 Color;
    float Intensity;  // cd or lux
    float Radius;
    float SpotInnerAngle, SpotOuterAngle;

    float InvRadiusSq, SpotScale, SpotOffset;  // Pre-computed

    void SetRadius(float radius) {
        Radius = radius > 0 ? radius : 1e+6; 
        InvRadiusSq = 1.0f / (Radius * Radius);
    }
    void SetSpotAngles(float innerAngle, float outerAngle) {
        SpotInnerAngle = innerAngle;
        SpotOuterAngle = outerAngle;
        SpotScale = 1.0f / std::max(std::cos(innerAngle) - std::cos(outerAngle), 1e-4f);
        SpotOffset = -std::cos(outerAngle) * SpotScale;
    }
};

struct Animation;
struct Scene;

struct Model {
    Scene* ParentScene;
    std::vector<Animation> Animations;

    std::vector<ModelNode> Nodes;
    std::vector<uint32_t> NodeIndicesPreDFS;

    ModelNode RootNode;

    ~Model();
};

template<typename T>
struct FlatPool {
    std::vector<T> Storage;

    T& operator[](uint32_t i) { return Storage[i]; }
    const T& operator[](uint32_t i) const { return Storage[i]; }

    FlatPool(size_t initialCap = 4) { Storage.reserve(initialCap); }

    // TODO: do better
    uint32_t AllocRange(uint32_t count) {
        uint32_t offset = Storage.size();
        Storage.resize(offset + count);
        return offset;
    }
    void FreeRange(uint32_t offset, uint32_t count) {
        if (offset + count == Storage.size()) {
            Storage.resize(offset);
        }
    }

    T* data() { return Storage.data(); }
    uint32_t size() const { return Storage.size(); }
};

struct Scene {
    FlatPool<Meshlet> Meshlets;
    FlatPool<Light> Lights;
    FlatPool<Material> Materials;
    std::vector<swr::RgbaTexture2D::Ptr> Textures;

    std::vector<std::unique_ptr<Model>> Models;

    Scene() : Meshlets(128 * 1024), Lights(1024), Materials(1024) {}

    Model* ImportGltf(const std::string& path);
};

struct Animation {
    enum LerpMode : uint8_t { kLerpNearest, kLerpLinear, kLerpSlerp, kLerpCubic };
    struct Sampler {
        uint32_t LastFrameIndex;
        uint32_t FrameCount;
        uint32_t DataOffset;
        LerpMode LerpMode;
    };
    struct Channel {
        Sampler SamplerTRS[3] = {};  // one per T/R/S
    };
    std::vector<Channel> Channels;
    std::vector<uint32_t> NodeToChannelMap; // UINT_MAX if empty
    std::vector<float> KeyframeData;
    float Duration = 0;

    void Interpolate(float timestamp, uint32_t channelIdx, TransformTRS& transform);
};