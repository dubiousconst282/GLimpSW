#pragma once

#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>

#include "SwRast.h"

struct Material {
    const swr::Texture2D* DiffuseTex;
    const swr::Texture2D* NormalTex;
};

struct Mesh {
    uint32_t VertexOffset, IndexOffset, IndexCount;
    Material* Material;
};

struct Vertex {
    float x, y, z;
    float u, v;
    int8_t nx, ny, nz;
    int8_t tx, ty, tz;
};

class Model {
public:
    using VertexIndex = uint16_t;

    std::string BasePath;

    std::vector<Mesh> Meshes;
    std::vector<Material> Materials;
    std::unordered_map<std::string, swr::Texture2D> Textures;

    std::unique_ptr<Vertex[]> VertexBuffer;
    std::unique_ptr<VertexIndex[]> IndexBuffer;

    Model(std::string_view path);

    const swr::Texture2D* LoadTexture(const std::string& name);
};