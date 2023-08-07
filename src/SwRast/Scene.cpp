#include "Scene.h"

#include <unordered_map>
#include <filesystem>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

namespace scene {

static void PackNorm(int8_t* dst, float* src) {
    for (uint32_t i = 0; i < 3; i++) {
        dst[i] = (int8_t)std::round(src[i] * 127.0f);
    }
}

const swr::Texture2D* Model::LoadTexture(const std::string& name) {
    auto entry = Textures.find(name);

    if (entry != Textures.end()) {
        return &entry->second;
    }

    auto fullPath = std::filesystem::path(BasePath) / name;

    if (name.empty() || !std::filesystem::exists(fullPath)) {
        return nullptr;
    }

    auto tex = Textures.insert({name, swr::Texture2D::LoadImage(fullPath.string())});
    return &tex.first->second;
}
std::string GetTexturePath(const aiMaterial* mat, aiTextureType type) {
    if (mat->GetTextureCount(type) <= 0) {
        return "";
    }
    aiString path;
    mat->GetTexture(type, 0, &path);
    return std::string(path.data, path.length);
}

Node ConvertNode(const Model& model, aiNode* node) {
    //TODO: figure out wtf is going on with empty nodes
    Node impNode = {
        .Transform = glm::transpose(*(glm::mat4*)&node->mTransformation),
        .BoundMin = glm::vec3(INFINITY),
        .BoundMax = glm::vec3(-INFINITY),
    };

    for (uint32_t i = 0; i < node->mNumMeshes; i++) {
        impNode.Meshes.push_back(node->mMeshes[i]);

        const Mesh& mesh = model.Meshes[node->mMeshes[i]];
        impNode.BoundMin = glm::min(impNode.BoundMin, mesh.BoundMin);
        impNode.BoundMax = glm::max(impNode.BoundMax, mesh.BoundMax);
    }
    for (uint32_t i = 0; i < node->mNumChildren; i++) {
        Node childNode = ConvertNode(model, node->mChildren[i]);

        impNode.BoundMin = glm::min(impNode.BoundMin, childNode.BoundMin);
        impNode.BoundMax = glm::max(impNode.BoundMax, childNode.BoundMax);

        impNode.Children.emplace_back(std::move(childNode));
    }
    return impNode;
}


Model::Model(std::string_view path) {
    const auto processFlags = aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_CalcTangentSpace | 
                              aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs | aiProcess_SplitLargeMeshes |
                              aiProcess_OptimizeGraph | aiProcess_OptimizeMeshes;

    Assimp::Importer imp;

    if (sizeof(Index) < 4) {
        imp.SetPropertyInteger(AI_CONFIG_PP_SLM_VERTEX_LIMIT, (1 << (sizeof(Index) * 8)) - 1);
    }

    const aiScene* scene = imp.ReadFile(path.data(), processFlags);

    if (!scene || !scene->HasMeshes()) {
        throw std::exception("Could not import scene");
    }

    BasePath = std::filesystem::path(path).parent_path().string();

    for (int i = 0; i < scene->mNumMaterials; i++) {
        aiMaterial* mat = scene->mMaterials[i];

        Materials.push_back(Material { 
            .DiffuseTex = LoadTexture(GetTexturePath(mat, aiTextureType_DIFFUSE)),
            .NormalTex = LoadTexture(GetTexturePath(mat, aiTextureType_NORMALS)),
        });
    }

    uint32_t numVertices = 0;
    uint32_t numIndices = 0;

    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
        auto mesh = scene->mMeshes[i];
        numVertices += mesh->mNumVertices;

        for (int j = 0; j < mesh->mNumFaces; j++) {
            numIndices += mesh->mFaces[j].mNumIndices;
        }
    }

    VertexBuffer = std::make_unique<Vertex[]>(numVertices);
    IndexBuffer = std::make_unique<Index[]>(numIndices);

    uint32_t vertexPos = 0, indexPos = 0;

    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];

        Mesh& impMesh = Meshes.emplace_back(Mesh{
            .VertexOffset = vertexPos,
            .IndexOffset = indexPos,
            .Material = &Materials[mesh->mMaterialIndex],
            .BoundMin = glm::vec3(INFINITY),
            .BoundMax = glm::vec3(-INFINITY),
        });

        for (uint32_t j = 0; j < mesh->mNumVertices; j++) {
            Vertex& v = VertexBuffer[vertexPos++];

            glm::vec3 pos = *(glm::vec3*)&mesh->mVertices[j];
            *(glm::vec3*)&v.x = pos;

            if (mesh->HasTextureCoords(0)) {
                *(glm::vec2*)&v.u = *(glm::vec2*)&mesh->mTextureCoords[0][j];
            }
            if (mesh->HasNormals() && mesh->HasTangentsAndBitangents()) {
                PackNorm(&v.nx, &mesh->mNormals[j].x);
                PackNorm(&v.tx, &mesh->mTangents[j].x);
            }

            impMesh.BoundMin = glm::min(impMesh.BoundMin, pos);
            impMesh.BoundMax = glm::max(impMesh.BoundMax, pos);
        }

        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
            aiFace& face = mesh->mFaces[j];

            for (uint32_t k = 0; k < face.mNumIndices; k++) {
                IndexBuffer[indexPos++] = (Index)face.mIndices[k];
            }
        }
        impMesh.IndexCount = indexPos - impMesh.IndexOffset;
    }

    RootNode = ConvertNode(*this, scene->mRootNode);
}


void DepthPyramid::Update(const swr::Framebuffer& fb) {
    if (_width != fb.Width || _height != fb.Height) {
        Realloc(fb.Width, fb.Height);
    }

    // Downsample original depth buffer
    for (uint32_t y = 0; y < _height; y += 4) {
        for (uint32_t x = 0; x < _width; x += 4) {
            auto tile = _mm512_load_ps(&fb.DepthBuffer[fb.GetPixelOffset(x, y)]);
            // A B C D  ->  max(AC, BD)
            tile = _mm512_shuffle_f32x4(tile, tile, _MM_SHUFFLE(3, 1, 2, 0));
            auto rows = _mm256_max_ps(_mm512_extractf32x8_ps(tile, 0), _mm512_extractf32x8_ps(tile, 1));
            auto cols = _mm256_permute_ps(_mm256_max_ps(rows, _mm256_movehdup_ps(rows)), _MM_SHUFFLE(0, 0, 3, 1));
            cols = _mm256_permutevar8x32_ps(cols, _mm256_setr_epi32(0, 1, 4, 5, -1, -1, -1, -1));
            
            _mm_storel_pi((__m64*)&_storage[(x / 2) + (y / 2 + 0) * (_width / 2)], _mm256_extractf128_ps(cols, 0));
            _mm_storel_pi((__m64*)&_storage[(x / 2) + (y / 2 + 1) * (_width / 2)], _mm256_extractf128_ps(cols, 1));
        }
    }

    for (size_t i = 2; i <= _levels; i++)
    {
        float* src = &_storage[_offsets[i - 1]];
        float* dst = &_storage[_offsets[i + 0]];

        uint32_t w = _width >> (i - 1);
        uint32_t h = _height >> (i - 1);

        for (uint32_t y = 0; y < h; y += 2) {
            for (uint32_t x = 0; x < w; x += 16) {
                auto rows = _mm512_max_ps(_mm512_load_ps(&src[x + (y + 0) * w]),
                                          _mm512_load_ps(&src[x + (y + 1) * w]));

                auto cols = _mm512_max_ps(rows, _mm512_movehdup_ps(rows));
                //auto res = _mm256_permutevar8x32_epi32(cols, _mm256_setr_epi32(0, 2, 4, 6, -1, -1, -1, -1));

                //_mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(res));
            }
        }
    }
    
}

void DepthPyramid::Realloc(uint32_t width, uint32_t height) {
    _width = width;
    _height = height;
    _levels = std::bit_floor(std::min(width, height));
    assert(_levels < 16);

    uint32_t offset = 0;

    for (uint32_t i = 1; i <= _levels; i++) {
        _offsets[i - 1] = offset;
        offset += (width >> i) * (height >> i);
    }
    _storage = (float*)_mm_malloc(offset * 4, 64);
}

}; // namespace scene