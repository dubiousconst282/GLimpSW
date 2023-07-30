#include "Model.h"

#include <unordered_map>
#include <filesystem>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

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


Model::Model(std::string_view path) {
    const auto processFlags = aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_CalcTangentSpace |
                              aiProcess_JoinIdenticalVertices | aiProcess_PreTransformVertices | aiProcess_FlipUVs;

    Assimp::Importer imp;
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
    IndexBuffer = std::make_unique<VertexIndex[]>(numIndices);

    uint32_t vertexPos = 0, indexPos = 0;

    for (uint32_t i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        
        Mesh& impMesh = Meshes.emplace_back(Mesh {
            .VertexOffset = vertexPos,
            .IndexOffset = indexPos,
            .Material = &Materials[mesh->mMaterialIndex],
        });

        for (uint32_t j = 0; j < mesh->mNumVertices; j++) {
            Vertex& v = VertexBuffer[vertexPos++];

            v.x = mesh->mVertices[j].x;
            v.y = mesh->mVertices[j].y;
            v.z = mesh->mVertices[j].z;

            v.u = mesh->mTextureCoords[0][j].x;
            v.v = mesh->mTextureCoords[0][j].y;

            PackNorm(&v.nx, &mesh->mNormals[j].x);
            PackNorm(&v.tx, &mesh->mTangents[j].x);
        }

        for (uint32_t j = 0; j < mesh->mNumFaces; j++) {
            aiFace& face = mesh->mFaces[j];

            for (uint32_t k = 0; k < face.mNumIndices; k++) {
                IndexBuffer[indexPos++] = (VertexIndex)face.mIndices[k];
            }
        }
        impMesh.IndexCount = indexPos - impMesh.IndexOffset;
    }
}