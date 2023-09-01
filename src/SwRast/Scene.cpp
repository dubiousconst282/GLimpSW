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
static std::string GetTexturePath(const aiMaterial* mat, aiTextureType type) {
    if (mat->GetTextureCount(type) <= 0) {
        return "";
    }
    aiString path;
    mat->GetTexture(type, 0, &path);
    return std::string(path.data, path.length);
}

static swr::RgbaTexture2D* LoadTexture(Model& m, const aiMaterial* mat, aiTextureType type) {
    std::string name = GetTexturePath(mat, type);

    auto basePath = std::filesystem::path(m.BasePath);

    auto fullPath = basePath / name;

    if (name.empty() || !std::filesystem::exists(fullPath)) {
        return nullptr;
    }

    auto cached = m.Textures.find(name);

    if (cached != m.Textures.end()) {
        return &cached->second;
    }

    if (type == aiTextureType_NORMALS) {
        // Merge Metallic-Roughness map into normals texture
        // RG: Normals, BA: MR
        std::string mrName = GetTexturePath(mat, aiTextureType_DIFFUSE_ROUGHNESS);
        auto mrPath = basePath / mrName;

        if (mrName.empty() || !std::filesystem::exists(mrPath)) {
            mrPath = "";
        }
        auto tex = swr::texutil::LoadNormalMap(fullPath.string(), mrPath.string());
        auto slot = m.Textures.insert({ name, std::move(tex) });
        return &slot.first->second;
    } else {
        auto tex = swr::texutil::LoadImage(fullPath.string());
        auto slot = m.Textures.insert({ name, std::move(tex) });
        return &slot.first->second;
    }
}

Node ConvertNode(const Model& model, aiNode* node) {
    //TODO: figure out wtf is going on with empty nodes
    //FIXME: apply transform on node AABBs
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
                              aiProcess_OptimizeGraph;// | aiProcess_OptimizeMeshes;

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

        Materials.push_back(Material{
            .DiffuseTex = LoadTexture(*this, mat, aiTextureType_BASE_COLOR),
            .NormalTex = LoadTexture(*this, mat, aiTextureType_NORMALS),
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

float DepthPyramid::GetDepth(float u, float v, float lod) const {
    uint32_t level = std::clamp((uint32_t)lod, 0u, _levels - 1);

    int32_t w = (int32_t)(_width >> level);
    int32_t h = (int32_t)(_height >> level);
    int32_t x = (int32_t)(u * w);
    int32_t y = (int32_t)(v * h);

    const auto Sample = [&](int32_t xo, int32_t yo) {
        int32_t i = std::clamp(x + xo, 0, w - 1) + std::clamp(y + yo, 0, h - 1) * w;
        return _storage[_offsets[level] + (uint32_t)i];
    };
    return std::max({ Sample(0, 0), Sample(1, 0), Sample(0, 1), Sample(1, 1) });
}

bool DepthPyramid::IsVisibleAABB(const glm::vec3& bbMin, const glm::vec3& bbMax) const {
    if (!_storage) return true;

    glm::vec3 rectMin = glm::vec3(INFINITY), rectMax = glm::vec3(-INFINITY);
    bool partialOut = false;
    bool combinedOut = true;

    for (uint32_t i = 0; i < 8; i++) {
        glm::bvec3 corner = { (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1 };
        glm::vec4 p = _viewProj * glm::vec4(glm::mix(bbMin, bbMax, corner), 1.0f);

        glm::vec3 rp = {
            p.x / p.w * 0.5f + 0.5f,
            p.y / p.w * 0.5f + 0.5f,
            p.z / p.w,
        };
        rectMin = glm::min(rectMin, rp);
        rectMax = glm::max(rectMax, rp);

        bool outflag = (p.x < -p.w || p.y > p.w) || (p.y < -p.w || p.y > p.w) || (p.z < 0 || p.z > p.w);
        partialOut |= outflag;
        combinedOut &= outflag;
    }

    // Hacky frustum check. Definitely won't work for AABBs bigger than the frustum, but that seems rare enough.
    if (combinedOut) return false;

    // We don't do clipping, so the ccclusion test doesn't work properly with
    // AABBs that are partially out the view frustum.
    // Consider them as visible, otherwise we'll get constant flicker
    if (partialOut) return true;

    float sizeX = (rectMax.x - rectMin.x) * _width;
    float sizeY = (rectMax.y - rectMin.y) * _height;
    float lod = std::ceil(std::log2(std::max(sizeX, sizeY) / 2.0f));

    float screenDepth = GetDepth((rectMin.x + rectMax.x) * 0.5f, (rectMin.y + rectMax.y) * 0.5f, lod);

    return rectMin.z <= screenDepth;
}

void DepthPyramid::Update(const swr::Framebuffer& fb, const glm::mat4& viewProj) {
    EnsureStorage(fb.Width, fb.Height);

    // Downsample original depth buffer
    for (uint32_t y = 0; y < fb.Height; y += 4) {
        for (uint32_t x = 0; x < fb.Width; x += 4) {
            auto tile = _mm512_load_ps(&fb.DepthBuffer[fb.GetPixelOffset(x, y)]);
            // A B C D  ->  max(AC, BD)
            tile = _mm512_shuffle_f32x4(tile, tile, _MM_SHUFFLE(3, 1, 2, 0));
            auto rows = _mm256_max_ps(_mm512_extractf32x8_ps(tile, 0), _mm512_extractf32x8_ps(tile, 1));
            auto cols = _mm256_max_ps(rows, _mm256_movehdup_ps(rows));
            cols = _mm256_permutevar8x32_ps(cols, _mm256_setr_epi32(0, 2, -1, -1, 4, 6, -1, -1));

            _mm_storel_pi((__m64*)&_storage[(x / 2) + (y / 2 + 0) * _width], _mm256_extractf128_ps(cols, 0));
            _mm_storel_pi((__m64*)&_storage[(x / 2) + (y / 2 + 1) * _width], _mm256_extractf128_ps(cols, 1));
        }
    }

    for (uint32_t i = 1; i < _levels; i++) {
        float* src = &_storage[_offsets[i - 1]];
        float* dst = &_storage[_offsets[i + 0]];

        uint32_t w = _width >> (i - 1);
        uint32_t h = _height >> (i - 1);

        // TODO: edge clamping and stuff
        for (uint32_t y = 0; y < h; y += 2) {
            for (uint32_t x = 0; x < w; x += 16) {
                auto rows = _mm512_max_ps(_mm512_loadu_ps(&src[x + (y + 0) * w]), 
                                          _mm512_loadu_ps(&src[x + (y + 1) * w]));

                auto cols = _mm512_max_ps(rows, _mm512_movehdup_ps(rows));
                auto res = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1), cols);

                _mm256_storeu_ps(&dst[(x / 2) + (y / 2) * (w / 2)], _mm512_extractf32x8_ps(res, 0));
            }
        }
    }

    _viewProj = viewProj;
}

void DepthPyramid::EnsureStorage(uint32_t width, uint32_t height) {
    if (_width == width / 2 && _height == height / 2) return;

    _width = width / 2;
    _height = height / 2;
    _levels = (uint32_t)std::bit_width(std::min(_width, _height));
    assert(_levels < 16);

    uint32_t offset = 0;

    for (uint32_t i = 0; i < _levels; i++) {
        _offsets[i] = offset;
        offset += (_width >> i) * (_height >> i);
    }
    _storage = (float*)_mm_malloc(offset * 4 + 128, 64);
}

}; // namespace scene