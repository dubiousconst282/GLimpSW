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
static std::string GetTextureName(const aiMaterial* mat, aiTextureType type) {
    if (mat->GetTextureCount(type) <= 0) {
        return "";
    }
    aiString path;
    mat->GetTexture(type, 0, &path);
    return std::string(path.data, path.length);
}

static void CombineNormalMR(swr::StbImage& normalMap, const swr::StbImage& mrMap) {
    uint32_t count = normalMap.Width * normalMap.Height * 4;
    uint8_t* pixels = normalMap.Data.get();
    uint8_t* mrPixels = mrMap.Data.get();

    for (uint32_t i = 0; i < count; i += 4) {
        // Re-normalize to get rid of JPEG artifacts
        glm::vec3 N = glm::vec3(pixels[i + 0], pixels[i + 1], pixels[i + 2]);
        N = glm::normalize(N / 127.0f - 1.0f) * 127.0f + 127.0f;

        pixels[i + 0] = (uint8_t)roundf(N.x);
        pixels[i + 1] = (uint8_t)roundf(N.y);

        // Overwrite BA channels from normal map with Metallic and Roughness.
        // The normal Z can be reconstructed with `sqrt(1.0f - dot(n.xy, n.xy))`
        if (mrPixels != nullptr) {
            pixels[i + 2] = mrPixels[i + 2];  // Metallic
            pixels[i + 3] = mrPixels[i + 1];  // Roughness
        }
    }
}

static void InsertEmissiveMask(swr::StbImage& baseMap, const swr::StbImage& emissiveMap) {
    uint32_t count = baseMap.Width * baseMap.Height * 4;
    uint8_t* pixels = baseMap.Data.get();
    uint8_t* emissivePixels = emissiveMap.Data.get();

    for (uint32_t i = 0; i < count; i += 4) {
        const uint8_t L = 8;
        bool isLit = emissivePixels[i + 0] > L || emissivePixels[i + 1] > L || emissivePixels[i + 2] > L;
        pixels[i + 3] = isLit ? 255 : std::min(pixels[i + 3], (uint8_t)254);
    }
}

static swr::StbImage LoadImage(Model& m, std::string_view name) {
    auto fullPath = std::filesystem::path(m.BasePath) / name;

    if (name.empty() || !std::filesystem::exists(fullPath)) {
        return { };
    }
    return swr::StbImage::Load(fullPath.string());
}

static swr::RgbaTexture2D* LoadTextures(Model& m, const aiMaterial* mat) {
    std::string name = GetTextureName(mat, aiTextureType_BASE_COLOR);
    auto cached = m.Textures.find(name);

    if (cached != m.Textures.end()) {
        return &cached->second;
    }
    swr::StbImage baseColorImg = LoadImage(m, name);
    swr::StbImage normalImg = LoadImage(m, GetTextureName(mat, aiTextureType_NORMALS));
    swr::StbImage metalRoughImg = LoadImage(m, GetTextureName(mat, aiTextureType_DIFFUSE_ROUGHNESS));
    swr::StbImage emissiveImg = LoadImage(m, GetTextureName(mat, aiTextureType_EMISSIVE));

    uint32_t numLayers = 1 + (normalImg.Width ? 1 : 0) + (emissiveImg.Width ? 1 : 0);
    swr::RgbaTexture2D tex(baseColorImg.Width, baseColorImg.Height, 8, numLayers);

    if (normalImg.Width) {
        CombineNormalMR(normalImg, metalRoughImg);
        tex.SetPixels(normalImg.Data.get(), normalImg.Width, 1);
    }
    if (emissiveImg.Width) {
        InsertEmissiveMask(baseColorImg, emissiveImg);
        tex.SetPixels(emissiveImg.Data.get(), emissiveImg.Width, 2);
    }
    tex.SetPixels(baseColorImg.Data.get(), baseColorImg.Width, 0);
    tex.GenerateMips();

    auto slot = m.Textures.insert({ name, std::move(tex) });
    return &slot.first->second;
}

Node ConvertNode(const Model& model, aiNode* node) {
    //TODO: figure out wtf is going on with empty nodes
    //FIXME: apply transform on node AABBs
    Node cn = {
        .Transform = glm::transpose(*(glm::mat4*)&node->mTransformation),
        .BoundMin = glm::vec3(INFINITY),
        .BoundMax = glm::vec3(-INFINITY),
    };

    for (uint32_t i = 0; i < node->mNumMeshes; i++) {
        cn.Meshes.push_back(node->mMeshes[i]);

        const Mesh& mesh = model.Meshes[node->mMeshes[i]];
        cn.BoundMin = glm::min(cn.BoundMin, mesh.BoundMin);
        cn.BoundMax = glm::max(cn.BoundMax, mesh.BoundMax);
    }
    for (uint32_t i = 0; i < node->mNumChildren; i++) {
        Node childNode = ConvertNode(model, node->mChildren[i]);

        cn.BoundMin = glm::min(cn.BoundMin, childNode.BoundMin);
        cn.BoundMax = glm::max(cn.BoundMax, childNode.BoundMax);

        cn.Children.emplace_back(std::move(childNode));
    }
    return cn;
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
            .Texture = LoadTextures(*this, mat),
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

bool DepthPyramid::IsVisible(const Mesh& mesh, const glm::mat4& transform) const {
    if (!_storage) return true;

    glm::vec3 rectMin = glm::vec3(INFINITY), rectMax = glm::vec3(-INFINITY);

    uint8_t combinedOut = 63;
    uint8_t partialOut = 0;

    for (uint32_t i = 0; i < 8; i++) {
        glm::bvec3 corner = { (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1 };
        glm::vec4 p = _viewProj * transform * glm::vec4(glm::mix(mesh.BoundMin, mesh.BoundMax, corner), 1.0f);

        glm::vec3 rp = {
            p.x / p.w * 0.5f + 0.5f,
            p.y / p.w * 0.5f + 0.5f,
            p.z / p.w,
        };
        rectMin = glm::min(rectMin, rp);
        rectMax = glm::max(rectMax, rp);

        uint8_t outcode = 0;
        outcode |= p.x < -p.w ? 1 : 0;
        outcode |= p.x > +p.w ? 2 : 0;
        outcode |= p.y < -p.w ? 4 : 0;
        outcode |= p.y > +p.w ? 8 : 0;
        outcode |= p.z < 0 ? 16 : 0;
        outcode |= p.z > p.w ? 32 : 0;

        combinedOut &= outcode;
        partialOut |= outcode;
    }

    // Hacky frustum check. Cull if all vertices are outside any of the frustum planes.
    // Not that this still have false positives for big objects (see below), but it's good enough for our purposes.
    // - https://bruop.github.io/improved_frustum_culling/
    // - https://iquilezles.org/articles/frustumcorrect/
    if (combinedOut != 0) return false;

    // We don't do clipping, so the ccclusion test wont't work properly with
    // AABBs that are partially out the view frustum.
    // Consider them as visible to prevent flickering.
    if (partialOut != 0) return true;

    float sizeX = (rectMax.x - rectMin.x) * _width;
    float sizeY = (rectMax.y - rectMin.y) * _height;
    float lod = std::ceil(std::log2(std::max(sizeX, sizeY) / 2.0f));

    float screenDepth = GetDepth((rectMin.x + rectMax.x) * 0.5f, (rectMin.y + rectMax.y) * 0.5f, lod);

    //ImGui::GetForegroundDrawList()->AddRect(ImVec2(rectMin.x * _width * 2, (1 - rectMin.y) * _height * 2),
    //                                        ImVec2(rectMax.x * _width * 2, (1 - rectMax.y) * _height * 2),
    //                                        rectMin.z <= screenDepth ? 0x8000FF00 : 0x80FFFFFF);

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
    _storage = swr::alloc_buffer<float>(offset + 16);
}

}; // namespace scene