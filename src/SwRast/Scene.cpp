#include "Scene.h"
#include "Rasterizer.h"

#include <glm/packing.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <meshoptimizer.h>
#include <stb_image.h>

#pragma clang diagnostic pop

namespace scene {

static void CombineNormalMR(swr::StbImage& normalMap, const swr::StbImage& mrMap) {
    uint32_t count = normalMap.Width * normalMap.Height * 4;
    uint8_t* pixels = normalMap.Data.get();
    uint8_t* mrPixels = mrMap.Data.get();

    for (uint32_t i = 0; i < count; i += 4) {
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

static swr::StbImage LoadImage(Model& m, const cgltf_texture_view& view) {
    if (!view.texture || !view.texture->image) return {};

    const cgltf_image* imgInfo = view.texture->image;
    int width, height;
    uint8_t* pixels = nullptr;

    if (imgInfo->uri && strncmp(imgInfo->uri, "data:", 5) != 0) {
        std::string path = m.BasePath + "/" + imgInfo->uri;
        pixels = stbi_load(path.c_str(), &width, &height, nullptr, 4);
    } else if (imgInfo->buffer_view) {
        auto* data = (const uint8_t*)imgInfo->buffer_view->buffer->data;
        pixels = stbi_load_from_memory(&data[imgInfo->buffer_view->offset], imgInfo->buffer_view->size, &width, &height, nullptr, 4);
    }
    if (pixels == nullptr) {
        throw std::runtime_error("Failed to load image");
    }
    return {
        .Width = (uint32_t)width,
        .Height = (uint32_t)height,
        .Type = swr::StbImage::PixelType::RGBA_U8,
        .Data = { pixels, stbi_image_free },
    };
}

static swr::RgbaTexture2D::Ptr LoadTextures(Model& m, const cgltf_material* mat) {
    swr::StbImage baseColorImg = LoadImage(m, mat->pbr_metallic_roughness.base_color_texture);

    if (!baseColorImg.Width) {
        return swr::CreateTexture<swr::RgbaTexture2D>(4, 4, 1, 1);
    }
    swr::StbImage normalImg = LoadImage(m, mat->normal_texture);
    swr::StbImage metalRoughImg = LoadImage(m, mat->pbr_metallic_roughness.metallic_roughness_texture);
    swr::StbImage emissiveImg = LoadImage(m, mat->emissive_texture);

    bool hasNormals = normalImg.Width == baseColorImg.Width && normalImg.Height == baseColorImg.Height;
    bool hasEmissive = emissiveImg.Width == baseColorImg.Width && emissiveImg.Height == baseColorImg.Height;

    uint32_t numLayers = hasEmissive ? 3 : (hasNormals ? 2 : 1);
    auto tex = swr::CreateTexture<swr::RgbaTexture2D>(baseColorImg.Width, baseColorImg.Height, 8, numLayers);

    if (hasNormals) {
        CombineNormalMR(normalImg, metalRoughImg);
        tex->SetPixels(normalImg.Data.get(), normalImg.Width, 1);
    }
    if (hasEmissive) {
        InsertEmissiveMask(baseColorImg, emissiveImg);
        tex->SetPixels(emissiveImg.Data.get(), emissiveImg.Width, 2);
    }
    tex->SetPixels(baseColorImg.Data.get(), baseColorImg.Width, 0);
    tex->GenerateMips();
    return tex;
}

static uint32_t PackRGB10(const glm::vec3& value) {
    int ri = (int)(value.x * 1023.0f + 0.5f);
    int gi = (int)(value.y * 1023.0f + 0.5f);
    int bi = (int)(value.z * 1023.0f + 0.5f);

    ri = glm::min(glm::max(ri, 0), 1023);
    gi = glm::min(glm::max(gi, 0), 1023);
    bi = glm::min(glm::max(bi, 0), 1023);

    return (uint32_t)(ri << 22 | gi << 12 | bi << 2);
}
Model::Model(const std::string& path) {
    cgltf_options gltfOptions = {};
    cgltf_data* gltf = NULL;
    cgltf_result parseResult = cgltf_parse_file(&gltfOptions, path.c_str(), &gltf);
    if (parseResult != cgltf_result_success) {
        throw std::runtime_error("Failed to parse GLTF");
    }

    parseResult = cgltf_load_buffers(&gltfOptions, gltf, path.c_str());

    if (parseResult != cgltf_result_success) {
        throw std::runtime_error("Failed to load associated GLTF data");
    }
    BasePath = path.substr(0, path.find_last_of("/\\"));

    Textures.reserve(gltf->materials_count);

    for (uint32_t i = 0; i < gltf->materials_count; i++) {
        cgltf_material* mat = &gltf->materials[i];
        Textures.emplace_back(LoadTextures(*this, mat));

        Materials.push_back({
            .Texture = Textures[i].get(),
            .IsDoubleSided = mat->double_sided != 0,
            .AlphaCutoff = (uint8_t)(mat->alpha_mode == cgltf_alpha_mode_mask ? mat->alpha_cutoff * 255.0f + 0.5f : 255),
        });
    }

    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;

    std::vector<glm::uvec2> primMeshletRanges;

    for (uint32_t meshIdx = 0; meshIdx < gltf->meshes_count; meshIdx++) {
        cgltf_mesh* mesh = &gltf->meshes[meshIdx];
        uint32_t startMeshletOffset = Meshlets.size();

        for (uint32_t primIdx = 0; primIdx < mesh->primitives_count; primIdx++) {
            cgltf_primitive* prim = &mesh->primitives[primIdx];

            indices.resize(prim->indices->count);
            cgltf_accessor_unpack_indices(prim->indices, indices.data(), 4, prim->indices->count);

            const cgltf_accessor* positionsAcc = cgltf_find_accessor(prim, cgltf_attribute_type_position, 0);
            positions.resize(positionsAcc->count);
            cgltf_accessor_unpack_floats(positionsAcc, &positions[0].x, positionsAcc->count * 3);

            const float cone_weight = 0.25f;

            size_t maxMeshlets = meshopt_buildMeshletsBound(indices.size(), swr::ShadedMeshlet::MaxVertices, swr::ShadedMeshlet::MaxPrims);
            std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
            std::vector<uint32_t> meshletVertices(indices.size());
            std::vector<uint8_t> meshletTriangles(indices.size());

            size_t numMeshlets = meshopt_buildMeshlets(meshlets.data(), meshletVertices.data(), meshletTriangles.data(), indices.data(),
                                                       indices.size(), &positions[0].x, positions.size(), sizeof(glm::vec3),
                                                       swr::ShadedMeshlet::MaxVertices, swr::ShadedMeshlet::MaxPrims, cone_weight);

            const cgltf_accessor* texcoordAcc = cgltf_find_accessor(prim, cgltf_attribute_type_texcoord, 0);
            const cgltf_accessor* normalAcc = cgltf_find_accessor(prim, cgltf_attribute_type_normal, 0);
            const cgltf_accessor* tangentAcc = cgltf_find_accessor(prim, cgltf_attribute_type_tangent, 0);

            for (uint32_t meshIdx = 0; meshIdx < numMeshlets; meshIdx++) {
                meshopt_Meshlet& m = meshlets[meshIdx];
                meshopt_optimizeMeshlet(&meshletVertices[m.vertex_offset], &meshletTriangles[m.triangle_offset], m.triangle_count,
                                        m.vertex_count);

                meshopt_Bounds bounds =
                    meshopt_computeMeshletBounds(&meshletVertices[m.vertex_offset], &meshletTriangles[m.triangle_offset], m.triangle_count,
                                                 &positions[0].x, positions.size(), sizeof(glm::vec3));

                Meshlet& om = this->Meshlets.emplace_back();
                om.BoundSphere = glm::vec4(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius);
                om.ConeApex = { bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2] };
                om.ConeAxis = { bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2] };
                om.ConeCutoff = bounds.cone_cutoff;

                om.NumVertices = m.vertex_count;
                om.NumTriangles = m.triangle_count;
                om.MaterialId = prim->material ? cgltf_material_index(gltf, prim->material) : UINT_MAX;

                for (uint32_t i = 0; i < m.vertex_count; i++) {
                    uint32_t vertIdx = meshletVertices[m.vertex_offset + i];

                    glm::vec3 pos = positions[vertIdx];
                    om.Positions[0][i] = pos.x;
                    om.Positions[1][i] = pos.y;
                    om.Positions[2][i] = pos.z;

                    if (texcoordAcc) {
                        glm::vec2 texcoord;
                        cgltf_accessor_read_float(texcoordAcc, vertIdx, &texcoord.x, 2);
                        om.TexCoords[i] = glm::packHalf2x16(texcoord);
                    }
                    if (normalAcc && tangentAcc) {
                        glm::vec3 normal;
                        glm::vec4 tangent;
                        cgltf_accessor_read_float(texcoordAcc, vertIdx, &normal.x, 3);
                        cgltf_accessor_read_float(texcoordAcc, vertIdx, &tangent.x, 4);
                        om.Normals[i] = PackRGB10(normal * 0.5f + 0.5f);
                        om.Tangents[i] = PackRGB10(tangent * 0.5f + 0.5f) | (tangent.w < 0 ? 1 : 0);
                    }
                }
                for (uint32_t i = 0; i < m.triangle_count; i++) {
                    uint32_t j = m.triangle_offset + i * 3;
                    om.Indices[0][i] = meshletTriangles[j + 0];
                    om.Indices[1][i] = meshletTriangles[j + 1];
                    om.Indices[2][i] = meshletTriangles[j + 2];
                }
            }
        }
        primMeshletRanges.push_back(glm::uvec2(startMeshletOffset, Meshlets.size()));
    }

    auto recurseNode = [&](auto& recurseNode, cgltf_node* srcNode, const glm::mat4& parentTransform) -> Node {
        uint32_t nodeIdx = cgltf_node_index(gltf, srcNode);
        Node node = { };

        cgltf_node_transform_local(srcNode, &node.LocalTransform[0][0]);
        node.GlobalTransform = parentTransform * node.LocalTransform;

        if (srcNode->mesh) {
            glm::uvec2 range = primMeshletRanges[cgltf_mesh_index(gltf, srcNode->mesh)];
            node.MeshletOffset = range.x;
            node.MeshletCount = range.y - range.x;

            for (uint32_t i = range.x; i < range.y; i++) {
                Meshlet& m = Meshlets[i];
                for (uint32_t j = 0; j < m.NumVertices; j++) {
                    glm::vec3 p = { m.Positions[0][j], m.Positions[1][j], m.Positions[2][j] };
                    node.BoundMin = glm::min(node.BoundMin, p);
                    node.BoundMax = glm::max(node.BoundMax, p);
                }
            }
        }

        for (uint32_t i = 0; i < srcNode->children_count; i++) {
            Node child = recurseNode(recurseNode, srcNode->children[i], node.GlobalTransform);
            node.Children.emplace_back(std::move(child));
        }
        return node;
    };

    auto& scene = gltf->scenes[0];
    RootNode = { };

    for (uint32_t i = 0; i < scene.nodes_count; i++) {
        Node child = recurseNode(recurseNode, scene.nodes[i], RootNode.GlobalTransform);
        RootNode.BoundMin = glm::min(RootNode.BoundMin, child.BoundMin);
        RootNode.BoundMax = glm::max(RootNode.BoundMax, child.BoundMax);
        RootNode.Children.emplace_back(std::move(child));
    }
}

}; // namespace scene