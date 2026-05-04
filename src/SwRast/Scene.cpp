#include "Scene.h"
#include "Rasterizer.h"

#include <glm/packing.hpp>
#include <glm/gtc/quaternion.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <meshoptimizer.h>
#include <stb_image.h>

#pragma clang diagnostic pop

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

static swr::StbImage LoadImage(const std::string& basePath, const cgltf_texture_view& view) {
    if (!view.texture || !view.texture->image) return {};

    const cgltf_image* imgInfo = view.texture->image;
    int width, height;
    uint8_t* pixels = nullptr;

    if (imgInfo->uri && strncmp(imgInfo->uri, "data:", 5) != 0) {
        std::string path = basePath + "/" + imgInfo->uri;
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

static swr::RgbaTexture2D::Ptr LoadTextures(const std::string& basePath, const cgltf_material* mat) {
    swr::StbImage baseColorImg = LoadImage(basePath, mat->pbr_metallic_roughness.base_color_texture);

    if (!baseColorImg.Width) {
        return swr::CreateTexture<swr::RgbaTexture2D>(4, 4, 1, 1);
    }
    swr::StbImage normalImg = LoadImage(basePath, mat->normal_texture);
    swr::StbImage metalRoughImg = LoadImage(basePath, mat->pbr_metallic_roughness.metallic_roughness_texture);
    swr::StbImage emissiveImg = LoadImage(basePath, mat->emissive_texture);

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

static Animation ParseAnimation(const cgltf_data* gltf, const cgltf_animation& srcAnim) {
    Animation anim;
    anim.NodeToChannelMap = std::vector<uint32_t>(gltf->nodes_count, UINT_MAX);

    for (uint32_t i = 0; i < srcAnim.channels_count; i++) {
        auto& channel = srcAnim.channels[i];
        auto& sampler = *channel.sampler;
        if (!channel.target_node) continue;

        int prop = channel.target_path == cgltf_animation_path_type_translation ? 0 :
                   channel.target_path == cgltf_animation_path_type_rotation    ? 1 :
                   channel.target_path == cgltf_animation_path_type_scale       ? 2 :
                                                                                  -1;
        if (prop < 0) continue;

        size_t offset = anim.KeyframeData.size();
        anim.KeyframeData.resize(offset + sampler.input->count + sampler.output->count * 4);

        float* timestamps = &anim.KeyframeData[offset];
        float* controlPoints = &anim.KeyframeData[offset + sampler.input->count];

        cgltf_accessor_unpack_floats(sampler.input, timestamps, sampler.input->count);

        if (sampler.output->type == cgltf_type_vec4) {
            cgltf_accessor_unpack_floats(sampler.output, controlPoints, sampler.output->count * 4);
        } else {
            assert(sampler.output->type == cgltf_type_vec3);
            for (uint32_t i = 0; i < sampler.output->count; i++) {
                cgltf_accessor_read_float(sampler.output, i, &controlPoints[i * 4], 3);
            }
        }

        anim.Duration = glm::max(anim.Duration, timestamps[sampler.input->count - 1]);

        uint32_t nodeIdx = cgltf_node_index(gltf, channel.target_node);
        uint32_t& channelIdx = anim.NodeToChannelMap[nodeIdx];
        if (channelIdx == UINT_MAX) {
            channelIdx = anim.Channels.size();
            anim.Channels.push_back({});
        }
        anim.Channels[channelIdx].SamplerTRS[prop] = {
            .FrameCount = (uint32_t)sampler.input->count,
            .DataOffset = (uint32_t)offset,
            .LerpMode = sampler.interpolation == cgltf_interpolation_type_linear ?
                            (channel.target_path == cgltf_animation_path_type_rotation ? Animation::kLerpSlerp : Animation::kLerpLinear) :
                        sampler.interpolation == cgltf_interpolation_type_cubic_spline ? Animation::kLerpCubic :
                                                                                         Animation::kLerpNearest,
        };
    }
    return anim;
}

Model* Scene::ImportGltf(const std::string& path) {
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
    std::string basePath = path.substr(0, path.find_last_of("/\\"));

    Model* model = Models.emplace_back(std::make_unique<Model>()).get();

    uint32_t materialOffset = Materials.AllocRange(gltf->materials_count);

    for (uint32_t i = 0; i < gltf->materials_count; i++) {
        cgltf_material* mat = &gltf->materials[i];
        Textures.emplace_back(LoadTextures(basePath, mat));

        Materials[materialOffset + i] = {
            .Texture = Textures[i].get(),
            .IsDoubleSided = mat->double_sided != 0,
            .AlphaCutoff = (uint8_t)(mat->alpha_mode == cgltf_alpha_mode_mask ? mat->alpha_cutoff * 255.0f + 0.5f : 255),
        };
    }

    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;

    std::vector<glm::uvec2> primMeshletRanges;
    
    for (uint32_t meshIdx = 0; meshIdx < gltf->meshes_count; meshIdx++) {
        cgltf_mesh* mesh = &gltf->meshes[meshIdx];

        uint32_t numMeshletsBound = 0;
        for (uint32_t primIdx = 0; primIdx < mesh->primitives_count; primIdx++) {
            cgltf_primitive* prim = &mesh->primitives[primIdx];
            numMeshletsBound += meshopt_buildMeshletsBound(prim->indices->count, swr::ShadedMeshlet::MaxVertices, swr::ShadedMeshlet::MaxPrims);
        }

        uint32_t startMeshletOffset = Meshlets.AllocRange(numMeshletsBound);
        uint32_t nextMeshletOffset = startMeshletOffset;

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

                Meshlet& om = Meshlets[nextMeshletOffset++];
                om.BoundCenter = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
                om.BoundRadius = bounds.radius;
                om.ConeApex = { bounds.cone_apex[0], bounds.cone_apex[1], bounds.cone_apex[2] };
                om.ConeAxis = { bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2] };
                om.ConeCutoff = bounds.cone_cutoff;

                om.NumVertices = m.vertex_count;
                om.NumTriangles = m.triangle_count;
                om.MaterialId = prim->material ? cgltf_material_index(gltf, prim->material) : UINT_MAX;
                om.AlphaCutoff = prim->material ? Materials[om.MaterialId].AlphaCutoff : 255;

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
                    if (normalAcc) {
                        glm::vec3 normal;
                        glm::vec4 tangent = glm::vec4(0);
                        cgltf_accessor_read_float(normalAcc, vertIdx, &normal.x, 3);
                        if (tangentAcc) cgltf_accessor_read_float(tangentAcc, vertIdx, &tangent.x, 4);

                        v_float2 normalOct = swr::texutil::MapOctahedron(normal);
                        v_float2 tangentOct = swr::texutil::MapOctahedron(glm::vec3(tangent));
                        om.NormalTangents[i] = glm::packUnorm4x8({ normalOct.x[0], normalOct.y[0], tangentOct.x[0], tangentOct.y[0] });
                        om.TangentHandedness |= (tangent.w < 0 ? (1ull << i) : 0);
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

        if (nextMeshletOffset < startMeshletOffset + numMeshletsBound) {
            Meshlets.FreeRange(nextMeshletOffset, startMeshletOffset + numMeshletsBound - nextMeshletOffset);
        }
        primMeshletRanges.push_back(glm::uvec2(startMeshletOffset, nextMeshletOffset));
    }

    // Convert nodes
    auto recurseNode = [&](auto& recurseNode, cgltf_node* srcNode, const float4x4& parentTransform) -> void {
        uint32_t nodeIdx = cgltf_node_index(gltf, srcNode);
        ModelNode& node = model->Nodes[nodeIdx];
        node.ParentIdx = srcNode->parent ? cgltf_node_index(gltf, srcNode->parent) : UINT_MAX;

        model->NodeIndicesPreDFS.push_back(nodeIdx);

        if (!srcNode->has_matrix) {
            node.LocalTRS = {
                .Translation = { srcNode->translation[0], srcNode->translation[1], srcNode->translation[2] },
                .Scale = { srcNode->scale[0], srcNode->scale[1], srcNode->scale[2] },
                .Rotation = { srcNode->rotation[0], srcNode->rotation[1], srcNode->rotation[2], srcNode->rotation[3] },
            };
        }
        float4x4 localTransform;
        cgltf_node_transform_local(srcNode, &localTransform[0][0]);
        node.LocalTransform = localTransform;
        node.GlobalTransform = parentTransform * localTransform;

        if (srcNode->mesh) {
            glm::uvec2 range = primMeshletRanges[cgltf_mesh_index(gltf, srcNode->mesh)];
            node.MeshOffset = range.x;
            node.MeshCount = range.y - range.x;
        }
        if (srcNode->skin) {
            // MaxJointMatrices += srcNode->skin->joints_count;

            node.Joints.resize(srcNode->skin->joints_count);
            for (uint32_t i = 0; i < srcNode->skin->joints_count; i++) {
                node.Joints[i] = cgltf_node_index(gltf, srcNode->skin->joints[i]);
            }

            if (auto* ibms = srcNode->skin->inverse_bind_matrices) {
                node.InverseBindMatrices.resize(ibms->count);

                for (uint32_t i = 0; i < ibms->count; i++) {
                    float4x4 m;
                    cgltf_accessor_read_float(ibms, i, &m[0][0], 16);
                    node.InverseBindMatrices[i] = glm::mat4x3(m);
                }
            }
        }

        if (srcNode->light) {
            cgltf_light* srcLight = srcNode->light;
            Light& dstLight = Lights[Lights.AllocRange(1)];

            dstLight.Type = srcLight->type == cgltf_light_type_directional ? Light::kTypeDirectional :
                            srcLight->type == cgltf_light_type_spot        ? Light::kTypeSpot :
                                                                             Light::kTypePoint;
            dstLight.Position = node.GlobalTransform[3];
            dstLight.Direction = glm::normalize(-node.GlobalTransform[2]);  // T * float3(0, 0, -1)

            dstLight.Color = { srcLight->color[0], srcLight->color[1], srcLight->color[2] };
            dstLight.Intensity = srcLight->intensity;
            dstLight.SetRadius(srcLight->range);
            dstLight.SetSpotAngles(srcLight->spot_inner_cone_angle, srcLight->spot_outer_cone_angle);
        }

        for (uint32_t i = 0; i < srcNode->children_count; i++) {
            recurseNode(recurseNode, srcNode->children[i], node.GlobalTransform);
        }
    };

    model->Nodes.resize(gltf->nodes_count);
    model->NodeIndicesPreDFS.reserve(gltf->nodes_count);

    auto& scene = gltf->scenes[0];
    for (uint32_t i = 0; i < scene.nodes_count; i++) {
        recurseNode(recurseNode, scene.nodes[i], float4x4(1));
    }

    for (uint32_t i = 0; i < gltf->animations_count; i++) {
        model->Animations.push_back(ParseAnimation(gltf, gltf->animations[i]));
    }
    return model;
}

Model::~Model() = default;

#if 0
void Model::UpdatePose(Animation* anim, double timestamp, std::span<float3x4> globalTransforms, std::span<float3x4> jointMatrices) {
    float cyclingTimestamp = fmod(timestamp, anim->Duration);

    for (uint32_t nodeIdx : NodeIndicesPreDFS) {
        ModelNode& node = Nodes[nodeIdx];
        float4x4 localTransform = float4x4(node.LocalTransform);

        if (anim->NodeToChannelMap[nodeIdx] != UINT_MAX) {
            TransformTRS localTRS = node.LocalTRS;  // copy
            anim->Interpolate(cyclingTimestamp, anim->NodeToChannelMap[nodeIdx], localTRS);
            localTransform = localTRS.ToMatrix();
        }
        float4x4 parentTransform = node.ParentIdx != UINT_MAX ? float4x4(Nodes[node.ParentIdx].GlobalTransform) : float4x4(1);
        globalTransforms[nodeIdx] = parentTransform * localTransform;

        if (node.Joints.size() > 0 && jointMatrices.size() >= node.Joints.size()) {
            float4x4 inverseTransform = InverseAffine(node.GlobalTransform);

            for (uint32_t i = 0; i < node.Joints.size(); i++) {
                float4x4 jointMatrix = inverseTransform * float4x4(Nodes[node.Joints[i]].GlobalTransform);

                if (i < node.InverseBindMatrices.size()) {
                    jointMatrix *= float4x4(node.InverseBindMatrices[i]);
                }
                jointMatrices[i] = TruncateMatrixCM34(jointMatrix);
            }
            jointMatrices.bump_slice(node.Joints.size());
        }
    }
    for (Light& light : Lights) {
        ModelNode& node = Nodes[light.ParentNodeIdx];
        light.Position = node.GlobalTransform[3];
        light.Direction = glm::normalize(-node.GlobalTransform[2]);  // T * float3(0, 0, -1)
    }
}
#endif

void Animation::Interpolate(float timestamp, uint32_t channelIdx, TransformTRS& transform) {
    Channel& ch = Channels[channelIdx];

    for (uint32_t j = 0; j < 3; j++) {
        Sampler& sampler = ch.SamplerTRS[j];
        if (sampler.FrameCount == 0) continue;

        const float* keyTimestamps = &KeyframeData[sampler.DataOffset];
        const float4* keyPoints = (float4*)&KeyframeData[sampler.DataOffset + sampler.FrameCount];

        // Advance or reset frame index based on current time
        uint32_t& currIdx = sampler.LastFrameIndex;
        if (timestamp < keyTimestamps[currIdx + 1]) currIdx = 0;
        while (currIdx + 1 < sampler.FrameCount - 1 && timestamp > keyTimestamps[currIdx + 1]) currIdx++;

        float prevTime = keyTimestamps[currIdx], nextTime = keyTimestamps[currIdx + 1];
        float td = nextTime - prevTime;
        float t = glm::clamp((timestamp - prevTime) / td, 0.0f, 1.0f);

        // Interpolate
        float4 res;

        if (sampler.LerpMode == kLerpLinear) {
            float4 p0 = keyPoints[currIdx], p1 = keyPoints[currIdx + 1];
            res = glm::mix(p0, p1, t);
        } else if (sampler.LerpMode == kLerpSlerp) {
            float4 p0 = keyPoints[currIdx], p1 = keyPoints[currIdx + 1];
            auto qr = glm::slerp(glm::quat::wxyz(p0.w, p0.x, p0.y, p0.z), glm::quat::wxyz(p1.w, p1.x, p1.y, p1.z), t);
            res = float4(qr.x, qr.y, qr.z, qr.w);
        } else if (sampler.LerpMode == kLerpNearest) {
            res = keyPoints[currIdx];
        } else if (sampler.LerpMode == kLerpCubic) {
            float4 v0 = keyPoints[(currIdx + 0) * 3 + 1];
            float4 b0 = keyPoints[(currIdx + 0) * 3 + 2];
            float4 a1 = keyPoints[(currIdx + 1) * 3 + 0];
            float4 v1 = keyPoints[(currIdx + 1) * 3 + 1];
            float t2 = t * t;
            float t3 = t2 * t;

            // clang-format off
            res = ( 2 * t3 - 3 * t2 + 1) * v0 + 
                  (     t3 - 2 * t2 + t) * (b0 * td) +
                  (-2 * t3 + 3 * t2    ) * v1 +
                  (     t3 -     t2    ) * (a1 * td);
            // clang-format on
        }

        // Store result into target node's transform
        if (j == 0) transform.Translation = float3(res);
        if (j == 1) transform.Rotation = glm::normalize(res);
        if (j == 2) transform.Scale = float3(res);
    }
}

float4x4 TransformTRS::ToMatrix() const {
    float4x4 M = glm::mat4_cast(glm::quat::wxyz(Rotation.w, Rotation.x, Rotation.y, Rotation.z));
    M[0] *= Scale.x;
    M[1] *= Scale.y;
    M[2] *= Scale.z;
    M[3] = float4(Translation, 1);
    return M;
}