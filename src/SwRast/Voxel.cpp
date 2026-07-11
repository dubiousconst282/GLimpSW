#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Rasterizer.h"
#include "Scene.h"
#include "Voxel.h"

#include <tracy/Tracy.hpp>

constexpr uint32_t VoxelizerChunkSize = 512;
constexpr uint32_t VoxelizerBrickGridSize = VoxelizerChunkSize / 16;

struct VoxelizerContext {
    const Meshlet* Meshlets;
    const Material* Materials;
    glm::mat4x3 ObjectToWorldMat;

    // Bump-allocated brickmap with three levels, two matching the compressed format (16^3 bricks + flat grid)
    uint32_t BrickNextOffset, BrickMaxOffset;
    std::unique_ptr<uint32_t[]> BrickStorage;

    void AllocStorage() {
        BrickNextOffset = VoxelizerBrickGridSize * VoxelizerBrickGridSize * VoxelizerBrickGridSize;
        BrickMaxOffset = 1024 * 1024 * 16;
        BrickStorage = std::make_unique<uint32_t[]>(BrickMaxOffset);
    }

    static uint32_t GetHierAddr(uint3 pos) {
        assert((pos.x | pos.y | pos.z) < (VoxelizerChunkSize / 4));

        return _pdep_u32(pos.x, 0b01'001'001'001'001'001'001'000011'000011) |  //
               _pdep_u32(pos.z, 0b10'010'010'010'010'010'010'001100'001100) |  //
               _pdep_u32(pos.y, 0b00'100'100'100'100'100'100'110000'110000);
    }

    [[clang::always_inline]]
    uint32_t FindOrAllocTile(uint3 pos) {
        if (BrickNextOffset + 128 >= BrickMaxOffset) [[unlikely]] {
            ExpandBrickStorage();
        }

        uint32_t addr = GetHierAddr(pos);
        uint32_t& brickOffset = BrickStorage[addr >> 6];
        if (brickOffset == 0) {
            brickOffset = BrickNextOffset;
            BrickNextOffset += 64;
        }

        uint32_t& tileOffset = BrickStorage[brickOffset + (addr & 63)];
        if (tileOffset == 0) {
            tileOffset = BrickNextOffset;
            BrickNextOffset += 64;
        }
        return tileOffset;
    }

    [[clang::noinline]]
    void ExpandBrickStorage() {
        if (BrickMaxOffset >= (UINT32_MAX / 2)) throw std::bad_alloc();

        BrickMaxOffset *= 2;
        auto newStorage = std::make_unique<uint32_t[]>(BrickMaxOffset);
        memcpy(newStorage.get(), BrickStorage.get(), BrickNextOffset * sizeof(uint32_t));
        BrickStorage = std::move(newStorage);
    }
};

static float2 UnpackHalf2x16(uint32_t x) {
    auto v = _mm_cvtph_ps(_mm_set1_epi32((int)x));
    return { v[0], v[1] };
}
static void UnpackNormal(v_uint packed, v_float3& norm) {
    v_float4 oct = swr::pixfmt::RGBA8u::Unpack(packed);
    norm = swr::texutil::UnmapOctahedron({ oct.x, oct.y });
}

static void MS_VoxelizeSurface(VoxelizerContext& ctx, uint32_t index, swr::ShadedMeshlet& output) {
    const Meshlet& mesh = ctx.Meshlets[index];

    for (uint32_t i = 0; i < mesh.NumVertices; i += simd::vec_width) {
        v_float3 pos = { simd::load(&mesh.Positions[0][i]), simd::load(&mesh.Positions[1][i]), simd::load(&mesh.Positions[2][i]) };
        output.SetPosition(i, simd::mul(ctx.ObjectToWorldMat, v_float4(pos, 1.0f)));
    }
    output.PrimCount = mesh.NumTriangles;
    memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));

    output.CullMode = swr::FaceCullMode::None;
    output.ReorientToMaxAxis = true;
}

static void FS_VoxelizeSurface(VoxelizerContext& ctx, swr::FragmentVars& vars) {
    const Meshlet& mesh = ctx.Meshlets[vars.MeshletId];
    const Material& material = ctx.Materials[mesh.MaterialId];

    constexpr swr::SamplerDesc SurfaceSampler = {
        .Wrap = swr::WrapMode::Repeat,
        .MagFilter = swr::FilterMode::Nearest,
        .MinFilter = swr::FilterMode::Nearest,
    };
    v_float2 uv0 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[0] & 63]);
    v_float2 uv1 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[1]]);
    v_float2 uv2 = UnpackHalf2x16(mesh.TexCoords[vars.VertexId[2]]);
    v_float2 uv = vars.Interpolate(uv0, uv1, uv2);
    v_uint baseColor = material.Texture->SampleImplicitLod<SurfaceSampler>(uv);

    vars.TileMask &= simd::movemask(baseColor >= uint32_t(material.AlphaCutoff << 24));
    if (vars.TileMask == 0) return;

    v_uint packedNMR = 0xFF00'8080;  // 100% roughness, norm(0,0,+1)

    if (material.Texture->NumLayers >= 2) {
        packedNMR = material.Texture->SampleImplicitLod<SurfaceSampler>(uv, 1);
    }

    v_float3 normal0, normal1, normal2;
    UnpackNormal(mesh.NormalTangents[vars.VertexId[0] & 63], normal0);
    UnpackNormal(mesh.NormalTangents[vars.VertexId[1]], normal1);
    UnpackNormal(mesh.NormalTangents[vars.VertexId[2]], normal2);
    v_float3 normalWS = simd::normalize(simd::mul(float3x3(ctx.ObjectToWorldMat), vars.Interpolate(normal0, normal1, normal2)));

    v_uint packedData = 0;

    // Color: RGB554
    packedData |= (baseColor >> 3 & 31) << 0;
    packedData |= (baseColor >> 11 & 31) << 5;
    packedData |= (baseColor >> 20 & 15) << 10;

    // Normal: Oct10
    v_int2 octNormal = simd::conv<int>(swr::texutil::MapOctahedron(normalWS) * 31.0f);
    packedData |= octNormal.x << 14;
    packedData |= octNormal.y << 19;

    // MetallicRoughness: 44
    packedData |= (packedNMR >> 20 & 15) << 24;
    packedData |= (packedNMR >> 28 & 15) << 28;

    uint32_t domAxis = vars.VertexId[0] >> 6;
    v_int3 fragPos = v_int3(int2(vars.TilePos) + swr::TilePixelOffsets, simd::conv<int>(vars.Depth * VoxelizerChunkSize));

    v_int3 gridPos = domAxis == 0 ? v_int3(fragPos.z, fragPos.x, fragPos.y) : // yzx
                     domAxis == 1 ? v_int3(fragPos.x, fragPos.z, fragPos.y) : // xzy
                                    v_int3(fragPos.x, fragPos.y, fragPos.z);  // xyz

    vars.TileMask &= simd::movemask(v_uint(gridPos.x | gridPos.y | gridPos.z) < VoxelizerChunkSize);

    v_uint tileIdx = v_uint((gridPos.x >> 2) | (gridPos.z >> 2) << 11 | (gridPos.y >> 2) << 22);
    v_uint voxelOffset = v_uint((gridPos.x & 3) | (gridPos.z & 3) << 2 | (gridPos.y & 3) << 4);

    // Volume overlaps per frag (Sponza 512³):
    // - 2x2x2: 2.504 (749.207K / 299.210K)
    // - 4x4x4: 1.027 (307.377K / 299.210K)
    // - 8x8x8: 1.013 (303.135K / 299.210K)
    simd_foreach_unique_lane(i, tileIdx, _mm512_movm_epi32(vars.TileMask)) {
        uint32_t tileOffset = ctx.FindOrAllocTile(i >> uint3(0, 22, 11) & uint3(2047, 1023, 2047));
        voxelOffset += v_exec ? tileOffset : 0;
    }
    _mm512_mask_i32scatter_epi32(ctx.BrickStorage.get(), vars.TileMask, voxelOffset, packedData, 4);
}

struct ClusterEncoder {
    uint64_t TileMasks[16 * 16 * 16];

    uint32_t AttribCount = 0;
    uint16_t AttribOffsets[16 * 16 * 16];
    uint32_t AttribData[64 * 64 * 64 + (simd::vec_width - 1)];

    BrickCluster Header = {};

    void StartBrick(uint32_t idx) {
        Header.ClusterMask |= 1ull << idx;
        Header.BrickOffsets[idx] = Header.TileCount;
    }
    void AddTile(uint32_t addr, uint32_t data[64]) {
        assert(AttribCount % 4 == 0 && AttribCount / 4 < UINT16_MAX);
        AttribOffsets[Header.TileCount] = AttribCount / 4;

        uint64_t tileMask = 0;

        for (uint32_t i = 0; i < 64; i += simd::vec_width) {
            v_uint chunkData = simd::load(&data[i]);
            v_mask chunkMask = simd::movemask(chunkData != 0);
            simd::store(&AttribData[AttribCount], _mm512_maskz_compress_epi32(chunkMask, chunkData));

            AttribCount += simd::popcnt(chunkMask);
            tileMask |= uint64_t(chunkMask) << i;
        }
        assert(tileMask != 0);

        TileMasks[Header.TileCount] = tileMask;
        Header.BrickMasks[addr >> 6] |= 1ull << (addr & 63);

        Header.TileCount++;
        AttribCount = (AttribCount + 3) & ~3u;
    }
    uint32_t GetFinalSize() {
        return sizeof(BrickCluster) +                               //
               sizeof(uint64_t) * Header.TileCount +                // Tile masks
               sizeof(uint16_t) * ((Header.TileCount + 1) & ~1u) +  // Attrib offses
               sizeof(uint32_t) * AttribCount;
    }
    void WriteTo(BrickCluster* dest) {
        memcpy(dest, &Header, sizeof(Header));
        memcpy(dest->TileMasks, TileMasks, Header.TileCount * sizeof(uint64_t));

        uint32_t* destAttribData = (uint32_t*)&dest->TileMasks[Header.TileCount];
        uint16_t* destAttribOffsets = (uint16_t*)destAttribData;
        destAttribData += ((Header.TileCount + 1) / 2);

        memcpy(destAttribOffsets, AttribOffsets, AttribCount * sizeof(uint16_t));
        memcpy(destAttribData, AttribData, AttribCount * sizeof(uint32_t));
    }
    void Reset() {
        Header = {};
        AttribCount = 0;
    }
};
static void CompressCluster(VoxelizerContext& src, BrickMap& dest, uint3 pos) {
    auto encoder = std::make_unique<ClusterEncoder>();

    for (uint32_t brickIdx = 0; brickIdx < 64; brickIdx++) {
        uint3 brickPos = brickIdx >> uint3(0, 4, 2) & 3u;
        uint32_t brickAddr = src.GetHierAddr(pos / 4u + brickPos * 4u) >> 6;
        uint32_t brickOffset = src.BrickStorage[brickAddr];
        if (brickOffset == 0) continue;

        encoder->StartBrick(brickIdx);

        for (uint32_t tileIdx = 0; tileIdx < 64; tileIdx++) {
            uint32_t tileOffset = src.BrickStorage[brickOffset + tileIdx];
            if (tileOffset == 0) continue;

            encoder->AddTile((brickIdx << 6) | tileIdx, &src.BrickStorage[tileOffset]);
        }
    }

    if (encoder->AttribCount != 0) {
        BrickCluster* cluster = dest.AllocCluster(pos, encoder->GetFinalSize());
        encoder->WriteTo(cluster);
        encoder->Reset();
    }
}

void BrickMap::VoxelizeScene(const Scene& scene, float3 centerPos) {
    auto raster = swr::Rasterizer(1);
    raster.EnableBinning = false;
    raster.SetViewport(uint2(VoxelizerChunkSize, VoxelizerChunkSize));

    // TODO: rolling/circular brickmaps and dynamic voxelization
    static_assert(VoxelizerChunkSize == VoxelGridSize);

    float scale = CascadeBaseScale;
    float3 minPos = centerPos - (VoxelGridSize / scale) * 0.5f;
    float3 maxPos = centerPos + (VoxelGridSize / scale) * 0.5f;
    float clipScale = 2.0f / (VoxelGridSize / scale);
    float3 clipOffset = -minPos * clipScale - 1.0f;
    float4x4 clipMatrix = glm::translate(glm::scale(float4x4(1), float3(clipScale)), clipOffset);

    auto shader = swr::GetDispatchTable<&MS_VoxelizeSurface, &FS_VoxelizeSurface>();

    VoxelizerContext ctx = { .Materials = scene.Materials.data() };
    ctx.AllocStorage();

    for (auto& model : scene.Models) {
        ZoneScopedN("Voxelize Model");

        for (auto& node : model->Nodes) {
            if (node.MeshCount == 0) continue;

            ctx.ObjectToWorldMat = clipMatrix * float4x4(node.GlobalTransform);
            ctx.Meshlets = &scene.Meshlets[node.MeshOffset];

            raster.DrawMeshlets(node.MeshCount, { shader, &ctx });
        }
    }

    {
        ZoneScopedN("Compress Brickmap");

        for (uint32_t by = 0; by < VoxelizerChunkSize; by += 64) {
            for (uint32_t bz = 0; bz < VoxelizerChunkSize; bz += 64) {
                for (uint32_t bx = 0; bx < VoxelizerChunkSize; bx += 64) {
                    CompressCluster(ctx, *this, uint3(bx, by, bz));
                }
            }
        }
    }
}

constexpr uint32_t kClusterGridBits = simd::tzcnt(BrickMap::VoxelGridSize / 64);
constexpr uint32_t kRootScale = 23;
constexpr uint32_t kClusterScale = 23 - kClusterGridBits;
constexpr uint32_t kBrickScale = kClusterScale - 2;
constexpr uint32_t kTileScale = kClusterScale - 4;
constexpr uint32_t kCellScale = kClusterScale - 6;

// Counts number of set bits before `index`
static uint32_t PrefixPopcnt(uint64_t mask, uint32_t index) {
    uint64_t prefix = (1ull << index) - 1u;
    return simd::popcnt(mask & prefix);
}
static v_uint PrefixPopcnt(v_uint2 mask, v_uint index) {
    v_uint lo_count = index >= 32 ? simd::popcnt(mask.x) : 0;
    v_uint hi_mask = index >= 32 ? mask.y : mask.x;
    hi_mask = _mm512_shrdv_epi32(v_uint(0), hi_mask, index);
    return lo_count + simd::popcnt(hi_mask);

    // v_uint prefix_mask = (1u << (index & 31u)) - 1u;
    // return lo_count + simd::popcnt(hi_mask & prefix_mask);
}

// Faster alternative to `(value >> offset & mask) != 0`
// Only equivalent if offset aligns with all bits in mask.
static v_int SplitBitTest64(v_uint2 value, v_uint offset, v_uint mask) {
    v_uint hi = offset >= 32 ? value.y : value.x;
    // hi >>= (offset & 31);
    hi = _mm512_shrdv_epi32(hi, hi, offset);
    return (hi & mask) != 0;
}

// Computes `dir > 0 ? 3.0 - pos : pos`
// with special case to avoid rounding.
static v_float3 GetMirroredPos(v_float3 pos, v_float3 dir, bool rangeCheck) {
    v_float3 mirrored = simd::as<v_float3>(simd::as<v_int3>(pos) ^ 0x7FFFFF);

    // XOR'ing will only work if the coords are in range [1.0, 2.0)
    if (rangeCheck) {
        simd::cmov(mirrored.x, 3.0 - pos.x, (pos.x < 1.0 || pos.x >= 2.0));
        simd::cmov(mirrored.y, 3.0 - pos.y, (pos.y < 1.0 || pos.y >= 2.0));
        simd::cmov(mirrored.z, 3.0 - pos.z, (pos.z < 1.0 || pos.z >= 2.0));
    }
    return {
        dir.x > 0 ? mirrored.x : pos.x,
        dir.y > 0 ? mirrored.y : pos.y,
        dir.z > 0 ? mirrored.z : pos.z,
    };
}
static v_uint GetNodeCellIndex(v_float3 pos, v_uint scaleExp) {
    v_uint3 cellPos = simd::as<v_uint3>(pos) >> scaleExp & 3;
    return cellPos.x | (cellPos.z << 2) | (cellPos.y << 4);
}

static v_uint GetBrickAddr(v_float3 pos) {
    v_uint addr = 0;
    addr |= GetNodeCellIndex(pos, kBrickScale) << 0;

    v_uint3 clusterPos = (simd::as<v_uint3>(pos) & 0x7FFFFF) >> kClusterScale;
    addr |= clusterPos.x << (kClusterGridBits * 0 + 6);
    addr |= clusterPos.y << (kClusterGridBits * 2 + 6);
    addr |= clusterPos.z << (kClusterGridBits * 1 + 6);

    return addr;
}

#define STEP_DBG(msg) // printf("%d %016b: %s\n", i, simd::movemask(v_exec), msg);

VoxelHitResult BrickMap::RayCast(v_float3 origin, v_float3 dir, v_float2 distRange, v_mask activeMask, bool fetchAttribs) const {
    // Mirror coordinates to negative ray octant
    v_float3 absDir = { simd::abs(dir.x), simd::abs(dir.y), simd::abs(dir.z) };
    v_float3 absInvDir = {
        1.0f / simd::max(absDir.x, 1e-8),
        1.0f / simd::max(absDir.y, 1e-8),
        1.0f / simd::max(absDir.z, 1e-8),
    };

    v_float3 mirrorOct = simd::as<v_float3>(simd::as<v_int3>(-dir) >> 31);
    v_uint mirrorMask = GetNodeCellIndex(mirrorOct, 0);
    v_uint brickMirrorMask = GetBrickAddr(mirrorOct);

    // TODO: unshittify coord scaling
    origin = origin * CascadeBaseScale + BrickMap::VoxelGridSize / 2.0f;
    origin = GetMirroredPos(origin / BrickMap::VoxelGridSize + 1.0, dir, true);

#if 0
    // Clip ray to be inside volume
    v_float3 exitDist = origin * absInvDir - absInvDir;  // (origin - 1.0) * absInvDir;
    v_float3 sideDist = exitDist - absInvDir;            // (origin - 2.0) * absInvDir;
    v_float tcurr = simd::max(simd::max(sideDist.x, sideDist.y), simd::max(sideDist.z, 1e-8f));
    v_float texit = simd::min(simd::min(exitDist.x, exitDist.y), exitDist.z);
    activeMask &= simd::movemask(tcurr < texit);
#else
    v_float3 sideDist = 0.0f;
    v_float tcurr = distRange.x * CascadeBaseScale / BrickMap::VoxelGridSize;
    v_float texit = distRange.y * CascadeBaseScale / BrickMap::VoxelGridSize;
#endif

    v_float3 pos = origin - absDir * tcurr;
    pos.x = simd::clamp(pos.x, 1.0f, 1.9999999f); // 0x3fffffff
    pos.y = simd::clamp(pos.y, 1.0f, 1.9999999f);
    pos.z = simd::clamp(pos.z, 1.0f, 1.9999999f);

    v_uint dirtyStepMask = 1u << kClusterScale;
    v_uint currScaleExp = kClusterScale;
    v_uint2 currLevelMask = 0;

    v_uint currBrickOffset = 0;
    v_uint2 currBrickMask = 0;

    for (uint32_t i = 0; i < 128; i++) {
        simd_if(simd::unmovemask(activeMask)) { STEP_DBG("Iter"); }

        simd_if(dirtyStepMask >= (1u << kBrickScale) && simd::unmovemask(activeMask)) {
            STEP_DBG("FetchBrick");

            // TODO: renormalize and continue on next cascade
            activeMask &= simd::movemask(tcurr < texit && dirtyStepMask < (1u << kRootScale));
            if (activeMask == 0) break;

            v_uint brickAddr = GetBrickAddr(pos) ^ brickMirrorMask;

            simd_foreach_unique_lane(addr, brickAddr, true) {
                uint32_t clusterOffset = Storage[addr >> 6];
                if (clusterOffset == 0) {
                    simd::cmov(currBrickOffset, 0, v_exec);
                    simd::cmov(currLevelMask, v_uint2(0), v_exec);
                    simd::cmov(currScaleExp, kClusterScale, v_exec);
                    continue;
                }

                auto* cluster = (BrickCluster*)&Storage[clusterOffset];
                uint32_t brickIdx = addr & 63;
                uint64_t brickMask = cluster->BrickMasks[brickIdx];
                uint32_t brickOffset = (uint32_t*)&cluster->TileMasks[cluster->BrickOffsets[brickIdx]] - Storage.get();

                if (brickMask != 0) {
                    simd::cmov(currBrickMask, simd::as<uint2>(brickMask), v_exec);
                    simd::cmov(currBrickOffset, brickOffset, v_exec);
                    simd::cmov(currScaleExp, kTileScale, v_exec);
                } else {
                    simd::cmov(currBrickOffset, 0, v_exec);
                    simd::cmov(currLevelMask, simd::as<uint2>(cluster->ClusterMask), v_exec);
                    simd::cmov(currScaleExp, kBrickScale, v_exec);
                }
            }
        }

        simd_if(dirtyStepMask >= (1u << kTileScale) && currBrickOffset != 0 && simd::unmovemask(activeMask)) {
            STEP_DBG("FetchTile");

            v_uint tileIdx = GetNodeCellIndex(pos, kTileScale) ^ mirrorMask;
            v_uint tileSlot = PrefixPopcnt(currBrickMask, tileIdx);
            v_int hasTile = SplitBitTest64(currBrickMask, tileIdx, 1);

#if 1
            v_uint2 tileMask = simd::gather2(Storage.get(), v_int(currBrickOffset + tileSlot * 2), v_exec && hasTile);

            simd::cmov(currLevelMask, simd::select(hasTile, tileMask, currBrickMask), v_exec);
#else
            v_uint tileOffset = currBrickOffset + tileSlot * 2;
            simd_foreach_unique_lane(i, tileOffset, hasTile) {
                simd::cmov(currLevelMask.x, Storage[(size_t)i + 0], v_exec);
                simd::cmov(currLevelMask.y, Storage[(size_t)i + 1], v_exec);
            }
            simd::cmov(currLevelMask, currBrickMask, v_exec && !hasTile);
#endif
            simd::cmov(currScaleExp, hasTile ? kCellScale : kTileScale, v_exec);
        }

        // TODO: try doing two/multiple steps to minimize divergence of tile/brick fetches

        v_uint advScaleExp = currScaleExp;
        simd_if((currLevelMask.x | currLevelMask.y) != 0) {
            v_uint cellIdx = GetNodeCellIndex(pos, advScaleExp) ^ mirrorMask;

            activeMask &= simd::movemask(advScaleExp > kCellScale || !SplitBitTest64(currLevelMask, cellIdx, 1));
            if (activeMask == 0) break;

            // Coalesce steps around 2³ empty cells
            advScaleExp += v_exec && !SplitBitTest64(currLevelMask, cellIdx & 0b101010, 0x00330033) ? 1u : 0;
        }

        // Compute next pos by intersecting with max cube edges
        v_uint3 edgePos = simd::as<v_uint3>(pos) & (~0u << advScaleExp);  // floor(pos / scale) * scale
        sideDist = (origin - simd::as<v_float3>(edgePos)) * absInvDir;

        simd::cmov(tcurr, simd::min(simd::min(sideDist.x, sideDist.y), sideDist.z), simd::unmovemask(activeMask));
        v_uint3 nextPos = simd::as<v_uint3>(origin - absDir * tcurr);

        // Clamp to next cell's bounding box
        v_uint advStepSize = (1 << advScaleExp) - 1;
        nextPos.x = simd::min(nextPos.x, edgePos.x + (sideDist.x == tcurr ? -1u : advStepSize));
        nextPos.y = simd::min(nextPos.y, edgePos.y + (sideDist.y == tcurr ? -1u : advStepSize));
        nextPos.z = simd::min(nextPos.z, edgePos.z + (sideDist.z == tcurr ? -1u : advStepSize));

        pos = simd::as<v_float3>(nextPos);

        // Bit diff tells which levels we have crossed and need to descend back from
        v_uint3 diffPos = nextPos ^ edgePos;
        dirtyStepMask = (diffPos.x | diffPos.y | diffPos.z);
    }

    v_int foundHit = currScaleExp == kCellScale && dirtyStepMask < (1u << kRootScale);
    VoxelHitResult hit = { .Dist = foundHit ? tcurr * BrickMap::VoxelGridSize / CascadeBaseScale : -1 };

    if (fetchAttribs) {
        v_uint cellAddr = GetBrickAddr(pos) ^ brickMirrorMask;
        cellAddr <<= 12;
        cellAddr |= (GetNodeCellIndex(pos, kTileScale) ^ mirrorMask) << 6;
        cellAddr |= (GetNodeCellIndex(pos, kCellScale) ^ mirrorMask) << 0;

        simd_foreach_unique_lane(addr, cellAddr, foundHit) {
            auto* cluster = (BrickCluster*)&Storage[Storage[addr >> 18]];
            uint32_t brickIdx = addr >> 12 & 63;
            uint32_t tileIdx = addr >> 6 & 63;

            uint32_t tileSlot = cluster->BrickOffsets[brickIdx];
            tileSlot += PrefixPopcnt(cluster->BrickMasks[brickIdx], tileIdx);

            const uint32_t* attribData = (uint32_t*)&cluster->TileMasks[cluster->TileCount];
            const uint16_t* attribOffsets = (uint16_t*)attribData;
            attribData += ((cluster->TileCount + 1) / 2) + attribOffsets[tileSlot] * 4;

            simd::cmov(hit.Attribs, attribData[PrefixPopcnt(cluster->TileMasks[tileSlot], addr & 63)], v_exec);
        }
    }
    return hit;
}

BrickCluster* BrickMap::AllocCluster(uint3 pos, uint32_t sizeBytes) {
    assert((pos.x | pos.y | pos.z) < VoxelGridSize);
    assert((pos.x | pos.y | pos.z) % 64 == 0);
    assert(Storage[GetClusterIndex(pos)] == 0);

    uint32_t sizeWords = ((sizeBytes + 7) & ~7u) / sizeof(uint32_t);

    if (NextOffset + sizeWords >= MaxOffset) [[unlikely]] {
        if (MaxOffset >= (UINT32_MAX / 2)) throw std::bad_alloc();

        MaxOffset *= 2;
        auto newStorage = std::make_unique<uint32_t[]>(MaxOffset);
        memcpy(newStorage.get(), Storage.get(), NextOffset);
        Storage = std::move(newStorage);
    }

    uint32_t offset = NextOffset;
    NextOffset += sizeWords;

    Storage[GetClusterIndex(pos)] = offset;
    return (BrickCluster*)&Storage[offset];
}
void BrickMap::FreeCluster(uint3 pos) {
    assert(!"TODO");
}