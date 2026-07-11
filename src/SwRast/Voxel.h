#pragma once

#include "SIMD.h"
#include "Scene.h"

struct BrickCluster {
    uint64_t ClusterMask;       // Brick occupancy (one bit covers 16^3)
    uint64_t BrickMasks[64];    // Tile occupancy (one bit covers 4^3)
    uint16_t BrickOffsets[64];  // Offset to tile masks (per brick)
    uint16_t TileCount;         // Size of following array
    uint64_t TileMasks[];       // Voxel occupancy
    // uint16_t AttribOffsets[];// Offset to attrib data (per tile, 4-strided)
    // uint32_t AttribData[];
};
struct VoxelHitResult {
    v_float Dist;           // `-1` if miss
    v_uint Attribs;
};

struct BrickMap {
    static constexpr uint32_t VoxelGridSize = 512, ClusterGridSize = VoxelGridSize / 64;
    static constexpr uint32_t ClustersPerCascade = ClusterGridSize * ClusterGridSize * ClusterGridSize;

    uint32_t NumCascades = 0;
    float CascadeBaseScale;

    uint32_t NextOffset, MaxOffset;
    std::unique_ptr<uint32_t[]> Storage;

    BrickMap(uint32_t numCascades_, float baseScale_) {
        NumCascades = numCascades_;

        CascadeBaseScale = baseScale_;

        NextOffset = ClustersPerCascade * numCascades_;
        MaxOffset = 1024 * 1024 * 16;
        Storage = std::make_unique<uint32_t[]>(MaxOffset);
    }

    // Clear grid and voxelize scene.
    void VoxelizeScene(const Scene& scene, float3 centerPos);

    VoxelHitResult RayCast(v_float3 origin, v_float3 dir, v_float2 distRange, v_mask activeMask, bool fetchAttribs) const;

    static uint32_t GetClusterIndex(uint3 voxelPos) {
        return (voxelPos.x >> 6) + (voxelPos.z >> 6) * ClusterGridSize + (voxelPos.y >> 6) * (ClusterGridSize * ClusterGridSize);
    }

    BrickCluster* GetCluster(uint3 voxelPos, uint32_t cascadeIdx = 0) {
        uint32_t index = GetClusterIndex(voxelPos) + cascadeIdx * ClustersPerCascade;
        uint32_t offset = Storage[index];
        return (BrickCluster*)&Storage[offset];
    }

    BrickCluster* AllocCluster(uint3 pos, uint32_t sizeBytes);
    void FreeCluster(uint3 pos);
};
