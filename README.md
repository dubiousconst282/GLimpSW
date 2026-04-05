# GLimpSW
High performance software rasterizer using AVX512

---
<!-- 
<div>
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4528415d-14b4-42fd-90d3-e1e47d2d9c17" width="49%" alt="Crytek Sponza">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4508d3a1-562e-4cb7-b43c-51a18f57e75b" width="49%" alt="Damaged Helmet">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/491bc38e-db84-408d-aa3f-2f2bf952d108" width="49%" alt="Polyhaven Ship Pinnace">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/0e39e425-102d-4f69-9802-41b654cd6aea" width="49%" alt="Amazon Bistro">
</div> -->

_Sample screenshots of the old pipeline, rendered at 1080p on a 4-core TigerLake CPU, running at ~3.5GHz_

## Features
- Programmable mesh and pixel shading via concepts and template specialization
  - Deferred rendering
  - PBR + Image Based Lighting
  - Shadow Mapping
  - Screen Space Ambient Occlusion at 1/2 resolution (with terrible results)
  - Temporal Anti-Aliasing
  - Hierarchical-Z occlusion
- Fully SIMD-driven pipeline: mesh/pixel shading and internal processing all work on 16 elements in parallel, per thread
- Texture sampling: bilinear filtering, mip mapping, seamless cube mapping, generic pixel formats
- Multi-threaded tiled rasterizer (binning)
- Guard-band clipping

## Implementation Notes
Some things I have and haven't tried.

### Raster Pipeline
Prior the rewrite, the raster pipeline was based around the traditional vertex shader model and was rather poorly optimized. All single-threaded except for bin rasterization.

One major source of inefficiency was in the lack of shaded vertex reuse based on the index buffer, since finding and referencing duplicates in an efficient way did not seem feasible without fast conflict and gather instructions, therefore the vertex shader would always be invoked three times per triangle. Vertex shading by itself was also expansive due to the many memory gathers needed to fetch custom attributes, most of which would be immediately thrown away due to clipping and culling (typically at least half of all triangles are culled in this early step).

Enter meshlets. They enable many interesting optimizations at scale, and at a lower level they can be defined around a very SIMD-friendly data layout, speeding up early triangle processing by 4~6x compared to the old pipeline. Of course, the rasterizer still needs to read the indexed clip-space positions, but it can take advantage of the meshlet vertex limit and [avoid gathers](#memory-gathers) in place of the amazingly fast `vpermi2d` instructions.

Because I had planned to port the renderer to a vis-buffer, and considering the aforementioned inefficiency with custom attributes, they are not supported in the new pipeline. The fragment shader is only given perspective-correct barycentrics, which it can use to interpolate whatever attributes it needs after fetching them from source, as opposed to an intermediate "varying" buffer. This is ignoring costlier work for per-model transforms, skinning, and other procedural stuff, but that could perhaps be amended with some sort of "late triangle setup" shader if it were to be a problem.

```cpp
// Produces up to 64 vertices + 128 triangles
void ShadeMeshlet(uint32_t index, swr::ShadedMeshlet& output) const {
    auto& mesh = Meshlets[index];
    if (CullMeshlet(mesh)) { output.PrimCount = 0; return; }

    // Nice and tight loop, memory bound! (compression could help)
    for (uint32_t i = 0; i < mesh.NumVertices; i += vec_width) {
        v_float3 worldPos = { load(&mesh.Positions[0][i]), load(&mesh.Positions[1][i]), load(&mesh.Positions[2][i]) };
        output.SetPosition(i, TransformVector(ProjMat, { worldPos, 1.0f }));
    }
    output.PrimCount = mesh.NumTriangles;
    memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));
}
struct Meshlet {
    uint16_t NumVertices, NumTriangles;
    uint32_t MaterialId;

    alignas(64) float Positions[3][64];
    alignas(64) uint8_t Indices[3][128];
    // ...
};

// The old pipeline. Invoked 3 times per 16 triangles (produces 16 vertices)
void ShadeVertices(const swr::VertexReader& data, swr::ShadedVertexPacket& vars) const {
    VFloat3 pos = data.ReadAttribs<VFloat3>(&scene::Vertex::x);       // 3 gathers
    vars.Position = TransformVector(ProjMat, { pos, 1.0f });

    vars.SetAttribs(0, data.ReadAttribs<VFloat2>(&scene::Vertex::u)); // 2 gathers

    VFloat3 norm = data.ReadAttribs<VFloat3>(&scene::Vertex::nx); // snorm8, 1 gather
    vars.SetAttribs(2, TransformNormal(ModelMat, norm));

    VFloat3 tang = data.ReadAttribs<VFloat3>(&scene::Vertex::tx); // snorm8, 1 gather
    vars.SetAttribs(5, TransformNormal(ModelMat, tang));
}
```

The rest of the pipeline is pretty standard, but I deem some details interesting enough for a summary:
- Clipping
  - Cohen-Sutherland outcodes are computed to determine trivial-accept and non-trivial masks
  - Usually <0.3% tris are non-trivial and need actual clipping (mainly to prevent overflow in fixed-point edge equations)
  - [TODO] Clipped triangles go through specialized "DrawTriangle<>" function that can adjust shader barycentrics to clipped factors (vertex attribs can't/don't need to be clipped!)
- Triangle setup
  - Perspective div, fixed-point snapping, bounding box, edge deltas at screen origin
  - Culling of triangles that are back-facing or covering no pixels
  - Edge equations are evaluated later, to avoid bloating binned triangle buffer
- Binning
  - Bins store packet IDs + masks rather than individual triangle IDs, for significantly lower overhead
  - Packets that are contained within a single bin are common (30~60%), slower path loops over entire packet's min/max bounding box
  - One bin queue per thread (might scale poorly over too many threads) 
  - Bitmap for fast skipping over empty bins (important due to limited batch size; per-thread bitmaps are merged once). 
- Rasterization
  - Once bin queues are full (a "batch" is ready), threads are synchronized to this stage [FIXME] this is bad design, stalls too much (~1/3 of time is waiting)
  - Late triangle setup: eval edge equations for incoming packets
  - [TODO] Different rasterizers for different triangle sizes
    - Medium: full BB scan, 4x4 fragments
    - Big (>16 pixels on both axes): 16x16 coarse pass + fine 4x4 pass
    - Small (<= 4x4 pixels on either axes, no wider than 16 pixels): 2x2 fragments (4 triangles per SIMD)
      - Requires specialized shader entrypoint
  - Draw order is not preserved due to laziness, but that's probably just a bit more coordination and bookkeeping. Not really necessary for most 3D renderering, except for proper ordering on alpha blending.

### Texture Sampling
Mip-mapping has a quite significant impact on performance, speeding up sampling by at least 2x thanks to better caching and reduced memory bandwidth. LOD selection requires screen-space derivatives, but these can be easily approximated through finite difference of SIMD lanes, requiring 2x shuffles per derivative.

For RGBA8 textures, bilinear interpolation is done in 16-bit fixed-point using `_mm512_mulhrs_epi16()`, operating on 2 channels at once. It still costs about 2.5x more than nearest sampling, so the shader switches between bilinear for magnification and nearest+nearest mipmap for minification at the fragment level. This hybrid filtering turns to be quite effective because most samples fall on lower mips for most camera angles, and the aliasing introduced by nearest sampling is relatively subtle.

A micro-optimization for sampling multiple textures of the same size is to pack them into a single layered texture, which can be implemented essentially for free with a single offset. If sample() calls are even partially inlined, the compiler can [eliminate](https://en.wikipedia.org/wiki/Common_subexpression_elimination) duplicated UV setup for all of those calls (wrapping, scaling, rounding, and mip selection), since it can more easily prove that all layers have the same size. This is currently done for the BaseColor + NormalMap/MetallicRoughness + Emission textures, saving about 3-5% from rasterization time.

Seamless cubemaps are relatively important as they will otherwise cause quite noticeable artifacts on high-roughness materials and pre-filtered environment maps. The current impl adjusts UVs and offsets near face edges to the nearest adjacent face when they are sampled using a [LUT](https://github.com/dubiousconst282/GLimpSW/blob/c1660c5bba70219798215898c8f744a1517be773/src/SwRast/Texture.h#L241-L246). An [easier and likely faster way](https://github.com/google/filament/blob/main/libs/ibl/src/Cubemap.cpp#L94) would be to pre-bake adjacent texels on the same face (or maybe some dummy memory location to avoid non-pow2 strides), so the sampling function could remain mostly unchanged. Alternatively, octahedron maps may prove to be far simpler to implement.

### Memory Gathers
Memory gathers are fundamental for more general SIMT-style programming, but they are not well accelerated on current consumer-level x86 CPUs and always scalarized to one memory access micro-op per lane (plus other overhead), providing significantly lower throughput compared to sequential vector loads.

For bilinear texture sampling, 4 gather instructions are needed to fetch the pixels neighboring each sample (P00, P10, P01, P11). Since we index textures linearly (`x + y * stride`), the two pixels neighboring across the X axis can be fetched at once for ~half the cost using 64-bit gather instructions, combined with 2-input permutes for unpacking. This improves raw throughput by ~1.3x on benchmarks, and ~8% when sampling once in fragment shader. [TexBench64.cpp](./src/SwRast/Benchmarks/TexGather64.cpp)

```cpp
    v_int row0_lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices00, 0), tex.Data.get(), 4);
    v_int row0_hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices00, 1), tex.Data.get(), 4);
    v_int row1_lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices01, 0), tex.Data.get(), 4);
    v_int row1_hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(indices01, 1), tex.Data.get(), 4);

    v_int data00 = _mm512_permutex2var_epi32(row0_lo, simd::lane_idx * 2 + 0, row0_hi);
    v_int data10 = _mm512_permutex2var_epi32(row0_lo, simd::lane_idx * 2 + 1, row0_hi);
    v_int data01 = _mm512_permutex2var_epi32(row1_lo, simd::lane_idx * 2 + 0, row1_hi);
    v_int data11 = _mm512_permutex2var_epi32(row1_lo, simd::lane_idx * 2 + 1, row1_hi);
```

For gathering values from a small array (64 elements), replacing gathers with loads + permutes provides a ~2.2x speedup. This is used to load clip-space positions output from the mesh shader. [GatherSmall.cpp](./src/SwRast/Benchmarks/GatherSmall.cpp)

```cpp
static v_float GatherPreload64(const float data[64], v_int idx) {
    v_float v0 = _mm512_load_ps(&data[0]);
    v_float v1 = _mm512_load_ps(&data[16]);
    v_float v2 = _mm512_load_ps(&data[32]);
    v_float v3 = _mm512_load_ps(&data[48]);
    v_float p01 = _mm512_permutex2var_ps(v0, idx, v1);
    v_float p23 = _mm512_permutex2var_ps(v2, idx, v3);
    return idx < 32 ? p01 : p23;
}
```

When the "Gather Data Sampling" vulnerability mitigation is enabled on TigerLake and older Intel CPUs, gather throughput is reduced by over 3x, which slows down the renderer noticeably. On Linux, it can be disabled by booting with kernel parameter `mitigations=off` or `gather_data_sampling=off`. [GatherThroughput.cpp](./src/SwRast/Benchmarks/GatherThroughput.cpp)

default
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|:----------
|              121.69 |        8,217,937.67 |    1.3% |        1,230.00 |          302.42 |  4.067 | `Load_Scalar_32`
|               18.70 |       53,467,269.08 |    0.7% |          171.00 |           46.36 |  3.688 | `Load_AVX2`
|               18.69 |       53,495,847.79 |    0.3% |           97.00 |           46.46 |  2.088 | `Load_AVX512`
|              410.49 |        2,436,102.25 |    0.6% |          294.00 |        1,018.32 |  0.289 | `Gather32_AVX512`

mitigations=off
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|:----------
|              120.67 |        8,287,178.49 |    0.6% |        1,230.00 |          299.41 |  4.108 | `Load_Scalar_32`
|               30.60 |       32,684,184.90 |    0.7% |          171.00 |           76.03 |  2.249 | `Load_AVX2`
|               18.66 |       53,602,141.29 |    0.3% |           97.00 |           46.34 |  2.093 | `Load_AVX512`
|              135.25 |        7,393,699.75 |    0.5% |          294.00 |          336.43 |  0.874 | `Gather32_AVX512`

### Texture Swizzling
GPUs typically arrange texture data in some way that improves spatial locality, since the traditional row-major layout is sub-optimal for texture accesses relative to 3D-mapped surfaces, as for example, samples that are transformed by shear and rotations will often straddle across multiple rows far apart in memory, despite being spatially close to each other.

In my experiments, tiling did not provide significant improvements outside of artificial benchmarks. After mip-mapping, sampling appears to get bound by compute more than memory latency. [TexSwizzle.cpp](./src/SwRast/Benchmarks/TexSwizzle.cpp)

Benchmark: Sampling a 4096x4096 texture row by row, with some rotation (million samples/sec). The prefetcher seems to have a big influence here, though there is some consistency with the other benchmark. Perhaps better analysis and understanding of the hardware behavior could help explain these somewhat unintuitive results.

| Layout   | 0deg   | 45deg  | 90deg  |
| -------- | ----   | -----  | -----  |
| Linear   | 1047.0 | 402.4  | 228.25 |
| TiledA   | 898.1  | 350.86 | 343.18 |
| TiledB   | 1001.2 | 374.85 | 369.54 |
| Z-Order  | 568.9  | 387.62 | 369.89 |

[TODO] Benchmark with randomized offsets/varying scales, convolutions

Benchmark: Bistro render (nearest only sampler, LQ textures @ 512x512)

| relative |            us/frame |    err% |       ins/frame |       cyc/frame |    IPC | benchmark
|---------:|--------------------:|--------:|----------------:|----------------:|-------:|:----------
|   100.0% |           79,052.68 |    0.5% |  313,581,110.50 |  185,763,012.50 |  1.688 | `Linear`
|    98.3% |           80,455.83 |    0.6% |  320,878,807.50 |  188,926,562.50 |  1.698 | `TiledA`
|    99.1% |           79,764.62 |    0.5% |  317,019,699.00 |  187,023,980.00 |  1.695 | `TiledB`
|    97.0% |           81,484.86 |    0.5% |  324,946,877.50 |  191,581,552.50 |  1.696 | `ZCurve`

### Coarse Rasterization
The basic rasterizer always traverses over the triangle's bounding box in fixed size steps (in this case, 4x4 pixels) to test whether any pixels are covered, before invoking the shader. A [coarse rasterizer](https://fgiesen.wordpress.com/2011/07/06/a-trip-through-the-graphics-pipeline-2011-part-6/) first tests bigger tiles to collect masks about which fragments are partially or fully covered, so it can skip through tiles of the bounding box that are outside the triangle much quicker.

My initial implementation was broken and didn't actually skip anything, so like the idiot I am got the wrong conclusion. [TODO: reimplement & retest]

Benchmark: skip rendering of triangles that are bigger/smaller than given amount (either/both XY axis of bounding-box).

|  frame/s |    err% |       ins/frame |       cyc/frame |bra miss%| benchmark
|---------:|--------:|----------------:|----------------:|--------:|:----------
|    33.43 |    0.4% |   99,981,906.00 |   70,600,855.00 |    6.4% | `Baseline`
|    65.52 |    0.6% |   53,917,656.67 |   36,031,948.33 |    1.8% | `either > 32`
|    60.29 |    0.5% |   51,462,850.33 |   39,057,720.00 |   10.7% | `both <= 32`
|    48.56 |    0.5% |   63,103,306.00 |   48,274,963.75 |   10.5% | `either <= 32`
|    52.93 |    0.7% |   57,930,090.00 |   44,236,053.33 |   10.5% | `either <= 16`


Other traversal approaches as suggested by Pineda and others (backtrack, zig-zag, middle-out) have shown to be difficult to implement correctly and efficiently. In another experiment, seeking to the start of a row and exiting after the first non-covered tile provided a ~10% speed up, but is not trivial as it first appears: simply checking for an empty tile coverage mask is not enough and causes occasional artifacts, since very thin triangles may produce gaps between tile rows; block corner thresholds are necessary for correctness.

### Visibility Buffer
[TODO1]

Vis-buffers do not require any complex shader operations and largely only write trivial parameters to the framebuffer, which should help reduce the problem of wasting SIMD lanes and register pressure.

## Building
The project uses CMake and CPM for build and dependency management. It depends on Clang's vector type extension and various related intrinsics.

```
cmake -S ./src/SwRast -B ./build/ -G Ninja -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe" -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang.exe"
cmake --build ./build/ --config RelWithDebInfo
```

## References (non-exhaustive)
- [Optimizing Software Occlusion Culling](https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [Rasterising a Triangle - Interactive Tutorial](https://jtsorlinis.github.io/rendering-tutorial/)
- [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.html)
- [Image Based Lighting with Multiple Scattering](https://bruop.github.io/ibl/)
