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
- Texture sampling: bilinear filtering, mip mapping, octahedron mapping, generic pixel formats
- Multi-threaded tiled rasterizer (binning)
- Guard-band clipping

## Implementation Notes
Some things I have and haven't tried.

### Raster Pipeline
Prior the rewrite, the raster pipeline was based around the traditional vertex shader model and was rather poorly optimized. All single-threaded except for bin rasterization and full-screen passes.

One major source of inefficiency was in the lack of shaded vertex reuse based on the index buffer, since finding and referencing duplicates in an efficient way did not seem feasible without fast conflict and gather instructions, therefore the vertex shader would always be invoked three times per triangle. Vertex shading by itself was also expansive due to the many memory gathers needed to fetch custom attributes, most of which would be immediately thrown away due to clipping and culling (typically at least half of all triangles are culled in this early step).

Enter meshlets. They enable many interesting optimizations at scale, and at a lower level they can be defined around a very SIMD-friendly data layout, speeding up early triangle processing by 4~6x compared to the old pipeline. Of course, the rasterizer still needs to read the indexed clip-space positions, but it can take advantage of the meshlet vertex limit and [avoid gathers](#memory-gathers) in place of the amazingly fast `vpermi2d` instructions.

Because I had planned to port the renderer to a vis-buffer, and considering the aforementioned inefficiency with custom attributes, they are not supported in the new pipeline. The fragment shader is only given perspective-correct barycentrics, which it can use to interpolate whatever attributes it needs after fetching them from source, as opposed to an intermediate "varying" buffer. This is ignoring costlier work for per-model transforms and other stuff like skinning, but that could perhaps be amended with some sort of "late triangle setup" shader if it were to be a problem.

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
- Early triangle setup
  - Perspective div, fixed-point snapping, bounding box
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
Mip-mapping has a quite significant impact on performance, speeding up sampling significantly thanks to better caching and reduced memory bandwidth. LOD selection requires screen-space derivatives, but these can be easily approximated through finite difference of SIMD lanes, requiring 2x shuffles per derivative.

For RGBA8 textures, bilinear interpolation is done in 16-bit fixed-point using `_mm512_mulhrs_epi16()`, operating on 2 channels at once. It still costs about 2.5x more than nearest sampling, so the shader switches between bilinear for magnification and nearest+nearest mipmap for minification at the fragment level. This hybrid filtering turns to be quite effective because samples typically fall on lower mips around distant objects, and the aliasing introduced by nearest sampling is relatively subtle due to the blurring introduced by mipping.

A micro-optimization for sampling multiple textures of the same size is to pack them into a single layered texture, which can be implemented essentially for free with a single offset. When sample() calls are inlined, the compiler can [eliminate](https://en.wikipedia.org/wiki/Common_subexpression_elimination) duplicated UV setup for all of those calls (wrapping, scaling, rounding, and mip selection), since it can more easily prove that all layers have the same size. This is currently done for the BaseColor + NormalMap/MetallicRoughness + Emission textures, saving about 3-5% from rasterization time.

### Memory Gathers
Memory gathers are fundamental for more general SIMT-style programming, but they are not well accelerated on current consumer-level x86 CPUs and always scalarized to one memory access micro-op per lane (plus other overhead), providing significantly lower throughput compared to sequential vector loads.

---

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

---

Bilinear filtering requires 4 gather instructions to needed to fetch neighboring pixels (P00, P10, P01, P11). For textures that are indexed linearly as `x + y * stride`, the two pixels neighboring across the X axis can be fetched at once for ~half the cost using 64-bit gather instructions, combined with 2-input permutes for unpacking. This improves raw throughput by ~1.3x on benchmarks, and by ~8% when sampling once in fragment shader. Unfortunately, this optimization cannot be applied to tiled layouts. [TexGather64.cpp](./src/SwRast/Benchmarks/TexGather64.cpp)

---

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
With the traditional row-major layout `x + y * stride`, texture accesses along the X and Y axis have dramatically different performance caracteristics due to poor cache locality. This is problematic for 3D rendering, since samples relative to object surfaces which are arbitrarily transformed by shear and rotations will often straddle across rows. GPUs have historically used swizzled/tiled layouts to improve spatial locality and minimize this issue.

In my experiments, tiling provided relatively small improvements outside of artificial throughput benchmarks. After mip-mapping, sampling appears to get bound primarily by compute, rather than memory latency. [TexSwizzle.cpp](./src/SwRast/Benchmarks/TexSwizzle.cpp)

<div align="center">
  <img src="heatmap_tiling_linear.jpg" width="40%"/>
  <img src="heatmap_tiling_y8.jpg" width="40%"/>
  <br><i>Plot of pixel exec timings</i>
</div>

Benchmark: Sampling visible fragments in Bistro (same camera view as in screenshot above; this is one of the more extreme cases, the difference for most other views is much lower)

| relative | us/frame | frame/s |    err% |       ins/frame |       cyc/frame |    IPC |  L1 cache refs |L1 miss% |  L3 cache refs |L3 miss% | benchmark
|---------:|---------:|--------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|---------------:|--------:|:----------
|   100.0% | 7,663.88 |  130.48 |    0.4% |    6,950,779.37 |    7,896,070.00 |  0.880 |  1,561,448.982 | 45.517% |    510,678.795 | 39.582% | `Linear`
|   109.6% | 6,994.87 |  142.96 |    0.2% |    7,108,171.15 |    7,130,771.74 |  0.997 |  1,525,567.214 | 30.913% |    412,985.321 | 44.510% | `TiledY4`
|   119.5% | 6,413.49 |  155.92 |    0.2% |    7,200,482.11 |    6,486,365.35 |  1.110 |  1,539,443.778 | 36.232% |    461,749.667 | 38.448% | `TiledY8`
|   110.4% | 6,942.70 |  144.04 |    0.2% |    7,651,839.90 |    7,107,611.88 |  1.077 |  1,557,348.420 | 35.117% |    448,449.236 | 38.333% | `Tiled16X4`
|   104.6% | 7,330.19 |  136.42 |    0.2% |    7,534,736.43 |    7,485,451.07 |  1.007 |  1,450,118.143 | 39.071% |    469,501.688 | 37.637% | `ZCurve`

Benchmark: Sampling a 4096x4096 texture, row by row, with some fixed rotation and scale (million samples/sec)

| Layout   | 0deg   | 45deg  | 90deg  |
| -------- | ----   | -----  | -----  |
| Linear   | 1364.59| 353.89 | 241.10 |
| TiledY4  |  802.55| 424.05 | 332.89 |
| TiledY8  |  668.19| 470.65 | 398.59 |
| Tiled16X4|  662.45| 448.10 | 451.42 |
| ZCurve   |  654.92| 407.36 | 469.11 |

The TiledY4 and TiledY8 layouts use 4x4 and 8x8 pixel tiles respectively. Tiled16X4 splits across two-levels, 16x16 and 4x4. The indexing formula has been simplified by placing pixels in column-major order, as described in [fatmap.txt](https://www.flipcode.com/documents/fatmap.txt): `(y & 7) | (x << 3) | ((y & ~7) << stride)` (3 cycles latency).

The [`vgf2p8affineqb`](https://gist.github.com/animetosho/d3ca95da2131b5813e16b5bb1b137ca0) instruction can be used to efficiently permute bits within bytes (replaces 2x shufb + 4x bitwise), it has been used to implement the Tiled16X4 and ZCurve layouts. Unfortunately, any additional latency between the path to memory appears to be harmful.

It is counterintuitive that 8x8 tiling outperforms 4x4, considering each 4x4 tile maps into exactly one cacheline. Perhaps this could be that since 4x4 samples in a SIMD group are fetched at once, there may be fewer of them straddling 8x8 tiles. Hardware details must also be a factor (cache associativity, prefetcher?), however the overall gains don't seem enough to warrant a deeper investigation. Larger tiles are inconvenient because they complicate bulk linear accesses (often need to shuffle data in SIMD groups). 

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

Rasterization basics:
- [Optimizing Software Occlusion Culling](https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [Rasterising a Triangle - Interactive Tutorial](https://jtsorlinis.github.io/rendering-tutorial/)

Shading:
- [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.html)
- [Image Based Lighting with Multiple Scattering](https://bruop.github.io/ibl/)

Other:
- https://themaister.net/blog/2024/01/17/modernizing-granites-mesh-rendering/
- https://liamtyler.github.io/posts/meshlet_compression/
- https://github.com/zeux/meshoptimizer?tab=readme-ov-file#mesh-shading