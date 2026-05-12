# GLimpSW
High performance AVX512 software rasterizer

---

<div>
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4528415d-14b4-42fd-90d3-e1e47d2d9c17" width="49%" alt="Crytek Sponza">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4508d3a1-562e-4cb7-b43c-51a18f57e75b" width="49%" alt="Damaged Helmet">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/491bc38e-db84-408d-aa3f-2f2bf952d108" width="49%" alt="Polyhaven Ship Pinnace">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/0e39e425-102d-4f69-9802-41b654cd6aea" width="49%" alt="Amazon Bistro">
</div>

_Sample screenshots of the [previous version](https://github.com/dubiousconst282/GLimpSW/tree/old-deferred) (deferred g-buffer), rendered at 1080p on a 4-core TigerLake CPU, running at ~3.5GHz_

<div>
  <img width="49%" src="https://github.com/user-attachments/assets/8be6debd-9f88-452d-89c9-4880a6c7a3f6" />
  <img width="49%" src="https://github.com/user-attachments/assets/11c3aa4d-23dc-4f66-a2f2-6fd591ae41f6" />
</div>

_Sample screenshots of the current version (deferred vis-buffer), showing performance improvements over the same config_

## Features
- Programmable mesh and pixel shading via concepts and template specialization
- Fully SIMD-driven pipeline: mesh/pixel shading and internal processing all work on 16 elements in parallel, per thread
- Texture sampling: bilinear filtering, mip mapping, octahedron mapping, generic pixel formats
- Multi-threaded tiled rasterizer (binning)
- Guard-band clipping

### Missing/broken features
- Vis-buffer resolve pass does not handle models with more than one node properly (need to support per-node transform matrices, maybe difficult)
- Clipping not implemented in binned rasterizer (to be rewritten)
- Non-binned rasterizer does not synchronize, artifacts when multi-threading
- Occlusion culling uses prev frame's depth map + reprojection, does not account for newly visible objects (should be easy to fix but doesn't bother me)
- IBL, SSAO, TAA not yet ported to meshlet branch

## Implementation Notes
Some things I have and haven't tried.

### Raster Pipeline
Prior the rewrite, the raster pipeline was based around the traditional vertex shader model and was rather poorly optimized. All single-threaded except for bin rasterization and full-screen passes.

One major source of inefficiency was in the lack of shaded vertex reuse based on the index buffer, since finding and referencing duplicates in an efficient way did not seem feasible without fast conflict and gather instructions, therefore the vertex shader would always be invoked three times per triangle. Vertex shading in by itself was also rather expansive because it would spend a lot of time gathering and copying custom attributes, most of which would be immediately thrown away after clipping and culling (typically at least half of all triangles are culled in this early step).

Meshlets are very elegant and enable many interesting optimizations, in particular they allow for very flexible data layouts. 
It is trivial to [avoid gathers](#memory-gathers) in the mesh shader by using a SoA layout, but of course the rasterizer still needs to read the resulting indexed positions. However, it can take advantage of the small vertex limit (64) and use the amazing 2-input permute instructions instead, which runs over twice as fast as plain gathers.

Because I had planned to port the renderer to a vis-buffer, and considering the aforementioned inefficiency with custom attributes, they are not supported in the new pipeline. The fragment shader is only given perspective-correct barycentrics, which it can use to interpolate whatever attributes it needs after fetching them from source, as opposed to an intermediate "varying" buffer. Although that is ignoring costlier work for per-model transforms and skinning, which is maybe not a big deal since some games have a dedicated compute pass for skinning.

```cpp
// New pipeline. Produces up to 64 vertices + 128 triangles
void ShadeMeshlet(const ShadingContext& ctx, uint32_t index, swr::ShadedMeshlet& output) {
    const Meshlet& mesh = ctx.Meshlets[index];
    if (CullMeshlet(mesh)) { output.PrimCount = 0; return; }

    // Nice and tight loop, memory bound! (compression could help)
    for (uint32_t i = 0; i < mesh.NumVertices; i += simd::vec_width) {
        v_float3 pos = { simd::load(&mesh.Positions[0][i]), simd::load(&mesh.Positions[1][i]), simd::load(&mesh.Positions[2][i]) };
        output.SetPosition(i, simd::mul(ctx.ObjectToClipMat, v_float4(pos, 1.0f)));
    }
    output.PrimCount = mesh.NumTriangles;
    memcpy(output.Indices, mesh.Indices, sizeof(mesh.Indices));
}

struct Meshlet {
    uint8_t NumVertices, NumTriangles;
    uint32_t MaterialId;

    alignas(64) float Positions[3][64];
    alignas(64) uint8_t Indices[3][128];
    // ...
};

// Old pipeline. Invoked 3 times per 16 triangles (produces 16 vertices)
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

The rest of the pipeline is pretty traditional, with a few details I find interesting:

- Clipping
  - Needed to prevent overflow in fixed-point edge equations (using 32-bit integers and 4-bit subpixel precision, max resolution is 2892x2892 (31/2-4 bits))
  - Packets that are fully inside the viewport are most common due to frustum culling, and accepted immediately (very cheap to test, `max_abs(x,y,z) < w` using `vrangeps`)
  - Otherwise, Cohen-Sutherland outcodes are computed to determine non-trivial triangles. Usually <0.1% tris need actual clipping (guardband is incredibly effective)
  - Clipped triangles go through specialized `DrawTriangle<IsClipped=true>()` function that can adjust shader barycentrics to clipped factors (vertex attribs can't/don't need to be clipped)
- Binning
  - Currently wastes up to 1/3 of time syncing mesh/raster (would probably be better to split batches per-bin rather than per-frame).
  - Triangle setup is split in two steps to reduce size of binned batches
    - Early: Perspective div, fixed-point snapping, bounding box, back-face/degenerate culling
    - Late: Edge equations
  - Bins store packet IDs + masks rather than individual triangle IDs, for significantly lower overhead
  - Packets that are contained within a single bin are common (30~60%), slower path loops over entire packet's min/max bounding box (edge tests are overkill since bins are very large, 128x128)
  - One bin queue per thread (probably scales poorly), bitmap helps skipping over empty bins

### Quad Inefficiency
The rasterizer maps the 16 SIMD lanes into 4x4 pixel fragments. This works reasonably well, but suffers from the characteristic quad utilization problem, where small triangles will cover very few lanes.

Still, measuring render times on Bistro after skipping triangles of certain sizes shows that the timings are very much lead by pixel count: 75% of all tris are smaller than 12x12 pixels on either axes, and the overall average is ~28ns per small triangle and ~200ns for larger tris.

Fragment lane usage is indeed pretty bad even before the depth test, averaging at 11-14% for small, and 40-60% for large-ish tris. Looking at the floor brings it up to 80-95%. It's unclear whether rasterizing multiple triangles per SIMD would improve throughput considerably, due to the overhead in gathering and scattering framebuffer data while also handling conflicts.

Both overdraw and the quad utilization issue can be mitigated by moving complexity out of the fragment shader using a vis-buffer. On Sponza, the complete raster+resolve passes take around 0.8-1.2x as long as just the deferred raster pass [TODO: more detailed comparison].

### Coarse Raster
The rasterizer uses a basic bounding-box traversal to step over the triangle, which results in many redundant steps over empty space for larger triangles. These could be avoided with [one of the many smarter traversal algorithms](https://fgiesen.wordpress.com/2011/07/06/a-trip-through-the-graphics-pipeline-2011-part-6/), but some profiling shows traversal is also not as problematic as it seems in theory: on a view with a bunch of moderately large tris, only ~14% of time is spent on the edge stepping loop for the vis-buffer shader which writes only the depth and surface ID:

<div align="center"> 
  <img width="400" alt="coarse_profile_drawtri" src="https://github.com/user-attachments/assets/43e2b3a1-8812-4168-baa6-5b3d235e9231" />
</div>

Depth testing takes pretty much all drawing time with an awful high cache-miss rate, since at that point all framebuffer data has to come from RAM. Fast clears could help here, but unfortunately there seems to be no way to populate the cache without having the CPU fetch associdated data from RAM, even for aligned and unmasked 512-bit stores that completely overwrite cachelines. Although it might still be possible to save one memory trip by delaying and splitting framebuffer clears to the first bin rasterization step. [TODO]

The `tileMask != 0` branch also has a pretty high misprediction rate, which makes sense since edge intervals vary a lot across rows and triangles. The hierarchical traversal would probably help avoid this as well, since we could just loop over set bits in the coarse tile coverage mask. It would also be a good place to implement HiZ depth tests, but overdraw should be less of an issue with meshlet occlusion culling, so meh.

TODO: write about how old benchmark was subtly broken due to stupidity, new results, and impl difficulties (buffering, vpmulld). [CoarseRaster.cpp](./src/SwRast/Benchmarks/CoarseRaster.cpp).

### Memory Gathers
Memory gathers are fundamental for general SIMT-style programming, but they are not well accelerated on current x86 CPUs and always scalarized to one memory access micro-op per lane (plus other overhead), providing significantly lower throughput compared to sequential vector loads. In some cases, they can be avoided in place of more efficient in-register permutes.

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

For strided accesses multiple of 2 x 32-bit components, 64-bit gathers can be used in combination with permutes. This runs at nearly double throughput for the reason above. [GatherThroughput.cpp](./src/SwRast/Benchmarks/GatherThroughput.cpp) [TexGather64.cpp](./src/SwRast/Benchmarks/TexGather64.cpp)

```cpp
auto lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(idx, 0), ptr, 4);
auto hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32(idx, 1), ptr, 4);
auto x = _mm512_permutex2var_epi32(lo, lane_idx<v_int> * 2 + 0, hi);
auto y = _mm512_permutex2var_epi32(lo, lane_idx<v_int> * 2 + 1, hi);
```

---

When the Gather Data Sampling vulnerability mitigation is enabled on TigerLake, gather throughput is reduced by over 3x, which slows down the renderer noticeably. On Linux, it can be disabled by booting with kernel parameter `mitigations=off` or `gather_data_sampling=off`. [GatherThroughput.cpp](./src/SwRast/Benchmarks/GatherThroughput.cpp)

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
|               78.83 |       12,685,063.34 |    0.7% |          295.00 |          196.10 |  1.504 | `Gather64_AVX512`

### Texture Sampling
Mip-mapping improves sampling performance significantly thanks to better cache locality and reduced memory bandwidth. In the fragment shader, screen-space derivatives can be easily approximated through finite difference of SIMD lanes, exactly as done by GPUs.

For RGBA8 textures, bilinear interpolation is done in 16-bit fixed-point using `_mm512_mulhrs_epi16()`, operating on 2 channels at once. It costs about 2.5x more than nearest sampling, so the shader switches between bilinear for magnification and nearest+nearest mipmap for minification at the fragment level. This hybrid filtering turns to be quite effective because most samples fall on lower mips around distant objects, and the aliasing introduced by nearest sampling is relatively subtle, although the lack of trilinear filtering can be noticeable.

A micro-optimization for sampling multiple textures of the same size is to pack them into a single layered texture, which can be implemented essentially for free using a single offset. When sample() calls are inlined (many functions are marked as `always_inline` to avoid stack spills of vector registers), the compiler can [eliminate](https://en.wikipedia.org/wiki/Common_subexpression_elimination) duplicated UV setup for all of those calls (wrapping, scaling, rounding, and mip selection), since it can more easily prove that all layers have the same size. This is currently done for the BaseColor + NormalMap/MetallicRoughness + Emission textures, and saves about 3-5% from rasterization time [old numbers].

### Texture Swizzling
With the traditional row-major layout `x + y * stride`, texture accesses along the X and Y axis have dramatically different performance caracteristics due to poor cache locality. This is problematic for 3D rendering, since samples relative to object surfaces which are arbitrarily transformed by shear and rotations will often straddle across rows. GPUs have historically used swizzled/tiled layouts to improve spatial locality and minimize this issue.

In my experiments, tiling provided relatively small improvements outside of artificial throughput benchmarks. After mip-mapping, sampling appears to get bound primarily by compute, rather than memory latency. [TexSwizzle.cpp](./src/SwRast/Benchmarks/TexSwizzle.cpp)

<div align="center">
  <img width="40%" alt="heatmap_tiling_y8" src="https://github.com/user-attachments/assets/69764904-baa8-4a39-b36e-7067b11546b1" />
  <img width="40%" alt="heatmap_tiling_linear" src="https://github.com/user-attachments/assets/2991af76-9b49-4c49-9ea2-0d5552adc482" />
  <br><i>Plot of pixel exec timings</i>
</div>

Benchmark: Sampling visible fragments on Bistro (same camera view as in screenshot above; this is one of the more extreme cases, the difference for most other views is much lower)

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

The TiledY4 and TiledY8 layouts swizzle across 4x4 and 8x8 pixel tiles respectively. Tiled16X4 swizzles across two-levels, 16x16 and 4x4. The indexing formula has been simplified by placing pixels in column-major order, as described in [fatmap.txt](https://www.flipcode.com/documents/fatmap.txt): `(y & 7) | (x << 3) | ((y & ~7) << stride)` (3 cycles latency).

It is a bit surprising that 8x8 tiling outperforms 4x4, considering each 4x4 tile maps into exactly one cacheline. This appears to be largely related to the hardware prefetcher and TLB misses (perf stat reports ~36% fewer unused prefetches, ~32% fewer TLB misses). The wider 16x4 swizzle slightly reduces TLB and prefetch misses further, but increases "cycle_activity.stalls_mem_any" by 12% for some unknown reason.

Constant folding and other compiler optimizations may introduce redundant constants, which can be harmful since vector constants need to be loaded from memory. Ternlog is mostly opaque to Clang and can be used to avoid some of this, e.g. `y & 7` and `ternlog<A & ~C>(y, y, 7)` need only one constant slot.

## Building
The project uses CMake and CPM for build and dependency management. It depends on Clang's vector type extension and various related intrinsics.

```
cmake -S . -B ./build/ -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build ./build/

./build/src/SwRast/Playground
```

## Useful/interesting refs

Rasterization basics:
- [Optimizing Software Occlusion Culling](https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/)
- [A trip through the Graphics Pipeline](https://fgiesen.wordpress.com/2011/07/09/a-trip-through-the-graphics-pipeline-2011-index/)
- [Rasterising a Triangle - Interactive Tutorial](https://jtsorlinis.github.io/rendering-tutorial/)

Shading:
- [Physically Based Rendering in Filament](https://google.github.io/filament/Filament.html)
- [Image Based Lighting with Multiple Scattering](https://bruop.github.io/ibl/)
- https://academysoftwarefoundation.github.io/OpenPBR/

Meshlets:
- [Visibility Buffer Rendering with Material Graphs – Filmic Worlds](https://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/)
- [Modernizing Granite’s mesh rendering – Maister's Graphics Adventures](https://themaister.net/blog/2024/01/17/modernizing-granites-mesh-rendering/)
- [Basic Meshlet Compression | Liam's Graphics Blog](https://liamtyler.github.io/posts/meshlet_compression/)
- https://github.com/zeux/meshoptimizer?tab=readme-ov-file#mesh-shading
- https://github.com/zeux/niagara

Other:
- [Algorithms for Modern Hardware - Algorithmica](https://en.algorithmica.org/hpc/)
- [High-Performance Software Rasterization on GPUs](https://research.nvidia.com/publication/2011-08_high-performance-software-rasterization-gpus)
- [Cuda-Based Software Rasterization for Billions of Triangles](https://github.com/m-schuetz/CuRast)