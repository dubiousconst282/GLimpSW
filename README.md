# GLimpSW
Real time Physically Based Rendering on the CPU using AVX512

---

<div>
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4528415d-14b4-42fd-90d3-e1e47d2d9c17" width="49%" alt="Crytek Sponza">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4508d3a1-562e-4cb7-b43c-51a18f57e75b" width="49%" alt="Damaged Helmet">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/491bc38e-db84-408d-aa3f-2f2bf952d108" width="49%" alt="Polyhaven Ship Pinnace">
  <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/0e39e425-102d-4f69-9802-41b654cd6aea" width="49%" alt="Amazon Bistro">
</div>

_Sample scenes rendered at 1080p on a 4-core laptop CPU @ ~3.5GHz (i5-11320H)_

## Features
- Programmable vertex and pixel shading via concepts and template specialization
  - Deferred rendering
  - PBR + Image Based Lighting (or something close to it)
  - Shadow Mapping with rotated-disk sampling
  - Screen Space Ambient Occlusion at 1/2 resolution (with terrible results)
  - Temporal Anti-Aliasing
  - Hierarchical-Z occlusion
- Highly SIMD-parallelized pipeline: vertex/pixel shading and triangle setup all work on 16 elements in parallel per thread
- Texture sampling: bilinear filtering, mip mapping, seamless cube mapping, generic pixel formats
- Multi-threaded tiled rasterizer (binning)
- Guard-band clipping

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

### Related Projects
- https://github.com/rswinkle/PortableGL
- https://github.com/karltechno/SoftRast
- https://github.com/Zielon/CPURasterizer
- https://github.com/Mesa3D/mesa
- https://github.com/google/swiftshader

## Implementation Notes
A boring brain dump about a few intricacies I have and haven't tried.

### Texture Sampling
Mip-mapping has a quite significant impact on performance, making sampling at least 2x faster thanks to better caching and reduced memory bandwidth. LOD selection requires screen-space derivatives, but these can be easily approximated by subtracting two permutations (depending on the axis) of the scaled UVs packed in a SIMD fragment - it only takes 3 instructions per dFdx/dFdy call.

For RGBA8 textures, bilinear interpolation is done in 16-bit fixed-point using `_mm512_mulhrs_epi16()`, operating on 2 channels at once. It still costs about 2.5x more than nearest sampling, so the shader switches between bilinear for magnification and nearest+nearest mipmap for minification at the fragment level. This hybrid filtering turns to be quite effective because most samples fall on lower mips for most camera angles, and the aliasing introduced by nearest sampling is relatively subtle.

A micro-optimization for sampling multiple textures of the same size is to pack them into a single layered texture, which can be implemented essentially for free with a single offset. If sample() calls are even partially inlined, the compiler can [eliminate](https://en.wikipedia.org/wiki/Common_subexpression_elimination) duplicated UV setup for all of those calls (wrapping, scaling, rounding, and mip selection), since it can more easily prove that all layers have the same size. This is currently done for the BaseColor + NormalMap/MetallicRoughness + Emission textures, saving about 3-5% from rasterization time.

Seamless cubemaps are relatively important as they will otherwise cause quite noticeable artifacts on high-roughness materials and pre-filtered environment maps. The current impl adjusts UVs and offsets near face edges to the nearest adjacent face when they are sampled using a [LUT](https://github.com/dubiousconst282/GLimpSW/blob/c1660c5bba70219798215898c8f744a1517be773/src/SwRast/Texture.h#L241-L246). An [easier and likely faster way](https://github.com/google/filament/blob/main/libs/ibl/src/Cubemap.cpp#L94) would be to pre-bake adjacent texels on the same face (or maybe some dummy memory location to avoid non-pow2 strides), so the sampling function could remain mostly unchanged. Alternatively, octahedron maps may prove to be far simpler to implement.

### Memory Gathers
Current consumer-level x86 CPUs do not implement coalescing logic for AVX gathers, instead always scalarizing to 1uop per lane. There is also a small overhead compared to scalar loads, so enough computation must be involved to make up for it.

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
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|              121.69 |        8,217,937.67 |    1.3% |        1,230.00 |          302.42 |  4.067 |          66.00 |    1.5% |      1.64 | `Load_Scalar_32`
|               18.70 |       53,467,269.08 |    0.7% |          171.00 |           46.36 |  3.688 |          10.00 |    0.0% |      1.67 | `Load_AVX2`
|               18.69 |       53,495,847.79 |    0.3% |           97.00 |           46.46 |  2.088 |           6.00 |    0.0% |      1.65 | `Load_AVX512`
|              410.49 |        2,436,102.25 |    0.6% |          294.00 |        1,018.32 |  0.289 |           6.00 |    0.0% |      1.64 | `Gather32_AVX512`

mitigations=off
|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|              120.67 |        8,287,178.49 |    0.6% |        1,230.00 |          299.41 |  4.108 |          66.00 |    1.5% |      1.56 | `Load_Scalar_32`
|               30.60 |       32,684,184.90 |    0.7% |          171.00 |           76.03 |  2.249 |          10.00 |    0.0% |      1.66 | `Load_AVX2`
|               18.66 |       53,602,141.29 |    0.3% |           97.00 |           46.34 |  2.093 |           6.00 |    0.0% |      1.66 | `Load_AVX512`
|              135.25 |        7,393,699.75 |    0.5% |          294.00 |          336.43 |  0.874 |           6.00 |    0.0% |      1.66 | `Gather32_AVX512`

### Texture Swizzling
GPUs typically rearrange texture data in some way to improve memory spatial locality, so that nearby texels are always close to each other independently of transformations like rotations. In my limited experiments, this didn't seem to have a significant impact outside of artificial benchmarks. After mip-mapping, sampling seems to get bound by compute more than memory latency.

Below are the [benchmark](https://github.com/dubiousconst282/GLimpSW/commit/d0896e9788f73c2372b424d031d55741006f0f2b) results from scanning over a 2048x2048 texture in row-major order, after applying the specified rotation or zeroing UVs. With smaller textures (1024x1024) the difference is quite negligible, since the data will fit almost entirely in the L3 cache.

| Indexing    | Zero    | 0deg    | 45deg   | 90deg   |
| --------    | ----    | ----    | -----   | -----   |
| Linear      | 4.12 ms | 4.30 ms | 11.7 ms | 17.8 ms |
| Tiled 4x4   | 4.49 ms | 4.94 ms | 13.7 ms | 15.6 ms |
| Z-Order     | 4.55 ms | 5.96 ms | 11.0 ms | 12.3 ms |

### Coarse Rasterization
The basic rasterizer always traverses over the triangle's bounding box in fixed size steps (in this case, 4x4 pixels) to test whether any pixels are covered, before invoking the shader. A [coarse rasterizer](https://fgiesen.wordpress.com/2011/07/06/a-trip-through-the-graphics-pipeline-2011-part-6/) first test bigger tiles to collect masks about which fragments are partially or fully covered, so it can skip through tiles of the bounding box that are outside the triangle much quicker.

My [initial implementation](https://github.com/dubiousconst282/GLimpSW/blob/hierarchical-rast/src/SwRast/SwRast.h#L427) did not show a considerable improvement, likely because most triangles on even relatively simple scenes are very small. For Sponza at 1080p, over 90% triangles have bounding boxes smaller than 16x16 when viewed from the center.

| Rasterizer  | Time    |
| ----------  | ----    |
| Fine only   | 6.17 ms |
| Coarse      | 6.63 ms |

[TODO3] This may prove more sucessful when applied in isolation to bigger triangles.

Other approaches as suggested by Pineda and others (backtrack, zig-zag, middle-out) have shown to be difficult to implement correctly and efficiently. In another experiment, seeking to the start of a row and exiting after the first non-covered tile provided a ~10% speed up, but is not trivial as it first appears: simply checking for an empty tile coverage mask is not enough and causes occasional artifacts, since very thin triangles may produce gaps between tile rows; separate edge weights/bias may be necessary.

[TODO2] Additionally, it may be feasible to rasterize multiple small triangles at a time, by splitting a 16 vector to 4 triangles and processing 2x2 pixels at a time. Scattering results to the framebuffer may be problematic and require scalarized conflict resolution.

### Visibility Buffer
[TODO1]

Vis-buffers do not require any complex shader operations and largely only write trivial parameters to the framebuffer, which should help reduce the problem of wasting SIMD lanes and register pressure.

### Multi-threading
Use of multi-threading is currently fairly limited and only done for bin rasterization and full-screen passes, using the parallel loops from `std::for_each(par_unseq)`. This is far from optimal because it leads to stalls between vertex shading/triangle setup and rasterization, so the CPU is never fully busy.
