# GLimpSW
Real time Physically Based Rendering on the CPU using AVX512

---

<table>
  <tr>
    <td>
      <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4528415d-14b4-42fd-90d3-e1e47d2d9c17" alt="Demo - Crytek Sponza" style="height: 100%; object-fit: scale-down;">
    </td>
    <td>
      <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/4508d3a1-562e-4cb7-b43c-51a18f57e75b" alt="Demo - Damaged Helmet" style="height: 100%; object-fit: scale-down;">
    </td>
  <tr>
  <tr>
    <td>
      <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/491bc38e-db84-408d-aa3f-2f2bf952d108" alt="Demo - Polyhaven Ship Pinnace" style="height: 100%; object-fit: scale-down;">
    </td>
    <td>
      <img src="https://github.com/dubiousconst282/GLimpSW/assets/87553666/0e39e425-102d-4f69-9802-41b654cd6aea" alt="Demo - Amazon Bistro" style="height: 100%; object-fit: scale-down;">
    </td>
  <tr>
</table>

## Features
- Programmable vertex and pixel shading via concepts and template specialization
  - Deferred rendering
  - PBR + Image Based Lighting (or something close to it)
  - Shadow Mapping with Poisson sampling
  - Screen Space Ambient Occlusion at 1/2 resolution (with terrible results)
  - Temporal Anti-Aliasing
  - Hierarchical-Z occlusion
- Highly SIMD-parallelized pipeline: vertex/pixel shading and triangle setup all work on 16 elements in parallel per thread
- Texture sampling: bilinear filtering, mip mapping, seamless cube mapping, generic pixel formats
- Multi-threaded tiled rasterizer (binning)
- Guard-band clipping

_Most of the effects aren't quite correct nor very well tuned, since this is a toy project and my focus was more on performance and simplicity than final quality._

## Building
The project uses CMake and vcpkg for project and dependency management. Clang or GCC must be used for building, as MSVC builds were initially about 2x slower, and there's now some `__builtin_*` stuff being used. Debug builds are also too slow for most of anything, so release/optimized builds are preferred except for heavy debugging.

- Ensure `VCPKG_ROOT` environment variable is set to vcpkg root directory. (and pray that CMake picks it up properly on the first try.)
- Open project on VS or VSC or wherever, or build through CLI:

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

## Appendix: Implementation Notes
A boring brain dump about a few intricacies I have and haven't tried.

### Texture Sampling
Mip-mapping has a quite significant impact on performance, making sampling at least 2x faster thanks to better caching and reduced memory bandwidth. LOD selection requires screen-space derivatives, but these can be easily approximated by subtracting two permutations (depending on the axis) of the scaled UVs packed in a SIMD fragment - it only takes 3 instructions per dFdx/dFdy call.

For RGBA8 textures, bilinear interpolation is done in 16-bit fixed-point using `_mm512_mulhrs_epi16()`, operating on 2 channels at once. It still costs about 2.5x more than nearest sampling, so the shader switches between bilinear for magnification and nearest+nearest mipmap for minification at the fragment level. This hybrid filtering turns to be quite effective because most samples fall on lower mips for most camera angles, and the aliasing introduced by nearest sampling is relatively subtle.

A micro-optimization for sampling multiple textures of the same size is to pack them into a single layered texture, which can be implemented essentially for free with a single offset. If sample() calls are even partially inlined, the compiler can [eliminate](https://en.wikipedia.org/wiki/Common_subexpression_elimination) duplicated UV setup for all of those calls (wrapping, scaling, rounding, and mip selection), since it can more easily prove that all layers have the same size. This is currently done for the BaseColor + NormalMap/MetallicRoughness + Emission textures, saving about 3-5% from rasterization time.

Seamless cubemaps are relatively important as they will otherwise cause quite noticeable artifacts on high-roughness materials and pre-filtered environment maps. The current impl adjusts UVs and offsets near face edges to the nearest adjacent face when they are sampled using a [LUT](https://github.com/dubiousconst282/GLimpSW/blob/c1660c5bba70219798215898c8f744a1517be773/src/SwRast/Texture.h#L241-L246). An [easier and likely faster way](https://github.com/google/filament/blob/main/libs/ibl/src/Cubemap.cpp#L94) would be to pre-bake adjacent texels on the same face (or maybe some dummy memory location to avoid non-pow2 strides), so the sampling function could remain mostly unchanged. (But I just don't want to think about cubemaps again any time soon.)

### Texture Swizzling
GPUs typically rearrange texture data in some way to improve memory spatial locality, so that nearby texels are always close to each other independent of transformations like rotations. In my limited experiments, this didn't seem to have a significant impact outside of artificial benchmarks. After mip-mapping, sampling seems to get bound by compute more than memory latency.

Below are the [benchmark](https://github.com/dubiousconst282/GLimpSW/commit/d0896e9788f73c2372b424d031d55741006f0f2b) results from scanning over a 2048x2048 texture in row-major order, after applying the specified rotation or zeroing UVs. With smaller textures (1024x1024) the difference is quite negligible, since the data will fit almost entirely in the L3 cache.

| Indexing    | Zero    | 0deg    | 45deg   | 90deg   |
| --------    | ----    | ----    | -----   | -----   |
| Linear      | 4.12 ms | 4.30 ms | 11.7 ms | 17.8 ms |
| Tiled 4x4   | 4.49 ms | 4.94 ms | 13.7 ms | 15.6 ms |
| Z-Order     | 4.55 ms | 5.96 ms | 11.0 ms | 12.3 ms |

(By the way, for the Z-order encoding function, it may be slightly faster to use the Galois field instructions instead of [SIMD LUTs](https://github.com/dubiousconst282/GLimpSW/blob/6f0c6b32e6e681469628e5d6d2a4d844b696be9c/src/SwRast/Texture.h#L274-L293), see [this post](https://news.ycombinator.com/item?id=37630391).)

### Coarse Rasterization
The basic rasterizer always traverses over fixed size tiles (in this case, 4x4 pixels) to test whether any pixels are covered or not, before doing the depth test and invoking the shader. A [coarse rasterizer](https://fgiesen.wordpress.com/2011/07/06/a-trip-through-the-graphics-pipeline-2011-part-6/) first rasterizes bigger tiles and collect masks about which fragments are partially or fully covered, so it can skip through tiles of the bounding box that are outside the triangle much quicker.

That sounds neat in theory, but the benchmark results for my [experimental implementation](https://github.com/dubiousconst282/GLimpSW/blob/hierarchical-rast/src/SwRast/SwRast.h#L427) weren't very promising, though:

| Rasterizer  | Time    |
| ----------  | ----    |
| Fine only   | 6.17 ms |
| Coarse      | 6.63 ms |

Ultimately, this isn't that surprising considering that the basic rasterizer can skip non-covered fragments in just about 5 cycles or so, and most triangles on even relatively simple scenes are very small. For Sponza at 1080p, more than 90% triangles have bounding boxes smaller than 16x16 when viewed from the center. Setup cost and other overhead dominates at least a third of processing time, so it might make sense to use smaller SIMD fragments, or even dynamically switch between them depending on triangle size (with maybe some preprocessor/template magic).

### Multi-threading
Use of multi-threading is currently fairly limited and only done for bin rasterization and full-screen passes, using the parallel loops from `std::for_each(par_unseq)`. This is far from optimal because it leads to stalls between vertex shading/triangle setup and rasterization, so the CPU is never fully busy. It could probably be improved to some extent without complicating state and memory management too much, but threading is hard... Maybe OpenMP would be nice for this.