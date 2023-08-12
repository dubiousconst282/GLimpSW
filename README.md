# GLimpSW
Software rasterizer using AVX512 intrinsics

![swr_sponza](https://github.com/dubiousconst282/GLimpSW/assets/87553666/092a611a-d8e0-4f11-a986-d38a738e0357)
_Crytek Sponza rendered entirely on the CPU (i5-11320H), in real time_

---

## Features
- Programmable vertex/pixel shading via template specialization
  - Renderer currently implements diffuse + normal mapping + shadow mapping (lazy 16-sample PCF)
- SIMD-parallelized pipeline (vertex/pixel shading, triangle setup)
- Multi-threaded tile-based rasterizer (binning)
- Guard-band clipping
- Texture sampling: mip mapping, HDR cube mapping
- Hierarchical-Z occlusion

## Maybe TODOs
 - Deferred shadows, SSAO at 1/2 or 1/4 resolution
 - PBR
 - Improve multi-threading
   - Rasterizer currently only peaks at most ~1/3 of total CPU, likely due to the lack of pipelining between rasterization and triangle setup
