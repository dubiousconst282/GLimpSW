add_library(SwRast STATIC
    Rasterizer.cpp
    ImageHelpers.cpp
    Scene.cpp
)
target_compile_options(SwRast PUBLIC "-march=native" "-mprefer-vector-width=512" "-ffast-math" "-Wno-unused" "-Wsign-conversion")
target_compile_definitions(SwRast PUBLIC GLM_FORCE_XYZW_ONLY GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_link_libraries(SwRast PUBLIC
    glm::glm
    meshoptimizer
    TracyClient
)
target_include_directories(SwRast PUBLIC ${stb_SOURCE_DIR} ${cgltf_SOURCE_DIR})
# target_compile_options(SwRast PRIVATE "-fsanitize=address")
# target_link_options(SwRast PRIVATE  "-fsanitize=address")

add_executable(Playground
    Main.cpp
    Shading.cpp
    Voxel.cpp
    Yson.cpp
)
target_link_libraries(Playground PRIVATE
    SwRast
    glfw glad imgui
    nfd
)
target_precompile_headers(SwRast PUBLIC <memory> <algorithm> <thread> <immintrin.h> <glm/glm.hpp>)

add_executable(BenchmarkApp
    Benchmarks/RasterBench.cpp Shading.cpp Voxel.cpp
    # Benchmarks/CoarseRaster.cpp
    # Benchmarks/TexGather64.cpp
    # Benchmarks/TexSwizzle.cpp
    # Benchmarks/GatherSmall.cpp
    # Benchmarks/GatherMany.cpp
)
target_link_libraries(BenchmarkApp PRIVATE SwRast nanobench)
