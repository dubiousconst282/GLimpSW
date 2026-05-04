add_library(SwRast STATIC
    Rasterizer.cpp
    ImageHelpers.cpp
    Scene.cpp
)
target_compile_options(SwRast PUBLIC "-march=native" "-mprefer-vector-width=512" "-ffast-math" "-Wno-unused" "-Wsign-conversion")
target_compile_definitions(SwRast PUBLIC GLM_FORCE_XYZW_ONLY)
target_link_libraries(SwRast PUBLIC
    glm::glm
    meshoptimizer
    TracyClient
)
target_include_directories(SwRast PUBLIC ${stb_SOURCE_DIR} ${cgltf_SOURCE_DIR})
# target_compile_options(SwRast PRIVATE "-fsanitize=address")
# target_link_options(SwRast PRIVATE  "-fsanitize=address")

add_executable(Playground Main.cpp Shading.cpp)
target_link_libraries(Playground PRIVATE
    SwRast
    glfw glad imgui
    nfd
)
add_executable(BenchmarkApp
    Benchmarks/RasterBench.cpp
    # Benchmarks/CoarseRaster.cpp
    # Benchmarks/TexGather64.cpp
    # Benchmarks/GatherSmall.cpp
    # Benchmarks/TexSwizzle.cpp
)
target_link_libraries(BenchmarkApp PRIVATE SwRast nanobench)
