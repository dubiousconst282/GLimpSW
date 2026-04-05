add_executable(SwRast 
    Main.cpp
    # Playground.cpp
    Scene.cpp
    
    Rasterizer.cpp
    ImageHelpers.cpp
)
target_compile_options(SwRast PRIVATE "-march=native" "-mprefer-vector-width=512" "-ffast-math" "-Wno-unused" "-Wsign-conversion")
target_compile_definitions(SwRast PRIVATE
    GLM_FORCE_XYZW_ONLY
)

target_link_libraries(SwRast PRIVATE
    imgui
    glfw
    glad
    glm::glm
    meshoptimizer
    TracyClient
)
target_include_directories(SwRast PRIVATE ${stb_SOURCE_DIR} ${cgltf_SOURCE_DIR})

add_executable(BenchmarkApp
    Benchmarks/RasterBench.cpp
    # Benchmarks/TexGather64.cpp
    # Benchmarks/GatherSmall.cpp

    Scene.cpp
    Rasterizer.cpp
    ImageHelpers.cpp
)
target_compile_options(BenchmarkApp PRIVATE "-march=native" "-mprefer-vector-width=512" "-ffast-math" "-Wno-unused" "-Wsign-conversion")
target_link_libraries(BenchmarkApp PRIVATE glm::glm nanobench meshoptimizer TracyClient)
target_include_directories(BenchmarkApp PRIVATE ${stb_SOURCE_DIR} ${cgltf_SOURCE_DIR})

