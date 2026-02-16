add_executable(SwRast 
    Main.cpp
    Scene.cpp
    
    Rasterizer.cpp
    ImageHelpers.cpp
)
target_compile_options(SwRast PRIVATE "-march=native" "-ffast-math" "-Wno-unused" "-Wsign-conversion")
target_compile_definitions(SwRast PRIVATE GLM_FORCE_XYZW_ONLY)

target_link_libraries(SwRast PRIVATE
    imgui
    glfw
    glad
    glm::glm
    meshoptimizer
)
target_include_directories(SwRast PRIVATE ${stb_SOURCE_DIR} ${cgltf_SOURCE_DIR})

# std::execution::par_unseq
if(CMAKE_HOST_LINUX)
    target_link_libraries(SwRast PRIVATE tbb)
endif()

add_executable(BenchmarkApp
    Benchmarks/TexGather64.cpp
)
target_link_libraries(BenchmarkApp PRIVATE glm::glm nanobench)
target_compile_options(BenchmarkApp PRIVATE "-march=native" "-ffast-math" "-Wno-unused" "-Wsign-conversion")
