add_executable(SwRast 
    Main.cpp
    Scene.cpp
    
    Rasterizer.cpp
    VertexReader.cpp
    Clipper.cpp
    ImageHelpers.cpp
)
target_compile_options(SwRast PRIVATE "-march=native" "-ffast-math" "-Wno-unused" "-Wsign-conversion")

target_link_libraries(SwRast PRIVATE
    imgui
    ImGuizmo
    glfw
    glad
    assimp
    glm::glm
    nanobench
)
target_include_directories(SwRast PRIVATE ${stb_SOURCE_DIR})

# std::execution::par_unseq
if(CMAKE_HOST_LINUX)
    target_link_libraries(SwRast PRIVATE tbb)
endif()