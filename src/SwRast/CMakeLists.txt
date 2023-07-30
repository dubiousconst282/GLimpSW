cmake_minimum_required(VERSION 3.25)

set(VCPKG_MANIFEST_MODE ON)
set(VCPKG_BUILD_TYPE "release")
set(CMAKE_BINARY_DIR "${CMAKE_CURRENT_LIST_DIR}/../../build")
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(SwRast VERSION 0.1.0)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 20)

add_executable(SwRast 
    Main.cpp
    Model.cpp
    
    Texture2D.cpp
    Rasterizer.cpp
    Clipper.cpp
    ImageHelpers.cpp
)

#set(CMAKE_FIND_DEBUG_MODE ON)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(SwRast PRIVATE
    imgui::imgui
    imguizmo::imguizmo
    glfw
    glad::glad
    assimp::assimp
    glm::glm
)
target_include_directories(SwRast PRIVATE ${Stb_INCLUDE_DIR})


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(SwRast PRIVATE "-march=native" "-ffast-math" "-Wall" "-Wno-unused")
else()
    target_compile_options(SwRast PRIVATE "/arch:AVX2") # MSVC seems to be pretty bad at inlining and SIMD
endif()