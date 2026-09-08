cmake_minimum_required(VERSION 3.30)

project(SwRast)
set(CMAKE_CXX_STANDARD 20)

# Setup CPM
include("deps/CPM.cmake")

if(NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_SOURCE_CACHE "${CMAKE_BINARY_DIR}/deps/")  # speeds up cmake configures
endif()

CPMAddPackage("gh:glfw/glfw#3.5.1")
CPMAddPackage("gh:g-truc/glm#1.0.3")
CPMAddPackage("gh:nothings/stb#master")

CPMAddPackage("gh:jkuhlmann/cgltf#v1.15")
CPMAddPackage("gh:zeux/meshoptimizer#v1.1")

CPMAddPackage("gh:martinus/nanobench#v4.3.11")

set(NFD_PORTAL ON)
CPMAddPackage("gh:btzy/nativefiledialog-extended#v1.3.0")

set(TRACY_ENABLE ON)
CPMAddPackage("gh:wolfpld/tracy#v0.14.1")

# Glad
add_library(glad STATIC deps/glad/src/glad.c)
target_include_directories(glad PUBLIC deps/glad/include/)

# ImGui
CPMAddPackage("gh:ocornut/imgui#v1.92.9b-docking")
# CPMAddPackage(GITHUB_REPOSITORY "CedricGuillemet/ImGuizmo" GIT_TAG "1.10" DOWNLOAD_ONLY TRUE)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    
    # ${ImGuizmo_SOURCE_DIR}/src/ImGuizmo.cpp
)
target_include_directories(imgui PUBLIC 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
    ${ImGuizmo_SOURCE_DIR}/src
)
target_compile_definitions(imgui PUBLIC -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_link_libraries(imgui PRIVATE glfw)

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

# Projects
add_subdirectory("src/SwRast/")
