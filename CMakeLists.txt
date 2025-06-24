cmake_minimum_required(VERSION 3.30)

project(SwRast)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 23)

# Setup CPM
include("deps/CPM.cmake")

if(NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_SOURCE_CACHE "${CMAKE_BINARY_DIR}/deps/")  # speeds up cmake configures
endif()

CPMAddPackage("gh:glfw/glfw#3.4")
CPMAddPackage("gh:g-truc/glm#1.0.1")
CPMAddPackage("gh:nothings/stb#master")

# Glad
add_library(glad STATIC deps/glad/src/glad.c)
target_include_directories(glad PUBLIC deps/glad/include/)

# ImGui
CPMAddPackage("gh:ocornut/imgui#v1.91.9b")
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
)
target_compile_definitions(imgui PUBLIC -DIMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_link_libraries(imgui PRIVATE glfw)

# ImGuizmo
CPMAddPackage("gh:CedricGuillemet/ImGuizmo#master")
add_library(ImGuizmo STATIC
    ${ImGuizmo_SOURCE_DIR}/ImGuizmo.cpp
)
target_include_directories(ImGuizmo PUBLIC 
    ${ImGuizmo_SOURCE_DIR}
)
target_link_libraries(ImGuizmo PUBLIC imgui)

add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
# Assimp
CPMAddPackage(
    NAME assimp
    GIT_REPOSITORY "https://github.com/assimp/assimp"
    GIT_TAG "v6.0.2"
    OPTIONS
        "ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT FALSE"
        "ASSIMP_BUILD_TESTS FALSE"
        "ASSIMP_NO_EXPORT TRUE"
        "ASSIMP_BUILD_GLTF_IMPORTER TRUE"
        "ASSIMP_BUILD_OBJ_IMPORTER TRUE"
)

# Projects
add_subdirectory("src/SwRast/")
