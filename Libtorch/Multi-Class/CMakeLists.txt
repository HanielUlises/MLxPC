cmake_minimum_required(VERSION 3.18)
project(MultiClassFNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(multiclass_fnn main.cpp)

target_link_libraries(multiclass_fnn "${TORCH_LIBRARIES}")
target_include_directories(multiclass_fnn PRIVATE ${TORCH_INCLUDE_DIRS})

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET multiclass_fnn
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:multiclass_fnn>)
endif (MSVC)