cmake_minimum_required(VERSION 3.14)
project(TinyQuant CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama/llama.cpp)

add_executable(model-inspection src/model-inspection.cpp)
target_link_libraries(model-inspection PRIVATE llama ggml)
target_include_directories(model-inspection PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/llama/llama.cpp
)
