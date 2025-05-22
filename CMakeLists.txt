cmake_minimum_required(VERSION 3.18)
project(TinyPolicyPro VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MISRA-style warning flags
add_compile_options(
  -Wall -Wextra -Wpedantic -Werror
  -Wconversion -Wsign-conversion
  -fstrict-aliasing -Wstrict-aliasing=2
)

include(cmake/options/BuildOptions.cmake) # placeholder
add_subdirectory(runtime)
