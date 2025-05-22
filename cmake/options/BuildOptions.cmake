if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wshadow -Wfloat-equal -Wcast-align)
endif()
