include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
  URL_HASH SHA256=f179ec217f9b3b3f3c6e8b02d3e7eda997b49e4ce26d6b235c9053bec9c0bf9f
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Windows: force shared CRT to match the rest of the build; harmless elsewhere.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(frameforge_engine_tests
  version_probe_test.cpp
  tensor_test.cpp
  tonemap_test.cpp
  queue_test.cpp
  exr_io_test.cpp
  engine_e2e_test.cpp
  pipeline_test.cpp
  tile_test.cpp
)
target_link_libraries(frameforge_engine_tests PRIVATE gtest_main frameforge_core)
target_compile_definitions(frameforge_engine_tests PRIVATE
  FRAMEFORGE_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures"
  FRAMEFORGE_ENGINE_EXE="$<TARGET_FILE:frameforge>"
  FRAMEFORGE_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
add_dependencies(frameforge_engine_tests frameforge)

# Windows: the test binary also needs onnxruntime.dll next to it.
if(WIN32)
  add_custom_command(TARGET frameforge_engine_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:frameforge_engine_tests>"
  )
endif()

include(GoogleTest)
gtest_discover_tests(frameforge_engine_tests)
