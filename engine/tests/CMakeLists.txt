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

add_executable(frameforge_engine_tests version_probe_test.cpp)
target_link_libraries(frameforge_engine_tests PRIVATE gtest_main frameforge_placeholder)

include(GoogleTest)
gtest_discover_tests(frameforge_engine_tests)
