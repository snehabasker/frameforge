cmake_minimum_required(VERSION 3.20)
# C is needed only for tinyexr's vendored miniz.c dependency (deps/miniz/,
# genuine C -- a `static const` array with a tentative (unsized/no-
# initializer) declaration followed by its real definition, which C
# permits and C++ does not, so it can't just be compiled as C++).
project(frameforge_engine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FRAMEFORGE_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(FRAMEFORGE_BUILD_BENCH "Build Google Benchmark microbenchmarks" OFF)
option(FRAMEFORGE_ENABLE_CUDA_EP "Enable ONNX Runtime CUDA execution provider" OFF)
option(FRAMEFORGE_ENABLE_ASAN "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(FRAMEFORGE_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if(FRAMEFORGE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(FRAMEFORGE_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Threads REQUIRED)

# --- ONNX Runtime: prebuilt binary release via FetchContent (CLAUDE.md 7.1:
# "prebuilt release fetched via CMake FetchContent or a documented download
# script" -- this is the FetchContent option; scripts/fetch_onnxruntime.py
# is the equivalent standalone script, used e.g. to pre-warm the download
# outside a CMake configure). URL_HASH values are real SHA256 sums computed
# from the actual downloaded archives, not guessed (see D1-style lesson
# learned the hard way with googletest's hash in M0).
include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(FRAMEFORGE_ORT_VERSION "1.20.1")
if(WIN32)
  FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${FRAMEFORGE_ORT_VERSION}/onnxruntime-win-x64-${FRAMEFORGE_ORT_VERSION}.zip
    URL_HASH SHA256=78d447051e48bd2e1e778bba378bec4ece11191c9e538cf7b2c4a4565e8f5581
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
elseif(UNIX AND NOT APPLE)
  FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${FRAMEFORGE_ORT_VERSION}/onnxruntime-linux-x64-${FRAMEFORGE_ORT_VERSION}.tgz
    URL_HASH SHA256=67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
else()
  message(FATAL_ERROR "No ONNX Runtime prebuilt release configured for this platform. "
                       "Add one to engine/CMakeLists.txt with a real (not guessed) SHA256.")
endif()
FetchContent_MakeAvailable(onnxruntime)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
  )
else()
  file(GLOB FRAMEFORGE_ORT_SO "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so*")
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
  )
endif()
target_include_directories(onnxruntime INTERFACE "${onnxruntime_SOURCE_DIR}/include")

# --- frameforge_core: the real engine library (M4 onward). ---
add_library(frameforge_core STATIC
  src/exr_io.cpp
  src/ort_session.cpp
  src/png_io.cpp
  src/preprocess.cpp
  src/sequence_runner.cpp
  src/pipeline.cpp
  src/tile.cpp
  include/tinyexr/miniz/miniz.c
)
# queue.hpp (BlockingQueue, SpscRingBuffer) is header-only (templates) --
# no corresponding .cpp. miniz.c (see THIRD_PARTY.md) compiles as real C
# per the `project(... LANGUAGES CXX C)` above.
target_include_directories(frameforge_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tinyexr/miniz)
target_link_libraries(frameforge_core PUBLIC onnxruntime Threads::Threads)
target_compile_features(frameforge_core PUBLIC cxx_std_17)

add_executable(frameforge src/main.cpp)
target_link_libraries(frameforge PRIVATE frameforge_core)

# Copy the ONNX Runtime shared library next to built executables (Windows
# has no rpath equivalent; this keeps `./frameforge.exe` runnable without
# manually copying onnxruntime.dll around).
if(WIN32)
  add_custom_command(TARGET frameforge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
      "$<TARGET_FILE_DIR:frameforge>"
  )
endif()

if(FRAMEFORGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(FRAMEFORGE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# baselines/fsr1_easu lives outside engine/ per the spec's repo layout
# (section 2), but shares this build tree (and, when tests are enabled,
# the gtest_main target that engine/tests/CMakeLists.txt just fetched --
# hence this add_subdirectory comes AFTER the tests one above).
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/fsr1_easu
  ${CMAKE_CURRENT_BINARY_DIR}/fsr1_easu
)
