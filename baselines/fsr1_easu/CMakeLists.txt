add_library(fsr1_core STATIC src/fsr1.cpp)
target_include_directories(fsr1_core PUBLIC include ${CMAKE_SOURCE_DIR}/include)
target_compile_features(fsr1_core PUBLIC cxx_std_17)

add_executable(fsr1_cli src/main.cpp)
target_link_libraries(fsr1_cli PRIVATE fsr1_core)
target_include_directories(fsr1_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(FRAMEFORGE_BUILD_TESTS)
  add_executable(fsr1_tests tests/fsr1_golden_test.cpp)
  target_link_libraries(fsr1_tests PRIVATE fsr1_core gtest_main)
  include(GoogleTest)
  gtest_discover_tests(fsr1_tests)
endif()
