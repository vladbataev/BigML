cmake_minimum_required(VERSION 3.9)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++17")

add_library(
    optimizers

    conjugated_gradients.cpp
    F_optimize.cpp
    X.cpp
)


add_executable(main main.cpp)

target_link_libraries(main optimizers)

# cmake -Dtest=ON
option(test "Build tests." OFF)

if (test)
    enable_testing()

    find_package(GTest REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    add_executable(runTests
        cg_test.cpp
        X_test.cpp
    )

    target_link_libraries(runTests optimizers GTest::GTest GTest::Main)

    add_test(test runTests)
endif()
