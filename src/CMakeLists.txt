cmake_minimum_required(VERSION 3.5.1)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost 1.62.0 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

SET(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++17")

add_library(
    optimizers

    conjugated_gradients.cpp
    F_optimize.cpp
    X.cpp
    W.cpp
    factor.cpp
)


add_executable(main main.cpp)

target_link_libraries(main LINK_PUBLIC ${Boost_LIBRARIES} optimizers)



# cmake -Dtest=ON
option(test "Build tests." OFF)

if (test)
    enable_testing()

    find_package(GTest REQUIRED)

    include_directories(${GTEST_INCLUDE_DIRS})

    add_executable(runTests
        cg_test.cc
        X_test.cpp
    )

    target_link_libraries(runTests optimizers GTest::GTest GTest::Main)

    add_test(test runTests)
endif()
