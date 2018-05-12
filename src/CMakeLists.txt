cmake_minimum_required(VERSION 3.5.1)

find_package(Eigen3 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Boost 1.62.0 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++17")

add_library(
    optimizers
    F.cpp
    X.cpp
    W.cpp
    factor.cpp
    predict.cpp
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
        W_test.cpp
        X_test.cpp
        factor_test.cpp
    )

    target_link_libraries(runTests optimizers GTest::GTest GTest::Main)

    add_test(test runTests)
endif()
