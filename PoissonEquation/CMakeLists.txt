cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_COMPILER, "mpicxx")
set(CMAKE_CXX_FLAGS 
"-std=gnu++98 -Wall -Werror -pedantic-errors -Wextra -ftrapv -O3 -g -fopenmp")
set(CMAKE_CXX_STANDARD, 98)

project(PoissonEquation)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(INCLUDE_DIR inc)
set(SOURCE_DIR src)
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp
${SOURCE_DIR}/SerialSolver.cpp
${SOURCE_DIR}/ParallelSolver.cpp
${SOURCE_DIR}/MultiThreadSolver.cpp
${SOURCE_DIR}/GPUSolver.cpp
${SOURCE_DIR}/Interactor.cpp)

include_directories(${INCLUDE_DIR} ${MPI_INCLUDE_PATH})
add_executable(PoissonEquation ${SOURCE_FILES})

target_link_libraries(PoissonEquation MPI::MPI_CXX)
target_link_libraries(PoissonEquation OpenMP::OpenMP_CXX)
target_link_libraries(PoissonEquation m)