cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_COMPILER, "mpixx")
set(CMAKE_CXX_FLAGS 
"-std=c++98 -Wall -Werror -pedantic-errors -Wextra -ftrapv -O3 -g")

project(MonteCarloMethod)

find_package(MPI REQUIRED)

set(INCLUDE_DIR inc)
set(SOURCE_DIR src)
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp)

include_directories(${INCLUDE_DIR} ${MPI_INCLUDE_PATH})
add_executable(MonteCarloMethod ${SOURCE_FILES})

target_link_libraries(MonteCarloMethod MPI::MPI_CXX)