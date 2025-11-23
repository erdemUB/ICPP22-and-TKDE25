####
# To build (with valid GCC and CMAKE):

# cmake -B <build_directory>
# cd <build_directory>
# make
####

project(par6cycle CXX)
cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

# disallow in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")

###################
# Setup Executable
###################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fopenmp")

set(TBB_INSTALL_DIR ${CMAKE_BINARY_DIR}/tbb)

set(SOURCES preProcessing.cpp utilities.cpp)

link_directories(${TBB_INSTALL_DIR}/lib)
include_directories(${TBB_INSTALL_DIR}/include)

add_executable(NJ NJ.cpp ${SOURCES})
target_link_libraries(NJ tbb)

add_executable(WJ WJ.cpp ${SOURCES})
target_link_libraries(WJ tbb)

add_executable(TJ TJ.cpp ${SOURCES})
target_link_libraries(TJ tbb)

add_executable(BTJ BTJ.cpp ${SOURCES})
target_link_libraries(BTJ tbb)

###############
# Download TBB
###############

include(ExternalProject)
ExternalProject_Add(
	tbb
	GIT_REPOSITORY https://github.com/wjakob/tbb.git
	GIT_TAG 9e219e24fe
	INSTALL_DIR ${TBB_INSTALL_DIR}
	CMAKE_ARGS
	-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
	-DTBB_BUILD_SHARED=ON
	-DTBB_BUILD_STATIC=OFF
	-DTBB_BUILD_TBBMALLOC=OFF
	-DTBB_BUILD_TBBMALLOC_PROXY=OFF
	-DTBB_BUILD_TESTS=OFF
)
add_dependencies(NJ tbb)
add_dependencies(WJ tbb)
add_dependencies(TJ tbb)
add_dependencies(BTJ tbb)
