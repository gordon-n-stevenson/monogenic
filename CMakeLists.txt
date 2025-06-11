# This is a stub CMakeLists.txt file for the monogenic project
# It provides basic configuration for building the monogenic library with OpenCV dependencies

cmake_minimum_required(VERSION 3.22)
project(monogenic CXX)

# Set the C++ standard for your wrapper project
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include # Include monogenic library headers
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(monogenic src/monogenicProcessor.cpp)
target_link_libraries(monogenic ${OpenCV_LIBS})

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs)

add_executable(monogenic_image_test example/cpp/monogenicImageTest.cpp)

target_include_directories(monogenic_image_test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(monogenic_image_test PUBLIC
    monogenic         # Link to the monogenic library target
    ${OpenCV_LIBS}    # Link to the necessary OpenCV libraries
)

add_subdirectory(pybind11)
add_subdirectory(cvnp)
add_subdirectory(cpp)