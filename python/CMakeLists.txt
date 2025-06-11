cmake_minimum_required(VERSION 3.15)
project(pymonogenic)

find_package(OpenCV REQUIRED) # Find OpenCV
find_package(pybind11 CONFIG) # Ensure pybind11 is found

# Include directories
include_directories(PRIVATE
    ${CMAKE_SOURCE_DIR}/monogenic/include       # Include monogenic headers
    ${CMAKE_SOURCE_DIR}/monogenic/src         # Include monogenic headers
    ${CMAKE_SOURCE_DIR}/cvnp/include
    ${CMAKE_SOURCE_DIR}/cvnp
    ${OpenCV_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}                  # Include pybind11 headers
)

# Add your wrapper source file
pybind11_add_module(pymonogenic pymonogenic.cpp) # Adjust module name if desired

target_include_directories(pymonogenic PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${cvnp_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/monogenic/src
)

target_link_libraries(pymonogenic PRIVATE pybind11::embed cvnp monogenic ${OpenCV_LIBS}) # Adjust monogenic_lib name if needed

install(TARGETS pymonogenic
        COMPONENT python
        LIBRARY DESTINATION "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin"
)

