# Set the minimum required CMake version.
# Using a recent version ensures access to modern CMake features.
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(monogenic CXX)

# Set the C++ standard. C++11 or higher is recommended for modern C++.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package. The example uses OpenCV for image loading/saving.
# COMPONENTS specify which parts of OpenCV are needed.
find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs)

# Define the library target for the monogenic signal computation.
# This creates a static library named 'monogenic'.
add_library(monogenic STATIC
    src/monogenicProcessor.cpp
    src/monogenicProcessor.h    
)

# Specify include directories for the library.
# This makes the header file available when building the library.
target_include_directories(monogenic PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link the library to necessary external libraries (like OpenCV core).
# Although monogenic.cpp itself might not directly call OpenCV functions,
# it's good practice to link dependencies if they were used internally
# or if the library's public interface relies on types from those libraries.
# In this case, the core monogenic code is pure C++, but if it were extended,
# this is where you'd add dependencies. For now, we'll keep it minimal.
target_link_libraries(monogenic PUBLIC ${OpenCV_LIBS}) # Uncomment if monogenic.cpp needs OpenCV

# --- Setup the Example Executable ---

# Define the executable target for the example.
add_executable(monogenic_video_example example/monogenicTest.cpp)

# Define the executable target for the example.
add_executable(monogenic_image_example example/monogenicImageTest.cpp)


# Specify include directories for the example.
# It needs access to the monogenic library headers and OpenCV headers.
target_include_directories(monogenic_video_example PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src # Include monogenic library headers
    ${OpenCV_INCLUDE_DIRS}          # Include OpenCV headers
)

# Link the example executable to the monogenic library and OpenCV libraries.
# This is crucial so the example can use functions from the library and OpenCV.
target_link_libraries(monogenic_video_example PUBLIC
    monogenic         # Link to the monogenic library we defined
    ${OpenCV_LIBS}    # Link to the necessary OpenCV libraries found by find_package
)

# Specify include directories for the example.
# It needs access to the monogenic library headers and OpenCV headers.
target_include_directories(monogenic_image_example PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src # Include monogenic library headers
    ${OpenCV_INCLUDE_DIRS}          # Include OpenCV headers
)

# Link the example executable to the monogenic library and OpenCV libraries.
# This is crucial so the example can use functions from the library and OpenCV.
target_link_libraries(monogenic_image_example PUBLIC
    monogenic         # Link to the monogenic library we defined
    ${OpenCV_LIBS}    # Link to the necessary OpenCV libraries found by find_package
)

# Install rules (optional, but good practice)
# Install the library
install(TARGETS monogenic
    DESTINATION lib
)

# Install the header file
install(FILES include/monogenic/monogenicProcessor.h
    DESTINATION bin
)

# Install the example executable
install(TARGETS monogenic_image_example
    DESTINATION bin
)

# Install the example executable
install(TARGETS monogenic_video_example
    DESTINATION bin
)
