cmake_minimum_required(VERSION 3.10)
project(LandmarkDetectionMNN)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(
        mnn/include
        src
)

link_directories(mnn/lib)

add_executable(LandmarkDetectionMNN src/main.cpp src/Landmark.cpp)
target_link_libraries(LandmarkDetectionMNN MNN ${OpenCV_LIBS})