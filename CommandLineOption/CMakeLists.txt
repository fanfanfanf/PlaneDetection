cmake_minimum_required(VERSION 2.6)
project(plane_detection)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(plane_detection 
               src/connectivitygraph.cpp src/plane.cpp src/circle.cpp src/cylinder.cpp 
               src/planarpatch.cpp src/planedetector.cpp main_pcl.cpp)
target_include_directories (plane_detection PUBLIC ./src)
target_include_directories (plane_detection PUBLIC ../eigen3)
target_link_libraries (plane_detection ${PCL_LIBRARIES})