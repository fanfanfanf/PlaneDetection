cmake_minimum_required(VERSION 2.6)
project(plane_detection)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# find_library(TBB_LIB
#     NAMES
#     tbbbind_debug
#     tbbbind
#     tbb_debug
#     tbbmalloc_debug
#     tbbmalloc_proxy_debug
#     tbbmalloc_proxy
#     tbbmalloc
#     tbb_preview_debug
#     tbb_preview
#     tbb
#     HINTS "${CMAKE_PREFIX_PATH}/deps/tbb/lib/intel64/gcc4.8"
# )

# find_package( TBB REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(plane_detection 
               src/connectivitygraph.cpp src/plane.cpp src/circle.cpp src/cylinder.cpp 
               src/planarpatch.cpp src/planedetector.cpp main_pcl.cpp
               src/CommonFunctions.cpp src/line_detection_3d.cpp)
target_include_directories (plane_detection PUBLIC ./src)
target_include_directories (plane_detection PUBLIC ../eigen3)
target_link_libraries (plane_detection ${PCL_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(plane_detection PUBLIC TBB::tbb)