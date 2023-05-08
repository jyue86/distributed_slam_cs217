cmake_minimum_required(VERSION 3.22.1)
project(distributed_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
set(THIRD_PARTY_LIBS ${realsense2_LIBRARY} ${OpenCV_LIBRARIES})

add_executable(
  main
  src/main.cpp
  src/utils/logger.hpp
  src/utils/logger.cpp
  src/utils/realsense.hpp
  src/utils/realsense.cpp
  src/frontend/featureMatcher.hpp
  src/frontend/featureMatcher.cpp)
target_link_libraries(main ${THIRD_PARTY_LIBS})

add_executable(aruco src/utils/aruco.cpp src/utils/aruco.hpp)
target_link_libraries(aruco ${OpenCV_LIBRARIES})
