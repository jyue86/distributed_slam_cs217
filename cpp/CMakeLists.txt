cmake_minimum_required(VERSION 3.22.1)
project(distributed_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(realsense2 REQUIRED)
find_package(OpenCV EXACT 4.7.0 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${GTSAM_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})
set(THIRD_PARTY_LIBS ${realsense2_LIBRARY} ${OpenCV_LIBRARIES}
                     ${Boost_LIBRARIES} ${GTSAM_LIBRARIES})

add_executable(
  main
  src/main.cpp
  src/utils/logger.hpp
  src/utils/logger.cpp
  src/utils/realsense.hpp
  src/utils/realsense.cpp
  src/frontend/featureMatcher.hpp
  src/frontend/featureMatcher.cpp
  src/frontend/arucoFrontEnd.hpp
  src/frontend/arucoFrontEnd.cpp)
target_link_libraries(main ${THIRD_PARTY_LIBS})

add_executable(aruco src/utils/generateAruco.cpp src/utils/generateAruco.hpp)
target_link_libraries(aruco ${OpenCV_LIBRARIES})

add_executable(
  getCalibData
  src/utils/calibrateWithCharuco.cpp src/frontend/arucoFrontEnd.hpp
  src/frontend/arucoFrontEnd.cpp src/utils/realsense.hpp
  src/utils/realsense.cpp)
target_link_libraries(getCalibData ${THIRD_PARTY_LIBS})
