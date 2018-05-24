cmake_minimum_required(VERSION 2.8.3)
project(lidar)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  pcl_ros
  pcl_conversions
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV 2 REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
)

# editted by sam in 2018-1-17
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})

add_executable(lidar src/lidar_main.cpp)
target_link_libraries(lidar
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${OpenCV_LIBS}
)
