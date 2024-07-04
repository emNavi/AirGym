cmake_minimum_required(VERSION 3.0.2)
project(vechicle_controller)

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  roscpp
  std_msgs
)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
  src/qr_controller.cpp
  src/regular_motion.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)