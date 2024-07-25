cmake_minimum_required(VERSION 3.0.2)
project(control_for_gym)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  roscpp
  std_msgs
  # map_operation_pkg
  # path_searching_pkg
  # poly_curves_pkg
)

catkin_package(
  # CATKIN_DEPENDS 
  # CATKIN_DEPENDS map_operation_pkg
  # CATKIN_DEPENDS path_searching_pkg
  # CATKIN_DEPENDS poly_curves_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(uav_ctrl src/uav_ctrl.cpp)
target_link_libraries(uav_ctrl
  ${catkin_LIBRARIES}
)
