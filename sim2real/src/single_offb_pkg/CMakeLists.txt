cmake_minimum_required(VERSION 3.0.2)
project(single_offb_pkg)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  roscpp
  std_msgs
  vechicle_controller
  quadrotor_msgs
  # map_operation_pkg
  # path_searching_pkg
  # poly_curves_pkg
)

catkin_package(
  CATKIN_DEPENDS vechicle_controller
  # CATKIN_DEPENDS map_operation_pkg
  # CATKIN_DEPENDS path_searching_pkg
  # CATKIN_DEPENDS poly_curves_pkg
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  "/usr/include/eigen3"
)

add_executable(uav_ctrl src/uav_ctrl.cpp)
target_link_libraries(uav_ctrl
  ${catkin_LIBRARIES}
)


# add_executable(plan_node src/path_plan_test.cpp)
# target_link_libraries(plan_node
#   ${catkin_LIBRARIES}
# )

# add_executable(ctrl_and_plan_node src/plan_and_ctrl_node.cpp)
# target_link_libraries(ctrl_and_plan_node
#   ${catkin_LIBRARIES}
# )

add_executable(mission_ctrl src/mission_ctrl.cpp)
target_link_libraries(mission_ctrl
  ${catkin_LIBRARIES}
)
