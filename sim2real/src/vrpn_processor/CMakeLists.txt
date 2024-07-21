cmake_minimum_required(VERSION 3.0.2)
project(vrpn_processor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/vrpn_processor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
