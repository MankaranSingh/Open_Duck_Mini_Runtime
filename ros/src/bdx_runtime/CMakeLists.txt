cmake_minimum_required(VERSION 2.8.3)
project(bdx_runtime)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  std_srvs
)

catkin_package()

## Install Python scripts (nodes)
catkin_install_python(PROGRAMS
  src/imu_node.py
  src/joystick_node.py
  src/motor_control_node.py
  src/feet_switch_node.py
  src/walk_policy_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Install URDF files
install(DIRECTORY urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
