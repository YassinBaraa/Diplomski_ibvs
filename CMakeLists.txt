cmake_minimum_required(VERSION 2.8.3)
project(ibvs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge
  opencv
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ibvs
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs cv_bridge
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Mark Python scripts for installation
catkin_install_python(PROGRAMS
  ibvs_ros/ibvs_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY ibvs_ros/launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY ibvs_ros/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# Install entire ibvs_ros python package
install(DIRECTORY ibvs_ros/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/ibvs_ros
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
