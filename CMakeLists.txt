cmake_minimum_required(VERSION 3.0.2)
project(human_detector)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  gazebo_ros
  gmapping
  turtlebot3_slam
)

## Find system dependencies
find_package(OpenCV REQUIRED)
find_package(gazebo REQUIRED)

## Declare the catkin package
catkin_package(
  CATKIN_DEPENDS rospy roscpp std_msgs sensor_msgs cv_bridge gazebo_ros gmapping turtlebot3_slam
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

## Link against Gazebo and other libraries
link_directories(${GAZEBO_LIBRARY_DIRS})

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/human_detector.py
  scripts/slam-script.py
  scripts/slam-detection.py
  scripts/yolo-detection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install Gazebo world files
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
  FILES_MATCHING PATTERN "*.world"
)

## Install Gazebo models
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

## Install configuration files if you have any
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

## Create maps directory in the install space
install(DIRECTORY maps
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Install YOLO models
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  FILES_MATCHING PATTERN "*.pt"
)