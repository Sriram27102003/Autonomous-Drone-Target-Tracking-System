cmake_minimum_required(VERSION 3.0.2)
project(human_detector)

find_package(catkin REQUIRED)

# Installs the pure-Python library in src/human_detector (see setup.py)
catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/person_tracker_node.py
  scripts/target_follower_node.py
  scripts/explorer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config worlds maps
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
