cmake_minimum_required(VERSION 2.8.3)
project(gpd)

# global options at the beginning
option(USE_CAFFE "use Caffe framework" ON)
option(USE_OPENVINO "use OpenVINO toolkit" OFF)
if (USE_OPENVINO STREQUAL "ON")
  set(USE_CAFFE OFF)
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS cmake_modules eigen_conversions geometry_msgs message_generation roscpp
  sensor_msgs std_msgs)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# custom messages
find_package(message_generation)

# Grasp Candidtes Generator
find_library(GENERATOR_LIB grasp_candidates_generator)
find_path(GENERATOR_LIB_INCLUDE_DIR gpg/grasp.h)
include_directories(${GENERATOR_LIB_INCLUDE_DIR})

# Caffe
if(USE_CAFFE STREQUAL "ON")
  find_package(Caffe)
  include_directories(${Caffe_INCLUDE_DIRS})
  add_definitions(${Caffe_DEFINITIONS})
  add_definitions(-DUSE_CAFFE)
endif()

## Set compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp -flto -Wno-deprecated -Wenum-compare -std=c++11")
# set(CMAKE_CXX_FLAGS "-DNDEBUG -O3 -fopenmp -flto -mavx -mfma -Wno-deprecated -Wenum-compare")

# Grasp Detection Library
add_subdirectory(src/gpd)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(FILES CloudIndexed.msg CloudSamples.msg CloudSources.msg GraspConfig.msg GraspConfigList.msg
  SamplesMsg.msg)

## Generate services in the 'srv' folder
add_service_files(DIRECTORY srv
 FILES
   SetParameters.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES geometry_msgs sensor_msgs std_msgs)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
INCLUDE_DIRS include
CATKIN_DEPENDS cmake_modules eigen_conversions geometry_msgs message_runtime roscpp sensor_msgs std_msgs
DEPENDS Eigen OpenCV PCL
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(grasp_candidates_classifier ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare C++ executables
add_executable(${PROJECT_NAME}_detect_grasps src/nodes/grasp_detection_node.cpp src/nodes/ros_params.cpp)
add_dependencies(${PROJECT_NAME}_detect_grasps ${GPD_LIB})

add_executable(${PROJECT_NAME}_generate_candidates src/nodes/generate_candidates.cpp)
add_dependencies(${PROJECT_NAME}_generate_candidates ${GPD_LIB})

add_executable(${PROJECT_NAME}_create_grasp_images src/nodes/create_grasp_images.cpp)
add_dependencies(${PROJECT_NAME}_create_grasp_images ${GPD_LIB})

if(USE_CAFFE STREQUAL "ON")
  add_executable(${PROJECT_NAME}_create_training_data src/nodes/create_training_data.cpp src/nodes/ros_params.cpp)
  add_dependencies(${PROJECT_NAME}_create_training_data ${GPD_LIB})
endif()

add_executable(${PROJECT_NAME}_classify_candidates src/nodes/classify_candidates.cpp src/nodes/ros_params.cpp)
add_dependencies(${PROJECT_NAME}_classify_candidates ${GPD_LIB})

if(USE_CAFFE STREQUAL "ON")
  add_executable(${PROJECT_NAME}_test_occlusion src/tests/test_occlusion.cpp)
  add_dependencies(${PROJECT_NAME}_test_occlusion ${GPD_LIB})
endif()

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(grasp_candidates_classifier_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_create_grasp_images
                      ${GPD_LIB}
                      ${GENERATOR_LIB}
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

if(USE_CAFFE STREQUAL "ON")
  target_link_libraries(${PROJECT_NAME}_create_training_data
                      ${GPD_LIB}
                      ${catkin_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME}_generate_candidates
                      ${GPD_LIB}
                      ${GENERATOR_LIB}
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_classify_candidates
                      ${GPD_LIB}
                      ${GENERATOR_LIB}
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_detect_grasps
                      ${GPD_LIB}
                      ${GENERATOR_LIB}
                      ${catkin_LIBRARIES}
                      ${PCL_LIBRARIES})

if(USE_CAFFE STREQUAL "ON")
  target_link_libraries(${PROJECT_NAME}_test_occlusion
                        ${GPD_LIB}
                        ${GENERATOR_LIB}
                        ${catkin_LIBRARIES}
                        ${PCL_LIBRARIES}
                        ${Caffe_LIBRARIES})
endif()

# Rename targets to simplify their names.
set_target_properties(${PROJECT_NAME}_detect_grasps
                      PROPERTIES OUTPUT_NAME detect_grasps
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_classify_candidates
                      PROPERTIES OUTPUT_NAME classify_candidates
                      PREFIX "")

set_target_properties(${PROJECT_NAME}_create_grasp_images
                      PROPERTIES OUTPUT_NAME create_grasp_images
                      PREFIX "")

if(USE_CAFFE STREQUAL "ON")
  set_target_properties(${PROJECT_NAME}_create_training_data
                      PROPERTIES OUTPUT_NAME create_training_data
                      PREFIX "")
endif()

if(USE_CAFFE STREQUAL "ON")
  set_target_properties(${PROJECT_NAME}_test_occlusion
                        PROPERTIES OUTPUT_NAME test_occlusion
                        PREFIX "")
endif()
