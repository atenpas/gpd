cmake_minimum_required(VERSION 2.8.3)
set(GPD_LIB grasp_pose_detection PARENT_SCOPE)
project(grasp_pose_detection)

# PCL
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# GPG
find_library(gpg_LIBRARIES grasp_candidates_generator)
find_path(gpg_INCLUDE_DIRS gpg/grasp.h)
include_directories(${gpg_INCLUDE_DIRS})

# Caffe
if(USE_CAFFE STREQUAL "ON")
  message(INFO " USE_CAFFE")
  find_package(Caffe)
  include_directories(${Caffe_INCLUDE_DIRS})
  add_definitions(-DUSE_CAFFE)
  add_definitions(${Caffe_DEFINITIONS})
endif()

# OpenVINO
if(USE_OPENVINO STREQUAL "ON")
  message(INFO " USE_OPENVINO")
  find_package(InferenceEngine)
  if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR " Please install OpenVINO https://github.com/opencv/dldt")
  endif()
  include_directories(${InferenceEngine_INCLUDE_DIRS})
  link_directories(${InferenceEngine_LIBRARY_DIRS})
  add_definitions(-DUSE_OPENVINO)
  get_filename_component(MODELS_DIR "../../models/openvino" ABSOLUTE)
  configure_file(../../include/gpd/openvino_classifier.h.in gpd/openvino_classifier.h)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  set(classifier_src openvino_classifier.cpp)
  set(classifier_dep ${InferenceEngine_LIBRARIES})
elseif (USE_CAFFE)
  set(classifier_src caffe_classifier.cpp)
  set(classifier_dep ${Caffe_LIBRARIES})
endif()

## Set compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fopenmp -flto -Wno-deprecated -Wenum-compare -std=c++11")

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(../../include)

set(libsrc
  ${libsrc}
  ${classifier_src}
  classifier.cpp
  clustering.cpp
  grasp_detector.cpp
  learning.cpp
  sequential_importance_sampling.cpp
  grasp_image.cpp
  grasp_image_15_channels.cpp
)
if(USE_CAFFE)
set(libsrc
  ${libsrc}
  data_generator.cpp
)
endif()

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
  ${libsrc}
)

set(libdep
  ${classifier_dep}
  ${GENERATOR_LIB}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
)
target_link_libraries(${PROJECT_NAME}
  ${libdep}
)

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib)

install(DIRECTORY ../../include/gpd
  DESTINATION include)
