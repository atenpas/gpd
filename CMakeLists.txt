cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpd)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)

# Eigen library
include_directories(${EIGEN3_INCLUDE_DIR})

# PCL
find_package(PCL 1.9 REQUIRED)
#find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV REQUIRED)
#find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

## Set compiler optimization flags
# set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -std=c++14")
#set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17")
#set(CMAKE_CXX_FLAGS "-fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17")
# set(CMAKE_CXX_FLAGS "-O3 -fopenmp -fPIC -Wno-deprecated -Wenum-compare")
# set(CMAKE_CXX_FLAGS "-O3 -fopenmp -march=native -mfpmath=sse -funroll-loops -fPIC -Wno-deprecated -Wenum-compare") # no improvement
# set(CMAKE_CXX_FLAGS "-frename-registers -Ofast -march=native -fopenmp -fPIC -Wno-deprecated -Wenum-compare") # no improvement

#set(CMAKE_CXX_FLAGS "-fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17")
set(CMAKE_CXX_FLAGS "-O3 -march=native -mtune=intel -msse4.2 -mavx2 -mfma -flto -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes -std=c++17")

## Specify additional locations of header files
include_directories(include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Classifier depends on what libraries are installed.
option(USE_OPENVINO "use classifier based on OpenVINO toolkit" OFF)
option(USE_CAFFE "use classifier based on Caffe framework" OFF)
option(USE_OPENCV "use classifier based on OpenCV framework" OFF)
if(USE_OPENVINO STREQUAL "ON")
  find_package(InferenceEngine 1.0)
  if (NOT InferenceEngine_FOUND)
    message(FATAL_ERROR "Please install OpenVINO https://software.intel.com/en-us/articles/OpenVINO-Install-Linux")
  endif()
  include_directories(${InferenceEngine_INCLUDE_DIRS})
  link_directories(${InferenceEngine_LIBRARY_DIRS})
  add_definitions(-DUSE_OPENVINO)
  get_filename_component(MODELS_DIR "models/openvino" ABSOLUTE)
  configure_file(include/gpd/net/openvino_classifier.h.in gpd/net/openvino_classifier.h)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  set(classifier_src src/${PROJECT_NAME}/net/classifier.cpp src/${PROJECT_NAME}/net/openvino_classifier.cpp)
  set(classifier_dep ${InferenceEngine_LIBRARIES})
  message("Using OpenVINO")
elseif(USE_CAFFE STREQUAL "ON")
  find_package(Caffe)
  if (NOT Caffe_FOUND)
    message(FATAL_ERROR "Please install Caffe https://caffe.berkeleyvision.org/installation.html")
  endif()
  include_directories(${Caffe_INCLUDE_DIRS})
  add_definitions(${Caffe_DEFINITIONS})
  add_definitions(-DUSE_CAFFE)
  set(classifier_src src/${PROJECT_NAME}/net/classifier.cpp src/${PROJECT_NAME}/net/caffe_classifier.cpp)
  set(classifier_dep ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})
elseif(USE_OPENCV STREQUAL "ON")
  message(FATAL_ERROR "Not supported yet")
else()
  add_library(${PROJECT_NAME}_conv_layer src/${PROJECT_NAME}/net/conv_layer.cpp)
  add_library(${PROJECT_NAME}_dense_layer src/${PROJECT_NAME}/net/dense_layer.cpp)
  set(classifier_src src/${PROJECT_NAME}/net/classifier.cpp src/${PROJECT_NAME}/net/eigen_classifier.cpp)
  set(classifier_dep ${PROJECT_NAME}_conv_layer ${PROJECT_NAME}_dense_layer ${OpenCV_LIBRARIES})
endif()

# Optional PCL GPU operations
option(USE_PCL_GPU "use PCL GPU for point cloud operations" OFF)
if(USE_PCL_GPU STREQUAL "ON")
  FIND_PACKAGE(CUDA)
  if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Please install CUDA and compile PCL with Nvidia GPU flags http://pointclouds.org/documentation/tutorials/gpu_install.php")
  endif()
  INCLUDE(FindCUDA)
  add_definitions(-DUSE_PCL_GPU)
  message("Using the GPU with PCL")
endif()

# Optional data generation (requires OpenCV Contrib)
option(BUILD_DATA_GENERATION "build data generation (requires OpenCV Contrib for HDF5)" OFF)
if(BUILD_DATA_GENERATION STREQUAL "ON")
  add_library(${PROJECT_NAME}_data_generator src/${PROJECT_NAME}/data_generator.cpp)
  target_link_libraries(${PROJECT_NAME}_data_generator
   ${PROJECT_NAME}_grasp_detector
   ${PCL_LIBRARIES}
   ${OpenCV_LIBS})
  add_executable(${PROJECT_NAME}_generate_data src/generate_data.cpp)
  target_link_libraries(${PROJECT_NAME}_generate_data
   ${PROJECT_NAME}_data_generator)
  message("Building data generation module")
endif()

# Generate the shared library from the sources
add_library(${PROJECT_NAME}_grasp_detector SHARED src/${PROJECT_NAME}/grasp_detector.cpp)

add_library(${PROJECT_NAME}_classifier ${classifier_src})
target_link_libraries(${PROJECT_NAME}_classifier
                      ${classifier_dep})

add_library(${PROJECT_NAME}_clustering src/${PROJECT_NAME}/clustering.cpp)
add_library(${PROJECT_NAME}_sequential_importance_sampling src/${PROJECT_NAME}/sequential_importance_sampling.cpp)

# namespace candidate
add_library(${PROJECT_NAME}_antipodal src/${PROJECT_NAME}/candidate/antipodal.cpp)
add_library(${PROJECT_NAME}_candidates_generator src/${PROJECT_NAME}/candidate/candidates_generator.cpp)
add_library(${PROJECT_NAME}_finger_hand src/${PROJECT_NAME}/candidate/finger_hand.cpp)
add_library(${PROJECT_NAME}_frame_estimator src/${PROJECT_NAME}/candidate/frame_estimator.cpp)
add_library(${PROJECT_NAME}_hand src/${PROJECT_NAME}/candidate/hand.cpp)
add_library(${PROJECT_NAME}_hand_set src/${PROJECT_NAME}/candidate/hand_set.cpp)
add_library(${PROJECT_NAME}_hand_geometry src/${PROJECT_NAME}/candidate/hand_geometry.cpp)
add_library(${PROJECT_NAME}_hand_search src/${PROJECT_NAME}/candidate/hand_search.cpp)
add_library(${PROJECT_NAME}_local_frame src/${PROJECT_NAME}/candidate/local_frame.cpp)

# namespace util
add_library(${PROJECT_NAME}_cloud src/${PROJECT_NAME}/util/cloud.cpp)
add_library(${PROJECT_NAME}_config_file src/${PROJECT_NAME}/util/config_file.cpp)
add_library(${PROJECT_NAME}_eigen_utils src/${PROJECT_NAME}/util/eigen_utils.cpp)
add_library(${PROJECT_NAME}_plot src/${PROJECT_NAME}/util/plot.cpp)
add_library(${PROJECT_NAME}_point_list src/${PROJECT_NAME}/util/point_list.cpp)

# namespace descriptor
add_library(${PROJECT_NAME}_image_strategy src/${PROJECT_NAME}/descriptor/image_strategy.cpp)
add_library(${PROJECT_NAME}_image_1_channels_strategy src/${PROJECT_NAME}/descriptor/image_1_channels_strategy.cpp)
add_library(${PROJECT_NAME}_image_3_channels_strategy src/${PROJECT_NAME}/descriptor/image_3_channels_strategy.cpp)
add_library(${PROJECT_NAME}_image_12_channels_strategy src/${PROJECT_NAME}/descriptor/image_12_channels_strategy.cpp)
add_library(${PROJECT_NAME}_image_15_channels_strategy src/${PROJECT_NAME}/descriptor/image_15_channels_strategy.cpp)
add_library(${PROJECT_NAME}_image_geometry src/${PROJECT_NAME}/descriptor/image_geometry.cpp)

add_library(${PROJECT_NAME}_image_generator src/${PROJECT_NAME}/descriptor/image_generator.cpp)

## Declare C++ executables
add_executable(${PROJECT_NAME}_cem_detect_grasps src/cem_detect_grasps.cpp)
add_executable(${PROJECT_NAME}_detect_grasps src/detect_grasps.cpp)
add_executable(${PROJECT_NAME}_generate_candidates src/generate_candidates.cpp)
add_executable(${PROJECT_NAME}_label_grasps src/label_grasps.cpp)
add_executable(${PROJECT_NAME}_test_grasp_image src/tests/test_grasp_image.cpp)
# add_executable(${PROJECT_NAME}_test_conv_layer src/tests/test_conv_layer.cpp)
# add_executable(${PROJECT_NAME}_test_hdf5 src/tests/test_hdf5.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_clustering
${PROJECT_NAME}_hand)

target_link_libraries(${PROJECT_NAME}_grasp_detector
  ${PROJECT_NAME}_clustering
  ${PROJECT_NAME}_image_generator
  ${PROJECT_NAME}_classifier
  ${PROJECT_NAME}_candidates_generator
  ${PROJECT_NAME}_hand_geometry
  ${PROJECT_NAME}_hand_set
  ${PROJECT_NAME}_config_file
  ${PROJECT_NAME}_plot)

target_link_libraries(${PROJECT_NAME}_generate_candidates
  ${PROJECT_NAME}_config_file
${PROJECT_NAME}_candidates_generator)

# Linking for libraries
target_link_libraries(${PROJECT_NAME}_antipodal
${PROJECT_NAME}_point_list)

target_link_libraries(${PROJECT_NAME}_cloud
  ${PROJECT_NAME}_eigen_utils
  ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_eigen_utils
${EIGEN_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_frame_estimator
  ${PROJECT_NAME}_cloud
${PROJECT_NAME}_local_frame)

target_link_libraries(${PROJECT_NAME}_hand
${PROJECT_NAME}_finger_hand)

target_link_libraries(${PROJECT_NAME}_hand_set
  ${PROJECT_NAME}_antipodal
  ${PROJECT_NAME}_config_file
  ${PROJECT_NAME}_hand
  ${PROJECT_NAME}_hand_geometry
  ${PROJECT_NAME}_local_frame
${PROJECT_NAME}_point_list)

target_link_libraries(${PROJECT_NAME}_hand_geometry
${PROJECT_NAME}_config_file)

target_link_libraries(${PROJECT_NAME}_hand_search
  ${PROJECT_NAME}_antipodal
  ${PROJECT_NAME}_cloud
  ${PROJECT_NAME}_frame_estimator
  ${PROJECT_NAME}_hand_set
  ${PROJECT_NAME}_hand_geometry
  ${PROJECT_NAME}_plot)

target_link_libraries(${PROJECT_NAME}_plot
  ${PROJECT_NAME}_cloud
  ${PROJECT_NAME}_hand_set
  ${PROJECT_NAME}_image_geometry
${PROJECT_NAME}_local_frame)

target_link_libraries(${PROJECT_NAME}_candidates_generator
  ${PROJECT_NAME}_config_file
  ${PROJECT_NAME}_hand_geometry
${PROJECT_NAME}_hand_search)

target_link_libraries(${PROJECT_NAME}_point_list
${PROJECT_NAME}_eigen_utils)

target_link_libraries(${PROJECT_NAME}_image_strategy
  ${PROJECT_NAME}_image_geometry
  ${PROJECT_NAME}_image_1_channels_strategy
  ${PROJECT_NAME}_image_3_channels_strategy
  ${PROJECT_NAME}_image_12_channels_strategy
  ${PROJECT_NAME}_image_15_channels_strategy
  ${PROJECT_NAME}_hand_set
${OpenCV_LIBS})


target_link_libraries(${PROJECT_NAME}_image_1_channels_strategy
${PROJECT_NAME}_image_strategy)

target_link_libraries(${PROJECT_NAME}_image_3_channels_strategy
${PROJECT_NAME}_image_strategy)

target_link_libraries(${PROJECT_NAME}_image_12_channels_strategy
${PROJECT_NAME}_image_strategy)

target_link_libraries(${PROJECT_NAME}_image_15_channels_strategy
${PROJECT_NAME}_image_strategy)

target_link_libraries(${PROJECT_NAME}_image_geometry
${PROJECT_NAME}_config_file)

target_link_libraries(${PROJECT_NAME}_image_generator
  ${PROJECT_NAME}_image_strategy
  ${PROJECT_NAME}_hand_set
  ${PROJECT_NAME}_image_strategy
  ${PROJECT_NAME}_cloud
${PROJECT_NAME}_eigen_utils)

target_link_libraries(${PROJECT_NAME}_sequential_importance_sampling
  ${PROJECT_NAME}_grasp_detector)

target_link_libraries(${PROJECT_NAME}_test_grasp_image
  ${PROJECT_NAME}_image_generator
  ${PROJECT_NAME}_classifier
  ${PROJECT_NAME}_plot
  ${PROJECT_NAME}_candidates_generator
${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_detect_grasps
  ${PROJECT_NAME}_grasp_detector
  ${PROJECT_NAME}_config_file
${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_label_grasps
  ${PROJECT_NAME}_grasp_detector
  ${PROJECT_NAME}_config_file
${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_cem_detect_grasps
  ${PROJECT_NAME}_sequential_importance_sampling
  ${PROJECT_NAME}_config_file
  ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_image_generator
  ${PROJECT_NAME}_hand_set
  ${PROJECT_NAME}_image_strategy)

# Rename targets to simplify their names.
set_target_properties(${PROJECT_NAME}_test_grasp_image
  PROPERTIES OUTPUT_NAME test_grasp_image PREFIX "")

set_target_properties(${PROJECT_NAME}_cem_detect_grasps
  PROPERTIES OUTPUT_NAME cem_detect_grasps PREFIX "")

set_target_properties(${PROJECT_NAME}_detect_grasps
  PROPERTIES OUTPUT_NAME detect_grasps PREFIX "")

set_target_properties(${PROJECT_NAME}_generate_candidates
  PROPERTIES OUTPUT_NAME generate_candidates PREFIX "")

if(BUILD_DATA_GENERATION STREQUAL "ON")
  set_target_properties(${PROJECT_NAME}_generate_data
    PROPERTIES OUTPUT_NAME generate_data PREFIX "")
endif()

set_target_properties(${PROJECT_NAME}_label_grasps
  PROPERTIES OUTPUT_NAME label_grasps PREFIX "")

set_target_properties(${PROJECT_NAME}_grasp_detector
  PROPERTIES OUTPUT_NAME gpd)


# Set the location for library installation
install(TARGETS ${PROJECT_NAME}_grasp_detector DESTINATION lib)

install(DIRECTORY include/gpd DESTINATION include)
