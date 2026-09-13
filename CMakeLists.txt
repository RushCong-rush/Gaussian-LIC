cmake_minimum_required(VERSION 3.0.2)
project(gaussian_lic)

SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf
  eigen_conversions
  message_generation
)

add_service_files(FILES RefinePose.srv)
generate_messages(DEPENDENCIES sensor_msgs geometry_msgs)

find_package(OpenCV 4 REQUIRED)
find_package(Torch REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY
  NAMES nvinfer libnvinfer.so.10
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(TENSORRT_PLUGIN_LIBRARY
  NAMES nvinfer_plugin libnvinfer_plugin.so.10
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_PLUGIN_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/runtime were not found")
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA toolkit root")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_120,code=sm_120")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(FFI_LIB ffi)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

catkin_package(
  CATKIN_DEPENDS message_runtime sensor_msgs geometry_msgs
#  INCLUDE_DIRS include
#  LIBRARIES gaussian_lic
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
# include
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(gs_mapping 
              ${PROJECT_SOURCE_DIR}/src/mapping.cpp 
              ${PROJECT_SOURCE_DIR}/src/pose_feedback.cpp
              ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
              ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
              ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp 
              ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
              ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
              ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
add_dependencies(gs_mapping ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(gs_mapping 
                      ${catkin_LIBRARIES} 
                      ${OpenCV_LIBS} 
                      ${TORCH_LIBRARIES} 
                      ${YAML_CPP_LIBRARIES} 
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_LIB_DIR}/libcudart.so
                      ${TENSORRT_LIBRARY}
                      ${TENSORRT_PLUGIN_LIBRARY})

add_executable(erp_rasterizer_forward_test
              ${PROJECT_SOURCE_DIR}/tests/test_erp_rasterizer_forward.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)
target_link_libraries(erp_rasterizer_forward_test
                      ${TORCH_LIBRARIES}
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_LIB_DIR}/libcudart.so)

add_executable(mapping_params_test tests/test_mapping_params.cpp)
target_link_libraries(mapping_params_test
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${YAML_CPP_LIBRARIES})
