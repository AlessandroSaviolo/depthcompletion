cmake_minimum_required(VERSION 3.6)
project(depthcompletion)

cmake_policy(SET CMP0057 NEW)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
set(TensorRT_DIR /usr/share/doc/tensorrt-8.5.2.2)
set(Torch_PATH $ENV{TORCH_PATH})
set(Torch_DIR "${Torch_PATH}/share/cmake/Torch")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(quadrotor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realsense_camera REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Torch REQUIRED)
find_library(NVINFER nvinfer)
find_library(NVINFER_PLUGIN nvinfer_plugin)

set(dependencies
  rclcpp
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  std_msgs
  sensor_msgs
  visualization_msgs
  Eigen3
  yaml-cpp
  tf2
  tf2_ros
  rclcpp_components
  cv_bridge
  OpenCV
  CUDA
  TensorRT
  pluginlib
  realsense_camera
  realsense2
  rosidl_default_generators
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DepthCompletion.msg"
  DEPENDENCIES std_msgs sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)

if (NOT CUDA_INCLUDE_DIRS)
  set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)
endif()
if (NOT CUDA_LIBRARIES)
  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib)
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${realsense2_INCLUDE_DIR}
  ${TensorRT_INCLUDE_DIRS}
  ${realsense2_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/../realsense-cpp/realsense_camera/include
  ${Torch_PATH}/include
  ${Torch_PATH}/include/torch/csrc/api/include
)

add_library(depth_completion SHARED src/nodelet.cpp)
set_target_properties(depth_completion PROPERTIES COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL")
ament_target_dependencies(depth_completion ${dependencies})
rosidl_target_interfaces(depth_completion ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(depth_completion 
  ${rclcpp_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TensorRT_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${NVINFER}
  ${NVINFER_PLUGIN}
  realsense2::realsense2
)

rclcpp_components_register_nodes(depth_completion "depthcompletion::DepthCompletionNodelet")
ament_export_dependencies(${dependencies})

install(TARGETS
  depth_completion
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch/)
install(DIRECTORY include/ DESTINATION share/${PROJECT_NAME}/include/)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()