cmake_minimum_required(VERSION 3.5)
project(pointcloud_processing)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_ros REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_geometry REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_library(PointCloudProcessing
  src/PointCloudProcessing.cpp
)
ament_target_dependencies(PointCloudProcessing
  sensor_msgs
  pcl_conversions
  pcl_ros
  OpenCV
  image_geometry
  cv_bridge
  Eigen3
)
target_include_directories(PointCloudProcessing PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

add_library(PointCloudProcessingNode
  src/PointCloudProcessingNode.cpp
)
ament_target_dependencies(PointCloudProcessingNode
  rclcpp
  rclcpp_components
)
target_link_libraries(PointCloudProcessingNode PointCloudProcessing)
target_include_directories(PointCloudProcessingNode PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

rclcpp_components_register_nodes(PointCloudProcessingNode
  "pointcloud_processing::PointCloudProcessingNode")

add_executable(pointcloud_processing_node
  src/pointcloud_processing.cpp
)

target_link_libraries(pointcloud_processing_node
  PointCloudProcessing
  PointCloudProcessingNode
)

install(DIRECTORY include
  DESTINATION include/
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS PointCloudProcessingNode
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install the executable
install(TARGETS
  pointcloud_processing_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
