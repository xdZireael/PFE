cmake_minimum_required(VERSION 3.8)
project(my_slam)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)

target_link_libraries(
  Eigen3::Eigen
)

include_directories(slam_node PRIVATE
  include)

add_executable(slam_node src/slam_node.cpp)

ament_target_dependencies(slam_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros tf2_eigen)

install(TARGETS slam_node
  DESTINATION lib/${PROJECT_NAME})
ament_package()