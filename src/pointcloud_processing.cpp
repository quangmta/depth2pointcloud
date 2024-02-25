#include <pointcloud_processing/PointCloudProcessingNode.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pointcloud_processing::PointCloudProcessingNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
