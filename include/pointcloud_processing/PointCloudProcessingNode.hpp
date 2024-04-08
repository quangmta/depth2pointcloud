#ifndef POINTCLOUD_PROCESSING_NODE
#define POINTCLOUD_PROCESSING_NODE

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <pointcloud_processing/PointCloudProcessing.hpp>

namespace pointcloud_processing
{
  class PointCloudProcessingNode final : public rclcpp::Node
  {
  public:
    explicit PointCloudProcessingNode(const rclcpp::NodeOptions &options);
    ~PointCloudProcessingNode();

  private:
    void ScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan);
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr image);
    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr camera_info);
    void PointCloudCallback(const sensor_msgs::msg::Image::SharedPtr depth_image);

    sensor_msgs::msg::LaserScan::SharedPtr scan_msg_;
    sensor_msgs::msg::Image::SharedPtr image_msg_;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr processed_point_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_original_sync_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_image_original_sync_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_from_depth_pub_;

    std::shared_ptr<pointcloud_processing::PointCloudProcessing> pointCloud_;
  };
}
#endif