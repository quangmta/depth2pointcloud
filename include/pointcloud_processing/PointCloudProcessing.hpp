#ifndef POINTCLOUD_PROCESSING
#define POINTCLOUD_PROCESSING

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_geometry/pinhole_camera_model.h>

namespace pointcloud_processing
{
  class PointCloudProcessing final
  {
  public:
    explicit PointCloudProcessing(
        const float tf_x,
        const float tf_y,
        const float tf_z,
        const float tf_roll,
        const float tf_pitch,
        const float tf_yaw);
    ~PointCloudProcessing();

    sensor_msgs::msg::PointCloud2::SharedPtr create_pc(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_image_msg,
                                                       const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg);
    sensor_msgs::msg::Image::SharedPtr AlignDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                       const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg,
                                                       const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg,
                                                       const sensor_msgs::msg::LaserScan::UniquePtr &scan_from_depth_msg);
    sensor_msgs::msg::Image::SharedPtr SimpleAlignDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg,
                                                             const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg,
                                                             const sensor_msgs::msg::LaserScan::UniquePtr &scan_from_depth_msg);
    sensor_msgs::msg::LaserScan::SharedPtr AlignLaserScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg);

  private:
    double angle_between_rays(const cv::Point3d &ray1, const cv::Point3d &ray2) const;
    bool use_point(const float new_value, const float old_value, const float range_min, const float range_max) const;
    double magnitude_of_ray(const cv::Point3d& ray) const;
    std::vector<double> FitPolynomial(const std::vector<double> &x,
                                      const std::vector<double> &y, int degree);
    double EvalPolynomial(const std::vector<double> &coeffs, double x);
    uint16_t EvalPolynomial(const std::vector<double> &coeffs, uint16_t x);    
    image_geometry::PinholeCameraModel cam_model_;
    float tf_x_;
    float tf_y_;
    float tf_z_;
    float tf_roll_;
    float tf_pitch_;
    float tf_yaw_;
  };
}
#endif