#include <pointcloud_processing/PointCloudProcessing.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rclcpp/rclcpp.hpp>
#include <rcutils/logging_macros.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace pointcloud_processing
{
    PointCloudProcessing::PointCloudProcessing(
        const float tf_x,
        const float tf_y,
        const float tf_z,
        const float tf_roll,
        const float tf_pitch,
        const float tf_yaw)
        : tf_x_(tf_x),
          tf_y_(tf_y),
          tf_z_(tf_z),
          tf_roll_(tf_roll),
          tf_pitch_(tf_pitch),
          tf_yaw_(tf_yaw)
    {
    }

    PointCloudProcessing::~PointCloudProcessing() {}

    sensor_msgs::msg::PointCloud2::SharedPtr PointCloudProcessing::create_pc(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_image_msg,
                                                                             const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg)
    {
        cam_model_.fromCameraInfo(cam_info_msg);
        cv::Mat rgb_image = cv_bridge::toCvShare(rgb_image_msg)->image;
        cv::Mat depth_image = cv_bridge::toCvShare(depth_image_msg)->image;

        // cv::Mat depth_image(depth_image_msg->height, depth_image_msg->width, CV_32FC1, const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(&depth_image_msg->data[0])), depth_image_msg->step);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        for (int i = 0; i < depth_image.rows; ++i)
        {
            for (int j = 0; j < depth_image.cols; ++j)
            {
                float d = depth_image.at<float>(i, j);

                // Check for invalid measurements
                if (std::fabs(d) <= 1e-9)
                    continue;
                // if d has a value, add a point to the point cloud

                pcl::PointXYZRGB pt;

                // Convert pixel coordinates to 3D point
                // cv::Point3d point_3d = cam_model_.projectPixelTo3dRay(cv::Point2d(i, j)) * d;
                pt.z = d;
                pt.x = (j - cam_model_.cx()) * pt.z / cam_model_.fx();
                pt.y = (i - cam_model_.cy()) * pt.z / cam_model_.fy();

                cv::Vec3b rgb_pixel = rgb_image.at<cv::Vec3b>(i, j);
                // std::uint32_t rgb = (rgb_pixel[2] << 16) | (rgb_pixel[1] << 8) | rgb_pixel[0];
                pt.rgb = *reinterpret_cast<float *>(&rgb_pixel);

                // add p to the point cloud
                pcl_cloud->points.push_back(pt);
            }
        }
        pcl_cloud->height = 1;
        pcl_cloud->width = pcl_cloud->points.size();
        pcl_cloud->is_dense = false;

        // Create the filtering object
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(pcl_cloud);
        sor.setLeafSize(0.01f, 0.01f, 0.01f);
        sor.filter(*cloud_filtered);

        auto pointcloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
        pcl::toROSMsg(*cloud_filtered, *pointcloud_msg);
        pointcloud_msg->header.frame_id = cam_info_msg->header.frame_id;
        pointcloud_msg->header.stamp = rclcpp::Clock().now();

        pcl_cloud->points.clear();
        return pointcloud_msg;
    }

    sensor_msgs::msg::LaserScan::SharedPtr PointCloudProcessing::AlignLaserScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg)
    {
        float cam_fov_hor_ = 2 * std::atan(cam_model_.fullResolution().width / (2 * cam_model_.fx()));
        size_t begin_point_scan_index, end_point_scan_index;
        unsigned int step = std::round(cam_fov_hor_ / 2 / scan_msg->angle_increment);
        float cam_angle_max_l = step * scan_msg->angle_increment;
        float cam_angle_max_r = cam_angle_max_l;
        begin_point_scan_index = (scan_msg->ranges.size() - 1) / 2 - step;
        end_point_scan_index = (scan_msg->ranges.size() - 1) / 2 + step;
        float eps = 1e-9;

        if (begin_point_scan_index > 0)
        {
            float coorY_l = scan_msg->ranges[end_point_scan_index] * std::tan(cam_angle_max_l);
            float coorY_r = scan_msg->ranges[begin_point_scan_index] * std::tan(cam_angle_max_r);
            if (tf_y_ < 0)
            {
                while (std::fabs(tf_y_) - eps > scan_msg->ranges[begin_point_scan_index] * std::tan(cam_angle_max_r) - coorY_r)
                {
                    begin_point_scan_index--;
                    cam_angle_max_r += scan_msg->angle_increment;
                }
                while (std::fabs(tf_y_) - eps > coorY_l - scan_msg->ranges[end_point_scan_index] * std::tan(cam_angle_max_l))
                {
                    end_point_scan_index--;
                    cam_angle_max_l -= scan_msg->angle_increment;
                }
            }
            else
            {
                while (std::fabs(tf_y_) - eps > coorY_r - scan_msg->ranges[begin_point_scan_index] * std::tan(cam_angle_max_r))
                {
                    begin_point_scan_index++;
                    cam_angle_max_r -= scan_msg->angle_increment;
                }
                while (std::fabs(tf_y_) - eps > scan_msg->ranges[end_point_scan_index] * std::tan(cam_angle_max_l) - coorY_l)
                {
                    end_point_scan_index++;
                    cam_angle_max_l += scan_msg->angle_increment;
                }
            }
        }
        else
        {
            begin_point_scan_index = 0;
            end_point_scan_index = scan_msg->ranges.size() - 1;
        }

        auto align_scan_msg = std::make_shared<sensor_msgs::msg::LaserScan>();
        // Create new scan message with new size
        align_scan_msg->header = scan_msg->header;
        align_scan_msg->angle_min = -cam_angle_max_r;
        align_scan_msg->angle_max = cam_angle_max_l;
        align_scan_msg->angle_increment = scan_msg->angle_increment;
        align_scan_msg->time_increment = scan_msg->time_increment;
        align_scan_msg->scan_time = scan_msg->scan_time;
        align_scan_msg->range_min = scan_msg->range_min;
        align_scan_msg->range_max = scan_msg->range_max;

        align_scan_msg->ranges.assign(scan_msg->ranges.begin() + begin_point_scan_index,
                                      scan_msg->ranges.begin() + end_point_scan_index);
        return align_scan_msg;
    }

    sensor_msgs::msg::Image::SharedPtr PointCloudProcessing::AlignDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg,
                                                                             const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg)
    {
        cam_model_.fromCameraInfo(cam_info_msg);
        // float cam_fov_hor_ = 2 * std::atan(cam_model_.fullResolution().width / (2 * cam_model_.fx()));
        cv::Mat depth_image = cv_bridge::toCvShare(depth_image_msg)->image;
        float coeff = 1, coeff_last = 1;
        std::ofstream csv_file("scan_data.csv");
        csv_file<<scan_msg->ranges.size()<<std::endl;
        for (int col = 0; col < depth_image.cols; col++)
        {
            double d = (depth_image.at<float>(cam_model_.cy(), col));
            int row_offset = cam_model_.cy() - tf_z_ * cam_model_.fy() / d;
            d = depth_image.at<float>(row_offset, col);
            long unsigned int indx = std::round(((scan_msg->ranges.size() - 1) * (depth_image.cols - col) / depth_image.cols));
            float d_real=d;
            float range = scan_msg->ranges[indx];
            if (!std::isinf(range))
            {                
                d_real = range*std::cos(scan_msg->angle_min+indx*scan_msg->angle_increment)+tf_x_;
                coeff = d_real / d;
                coeff_last = coeff;
            }
            else
            {
                coeff = coeff_last;
            }
            csv_file << col << "\t" << indx << "\t" << range << "\t" << d_real << "\t" << d << "\t" << coeff << std::endl;

            for (int row = 0; row < depth_image.rows; row++)
            {
                depth_image.at<float>(row, col) *= coeff;
            }
        }
        csv_file.close();

        auto depth_image_msg_copy = std::make_shared<sensor_msgs::msg::Image>(*depth_image_msg);
        memcpy(depth_image_msg_copy->data.data(), depth_image.data, depth_image_msg->data.size());
        return depth_image_msg_copy;
    }
}