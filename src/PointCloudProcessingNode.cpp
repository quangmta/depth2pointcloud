#include <pointcloud_processing/PointCloudProcessingNode.hpp>

#include <rcutils/logging_macros.h>
#include <rclcpp_components/register_node_macro.hpp>
#include <cmath>

namespace pointcloud_processing
{
    PointCloudProcessingNode::PointCloudProcessingNode(const rclcpp::NodeOptions &options) : rclcpp::Node("pointcloud_processing_node", options)
    {
        auto qos = rclcpp::SystemDefaultsQoS();

        // Subscribe to the scan (lidar) topic
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/laser", qos,
            std::bind(&PointCloudProcessingNode::ScanCallback, this, std::placeholders::_1));

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image", qos,
            std::bind(&PointCloudProcessingNode::imageCallback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", qos,
            std::bind(&PointCloudProcessingNode::infoCallback, this, std::placeholders::_1));

        depth_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/depth", qos,
            std::bind(&PointCloudProcessingNode::PointCloudCallback, this, std::placeholders::_1));

        // Advertise the output point cloud topic
        scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/processed_scan", qos);
        point_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/original_points", qos);
        processed_point_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_points", qos);
        depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/processed_depth_image", qos);
        depth_image_original_sync_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/original_sync_depth_image", qos);
        rgb_image_original_sync_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/original_sync_rgb_image", qos);
        scan_from_depth_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan_from_depth", qos);

        // float tf_x, tf_y, tf_z, tf_roll, tf_pitch, tf_yaw;
        // int cam_width, cam_height;
        this->declare_parameter("tf_x", 0.0);
        this->declare_parameter("tf_y", 0.0);
        this->declare_parameter("tf_z", 0.0);
        this->declare_parameter("tf_roll", 0.0);
        this->declare_parameter("tf_pitch", 0.0);
        this->declare_parameter("tf_yaw", 0.0);

        auto tf_x = this->get_parameter("tf_x").as_double();
        auto tf_y = this->get_parameter("tf_y").as_double();
        auto tf_z = this->get_parameter("tf_z").as_double();
        auto tf_roll = this->get_parameter("tf_roll").as_double();
        auto tf_pitch = this->get_parameter("tf_pitch").as_double();
        auto tf_yaw =  this->get_parameter("tf_yaw").as_double();

        std::string output_frame = this->declare_parameter("output_frame", "/processed_points");

        pointCloud_ = std::make_shared<pointcloud_processing::PointCloudProcessing>(tf_x, tf_y, tf_z, tf_roll, tf_pitch, tf_yaw);
    }

    PointCloudProcessingNode::~PointCloudProcessingNode() {}

    void PointCloudProcessingNode::ScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        scan_msg_ = scan;
    }

    void PointCloudProcessingNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr image)
    {
        image_msg_ = image;
    }

    void PointCloudProcessingNode::infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr camera_info)
    {
        camera_info_msg_ = camera_info;
    }

    void PointCloudProcessingNode::PointCloudCallback(const sensor_msgs::msg::Image::SharedPtr depth_image)
    {
        bool flag_return = 0;
        if (nullptr == scan_msg_)
        {
            RCLCPP_INFO(get_logger(), "No laser scan, skipping point cloud processing");
            flag_return = 1;
        }
        if (nullptr == image_msg_)
        {
            RCLCPP_INFO(get_logger(), "No image, skipping point cloud processing");
            flag_return = 1;
        }
        if (nullptr == camera_info_msg_)
        {
            RCLCPP_INFO(get_logger(), "No camera info, skipping point cloud processing");
            flag_return = 1;
        }

        if (nullptr == depth_image)
        {
            RCLCPP_INFO(get_logger(), "No depth camera info, skipping point cloud processing");
            flag_return = 1;
        }

        if (flag_return)
            return;

        if (image_msg_->height != depth_image->height || image_msg_->width != depth_image->width)
        {
            RCLCPP_INFO(get_logger(), "Difference size of image, skipping point cloud processing");
            return;
        }

        depth_image_original_sync_pub_->publish(*depth_image);
        rgb_image_original_sync_pub_->publish(*image_msg_);

        try
        {
            // Fill in laserscan message
            sensor_msgs::msg::LaserScan::UniquePtr scan_from_depth_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
            scan_from_depth_msg->header = depth_image->header;
            scan_from_depth_msg->header.frame_id = "camera_color_frame";
            // float cam_fov_hor_half_ = std::atan(camera_info_msg_->width / (2 * camera_info_msg_->k[0]));
            // scan_from_depth_msg->angle_min = -cam_fov_hor_half_;
            // scan_from_depth_msg->angle_max = +cam_fov_hor_half_;
            // scan_from_depth_msg->angle_increment = (scan_from_depth_msg->angle_max - scan_from_depth_msg->angle_min) / (depth_image->width - 1);
            scan_from_depth_msg->time_increment = 0.0;
            scan_from_depth_msg->scan_time = scan_msg_->scan_time;
            scan_from_depth_msg->range_min = 0.1;
            scan_from_depth_msg->range_max = 5.6;            

            // RCLCPP_INFO(get_logger(), "Processing started!");
            auto original_pointcloud_msg_ = pointCloud_->create_pc(image_msg_, depth_image, camera_info_msg_);
            auto processed_scan_msg_ = pointCloud_->AlignLaserScan(scan_msg_);
            auto processed_depth_image_msg_ = pointCloud_->SimpleAlignDepthImage(depth_image, camera_info_msg_, processed_scan_msg_, scan_from_depth_msg);
            auto processed_pointcloud_msg_ = pointCloud_->create_pc(image_msg_, processed_depth_image_msg_, camera_info_msg_);

            image_msg_->header.stamp = depth_image->header.stamp;

            point_pub_->publish(*original_pointcloud_msg_);
            scan_pub_->publish(*processed_scan_msg_);
            depth_image_pub_->publish(*processed_depth_image_msg_);            
            scan_from_depth_pub_->publish(*scan_from_depth_msg);
            processed_point_pub_->publish(*processed_pointcloud_msg_);
            RCLCPP_INFO(get_logger(), "Processing completed!");
        }
        catch (const std::runtime_error &e)
        {
            RCLCPP_ERROR(get_logger(), "Could not convert depth image to point cloud: %s", e.what());
        }
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_processing::PointCloudProcessingNode)