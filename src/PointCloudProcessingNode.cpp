#include <pointcloud_processing/PointCloudProcessingNode.hpp>

#include <rcutils/logging_macros.h>
#include <rclcpp_components/register_node_macro.hpp>

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

        float tf_x, tf_y, tf_z, tf_roll, tf_pitch, tf_yaw;
        //int cam_width, cam_height;
        this->declare_parameter("tf_x", -0.04);
        this->declare_parameter("tf_y", 0.0);
        this->declare_parameter("tf_z", 0.02);
        this->declare_parameter("tf_roll", 0.0);
        this->declare_parameter("tf_pitch", 0.0);
        this->declare_parameter("tf_yaw", 0.0);

        this->get_parameter("tf_x", tf_x);
        this->get_parameter("tf_y", tf_y);
        this->get_parameter("tf_z", tf_z);
        this->get_parameter("tf_roll", tf_roll);
        this->get_parameter("tf_pitch", tf_pitch);
        this->get_parameter("tf_yaw", tf_yaw);

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
        if (nullptr == camera_info_msg_)
        {
            RCLCPP_INFO(get_logger(), "No laser scan info, skipping point cloud processing");
            return;
        }

        try
        {
            auto original_pointcloud_msg_ = pointCloud_->create_pc(image_msg_, depth_image, camera_info_msg_);
            auto processed_scan_msg_ = pointCloud_->AlignLaserScan(scan_msg_);
            auto processed_depth_image_msg_ = pointCloud_->AlignDepthImage(depth_image,camera_info_msg_,processed_scan_msg_);
            auto processed_pointcloud_msg_ = pointCloud_->create_pc(image_msg_,processed_depth_image_msg_,camera_info_msg_);
            
            point_pub_->publish(*original_pointcloud_msg_);
            scan_pub_->publish(*processed_scan_msg_);
            depth_image_pub_->publish(*processed_depth_image_msg_);
            processed_point_pub_->publish(*processed_pointcloud_msg_);
        }
        catch (const std::runtime_error &e)
        {
            RCLCPP_ERROR(get_logger(), "Could not convert depth image to laserscan: %s", e.what());
        }
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_processing::PointCloudProcessingNode)