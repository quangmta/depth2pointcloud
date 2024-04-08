#include <pointcloud_processing/PointCloudProcessing.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rclcpp/rclcpp.hpp>
#include <rcutils/logging_macros.h>
// #include <rclcpp_components/register_node_macro.hpp>
#include <eigen3/Eigen/Dense>

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

    double PointCloudProcessing::angle_between_rays(const cv::Point3d &ray1, const cv::Point3d &ray2) const
    {
        double dot_product = ray1.x * ray2.x + ray1.y * ray2.y + ray1.z * ray2.z;
        double magnitude1 = magnitude_of_ray(ray1);
        double magnitude2 = magnitude_of_ray(ray2);
        return std::acos(dot_product / (magnitude1 * magnitude2));
    }
    double PointCloudProcessing::magnitude_of_ray(const cv::Point3d &ray) const
    {
        return std::sqrt(std::pow(ray.x, 2.0) + std::pow(ray.y, 2.0) + std::pow(ray.z, 2.0));
    }

    bool PointCloudProcessing::use_point(const float new_value, const float old_value, const float range_min, const float range_max) const
    {
        // Check for NaNs and Infs, a real number within our limits is more desirable than these.
        bool new_finite = std::isfinite(new_value);
        bool old_finite = std::isfinite(old_value);

        // Infs are preferable over NaNs (more information)
        if (!new_finite && !old_finite)
        {                                  // Both are not NaN or Inf.
            return !std::isnan(new_value); // new is not NaN, so use it's +-Inf value.
        }

        // If not in range, don't bother
        bool range_check = range_min <= new_value && new_value <= range_max;
        if (!range_check)
        {
            return false;
        }

        if (!old_finite)
        { // New value is in range and finite, use it.
            return true;
        }

        // Finally, if they are both numerical and new_value is closer than old_value, use new_value.
        bool shorter_check = new_value < old_value;
        return shorter_check;
    }

    std::vector<double> PointCloudProcessing::FitPolynomial(const std::vector<double> &x,
                                                            const std::vector<double> &y, int degree)
    {
        assert(x.size() == y.size());
        // This maps the doubles in x to something that behaves like a const
        // Eigen::VectorXd object so we can use Eigen functionality on the underlying
        // data.
        auto xvec = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());

        // Create a matrix with one row for each observation where xs(i, j) = x^j.
        // For example, if x = {1, 2, 3, 4}, then:
        // xs =
        //   1  1  1  1
        //   1  2  4  8
        //   1  3  9 27
        //   1  4 16 64
        Eigen::MatrixXd xs(x.size(), degree + 1);
        xs.col(0).setOnes();
        // xs.col(0).array() = xvec.array();
        for (int i = 1; i <= degree; ++i)
        {
            xs.col(i).array() = xs.col(i - 1).array() * xvec.array();
        }
        // xs.col(0).setZero();

        // Map y to an object ys that behaves like an Eigen::VectorXd.
        auto ys = Eigen::Map<const Eigen::VectorXd>(y.data(), y.size());

        std::vector<double> result(degree + 1);
        // Again we use Eigen::Map to enable treating a std::vector<double> as an
        // Eigen object (this time a non-const one since we need to write to it).
        auto result_map = Eigen::Map<Eigen::VectorXd>(result.data(), result.size());

        // Compute a decomposition of the matrix xs; in this case we are using a QR
        // decomposition computed via Householder reflections. There are other
        // decompositions that can be used here as well with differing accuracy and
        // performance characteristics. For a list, see
        // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
        //  and
        // https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html
        // Note that this step is expensive and if you may ever be trying to set the
        // same set of x values to different y values, then you would want to compute
        // this value once and reuse it. This would be the case if, for example, you
        // always sample some real world data at the same x points and want to fit
        // a polynomial based on those xs and the observed ys.
        auto decomposition = xs.householderQr();
        result_map = decomposition.solve(ys);
        return result;
    }

    double PointCloudProcessing::EvalPolynomial(const std::vector<double> &coeffs, double x)
    {
        double result = 0;
        double xp = 1;
        for (auto &c : coeffs)
        {
            result += xp * c;
            xp *= x;
        }
        return result;
    }

    uint16_t PointCloudProcessing::EvalPolynomial(const std::vector<double> &coeffs, uint16_t x)
    {
        float x_float = x * 0.001;
        float result = 0;
        float xp = 1;
        for (auto &c : coeffs)
        {
            result += xp * c;
            xp *= x_float;
        }
        result *= 1000;
        return (uint16_t)result;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr PointCloudProcessing::create_pc(const sensor_msgs::msg::Image::ConstSharedPtr &rgb_image_msg,
                                                                             const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg)
    {
        cam_model_.fromCameraInfo(cam_info_msg);
        cv::Mat rgb_image = cv_bridge::toCvShare(rgb_image_msg)->image;
        // cv::Mat depth_image = cv_bridge::toCvShare(depth_image_msg)->image;
        cv::Mat depth_image;
        if (depth_image_msg->encoding == "32FC1")
            depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_32FC1, const_cast<float *>(reinterpret_cast<const float *>(&depth_image_msg->data[0])), depth_image_msg->step);
        else if (depth_image_msg->encoding == "16UC1")
            depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_16UC1, const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(&depth_image_msg->data[0])), depth_image_msg->step);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        for (int i = 0; i < depth_image.rows; ++i)
        {
            for (int j = 0; j < depth_image.cols; ++j)
            {
                float d;
                if (depth_image_msg->encoding == "32FC1")
                    d = static_cast<double>(depth_image.at<float>(i, j));
                else if (depth_image_msg->encoding == "16UC1")
                    d = static_cast<double>(depth_image.at<uint16_t>(i, j)) * 0.001;

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
                std::uint32_t rgb = (rgb_pixel[0] << 16) | (rgb_pixel[1] << 8) | rgb_pixel[2];
                pt.rgb = *reinterpret_cast<float *>(&rgb);
                // pt.rgb = *reinterpret_cast<float *>(&rgb_pixel);

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
        // Calculate angle_min and angle_max by measuring angles between the left ray, right ray, and optical center ray
        cv::Point2d raw_pixel_left(0, cam_model_.cy());
        cv::Point2d rect_pixel_left = cam_model_.rectifyPoint(raw_pixel_left);
        cv::Point3d left_ray = cam_model_.projectPixelTo3dRay(rect_pixel_left);

        cv::Point2d raw_pixel_right(cam_model_.fullResolution().width - 1, cam_model_.cy());
        cv::Point2d rect_pixel_right = cam_model_.rectifyPoint(raw_pixel_right);
        cv::Point3d right_ray = cam_model_.projectPixelTo3dRay(rect_pixel_right);

        cv::Point2d raw_pixel_center(cam_model_.cx(), cam_model_.cy());
        cv::Point2d rect_pixel_center = cam_model_.rectifyPoint(raw_pixel_center);
        cv::Point3d center_ray = cam_model_.projectPixelTo3dRay(rect_pixel_center);

        double cam_angle_max_l = angle_between_rays(left_ray, center_ray);
        double cam_angle_max_r = angle_between_rays(center_ray, right_ray);

        int step_l = cam_angle_max_l / scan_msg->angle_increment;
        int step_r = cam_angle_max_r / scan_msg->angle_increment;
        cam_angle_max_l = step_l * scan_msg->angle_increment;
        cam_angle_max_r = step_r * scan_msg->angle_increment;

        // float cam_angle_max_l = std::atan(cam_model_.cx() / cam_model_.fx());
        // float cam_angle_max_r = std::atan((cam_model_.fullResolution().width - cam_model_.cx()) / cam_model_.fx());
        size_t begin_point_scan_index = (scan_msg->ranges.size() - 1) / 2 - step_r;
        size_t end_point_scan_index = (scan_msg->ranges.size() - 1) / 2 + step_l;
        float eps = 1e-3;

        if (begin_point_scan_index > 0)
        {
            float coorY_l = scan_msg->ranges[end_point_scan_index] * std::sin(cam_angle_max_l);
            float coorY_r = scan_msg->ranges[begin_point_scan_index] * std::sin(cam_angle_max_r);
            if (tf_y_ < 0)
            {
                while (std::fabs(tf_y_) - eps > scan_msg->ranges[begin_point_scan_index] * std::sin(cam_angle_max_r) - coorY_r)
                {
                    begin_point_scan_index--;
                    cam_angle_max_r += scan_msg->angle_increment;
                }
                while (std::fabs(tf_y_) - eps > coorY_l - scan_msg->ranges[end_point_scan_index] * std::sin(cam_angle_max_l))
                {
                    end_point_scan_index--;
                    cam_angle_max_l -= scan_msg->angle_increment;
                }
            }
            else
            {
                while (std::fabs(tf_y_) - eps > coorY_r - scan_msg->ranges[begin_point_scan_index] * std::sin(cam_angle_max_r))
                {
                    begin_point_scan_index++;
                    cam_angle_max_r -= scan_msg->angle_increment;
                }
                while (std::fabs(tf_y_) - eps > scan_msg->ranges[end_point_scan_index] * std::sin(cam_angle_max_l) - coorY_l)
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
                                      scan_msg->ranges.begin() + end_point_scan_index + 1);
        return align_scan_msg;
    }

    sensor_msgs::msg::Image::SharedPtr PointCloudProcessing::AlignDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                                             const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg,
                                                                             const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg,
                                                                             const sensor_msgs::msg::LaserScan::UniquePtr &scan_from_depth_msg)
    {
        cam_model_.fromCameraInfo(cam_info_msg);
        // float cam_fov_hor_ = 2 * std::atan(cam_model_.fullResolution().width / (2 * cam_model_.fx()));
        // cv::Mat depth_image = cv_bridge::toCvShare(depth_image_msg)->image;
        cv::Mat depth_image;
        if (depth_image_msg->encoding == "32FC1")
            depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_32FC1, const_cast<float *>(reinterpret_cast<const float *>(&depth_image_msg->data[0])), depth_image_msg->step);
        else if (depth_image_msg->encoding == "16UC1")
            depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_16UC1, const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(&depth_image_msg->data[0])), depth_image_msg->step);
        float coeff = 1, coeff_last = 1;
        std::ofstream csv_file("scan_data.csv");
        csv_file << scan_msg->ranges.size() << "\t" << depth_image.cols << std::endl;
        for (int col = 0; col < depth_image.cols; col++)
        {
            double d;
            if (depth_image_msg->encoding == "32FC1")
                d = static_cast<double>(depth_image.at<float>(cam_model_.cy(), col));
            else if (depth_image_msg->encoding == "16UC1")
                d = static_cast<double>(depth_image.at<uint16_t>(cam_model_.cy(), col)) * 0.001;
            long unsigned int indx = std::round(((scan_msg->ranges.size() - 1) * (depth_image.cols - col) / depth_image.cols));
            float d_real = 0;
            float range = scan_msg->ranges[indx];

            int row_offset;
            row_offset = cam_model_.cy() - tf_z_ * cam_model_.fy() / d;
            if (row_offset >= 0 && row_offset < depth_image.rows)
            {
                if (depth_image_msg->encoding == "32FC1")
                    d = static_cast<double>(depth_image.at<float>(row_offset, col));
                else if (depth_image_msg->encoding == "16UC1")
                    d = static_cast<double>(depth_image.at<uint16_t>(row_offset, col)) * 0.001;

                // Determine if this point should be used.
                if (use_point(d, scan_msg->ranges[depth_image.cols - col], scan_msg->range_min, scan_msg->range_max))
                {
                    scan_from_depth_msg->ranges[depth_image.cols - col] = d;
                }
            }

            if (!std::isnan(range))
            {
                d_real = range * std::cos(scan_msg->angle_min + indx * scan_msg->angle_increment) + tf_x_;
                row_offset = cam_model_.cy() - tf_z_ * cam_model_.fy() / d_real;
                if (row_offset >= 0 && row_offset < depth_image.rows)
                {
                    if (depth_image_msg->encoding == "32FC1")
                        d = static_cast<double>(depth_image.at<float>(row_offset, col));
                    else if (depth_image_msg->encoding == "16UC1")
                        d = static_cast<double>(depth_image.at<uint16_t>(row_offset, col)) * 0.001;

                    // Determine if this point should be used.
                    if (use_point(d, scan_msg->ranges[depth_image.cols - col], scan_msg->range_min, scan_msg->range_max))
                    {
                        scan_from_depth_msg->ranges[depth_image.cols - col] = d;
                    }
                    if (d < 0.005 || d_real <= 0.005)
                    {
                        coeff = coeff_last;
                    }
                    else
                    {
                        coeff = d_real / d;
                        coeff_last = coeff;
                    }
                }
            }
            else
            {
                coeff = coeff_last;
            }
            csv_file << col << "\t" << indx << "\t" << range << "\t" << row_offset << "\t" << d_real << "\t" << d << "\t" << coeff;

            for (int row = 0; row < depth_image.rows; row++)
            {
                if (depth_image_msg->encoding == "32FC1")
                    depth_image.at<float>(row, col) *= coeff;
                else if (depth_image_msg->encoding == "16UC1")
                    depth_image.at<uint16_t>(row, col) *= coeff;
            }
            csv_file << std::endl;
        }
        csv_file << "end";
        csv_file.close();

        auto depth_image_msg_copy = std::make_shared<sensor_msgs::msg::Image>(*depth_image_msg);
        memcpy(depth_image_msg_copy->data.data(), depth_image.data, depth_image_msg->data.size());
        return depth_image_msg_copy;
    }

    sensor_msgs::msg::Image::SharedPtr PointCloudProcessing::SimpleAlignDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
                                                                                   const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg,
                                                                                   const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg,
                                                                                   const sensor_msgs::msg::LaserScan::UniquePtr &scan_from_depth_msg)
    {
        cam_model_.fromCameraInfo(cam_info_msg);
        // float cam_fov_hor_ = 2 * std::atan(cam_model_.fullResolution().width / (2 * cam_model_.fx()));
        // cv::Mat depth_image = cv_bridge::toCvShare(depth_image_msg)->image;
        cv::Mat depth_image;
        if (depth_image_msg->encoding == "32FC1")
            depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_32FC1, const_cast<float *>(reinterpret_cast<const float *>(&depth_image_msg->data[0])), depth_image_msg->step);
        else if (depth_image_msg->encoding == "16UC1")
            depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_16UC1, const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(&depth_image_msg->data[0])), depth_image_msg->step);

        cv::Point2d raw_pixel_left(0, cam_model_.cy());
        cv::Point2d rect_pixel_left = cam_model_.rectifyPoint(raw_pixel_left);
        cv::Point3d left_ray = cam_model_.projectPixelTo3dRay(rect_pixel_left);

        // cv::Point2d raw_pixel_right(cam_model_.fullResolution().width - 1, cam_model_.cy());
        // cv::Point2d rect_pixel_right = cam_model_.rectifyPoint(raw_pixel_right);
        // cv::Point3d right_ray = cam_model_.projectPixelTo3dRay(rect_pixel_right);

        cv::Point2d raw_pixel_center(cam_model_.cx(), cam_model_.cy());
        cv::Point2d rect_pixel_center = cam_model_.rectifyPoint(raw_pixel_center);
        cv::Point3d center_ray = cam_model_.projectPixelTo3dRay(rect_pixel_center);

        double cam_angle_max_l = angle_between_rays(left_ray, center_ray);
        // double cam_angle_max_r = angle_between_rays(center_ray, right_ray);

        double sum_num = 0, sum_den = 0;
        std::vector<double> x, y;

        std::ofstream csv_file("align_data.csv");
        csv_file << scan_msg->ranges.size() << "\t" << depth_image.cols << std::endl;

        for (int indx = 0; indx < (int)scan_msg->ranges.size(); indx++)
        {
            float range = scan_msg->ranges[indx];
            if (!std::isnan(range))
            {
                float angle = scan_msg->angle_min + indx * scan_msg->angle_increment;
                float d_real = range * std::cos(angle) + tf_x_;
                int row_offset = cam_model_.cy() - tf_z_ * cam_model_.fy() / d_real;
                if (row_offset >= 0 && row_offset < depth_image.rows)
                {
                    int index_col = cam_model_.cx() - cam_model_.cx() / std::tan(cam_angle_max_l) * std::tan(angle);
                    float d;
                    if (depth_image_msg->encoding == "32FC1")
                        d = static_cast<double>(depth_image.at<float>(row_offset, index_col));
                    else if (depth_image_msg->encoding == "16UC1")
                        d = static_cast<double>(depth_image.at<uint16_t>(row_offset, index_col)) * 0.001;

                    scan_from_depth_msg->ranges[depth_image.cols - index_col] = d / std::cos(scan_msg->angle_min + index_col * scan_msg->angle_increment);

                    // Determine if this point should be used.
                    // if (use_point(d, scan_msg->ranges[depth_image.cols - col], scan_msg->range_min, scan_msg->range_max))
                    // {
                    //     scan_from_depth_msg->ranges[depth_image.cols - col] = d;
                    // }

                    if (d > 0.005 && d_real > 0.005)
                    {
                        x.push_back(d);
                        y.push_back(d_real);
                        sum_num += d * d_real;
                        sum_den += d * d;
                    }
                    csv_file << indx << "\t" << index_col << "\t" << row_offset << "\t" << range << "\t" << d_real << "\t" << d << "\t" << d_real / d << std::endl;
                }
            }
        }
        csv_file.close();

        std::ofstream csv_result_file("result.csv");
        csv_result_file << "Number of input:  " << x.size() << "\n";

        std::vector<double> polynom = FitPolynomial(x, y, 1);
        csv_result_file << "a:\t" << polynom[1] << "\tb:\t" << polynom[0] << std::endl;

        float coeff_k = sum_num / sum_den;

        csv_result_file << "coeff k\t" << coeff_k << std::endl;

        sum_num = 0;
        sum_den = 0;

        for (int i = 0; i < (int)x.size(); i++)
        {
            if (std::abs((y[i] - x[i] * coeff_k) / x[i]) > 0.1)
            {
                csv_result_file << x[i] << "\t" << y[i] << "\t" << EvalPolynomial(polynom, x[i]) << "\t" << x[i] * coeff_k << "\t ignored" << std::endl;
                x.erase(x.begin() + i);
                y.erase(y.begin() + i);
                i--;
            }
            else
            {
                sum_num += x[i] * y[i];
                sum_den += x[i] * x[i];
                csv_result_file << x[i] << "\t" << y[i] << "\t" << EvalPolynomial(polynom, x[i]) << "\t" << x[i] * coeff_k << std::endl;
            }
        }
        coeff_k = sum_num / sum_den;

        csv_result_file << "coeff k\t" << coeff_k << std::endl;

        csv_result_file.close();

        for (int col = 0; col < depth_image.cols; col++)
            for (int row = 0; row < depth_image.rows; row++)
            {
                if (depth_image_msg->encoding == "32FC1")
                    // depth_image.at<float>(row, col) = EvalPolynomial(polynom, depth_image.at<float>(row, col));
                    depth_image.at<float>(row, col) = depth_image.at<float>(row, col) * coeff_k;
                else if (depth_image_msg->encoding == "16UC1")
                    depth_image.at<uint16_t>(row, col) = depth_image.at<uint16_t>(row, col) * coeff_k;
            }

        auto depth_image_msg_copy = std::make_shared<sensor_msgs::msg::Image>(*depth_image_msg);
        memcpy(depth_image_msg_copy->data.data(), depth_image.data, depth_image_msg->data.size());
        return depth_image_msg_copy;
    }

    // sensor_msgs::msg::Image::SharedPtr PointCloudProcessing::SimpleAlignDepthImage(const sensor_msgs::msg::Image::ConstSharedPtr &depth_image_msg,
    //                                                                                const sensor_msgs::msg::CameraInfo::ConstSharedPtr &cam_info_msg,
    //                                                                                const sensor_msgs::msg::LaserScan::ConstSharedPtr &scan_msg,
    //                                                                                const sensor_msgs::msg::LaserScan::UniquePtr &scan_from_depth_msg)
    // {
    //     cam_model_.fromCameraInfo(cam_info_msg);
    //     // float cam_fov_hor_ = 2 * std::atan(cam_model_.fullResolution().width / (2 * cam_model_.fx()));
    //     // cv::Mat depth_image = cv_bridge::toCvShare(depth_image_msg)->image;
    //     cv::Mat depth_image;
    //     if (depth_image_msg->encoding == "32FC1")
    //         depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_32FC1, const_cast<float *>(reinterpret_cast<const float *>(&depth_image_msg->data[0])), depth_image_msg->step);
    //     else if (depth_image_msg->encoding == "16UC1")
    //         depth_image = cv::Mat(depth_image_msg->height, depth_image_msg->width, CV_16UC1, const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(&depth_image_msg->data[0])), depth_image_msg->step);
    //     float coeff = 1, coeff_last = 1;
    //     double sum_num = 0, sum_den = 0;

    //     std::vector<double> x, y;

    //     std::ofstream csv_file("align_data.csv");
    //     csv_file << scan_msg->ranges.size() << "\t" << depth_image.cols << std::endl;
    //     for (int col = 0; col < depth_image.cols; col++)
    //     {
    //         double d;
    //         if (depth_image_msg->encoding == "32FC1")
    //             d = static_cast<double>(depth_image.at<float>(cam_model_.cy(), col));
    //         else if (depth_image_msg->encoding == "16UC1")
    //             d = static_cast<double>(depth_image.at<uint16_t>(cam_model_.cy(), col)) * 0.001;
    //         long unsigned int indx = std::round(((scan_msg->ranges.size() - 1) * (depth_image.cols - col) / depth_image.cols));
    //         float d_real = 0;
    //         float range = scan_msg->ranges[indx];

    //         int row_offset;
    //         row_offset = cam_model_.cy() - tf_z_ * cam_model_.fy() / d;
    //         if (row_offset >= 0 && row_offset < depth_image.rows)
    //         {
    //             if (depth_image_msg->encoding == "32FC1")
    //                 d = static_cast<double>(depth_image.at<float>(row_offset, col));
    //             else if (depth_image_msg->encoding == "16UC1")
    //                 d = static_cast<double>(depth_image.at<uint16_t>(row_offset, col)) * 0.001;

    //             scan_from_depth_msg->ranges[depth_image.cols - col] = d;
    //             // Determine if this point should be used.
    //             // if (use_point(d, scan_msg->ranges[depth_image.cols - col], scan_msg->range_min, scan_msg->range_max))
    //             // {
    //             //     scan_from_depth_msg->ranges[depth_image.cols - col] = d;
    //             // }
    //         }

    //         if (!std::isnan(range))
    //         {
    //             d_real = range * std::cos(scan_msg->angle_min + indx * scan_msg->angle_increment) + tf_x_;
    //             row_offset = cam_model_.cy() - tf_z_ * cam_model_.fy() / d_real;
    //             if (row_offset >= 0 && row_offset < depth_image.rows)
    //             {
    //                 if (depth_image_msg->encoding == "32FC1")
    //                     d = static_cast<double>(depth_image.at<float>(row_offset, col));
    //                 else if (depth_image_msg->encoding == "16UC1")
    //                     d = static_cast<double>(depth_image.at<uint16_t>(row_offset, col)) * 0.001;

    //                 scan_from_depth_msg->ranges[depth_image.cols - col] = d / std::cos(scan_msg->angle_min + indx * scan_msg->angle_increment);

    //                 // Determine if this point should be used.
    //                 // if (use_point(d, scan_msg->ranges[depth_image.cols - col], scan_msg->range_min, scan_msg->range_max))
    //                 // {
    //                 //     scan_from_depth_msg->ranges[depth_image.cols - col] = d;
    //                 // }

    //                 if (d < 0.005 || d_real <= 0.005)
    //                 {
    //                     coeff = coeff_last;
    //                 }
    //                 else
    //                 {
    //                     x.push_back(d);
    //                     y.push_back(d_real);
    //                     sum_num += d * d_real;
    //                     sum_den += d * d;
    //                     coeff = d_real / d;
    //                     coeff_last = coeff;
    //                 }
    //             }
    //         }
    //         else
    //         {
    //             coeff = coeff_last;
    //         }
    //         csv_file << col << "\t" << indx << "\t" << row_offset << "\t" << range << "\t" << d_real << "\t" << d << "\t" << coeff << std::endl;
    //     }

    //     csv_file.close();

    //     std::ofstream csv_result_file("result.csv");
    //     csv_result_file << "Number of input:  " << x.size() << "\n";

    //     std::vector<double> polynom = FitPolynomial(x, y, 1);
    //     csv_result_file << "a:\t" << polynom[1] << "\tb:\t" << polynom[0] << std::endl;

    //     float coeff_k = sum_num / sum_den;

    //     csv_result_file << "coeff k\t" << coeff_k << std::endl;

    //     sum_num = 0;
    //     sum_den = 0;

    //     for (int i = 0; i < (int)x.size(); i++)
    //     {
    //         if (std::abs((y[i] - x[i] * coeff_k) / x[i]) > 0.1)
    //         {
    //             csv_result_file << x[i] << "\t" << y[i] << "\t" << EvalPolynomial(polynom, x[i]) << "\t" << x[i] * coeff_k << "\t ignored" << std::endl;
    //             x.erase(x.begin() + i);
    //             y.erase(y.begin() + i);
    //             i--;
    //         }
    //         else
    //         {
    //             sum_num += x[i] * y[i];
    //             sum_den += x[i] * x[i];
    //             csv_result_file << x[i] << "\t" << y[i] << "\t" << EvalPolynomial(polynom, x[i]) << "\t" << x[i] * coeff_k << std::endl;
    //         }
    //     }
    //     coeff_k = sum_num / sum_den;

    //     csv_result_file << "coeff k\t" << coeff_k << std::endl;

    //     csv_result_file.close();

    //     for (int col = 0; col < depth_image.cols; col++)
    //         for (int row = 0; row < depth_image.rows; row++)
    //         {
    //             if (depth_image_msg->encoding == "32FC1")
    //                 // depth_image.at<float>(row, col) = EvalPolynomial(polynom, depth_image.at<float>(row, col));
    //                 depth_image.at<float>(row, col) = depth_image.at<float>(row, col) * coeff_k;
    //             else if (depth_image_msg->encoding == "16UC1")
    //                 depth_image.at<uint16_t>(row, col) = depth_image.at<uint16_t>(row, col) * coeff_k;
    //         }

    //     auto depth_image_msg_copy = std::make_shared<sensor_msgs::msg::Image>(*depth_image_msg);
    //     memcpy(depth_image_msg_copy->data.data(), depth_image.data, depth_image_msg->data.size());
    //     return depth_image_msg_copy;
    // }
}
