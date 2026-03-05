#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

using Point2D = Eigen::Vector2f;
using PointCloud = std::vector<Point2D>;

struct ICPResult
{
    Eigen::Matrix2f R;
    Eigen::Vector2f t;
    float error;
    bool converged;
};

class SLAMNode : public rclcpp::Node
{
public:
    SLAMNode() : Node("slam_node"), robot_pose_(Eigen::Vector3d::Zero())
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&SLAMNode::laserScanCallback, this, std::placeholders::_1));

        map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", 10);
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pose", 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
    // SLAM State
    Eigen::Vector3d robot_pose_; // (x, y, theta)
    nav_msgs::msg::OccupancyGrid map_;
    PointCloud prev_cloud_;

    // ROS Interfaces
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ICP Params
    static constexpr float ICP_MAX_ITER = 50;
    static constexpr float ICP_TOLERANCE = 1e-4;
    static constexpr float ICP_MAX_DIST = 0.5;
    static constexpr float ICP_MIN_POINTS = 20;

    PointCloud parseScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        PointCloud cloud;
        cloud.reserve(msg->ranges.size());

        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            float range = msg->ranges[i];
            if (std::isfinite(range) && range >= msg->range_min && range <= msg->range_max)
            {
                float angle = msg->angle_min + i * msg->angle_increment;
                cloud.push_back({range * std::cos(angle), range * std::sin(angle)});
            }
        }
        return cloud;
    }

    void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        PointCloud current_cloud = parseScan(msg);
        if (current_cloud.size() < ICP_MIN_POINTS)
        {
            RCLCPP_WARN(this->get_logger(), "Not enough points in the scan for ICP");
            return;
        }

        if (prev_cloud_.size() >= ICP_MIN_POINTS)
        {
            ICPResult icp_result = runICP(current_cloud, prev_cloud_);
            if (icp_result.converged)
            {
                float dtheta = std::atan2(icp_result.R(1, 0), icp_result.R(0, 0));

                // Rotate transaltion into map with current orientation
                float cos_h = std::cos(robot_pose_.z());
                float sin_h = std::sin(robot_pose_.z());
                float dx = cos_h * icp_result.t.x() - sin_h * icp_result.t.y();
                float dy = sin_h * icp_result.t.x() + cos_h * icp_result.t.y();

                // Update robot pose
                robot_pose_.x() += dx;
                robot_pose_.y() += dy;
                robot_pose_.z() += dtheta;
                robot_pose_.z() = normalizeAngle(robot_pose_.z());

                RCLCPP_INFO(this->get_logger(),
                            "ICP OK | pose=(%.3f, %.3f, %.3f°) | err=%.5f | Δ=(%.4f, %.4f, %.3f°)",
                            robot_pose_.x(), robot_pose_.y(), robot_pose_.z() * 180.0 / M_PI,
                            icp_result.error, dx, dy, dtheta * 180.0 / M_PI);

                publishPose(msg->header.stamp);
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "ICP did not converge");
            }
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "First scan received, initializing reference cloud");
        }
        prev_cloud_ = current_cloud;
    }

    ICPResult runICP(const PointCloud &source, const PointCloud &target)
    {
        ICPResult result;
        result.R = Eigen::Matrix2f::Identity();
        result.t = Eigen::Vector2f::Zero();
        result.error = std::numeric_limits<float>::max();
        result.converged = false;

        PointCloud src = source;

        for (int iter = 0; iter < ICP_MAX_ITER; ++iter)
        {
            std::vector<std::pair<int, int>> correspondences;
            correspondences.reserve(src.size());
            float total_error = 0.0f;

            for (size_t i = 0; i < src.size(); ++i)
            {
                float best_dist = ICP_MAX_DIST * ICP_MAX_DIST;
                int best_j = -1;
                for (size_t j = 0; j < target.size(); ++j)
                {
                    float dist = (src[i] - target[j]).squaredNorm();
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        best_j = static_cast<int>(j);
                    }
                }
                if (best_j >= 0)
                {
                    correspondences.push_back({static_cast<int>(i), best_j});
                    total_error += best_dist;
                }
            }
            if (correspondences.size() < ICP_MIN_POINTS)
                break;

            float mean_error = total_error / correspondences.size();
            Eigen::Vector2f src_centroid = Eigen::Vector2f::Zero();
            Eigen::Vector2f tgt_centroid = Eigen::Vector2f::Zero();
            for (const auto &[i, j] : correspondences)
            {
                src_centroid += src[i];
                tgt_centroid += target[j];
            }

            src_centroid /= static_cast<float>(correspondences.size());
            tgt_centroid /= static_cast<float>(correspondences.size());

            // Compute covariance matrix
            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            for (const auto &[i, j] : correspondences)
            {
                Eigen::Vector2f ps = src[i] - src_centroid;
                Eigen::Vector2f pt = target[j] - tgt_centroid;
                H += ps * pt.transpose();
            }

            // SVD for rotation
            Eigen::JacobiSVD<Eigen::Matrix2f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f R_iter = svd.matrixV() * svd.matrixU().transpose();

            if (R_iter.determinant() < 0)
            {
                Eigen::Matrix2f V = svd.matrixV();
                V.col(1) *= -1;
                R_iter = V * svd.matrixU().transpose();
            }

            Eigen::Vector2f t_iter = tgt_centroid - R_iter * src_centroid;

            // Transform to working frame
            for (auto &p : src)
            {
                p = R_iter * p + t_iter;
            }

            result.t = R_iter * result.t + t_iter;
            result.R = R_iter * result.R;

            float delta = t_iter.norm() + std::abs(std::acos(R_iter(0, 0)));
            if (delta < ICP_TOLERANCE && iter > 0)
            {
                result.error = mean_error;
                result.converged = true;
                RCLCPP_INFO(this->get_logger(), "ICP converged in %d iterations with error %.4f", iter, mean_error);
                break;
            }
            result.error = mean_error;
        }

        return result;
    };

    void publishPose(const rclcpp::Time &stamp)
    {
        // PoseStamped
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = robot_pose_.x();
        pose_msg.pose.position.y = robot_pose_.y();
        pose_msg.pose.position.z = 0.0;

        // Yaw → quaternion
        double half = robot_pose_.z() * 0.5;
        pose_msg.pose.orientation.z = std::sin(half);
        pose_msg.pose.orientation.w = std::cos(half);
        pose_publisher_->publish(pose_msg);

        // TF: map → base_link
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header = pose_msg.header;
        tf_msg.child_frame_id = "base_link";
        tf_msg.transform.translation.x = robot_pose_.x();
        tf_msg.transform.translation.y = robot_pose_.y();
        tf_msg.transform.rotation = pose_msg.pose.orientation;
        tf_broadcaster_->sendTransform(tf_msg);
    }

    static float normalizeAngle(float angle)
    {
        while (angle > M_PI)
            angle -= 2.0f * M_PI;
        while (angle < -M_PI)
            angle += 2.0f * M_PI;
        return angle;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SLAMNode>());
    rclcpp::shutdown();
    return 0;
}