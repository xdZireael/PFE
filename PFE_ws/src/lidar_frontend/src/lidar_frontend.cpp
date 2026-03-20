#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "slam_common/scan_matcher.hpp"
#include "slam_msgs/msg/loop_constraint.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <algorithm>

static constexpr float KF_DIST        = 0.30f;
static constexpr float KF_ANGLE       = 0.20f;

static constexpr float LC_SEARCH_RADIUS = 1.0f;
static constexpr int   LC_MIN_AGE       = 5;
static constexpr float LC_ICP_SCORE_THR = 0.10f;
static constexpr float LC_INLIER_RATIO  = 0.65f;
static constexpr float LC_INLIER_DIST   = 0.10f;

struct KeyFrame {
    int             id;
    Eigen::Vector3d pose;
    PointCloud      cloud;
};


class LidarFrontendNode : public rclcpp::Node
{
public:
    LidarFrontendNode()
    : Node("lidar_frontend"),
      robot_pose_(Eigen::Vector3d(-0.040, 0.0, 0.193)),
      last_kf_pose_(Eigen::Vector3d::Zero())
    {
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&LidarFrontendNode::onScan, this, std::placeholders::_1));

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&LidarFrontendNode::onOdom, this, std::placeholders::_1));

        map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(1).transient_local(),
            std::bind(&LidarFrontendNode::onMap, this, std::placeholders::_1));

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/lidar/odometry", 10);
        loop_pub_ = create_publisher<slam_msgs::msg::LoopConstraint>("/lidar/loop_constraint", 10);

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock(), tf2::Duration(std::chrono::seconds(10)));
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
    }

private:
    ScanMatcher      matcher_;
    Eigen::Vector3d  robot_pose_;
    Eigen::Vector3d  last_kf_pose_;
    Eigen::Vector3d  odom_prev_;
    bool             first_odom_     = true;
    bool             lidar_tf_ready_ = false;
    Eigen::Affine2f  lidar_to_base_;
    float heading_at_last_icp_ = 0.0f;

    std::vector<KeyFrame> keyframes_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr      scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr     map_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr             odom_pub_;
    rclcpp::Publisher<slam_msgs::msg::LoopConstraint>::SharedPtr      loop_pub_;
    std::shared_ptr<tf2_ros::Buffer>                                  tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                       tf_listener_;

    void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        double qx  = msg->pose.pose.orientation.x;
        double qy  = msg->pose.pose.orientation.y;
        double qz  = msg->pose.pose.orientation.z;
        double qw  = msg->pose.pose.orientation.w;
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        double yaw = std::atan2(siny_cosp, cosy_cosp);
        Eigen::Vector3d cur(msg->pose.pose.position.x, msg->pose.pose.position.y, yaw);

        if (!first_odom_) {
            double dx  = cur.x() - odom_prev_.x();
            double dy  = cur.y() - odom_prev_.y();
            double dth = normalizeAngle(cur.z() - odom_prev_.z());

            robot_pose_.x() += dx;
            robot_pose_.y() += dy;
            robot_pose_.z()  = normalizeAngle(robot_pose_.z() + dth);
        }

        odom_prev_  = cur;
        first_odom_ = false;
        
    }

    void onMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        matcher_.updateMap(*msg);
    }


    void onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        PointCloud cloud = parseScan(msg);
        if (cloud.size() < 20) return;

        if (matcher_.hasMap()) {
            Eigen::Vector3f guess = robot_pose_.cast<float>();
            ICPResult res = matcher_.match(cloud, guess);

            if (res.converged && res.mean_error < 0.1f) {
                Eigen::Vector2f corrected_xy = res.R * guess.head<2>() + res.t;
                float icp_dth = std::atan2(res.R(1,0), res.R(0,0));

                float odom_delta_since_last_icp = static_cast<float>(
                    normalizeAngle(robot_pose_.z() - heading_at_last_icp_));

                float heading_disagreement = std::abs(
                    static_cast<float>(normalizeAngle(icp_dth - odom_delta_since_last_icp)));

                robot_pose_.x() = static_cast<double>(corrected_xy.x());
                robot_pose_.y() = static_cast<double>(corrected_xy.y());

                if (heading_disagreement < 0.26f) {
                    robot_pose_.z() = normalizeAngle(
                        static_cast<double>(guess.z() + icp_dth));
                }

                heading_at_last_icp_ = static_cast<float>(robot_pose_.z());
            } else {
                RCLCPP_WARN(get_logger(), "ICP failed: score=%.3f", res.mean_error);
            }
        }

        double dist  = (robot_pose_.head<2>() - last_kf_pose_.head<2>()).norm();
        double angle = std::abs(normalizeAngle(robot_pose_.z() - last_kf_pose_.z()));

        if (dist > KF_DIST || angle > KF_ANGLE || keyframes_.empty()) {
            KeyFrame kf;
            kf.id    = static_cast<int>(keyframes_.size());
            kf.pose  = robot_pose_;
            kf.cloud = cloud;
            keyframes_.push_back(kf);
            last_kf_pose_ = robot_pose_;

            if (kf.id >= LC_MIN_AGE)
                detectLoop(kf);
        }

        publishOdometry(msg->header.stamp);
    }

    void detectLoop(const KeyFrame& cur)
    {
        for (int i = 0; i < cur.id - LC_MIN_AGE; ++i) {
            const KeyFrame& cand = keyframes_[i];

            double dx = cur.pose.x() - cand.pose.x();
            double dy = cur.pose.y() - cand.pose.y();
            if (std::sqrt(dx*dx + dy*dy) > LC_SEARCH_RADIUS) continue;

            double cos_c = std::cos(cand.pose.z());
            double sin_c = std::sin(cand.pose.z());
            Eigen::Vector3f guess(
                static_cast<float>( cos_c * dx + sin_c * dy),
                static_cast<float>(-sin_c * dx + cos_c * dy),
                static_cast<float>(normalizeAngle(cur.pose.z() - cand.pose.z())));

            ICPResult res = matcher_.matchClouds(cur.cloud, cand.cloud, guess);
            if (!res.converged || res.mean_error > LC_ICP_SCORE_THR) continue;
    
            slam_msgs::msg::LoopConstraint lc;
            lc.from_id = cur.id;
            lc.to_id   = i;
            lc.dx      = res.t.x();
            lc.dy      = res.t.y();
            lc.dtheta  = std::atan2(res.R(1,0), res.R(0,0));
            lc.score   = res.mean_error;
            loop_pub_->publish(lc);

            RCLCPP_INFO(get_logger(),
                "[lidar] loop: KF%d → KF%d  score=%.3f",
                cur.id, i, res.mean_error);
            return;
        }
    }

    void publishOdometry(const rclcpp::Time& stamp)
    {
        nav_msgs::msg::Odometry msg;
        msg.header.stamp    = stamp;
        msg.header.frame_id = "map";
        msg.child_frame_id  = "base_link";
        msg.pose.pose.position.x = robot_pose_.x();
        msg.pose.pose.position.y = robot_pose_.y();
        double half = robot_pose_.z() * 0.5;
        msg.pose.pose.orientation.z = std::sin(half);
        msg.pose.pose.orientation.w = std::cos(half);
        odom_pub_->publish(msg);
    }

    PointCloud parseScan(const sensor_msgs::msg::LaserScan::SharedPtr& msg)
    {
        if (!lidar_tf_ready_) {
            lidar_tf_ready_ = fetchLidarTF();
            if (!lidar_tf_ready_) return {};
        }
        PointCloud cloud;
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            float r = msg->ranges[i];
            if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) continue;
            float angle = msg->angle_min + i * msg->angle_increment;
            Eigen::Vector2f p(r * std::cos(angle), r * std::sin(angle));
            cloud.push_back(lidar_to_base_ * p);
        }
        return cloud;
    }

    bool fetchLidarTF()
    {
        try {
            if (!tf_buffer_->canTransform("base_link", "rplidar_link",
                    tf2::TimePointZero,
                    tf2::Duration(std::chrono::milliseconds(100))))
            {
                RCLCPP_WARN(get_logger(), "Waiting for TF base_link -> rplidar_link...");
                return false;
            }

            auto tf = tf_buffer_->lookupTransform(
                "base_link", "rplidar_link",
                tf2::TimePointZero,
                tf2::Duration(std::chrono::milliseconds(100)));

            float tx = tf.transform.translation.x;
            float ty = tf.transform.translation.y;

            double qx = tf.transform.rotation.x;
            double qy = tf.transform.rotation.y;
            double qz = tf.transform.rotation.z;
            double qw = tf.transform.rotation.w;

            double siny_cosp = 2.0 * (qw * qz + qx * qy);
            double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
            float yaw = static_cast<float>(std::atan2(siny_cosp, cosy_cosp));

            Eigen::Matrix2f R;
            R << std::cos(yaw), -std::sin(yaw),
                 std::sin(yaw),  std::cos(yaw);
            lidar_to_base_ = Eigen::Affine2f::Identity();
            lidar_to_base_.linear()      = R;
            lidar_to_base_.translation() = Eigen::Vector2f(tx, ty);
            return true;

        } catch (...) { return false; }
    }

    static double normalizeAngle(double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarFrontendNode>());
    rclcpp::shutdown();
    return 0;
}