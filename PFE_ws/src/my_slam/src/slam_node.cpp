#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "../include/scan_matcher.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

// Map Constants
static constexpr float MAP_RESOLUTION = 0.05f;
static constexpr int   MAP_WIDTH      = 400;
static constexpr int   MAP_HEIGHT     = 400;
static constexpr float MAP_ORIGIN_X   = -10.0f;
static constexpr float MAP_ORIGIN_Y   = -10.0f;

// Log-Odds Constants
static constexpr float L_OCC  =  0.85f;
static constexpr float L_FREE = -0.40f;
static constexpr float L_MIN  = -5.0f;
static constexpr float L_MAX  =  5.0f;

// Keyframe thresholds
static constexpr float KEYFRAME_DIST  = 0.30f;
static constexpr float KEYFRAME_ANGLE = 0.20f;

// Loop Closure Constants 
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

struct LoopConstraint {
    int             from_id;
    int             to_id;
    Eigen::Vector3d relative_pose;  
    float           score;
};

class SLAMNode : public rclcpp::Node
{
public:
    SLAMNode() : Node("slam_node"),
                 robot_pose_(Eigen::Vector3d::Zero()),
                 last_keyframe_pose_(Eigen::Vector3d::Zero())
    {
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&SLAMNode::laserScanCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&SLAMNode::odomCallback, this, std::placeholders::_1));

        map_publisher_  = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).transient_local());
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pose", 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        tf_buffer_      = std::make_shared<tf2_ros::Buffer>(this->get_clock(), tf2::Duration(std::chrono::seconds(10)));
        tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);

        initMap();
    }

private:
    ScanMatcher      scan_matcher_;
    Eigen::Vector3d  robot_pose_;
    Eigen::Vector3d  last_keyframe_pose_;
    Eigen::Vector3d  odom_pose_prev_;
    bool             first_odom_ = true;

    nav_msgs::msg::OccupancyGrid map_;
    std::vector<float>           log_odds_;
    Eigen::Affine2f              lidar_to_base_;
    bool                         lidar_tf_ready_ = false;

    std::vector<KeyFrame>       keyframes_;
    std::vector<LoopConstraint> loop_constraints_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr   map_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster>               tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer>                             tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                  tf_listener_;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        double qz  = msg->pose.pose.orientation.z;
        double qw  = msg->pose.pose.orientation.w;
        double yaw = 2.0 * std::atan2(qz, qw);
        Eigen::Vector3d current_odom(msg->pose.pose.position.x, msg->pose.pose.position.y, yaw);

        if (!first_odom_) {
            double dx  = current_odom.x() - odom_pose_prev_.x();
            double dy  = current_odom.y() - odom_pose_prev_.y();
            double dth = normalizeAngle(current_odom.z() - odom_pose_prev_.z());

            double cos_h = std::cos(robot_pose_.z());
            double sin_h = std::sin(robot_pose_.z());
            robot_pose_.x() += dx * cos_h - dy * sin_h;
            robot_pose_.y() += dx * sin_h + dy * cos_h;
            robot_pose_.z()  = normalizeAngle(robot_pose_.z() + dth);
        }

        odom_pose_prev_ = current_odom;
        first_odom_     = false;
    }

    void laserScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        PointCloud current_cloud = parseScan(msg);
        if (current_cloud.size() < 20) return;

        if (scan_matcher_.hasMap()) {
            Eigen::Vector3f initial_guess = robot_pose_.cast<float>();
            ICPResult map_result = scan_matcher_.match(current_cloud, initial_guess);

            if (map_result.converged) {
                float dtheta = std::atan2(map_result.R(1,0), map_result.R(0,0));
                robot_pose_.x() += map_result.t.x();
                robot_pose_.y() += map_result.t.y();
                robot_pose_.z()  = normalizeAngle(robot_pose_.z() + dtheta);
            }
        }

        double dist_moved  = (robot_pose_.head<2>() - last_keyframe_pose_.head<2>()).norm();
        double angle_moved = std::abs(normalizeAngle(robot_pose_.z() - last_keyframe_pose_.z()));

        if (dist_moved > KEYFRAME_DIST || angle_moved > KEYFRAME_ANGLE || !scan_matcher_.hasMap()) {

            KeyFrame kf;
            kf.id    = static_cast<int>(keyframes_.size());
            kf.pose  = robot_pose_;
            kf.cloud = current_cloud;
            keyframes_.push_back(kf);

            if (kf.id >= LC_MIN_AGE) {
                detectAndCloseLoop(kf);
            }

            updateMap(current_cloud, msg->header.stamp);
            scan_matcher_.updateMap(map_);
            last_keyframe_pose_ = robot_pose_;
        }

        publishPose(msg->header.stamp);
    }

    void detectAndCloseLoop(const KeyFrame& current_kf)
    {
        for (int i = 0; i < current_kf.id - LC_MIN_AGE; ++i) {
            const KeyFrame& candidate = keyframes_[i];

            double dx   = current_kf.pose.x() - candidate.pose.x();
            double dy   = current_kf.pose.y() - candidate.pose.y();
            double dist = std::sqrt(dx*dx + dy*dy);
            if (dist > LC_SEARCH_RADIUS) continue;

            Eigen::Vector3f initial_guess(
                static_cast<float>(dx),
                static_cast<float>(dy),
                static_cast<float>(normalizeAngle(current_kf.pose.z() - candidate.pose.z()))
            );

            ICPResult result = scan_matcher_.matchClouds(current_kf.cloud, candidate.cloud, initial_guess);
            if (!result.converged)          continue;
            if (result.mean_error > LC_ICP_SCORE_THR) continue;

            int inliers = 0;
            for (const auto& p : current_kf.cloud) {
                Eigen::Vector2f pt = result.R * p + result.t;
                for (const auto& q : candidate.cloud) {
                    if ((pt - q).norm() < LC_INLIER_DIST) { ++inliers; break; }
                }
            }
            float inlier_ratio = static_cast<float>(inliers) / static_cast<float>(current_kf.cloud.size());
            if (inlier_ratio < LC_INLIER_RATIO) continue;

            float dth = std::atan2(result.R(1,0), result.R(0,0));

            LoopConstraint lc;
            lc.from_id       = current_kf.id;
            lc.to_id         = i;
            lc.relative_pose = Eigen::Vector3d(result.t.x(), result.t.y(), dth);
            lc.score         = result.mean_error;
            loop_constraints_.push_back(lc);

            RCLCPP_INFO(this->get_logger(),
                "Loop closed: KF%d -> KF%d  score=%.3f  inliers=%.0f%%",
                current_kf.id, i, result.mean_error, inlier_ratio * 100.0f);

            applyPoseGraphCorrection(lc);
            rebuildMap();
            return; 
        }
    }

    void applyPoseGraphCorrection(const LoopConstraint& lc)
    {
        robot_pose_.x() += lc.relative_pose.x();
        robot_pose_.y() += lc.relative_pose.y();
        robot_pose_.z()  = normalizeAngle(robot_pose_.z() + lc.relative_pose.z());

        keyframes_.back().pose = robot_pose_;
    }

    void rebuildMap()
    {
        std::fill(log_odds_.begin(), log_odds_.end(), 0.0f);
        std::fill(map_.data.begin(), map_.data.end(), -1);

        Eigen::Vector3d saved_pose = robot_pose_;

        for (const auto& kf : keyframes_) {
            robot_pose_ = kf.pose;
            updateMap(kf.cloud, map_.header.stamp);
        }

        robot_pose_ = saved_pose;

        scan_matcher_.updateMap(map_);
    }

    void initMap()
    {
        log_odds_.assign(MAP_WIDTH * MAP_HEIGHT, 0.0f);
        map_.header.frame_id           = "map";
        map_.info.resolution           = MAP_RESOLUTION;
        map_.info.width                = MAP_WIDTH;
        map_.info.height               = MAP_HEIGHT;
        map_.info.origin.position.x    = MAP_ORIGIN_X;
        map_.info.origin.position.y    = MAP_ORIGIN_Y;
        map_.info.origin.orientation.w = 1.0;
        map_.data.assign(MAP_WIDTH * MAP_HEIGHT, -1);
    }

    PointCloud parseScan(const sensor_msgs::msg::LaserScan::SharedPtr& msg)
    {
        if (!lidar_tf_ready_) {
            lidar_tf_ready_ = fetchLidarToBase();
            if (!lidar_tf_ready_) return {};
        }
        PointCloud cloud;
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            float r = msg->ranges[i];
            if (std::isfinite(r) && r >= msg->range_min && r <= msg->range_max) {
                float angle = msg->angle_min + i * msg->angle_increment;
                Eigen::Vector2f p_lidar(r * std::cos(angle), r * std::sin(angle));
                cloud.push_back(lidar_to_base_ * p_lidar);
            }
        }
        return cloud;
    }

    bool fetchLidarToBase()
    {
        try {
            auto tf = tf_buffer_->lookupTransform("base_link", "rplidar_link", tf2::TimePointZero);
            float tx  = tf.transform.translation.x;
            float ty  = tf.transform.translation.y;
            float qz  = tf.transform.rotation.z;
            float qw  = tf.transform.rotation.w;
            float yaw = 2.0f * std::atan2(qz, qw);
            Eigen::Matrix2f R;
            R << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
            lidar_to_base_ = Eigen::Affine2f::Identity();
            lidar_to_base_.linear()      = R;
            lidar_to_base_.translation() = Eigen::Vector2f(tx, ty);
            return true;
        } catch (...) { return false; }
    }

    void updateMap(const PointCloud& cloud, const rclcpp::Time& stamp)
    {
        int rx = worldToCell(robot_pose_.x(), MAP_ORIGIN_X);
        int ry = worldToCell(robot_pose_.y(), MAP_ORIGIN_Y);
        float cos_h = std::cos(robot_pose_.z());
        float sin_h = std::sin(robot_pose_.z());

        for (const auto& p : cloud) {
            float wx = robot_pose_.x() + cos_h * p.x() - sin_h * p.y();
            float wy = robot_pose_.y() + sin_h * p.x() + cos_h * p.y();
            int ex = worldToCell(wx, MAP_ORIGIN_X);
            int ey = worldToCell(wy, MAP_ORIGIN_Y);

            bresenham(rx, ry, ex, ey, [&](int cx, int cy) { updateCell(cx, cy, L_FREE); });
            updateCell(ex, ey, L_OCC);
        }

        for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; ++i) {
            if      (log_odds_[i] == 0) map_.data[i] = -1;
            else if (log_odds_[i]  > 0) map_.data[i] = 100;
            else                         map_.data[i] = 0;
        }
        map_.header.stamp = stamp;
        map_publisher_->publish(map_);
    }

    void bresenham(int x0, int y0, int x1, int y1, const std::function<void(int,int)>& visit)
    {
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;
        while (true) {
            if (x0 == x1 && y0 == y1) break;
            if (!inBounds(x0, y0))    break;
            visit(x0, y0);
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 <  dx) { err += dx; y0 += sy; }
        }
    }

    void updateCell(int cx, int cy, float delta)
    {
        if (!inBounds(cx, cy)) return;
        int idx = cy * MAP_WIDTH + cx;
        log_odds_[idx] = std::clamp(log_odds_[idx] + delta, L_MIN, L_MAX);
    }

    void publishPose(const rclcpp::Time& stamp)
    {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp    = stamp;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = robot_pose_.x();
        pose_msg.pose.position.y = robot_pose_.y();
        double half = robot_pose_.z() * 0.5;
        pose_msg.pose.orientation.z = std::sin(half);
        pose_msg.pose.orientation.w = std::cos(half);
        pose_publisher_->publish(pose_msg);

        double odom_x   = odom_pose_prev_.x();
        double odom_y   = odom_pose_prev_.y();
        double odom_yaw = odom_pose_prev_.z();

        double cos_odom = std::cos(odom_yaw);
        double sin_odom = std::sin(odom_yaw);

        double correction_x   = robot_pose_.x() - (cos_odom * odom_x - sin_odom * odom_y);
        double correction_y   = robot_pose_.y() - (sin_odom * odom_x + cos_odom * odom_y);
        double correction_yaw = normalizeAngle(robot_pose_.z() - odom_yaw);

        double correction_half = correction_yaw * 0.5;

        geometry_msgs::msg::TransformStamped map_to_odom;
        map_to_odom.header.stamp       = stamp;
        map_to_odom.header.frame_id    = "map";
        map_to_odom.child_frame_id     = "odom";
        map_to_odom.transform.translation.x = correction_x;
        map_to_odom.transform.translation.y = correction_y;
        map_to_odom.transform.translation.z = 0.0;
        map_to_odom.transform.rotation.z    = std::sin(correction_half);
        map_to_odom.transform.rotation.w    = std::cos(correction_half);
        tf_broadcaster_->sendTransform(map_to_odom);
    }

    static int  worldToCell(float world, float origin) { return static_cast<int>((world - origin) / MAP_RESOLUTION); }
    static bool inBounds(int cx, int cy) { return cx >= 0 && cx < MAP_WIDTH && cy >= 0 && cy < MAP_HEIGHT; }
    static float normalizeAngle(float angle) {
        while (angle >  M_PI) angle -= 2.0f * M_PI;
        while (angle < -M_PI) angle += 2.0f * M_PI;
        return angle;
    }
    static double normalizeAngle(double angle) {
        while (angle >  M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SLAMNode>());
    rclcpp::shutdown();
    return 0;
}