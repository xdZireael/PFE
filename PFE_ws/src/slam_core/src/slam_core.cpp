#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_broadcaster.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "slam_msgs/msg/loop_constraint.hpp"
#include "../include/scan_matcher.hpp"  

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

static constexpr float MAP_RESOLUTION = 0.05f;
static constexpr int   MAP_WIDTH      = 400;
static constexpr int   MAP_HEIGHT     = 400;
static constexpr float MAP_ORIGIN_X   = -10.0f;
static constexpr float MAP_ORIGIN_Y   = -10.0f;

static constexpr float L_OCC  =  0.85f;
static constexpr float L_FREE = -0.40f;
static constexpr float L_MIN  = -5.0f;
static constexpr float L_MAX  =  5.0f;

static constexpr double ODOM_NOISE_XY  = 0.10;   // wheel odom (m)
static constexpr double ODOM_NOISE_TH  = 0.05;   // wheel odom (rad)
static constexpr double LIDAR_NOISE_XY = 0.03;   // LiDAR ICP (m)
static constexpr double LIDAR_NOISE_TH = 0.02;   // LiDAR ICP (rad)
static constexpr double CAM_NOISE_XY   = 0.05;   // visual VO (m)
static constexpr double CAM_NOISE_TH   = 0.03;   // visual VO (rad)

struct ScanKeyFrame {
    Eigen::Vector3d pose;
    PointCloud      cloud;
};

//AI
class EKF3DOF
{
public:
    EKF3DOF()
    {
        mu_  = Eigen::Vector3d::Zero();
        // Start with high uncertainty; first good measurement sets the scale.
        sigma_ = Eigen::Matrix3d::Identity() * 1e6;
    }

    // Process update – propagate state with additive Gaussian noise Q
    void predict(const Eigen::Vector3d& delta, const Eigen::Matrix3d& Q)
    {
        // Jacobian of the motion model F (identity for additive delta)
        // Full nonlinear Jacobian for (dx, dy rotated by current yaw):
        double ch = std::cos(mu_.z()), sh = std::sin(mu_.z());
        Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
        F(0,2) = -delta.x() * sh - delta.y() * ch;
        F(1,2) =  delta.x() * ch - delta.y() * sh;

        mu_.x() += delta.x() * ch - delta.y() * sh;
        mu_.y() += delta.x() * sh + delta.y() * ch;
        mu_.z()  = normalizeAngle(mu_.z() + delta.z());

        sigma_ = F * sigma_ * F.transpose() + Q;
    }

    // Measurement update – absolute pose observation z with noise R (diagonal)
    void update(const Eigen::Vector3d& z, const Eigen::Matrix3d& R)
    {
        // H = I (direct state observation)
        Eigen::Matrix3d S = sigma_ + R;
        Eigen::Matrix3d K = sigma_ * S.inverse();           // Kalman gain

        Eigen::Vector3d innov = z - mu_;
        innov.z() = normalizeAngle(innov.z());              // angle wrap

        mu_   += K * innov;
        mu_.z() = normalizeAngle(mu_.z());
        sigma_ = (Eigen::Matrix3d::Identity() - K) * sigma_;
    }

    Eigen::Vector3d mean()       const { return mu_; }
    Eigen::Matrix3d covariance() const { return sigma_; }

private:
    Eigen::Vector3d mu_;
    Eigen::Matrix3d sigma_;

    static double normalizeAngle(double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }
};

class SlamCoreNode : public rclcpp::Node
{
public:
    SlamCoreNode()
    : Node("slam_core"),
      last_odom_pose_(Eigen::Vector3d::Zero())
    {
        initMap();

        R_odom_  = makeDiag(ODOM_NOISE_XY,  ODOM_NOISE_XY,  ODOM_NOISE_TH);
        R_lidar_ = makeDiag(LIDAR_NOISE_XY, LIDAR_NOISE_XY, LIDAR_NOISE_TH);
        R_cam_   = makeDiag(CAM_NOISE_XY,   CAM_NOISE_XY,   CAM_NOISE_TH);

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&SlamCoreNode::onWheelOdom, this, std::placeholders::_1));

        lidar_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/lidar/odometry", 10,
            std::bind(&SlamCoreNode::onLidarOdom, this, std::placeholders::_1));

        cam_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/camera/odometry", 10,
            std::bind(&SlamCoreNode::onCamOdom, this, std::placeholders::_1));

        lidar_loop_sub_ = create_subscription<slam_msgs::msg::LoopConstraint>(
            "/lidar/loop_constraint", 10,
            std::bind(&SlamCoreNode::onLoopConstraint, this, std::placeholders::_1));

        cam_loop_sub_ = create_subscription<slam_msgs::msg::LoopConstraint>(
            "/camera/loop_constraint", 10,
            std::bind(&SlamCoreNode::onLoopConstraint, this, std::placeholders::_1));

        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&SlamCoreNode::onScan, this, std::placeholders::_1));

        map_pub_  = create_publisher<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(1).transient_local());
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/pose", 10);

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
        tf_buffer_      = std::make_shared<tf2_ros::Buffer>(
            get_clock(), tf2::Duration(std::chrono::seconds(10)));
        tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
    }

private:
    EKF3DOF ekf_;
    Eigen::Matrix3d R_odom_, R_lidar_, R_cam_;

    Eigen::Vector3d last_odom_pose_;
    bool            first_odom_ = true;

    nav_msgs::msg::OccupancyGrid  map_;
    std::vector<float>            log_odds_;

    std::vector<ScanKeyFrame>     scan_kfs_;
    Eigen::Affine2f               lidar_to_base_;
    bool                          lidar_tf_ready_ = false;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          lidar_odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          cam_odom_sub_;
    rclcpp::Subscription<slam_msgs::msg::LoopConstraint>::SharedPtr   lidar_loop_sub_;
    rclcpp::Subscription<slam_msgs::msg::LoopConstraint>::SharedPtr   cam_loop_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr      scan_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr        map_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr     pose_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster>                    tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer>                                  tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                       tf_listener_;

    void onWheelOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        double qz  = msg->pose.pose.orientation.z;
        double qw  = msg->pose.pose.orientation.w;
        double yaw = 2.0 * std::atan2(qz, qw);
        Eigen::Vector3d cur(msg->pose.pose.position.x, msg->pose.pose.position.y, yaw);

        if (first_odom_) {
            last_odom_pose_ = cur;
            first_odom_ = false;
            return;
        }

        double dx  = cur.x() - last_odom_pose_.x();
        double dy  = cur.y() - last_odom_pose_.y();
        double dth = normalizeAngle(cur.z() - last_odom_pose_.z());
        last_odom_pose_ = cur;

        double motion = std::sqrt(dx*dx + dy*dy);
        Eigen::Matrix3d Q = makeDiag(
            ODOM_NOISE_XY * motion,
            ODOM_NOISE_XY * motion,
            ODOM_NOISE_TH * std::abs(dth));

        ekf_.predict({dx, dy, dth}, Q);

        publishPose(msg->header.stamp);
    }

    void onLidarOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        Eigen::Vector3d z = poseFromOdom(msg);
        ekf_.update(z, R_lidar_);
        publishPose(msg->header.stamp);
    }

    void onCamOdom(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        Eigen::Vector3d z = poseFromOdom(msg);
        ekf_.update(z, R_cam_);
        publishPose(msg->header.stamp);
    }

    void onLoopConstraint(const slam_msgs::msg::LoopConstraint::SharedPtr msg)
    {
        RCLCPP_INFO(get_logger(),
            "[core] loop: KF%d → KF%d  score=%.3f",
            msg->from_id, msg->to_id, msg->score);

        applyPoseGraphCorrection(msg);
        rebuildMap();
    }
    void onScan(const sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        if (!lidar_tf_ready_) {
            lidar_tf_ready_ = fetchLidarTF();
            if (!lidar_tf_ready_) return;
        }

        PointCloud cloud = parseScan(msg);
        if (cloud.size() < 20) return;

        Eigen::Vector3d cur = ekf_.mean();

        if (!scan_kfs_.empty()) {
            double d = (cur.head<2>() - scan_kfs_.back().pose.head<2>()).norm();
            double a = std::abs(normalizeAngle(cur.z() - scan_kfs_.back().pose.z()));
            if (d < 0.30 && a < 0.20) goto publish_map;
        }

        scan_kfs_.push_back({cur, cloud});
        updateMapFromScan(cloud, cur, msg->header.stamp);

        publish_map:
        map_.header.stamp = msg->header.stamp;
        map_pub_->publish(map_);
    }
    void applyPoseGraphCorrection(const slam_msgs::msg::LoopConstraint::SharedPtr lc)
    {
        Eigen::Vector3d mu = ekf_.mean();
        mu.x() += lc->dx;
        mu.y() += lc->dy;
        mu.z()  = normalizeAngle(mu.z() + lc->dtheta);

        EKF3DOF fresh;
        fresh.predict(mu, makeDiag(0.05, 0.05, 0.02));
        ekf_ = fresh;

        if (scan_kfs_.empty()) return;
        int from = std::min(lc->from_id, static_cast<int>(scan_kfs_.size()) - 1);
        double n  = static_cast<double>(scan_kfs_.size() - from);
        for (int i = from; i < static_cast<int>(scan_kfs_.size()); ++i) {
            double alpha = (i - from) / n;          // 0 at from_id, 1 at end
            scan_kfs_[i].pose.x() += alpha * lc->dx;
            scan_kfs_[i].pose.y() += alpha * lc->dy;
            scan_kfs_[i].pose.z()  = normalizeAngle(
                scan_kfs_[i].pose.z() + alpha * lc->dtheta);
        }
    }

    void rebuildMap()
    {
        std::fill(log_odds_.begin(), log_odds_.end(), 0.0f);
        std::fill(map_.data.begin(), map_.data.end(), -1);

        rclcpp::Time now = get_clock()->now();
        for (const auto& kf : scan_kfs_)
            updateMapFromScan(kf.cloud, kf.pose, now);

        map_.header.stamp = now;
        map_pub_->publish(map_);
    }


    void updateMapFromScan(const PointCloud& cloud,
                           const Eigen::Vector3d& pose,
                           const rclcpp::Time& /*stamp*/)
    {
        int rx = worldToCell(static_cast<float>(pose.x()), MAP_ORIGIN_X);
        int ry = worldToCell(static_cast<float>(pose.y()), MAP_ORIGIN_Y);
        float ch = static_cast<float>(std::cos(pose.z()));
        float sh = static_cast<float>(std::sin(pose.z()));

        for (const auto& p : cloud) {
            float wx = static_cast<float>(pose.x()) + ch * p.x() - sh * p.y();
            float wy = static_cast<float>(pose.y()) + sh * p.x() + ch * p.y();
            int ex = worldToCell(wx, MAP_ORIGIN_X);
            int ey = worldToCell(wy, MAP_ORIGIN_Y);

            bresenham(rx, ry, ex, ey,
                [this](int cx, int cy){ updateCell(cx, cy, L_FREE); });
            updateCell(ex, ey, L_OCC);
        }

        for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; ++i) {
            if      (log_odds_[i] == 0.0f) map_.data[i] = -1;
            else if (log_odds_[i]  > 0.0f) map_.data[i] = 100;
            else                            map_.data[i] = 0;
        }
    }

    void publishPose(const rclcpp::Time& stamp)
    {
        Eigen::Vector3d mu = ekf_.mean();

        // PoseStamped
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp    = stamp;
        ps.header.frame_id = "map";
        ps.pose.position.x = mu.x();
        ps.pose.position.y = mu.y();
        double half = mu.z() * 0.5;
        ps.pose.orientation.z = std::sin(half);
        ps.pose.orientation.w = std::cos(half);
        pose_pub_->publish(ps);

        // map → odom TF correction (same logic as original slam_node.cpp)
        double ox   = last_odom_pose_.x();
        double oy   = last_odom_pose_.y();
        double oyaw = last_odom_pose_.z();
        double co   = std::cos(oyaw), so = std::sin(oyaw);

        double cx   = mu.x() - (co * ox - so * oy);
        double cy   = mu.y() - (so * ox + co * oy);
        double cyaw = normalizeAngle(mu.z() - oyaw);
        double ch   = cyaw * 0.5;

        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp       = stamp;
        tf.header.frame_id    = "map";
        tf.child_frame_id     = "odom";
        tf.transform.translation.x = cx;
        tf.transform.translation.y = cy;
        tf.transform.translation.z = 0.0;
        tf.transform.rotation.z    = std::sin(ch);
        tf.transform.rotation.w    = std::cos(ch);
        tf_broadcaster_->sendTransform(tf);
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
            auto tf = tf_buffer_->lookupTransform("base_link", "rplidar_link", tf2::TimePointZero);
            float tx  = tf.transform.translation.x;
            float ty  = tf.transform.translation.y;
            float qz  = tf.transform.rotation.z;
            float qw  = tf.transform.rotation.w;
            float yaw = 2.f * std::atan2(qz, qw);
            Eigen::Matrix2f R;
            R << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
            lidar_to_base_ = Eigen::Affine2f::Identity();
            lidar_to_base_.linear()      = R;
            lidar_to_base_.translation() = Eigen::Vector2f(tx, ty);
            return true;
        } catch (...) { return false; }
    }

    void bresenham(int x0, int y0, int x1, int y1,
                   const std::function<void(int,int)>& visit)
    {
        int dx = std::abs(x1-x0), sx = x0 < x1 ? 1 : -1;
        int dy = std::abs(y1-y0), sy = y0 < y1 ? 1 : -1;
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

    static Eigen::Vector3d poseFromOdom(const nav_msgs::msg::Odometry::SharedPtr& msg)
    {
        double qz  = msg->pose.pose.orientation.z;
        double qw  = msg->pose.pose.orientation.w;
        return {msg->pose.pose.position.x,
                msg->pose.pose.position.y,
                2.0 * std::atan2(qz, qw)};
    }

    static Eigen::Matrix3d makeDiag(double a, double b, double c)
    {
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
        M(0,0) = a; M(1,1) = b; M(2,2) = c;
        return M;
    }

    static int  worldToCell(float w, float origin) {
        return static_cast<int>((w - origin) / MAP_RESOLUTION);
    }
    static bool inBounds(int cx, int cy) {
        return cx >= 0 && cx < MAP_WIDTH && cy >= 0 && cy < MAP_HEIGHT;
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
    rclcpp::spin(std::make_shared<SlamCoreNode>());
    rclcpp::shutdown();
    return 0;
}