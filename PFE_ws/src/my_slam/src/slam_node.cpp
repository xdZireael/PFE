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

static constexpr float MAP_RESOLUTION = 0.05f; 
static constexpr int MAP_WIDTH = 400;           
static constexpr int MAP_HEIGHT = 400;         
static constexpr float MAP_ORIGIN_X = -10.0f; 
static constexpr float MAP_ORIGIN_Y = -10.0f;

// Log-odds params for occupancy grid mapping
static constexpr float L_OCC = 0.85f; 
static constexpr float L_FREE = 0.40f;
static constexpr float L_MIN = -5.0f;
static constexpr float L_MAX = 5.0f;

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
    std::vector<float> log_odds_;

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

    void initMap(){
        log_odds_.assign(MAP_WIDTH * MAP_HEIGHT, 0.0f);
        map_.header.frame_id = "map";
        map_.info.resolution = MAP_RESOLUTION;
        map_.info.width = MAP_WIDTH;
        map_.info.height = MAP_HEIGHT;
        map_.info.origin.position.x = MAP_ORIGIN_X;
        map_.info.origin.position.y = MAP_ORIGIN_Y;
        map_.info.origin.position.z = 1.0;
        map_.data.assign(MAP_WIDTH * MAP_HEIGHT, -1); 

        RCLCPP_INFO(this->get_logger(), "Map initialized");
    }
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
        if (current_cloud.size() < ICP_MIN_POINTS) return;

        if (prev_cloud_.size() >= ICP_MIN_POINTS) {
            ICPResult result = runICP(current_cloud, prev_cloud_);

            if (result.converged) {
                float dtheta = std::atan2(result.R(1,0), result.R(0,0));
                float cos_h  = std::cos(robot_pose_.z());
                float sin_h  = std::sin(robot_pose_.z());

                robot_pose_.x() += cos_h * result.t.x() - sin_h * result.t.y();
                robot_pose_.y() += sin_h * result.t.x() + cos_h * result.t.y();
                robot_pose_.z()  = normalizeAngle(robot_pose_.z() + dtheta);

                RCLCPP_INFO(this->get_logger(),
                    "pose=(%.2f, %.2f, %.1f°) err=%.5f",
                    robot_pose_.x(), robot_pose_.y(),
                    robot_pose_.z() * 180.0 / M_PI, result.error);
            }
        } else {
            RCLCPP_INFO(this->get_logger(), "First scan — initialising reference cloud.");
        }

        prev_cloud_ = current_cloud;

        
        updateMap(current_cloud, msg->header.stamp);
        publishPose(msg->header.stamp);
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

            bresenham(rx, ry, ex, ey, [&](int cx, int cy) {
                updateCell(cx, cy, L_FREE);
            });

            updateCell(ex, ey, L_OCC);
        }

        for (int i = 0; i < MAP_WIDTH * MAP_HEIGHT; ++i) {
            float lo = log_odds_[i];
            if      (lo == 0.0f) map_.data[i] = -1;   // never touched = unknown
            else if (lo  > 0.0f) map_.data[i] = static_cast<int8_t>(
                                     std::min(100.0f, lo / L_MAX * 100.0f));
            else                 map_.data[i] = 0;    // free
        }

        map_.header.stamp = stamp;
        map_publisher_->publish(map_);
    }

    void bresenham(int x0, int y0, int x1, int y1,
                   const std::function<void(int,int)>& visit)
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

    static int worldToCell(float world, float origin)
    {
        return static_cast<int>((world - origin) / MAP_RESOLUTION);
    }

    static bool inBounds(int cx, int cy)
    {
        return cx >= 0 && cx < MAP_WIDTH && cy >= 0 && cy < MAP_HEIGHT;
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