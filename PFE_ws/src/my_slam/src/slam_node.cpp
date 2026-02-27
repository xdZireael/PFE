#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include "my_slam/scan_matcher.hpp"
#include "my_slam/occupancy_map.hpp"

class SlamNode : public rclcpp::Node {
public:
    SlamNode() : Node("slam_node"),
        map_(400, 400, 0.05, -10.0, -10.0),  // 20m x 20m map, 5cm resolution
        has_prev_scan_(false),
        robot_x_(0.0), robot_y_(0.0), robot_theta_(0.0)
    {
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&SlamNode::scanCallback, this, std::placeholders::_1));

        map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/map", 10);
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/slam_pose", 10);

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        RCLCPP_INFO(get_logger(), "SLAM node started");
    }

private:
    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {

        // 1. Convert scan to 2D points
        auto current_points = scanToPoints(msg);

        if (!has_prev_scan_) {
            prev_points_ = current_points;
            has_prev_scan_ = true;
            return;
        }

        // 2. Run ICP to estimate motion since last scan
        ICP icp;
        Transform2D delta = icp.align(current_points, prev_points_);

        // 3. Update robot pose
        double cos_t = std::cos(robot_theta_);
        double sin_t = std::sin(robot_theta_);
        robot_x_     += cos_t * delta.dx - sin_t * delta.dy;
        robot_y_     += sin_t * delta.dx + cos_t * delta.dy;
        robot_theta_ += delta.dtheta;

        // Normalize angle
        while (robot_theta_ >  M_PI) robot_theta_ -= 2 * M_PI;
        while (robot_theta_ < -M_PI) robot_theta_ += 2 * M_PI;

        // 4. Update map with current scan at estimated pose
        for (auto& p : current_points) {
            // Transform point to world frame
            double wx = robot_x_ + std::cos(robot_theta_) * p.x
                                 - std::sin(robot_theta_) * p.y;
            double wy = robot_y_ + std::sin(robot_theta_) * p.x
                                 + std::cos(robot_theta_) * p.y;
            map_.updateRay(robot_x_, robot_y_, wx, wy);
        }

        // 5. Publish map
        map_pub_->publish(map_.toMsg("map", msg->header.stamp));

        // 6. Publish pose
        publishPose(msg->header.stamp);

        // 7. Broadcast TF map -> odom
        publishTF(msg->header.stamp);

        // 8. Store current scan for next iteration
        prev_points_ = current_points;
    }

    std::vector<Point2D> scanToPoints(
        const sensor_msgs::msg::LaserScan::SharedPtr& msg)
    {
        std::vector<Point2D> points;
        float angle = msg->angle_min;

        for (auto& r : msg->ranges) {
            if (r > msg->range_min && r < msg->range_max) {
                points.push_back({
                    r * std::cos(angle),
                    r * std::sin(angle)
                });
            }
            angle += msg->angle_increment;
        }
        return points;
    }

    void publishPose(rclcpp::Time stamp) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = "map";
        pose.pose.position.x = robot_x_;
        pose.pose.position.y = robot_y_;

        tf2::Quaternion q;
        q.setRPY(0, 0, robot_theta_);
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();

        pose_pub_->publish(pose);
    }

    void publishTF(rclcpp::Time stamp) {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = "map";
        t.child_frame_id = "odom";
        t.transform.translation.x = robot_x_;
        t.transform.translation.y = robot_y_;

        tf2::Quaternion q;
        q.setRPY(0, 0, robot_theta_);
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        tf_broadcaster_->sendTransform(t);
    }

    // Members
    OccupancyMap map_;
    ICP icp_;
    std::vector<Point2D> prev_points_;
    bool has_prev_scan_;
    double robot_x_, robot_y_, robot_theta_;

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamNode>());
    rclcpp::shutdown();
    return 0;
}
