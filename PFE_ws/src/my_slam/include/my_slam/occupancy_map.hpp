#pragma once
#include <vector>
#include <cmath>
#include <nav_msgs/msg/occupancy_grid.hpp>

class OccupancyMap {
public:
    int width, height;
    double resolution;    // meters per cell
    double origin_x, origin_y;

    // Log-odds values per cell (more robust than 0/1)
    std::vector<double> log_odds;

    // Constants
    const double L_OCC  =  0.85;   // log-odds to add when cell is hit
    const double L_FREE = -0.4;    // log-odds to add when cell is passed through
    const double L_MAX  =  3.5;
    const double L_MIN  = -3.5;

    OccupancyMap(int w, int h, double res, double ox, double oy)
    : width(w), height(h), resolution(res), origin_x(ox), origin_y(oy),
      log_odds(w * h, 0.0) {}

    // Convert world coords to grid cell
    int worldToCell(double wx, double wy) const {
        int cx = static_cast<int>((wx - origin_x) / resolution);
        int cy = static_cast<int>((wy - origin_y) / resolution);
        if (cx < 0 || cx >= width || cy < 0 || cy >= height) return -1;
        return cy * width + cx;
    }

    // Bresenham ray tracing — marks free cells along ray, occupied at endpoint
    void updateRay(double robot_x, double robot_y,
                   double hit_x,   double hit_y) {

        // Trace the ray and mark free cells
        int x0 = (robot_x - origin_x) / resolution;
        int y0 = (robot_y - origin_y) / resolution;
        int x1 = (hit_x   - origin_x) / resolution;
        int y1 = (hit_y   - origin_y) / resolution;

        // Bresenham line algorithm
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = (dx > dy ? dx : -dy) / 2;

        int x = x0, y = y0;
        while (true) {
            if (x != x1 || y != y1) {  // free cell along ray
                int idx = y * width + x;
                if (idx >= 0 && idx < (int)log_odds.size())
                    log_odds[idx] = std::max(L_MIN, log_odds[idx] + L_FREE);
            }
            if (x == x1 && y == y1) break;
            int e2 = err;
            if (e2 > -dx) { err -= dy; x += sx; }
            if (e2 <  dy) { err += dx; y += sy; }
        }

        // Mark hit cell as occupied
        int hit_idx = worldToCell(hit_x, hit_y);
        if (hit_idx >= 0)
            log_odds[hit_idx] = std::min(L_MAX, log_odds[hit_idx] + L_OCC);
    }

    // Convert to ROS OccupancyGrid message
    nav_msgs::msg::OccupancyGrid toMsg(const std::string& frame_id,
                                       rclcpp::Time stamp) const {
        nav_msgs::msg::OccupancyGrid msg;
        msg.header.frame_id = frame_id;
        msg.header.stamp = stamp;
        msg.info.resolution = resolution;
        msg.info.width = width;
        msg.info.height = height;
        msg.info.origin.position.x = origin_x;
        msg.info.origin.position.y = origin_y;

        msg.data.resize(width * height);
        for (int i = 0; i < width * height; i++) {
            if      (log_odds[i] >  0.1) msg.data[i] = 100;   // occupied
            else if (log_odds[i] < -0.1) msg.data[i] = 0;     // free
            else                          msg.data[i] = -1;    // unknown
        }
        return msg;
    }
};
