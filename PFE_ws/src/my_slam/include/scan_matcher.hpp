#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <cmath>
#include <limits>
#include "nav_msgs/msg/occupancy_grid.hpp"

using Point2D    = Eigen::Vector2f;
using PointCloud = std::vector<Point2D>;

struct ICPResult
{
    Eigen::Matrix2f R;
    Eigen::Vector2f t;
    float           error;
    bool            converged;
};

class ScanMatcher
{
public:
    struct Config
    {
        int   max_iterations  = 50;
        float tolerance       = 1e-4f;
        float max_dist        = 0.3f;   // tighter than scan-to-scan
        int   min_points      = 20;
        float occupied_thresh = 50.0f;  // min occupancy value to extract as point
    };

    explicit ScanMatcher(const Config& config = Config{}) : cfg_(config) {}

    // Call this every time the map is updated
    void updateMap(const nav_msgs::msg::OccupancyGrid& map)
    {
        map_cloud_.clear();

        const float res    = map.info.resolution;
        const float orig_x = map.info.origin.position.x;
        const float orig_y = map.info.origin.position.y;
        const int   width  = static_cast<int>(map.info.width);
        const int   height = static_cast<int>(map.info.height);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int8_t val = map.data[y * width + x];
                if (val >= static_cast<int8_t>(cfg_.occupied_thresh)) {
                    float wx = orig_x + (x + 0.5f) * res;
                    float wy = orig_y + (y + 0.5f) * res;
                    map_cloud_.push_back({wx, wy});
                }
            }
        }
    }

    // Returns the corrected pose delta — add this to your robot_pose_
    // initial_pose: current robot pose (x, y, theta) used to seed the search
    ICPResult match(const PointCloud& scan_in_base,
                    const Eigen::Vector3f& initial_pose)
    {
        ICPResult result;
        result.R         = Eigen::Matrix2f::Identity();
        result.t         = Eigen::Vector2f::Zero();
        result.error     = std::numeric_limits<float>::max();
        result.converged = false;

        if ((int)map_cloud_.size() < cfg_.min_points ||
            (int)scan_in_base.size() < cfg_.min_points)
            return result;

        // Transform scan into map frame using initial pose
        PointCloud src = transformCloud(scan_in_base, initial_pose);

        for (int iter = 0; iter < cfg_.max_iterations; ++iter)
        {
            // ── Nearest neighbour correspondences ────────────────────────
            std::vector<std::pair<int,int>> correspondences;
            correspondences.reserve(src.size());
            float total_error = 0.0f;

            for (size_t i = 0; i < src.size(); ++i) {
                float best_dist = cfg_.max_dist * cfg_.max_dist;
                int   best_j    = -1;
                for (size_t j = 0; j < map_cloud_.size(); ++j) {
                    float d = (src[i] - map_cloud_[j]).squaredNorm();
                    if (d < best_dist) { best_dist = d; best_j = static_cast<int>(j); }
                }
                if (best_j >= 0) {
                    correspondences.push_back({static_cast<int>(i), best_j});
                    total_error += best_dist;
                }
            }

            if ((int)correspondences.size() < cfg_.min_points) break;

            float mean_error = total_error / correspondences.size();

            // ── Centroids ────────────────────────────────────────────────
            Eigen::Vector2f src_centroid = Eigen::Vector2f::Zero();
            Eigen::Vector2f tgt_centroid = Eigen::Vector2f::Zero();
            for (auto& [i,j] : correspondences) {
                src_centroid += src[i];
                tgt_centroid += map_cloud_[j];
            }
            src_centroid /= (float)correspondences.size();
            tgt_centroid /= (float)correspondences.size();

            // ── Cross-covariance + SVD ───────────────────────────────────
            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            for (auto& [i,j] : correspondences)
                H += (src[i] - src_centroid) * (map_cloud_[j] - tgt_centroid).transpose();

            Eigen::JacobiSVD<Eigen::Matrix2f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f R_iter = svd.matrixV() * svd.matrixU().transpose();

            if (R_iter.determinant() < 0) {
                Eigen::Matrix2f V = svd.matrixV();
                V.col(1) *= -1.0f;
                R_iter = V * svd.matrixU().transpose();
            }

            Eigen::Vector2f t_iter = tgt_centroid - R_iter * src_centroid;

            // ── Apply + accumulate ───────────────────────────────────────
            for (auto& p : src) p = R_iter * p + t_iter;
            result.t = R_iter * result.t + t_iter;
            result.R = R_iter * result.R;

            float delta  = t_iter.norm() + std::abs(std::atan2(R_iter(1,0), R_iter(0,0)));
            result.error = mean_error;

            if (delta < cfg_.tolerance && iter > 0) {
                result.converged = true;
                break;
            }
        }

        return result;
    }

    bool hasMap() const { return !map_cloud_.empty(); }
    int  mapPoints() const { return static_cast<int>(map_cloud_.size()); }

private:
    Config     cfg_;
    PointCloud map_cloud_;   // occupied cells extracted from the grid

    // Transform a cloud from base frame into map frame given pose (x,y,theta)
    static PointCloud transformCloud(const PointCloud& cloud,
                                     const Eigen::Vector3f& pose)
    {
        float cos_h = std::cos(pose.z());
        float sin_h = std::sin(pose.z());

        PointCloud out;
        out.reserve(cloud.size());
        for (const auto& p : cloud) {
            out.push_back({
                pose.x() + cos_h * p.x() - sin_h * p.y(),
                pose.y() + sin_h * p.x() + cos_h * p.y()
            });
        }
        return out;
    }
};