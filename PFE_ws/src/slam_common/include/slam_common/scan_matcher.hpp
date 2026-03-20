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
    float           mean_error;
    bool            converged;
};

class ScanMatcher
{
public:
    struct Config
    {
        int   max_iterations  = 30;
        float tolerance       = 1e-3f;
        float max_dist        = 1.0f;
        int   min_points      = 15;
        float occupied_thresh = 50.0f;
        float map_downsample  = 0.05f;
    };

    ScanMatcher()                        : cfg_(Config{}) {}
    explicit ScanMatcher(const Config& c): cfg_(c) {}

    void updateMap(const nav_msgs::msg::OccupancyGrid& map)
    {
        map_cloud_.clear();

        const float res    = map.info.resolution;
        const float orig_x = map.info.origin.position.x;
        const float orig_y = map.info.origin.position.y;
        const int   width  = static_cast<int>(map.info.width);
        const int   height = static_cast<int>(map.info.height);

        int skip = std::max(1, static_cast<int>(cfg_.map_downsample / res));

        for (int y = 0; y < height; y += skip) {
            for (int x = 0; x < width; x += skip) {
                int8_t val = map.data[y * width + x];
                if (val > 0 && val >= static_cast<int8_t>(cfg_.occupied_thresh)) {
                    float wx = orig_x + (x + 0.5f) * res;
                    float wy = orig_y + (y + 0.5f) * res;
                    map_cloud_.push_back({wx, wy});
                }
            }
        }
    }

    ICPResult match(const PointCloud& scan_in_base,
                    const Eigen::Vector3f& initial_pose)
    {
        if ((int)map_cloud_.size() < cfg_.min_points ||
            (int)scan_in_base.size() < cfg_.min_points)
            return makeFailedResult();

        PointCloud src = transformCloud(scan_in_base, initial_pose);
        return runICP(src, map_cloud_);
    }

    ICPResult matchClouds(const PointCloud& source,
                          const PointCloud& target,
                          const Eigen::Vector3f& initial_guess)
    {
        if ((int)source.size() < cfg_.min_points ||
            (int)target.size() < cfg_.min_points)
            return makeFailedResult();

        PointCloud src = transformCloud(source, initial_guess);
        return runICP(src, target);
    }

    bool hasMap()    const { return !map_cloud_.empty(); }
    int  mapPoints() const { return static_cast<int>(map_cloud_.size()); }

private:
    Config     cfg_;
    PointCloud map_cloud_;

    ICPResult runICP(PointCloud src, const PointCloud& target)
    {
        ICPResult result;
        result.R          = Eigen::Matrix2f::Identity();
        result.t          = Eigen::Vector2f::Zero();
        result.error      = std::numeric_limits<float>::max();
        result.mean_error = std::numeric_limits<float>::max();
        result.converged  = false;

        for (int iter = 0; iter < cfg_.max_iterations; ++iter)
        {
            std::vector<std::pair<int,int>> correspondences;
            correspondences.reserve(src.size());
            float total_error = 0.0f;

            for (size_t i = 0; i < src.size(); ++i) {
                float best_dist = cfg_.max_dist * cfg_.max_dist;
                int   best_j    = -1;
                for (size_t j = 0; j < target.size(); ++j) {
                    float d = (src[i] - target[j]).squaredNorm();
                    if (d < best_dist) { best_dist = d; best_j = static_cast<int>(j); }
                }
                if (best_j >= 0) {
                    correspondences.push_back({static_cast<int>(i), best_j});
                    total_error += best_dist;
                }
            }

            if ((int)correspondences.size() < cfg_.min_points) break;

            float mean_error = total_error / correspondences.size();

            Eigen::Vector2f src_centroid = Eigen::Vector2f::Zero();
            Eigen::Vector2f tgt_centroid = Eigen::Vector2f::Zero();
            for (auto& [i,j] : correspondences) {
                src_centroid += src[i];
                tgt_centroid += target[j];
            }
            src_centroid /= static_cast<float>(correspondences.size());
            tgt_centroid /= static_cast<float>(correspondences.size());

            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            for (auto& [i,j] : correspondences)
                H += (src[i] - src_centroid) * (target[j] - tgt_centroid).transpose();

            Eigen::JacobiSVD<Eigen::Matrix2f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f R_iter = svd.matrixV() * svd.matrixU().transpose();

            if (R_iter.determinant() < 0) {
                Eigen::Matrix2f V = svd.matrixV();
                V.col(1) *= -1.0f;
                R_iter = V * svd.matrixU().transpose();
            }

            Eigen::Vector2f t_iter = tgt_centroid - R_iter * src_centroid;

            for (auto& p : src) p = R_iter * p + t_iter;

            result.t = R_iter * result.t + t_iter;
            result.R = R_iter * result.R;

            float delta   = t_iter.norm() + std::abs(std::atan2(R_iter(1,0), R_iter(0,0)));
            result.error      = total_error;
            result.mean_error = std::sqrt(mean_error);

            if (delta < cfg_.tolerance && iter > 0) {
                result.converged = true;
                break;
            }
        }

        return result;
    }

    static ICPResult makeFailedResult()
    {
        ICPResult r;
        r.R          = Eigen::Matrix2f::Identity();
        r.t          = Eigen::Vector2f::Zero();
        r.error      = std::numeric_limits<float>::max();
        r.mean_error = std::numeric_limits<float>::max();
        r.converged  = false;
        return r;
    }

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