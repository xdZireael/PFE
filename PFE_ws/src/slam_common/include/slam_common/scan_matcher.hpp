#include "nav_msgs/msg/occupancy_grid.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

using Point2D = Eigen::Vector2f;
using PointCloud = std::vector<Point2D>;

struct ICPResult {
  Eigen::Matrix2f R;
  Eigen::Vector2f t;
  float error;
  float mean_error;
  bool converged;
};

// ---------------------------------------------------------------------------
// GridIndex: O(1) approximate nearest-neighbour for 2-D point clouds.
//
// The brute-force search in the original runICP was O(n_src * n_target) per
// iteration — with 360 scan points and ~5 000 map points that is ~1.8 M
// squaredNorm calls per iteration × 30 iterations × 10 Hz = hundreds of
// millions of operations per second on a TurtleBot CPU.  The result: ICP
// takes longer than one scan period, corrections arrive a scan cycle late,
// and the stale EKF update looks exactly like rotation-induced drift.
//
// This index snaps every target point into a grid cell of size `cell_size`.
// Nearest-neighbour lookup checks the 3×3 neighbourhood of cells around the
// query point — O(1) per query, O(n_target) to build, rebuilds only when the
// target cloud changes (i.e. on map update or new keyframe).
// ---------------------------------------------------------------------------
class GridIndex {
public:
  explicit GridIndex(float cell_size = 0.10f) : cell_(cell_size) {}

  void build(const PointCloud &pts) {
    grid_.clear();
    grid_.reserve(pts.size() * 2);
    for (int i = 0; i < static_cast<int>(pts.size()); ++i)
      grid_[key(pts[i])].push_back(i);
    pts_ = &pts;
  }

  int nearest(const Point2D &q, float max_dist) const {
    int best_idx = -1;
    float best_d2 = max_dist * max_dist;
    int cx = cellCoord(q.x());
    int cy = cellCoord(q.y());

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        auto it = grid_.find(pack(cx + dx, cy + dy));
        if (it == grid_.end())
          continue;
        for (int idx : it->second) {
          float d2 = (q - (*pts_)[idx]).squaredNorm();
          if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = idx;
          }
        }
      }
    }
    return best_idx;
  }

private:
  float cell_;
  const PointCloud *pts_ = nullptr;
  std::unordered_map<int64_t, std::vector<int>> grid_;

  int cellCoord(float v) const {
    return static_cast<int>(std::floor(v / cell_));
  }
  int64_t pack(int cx, int cy) const {
    return (static_cast<int64_t>(cx) << 32) | static_cast<uint32_t>(cy);
  }
  int64_t key(const Point2D &p) const {
    return pack(cellCoord(p.x()), cellCoord(p.y()));
  }
};

class ScanMatcher {
public:
  struct Config {
    int max_iterations = 20;
    float tolerance = 1e-3f;
    float max_dist = 1.0f;
    int min_points = 15;
    float occupied_thresh = 50.0f;
    float map_downsample = 0.05f;
    float grid_cell_size = 0.10f;
  };

  ScanMatcher() : cfg_(Config{}) {}
  explicit ScanMatcher(const Config &c) : cfg_(c) {}

  void updateMap(const nav_msgs::msg::OccupancyGrid &map) {
    map_cloud_.clear();

    const float res = map.info.resolution;
    const float orig_x = map.info.origin.position.x;
    const float orig_y = map.info.origin.position.y;
    const int width = static_cast<int>(map.info.width);
    const int height = static_cast<int>(map.info.height);

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

    map_index_ = GridIndex(cfg_.grid_cell_size);
    map_index_.build(map_cloud_);
  }

  ICPResult match(const PointCloud &scan_in_base,
                  const Eigen::Vector3f &initial_pose) {
    if ((int)map_cloud_.size() < cfg_.min_points ||
        (int)scan_in_base.size() < cfg_.min_points)
      return makeFailedResult();

    PointCloud src = transformCloud(scan_in_base, initial_pose);
    return runICP(src, map_cloud_, map_index_);
  }

  ICPResult matchClouds(const PointCloud &source, const PointCloud &target,
                        const Eigen::Vector3f &initial_guess) {
    if ((int)source.size() < cfg_.min_points ||
        (int)target.size() < cfg_.min_points)
      return makeFailedResult();

    GridIndex tmp_index(cfg_.grid_cell_size);
    tmp_index.build(target);

    PointCloud src = transformCloud(source, initial_guess);
    return runICP(src, target, tmp_index);
  }

  bool hasMap() const { return !map_cloud_.empty(); }
  int mapPoints() const { return static_cast<int>(map_cloud_.size()); }

private:
  Config cfg_;
  PointCloud map_cloud_;
  GridIndex
      map_index_; // persistent index over map_cloud_, rebuilt on updateMap()

  // FIX (performance): correspondence search is now O(1) per point via
  // GridIndex instead of O(n_target) brute-force.  Full iteration cost is
  // O(n_src) instead of O(n_src × n_target), cutting per-scan ICP time by 1–2
  // orders of magnitude.
  ICPResult runICP(PointCloud src, const PointCloud &target,
                   const GridIndex &index) {
    ICPResult result;
    result.R = Eigen::Matrix2f::Identity();
    result.t = Eigen::Vector2f::Zero();
    result.error = std::numeric_limits<float>::max();
    result.mean_error = std::numeric_limits<float>::max();
    result.converged = false;

    for (int iter = 0; iter < cfg_.max_iterations; ++iter) {
      std::vector<std::pair<int, int>> correspondences;
      correspondences.reserve(src.size());
      float total_error = 0.0f;

      for (size_t i = 0; i < src.size(); ++i) {
        // O(1) lookup — checks only the 9 neighbouring grid cells.
        int best_j = index.nearest(src[i], cfg_.max_dist);
        if (best_j >= 0) {
          float d2 = (src[i] - target[best_j]).squaredNorm();
          correspondences.push_back({static_cast<int>(i), best_j});
          total_error += d2;
        }
      }

      // FIX: only update mean_error when we have enough correspondences.
      // Previously mean_error was written every iteration including the
      // early-exit path (too few correspondences), so a failed ICP could
      // report a misleadingly low error from a prior iteration.
      if ((int)correspondences.size() < cfg_.min_points)
        break;

      float mean_error = total_error / correspondences.size();
      result.error = total_error;
      result.mean_error =
          std::sqrt(mean_error); // now only set on valid iterations

      Eigen::Vector2f src_centroid = Eigen::Vector2f::Zero();
      Eigen::Vector2f tgt_centroid = Eigen::Vector2f::Zero();
      for (auto &[i, j] : correspondences) {
        src_centroid += src[i];
        tgt_centroid += target[j];
      }
      src_centroid /= static_cast<float>(correspondences.size());
      tgt_centroid /= static_cast<float>(correspondences.size());

      Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
      for (auto &[i, j] : correspondences)
        H += (src[i] - src_centroid) * (target[j] - tgt_centroid).transpose();

      Eigen::JacobiSVD<Eigen::Matrix2f> svd(H, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV);
      Eigen::Matrix2f R_iter = svd.matrixV() * svd.matrixU().transpose();

      if (R_iter.determinant() < 0) {
        Eigen::Matrix2f V = svd.matrixV();
        V.col(1) *= -1.0f;
        R_iter = V * svd.matrixU().transpose();
      }

      Eigen::Vector2f t_iter = tgt_centroid - R_iter * src_centroid;

      for (auto &p : src)
        p = R_iter * p + t_iter;

      result.t = R_iter * result.t + t_iter;
      result.R = R_iter * result.R;

      float delta =
          t_iter.norm() + std::abs(std::atan2(R_iter(1, 0), R_iter(0, 0)));
      if (delta < cfg_.tolerance && iter > 0) {
        result.converged = true;
        break;
      }
    }

    return result;
  }

  static ICPResult makeFailedResult() {
    ICPResult r;
    r.R = Eigen::Matrix2f::Identity();
    r.t = Eigen::Vector2f::Zero();
    r.error = std::numeric_limits<float>::max();
    r.mean_error = std::numeric_limits<float>::max();
    r.converged = false;
    return r;
  }

  static PointCloud transformCloud(const PointCloud &cloud,
                                   const Eigen::Vector3f &pose) {
    float cos_h = std::cos(pose.z());
    float sin_h = std::sin(pose.z());
    PointCloud out;
    out.reserve(cloud.size());
    for (const auto &p : cloud) {
      out.push_back({pose.x() + cos_h * p.x() - sin_h * p.y(),
                     pose.y() + sin_h * p.x() + cos_h * p.y()});
    }
    return out;
  }
};