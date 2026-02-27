#pragma once
#include <vector>
#include <cmath>
#include <limits>

struct Point2D {
    double x, y;
};

struct Transform2D {
    double dx, dy, dtheta;
};

class ICP {
public:
    // Max iterations and convergence threshold
    int max_iterations = 50;
    double tolerance = 1e-4;

    Transform2D align(
        const std::vector<Point2D>& source,   // current scan
        const std::vector<Point2D>& target    // previous scan
    ) {
        Transform2D T = {0, 0, 0};
        std::vector<Point2D> src = source;

        for (int iter = 0; iter < max_iterations; iter++) {

            // 1. Find nearest neighbors
            std::vector<std::pair<Point2D, Point2D>> pairs;
            for (auto& s : src) {
                double best_dist = std::numeric_limits<double>::max();
                Point2D best_t;
                for (auto& t : target) {
                    double d = std::pow(s.x - t.x, 2) + std::pow(s.y - t.y, 2);
                    if (d < best_dist) {
                        best_dist = d;
                        best_t = t;
                    }
                }
                if (best_dist < 1.0) // max correspondence distance
                    pairs.push_back({s, best_t});
            }

            if (pairs.empty()) break;

            // 2. Compute centroids
            Point2D cs = {0,0}, ct = {0,0};
            for (auto& [s, t] : pairs) {
                cs.x += s.x; cs.y += s.y;
                ct.x += t.x; ct.y += t.y;
            }
            cs.x /= pairs.size(); cs.y /= pairs.size();
            ct.x /= pairs.size(); ct.y /= pairs.size();

            // 3. Compute rotation using SVD-like cross correlation
            double sxx=0, sxy=0, syx=0, syy=0;
            for (auto& [s, t] : pairs) {
                double sx = s.x - cs.x, sy = s.y - cs.y;
                double tx = t.x - ct.x, ty = t.y - ct.y;
                sxx += sx * tx; sxy += sx * ty;
                syx += sy * tx; syy += sy * ty;
            }
            double dtheta = std::atan2(syx - sxy, sxx + syy);

            // 4. Compute translation
            double cos_t = std::cos(dtheta), sin_t = std::sin(dtheta);
            double dx = ct.x - (cos_t * cs.x - sin_t * cs.y);
            double dy = ct.y - (sin_t * cs.x + cos_t * cs.y);

            // 5. Apply transform to source
            for (auto& p : src) {
                double nx = cos_t * p.x - sin_t * p.y + dx;
                double ny = sin_t * p.x + cos_t * p.y + dy;
                p.x = nx; p.y = ny;
            }

            // 6. Accumulate transform
            T.dx += dx; T.dy += dy; T.dtheta += dtheta;

            // 7. Check convergence
            if (std::sqrt(dx*dx + dy*dy) < tolerance && std::abs(dtheta) < tolerance)
                break;
        }

        return T;
    }
};
