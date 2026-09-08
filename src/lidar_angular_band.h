#pragma once

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

// Equal-solid-angle vertical crop in the LiDAR frame, queried in camera coordinates.
struct LidarAngularBand
{
    double fraction = 1.0;
    double sin_lower = -1.0;
    double sin_upper = 1.0;
    Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();

    LidarAngularBand() = default;
    LidarAngularBand(double keep, double lower_deg, double upper_deg,
                     const Eigen::Vector3d& z_in_camera,
                     const Eigen::Vector3d& origin_in_camera)
        : fraction(keep), axis(z_in_camera), origin(origin_in_camera)
    {
        if (!(keep > 0.0 && keep <= 1.0) ||
            !(lower_deg >= -90.0 && upper_deg <= 90.0 && lower_deg < upper_deg) ||
            !axis.allFinite() || !origin.allFinite() || std::abs(axis.norm() - 1.0) > 1e-6)
            throw std::invalid_argument("Invalid mapping LiDAR angular band");
        const double low = std::sin(lower_deg * M_PI / 180.0);
        const double high = std::sin(upper_deg * M_PI / 180.0);
        // Removing 10% of the original solid angle trims 5% from each end.
        const double center = (low + high) * 0.5;
        const double half = (high - low) * keep * 0.5;
        sin_lower = center - half;
        sin_upper = center + half;
    }

    bool contains(const Eigen::Vector3d& point_camera) const
    {
        if (fraction == 1.0) return true;
        const Eigen::Vector3d ray = point_camera - origin;
        const double sine = axis.dot(ray) / ray.norm();
        return sine >= sin_lower && sine <= sin_upper;
    }
};
