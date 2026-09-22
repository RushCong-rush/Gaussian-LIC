#pragma once

#include "gaussian.h"

// Export-only cameras and annotations; never modify the Gaussian map or poses.
class ThirdPersonExport
{
public:
    ThirdPersonExport(const std::string& result_path, const Eigen::Vector3d& front_axis_camera);
    void addFrame(const std::shared_ptr<Camera>& camera, double timestamp);
    void saveOnline(const std::shared_ptr<GaussianModel>& map);
    void saveFinal(const std::shared_ptr<GaussianModel>& map);

private:
    void saveFrame(size_t index, const std::shared_ptr<GaussianModel>& map,
                   const std::string& phase);
    std::string directory_;
    std::vector<std::shared_ptr<Camera>> views_;
    std::vector<Eigen::Vector3d> centers_;
    std::vector<Eigen::Vector3d> front_axes_;
    Eigen::Vector3d front_axis_camera_;
    Eigen::Matrix<double, 3, Eigen::Dynamic> rays_;
    Eigen::Vector3d heading_ = Eigen::Vector3d::UnitX();
    double last_timestamp_ = 0.0;
    std::ofstream poses_;
};
