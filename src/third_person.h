#pragma once

#include "gaussian.h"

// Export-only cameras and annotations; never modify the Gaussian map or poses.
class ThirdPersonExport
{
public:
    explicit ThirdPersonExport(const std::string& result_path);
    void addFrame(const std::shared_ptr<Camera>& camera);
    void saveOnline(const std::shared_ptr<GaussianModel>& map);
    void saveFinal(const std::shared_ptr<GaussianModel>& map);

private:
    void saveFrame(size_t index, const std::shared_ptr<GaussianModel>& map,
                   const std::string& phase);
    std::string directory_;
    std::vector<std::shared_ptr<Camera>> views_;
    std::vector<Eigen::Vector3d> centers_;
    Eigen::Vector3d heading_ = Eigen::Vector3d::UnitX();
    std::ofstream poses_;
};
