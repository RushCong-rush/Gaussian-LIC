/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "renderer.h"
#include <filesystem>
#include <fstream>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render(const std::shared_ptr<Camera>& viewpoint_camera,
       std::shared_ptr<GaussianModel> pc,
       torch::Tensor& bg_color,
       bool use_trained_exposure,
       bool no_color,
       float scaling_modifier,
       bool apply_valid_mask, bool training_visibility, torch::Tensor* visibility_weights)
{
    auto screenspace_points = torch::zeros_like(pc->getXYZ(), torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));

    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);  // w / (2 * fx)
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);  // h / (2 * fy)
    bool prefiltered = false;
    bool debug = false;
    GaussianRasterizationSettings raster_settings(
        viewpoint_camera->image_height_,
        viewpoint_camera->image_width_,
        tanfovx,
        tanfovy,
        viewpoint_camera->limx_neg_,
        viewpoint_camera->limx_pos_,
        viewpoint_camera->limy_neg_,
        viewpoint_camera->limy_pos_,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->sh_degree_,
        viewpoint_camera->camera_center_,
        prefiltered,
        debug,
        no_color,
        pc->lambda_erank_,
        viewpoint_camera->is_equirectangular_
    );
    if (apply_valid_mask)
        raster_settings.valid_mask_ = no_color
            ? viewpoint_camera->extension_raster_mask_ : viewpoint_camera->raster_mask_;
    raster_settings.normalize_depth_gradient_ = pc->normalize_depth_gradient_;
    if (visibility_weights || (training_visibility &&
        (pc->depth_visibility_lidar_strength_ > 0.0 || pc->depth_visibility_dap_strength_ > 0.0)))
    {
        auto depth = viewpoint_camera->diagnostic_depth_.to(torch::kCUDA).squeeze();
        auto lidar = viewpoint_camera->lidar_valid_mask_.to(torch::kCUDA).squeeze().to(torch::kBool);
        auto valid = torch::isfinite(depth) & (depth > 0.f);
        if (viewpoint_camera->raster_mask_.defined()) valid &= viewpoint_camera->raster_mask_;
        depth = torch::where(valid, depth, torch::zeros_like(depth));
        // DAP receives a wider tolerance and a weaker maximum attenuation.
        auto margin = torch::where(lidar, torch::clamp_min(depth * .1f, .25f),
                                          torch::clamp_min(depth * .2f, .5f));
        auto strength = training_visibility
            ? torch::where(lidar, torch::full_like(depth, pc->depth_visibility_lidar_strength_),
                                  torch::full_like(depth, pc->depth_visibility_dap_strength_))
            : torch::zeros_like(depth);
        raster_settings.depth_visibility_ = torch::stack({depth, margin, strength}).contiguous();
        if (visibility_weights)
        {
            *visibility_weights = torch::zeros_like(raster_settings.depth_visibility_);
            raster_settings.visibility_weights_ = *visibility_weights;
        }
    }
    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();  // (n, 3)
    auto means2D = screenspace_points;  // (n, 3)
    auto opacity = pc->getOpacity();  // (n, 1) 0-1
    auto scales = pc->getScaling();  // (n, 3) 0-inf
    auto rotations = pc->getRotation();  // (n, 4)
    torch::Tensor dc = pc->getFeaturesDc();  // (n, 1, 3)
    torch::Tensor shs = pc->getFeaturesRest();  // (n, 15, 3)
    torch::Tensor colors_precomp; 
    torch::Tensor cov3D_precomp;

    auto rasterizer_result = rasterizer.forward(
                                    means3D,
                                    means2D,
                                    opacity,
                                    dc,
                                    shs,
                                    colors_precomp,
                                    scales,
                                    rotations,
                                    cov3D_precomp);
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);
    auto rendered_depth = std::get<2>(rasterizer_result);
    auto rendered_final_T = std::get<3>(rasterizer_result);

    return std::make_tuple(
        rendered_image,
        rendered_depth,   
        rendered_final_T,
        screenspace_points, 
        radii > 0,          
        radii
    );
}


void saveVisibilityDiagnosis(const std::shared_ptr<Camera>& camera,
    std::shared_ptr<GaussianModel> pc, torch::Tensor& background,
    const std::string& directory, const std::string& phase)
{
    const int frame = camera->frame_index_;
    if (!pc->diagnose_depth_visibility_ || directory.empty() ||
        !(frame == 649 || frame == 699 || frame == 799 || frame == 934 || frame == 999 || frame == 1034)) return;
    torch::NoGradGuard guard;
    torch::Tensor weights;
    render(camera, pc, background, pc->apply_exposure_, false, 1.f, true, false, &weights);
    auto cpu = weights.cpu().contiguous();
    auto depth = camera->diagnostic_depth_.cpu().squeeze().contiguous();
    auto lidar = camera->lidar_valid_mask_.cpu().squeeze().to(torch::kBool);
    auto valid = torch::isfinite(depth) & (depth > 0.f);
    if (camera->raster_mask_.defined()) valid &= camera->raster_mask_.cpu();
    const std::string folder = directory + "/visibility";
    std::filesystem::create_directories(folder);
    auto packed = torch::cat({cpu, depth.unsqueeze(0), lidar.to(torch::kFloat32).unsqueeze(0),
                              valid.to(torch::kFloat32).unsqueeze(0)}, 0).contiguous();
    std::ofstream raw(folder + "/" + phase + "_" + camera->image_name_ + ".f32", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(packed.data_ptr<float>()), packed.numel() * sizeof(float));
    const std::string path = folder + "/contributions.csv";
    const bool header = !std::filesystem::exists(path);
    std::ofstream csv(path, std::ios::app);
    if (header) csv << "phase,frame,width,height,source,pixels,front,near,behind,total_alpha\n";
    for (int source = 0; source < 2; ++source)
    {
        auto mask = valid & (source == 0 ? lidar : ~lidar);
        const auto count = mask.sum().item<int64_t>();
        if (count == 0) continue;
        auto means = cpu.index({torch::indexing::Slice(), mask}).mean(1);
        csv << phase << ',' << frame << ',' << depth.size(1) << ',' << depth.size(0) << ','
            << (source == 0 ? "LiDAR" : "DAP") << ',' << count;
        for (int band = 0; band < 3; ++band) csv << ',' << means[band].item<float>();
        csv << ',' << means.sum().item<float>() << '\n';
    }
}
