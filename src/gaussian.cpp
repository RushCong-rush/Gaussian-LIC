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

#include "gaussian.h"
#include "tensor_utils.h"
#include "loss_utils.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <torch/script.h>
#include <memory>

namespace fs = std::filesystem;

struct PixelPosition 
{
    int u, v;
};

std::vector<PixelPosition> selectFromDepthCompletion(const cv::Mat& depth_A, const cv::Mat& depth_B, int patch_size = 20) 
{
    CV_Assert(depth_A.size() == depth_B.size());
    CV_Assert(depth_A.type() == depth_B.type());
    
    int H = depth_A.rows;
    int W = depth_A.cols;
    std::vector<PixelPosition> result;
    result.reserve((H / patch_size) * (W / patch_size));

    for (int i = 0; i < H; i += patch_size) 
    {
        for (int j = 0; j < W; j += patch_size) 
        {
            int h_end = std::min(i + patch_size, H);
            int w_end = std::min(j + patch_size, W);
            
            bool has_valid_A = false;
            bool has_valid_B = false;
            float min_val = std::numeric_limits<float>::max();
            PixelPosition min_pos;
            
            for (int y = i; y < h_end; ++y) 
            {
                const float* ptr_A = depth_A.ptr<float>(y);
                const float* ptr_B = depth_B.ptr<float>(y);
                
                for (int x = j; x < w_end; ++x) 
                {
                    if (ptr_A[x] > 0) 
                    {
                        has_valid_A = true;
                        y = h_end;
                        break;
                    }
                    
                    if (ptr_B[x] > 0) 
                    {
                        has_valid_B = true;
                        if (ptr_B[x] < min_val) 
                        {
                            min_val = ptr_B[x];
                            min_pos = {x, y};
                        }
                    }
                }
            }
            
            if (has_valid_A || !has_valid_B) 
            {
                continue;
            }
            
            result.push_back(min_pos);
        }
    }
    
    return result;
}

void Dataset::addFrame(Frame& cur_frame)
{
    /// image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_bgr = cv_ptr->image;
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);  // 0-255
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);  // 0-1

    /// depth
    cv_bridge::CvImagePtr dp_ptr;
    dp_ptr = cv_bridge::toCvCopy(cur_frame.depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    cv::Mat lidar_depth = dp_ptr->image.clone();  // metric float32, sparse LiDAR measurements
    cv::Mat depth_map = lidar_depth.clone();
    if (cur_frame.dap_depth_msg)
    {
        cv::Mat dap_depth;
        auto dap_ptr = cv_bridge::toCvCopy(cur_frame.dap_depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        dap_depth = dap_ptr->image.clone();
        if (dap_depth.size() != depth_map.size())
            throw std::runtime_error("DAP ERP depth size does not match the input image");

        // Align the offline DAP metric to the current LiDAR scale, then keep
        // measured LiDAR pixels as hard anchors in the fused depth image.
        std::vector<float> scale_ratios;
        scale_ratios.reserve(static_cast<size_t>(depth_map.total() / 20));
        for (int y = 0; y < depth_map.rows; ++y)
        {
            const float* lidar_row = lidar_depth.ptr<float>(y);
            const float* dap_row = dap_depth.ptr<float>(y);
            for (int x = 0; x < depth_map.cols; ++x)
            {
                if (lidar_row[x] > 0.0f && dap_row[x] > 0.0f && std::isfinite(lidar_row[x]) && std::isfinite(dap_row[x]))
                    scale_ratios.push_back(lidar_row[x] / dap_row[x]);
            }
        }
        if (scale_ratios.empty())
            throw std::runtime_error("No valid LiDAR/DAP overlap for depth scale alignment");
        const auto middle = scale_ratios.begin() + scale_ratios.size() / 2;
        std::nth_element(scale_ratios.begin(), middle, scale_ratios.end());
        const float dap_scale = *middle;
        depth_map = dap_depth * dap_scale;
        lidar_depth.copyTo(depth_map, lidar_depth > 0.0f);
        if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
            std::cout << std::fixed << std::setprecision(4)
                      << "[DepthFusion] DAP scale " << dap_scale
                      << ", LiDAR anchors " << scale_ratios.size() << std::endl;

        if (!diagnosis_dir_.empty() && equirectangular_ && (all_frame_num_ % (select_every_k_frame_ * 5) == 0))
        {
            fs::create_directories(diagnosis_dir_);
            auto colorize = [](const cv::Mat& depth, double max_depth) {
                cv::Mat clipped;
                cv::max(depth, 0.0, clipped);
                cv::min(clipped, max_depth, clipped);
                clipped.convertTo(clipped, CV_8UC1, 255.0 / max_depth);
                cv::Mat colored;
                cv::applyColorMap(clipped, colored, cv::COLORMAP_TURBO);
                colored.setTo(cv::Scalar(0, 0, 0), depth <= 0.0f);
                return colored;
            };
            const std::string stamp = std::to_string(cur_frame.image_msg->header.stamp.toNSec());
            cv::imwrite(diagnosis_dir_ + "/dap_" + stamp + ".png", colorize(dap_depth, 80.0));
            cv::imwrite(diagnosis_dir_ + "/fused_" + stamp + ".png", colorize(depth_map, 80.0));
        }
    }

    /// pose
    Eigen::Quaterniond q_wc;
    Eigen::Vector3d t_wc;
    tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
    R_wc_.push_back(q_wc.toRotationMatrix());
    t_wc_.push_back(t_wc);

    /// point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
    for (const auto& pt : cloud->points)
    {
        pointcloud_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        pointcolor_.emplace_back(Eigen::Vector3d(pt.r, pt.g, pt.b) / 255.0);
        Eigen::Matrix3d R_cw = q_wc.toRotationMatrix().transpose();
        Eigen::Vector3d t_cw = - R_cw * t_wc;
        Eigen::Vector3d pt_c = R_cw * pointcloud_.back() + t_cw;
        if (!equirectangular_)
            assert(pt_c(2) > 0);
        pointdepth_.push_back(static_cast<float>(
            equirectangular_ ? pt_c.norm() : pt_c(2)));
    }

    /// train & test
    int width = image_rgb.cols, height = image_rgb.rows;
    if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
    {
        is_keyframe_current_ = true;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        if (depth_completion_)
        {
            cv::Mat completed_depth;  // metric float32
            completed_depth = depth_completer_->complete(image_rgb, depth_map);

            cv::Mat mask_known = depth_map > 0;  // 0/255 uint8
            cv::Mat completed_depth_known;
            completed_depth.copyTo(completed_depth_known, mask_known);
            cv::Mat depth_difference = completed_depth_known - depth_map;
            double mean_depth_difference = cv::mean(depth_difference, mask_known)[0];

            if (std::abs(mean_depth_difference) < 0.1)
            {
                // wanted_depth：non-edge && positive
                cv::Mat depth_gradient_x, depth_gradient_y;
                cv::Sobel(completed_depth, depth_gradient_x, CV_32F, 1, 0, 3);
                cv::Sobel(completed_depth, depth_gradient_y, CV_32F, 0, 1, 3);
                cv::Mat depth_edges;
                cv::magnitude(depth_gradient_x, depth_gradient_y, depth_edges);
                double edge_threshold = 0.1;
                cv::Mat mask_not_edges = depth_edges < edge_threshold;  // 0/255 uint8
                completed_depth -= mean_depth_difference;
                cv::Mat mask = (completed_depth > 0) & mask_not_edges;  // 0/255 uint8
                cv::Mat wanted_depth;
                completed_depth.copyTo(wanted_depth, mask);

                // select
                std::vector<PixelPosition> new_positions = selectFromDepthCompletion(depth_map, wanted_depth, patch_size_);
                for (const auto& pt : new_positions) 
                {
                    int u = pt.u, v = pt.v;
                    float depth = wanted_depth.at<float>(v, u);
                    assert(depth > 0);
                    if (depth > max_depth_) continue;

                    cv::Vec3f color = image_rgb.at<cv::Vec3f>(v, u);
                    Eigen::Vector3d eigen_color(color[0], color[1], color[2]);

                    Eigen::Vector3d cam_point((u - cx_) * depth / fx_, 
                                            (v - cy_) * depth / fy_, 
                                            depth);
                    Eigen::Vector3d world_point = q_wc * cam_point + t_wc;

                    pointcloud_.emplace_back(world_point);
                    pointcolor_.emplace_back(eigen_color);
                    pointdepth_.emplace_back(static_cast<float>(depth));
                }
            }
            else
            {
                // std::cout << "[bef vs aft diff]: " << mean_depth_difference << " m" << std::endl;
            }
        }
        else if (cur_frame.dap_depth_msg && equirectangular_)
        {
            // Use the fused depth only in LiDAR blind patches. This preserves
            // the original sparse LiDAR initialization while adding a small,
            // controlled number of DAP-supported ERP Gaussians.
            cv::Mat depth_gradient_x, depth_gradient_y;
            cv::Sobel(depth_map, depth_gradient_x, CV_32F, 1, 0, 3);
            cv::Sobel(depth_map, depth_gradient_y, CV_32F, 0, 1, 3);
            cv::Mat depth_edges;
            cv::magnitude(depth_gradient_x, depth_gradient_y, depth_edges);
            cv::Mat mask_not_edges = depth_edges < 0.1;
            cv::Mat wanted_depth;
            depth_map.copyTo(wanted_depth, (depth_map > 0) & mask_not_edges & (depth_map < max_depth_));

            std::vector<PixelPosition> new_positions = selectFromDepthCompletion(lidar_depth, wanted_depth, patch_size_);
            for (const auto& pt : new_positions)
            {
                const int u = pt.u, v = pt.v;
                const float depth = wanted_depth.at<float>(v, u);
                if (depth <= 0.0f || depth > max_depth_) continue;

                const double longitude = (static_cast<double>(u) / width - 0.5) * 2.0 * M_PI;
                const double latitude = (0.5 - static_cast<double>(v) / height) * M_PI;
                const double cos_latitude = std::cos(latitude);
                Eigen::Vector3d cam_point(
                    depth * cos_latitude * std::sin(longitude),
                    -depth * std::sin(latitude),
                    depth * cos_latitude * std::cos(longitude));
                Eigen::Vector3d world_point = q_wc * cam_point + t_wc;
                cv::Vec3f color = image_rgb.at<cv::Vec3f>(v, u);
                pointcloud_.emplace_back(world_point);
                pointcolor_.emplace_back(Eigen::Vector3d(color[0], color[1], color[2]));
                pointdepth_.emplace_back(depth);
            }
        }

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU, true);
        
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "train_" + formatted_str + ".jpg";

        cam->setCameraModel(equirectangular_);
        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU);

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "test_" + formatted_str + ".jpg";

        cam->setCameraModel(equirectangular_);
        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;
}

GaussianModel::GaussianModel(const Params& prm)
{
    sh_degree_ = prm.sh_degree;
    white_background_ = prm.white_background;
    random_background_ = prm.random_background;
    convert_SHs_python_ = prm.convert_SHs_python;
    compute_cov3D_python_ = prm.compute_cov3D_python;
    lambda_erank_ = prm.lambda_erank;
    scaling_scale_ = prm.scaling_scale;

    position_lr_ = prm.position_lr;
    feature_lr_ = prm.feature_lr;
    opacity_lr_ = prm.opacity_lr;
    scaling_lr_ = prm.scaling_lr;
    rotation_lr_ = prm.rotation_lr;
    lambda_dssim_ = prm.lambda_dssim;
    optimize_depth_ = prm.optimize_depth;
    lambda_depth_ = prm.lambda_depth;
    iteration_decay_ = prm.iteration_decay;

    apply_exposure_ = prm.apply_exposure;
    exposure_lr_ = prm.exposure_lr;
    skybox_points_num_ = prm.skybox_points_num;
    skybox_radius_ = prm.skybox_radius;

    auto device_type = torch::kCUDA;
    GAUSSIAN_MODEL_INIT_TENSORS(device_type)

    is_init_ = false;

    t_forward_ = 0;
    t_backward_ = 0;
    t_step_ = 0;
    t_optlist_ = 0;
    t_tocuda_ = 0;
}

torch::Tensor GaussianModel::getScaling()
{
    return torch::exp(scaling_);
}

torch::Tensor GaussianModel::getRotation()
{
    return torch::nn::functional::normalize(rotation_);
}

torch::Tensor GaussianModel::getXYZ()
{
    return xyz_;
}

torch::Tensor GaussianModel::getFeaturesDc()
{
    return features_dc_;
}

torch::Tensor GaussianModel::getFeaturesRest()
{
    return features_rest_;
}

torch::Tensor GaussianModel::getOpacity()
{
    return torch::sigmoid(opacity_);
}

torch::Tensor GaussianModel::getCovariance(int scaling_modifier)
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScaling();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

torch::Tensor GaussianModel::getExposure()
{
    return exposure_;
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset)
{
    /// foreground
    int num = static_cast<int>(dataset->pointcloud_.size());
    assert(num > 0);
    torch::Tensor fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();  // (n, 3)
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    torch::Tensor scales = torch::zeros({num}, torch::kFloat32).cuda();

    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num; ++i) 
    {
        auto& pt_w = dataset->pointcloud_[i];
        auto& color = dataset->pointcolor_[i];
        fused_point_cloud.index({i, 0}) = pt_w.x();
        fused_point_cloud.index({i, 1}) = pt_w.y();
        fused_point_cloud.index({i, 2}) = pt_w.z();
        features.index({i, 0, 0}) = RGB2SH(color.x());
        features.index({i, 1, 0}) = RGB2SH(color.y());
        features.index({i, 2, 0}) = RGB2SH(color.z());

        double d = dataset->pointdepth_[i];
        scales.index({i}) = std::log(scaling_scale_ * d / f);
    }
    scales = scales.unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)

    /// sky
    if (skybox_points_num_ > 0)
    {
        int num = skybox_points_num_;
        double radius = skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);

        torch::Tensor sky_features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;

        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({num, 1}, torch::kFloat32).cuda());

        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }

    this->xyz_ = fused_point_cloud.requires_grad_();  // (n, 3)
    // this->xyz_ = fused_point_cloud.requires_grad_(false);  // fix xyz
    this->features_dc_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();  // (n, 1, 3)
    this->features_rest_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();  // (n, 15, 3)
    this->scaling_ = scales.requires_grad_();  // (n, 3)
    this->rotation_ = rots.requires_grad_();  // (n, 4)
    this->opacity_ = opacities.requires_grad_();  // (n, 1)

    if (apply_exposure_)
    {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();  // (3, 4)
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC
    
    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;37m Init Map with " 
              << double(fused_point_cloud.size(0)) / 10000 << "w GS" 
              << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void GaussianModel::saveMap(const std::string& result_path)
{
    std::string pc_path = result_path + "/point_cloud.ply";

    torch::Tensor xyz = this->xyz_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    // torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor f_dc = this->features_dc_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor scale = this->scaling_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor rotation = this->rotation_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(pc_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // // normals
    // result_file.add_properties_to_element(
    //     "vertex", {"nx", "ny", "nz"},
    //     tinyply::Type::FLOAT32, normals.size(0),
    //     reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
    //     tinyply::Type::INVALID, 0);

    // f_dc
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    if (n_f_rest > 0)
    {
        result_file.add_properties_to_element(
            "vertex", property_names_f_rest,
            tinyply::Type::FLOAT32, this->features_rest_.size(0),
            reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
            tinyply::Type::INVALID, 0);
    }

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

void GaussianModel::trainingSetup()
{
    this->sparse_optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, 0.0, 1e-15));
    sparse_optimizer_->param_groups()[0].options().set_lr(position_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_dc_);
    sparse_optimizer_->param_groups()[1].options().set_lr(feature_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_rest_);
    sparse_optimizer_->param_groups()[2].options().set_lr(feature_lr_ / 20.0);

    sparse_optimizer_->add_param_group(Tensor_vec_opacity_);
    sparse_optimizer_->param_groups()[3].options().set_lr(opacity_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_scaling_);
    sparse_optimizer_->param_groups()[4].options().set_lr(scaling_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_rotation_);
    sparse_optimizer_->param_groups()[5].options().set_lr(rotation_lr_);

    if (apply_exposure_)
    {
        this->exposure_optimizer_.reset(new torch::optim::Adam(Tensor_vec_exposure_, {}));
        exposure_optimizer_->param_groups()[0].options().set_lr(exposure_lr_);
    }
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation)
{
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = 
    {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->sparse_optimizer_->param_groups();
    auto& optimizer_state = this->sparse_optimizer_->get_state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) 
    {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];

        auto old_param_impl = param.unsafeGetTensorImpl();

        param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
        // if (group_idx == 0) param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_(false);  // fix xyz
        // else param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();  // fix xyz
        group.params()[0] = param;

        auto new_param_impl = param.unsafeGetTensorImpl();

        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end()) 
        {
            auto stored_state = state_it->second;

            stored_state.exp_avg = torch::cat({stored_state.exp_avg.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);
            stored_state.exp_avg_sq = torch::cat({stored_state.exp_avg_sq.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);

            optimizer_state.erase(state_it);

            optimizer_state[new_param_impl] = stored_state;
        }
        else 
        {
            State new_state;
            new_state.step = 0;
            new_state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.initialized = true;

            optimizer_state[new_param_impl] = new_state;
        }

        optimizable_tensors[group_idx] = param;
    }

    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC
}

static void saveDepthDiagnosis(const std::shared_ptr<Camera>& camera,
                               const torch::Tensor& rendered_depth,
                               const std::string& diagnosis_dir)
{
    auto fused_cpu = camera->original_depth_.detach().to(torch::kCPU).contiguous();
    auto rendered_cpu = rendered_depth.detach().to(torch::kCPU).contiguous();
    const int height = static_cast<int>(fused_cpu.size(0));
    const int width = static_cast<int>(fused_cpu.size(1));
    cv::Mat fused(height, width, CV_32FC1, fused_cpu.data_ptr<float>());
    cv::Mat rendered(height, width, CV_32FC1, rendered_cpu.data_ptr<float>());
    cv::Mat valid = (fused > 0.0f) & (rendered > 0.0f);

    auto colorize = [](const cv::Mat& depth, const cv::Mat& mask, double max_depth) {
        cv::Mat clipped;
        depth.copyTo(clipped);
        clipped.setTo(0, ~mask);
        clipped = cv::min(clipped, max_depth);
        clipped.convertTo(clipped, CV_8UC1, 255.0 / max_depth);
        cv::Mat colored;
        cv::applyColorMap(clipped, colored, cv::COLORMAP_TURBO);
        colored.setTo(cv::Scalar(0, 0, 0), ~mask);
        return colored;
    };

    cv::Mat fused_valid, rendered_valid;
    fused.copyTo(fused_valid, fused > 0.0f);
    rendered.copyTo(rendered_valid, rendered > 0.0f);
    double fused_max = 0.0, rendered_max = 0.0;
    cv::minMaxLoc(fused_valid, nullptr, &fused_max);
    cv::minMaxLoc(rendered_valid, nullptr, &rendered_max);
    const double visualization_max = std::max(1.0, std::min(80.0, std::max(fused_max, rendered_max)));
    cv::Mat difference = cv::abs(fused - rendered);
    difference.setTo(0, ~valid);
    cv::imwrite(diagnosis_dir + "/composite_" + camera->image_name_, colorize(fused, fused > 0.0f, visualization_max));
    cv::imwrite(diagnosis_dir + "/rendered_" + camera->image_name_, colorize(rendered, rendered > 0.0f, visualization_max));
    cv::imwrite(diagnosis_dir + "/absdiff_" + camera->image_name_, colorize(difference, valid, std::min(10.0, visualization_max)));
}

void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{
    torch::NoGradGuard no_grad;
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    std::shared_ptr<Camera> viewpoint_cam = dataset->train_cameras_.back();
    auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_, true);
    auto rendered_alpha = 1 - std::get<2>(render_pkg).squeeze(0);

    int n = dataset->pointcloud_.size();
    std::vector<float> float_point(n * 3);
    std::vector<float> float_color(n * 3);
    for (size_t i = 0; i < n; ++i) 
    {
        float_point[3 * i + 0] = static_cast<float>(dataset->pointcloud_[i][0]);
        float_point[3 * i + 1] = static_cast<float>(dataset->pointcloud_[i][1]);
        float_point[3 * i + 2] = static_cast<float>(dataset->pointcloud_[i][2]);
        float_color[3 * i + 0] = static_cast<float>(dataset->pointcolor_[i][0]);
        float_color[3 * i + 1] = static_cast<float>(dataset->pointcolor_[i][1]);
        float_color[3 * i + 2] = static_cast<float>(dataset->pointcolor_[i][2]);
    }
    torch::Tensor points = torch::from_blob(float_point.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor colors = torch::from_blob(float_color.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor depths_in_rsp_frame = torch::from_blob(dataset->pointdepth_.data(), {n}).to(torch::kFloat32).cuda();

    /// filter
    auto R_wc = dataset->R_wc_.back();
    auto t_wc = dataset->t_wc_.back();
    auto R_cw = R_wc.transpose();
    auto t_cw = - R_cw * t_wc;
    std::vector<float> float_R_cw(3 * 3);
    std::vector<float> float_t_cw(3);
    for (size_t i = 0; i < 3; ++i)
    {
        float_R_cw[3 * i + 0] = static_cast<float>(R_cw(i, 0));
        float_R_cw[3 * i + 1] = static_cast<float>(R_cw(i, 1));
        float_R_cw[3 * i + 2] = static_cast<float>(R_cw(i, 2));
        float_t_cw[i] = static_cast<float>(t_cw[i]);
    }
    torch::Tensor R_cw_tensor = torch::from_blob(float_R_cw.data(), {3, 3}).to(torch::kFloat32).cuda();
    torch::Tensor t_cw_tensor = torch::from_blob(float_t_cw.data(), {3, 1}).to(torch::kFloat32).cuda();
    auto points_camera = torch::matmul(points, R_cw_tensor.t()) + t_cw_tensor.view({1, 3});  // (n, 3)
    const int H = viewpoint_cam->image_height_;
    const int W = viewpoint_cam->image_width_;
    float fx = static_cast<float>(viewpoint_cam->fx_);
    float fy = static_cast<float>(viewpoint_cam->fy_);
    float cx = static_cast<float>(viewpoint_cam->cx_);
    float cy = static_cast<float>(viewpoint_cam->cy_);
    float focal = (fx + fy) / 2.0;
    torch::Tensor depths;
    torch::Tensor x_pixel;
    torch::Tensor y_pixel;
    if (viewpoint_cam->is_equirectangular_)
    {
        auto x = points_camera.index({torch::indexing::Slice(), 0});
        auto y = points_camera.index({torch::indexing::Slice(), 1});
        auto z = points_camera.index({torch::indexing::Slice(), 2});
        depths = torch::linalg_vector_norm(points_camera, 2, {1});
        auto longitude = torch::atan2(x, z);
        auto latitude = torch::atan2(-y, torch::sqrt(x * x + z * z));
        x_pixel = torch::remainder((longitude / M_PI + 1.0) * (0.5 * W), W);
        y_pixel = (0.5 - latitude / M_PI) * H;
        focal = 0.5f * (static_cast<float>(W) / (2.0f * M_PI) +
                        static_cast<float>(H) / M_PI);
    }
    else
    {
        depths = points_camera.index({torch::indexing::Slice(), 2});  // (n)
        x_pixel = (points_camera.index({torch::indexing::Slice(), 0}) * fx) / depths + cx;
        y_pixel = (points_camera.index({torch::indexing::Slice(), 1}) * fy) / depths + cy;
    }
    auto pixels = torch::stack({x_pixel, y_pixel}, 1);  // (n, 2)
    pixels = pixels.floor().to(torch::kInt32);

    auto pixels_float = pixels.to(torch::kFloat32);
    auto pixels_with_depth = torch::cat({pixels_float, depths.unsqueeze(1)}, 1).to(torch::kCPU);
    auto pixels_depth_a = pixels_with_depth.accessor<float, 2>();

    std::unordered_map<std::string, std::pair<int, float>> pixel_depth_map;
    for (int i = 0; i < pixels_with_depth.size(0); ++i) {
        int x = static_cast<int>(pixels_depth_a[i][0]);
        int y = static_cast<int>(pixels_depth_a[i][1]);
        float depth = pixels_depth_a[i][2];
        
        std::string key = std::to_string(x) + "_" + std::to_string(y);
        if (!pixel_depth_map.count(key) || depth < pixel_depth_map[key].second) {
            pixel_depth_map[key] = {i, depth};
        }
    }

    std::vector<int64_t> keep_indices;
    for (const auto& item : pixel_depth_map) {
        keep_indices.push_back(item.second.first);
    }

    auto keep_indices_tensor = torch::from_blob(
        keep_indices.data(), 
        {static_cast<int64_t>(keep_indices.size())}, 
        torch::kInt64
    ).to(points.device());
    auto filtered_points = points.index_select(0, keep_indices_tensor);
    auto filtered_colors = colors.index_select(0, keep_indices_tensor);
    auto filtered_depths_in_rsp_frame = depths_in_rsp_frame.index_select(0, keep_indices_tensor);
    auto filtered_pixels = pixels.index_select(0, keep_indices_tensor);

    auto filter = [H, W, &rendered_alpha](const torch::Tensor& points, 
                                        const torch::Tensor& colors, 
                                        const torch::Tensor& depths_in_rsp_frame, 
                                        const torch::Tensor& pixels) 
    {
        auto in_image = (pixels.index({torch::indexing::Slice(), 0}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 0}) < W) &
                        (pixels.index({torch::indexing::Slice(), 1}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 1}) < H);  // (n) bool
        
        auto positive_depth = depths_in_rsp_frame > 0;

        auto x_coords = pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1);
        auto y_coords = pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1);
        auto opaque = rendered_alpha.index({y_coords, x_coords}) < 0.99;  // (n) bool

        auto valid_flag = torch::logical_and(torch::logical_and(in_image, positive_depth), opaque);
        auto filtered_points = points.index({valid_flag, torch::indexing::Slice()});
        auto filtered_colors = colors.index({valid_flag, torch::indexing::Slice()});
        auto filtered_depths = depths_in_rsp_frame.index({valid_flag});
        return std::make_tuple(filtered_points, filtered_colors, filtered_depths);
    };

    // auto filtered_pkg = filter(points, colors, depths_in_rsp_frame, pixels);
    auto filtered_pkg = filter(filtered_points, filtered_colors, filtered_depths_in_rsp_frame, filtered_pixels);
    
    /// densification
    torch::Tensor fused_point_cloud = std::get<0>(filtered_pkg);  // (n, 3)
    torch::Tensor fused_color = RGB2SH(std::get<1>(filtered_pkg));
    int num = fused_point_cloud.size(0);
    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;
    torch::Tensor features_dc = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();  // (n, 1, 3)
    torch::Tensor features_rest = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous();  // (n, 15, 3)
    torch::Tensor scales = torch::log(pc->scaling_scale_ * std::get<2>(filtered_pkg) / focal).unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)

    pc->densificationPostfix(fused_point_cloud, features_dc, features_rest, opacities, scales, rots);

    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;32m Insert " << double(fused_point_cloud.size(0)) / 1000 
              << "k GS" << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void decayOptList(int max_iters, const int train_camera_num, 
                  const std::shared_ptr<Dataset>& dataset, const std::vector<int>& all_list, std::vector<int>& opt_list)
{
    Eigen::Vector3d t0 = dataset->t_wc_[0];
    double dist = (dataset->t_wc_.back() - t0).norm();
    if (dist > 120)
    {
        max_iters /= 2;
        opt_list.clear();
        std::random_device rd;
        std::mt19937 gen(rd());
        int split = train_camera_num * 2 / 3;
        int half = max_iters / 2;
        std::sample(all_list.begin(), all_list.begin() + split,
                    std::back_inserter(opt_list), std::min(half, split), gen);
        std::sample(all_list.begin() + split, all_list.end(),
                    std::back_inserter(opt_list), std::min(half, train_camera_num - split), gen);
    }
}

double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{
    pc->t_start_ = std::chrono::steady_clock::now();
    int updated_num = 0;
    std::vector<int> opt_list;
    int max_iters = 100;

    int train_camera_num = dataset->train_cameras_.size();
    std::vector<int> all_list(train_camera_num);
    std::iota(all_list.begin(), all_list.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    if (train_camera_num <= max_iters) 
    {
        opt_list = all_list;
    }
    else
    {
        std::sample(all_list.begin(), all_list.end(), 
                    std::back_inserter(opt_list), max_iters, gen);
    } 
    if (pc->iteration_decay_) decayOptList(max_iters, train_camera_num, dataset, all_list, opt_list);
    std::shuffle(opt_list.begin(), opt_list.end(), gen);
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_optlist_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

    pc->t_start_ = std::chrono::steady_clock::now();
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    for (int idx : opt_list)
    {
        pc->t_start_ = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera>& viewpoint_cam = dataset->train_cameras_[idx];
        auto gt_image = viewpoint_cam->original_image_.to(torch::kCUDA, /*non_blocking=*/true);
        auto gt_depth = viewpoint_cam->original_depth_.to(torch::kCUDA, /*non_blocking=*/true);
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        pc->t_start_ = std::chrono::steady_clock::now();
        auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_);
        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<1>(render_pkg);
        auto mask = (gt_depth > 0) & (rendered_depth > 0);
        auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);
        auto Ll1_depth = torch::abs(rendered_depth.masked_select(mask) - gt_depth.masked_select(mask)).mean();
        float lambda_dssim = pc->lambda_dssim_;
        float lambda_depth = pc->lambda_depth_;
        torch::Tensor ssim_value;
        torch::Tensor rendered_image_unsq = rendered_image.unsqueeze(0);
        torch::Tensor gt_image_unsq = gt_image.unsqueeze(0);
        ssim_value = loss_utils::fused_ssim(rendered_image_unsq, gt_image_unsq);
        auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value);
        if (pc->optimize_depth_) loss += lambda_depth * Ll1_depth;
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        
        pc->t_start_ = std::chrono::steady_clock::now();
        loss.backward();
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_backward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

        pc->t_start_ = std::chrono::steady_clock::now();
        auto visible = std::get<4>(render_pkg);
        updated_num += visible.sum().item<int>();
        pc->sparse_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0));
        pc->sparse_optimizer_->step();
        pc->sparse_optimizer_->zero_grad(true);
        if (pc->apply_exposure_)
        {
            pc->exposure_optimizer_->step();
            pc->exposure_optimizer_->zero_grad(true);
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_step_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    }

    return updated_num / opt_list.size();
}

void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, 
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path)
{
    std::cout << "\n     🎉 Evaluate Visual Quality 🎉\n";
    std::cout << "\n        [Number of Final Gaussians] " << pc->getXYZ().size(0) << std::endl;

    if (fs::exists(result_path)) fs::remove_all(result_path);
    fs::create_directories(result_path);

    std::string render_dir_path = result_path + "/render";
    fs::create_directories(render_dir_path);
    std::string render_depth_dir_path = result_path + "/render_depth";
    fs::create_directories(render_depth_dir_path);
    std::string gt_dir_path = result_path + "/gt";
    fs::create_directories(gt_dir_path);
    std::string diagnosis_dir_path = result_path + "/depth_diagnose";
    fs::create_directories(diagnosis_dir_path);

    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::jit::script::Module m_lpips;
    try 
    {
        m_lpips = torch::jit::load(lpips_path + "/lpips_alex.pt");
        m_lpips.to(torch::kCUDA);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "lpips model loading failed: " << e.what() << std::endl;
    }

    torch::Tensor metric_mask;
    if (dataset->metric_mask_.defined())
        metric_mask = dataset->metric_mask_.to(torch::kCUDA);

    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        for (const auto& train_camera : dataset->train_cameras_)
        {
            auto render_pkg = render(train_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto rendered_depth = std::get<1>(render_pkg);
            saveDepthDiagnosis(train_camera, rendered_depth, diagnosis_dir_path);
            auto gt_image = train_camera->original_image_.cuda().clamp(0, 1);
            const bool use_metric_mask = metric_mask.defined() && train_camera->is_equirectangular_;
            auto metric_rendered = rendered_image;
            auto metric_gt = gt_image;
            if (use_metric_mask)
            {
                auto mask = metric_mask.unsqueeze(0);
                metric_rendered = rendered_image * mask;
                metric_gt = gt_image * mask;
            }
            double psnr = use_metric_mask
                ? loss_utils::masked_psnr(rendered_image, gt_image, metric_mask).item<double>()
                : loss_utils::psnr(rendered_image, gt_image).item<double>();
            double ssim = use_metric_mask
                ? loss_utils::ssim_masked(metric_rendered, metric_gt, metric_mask).item<double>()
                : loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(metric_rendered.unsqueeze(0));
            inputs.push_back(metric_gt.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            int H = rendered_image.size(1), W = rendered_image.size(2);

            torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
            cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
            cv::imwrite(render_dir_path + "/" + train_camera->image_name_, a_img);

            torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
            cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
            cv::imwrite(gt_dir_path + "/" + train_camera->image_name_, b_img);

            torch::Tensor depth_map_normalized = (rendered_depth - rendered_depth.min()) / 
                                                     (rendered_depth.max() - rendered_depth.min()) * 255;
            torch::Tensor c_cpu = depth_map_normalized.to(torch::kCPU);
            cv::Mat c_img(H, W, CV_32FC1, c_cpu.data_ptr<float>());
            c_img.convertTo(c_img, CV_8UC1);
            cv::applyColorMap(c_img, c_img, cv::COLORMAP_JET);
            cv::imwrite(render_depth_dir_path + "/" + train_camera->image_name_, c_img);
        }
        psnrs /= dataset->train_cameras_.size();
        ssims /= dataset->train_cameras_.size();
        lpipss /= dataset->train_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [Training View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View LPIPS] " << lpipss << std::endl;
    }
    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        for (const auto& test_camera : dataset->test_cameras_)
        {
            auto render_pkg = render(test_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto rendered_depth = std::get<1>(render_pkg);
            auto gt_image = test_camera->original_image_.cuda().clamp(0, 1);
            const bool use_metric_mask = metric_mask.defined() && test_camera->is_equirectangular_;
            auto metric_rendered = rendered_image;
            auto metric_gt = gt_image;
            if (use_metric_mask)
            {
                auto mask = metric_mask.unsqueeze(0);
                metric_rendered = rendered_image * mask;
                metric_gt = gt_image * mask;
            }
            double psnr = use_metric_mask
                ? loss_utils::masked_psnr(rendered_image, gt_image, metric_mask).item<double>()
                : loss_utils::psnr(rendered_image, gt_image).item<double>();
            double ssim = use_metric_mask
                ? loss_utils::ssim_masked(metric_rendered, metric_gt, metric_mask).item<double>()
                : loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(metric_rendered.unsqueeze(0));
            inputs.push_back(metric_gt.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            int H = rendered_image.size(1), W = rendered_image.size(2);

            torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
            cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
            cv::imwrite(render_dir_path + "/" + test_camera->image_name_, a_img);

            torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
            cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
            cv::imwrite(gt_dir_path + "/" + test_camera->image_name_, b_img);

            torch::Tensor depth_map_normalized = (rendered_depth - rendered_depth.min()) / 
                                                     (rendered_depth.max() - rendered_depth.min()) * 255;
            torch::Tensor c_cpu = depth_map_normalized.to(torch::kCPU);
            cv::Mat c_img(H, W, CV_32FC1, c_cpu.data_ptr<float>());
            c_img.convertTo(c_img, CV_8UC1);
            cv::applyColorMap(c_img, c_img, cv::COLORMAP_JET);
            cv::imwrite(render_depth_dir_path + "/" + test_camera->image_name_, c_img);
        }
        psnrs /= dataset->test_cameras_.size();
        ssims /= dataset->test_cameras_.size();
        lpipss /= dataset->test_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [In-Sequence Novel View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View LPIPS] " << lpipss << std::endl;
    }
}
