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

#pragma once

#include "yaml_utils.h"

#include <chrono>
#include <cmath>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();
        const std::string camera_model =
            node["camera_model"] ? node["camera_model"].as<std::string>() : "pinhole";
        if (camera_model != "pinhole" && camera_model != "equirectangular" && camera_model != "erp")
            throw std::invalid_argument("Unsupported camera model: " + camera_model);
        equirectangular = camera_model != "pinhole";

        select_every_k_frame = node["select_every_k_frame"].as<int>();
        depth_completion = node["depth_completion"].as<bool>();
        patch_size = node["patch_size"].as<int>();
        if (patch_size <= 0)
            throw std::invalid_argument("patch_size must be positive");
        lidar_patch_size = node["lidar_patch_size"] ? node["lidar_patch_size"].as<int>() : 3;
        if (lidar_patch_size < 0)
            throw std::invalid_argument("lidar_patch_size must be nonnegative (0 disables sampling)");
        min_point_depth = node["min_point_depth"] ? node["min_point_depth"].as<double>() : 0.0;
        if (const auto body = node["body_filter"])
        {
            body_radius = body["radius_m"].as<double>();
            body_height = body["height_m"].as<double>();
            dap_body_radius = body["dap_radius_m"] ? body["dap_radius_m"].as<double>() : body_radius;
            dap_body_height = body["dap_height_m"] ? body["dap_height_m"].as<double>() : body_height;
            for (int k = 0; k < 3; ++k)
                body_axis[k] = body["axis_camera"][k].as<double>();
        }
        max_depth = node["max_depth"].as<double>();
        map_extension_min_depth_gap_m = node["map_extension_min_depth_gap_m"]
            ? node["map_extension_min_depth_gap_m"].as<double>() : 0.5;
        if (map_extension_min_depth_gap_m <= 0.0)
            throw std::invalid_argument("map_extension_min_depth_gap_m must be positive");
        map_extension_relative_depth_gap = node["map_extension_relative_depth_gap"]
            ? node["map_extension_relative_depth_gap"].as<double>() : 0.1;
        if (map_extension_relative_depth_gap < 0.0)
            throw std::invalid_argument("map_extension_relative_depth_gap must be nonnegative");
        map_extension_depth_rescued_opacity = node["map_extension_depth_rescued_opacity"]
            ? node["map_extension_depth_rescued_opacity"].as<double>() : 0.1;
        if (!(map_extension_depth_rescued_opacity > 0.0 && map_extension_depth_rescued_opacity < 1.0))
            throw std::invalid_argument("map_extension_depth_rescued_opacity must be between 0 and 1 exclusively");
        map_extension_depth_rescued_scale_multiplier = node["map_extension_depth_rescued_scale_multiplier"]
            ? node["map_extension_depth_rescued_scale_multiplier"].as<double>() : 1.0;
        if (!(map_extension_depth_rescued_scale_multiplier > 0.0) ||
            !std::isfinite(map_extension_depth_rescued_scale_multiplier))
            throw std::invalid_argument("map_extension_depth_rescued_scale_multiplier must be finite and positive");
        online_dap = node["online_dap"] ? node["online_dap"].as<bool>() : false;
        dap_topic = node["dap_topic"] ? node["dap_topic"].as<std::string>() : "/depth_dap_for_gs";
        dap_dense_depth_supervision = node["dap_dense_depth_supervision"]
            ? node["dap_dense_depth_supervision"].as<bool>() : true;
        dap_initialize_gaussians = node["dap_initialize_gaussians"]
            ? node["dap_initialize_gaussians"].as<bool>() : true;
        dap_seed_lidar_dilation_pixels = node["dap_seed_lidar_dilation_pixels"]
            ? node["dap_seed_lidar_dilation_pixels"].as<int>() : 0;
        dap_seed_max_depth_gradient = node["dap_seed_max_depth_gradient"]
            ? node["dap_seed_max_depth_gradient"].as<double>() : 1.0;
        if (!std::isfinite(dap_seed_max_depth_gradient) || dap_seed_max_depth_gradient <= 0.0)
            throw std::invalid_argument("dap_seed_max_depth_gradient must be finite and positive");
        std::string pkg_path = ros::package::getPath("gaussian_lic");
        if (height == 512 && width == 640) engine_path = pkg_path + "/ckpt/spnet_512_640.engine";
        if (height == 480 && width == 640) engine_path = pkg_path + "/ckpt/spnet_480_640.engine";

        sh_degree = node["sh_degree"] ? node["sh_degree"].as<int>() : 1;
        if (sh_degree < 0 || sh_degree > 3)
            throw std::invalid_argument("sh_degree must be between 0 and 3");
        metric_mask_path = node["metric_mask_path"] ? node["metric_mask_path"].as<std::string>() : "";
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<double>();

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();
        latitude_weighting = node["latitude_weighting"] ? node["latitude_weighting"].as<bool>() : true;
        optimize_depth = node["optimize_depth"].as<bool>();
        normalize_depth_gradient = node["normalize_depth_gradient"] ? node["normalize_depth_gradient"].as<bool>() : true;
        diagnose_depth_visibility = node["diagnose_depth_visibility"] ? node["diagnose_depth_visibility"].as<bool>() : false;
        depth_visibility_lidar_strength = node["depth_visibility_lidar_strength"] ? node["depth_visibility_lidar_strength"].as<double>() : 0.0;
        depth_visibility_dap_strength = node["depth_visibility_dap_strength"] ? node["depth_visibility_dap_strength"].as<double>() : 0.0;
        if (!(depth_visibility_lidar_strength >= 0.0 && depth_visibility_lidar_strength <= 1.0 &&
              depth_visibility_dap_strength >= 0.0 && depth_visibility_dap_strength <= 1.0))
            throw std::invalid_argument("Depth visibility strengths must lie in [0, 1]");
        dap_depth_loss_relative_weight = node["dap_depth_loss_relative_weight"]
            ? node["dap_depth_loss_relative_weight"].as<double>() : 0.1;
        if (!(dap_depth_loss_relative_weight >= 0.0) || !std::isfinite(dap_depth_loss_relative_weight))
            throw std::invalid_argument("dap_depth_loss_relative_weight must be finite and nonnegative");
        lambda_depth = node["lambda_depth"].as<double>();
        iteration_decay = node["iteration_decay"].as<bool>();
        optimization_recent_keyframes = node["optimization_recent_keyframes"]
            ? node["optimization_recent_keyframes"].as<int>() : 15;
        if (optimization_recent_keyframes < 0 || optimization_recent_keyframes > 100)
            throw std::invalid_argument("optimization_recent_keyframes must be between 0 and 100");

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;
    bool equirectangular;

    int select_every_k_frame;
    bool depth_completion;
    int patch_size;
    int lidar_patch_size;
    double min_point_depth;
    double body_radius = 0.0, body_height = 0.0;
    double dap_body_radius = 0.0, dap_body_height = 0.0;
    Eigen::Vector3d body_axis = Eigen::Vector3d::Zero();
    double max_depth;
    double map_extension_min_depth_gap_m;
    double map_extension_relative_depth_gap;
    double map_extension_depth_rescued_opacity;
    double map_extension_depth_rescued_scale_multiplier;
    bool online_dap;
    std::string dap_topic;
    bool dap_dense_depth_supervision;
    bool dap_initialize_gaussians;
    int dap_seed_lidar_dilation_pixels;
    double dap_seed_max_depth_gradient;
    std::string engine_path;

    /// gaussian
    int sh_degree;
    std::string metric_mask_path;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;
    bool latitude_weighting = true;
    bool optimize_depth;
    bool normalize_depth_gradient;
    bool diagnose_depth_visibility;
    double depth_visibility_lidar_strength;
    double depth_visibility_dap_strength;
    double dap_depth_loss_relative_weight;
    double lambda_depth;
    bool iteration_decay;
    int optimization_recent_keyframes;

    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;
};

struct Frame 
{
    sensor_msgs::PointCloud2ConstPtr point_msg;
    geometry_msgs::PoseStampedConstPtr pose_msg;
    sensor_msgs::ImageConstPtr image_msg;
    sensor_msgs::ImageConstPtr depth_msg;
    sensor_msgs::ImageConstPtr dap_depth_msg;
};
