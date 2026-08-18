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
        max_depth = node["max_depth"].as<double>();
        external_depth_dir = node["external_depth_dir"] ? node["external_depth_dir"].as<std::string>() : "";
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
        optimize_depth = node["optimize_depth"].as<bool>();
        lambda_depth = node["lambda_depth"].as<double>();
        iteration_decay = node["iteration_decay"].as<bool>();

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
    double max_depth;
    std::string external_depth_dir;
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
    bool optimize_depth;
    double lambda_depth;
    bool iteration_decay;

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
};
