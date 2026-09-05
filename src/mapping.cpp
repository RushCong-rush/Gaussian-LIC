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

#include "mapping.h"
#include "gaussian.h"

#include <atomic>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

namespace fs = std::filesystem;

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;
std::queue<sensor_msgs::ImageConstPtr> depth_buf;
std::queue<sensor_msgs::ImageConstPtr> dap_depth_buf;

std::atomic<bool> exit_flag(false);
std::atomic<double> last_point_time(0.0);
std::atomic<bool> gaussians_initialized(false);
std::atomic<bool> online_dap_enabled(false);
std::atomic<bool> dap_sync_error(false);
std::atomic<double> dap_wait_start(0.0);

namespace
{
bool mapExtensionDiagnosisEnabled()
{
    const char* value = std::getenv("ODGS_MAP_EXTENSION_DIAG");
    return value != nullptr && std::string(value) == "1";
}

void saveRgbTensor(const torch::Tensor& image, const std::string& path)
{
    torch::Tensor image_cpu = image.detach().clamp(0, 1).to(torch::kCPU)
                                  .permute({1, 2, 0}).contiguous()
                                  .mul(255).clamp(0, 255).to(torch::kU8);
    cv::Mat image_rgb(image_cpu.size(0), image_cpu.size(1), CV_8UC3,
                      image_cpu.data_ptr<uint8_t>());
    cv::Mat image_bgr;
    cv::cvtColor(image_rgb, image_bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path, image_bgr))
        throw std::runtime_error("Failed to save online RGB image: " + path);
}

void saveCausalDepthDiagnosis(const torch::Tensor& rendered_depth,
                              const torch::Tensor& rendered_final_transmittance,
                              const torch::Tensor& valid_mask,
                              const std::string& diagnosis_dir,
                              const std::string& image_name)
{
    auto depth_cpu = rendered_depth.detach().squeeze().to(torch::kCPU).contiguous();
    auto alpha_cpu = (1.0 - rendered_final_transmittance).detach().squeeze()
                         .to(torch::kCPU).contiguous();
    const int height = static_cast<int>(depth_cpu.size(0));
    const int width = static_cast<int>(depth_cpu.size(1));
    cv::Mat depth(height, width, CV_32FC1, depth_cpu.data_ptr<float>());
    cv::Mat alpha(height, width, CV_32FC1, alpha_cpu.data_ptr<float>());
    depth = depth.clone();
    alpha = alpha.clone();
    cv::patchNaNs(depth, 0.0);
    cv::patchNaNs(alpha, 0.0);

    cv::Mat valid_region(height, width, CV_8UC1, cv::Scalar(255));
    if (valid_mask.defined())
    {
        auto mask_cpu = valid_mask.detach().to(torch::kCPU).to(torch::kUInt8).contiguous();
        cv::Mat mask(height, width, CV_8UC1, mask_cpu.data_ptr<uint8_t>());
        valid_region = mask.clone() * 255;
    }

    cv::Mat clipped_depth;
    cv::max(depth, 0.0, clipped_depth);
    cv::min(clipped_depth, 80.0, clipped_depth);
    cv::Mat depth_u8;
    clipped_depth.convertTo(depth_u8, CV_8UC1, 255.0 / 80.0);
    cv::Mat depth_color;
    cv::applyColorMap(depth_u8, depth_color, cv::COLORMAP_TURBO);
    depth_color.setTo(cv::Scalar(0, 0, 0), (depth <= 0.0f) | (valid_region == 0));

    cv::Mat clipped_alpha;
    cv::max(alpha, 0.0, clipped_alpha);
    cv::min(clipped_alpha, 1.0, clipped_alpha);
    cv::Mat alpha_u8;
    clipped_alpha.convertTo(alpha_u8, CV_8UC1, 255.0);
    alpha_u8.setTo(0, valid_region == 0);

    cv::imwrite(diagnosis_dir + "/causal_depth/" + image_name, depth_color);
    cv::imwrite(diagnosis_dir + "/causal_alpha/" + image_name, alpha_u8);
}

void appendOnlineDepthMetric(const std::shared_ptr<Camera>& camera,
                             const torch::Tensor& rendered_depth,
                             const torch::Tensor& valid_mask,
                             const std::string& metrics_path)
{
    auto rendered_cpu = rendered_depth.detach().squeeze().to(torch::kCPU).contiguous();
    auto lidar_depth = camera->diagnostic_depth_.detach().squeeze().to(torch::kCPU).contiguous();
    auto lidar_valid = camera->lidar_valid_mask_.detach().squeeze().to(torch::kCPU).to(torch::kBool);
    auto overlap = lidar_valid & torch::isfinite(lidar_depth) & (lidar_depth > 0.0) &
                   torch::isfinite(rendered_cpu) & (rendered_cpu > 0.0);
    if (valid_mask.defined())
        overlap &= valid_mask.detach().squeeze().to(torch::kCPU).to(torch::kBool);

    const int64_t overlap_pixels = overlap.sum().item<int64_t>();
    const double depth_l1 = overlap_pixels > 0
        ? torch::abs(rendered_cpu.masked_select(overlap) - lidar_depth.masked_select(overlap))
              .mean().item<double>()
        : std::numeric_limits<double>::quiet_NaN();

    const bool write_header = !fs::exists(metrics_path) || fs::file_size(metrics_path) == 0;
    std::ofstream metrics(metrics_path, std::ios::app);
    if (write_header)
        metrics << "image_name,lidar_overlap_pixels,lidar_mae_m\n";
    metrics << camera->image_name_ << ',' << overlap_pixels << ',' << depth_l1 << '\n';
}

void saveOnlineFrame(const std::shared_ptr<Camera>& camera,
                     const std::shared_ptr<Dataset>& dataset,
                     const std::shared_ptr<GaussianModel>& gaussians,
                     torch::Tensor& background,
                     const std::string& render_dir,
                     const std::string& gt_dir,
                     const std::string& depth_metrics_path,
                     const std::string& diagnosis_dir = "")
{
    torch::NoGradGuard no_grad;
    const auto render_pkg = render(camera, gaussians, background,
                                   gaussians->apply_exposure_);
    torch::Tensor valid_mask;
    if (camera->is_equirectangular_ && dataset->metric_mask_.defined())
        valid_mask = dataset->metric_mask_;
    auto rendered = compositeMaskedImage(std::get<0>(render_pkg), valid_mask, background);
    auto ground_truth = compositeMaskedImage(camera->original_image_, valid_mask, background);
    saveRgbTensor(rendered, render_dir + "/" + camera->image_name_);
    saveRgbTensor(ground_truth, gt_dir + "/" + camera->image_name_);
    appendOnlineDepthMetric(camera, std::get<1>(render_pkg), valid_mask, depth_metrics_path);
    if (!diagnosis_dir.empty())
        saveCausalDepthDiagnosis(std::get<1>(render_pkg), std::get<2>(render_pkg),
                                 valid_mask, diagnosis_dir, camera->image_name_);
}
}  // namespace

void pointCallback(const sensor_msgs::PointCloud2ConstPtr& point_msg) 
{
    m_buf.lock();
    point_buf.push(point_msg);
    last_point_time = ros::WallTime::now().toSec();
    m_buf.unlock();
}

void poseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg) 
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg) 
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg) 
{
    m_buf.lock();
    depth_buf.push(depth_msg);
    m_buf.unlock();
}

void dapDepthCallback(const sensor_msgs::ImageConstPtr& depth_msg)
{
    m_buf.lock();
    dap_depth_buf.push(depth_msg);
    m_buf.unlock();
}

bool getAlignedData(Frame& cur_frame, bool require_dap)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty() || depth_buf.empty())
    {
        return false;
    }

    double frame_time = point_buf.front()->header.stamp.toSec();

    while (1) 
    {
        if (pose_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            pose_buf.pop();
            if (pose_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (pose_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (image_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            image_buf.pop();
            if (image_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (image_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    while (1)
    {
        if (depth_buf.front()->header.stamp.toSec() < frame_time - 0.01)
        {
            depth_buf.pop();
            if (depth_buf.empty()) return false;
        }
        else break;
    }
    if (depth_buf.front()->header.stamp.toSec() > frame_time + 0.01)
    {
        point_buf.pop();
        return false;
    }

    // Non-keyframes do not consume DAP depth. Keyframes require the prediction
    // copied from their exact ERP timestamp; a mismatch is a protocol error.
    if (require_dap)
    {
        if (dap_depth_buf.empty())
        {
            double expected = 0.0;
            dap_wait_start.compare_exchange_strong(expected, ros::WallTime::now().toSec());
            return false;
        }
        const uint64_t image_stamp = image_buf.front()->header.stamp.toNSec();
        const uint64_t dap_stamp = dap_depth_buf.front()->header.stamp.toNSec();
        if (dap_stamp != image_stamp)
        {
            ROS_FATAL_STREAM("Keyframe DAP timestamp mismatch: ERP=" << image_stamp
                             << ", DAP=" << dap_stamp);
            dap_sync_error = true;
            exit_flag = true;
            return false;
        }
        dap_wait_start = 0.0;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();
    auto cur_depth = depth_buf.front();

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.image_msg = cur_image;
    cur_frame.depth_msg = cur_depth;
    cur_frame.dap_depth_msg = require_dap ? dap_depth_buf.front() : sensor_msgs::ImageConstPtr();

    point_buf.pop();
    pose_buf.pop();
    image_buf.pop();
    depth_buf.pop();
    if (require_dap) dap_depth_buf.pop();

    return true;
}

void mapping(const YAML::Node& node, const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    online_dap_enabled = prm.online_dap;
    std::cout << "        [SH Degree] " << prm.sh_degree << std::endl;
    std::cout << "        [Map Extension Min Depth Gap] "
              << prm.map_extension_min_depth_gap_m << " m" << std::endl;
    std::cout << "        [Map Extension Relative Depth Gap] "
              << prm.map_extension_relative_depth_gap << std::endl;
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);
    dataset->setDiagnosisDirectory(result_path + "/depth_diagnose");

    const std::string online_render_dir = result_path + "/online_render";
    const std::string online_gt_dir = result_path + "/online_gt";
    const std::string depth_diagnosis_dir = result_path + "/depth_diagnose";
    const std::string online_depth_metrics_path =
        depth_diagnosis_dir + "/online_render_depth_metrics.csv";
    if (fs::exists(online_render_dir)) fs::remove_all(online_render_dir);
    if (fs::exists(online_gt_dir)) fs::remove_all(online_gt_dir);
    fs::create_directories(online_render_dir);
    fs::create_directories(online_gt_dir);
    fs::create_directories(depth_diagnosis_dir);
    if (fs::exists(online_depth_metrics_path)) fs::remove(online_depth_metrics_path);
    const bool extension_diagnosis = mapExtensionDiagnosisEnabled();
    const std::string extension_diagnosis_dir = result_path + "/depth_diagnose";
    if (extension_diagnosis)
    {
        fs::create_directories(extension_diagnosis_dir + "/causal_depth");
        fs::create_directories(extension_diagnosis_dir + "/causal_alpha");
    }
    torch::Tensor online_background = prm.white_background
        ? torch::ones({3}, torch::kFloat32).cuda()
        : torch::zeros({3}, torch::kFloat32).cuda();

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;
    double total_online_render_time = 0;
    size_t online_rendered_frames = 0;

    Frame cur_frame;
    while (!exit_flag)
    {
        /// [1] data alignment
        m_buf.lock();
        const bool require_dap = online_dap_enabled &&
            ((dataset->all_frame_num_ + 1) % prm.select_every_k_frame == 0);
        bool align_flag = getAlignedData(cur_frame, require_dap);
        m_buf.unlock();
        if (!align_flag) continue;
        
        /// [2] add every frame
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera> current_camera = dataset->is_keyframe_current_
            ? dataset->train_cameras_.back()
            : dataset->test_cameras_.back();
        if (dataset->is_keyframe_current_)
        {
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
        }
        else
        {
            if (gaussians->is_init_)
            {
                t_start = std::chrono::steady_clock::now();
                saveOnlineFrame(current_camera, dataset, gaussians, online_background,
                                online_render_dir, online_gt_dir, online_depth_metrics_path);
                torch::cuda::synchronize();
                t_end = std::chrono::steady_clock::now();
                total_online_render_time +=
                    std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
                ++online_rendered_frames;
            }
            continue;
        }

        if (!gaussians->is_init_)
        {
            /// [3] initialize map
            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset);
            gaussians->trainingSetup();
        }
        else 
        {
            /// [4] extend map
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
            total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        }

        /// [5] optimize map
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << std::fixed << std::setprecision(2) 
                  << "\033[1;36m Update " << updated_num / 10000 
                  << "w GS per Iter \033[0m" << std::endl;

        t_start = std::chrono::steady_clock::now();
        saveOnlineFrame(current_camera, dataset, gaussians, online_background,
                        online_render_dir, online_gt_dir, online_depth_metrics_path,
                        extension_diagnosis ? extension_diagnosis_dir : "");
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_online_render_time +=
            std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        ++online_rendered_frames;
    }

    if (dap_sync_error)
    {
        ros::shutdown();
        return;
    }

    /// [6] evaluation
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2) << "\n        [Total Mapping Time] " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         1) Forward " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         2) Backward " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         3) Step " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         4) CPU2GPU " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Adding Time] " << total_adding_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Extending Time] " << total_extending_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Online Render Time] " << total_online_render_time << "s" << std::endl;
    std::cout << "        [Online Rendered Frames] " << online_rendered_frames << std::endl;
    torch::NoGradGuard no_grad;
    evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    gaussians->saveMap(result_path);

    std::cout << "\n\n😋 Gaussian-LIC Done!\n\n\n";
    ros::shutdown();
}

int main(int argc, char** argv)
{
    std::cout << "\n\n😋 Gaussian-LIC Ready!\n\n\n";
    ros::init(argc, argv, "gaussianlic");
    ros::NodeHandle nh("~");
    ros::Rate loop_rate(1000);
    image_transport::ImageTransport it_(nh);

    ros::Subscriber sub_point = nh.subscribe("/points_for_gs", 10000, pointCallback);
    ros::Subscriber sub_pose = nh.subscribe("/pose_for_gs", 10000, poseCallback);
    image_transport::Subscriber image_sub = it_.subscribe("/image_for_gs", 10000, imageCallback);
    image_transport::Subscriber depth_sub = it_.subscribe("/depth_for_gs", 10000, depthCallback);
    image_transport::Subscriber dap_depth_sub;

    std::string config_path;
    nh.param<std::string>("config_path", config_path, "");
    YAML::Node config_node = YAML::LoadFile(config_path);
    online_dap_enabled = config_node["online_dap"] ? config_node["online_dap"].as<bool>() : false;
    const int keyframe_interval = config_node["select_every_k_frame"].as<int>();
    if (keyframe_interval <= 0)
    {
        ROS_FATAL_STREAM("select_every_k_frame must be positive, got " << keyframe_interval);
        return 2;
    }
    std::string dap_topic = config_node["dap_topic"] ? config_node["dap_topic"].as<std::string>() : "/depth_dap_for_gs";
    if (online_dap_enabled)
        dap_depth_sub = it_.subscribe(dap_topic, 100, dapDepthCallback);
    int sh_degree_override;
    if (nh.getParam("sh_degree", sh_degree_override))
    {
        if (sh_degree_override < 0 || sh_degree_override > 3)
        {
            ROS_FATAL_STREAM("sh_degree must be between 0 and 3, got " << sh_degree_override);
            return 2;
        }
        config_node["sh_degree"] = sh_degree_override;
    }
    std::string result_path;
    nh.param<std::string>("result_path", result_path, "");
    std::string lpips_path;
    nh.param<std::string>("lpips_path", lpips_path, "");

    std::thread mapping_process(mapping, config_node, result_path, lpips_path);
    std::thread monitor_thread([](){
        while (!exit_flag) 
        {
            double now = ros::WallTime::now().toSec();
            if (gaussians_initialized && (now - last_point_time > 5.0)) 
            {
                m_buf.lock();
                if (point_buf.empty())
                    exit_flag = true;
                else
                {
                    const double wait_start = dap_wait_start.load();
                    if (online_dap_enabled && wait_start > 0.0 && now - wait_start > 5.0)
                    {
                        ROS_FATAL("Timed out waiting for exact-timestamp DAP depth for a keyframe");
                        dap_sync_error = true;
                        exit_flag = true;
                    }
                }
                m_buf.unlock();
            } 
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    
    ros::spin();

    mapping_process.join();
    monitor_thread.join();
    
    return dap_sync_error ? 2 : 0;
}
