#include "third_person.h"

#include <iomanip>
#include <sstream>

namespace {
constexpr int width = 1280, height = 720;
constexpr double back = 1.5, side = 0.6, above = 0.2, sphere_radius = 0.12;

cv::Point project(const Eigen::Vector3d& point, const Camera& view)
{
    return cv::Point(cvRound(view.fx_ * point.x() / point.z() + view.cx_),
                     cvRound(view.fy_ * point.y() / point.z() + view.cy_));
}
}

ThirdPersonExport::ThirdPersonExport(const std::string& result_path)
    : directory_(result_path + "/third_person")
{
    std::filesystem::create_directories(directory_);
    poses_.open(directory_ + "/views.csv");
    poses_ << "frame,camera_x,camera_y,camera_z,view_x,view_y,view_z,heading_x,heading_y\n";
    poses_ << std::setprecision(12);
}

void ThirdPersonExport::addFrame(const std::shared_ptr<Camera>& camera)
{
    const Eigen::Matrix3d rotation = camera->R_cw_.transpose();
    const Eigen::Vector3d center = -rotation * camera->t_cw_;
    Eigen::Vector3d heading = rotation.col(2);
    heading.z() = 0;
    // Keep the last horizontal heading when the source camera points vertically.
    if (heading.norm() > 1.e-4)
    {
        heading.normalize();
        heading_ = centers_.empty() ? heading : (0.85 * heading_ + 0.15 * heading).normalized();
    }
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d right = heading_.cross(up).normalized();
    const Eigen::Vector3d eye = center - back * heading_ + side * right + above * up;
    const Eigen::Vector3d forward = (center + 0.35 * heading_ - eye).normalized();
    Eigen::Matrix3d view_rotation;
    view_rotation.col(0) = forward.cross(up).normalized();
    view_rotation.col(1) = forward.cross(view_rotation.col(0));
    view_rotation.col(2) = forward;
    auto view = std::make_shared<Camera>();
    const double focal = width / (2.0 * std::tan(85.0 * M_PI / 360.0));
    view->setCameraModel(false);
    view->setIntrinsic(width, height, focal, focal, width / 2.0, height / 2.0);
    view->setPose(view_rotation, eye);
    view->frame_index_ = camera->frame_index_;
    centers_.push_back(center);
    views_.push_back(view);
    poses_ << camera->frame_index_ << ',' << center.x() << ',' << center.y() << ',' << center.z()
           << ',' << eye.x() << ',' << eye.y() << ',' << eye.z()
           << ',' << heading_.x() << ',' << heading_.y() << '\n';
}

void ThirdPersonExport::saveOnline(const std::shared_ptr<GaussianModel>& map)
{
    saveFrame(views_.size() - 1, map, "online");
}

void ThirdPersonExport::saveFinal(const std::shared_ptr<GaussianModel>& map)
{
    for (size_t index = 0; index < views_.size(); ++index)
    {
        saveFrame(index, map, "final");
        if ((index + 1) % 200 == 0)
            std::cout << "[Third-person final] " << index + 1 << '/' << views_.size() << std::endl;
    }
}

void ThirdPersonExport::saveFrame(size_t index, const std::shared_ptr<GaussianModel>& map,
                                const std::string& phase)
{
    torch::NoGradGuard no_grad;
    const auto& view = views_.at(index);
    cv::Mat image(height, width, CV_8UC3, cv::Scalar::all(0));
    if (map->is_init_)
    {
        auto background = torch::zeros({3}, torch::TensorOptions().device(torch::kCUDA));
        auto pkg = render(view, map, background, false, false, 1.0f, false);
        // Same native RGB convention as evaluation; no extra background compositing.
        auto rgb = std::get<0>(pkg).detach().clamp(0, 1).cpu().permute({1, 2, 0})
            .contiguous().mul(255).to(torch::kUInt8);
        cv::Mat rgb_image(height, width, CV_8UC3, rgb.data_ptr<uint8_t>());
        cv::cvtColor(rgb_image, image, cv::COLOR_RGB2BGR);
    }
    // Trajectory and camera are foreground annotations, visible through the map.
    for (size_t i = 1; i <= index; ++i)
    {
        const Eigen::Vector3d a = view->R_cw_ * centers_[i - 1] + view->t_cw_;
        const Eigen::Vector3d b = view->R_cw_ * centers_[i] + view->t_cw_;
        if (a.z() <= .05 || b.z() <= .05) continue;
        auto p = project(a, *view), q = project(b, *view);
        if (cv::clipLine(image.size(), p, q))
            cv::line(image, p, q, cv::Scalar(255, 220, 20), 3, cv::LINE_AA);
    }
    const Eigen::Vector3d center = view->R_cw_ * centers_[index] + view->t_cw_;
    const cv::Point pixel = project(center, *view);
    const int radius = std::max(1, cvRound(view->fx_ * sphere_radius / center.z()));
    for (int y = std::max(0, pixel.y - radius); y <= std::min(height - 1, pixel.y + radius); ++y)
        for (int x = std::max(0, pixel.x - radius); x <= std::min(width - 1, pixel.x + radius); ++x)
        {
            const double u = double(x - pixel.x) / radius, v = double(y - pixel.y) / radius;
            if (u * u + v * v > 1) continue;
            const double z = std::sqrt(1 - u * u - v * v);
            const double shade = .45 + .55 * std::max(0.0, -.3 * u - .4 * v + .866 * z);
            const double alpha = .35 + .2 * z;
            const cv::Vec3d blue(255 * shade, 145 * shade, 40 * shade);
            auto& value = image.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c)
                value[c] = cv::saturate_cast<uchar>((1 - alpha) * value[c] + alpha * blue[c]);
        }
    cv::circle(image, pixel, radius, cv::Scalar(255, 165, 65), 1, cv::LINE_AA);
    const std::string label = "Third person | " + phase + " map | frame " + std::to_string(view->frame_index_);
    cv::putText(image, label, {18, 32}, cv::FONT_HERSHEY_SIMPLEX, .7, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(image, label, {18, 32}, cv::FONT_HERSHEY_SIMPLEX, .7, {255, 255, 255}, 1, cv::LINE_AA);
    const std::string output = directory_ + '/' + phase;
    std::filesystem::create_directories(output);
    std::ostringstream name;
    name << output << "/frame_" << std::setw(6) << std::setfill('0') << view->frame_index_ << ".jpg";
    if (!cv::imwrite(name.str(), image)) throw std::runtime_error("Cannot write third-person frame");
}
