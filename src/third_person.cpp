#include "third_person.h"

#include <iomanip>
#include <sstream>

namespace {
constexpr int width = 1280, height = 640;
constexpr double back = 1.0, left = 0.3, above = 0.2, sphere_radius = 0.12;
constexpr double heading_time_constant = 1.0, max_turn_rate = M_PI / 4;

cv::Point project(const Eigen::Vector3d& point)
{
    return cv::Point(cvRound(width * (.5 + std::atan2(point.x(), point.z()) / (2 * M_PI))),
                     cvRound(height * (.5 + std::atan2(point.y(), std::hypot(point.x(), point.z())) / M_PI)));
}

void drawSegment(cv::Mat& image, cv::Point a, cv::Point b)
{
    // Draw the short arc across the ERP seam, never a line through the whole image.
    if (b.x - a.x > width / 2) b.x -= width;
    if (a.x - b.x > width / 2) b.x += width;
    for (int offset : {-width, 0, width})
    {
        auto p = a + cv::Point(offset, 0), q = b + cv::Point(offset, 0);
        if (cv::clipLine(image.size(), p, q))
            cv::line(image, p, q, cv::Scalar(255, 220, 20), 2, cv::LINE_AA);
    }
}
}

ThirdPersonExport::ThirdPersonExport(const std::string& result_path, const Eigen::Vector3d& front_axis_camera)
    : directory_(result_path + "/third_person"), front_axis_camera_(front_axis_camera)
{
    std::filesystem::create_directories(directory_);
    poses_.open(directory_ + "/views.csv");
    poses_ << "frame,camera_x,camera_y,camera_z,view_x,view_y,view_z,heading_x,heading_y,front_x,front_y,front_z,timestamp\n";
    poses_ << std::setprecision(17);
    rays_.resize(3, width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const double lon = (double(x) / width - .5) * 2 * M_PI;
            const double lat = (double(y) / height - .5) * M_PI;
            rays_.col(y * width + x) = Eigen::Vector3d(
                std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon));
        }
}

void ThirdPersonExport::addFrame(const std::shared_ptr<Camera>& camera, double timestamp)
{
    const Eigen::Matrix3d rotation = camera->R_cw_.transpose();
    const Eigen::Vector3d center = -rotation * camera->t_cw_;
    const Eigen::Vector3d front = (rotation * front_axis_camera_).normalized();
    if (centers_.empty())
    {
        // Before any motion, use the physical front axis as an initial heading.
        Eigen::Vector3d horizontal = front;
        horizontal.z() = 0;
        if (horizontal.norm() > 1.e-4) heading_ = horizontal.normalized();
    }
    else
    {
        // Causal displacement over five processed frames (0.5 s at 10 Hz).
        const size_t previous = centers_.size() > 5 ? centers_.size() - 5 : 0;
        Eigen::Vector3d displacement = center - centers_[previous];
        displacement.z() = 0;
        // Hold the last heading while stationary instead of following pose jitter.
        if (displacement.norm() > .05)
        {
            const double dt = timestamp - last_timestamp_;
            const double yaw = std::atan2(heading_.y(), heading_.x());
            const double target = std::atan2(displacement.y(), displacement.x());
            const double difference = std::atan2(std::sin(target - yaw), std::cos(target - yaw));
            // Smooth the shortest angular difference, including the +/-pi boundary.
            const double step = std::clamp((1 - std::exp(-dt / heading_time_constant)) * difference,
                                           -max_turn_rate * dt, max_turn_rate * dt);
            heading_ = Eigen::Vector3d(std::cos(yaw + step), std::sin(yaw + step), 0);
        }
    }
    last_timestamp_ = timestamp;
    const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d eye = center - back * heading_ + left * up.cross(heading_) + above * up;
    Eigen::Matrix3d view_rotation;
    view_rotation.col(0) = heading_.cross(up).normalized();
    view_rotation.col(1) = -up;
    view_rotation.col(2) = heading_;
    auto view = std::make_shared<Camera>();
    view->setCameraModel(true);
    view->setIntrinsic(width, height, width / (2 * M_PI), height / M_PI, width / 2.0, height / 2.0);
    view->setPose(view_rotation, eye);
    view->frame_index_ = camera->frame_index_;
    centers_.push_back(center);
    front_axes_.push_back(front);
    views_.push_back(view);
    poses_ << camera->frame_index_ << ',' << center.x() << ',' << center.y() << ',' << center.z()
           << ',' << eye.x() << ',' << eye.y() << ',' << eye.z()
           << ',' << heading_.x() << ',' << heading_.y()
           << ',' << front.x() << ',' << front.y() << ',' << front.z() << ',' << timestamp << '\n';
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
        const int steps = std::max(1, int(std::ceil((b - a).norm() / .03)));
        for (int j = 0; j < steps; ++j)
        {
            const Eigen::Vector3d p = a + (b - a) * (double(j) / steps);
            const Eigen::Vector3d q = a + (b - a) * (double(j + 1) / steps);
            if (p.norm() > .05 && q.norm() > .05) drawSegment(image, project(p), project(q));
        }
    }
    const Eigen::Vector3d center = view->R_cw_ * centers_[index] + view->t_cw_;
    const Eigen::Vector3d front = view->R_cw_ * front_axes_[index];
    // Ray/sphere intersections preserve hemisphere orientation in the ERP projection.
    // Blend the far and near surfaces in that order to show a translucent globe.
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            const Eigen::Vector3d ray = rays_.col(y * width + x);
            const double along = ray.dot(center);
            const double discriminant = along * along - center.squaredNorm() + sphere_radius * sphere_radius;
            if (along <= 0 || discriminant < 0) continue;
            auto& value = image.at<cv::Vec3b>(y, x);
            for (double sign : {1.0, -1.0})
            {
                const Eigen::Vector3d normal = ((along + sign * std::sqrt(discriminant)) * ray - center) / sphere_radius;
                const double shade = .65 + .35 * std::abs(normal.dot(ray));
                const cv::Vec3d color = normal.dot(front) >= 0
                    ? cv::Vec3d(55, 60, 255) : cv::Vec3d(255, 125, 35);
                const double alpha = sign > 0 ? .12 : .5;
                for (int c = 0; c < 3; ++c)
                    value[c] = cv::saturate_cast<uchar>((1 - alpha) * value[c] + alpha * shade * color[c]);
            }
        }
    const std::string output = directory_ + '/' + phase;
    std::filesystem::create_directories(output);
    std::ostringstream name;
    name << output << "/frame_" << std::setw(6) << std::setfill('0') << view->frame_index_ << ".jpg";
    if (!cv::imwrite(name.str(), image)) throw std::runtime_error("Cannot write third-person frame");
}
